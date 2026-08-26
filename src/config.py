"""Nạp config, dò phần cứng, giải quyết ô sweep.

`python -m src.config` in ra cấu hình đã giải quyết mà không nạp model —
dùng để kiểm tra trước khi chạy thật.

PHASE 1 của sweep (xem README §3): adapter đã CHỐT là QDoRA r8, số expert và
top-k cố định (4 / 2). Biến duy nhất còn quét là cặp (biến thể Recombinator,
số block) — vì vậy `sweep` ở đây chỉ còn hai trường: `variant` và `n_block`.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from types import SimpleNamespace

import yaml

from .router import BLOCK_RATIO, VARIANTS as ROUTER_VARIANTS, estimate_params

# ══════════════ OPTION 2 — container GPU thuê theo giờ (§9) ══════════════════
# Chỉ đổi CHỖ ĐỂ FILE (cache, outputs, ckpt) và cách chạy. Không đụng tới
# `fixed`/`sweep`, nên một ô chạy trên máy thuê so sánh được với ô chạy cục bộ.
REMOTE_DEFAULTS = dict(
    enabled="auto", volume="/workspace", workdir=None,
    hf_home=None, torch_home=None, image_cache=None,
    out_dir=None, ckpt_dir=None, use_venv=False, tmux_session="sweep",
    ssh_host=None, ssh_port=None, ssh_user="root", ports={},
)
_FALSEY = ("0", "false", "no", "off")


def remote_is_on(enabled="auto"):
    """RGEXP_REMOTE (env) > remote.enabled (yaml) > tự dò /.dockerenv."""
    env = os.environ.get("RGEXP_REMOTE", "").strip().lower()
    if env:
        return env not in _FALSEY
    e = str(enabled).strip().lower()
    if e in ("on", "true", "1", "yes"):
        return True
    if e in _FALSEY:
        return False
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")


def read_remote(path):
    """Đọc RIÊNG khối `remote`, không đụng phần còn lại của config.

    Tách ra vì phải gọi được rất sớm (xem bootstrap_remote_env) và phải chịu
    được config thiếu/hỏng — lúc đó cứ coi như không bật remote.
    """
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    except Exception:
        raw = {}
    r = dict(REMOTE_DEFAULTS)
    r.update(raw.get("remote") or {})
    r["active"] = remote_is_on(r.get("enabled", "auto"))
    return SimpleNamespace(**r)


def _argv_config_path(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return os.environ.get("RGEXP_CONFIG", "configs/default.yaml")


def bootstrap_remote_env(path=None):
    """Export HF_HOME/TORCH_HOME TRƯỚC khi huggingface_hub kịp import.

    `huggingface_hub` đọc HF_HOME đúng một lần lúc import; đặt muộn hơn là vô
    tác dụng và ~9GB weight sẽ rơi vào $HOME của container — mất sạch sau
    Restart. `src/train.py` import `.config` trước `.data` nên hàm này chạy kịp.
    scripts/_env.sh cũng export sẵn hai biến này; đây là lớp dự phòng cho ai
    chạy thẳng `python -m src.train`.
    """
    rm = read_remote(path or _argv_config_path())
    if not rm.active:
        return rm
    for var, val in (("HF_HOME", rm.hf_home), ("TORCH_HOME", rm.torch_home)):
        if not val or os.environ.get(var):
            continue
        try:
            os.makedirs(val, exist_ok=True)
            os.environ[var] = val
        except OSError as e:      # volume chưa mount / không có quyền
            print(f"⚠ remote: không tạo được {var}={val} ({e}) — dùng mặc định")
    return rm


REMOTE = bootstrap_remote_env()


def _apply_remote(rm, run, explicit):
    """Chuyển out_dir / ckpt_dir / image_cache sang volume lớn.

    `explicit` là các khoá người dùng đã tự đặt bằng --set: đặt tay thì thắng,
    remote không giành lại.
    """
    for key, val in (("out_dir", rm.out_dir), ("ckpt_dir", rm.ckpt_dir)):
        if val and f"run.{key}" not in explicit:
            run[key] = val
    if rm.image_cache and "run.data.image_cache" not in explicit:
        run["data"]["image_cache"] = rm.image_cache


# ─────────────────────────── Preset theo VRAM ────────────────────────────────
# Mọi preset giữ effective_batch = train_bs × grad_accum = 32.
# STAGE 0 in VRAM đỉnh để bạn nới ra nếu còn dư.
#
# Thấp hơn preset của sweep adapter cũ MỘT BẬC: MoE giữ E bản delta cùng lúc và
# QDoRA phải dequantize base weight trong forward, nên đỉnh VRAM cao hơn LoRA
# thường ở cùng batch.
PRESETS = [
    # (VRAM tối thiểu, tên, train_bs, grad_accum, infer_bs, grad_ckpt)
    (70, "A100-80 / H100", 8, 4, 8, True),
    (38, "A100-40 / L40S", 4, 8, 4, True),
    (22, "RTX 3090/4090", 2, 16, 2, True),
    # 16GB (4070 Ti Super / 4080 / L4): base NF4 ~2.5GB + dequantize base weight
    # trong forward + E bản delta lúc backward -> chỉ vừa đúng train_bs=1.
    (14, "16GB (4070 Ti S / 4080 / L4)", 1, 32, 1, True),
    (0, "GPU <14GB — chật", 1, 32, 1, True),
]

# ── Ô sweep Phase 1: biến thể Recombinator × số block (§2) ───────────────────
# Sweep chỉ còn HAI biến thể: BRM (baseline không attention) và GSLF. MBRF vẫn
# nằm trong src/router.py và chạy tay được (--variant mbrf --n-block 4), nhưng
# không còn ô nào trong bảng mặc định.
# Tỉ lệ giá một block: 1 GSLF ≈ 7 BRM. brm 10→60 phủ 67.6M→74.2M, gslf 1→6 phủ
# 67.2M→71.8M — phần chồng nhau (~67M → ~72M) là chỗ so "cùng ngân sách tham
# số thì kiến trúc nào hơn".
SWEEP_BLOCKS = {
    "brm": (10, 20, 30, 40, 50, 60),
    "gslf": (1, 2, 3, 4, 5, 6),
}
assert set(SWEEP_BLOCKS) <= set(ROUTER_VARIANTS), "biến thể sweep lạ với router"


def variant_desc(cfg):
    """Một dòng mô tả ô sweep, dùng chung cho log / metrics / print."""
    s, f = cfg.sweep, cfg.fixed
    est = estimate_params(s.variant, s.n_block, f.n_expert)
    return (f"{s.variant}×{s.n_block} block "
            f"(1 block ≈ {BLOCK_RATIO[s.variant]:.0f} BRM) | "
            f"router {est['router'] / 1e6:.2f}M + "
            f"{f.n_expert} expert QDoRA r{f.lora_r} top-{f.top_k} "
            f"≈ {est['total'] / 1e6:.1f}M")


def adapter_desc(f, sweep=None):
    """Adapter đã CHỐT ở Phase này — giữ hàm để log/metrics gọi được như cũ."""
    return f"qdora r{f.lora_r} a{f.lora_alpha} × {f.n_expert}E top-{f.top_k}"


def detect_hardware():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy GPU CUDA.")
    p = torch.cuda.get_device_properties(0)
    sm = torch.cuda.get_device_capability(0)
    hw = SimpleNamespace(name=p.name, vram_gb=p.total_memory / 1024**3, sm=sm,
                         has_bf16=sm[0] >= 8,
                         n_gpu=torch.cuda.device_count())
    # §4.1.3 yêu cầu bf16. sm75 (T4) không có -> fp16 tràn số với Gemma 3 ngay
    # micro-batch đầu. Đó là nhánh riêng, không dùng project này.
    if not hw.has_bf16:
        raise RuntimeError(
            f"GPU sm{sm[0]}{sm[1]} không hỗ trợ bf16. Cần Ampere trở lên (sm80+).")
    return hw


def pick_attn_impl():
    """flash_attention_2 > sdpa. Cả hai đều dùng kernel hợp nhất trên GPU."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def pick_preset(vram_gb):
    for min_vram, name, bs, accum, ibs, ckpt in PRESETS:
        if vram_gb >= min_vram:
            if vram_gb < 12:
                print(f"⚠ Chỉ {vram_gb:.1f}GB VRAM — dưới sàn thực tế của QDoRA "
                      f"4 expert. Nếu OOM ở STAGE 0 thì không có preset nào cứu "
                      f"được, phải hạ fixed.n_expert (mất ý nghĩa sweep) hoặc "
                      f"đổi GPU.")
            return dict(name=name, train_bs=bs, grad_accum=accum,
                        infer_bs=ibs, grad_ckpt=ckpt)
    raise RuntimeError("Không khớp preset nào")


def next_pow2(n):
    """§4.1.2 — làm tròn LÊN lũy thừa 2 gần nhất."""
    return 1 << math.ceil(math.log2(max(n, 1)))


def image_desc(f):
    """Một dòng mô tả đường đi của ảnh, dùng chung cho print và log."""
    if f.image_size_model is None:
        kw = ", ".join(f"{k}={v}" for k, v in f.processor_kwargs.items())
        return (f"{f.image_size_src}² giữ NGUYÊN (processor tự resize"
                + (f" | {kw}" if kw else "") + ")")
    return (f"{f.image_size_src} -> {f.image_size_model} "
            f"({f.resize_filter}, antialias={f.antialias})")


def _deep_set(d, dotted, value):
    """--set sweep.n_block=20  ->  d['sweep']['n_block'] = 20"""
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)   # tự ép int/float/bool/null


def load_config(path="configs/default.yaml", overrides=None):
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set cần dạng khoa=gia_tri, nhận: {ov!r}")
        _deep_set(raw, *ov.split("=", 1))

    hw = detect_hardware()
    preset = pick_preset(hw.vram_gb)
    fixed, sweep, run = raw["fixed"], raw["sweep"], raw["run"]
    explicit = {ov.split("=", 1)[0] for ov in (overrides or [])}

    # §1.6 — số mũ của bảng α (0.125 = nghịch đảo căn bậc 8). Đặt default ở đây
    # để config cũ (chưa có khoá này) không chết giữa chừng vì AttributeError.
    run.setdefault("focal_alpha_pow", 0.125)

    # §9 — container GPU thuê: dời cache/outputs/ckpt sang volume lớn. Làm
    # TRƯỚC khi ghép tag vào out_dir để mỗi ô vẫn có thư mục riêng.
    rm = read_remote(path)
    if rm.active:
        _apply_remote(rm, run, explicit)

    # ── Giải quyết ô sweep ───────────────────────────────────────────────────
    if sweep["variant"] not in ROUTER_VARIANTS:
        raise ValueError(f"variant lạ: {sweep['variant']} "
                         f"(chọn: {list(ROUTER_VARIANTS)})")
    n_block = int(sweep["n_block"])
    if n_block not in SWEEP_BLOCKS.get(sweep["variant"], ()):
        # Không chặn: dải block là gợi ý để hai biến thể so được với nhau, không
        # phải luật — và mbrf thì cố tình nằm ngoài sweep. Nhưng phải BÁO, vì ô
        # ngoài bảng không có ô đối chứng cùng ngân sách tham số ở biến thể kia.
        print(f"⚠ sweep: {sweep['variant']}-n{n_block} nằm ngoài bảng "
              f"{SWEEP_BLOCKS.get(sweep['variant']) or '(biến thể ngoài sweep)'}"
              f" — ô này không có đối chứng cùng lượng tham số ở biến thể khác.")
    sweep["n_block"] = n_block
    if sweep["model"] not in raw["models"]:
        raise ValueError(f"model lạ: {sweep['model']}")

    mdl = raw["models"][sweep["model"]]
    fixed["model_id"] = mdl["model_id"]
    fixed["lora_exclude"] = mdl["exclude"]
    # §4.1.1 — mỗi model MỘT processor riêng. Gemma 3 buộc 896² (SigLIP có
    # bảng position embedding cố định); Qwen2.5-VL phân giải động nên nhận
    # thẳng ảnh gốc. image_size = null -> không resize, processor tự lo.
    fixed["image_size_model"] = mdl.get("image_size")
    fixed["processor_kwargs"] = dict(mdl.get("processor_kwargs") or {})
    # d_enc thật lấy từ config của model lúc build (hidden_size); giá trị ở đây
    # chỉ để in bảng tham số trước khi nạp weight.
    fixed["d_enc"] = int(mdl.get("hidden_size", 2560))
    fixed["lora_alpha"] = fixed["lora_r"] * fixed["alpha_ratio"]
    # Adapter đã chốt: QDoRA = LoRA + DoRA + base NF4. Không còn nhánh nào khác
    # trong Phase này, nên khoá cứng thay vì để config đổi ngầm.
    fixed["use_dora"] = True
    run["quant"] = True

    tag = f"{sweep['model']}-{sweep['variant']}-n{n_block}"
    if run.get("hub_repo") is None:
        run["hub_repo"] = f"RGExp123/mimic-cxr-{tag}"
    if run.get("wandb_run") is None:
        run["wandb_run"] = tag
    # Mỗi ô sweep có thư mục riêng: chạy song song/nối tiếp không đè checkpoint.
    run["out_dir"] = os.path.join(run["out_dir"], tag)
    run["ckpt_dir"] = os.path.join(run["ckpt_dir"], tag)

    # null trong YAML = để preset quyết định
    for key in ("train_bs", "grad_accum", "infer_bs", "grad_ckpt"):
        if run.get(key) is None:
            run[key] = preset[key]

    import torch
    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32,
                 "float16": torch.float16}
    fixed["compute_dtype"] = dtype_map[fixed["compute_dtype"]]
    fixed["trainable_dtype"] = dtype_map[fixed["trainable_dtype"]]
    fixed["attn_impl"] = pick_attn_impl()

    cfg = SimpleNamespace(
        fixed=SimpleNamespace(**fixed),
        run=SimpleNamespace(**run),
        data=SimpleNamespace(**run["data"]),
        sweep=SimpleNamespace(**sweep, tag=tag),
        hw=hw, preset=preset, remote=rm,
    )
    _check_phases(cfg)
    for d in (cfg.run.out_dir, cfg.run.ckpt_dir):
        os.makedirs(d, exist_ok=True)
    return cfg


def _check_phases(cfg):
    """§1.7 — lịch hai phase phải chia hết cho effective batch.

    Lệch một chút thì DataLoader drop_last cắt mất phần dư và số mẫu thật sự
    dùng khác con số ghi trong tài liệu — im lặng nhưng làm hỏng tính so sánh
    giữa các ô.
    """
    r = cfg.run
    eff = r.train_bs * r.grad_accum
    for name in ("phase1_samples", "phase2_samples"):
        n = getattr(r, name)
        if n % eff:
            print(f"⚠ {name}={n} không chia hết cho effective batch {eff} "
                  f"-> dùng {n // eff * eff} mẫu ({n // eff} step)")
    if r.phase1_samples // eff < 1 or r.phase2_samples // eff < 1:
        raise ValueError(
            f"effective batch {eff} lớn hơn số mẫu của một phase "
            f"({r.phase1_samples}/{r.phase2_samples}) — không đủ một step.")


def phase_steps(cfg, phase):
    """Số optimizer step của một phase, suy từ số mẫu (§1.7)."""
    r = cfg.run
    n = r.phase1_samples if phase == 1 else r.phase2_samples
    return n // (r.train_bs * r.grad_accum)


def resolve_seq_len(cfg, n_prompt_tok, n_image_tok, log):
    """§4.1.2 — max_seq_len = 2^ceil(log2(prompt + img + delta)).

    Gọi SAU khi có processor (mới biết số image token thật). Lũy thừa 2 giúp
    kernel attention chạy trên kích thước "tròn", tránh padding ngầm.
    """
    if cfg.run.max_seq_len is not None:
        log("seq", f"max_seq_len ép cứng = {cfg.run.max_seq_len}")
        return cfg.run.max_seq_len
    need = n_prompt_tok + n_image_tok + cfg.run.seq_delta
    val = next_pow2(need)
    log("seq", f"max_seq_len = 2^k: prompt {n_prompt_tok} + img {n_image_tok} "
               f"+ delta {cfg.run.seq_delta} = {need} -> {val}")
    cfg.run.max_seq_len = val
    return val


def build_argparser():
    ap = argparse.ArgumentParser(
        description="MIMIC-CXR MoE-LoRA — sweep Phase 1 (Recombinator × n_block)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="Ghi đè config, vd: --set run.phase1_samples=3200")
    ap.add_argument("--model", choices=["gemma3", "qwen25vl"])
    ap.add_argument("--variant", choices=list(ROUTER_VARIANTS))
    ap.add_argument("--n-block", type=int, dest="n_block")
    ap.add_argument("--mode", choices=["train", "test", "both"])
    ap.add_argument("--phase", type=int, choices=[1, 2],
                    help="Chạy riêng một phase (mặc định: cả hai, nối tiếp)")
    # dry_run mặc định false trong config (chạy thật). Hai cờ này để ép:
    ap.add_argument("--dry-run", action="store_true",
                    help="Dừng sau STAGE 0 (2 step) để kiểm cấu hình")
    ap.add_argument("--no-dry-run", action="store_true",
                    help="Ép chạy thật (ghi đè config)")
    return ap


def config_from_args(argv=None):
    args = build_argparser().parse_args(argv)
    ov = list(args.overrides)
    for name, key in (("model", "sweep.model"), ("variant", "sweep.variant"),
                      ("n_block", "sweep.n_block"), ("mode", "run.mode"),
                      ("phase", "run.only_phase")):
        val = getattr(args, name)
        if val is not None:
            ov.append(f"{key}={val}")
    if args.dry_run:
        ov.append("run.dry_run=true")
    if args.no_dry_run:
        ov.append("run.dry_run=false")
    return load_config(args.config, ov)


if __name__ == "__main__":
    c = config_from_args()
    est = estimate_params(c.sweep.variant, c.sweep.n_block, c.fixed.n_expert)
    print(f"Ô sweep : {c.sweep.tag}")
    print(f"GPU     : {c.hw.name} | {c.hw.vram_gb:.0f}GB | "
          f"sm{c.hw.sm[0]}{c.hw.sm[1]} | {c.hw.n_gpu} GPU")
    print(f"Preset  : {c.preset['name']}")
    print(f"Model   : {c.fixed.model_id} | attn={c.fixed.attn_impl}")
    print(f"Router  : {variant_desc(c)}")
    print(f"          router {est['router']:,} + expert {est['experts']:,} "
          f"= {est['total']:,} param trainable")
    print(f"Adapter : {adapter_desc(c.fixed)} | quant={c.run.quant}")
    print(f"Ảnh     : {image_desc(c.fixed)}")
    print(f"Batch   : {c.run.train_bs} × {c.run.grad_accum} = "
          f"{c.run.train_bs * c.run.grad_accum} mẫu/step")
    print(f"Phase 1 : {c.run.phase1_samples} mẫu = {phase_steps(c, 1)} step "
          f"| loss = Focal (α = 1/f^{c.run.focal_alpha_pow})")
    print(f"Phase 2 : {c.run.phase2_samples} mẫu = {phase_steps(c, 2)} step "
          f"| loss = CE + {c.run.w_sup}·Focal + {c.run.w_aux}·Aux")
    print(f"Output  : {c.run.out_dir} | hub={c.run.hub_repo}")
    print(f"Mode    : {c.run.mode} | dry_run={c.run.dry_run} | "
          f"policy={'BẬT' if c.run.use_policy else 'TẮT (chấm raw_output)'}")
    if c.remote.active:
        print(f"Remote  : BẬT (§9) | venv={'có' if c.remote.use_venv else 'không'}"
              f" | tmux={c.remote.tmux_session}")
        print(f"Cache   : HF_HOME={os.environ.get('HF_HOME', '(mặc định)')} | "
              f"ảnh={c.data.image_cache} | ckpt={c.run.ckpt_dir}")
    else:
        print("Remote  : tắt (chạy cục bộ — option 1)")

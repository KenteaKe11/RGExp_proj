"""Nạp config, dò phần cứng, giải quyết ô sweep.

`python -m src.config` in ra cấu hình đã giải quyết mà không nạp model —
dùng để kiểm tra trước khi chạy thật.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from types import SimpleNamespace

import yaml

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
# Mọi preset giữ effective_batch = train_bs × grad_accum = 32 (§3.2).
# Thận trọng; STAGE 0 in VRAM đỉnh để bạn nới ra nếu còn dư.
PRESETS = [
    # (VRAM tối thiểu, tên, train_bs, grad_accum, infer_bs, grad_ckpt)
    (70, "A100-80 / H100", 8, 4, 8, False),
    (38, "A100-40 / L40S", 4, 8, 4, False),
    (22, "RTX 3090/4090", 2, 16, 2, True),
    (0, "GPU <22GB", 1, 32, 1, True),
]

# §3.2 — bảng biến thể. quant: nền 4-bit NF4. dora: phân rã magnitude/direction.
VARIANTS = {
    "lora": dict(quant=False, dora=False, ranks=(16, 32)),
    "qlora": dict(quant=True, dora=False, ranks=(16, 32)),
    "dora": dict(quant=False, dora=True, ranks=(8, 16)),
    "qdora": dict(quant=True, dora=True, ranks=(8, 16)),
}


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
    """--set sweep.rank=8  ->  d['sweep']['rank'] = 8"""
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

    # §9 — container GPU thuê: dời cache/outputs/ckpt sang volume lớn. Làm
    # TRƯỚC khi ghép tag vào out_dir để mỗi ô vẫn có thư mục riêng.
    rm = read_remote(path)
    if rm.active:
        _apply_remote(rm, run, {ov.split("=", 1)[0] for ov in (overrides or [])})

    # ── Giải quyết ô sweep ───────────────────────────────────────────────────
    if sweep["variant"] not in VARIANTS:
        raise ValueError(f"variant lạ: {sweep['variant']} "
                         f"(chọn: {list(VARIANTS)})")
    v = VARIANTS[sweep["variant"]]
    if sweep["rank"] not in v["ranks"]:
        raise ValueError(
            f"§3.2: {sweep['variant']} chỉ sweep rank {v['ranks']}, "
            f"nhận {sweep['rank']}. Muốn rank khác thì ghi vào sweep log.")
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
    fixed["lora_r"] = sweep["rank"]
    fixed["lora_alpha"] = sweep["rank"] * fixed["alpha_ratio"]
    fixed["use_dora"] = v["dora"]
    run["quant"] = v["quant"]

    tag = f"{sweep['model']}-{sweep['variant']}-r{sweep['rank']}"
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
    for d in (cfg.run.out_dir, cfg.run.ckpt_dir):
        os.makedirs(d, exist_ok=True)
    return cfg


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
        description="MIMIC-CXR VLM fine-tuning sweep (§3.2)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="Ghi đè config, vd: --set sweep.rank=8")
    ap.add_argument("--model", choices=["gemma3", "qwen25vl"])
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--rank", type=int)
    ap.add_argument("--mode", choices=["train", "test", "both"])
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
                      ("rank", "sweep.rank"), ("mode", "run.mode")):
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
    print(f"Ô sweep : {c.sweep.tag}")
    print(f"GPU     : {c.hw.name} | {c.hw.vram_gb:.0f}GB | "
          f"sm{c.hw.sm[0]}{c.hw.sm[1]} | {c.hw.n_gpu} GPU")
    print(f"Preset  : {c.preset['name']}")
    print(f"Model   : {c.fixed.model_id} | attn={c.fixed.attn_impl}")
    print(f"Adapter : {'DoRA' if c.fixed.use_dora else 'LoRA'} "
          f"r{c.fixed.lora_r} a{c.fixed.lora_alpha} | quant={c.run.quant}")
    print(f"Ảnh     : {image_desc(c.fixed)}")
    print(f"Batch   : {c.run.train_bs} × {c.run.grad_accum} = "
          f"{c.run.train_bs * c.run.grad_accum} mẫu/step | "
          f"{c.run.num_epochs} epoch")
    print(f"Output  : {c.run.out_dir} | hub={c.run.hub_repo}")
    print(f"Mode    : {c.run.mode} | dry_run={c.run.dry_run}")
    if c.remote.active:
        print(f"Remote  : BẬT (§9) | venv={'có' if c.remote.use_venv else 'không'}"
              f" | tmux={c.remote.tmux_session}")
        print(f"Cache   : HF_HOME={os.environ.get('HF_HOME', '(mặc định)')} | "
              f"ảnh={c.data.image_cache} | ckpt={c.run.ckpt_dir}")
    else:
        print("Remote  : tắt (chạy cục bộ — option 1)")

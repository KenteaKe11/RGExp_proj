"""Nạp config, dò phần cứng, giải quyết cấu hình một lần chạy.

`python -m src.config` in ra cấu hình đã giải quyết mà không nạp model —
dùng để kiểm tra trước khi chạy thật.

KHÔNG CÒN SWEEP. Trước đây file này giải quyết một "ô sweep" = (model,
variant, số block), và `SWEEP_BLOCKS` là bảng các ô hợp lệ.
Kiến trúc router giờ đã chốt hẳn — Feat-Extractor CNN + QFormer, không
có siêu tham số kiến trúc nào để quét (xem src/router.py) — nên khối `sweep`
trong YAML đổi tên thành `arch` và chỉ còn giữ đúng một trường: `model`.
`arch.tag` vẫn là thứ đặt tên thư mục outputs/ckpt và repo Hub, để nhiều lần
chạy khác model không đè lên nhau.

Adapter đã CHỐT là QDoRA r8. MoE: E expert / top-k, thuần top-k, không còn
expert hứng và không còn trần tải; cân bằng tải là MỘT số hạng aux trong
Total_Loss (run.w_aux).

Lịch train là MỘT phase multi-task duy nhất và KHÔNG có đích (xem
configs/default.yaml, khối `run`): không còn phase1_samples/phase2_samples,
không còn only_phase, không còn dừng sớm. Vì thế file này cũng không còn hàm
nào chia lịch — chỉ còn kiểm tra lịch LR (cycle_samples) cho hợp lệ.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from types import SimpleNamespace

import yaml

from .router import (D_MODEL, IMG_SIZE, N_BLOCK, N_TOKEN, STRIDE,
                     arch_desc, decay_desc, estimate_params)

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


def _deep_merge(base, over):
    """Gộp `over` VÀO `base` theo chiều sâu; dict thì đệ quy, còn lại thì đè."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def read_yaml(path):
    """Đọc một file config, theo `extends:` lên config cha nếu có.

    Một file profile (vd configs/v100_32g.yaml) chỉ ghi ĐÚNG những khoá nó
    đổi, dòng đầu là:

        extends: default.yaml     # tương đối với chính file đó

    rồi được gộp SÂU lên trên default.yaml. Chép cả 430 dòng ra file thứ hai
    là cách chắc chắn nhất để hai file lệch nhau sau vài lần sửa: khoá mới
    thêm ở default sẽ không bao giờ tới được profile, và không ai nhận ra.

    encoding="utf-8" là BẮT BUỘC, không phải trang trí: config có tiếng Việt,
    và trên Windows `open()` mặc định là cp1252 -> UnicodeDecodeError ngay
    dòng đầu.
    """
    return _read_yaml_chain(os.path.abspath(path), [])


def _read_yaml_chain(path, seen):
    if path in seen:
        raise ValueError("`extends:` tạo vòng lặp: "
                         + " -> ".join(seen + [path]))
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)
    if not parent:
        return raw
    ppath = (parent if os.path.isabs(parent)
             else os.path.join(os.path.dirname(path), parent))
    ppath = os.path.abspath(ppath)
    if not os.path.exists(ppath):
        raise FileNotFoundError(
            f"{path}: extends: {parent!r} -> không thấy {ppath}")
    return _deep_merge(_read_yaml_chain(ppath, seen + [path]), raw)


def read_remote(path):
    """Đọc RIÊNG khối `remote`, không đụng phần còn lại của config.

    Tách ra vì phải gọi được rất sớm (xem bootstrap_remote_env) và phải chịu
    được config thiếu/hỏng — lúc đó cứ coi như không bật remote.
    """
    try:
        raw = read_yaml(path)
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
# thường ở cùng batch. Từ khi gộp một phase multi-task, MỌI step đều là step
# "nặng" (router + expert + decoder) — không còn phase 1 rẻ tiền để làm đệm.
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

# Tên kiến trúc, đi vào `arch.tag` -> tên thư mục outputs/ckpt và repo Hub.
# Không còn siêu tham số kiến trúc nào để quét, nên đây là hằng số chứ không
# phải một trường trong YAML.
ARCH_NAME = f"cnnqf{D_MODEL}"


def arch_summary(cfg):
    """Một dòng mô tả kiến trúc, dùng chung cho log / metrics / print."""
    f = cfg.fixed
    est = estimate_params(f.n_expert)
    return (arch_desc(f.n_expert, f.top_k)
            + f" + {f.n_expert} expert QDoRA r{f.lora_r} "
              f"≈ {est['total'] / 1e6:.1f}M trainable")


def adapter_desc(f, arch=None):
    """Adapter đã CHỐT — giữ hàm để log/metrics gọi được như cũ."""
    return (f"qdora r{f.lora_r} a{f.lora_alpha} × {f.n_expert}E "
            f"top-{f.top_k} (thuần top-k, không expert hứng)")


def detect_hardware(require_bf16=True, compute_dtype="bfloat16"):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Không thấy GPU CUDA.")
    p = torch.cuda.get_device_properties(0)
    sm = torch.cuda.get_device_capability(0)
    hw = SimpleNamespace(name=p.name, vram_gb=p.total_memory / 1024**3, sm=sm,
                         has_bf16=sm[0] >= 8,
                         n_gpu=torch.cuda.device_count())
    # §4.1.3 yêu cầu bf16. sm60 (P100) / sm70 (V100) / sm75 (T4) không có.
    # Mặc định CHẶN: fp16 có dải số hẹp hơn nhiều và Gemma 3 dễ tràn (inf/NaN)
    # ở activation. Muốn chạy trên GPU đời cũ thì phải khai báo RÕ trong config
    #     fixed.require_bf16: false
    #     fixed.compute_dtype: float16
    # để không ai vô tình train fp16 rồi tưởng kết quả so được với các ô bf16.
    if not hw.has_bf16:
        if require_bf16:
            raise RuntimeError(
                f"GPU sm{sm[0]}{sm[1]} không hỗ trợ bf16. Cần Ampere trở lên "
                f"(sm80+), HOẶC chạy fp32 bằng profile có sẵn:\n"
                f"    bash scripts/run_v100.sh --dry-run\n"
                f"    python -m src.train --config configs/v100_32g.yaml\n"
                f"(V100 sm70 / T4 sm75 — xem README §0.3.)")
        dt = str(compute_dtype)
        if "float32" in dt:
            print(f"✓ GPU sm{sm[0]}{sm[1]} không có bf16 -> chạy fp32. Chậm "
                  f"hơn bf16 khoảng 2-3× nhưng KHÔNG tràn số, và số học rộng "
                  f"hơn bf16 nên checkpoint vẫn bàn giao được với máy sm80+.")
        elif "float16" in dt:
            print(f"⚠ GPU sm{sm[0]}{sm[1]} KHÔNG có bf16 và bạn chọn fp16. "
                  f"Gemma 3 tràn activation quanh layer 5 (đo được: 51072 ở "
                  f"layer 4, trần fp16 là 65504) -> loss NaN/inf ngay STAGE 0. "
                  f"Dùng compute_dtype=float32 thay vì đi tìm lỗi wiring.")
    return hw


def pick_attn_impl(sm=None, compute_dtype=None):
    """flash_attention_2 > sdpa. Cả hai đều dùng kernel hợp nhất trên GPU.

    HAI cửa loại FA2, và cả hai đều nổ MUỘN (lúc forward, không phải lúc nạp
    model) nếu để lọt:
      * kernel FA2 chỉ build cho sm80+  -> V100 (sm70) / T4 (sm75) không có;
      * FA2 chỉ nhận fp16/bf16          -> mọi lần chạy fp32 đều loại.
    sdpa không mất gì đáng kể ở đây: nó cũng là kernel hợp nhất, chỉ là chọn
    backend mem-efficient/math thay vì flash.
    """
    if sm is not None and sm[0] < 8:
        return "sdpa"
    if compute_dtype is not None and "float32" in str(compute_dtype):
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def pick_preset(vram_gb, compute_dtype=None):
    """Chọn batch theo VRAM. fp32 thì VRAM hiệu dụng chỉ còn MỘT NỬA.

    Bảng PRESETS đo trên đường bf16. Ở fp32 mọi activation, mọi bản base
    weight vừa dequantize và mọi bản delta của expert đều rộng gấp đôi, nên
    một V100 32GB chạy fp32 xếp cùng bậc với một card 16GB chạy bf16 — chia
    đôi rồi mới tra bảng, thay vì để STAGE 0 phát hiện bằng một cú OOM.
    """
    fp32 = compute_dtype is not None and "float32" in str(compute_dtype)
    eff_gb = vram_gb / 2 if fp32 else vram_gb
    for min_vram, name, bs, accum, ibs, ckpt in PRESETS:
        if eff_gb >= min_vram:
            if eff_gb < 12:
                print(f"⚠ VRAM hiệu dụng {eff_gb:.1f}GB"
                      + (f" ({vram_gb:.1f}GB thật, fp32 ăn gấp đôi)" if fp32
                         else "")
                      + f" — dưới sàn thực tế của QDoRA 12 expert. Nếu OOM ở "
                        f"STAGE 0 thì không có preset nào cứu được, phải hạ "
                        f"fixed.n_expert (mất tính so sánh) hoặc đổi GPU.")
            if fp32:
                name = f"{name} · fp32 ({vram_gb:.0f}GB thật)"
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
    """--set run.train_bs=2  ->  d['run']['train_bs'] = 2"""
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)   # tự ép int/float/bool/null


# Khoá của bản HAI PHASE cũ. Không còn ý nghĩa gì trong lịch một phase; để
# nguyên trong config thì người đọc tưởng nó vẫn có tác dụng, nên báo rõ.
_DEAD_KEYS = {
    "phase1_samples": "lịch train không còn trần số mẫu",
    "phase2_samples": "lịch train không còn trần số mẫu",
    "only_phase": "chỉ còn MỘT phase multi-task",
    "p2_freeze_trunk": "không còn phase 2 để đóng băng trunk",
    "p2_freeze_sup_head": "không còn phase 2 để đóng băng sup_head",
    "focal_gamma": "nhánh supervised dùng ASL, không còn Focal (run.asl_*)",
    "focal_alpha_pow": "ASL không có bảng α theo tần suất — γ⁻ lo phần đó",
    "focal_alpha_normalize": "ASL không có bảng α theo tần suất",
    "early_stop_patience": "lịch train không có đích, không có dừng sớm",
    "early_stop_min_delta": "lịch train không có đích, không có dừng sớm",
    "early_stop_phases": "lịch train không có đích, không có dừng sớm",
    "resume": "LUÔN resume nếu có checkpoint",
    "save_every": "ckpt đếm theo lần val (save_every_n_vals)",
    "warmup_steps": "warmup đếm bằng mẫu (warmup_samples)",
    # ── Khoá của bản CÂN BẰNG TẢI AUX-FREE (bias + trần tải + expert hứng) ──
    # Cả ba cơ chế đã bị thay bằng ĐÚNG một số hạng aux trong Total_Loss
    # (run.w_aux). Để nguyên trong config thì người đọc tưởng chúng còn tác
    # dụng, mà cái nguy hơn: `n_shared` cũ đổi VAI của những expert cuối.
    "bias_gamma": "cân bằng tải giờ là số hạng aux trong loss (run.w_aux)",
    "bias_gamma_start": "không còn bias để nắn, nên không còn lịch cho γ",
    "bias_clip": "không còn bias để chặn trần",
    "router_bias_at_eval": "không còn bias nào để cộng vào điểm chọn",
    "capacity_factor": "không còn trần tải mỗi micro-batch — top-k thuần",
    "n_shared": "không còn expert HỨNG — cả E expert đều được định tuyến",
    "shared_weight": "không còn expert hứng thì không có suất cố định nào",
    "decay_warmup_steps": ("lịch decay của σ đếm bằng MẪU "
                           "(fixed.decay_warmup_samples)"),
    # ── Khoá của bản CÓ SWEEP ───────────────────────────────────────────────
    "variant": "kiến trúc router đã chốt, không còn biến thể để quét",
    "n_block": "kiến trúc router đã chốt, không còn số block để quét",
    "d_enc": "router đọc ảnh, không đọc hidden state của Gemma",
    # ── Khoá của bản có BACKBONE PRETRAIN + SWA ─────────────────────────────
    "router_backbone": ("nhánh router không nạp trọng số ngoài nào nữa — "
                        "Feat-Extractor train từ số 0"),
    "router_frozen_upto": "không còn phần nào bị đóng băng trong nhánh router",
    "swa": "đã bỏ hẳn SWA — chấm trên trọng số cuối (weights.pt)",
    "swa_use": "đã bỏ hẳn SWA",
    "hub_include_swa": "đã bỏ hẳn SWA — không còn swa.pt để đẩy lên Hub",
}


def load_config(path="configs/default.yaml", overrides=None):
    raw = read_yaml(path)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set cần dạng khoa=gia_tri, nhận: {ov!r}")
        _deep_set(raw, *ov.split("=", 1))

    # dtype phải đọc TRƯỚC khi dò phần cứng: cùng một GPU sm70, fp32 là
    # đường chạy được còn fp16 là đường tràn số, và thông báo phải nói đúng
    # cái đang xảy ra. Lúc này nó vẫn là CHUỖI ("float32"), map sang
    # torch.dtype mãi tới cuối hàm.
    _dtype_name = str(raw["fixed"].get("compute_dtype", "bfloat16"))
    hw = detect_hardware(bool(raw["fixed"].get("require_bf16", True)),
                         _dtype_name)
    preset = pick_preset(hw.vram_gb, _dtype_name)
    # Khối `sweep` cũ (model + variant + n_block) đổi tên thành `arch` và chỉ
    # còn `model`. Đọc được cả hai để config cũ không nổ ngay ở dòng đầu —
    # nhưng variant/n_block trong đó sẽ rơi vào _DEAD_KEYS và bị báo.
    if "arch" not in raw and "sweep" in raw:
        raw["arch"] = raw.pop("sweep")
        print("⚠ config: khối `sweep:` đã đổi tên thành `arch:` — không còn "
              "sweep nào để quét (kiến trúc router đã chốt).")
    fixed, arch, run = raw["fixed"], raw["arch"], raw["run"]
    explicit = {ov.split("=", 1)[0] for ov in (overrides or [])}

    # Khoá chết của các bản trước: BỎ QUA nhưng nói rõ, đừng để im lặng.
    for key, why in _DEAD_KEYS.items():
        for block, name in ((fixed, "fixed"), (arch, "arch"), (run, "run")):
            if key in block:
                block.pop(key)
                print(f"⚠ config: bỏ qua {name}.{key} — {why}")

    # §1.7 — Total_Loss = 1·CE + w_sup·ASL + w_aux·Aux.
    run.setdefault("w_sup", 0.5)
    run.setdefault("w_aux", 0.05)
    # §1.6 — ASL (Ridnik et al. 2021) trên 14 finding. γ giữ ĐÚNG default của
    # bản gốc; chỉ ngưỡng âm tính đi theo chuẩn đọc phim (xem yaml).
    run.setdefault("asl_gamma_pos", 0.0)
    run.setdefault("asl_gamma_neg", 4.0)
    run.setdefault("asl_clip", 0.02)
    run.setdefault("uncertain_soft", 0.7)
    run.setdefault("save_best", True)
    run.setdefault("save_every_n_vals", 2)
    run.setdefault("num_workers", 4)
    run.setdefault("hub_sync", False)
    run.setdefault("hub_include_optimizer", True)
    fixed.setdefault("require_bf16", True)

    # Cắt gradient theo CHUẨN TOÀN CỤC. 1.0 là giá trị chốt — xem docstring
    # tại chỗ gọi clip_grad_norm_ trong src/train.py.
    run.setdefault("max_grad_norm", 1.0)
    run["max_grad_norm"] = float(run["max_grad_norm"])
    if run["max_grad_norm"] <= 0:
        raise ValueError(
            f"run.max_grad_norm phải > 0, nhận {run['max_grad_norm']}. Muốn "
            f"TẮT cắt gradient thì đặt một số rất lớn (vd 1e9) — để 0 thì "
            f"clip_grad_norm_ nhân mọi gradient với 0 và không có gì học "
            f"được, im lặng.")
    if run["max_grad_norm"] > 10:
        print(f"⚠ max_grad_norm={run['max_grad_norm']:g} — gần như không cắt "
              f"gì. Router train từ số 0 và ASL có thể cho gradient rất lớn ở "
              f"vài trăm step đầu; 1.0 là giá trị chốt.")

    for _k in ("asl_gamma_pos", "asl_gamma_neg", "asl_clip", "uncertain_soft"):
        run[_k] = float(run[_k])
    if not 0.0 <= run["asl_clip"] < 0.5:
        raise ValueError(f"run.asl_clip phải trong [0, 0.5), nhận "
                         f"{run['asl_clip']}. Đây là NGƯỠNG ÂM TÍNH: xác suất "
                         f"mà dưới đó một ô âm coi như đã đúng và hết "
                         f"gradient.")
    if run["asl_clip"] > 0.05:
        print(f"⚠ asl_clip={run['asl_clip']:g} > 0.05 (default của ASL) — "
              f"nghĩa là tha cho model ở mức bỏ sót cao hơn cả ngưỡng của một "
              f"bài báo thị giác máy tính phổ thông, chưa nói tới chuẩn đọc "
              f"phim (BI-RADS 3: ≤ 0.02).")
    if not 0.0 <= run["uncertain_soft"] <= 1.0:
        raise ValueError(f"run.uncertain_soft phải trong [0, 1], nhận "
                         f"{run['uncertain_soft']} (1 = dương, 0 = âm)")
    if run["asl_gamma_pos"] > run["asl_gamma_neg"]:
        print(f"⚠ asl_gamma_pos={run['asl_gamma_pos']:g} > asl_gamma_neg="
              f"{run['asl_gamma_neg']:g} — ASL đang bị lật ngược: nó sẽ dập "
              f"gradient của ô DƯƠNG và giữ ô âm, tức phạt nhẹ chuyện bỏ sót "
              f"bệnh. Default là 0 và 4.")

    # ── mode="probe": khảo sát chuyên môn expert ─────────────────────────────
    run.setdefault("probe_repo", None)        # null -> "<hub_repo>-probe"
    run.setdefault("probe_commit_every", 1.0)   # phút, CommitScheduler
    run.setdefault("probe_top_tokens", 10)
    run.setdefault("probe_log_every", 20)     # ảnh
    run.setdefault("probe_flush_every", 20)   # ảnh, ghi lại reports.csv
    run.setdefault("probe_n_images", None)    # null = TRỌN tập test
    # Số ảnh ĐẦU của tập test được probe với ĐỦ n_expert expert (1+E forward)
    # thay vì chỉ top-k (1+K forward). Chỉ những ảnh này cho hồ sơ vô điều
    # kiện — bảng chuyên môn và cosine trong báo cáo dựng trên chúng.
    #   300  mặc định: đủ để trung bình hết nhiễu, probe không chậm gấp 2.6×
    #        trên trọn tập.
    #   0    tắt hẳn, quay về hành vi cũ (chỉ top-k, không có hồ sơ vô điều kiện)
    #   null MỌI ảnh chạy đủ E expert — chính xác nhất, chậm nhất.
    run.setdefault("probe_full_n", 300)
    if str(run.get("mode", "both")) == "probe":
        # Probe không train và không được bỏ qua infer vì "ô này đã có eval":
        # eval cũ là của mode=test, không có bản ghi probe nào.
        run["force_eval"] = True
        # n_test_samples là trần của mode=test (200 mẫu để chấm metrics nhanh).
        # Probe chạy TRỌN test, trừ khi probe_n_images ghi số cứng.
        run["n_test_samples"] = run.get("probe_n_images") or 0

    # ── w_aux: hệ số của số hạng cân bằng tải (Switch-Transformer) ──────────
    # L_aux = E·Σ f_e·P_e ≥ 1, = 1 khi cân bằng hoàn hảo (src/losses.py).
    # Nó THAY HẲN cơ chế bias aux-free của bản trước: không còn núm nào cộng
    # vào điểm chọn top-k, nên `chosen` lúc probe và top-k lúc train là cùng
    # một hàm của route_logit.
    run["w_aux"] = float(run["w_aux"])
    if run["w_aux"] < 0:
        raise ValueError(f"run.w_aux phải >= 0, nhận {run['w_aux']}")
    if run["w_aux"] == 0:
        print("⚠ w_aux=0 — TẮT cân bằng tải. Không còn bias, không còn trần "
              "tải, nên không còn gì giữ router khỏi dồn hết vào vài expert; "
              "chỉ dùng khi cố ý đo hiện tượng đó.")
    if run["w_aux"] > 0.5:
        print(f"⚠ w_aux={run['w_aux']:g} rất lớn — L_aux có sàn 1.0 nên nó "
              f"cộng thẳng {run['w_aux']:g} vào Total_Loss, và ở mức này nó "
              f"lấn át CE. Chuẩn phổ thông là 0.01–0.05.")
    run["w_sup"] = float(run["w_sup"])
    if run["w_sup"] < 0:
        raise ValueError(f"run.w_sup phải >= 0, nhận {run['w_sup']}")

    # ── Nhiễu Gumbel trên route logit khi CHỌN top-k ────────────────────────
    # σ = 1.0 là lấy mẫu ĐÚNG CHUẨN Gumbel-top-k. Giá trị ở đây là σ SAU
    # warmup (xem decay_warmup_samples ngay dưới).
    fixed.setdefault("route_noise", 0.5)
    fixed["route_noise"] = float(fixed["route_noise"])
    if fixed["route_noise"] < 0:
        raise ValueError(f"fixed.route_noise phải >= 0, nhận "
                         f"{fixed['route_noise']}")
    if fixed["route_noise"] > 2.0:
        print(f"⚠ route_noise={fixed['route_noise']:g} rất lớn — σ ≳ 2 thì "
              f"top-k gần như bốc ngẫu nhiên đều, gate hết mang thông tin.")
    if fixed["route_noise"] == 0:
        print("⚠ route_noise=0 sau warmup — top-k thành tất định hoàn toàn. "
              "Từ khi bỏ bias, Gumbel là cơ chế KHÁM PHÁ DUY NHẤT còn lại: "
              "expert xếp thứ k+1 sẽ không bao giờ được chạy nữa, và aux "
              "loss không sửa được điều đó (nó chỉ nắn qua P_e).")

    # ── Lịch decay MŨ của σ trong n MẪU đầu ─────────────────────────────────
    # `route_noise` ở trên là giá trị CUỐI; `route_noise_start` là giá trị ở
    # mẫu 0. `route_noise_start: null` hoặc `decay_warmup_samples: 0` -> tắt
    # lịch, dùng thẳng giá trị cuối.
    #
    # ĐẾM BẰNG MẪU, không bằng step — đổi so với bản trước, cùng lý do đã
    # buộc lịch LR đếm bằng mẫu: step phụ thuộc train_bs × grad_accum của
    # từng máy, còn mẫu thì không.
    fixed.setdefault("decay_warmup_samples", 3200)
    fixed["decay_warmup_samples"] = int(fixed["decay_warmup_samples"])
    if fixed["decay_warmup_samples"] < 0:
        raise ValueError(f"fixed.decay_warmup_samples phải >= 0, nhận "
                         f"{fixed['decay_warmup_samples']}")
    fixed.setdefault("route_noise_start", None)
    if fixed["route_noise_start"] is not None:
        fixed["route_noise_start"] = float(fixed["route_noise_start"])
        if fixed["route_noise_start"] < 0:
            raise ValueError(f"fixed.route_noise_start phải >= 0 hoặc null, "
                             f"nhận {fixed['route_noise_start']}")
    if (fixed["route_noise_start"] is not None
            and fixed["decay_warmup_samples"] == 0):
        print("⚠ có route_noise_start nhưng decay_warmup_samples=0 — lịch "
              "decay KHÔNG chạy, σ dùng thẳng giá trị cuối từ mẫu 0.")
    if (fixed["route_noise_start"] is not None
            and fixed["route_noise_start"] < fixed["route_noise"]):
        print(f"⚠ route_noise_start={fixed['route_noise_start']:g} < "
              f"route_noise={fixed['route_noise']:g} — lịch chạy NGƯỢC (σ "
              f"tăng dần). Ý đồ là σ lớn lúc đầu để tuyến đường còn đảo "
              f"được, rồi mới siết lại.")

    # ── Hình học của router: BỊ BUỘC bởi số bậc downsample ─────────────────
    # 5 bậc, mỗi bậc chia đôi cạnh và nhân bốn kênh, nên
    #     cạnh img/2^5 = 16 -> 16×16 = 256 token
    #     kênh 1 × 4^5      = 1024   = chiều làm việc
    # Xem docstring src/router.py.
    fixed.setdefault("router_img_size", IMG_SIZE)
    fixed["router_img_size"] = int(fixed["router_img_size"])
    _grid = fixed["router_img_size"] // STRIDE
    if _grid * STRIDE != fixed["router_img_size"] or _grid * _grid != N_TOKEN:
        raise ValueError(
            f"fixed.router_img_size={fixed['router_img_size']} cho lưới "
            f"{_grid}×{_grid} = {_grid*_grid} token, nhưng nhánh router được "
            f"dựng cho {N_TOKEN} token ({N_BLOCK} bậc downsample, tổng stride "
            f"{STRIDE}). Giá trị hợp lệ duy nhất là "
            f"{int(round(N_TOKEN ** 0.5)) * STRIDE}.")
    # §9 — container GPU thuê: dời cache/outputs/ckpt sang volume lớn. Làm
    # TRƯỚC khi ghép tag vào out_dir để mỗi ô vẫn có thư mục riêng.
    rm = read_remote(path)
    if rm.active:
        _apply_remote(rm, run, explicit)

    # ── Giải quyết cấu hình một lần chạy ────────────────────────────────────
    if arch["model"] not in raw["models"]:
        raise ValueError(f"model lạ: {arch['model']} "
                         f"(chọn: {list(raw['models'])})")

    mdl = raw["models"][arch["model"]]
    fixed["model_id"] = mdl["model_id"]
    fixed["lora_exclude"] = mdl["exclude"]
    # §4.1.1 — mỗi model MỘT processor riêng. Gemma 3 buộc 896² (SigLIP có
    # bảng position embedding cố định); Qwen2.5-VL phân giải động nên nhận
    # thẳng ảnh gốc. image_size = null -> không resize, processor tự lo.
    fixed["image_size_model"] = mdl.get("image_size")
    fixed["processor_kwargs"] = dict(mdl.get("processor_kwargs") or {})
    fixed["lora_alpha"] = fixed["lora_r"] * fixed["alpha_ratio"]
    # Adapter đã chốt: QDoRA = LoRA + DoRA + base NF4.
    fixed["use_dora"] = True
    run["quant"] = True
    if not 1 <= int(fixed["top_k"]) <= int(fixed["n_expert"]):
        raise ValueError(
            f"top_k={fixed['top_k']} phải nằm trong [1, n_expert="
            f"{fixed['n_expert']}] — top-k thuần trên CẢ E expert, không còn "
            f"expert hứng nào bị loại khỏi vòng chọn (xem src/router.py).")

    tag = f"{arch['model']}-{ARCH_NAME}"
    if run.get("hub_repo") is None:
        run["hub_repo"] = f"RGExp123/mimic-cxr-{tag}"
    # Bản ghi probe là DATASET, không phải model: để chung repo với checkpoint
    # thì `list_repo_files` phải lọc lẫn giữa weights.pt và vài nghìn json, và
    # ai tải ckpt cũng kéo theo cả đống bản ghi. Mặc định tách hẳn một repo.
    if run.get("probe_repo") is None:
        run["probe_repo"] = f"{run['hub_repo']}-probe"
    if run.get("wandb_run") is None:
        run["wandb_run"] = tag
    # Mỗi lần chạy có thư mục riêng theo tag: đổi model không đè checkpoint.
    run["out_dir"] = os.path.join(run["out_dir"], tag)
    run["ckpt_dir"] = os.path.join(run["ckpt_dir"], tag)

    # null trong YAML = để preset quyết định
    for key in ("train_bs", "grad_accum", "infer_bs", "grad_ckpt"):
        if run.get(key) is None:
            run[key] = preset[key]

    # ── Lịch LR: cosine warm restarts VÔ HẠN (src/optim.py) ────────────────
    # Đặt SAU khối preset vì mọi kiểm tra ở đây đều so với effective batch, mà
    # train_bs/grad_accum có thể còn là null cho tới dòng trên.
    run.setdefault("cycle_samples", 1600)
    run.setdefault("warmup_samples", 320)
    run.setdefault("min_lr_ratio", 0.0)
    run.setdefault("cycle_lr_decay", 1.0)
    # Số revision tối đa được giữ trên repo checkpoint. HF tính quota theo mọi
    # blob LFS còn trong lịch sử, nên ghi đè last/ KHÔNG giải phóng gì; tới
    # revision thứ N+1, train.py gộp lịch sử còn một commit (HEAD giữ nguyên).
    # 0 / null = tắt hẳn, lịch sử để nguyên và quota cứ thế tăng.
    run.setdefault("hub_keep_revisions", 7)
    run["hub_keep_revisions"] = int(run["hub_keep_revisions"] or 0)
    if run["hub_keep_revisions"] < 0:
        raise ValueError(
            f"run.hub_keep_revisions phải >= 0, nhận "
            f"{run['hub_keep_revisions']}. 0 = không bao giờ gộp lịch sử.")
    eff = run["train_bs"] * run["grad_accum"]
    cs = int(run["cycle_samples"] or 0)
    if cs <= 0:
        raise ValueError(
            "run.cycle_samples phải > 0. Lịch train không có đích nên không "
            "còn tổng số mẫu nào để suy ra độ dài một cycle — nó phải được "
            "ghi thẳng trong config.")
    if cs < eff:
        raise ValueError(
            f"cycle_samples={cs} nhỏ hơn effective batch {eff}: một cycle "
            f"không đủ một optimizer step, LR sẽ nhảy cả chu kỳ trong một "
            f"lần cập nhật.")
    if cs % eff:
        print(f"⚠ cycle_samples={cs} không chia hết cho effective batch "
              f"{eff} — biên cycle rơi vào giữa một step, lệch tối đa {eff} "
              f"mẫu. Dùng bội của {eff} (vd "
              f"{max(round(cs / eff), 1) * eff}) nếu muốn khớp chính xác.")
    if int(run["warmup_samples"]) >= cs:
        raise ValueError(
            f"warmup_samples={run['warmup_samples']} >= cycle_samples={cs}: "
            f"cycle đầu không còn đoạn cosine nào.")
    run["cycle_samples"] = cs
    if float(run["cycle_lr_decay"]) != 1.0:
        print(f"⚠ cycle_lr_decay={run['cycle_lr_decay']} với lịch VÔ HẠN: "
              f"đỉnh LR tắt dần về 0 và train sẽ tự chết. Giữ 1.0 trừ khi "
              f"bạn cố ý muốn thế.")

    import torch
    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32,
                 "float16": torch.float16}
    fixed["compute_dtype"] = dtype_map[fixed["compute_dtype"]]
    fixed["trainable_dtype"] = dtype_map[fixed["trainable_dtype"]]
    fixed["attn_impl"] = pick_attn_impl(hw.sm, _dtype_name)

    cfg = SimpleNamespace(
        fixed=SimpleNamespace(**fixed),
        run=SimpleNamespace(**run),
        data=SimpleNamespace(**run["data"]),
        arch=SimpleNamespace(**arch, tag=tag),
        hw=hw, preset=preset, remote=rm,
    )
    for d in (cfg.run.out_dir, cfg.run.ckpt_dir):
        os.makedirs(d, exist_ok=True)
    return cfg


def cycles_desc(cfg):
    """Một dòng mô tả lịch LR. Không còn "bao nhiêu cycle cả phase": lịch
    train không có đích, cycle cứ lặp cho tới khi người dùng dừng tay."""
    r = cfg.run
    eff = r.train_bs * r.grad_accum
    s = (f"lịch LR: cosine warm restarts VÔ HẠN | cycle = {r.cycle_samples} "
         f"mẫu ({r.cycle_samples // eff} step ở batch {eff}) | warmup "
         f"{r.warmup_samples} mẫu ở đầu MỌI cycle")
    if r.min_lr_ratio:
        s += f" | đáy = {r.min_lr_ratio:g}×lr"
    return s


def epoch_desc(cfg, n_train, samples):
    """`samples` tích luỹ -> đọc thành (epoch thứ mấy, đang ở đâu trong epoch)."""
    n = max(int(n_train), 1)
    return int(samples) // n, int(samples) % n


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
        description="MIMIC-CXR MoE-LoRA — một phase multi-task, lịch vô hạn")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="Ghi đè config, vd: --set run.cycle_samples=3200")
    ap.add_argument("--model", choices=["gemma3", "qwen25vl"])
    # `--variant` / `--n-block` đã BỎ: kiến trúc router chốt hẳn, không còn
    # siêu tham số nào để quét. Gọi kèm hai cờ đó thì argparse báo lỗi ngay
    # thay vì im lặng bỏ qua — đúng ý, vì script cũ gọi chúng sẽ chạy MỘT
    # kiến trúc khác hẳn với cùng dòng lệnh.
    ap.add_argument("--mode", choices=["train", "test", "both", "probe"],
                    help="probe = không train, nạp last/ từ Hub, sinh báo "
                         "cáo rồi bật lần lượt từng expert trong top-k để đo "
                         "chuyên môn (xem src/probe.py).")
    # dry_run mặc định false trong config (chạy thật). Hai cờ này để ép:
    ap.add_argument("--dry-run", action="store_true",
                    help="Dừng sau STAGE 0 (2 step thử) để kiểm cấu hình")
    ap.add_argument("--no-dry-run", action="store_true",
                    help="Ép chạy thật (ghi đè config)")
    return ap


def config_from_args(argv=None):
    args = build_argparser().parse_args(argv)
    ov = list(args.overrides)
    for name, key in (("model", "arch.model"), ("mode", "run.mode")):
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
    est = estimate_params(c.fixed.n_expert)
    print(f"Lần chạy: {c.arch.tag}")
    print(f"GPU     : {c.hw.name} | {c.hw.vram_gb:.0f}GB | "
          f"sm{c.hw.sm[0]}{c.hw.sm[1]} | {c.hw.n_gpu} GPU")
    print(f"Preset  : {c.preset['name']}")
    print(f"Model   : {c.fixed.model_id} | attn={c.fixed.attn_impl}")
    print(f"Router  : {arch_summary(c)}")
    _g = c.fixed.router_img_size // STRIDE
    print(f"          ảnh {c.fixed.router_img_size}² -> {N_BLOCK} bậc -> lưới "
          f"{_g}×{_g} = {_g*_g} token × {D_MODEL} chiều (train từ số 0) | "
          f"prefix {N_TOKEN} token vào decoder")
    print(f"          router {est['router']:,} + expert "
          f"{est['experts']:,} = {est['total']:,} param trainable")
    print(f"Adapter : {adapter_desc(c.fixed)} | quant={c.run.quant}")
    print(f"Ảnh     : {image_desc(c.fixed)}")
    print(f"Batch   : {c.run.train_bs} × {c.run.grad_accum} = "
          f"{c.run.train_bs * c.run.grad_accum} mẫu/step")
    print("Lịch    : MỘT phase multi-task, KHÔNG có đích — chạy tuần tự tập "
          "train, hết thì epoch += 1 và chạy tiếp")
    print(f"Loss    : 1·CE + {c.run.w_sup}·ASL + {c.run.w_aux}·Aux")
    print(f"          ASL γ+={c.run.asl_gamma_pos:g} γ-={c.run.asl_gamma_neg:g}"
          f" | ngưỡng âm m={c.run.asl_clip:g} (BI-RADS 3)"
          f" | uncertain={c.run.uncertain_soft:g}, blank=MASK")
    print(f"Cân bằng: aux loss w={c.run.w_aux:g} (Switch: E·Σf·P, sàn 1.0) — "
          f"KHÔNG còn bias, trần tải hay expert hứng")
    _sig = decay_desc(getattr(c.fixed, "route_noise_start", None),
                      c.fixed.route_noise, c.fixed.decay_warmup_samples)
    print(f"Gumbel  : {_sig}")
    print(f"LR      : {cycles_desc(c)}")
    print(f"          lr_router {c.run.lr_router:.1e} | lr_expert "
          f"{c.run.lr:.1e} | đáy = {c.run.min_lr_ratio:.3g}× | clip chuẩn "
          f"toàn cục {c.run.max_grad_norm:g}")
    print(f"Val     : mỗi {c.run.eval_every} step ({c.run.eval_max_batches} "
          f"batch) | ckpt mỗi {c.run.save_every_n_vals} lần val | dừng sớm: "
          f"KHÔNG (chạy tới khi dừng tay)")
    _keep = c.run.hub_keep_revisions
    print(f"Output  : {c.run.out_dir} | hub={c.run.hub_repo} | lịch sử Hub: "
          + (f"giữ {_keep} revision rồi gộp" if _keep else "để nguyên"))
    print(f"Mode    : {c.run.mode} | dry_run={c.run.dry_run} | "
          f"policy={'BẬT' if c.run.use_policy else 'TẮT (chấm raw_output)'}")
    if c.remote.active:
        print(f"Remote  : BẬT (§9) | venv={'có' if c.remote.use_venv else 'không'}"
              f" | tmux={c.remote.tmux_session}")
        print(f"Cache   : HF_HOME={os.environ.get('HF_HOME', '(mặc định)')} | "
              f"ảnh={c.data.image_cache} | ckpt={c.run.ckpt_dir}")
    else:
        print("Remote  : tắt (chạy cục bộ — option 1)")

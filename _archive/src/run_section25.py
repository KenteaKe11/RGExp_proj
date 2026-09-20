"""§2.5 — đối chứng với SOTA. Trọn một thí nghiệm trong MỘT lệnh.

    python -m src.run_section25                      # data -> … -> push
    python -m src.run_section25 --dry-run            # 8 ảnh, 16 mẫu, 4 bước
    python -m src.run_section25 --stages stats,report
    python -m src.run_section25 --arms my_gemma
    python -m src.run_section25 --init-from hub:OWNER/NAME/latest
    python -m src.run_section25 --init-from none     # My Gemma KHÔNG pretrain
    python -m src.run_section25 --no-resume          # decoder_ft My Gemma làm LẠI từ đầu
    python -m src.run_section25 --mode comparer_pretrain   # CHỈ R2Gen+AdaAtt -> Hub
    python -m src.run_section25 --mode infer_test          # sinh+chấm, ckpt kéo từ Hub

Mã thoát:
    0  xong
    2  tham số/cấu hình sai
    3  một candidate không huấn luyện được
    4  rò rỉ train/test
    5  nhãn phân loại không dùng được
    6  một candidate không sinh được
    7  stage `report` hỏng   -> chạy lại `--stages report`, kết quả còn nguyên
    8  bước đẩy Hub hỏng     -> chạy lại `--stages push`
    9  thiếu checkpoint Stage 0 của My Gemma -> chạy `scripts/run_pretrain.sh`
    10 `infer_test`: một candidate không có checkpoint (cục bộ lẫn Hub)

── BA CANDIDATE, HAI HỌ CODE, MỘT ĐIỀU KIỆN ────────────────────────────────
`r2gen` và `adaatt` đi qua `src/train_sota.py` + `src/runners/sota.py`;
`my_gemma` đi qua `src/train_recomb.py` + `src/runners/gemma3_recomb.py`. Hai
đường huấn luyện khác nhau là BẮT BUỘC — một ResNet+LSTM không huấn luyện được
bằng vòng lặp LoRA của một VLM 4-bit — nhưng mọi thứ ở NGOÀI mô hình thì không:
cùng `data`, cùng ảnh đã qua `image_policy`, cùng 3000 ảnh chấm, cùng
`src/metrics.py` theo từng mẫu, cùng `src/stats_multi.py`.

File này là chỗ ranh giới ấy được giữ, nên nó nói ra ranh giới ấy ở hai chỗ:
`_deviations()` ghi mọi thứ §2.5 áp lên setup gốc, và stage `stats` DỪNG nếu
thiếu bất kỳ cột nào (ba cặp đòi đủ ba cột — thiếu một là đổi thiết kế trong
khi bảng vẫn in ra bình thường).

── `--no-resume` — decoder_ft làm lại từ đầu ────────────────────────────────
Mặc định mọi stage ĐI TIẾP từ đĩa: adapter đã có -> bỏ qua train, predictions
đã có -> chỉ sinh phần thiếu, per_sample đã có -> dùng lại. Với decoder_ft của
My Gemma, `--no-resume` bỏ qua cả ba: checkpoint decoder_ft, predictions và
điểm của candidate `ours` được ĐỔI TÊN sang `*.old_<thời điểm>` (không xoá),
rồi train (vẫn nạp Stage 0) -> sinh -> chấm lại từ đầu. `--retrain` thì chỉ
huấn luyện lại và giữ predictions cũ — sinh tiếp trên file của model CŨ.
Hai baseline không bị đụng; muốn làm lại chúng thì `--retrain`.

── `--mode` — tách hai baseline khỏi phần còn lại ──────────────────────────
    full               (mặc định) data -> train -> infer -> score -> stats ->
                       report -> push, cả ba candidate — như trước.
    comparer_pretrain  CHỈ R2Gen + AdaAtt: data -> train -> push, trong đó
                       `push` đẩy CHECKPOINT (model_best.pt + train_stats.json +
                       tokenizer.json + hub_meta.json) lên repo MODEL
                       `sota_hub.repo`, mỗi candidate ngay khi nó huấn luyện
                       xong. Candidate đã có checkpoint cục bộ thì không huấn
                       luyện lại mà chỉ đẩy (--retrain để làm lại). Không đụng
                       predictions/metrics/manifest.json của run chính.
    infer_test         data -> infer -> score -> stats -> report -> push, KHÔNG
                       huấn luyện gì. Checkpoint baseline thiếu cục bộ thì kéo
                       từ `sota_hub.repo`; vẫn thiếu thì DỪNG (mã 10) thay vì
                       lặng lẽ huấn luyện.
Ở `full`, baseline thiếu checkpoint cục bộ cũng được kéo từ Hub TRƯỚC khi quyết
định huấn luyện; không kéo được thì huấn luyện như cũ. `--no-pull` tắt việc kéo.
Chi tiết gói checkpoint: `src/sota_hub.py`.

── VÌ SAO STAGE 0 CỦA MY GEMMA KHÔNG CHẠY Ở ĐÂY ────────────────────────────
`sup_pretrain` có điểm vào riêng (`scripts/run_pretrain.sh`), ghi ra
`runs/<pretrain.subdir>/ckpt/latest`, và nó chạy trên TOÀN BỘ train split. Gộp nó
vào đây nghĩa là mỗi lần chạy lại §2.5 để in một bảng cũng kéo theo 5 epoch
pretrain — và tệ hơn, `--arms my_gemma` sẽ lặng lẽ pretrain LẠI trên một run
dir khác. §2.5 chỉ NẠP checkpoint ấy và dừng nếu không thấy (mã 9).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import (Cfg, add_common_args, hf_token, load_config,
                     resolved_snapshot, seed_everything)
from .tracking import Tracker

STAGES = ["data", "train", "infer", "score", "stats", "report", "push"]
KINDS = ("sota", "ours")
# Stage mặc định của từng mode, và những stage mode ấy CHO PHÉP khai qua --stages.
MODES = {
    "full": (STAGES, STAGES),
    "comparer_pretrain": (["data", "train", "push"], ["data", "train", "push"]),
    "infer_test": ([s for s in STAGES if s != "train"],
                   [s for s in STAGES if s != "train"]),
}
DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "section25.yaml"


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


def _read_preds(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[r["key"]] = r.get("pred", "")
    return out


def _count_preds(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except Exception:
                continue
            n += 1
    return n


def _env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "torchvision", "transformers", "peft", "bitsandbytes",
                "safetensors"):
        try:
            info[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
    except Exception:
        pass
    return info


# ── candidate ───────────────────────────────────────────────────────────────
def enabled_arms(cfg: Cfg, only: set[str] | None = None) -> list[dict]:
    """`arms_s25` đã kiểm — mọi cờ phân biệt candidate đều BẮT BUỘC.

    Không cờ nào có mặc định ngầm, cùng lý lẽ với `recomb` của §2.3: một arm
    quên khai `kind` mà lặng lẽ nhận "sota" nghĩa là candidate C được huấn
    luyện bằng đường của baseline, và bảng vẫn có đủ ba cột.
    """
    raw = cfg.get("arms_s25") or []
    if not raw:
        raise ValueError(
            "config không có `arms_s25`. Đó là danh sách candidate của §2.5 — "
            "đọc nhầm `arms` (khoá của §2.3, nay ở _archive/) sẽ cho ra hai "
            "candidate của một thí nghiệm khác với cùng số cột.")
    arms = []
    for a in raw:
        a = dict(a)
        aid = a.get("id")
        if not a.get("enabled", True):
            continue
        kind = str(a.get("kind", "")).lower()
        if kind not in KINDS:
            raise ValueError(
                f"arm {aid!r} khai `kind: {a.get('kind')!r}`; hợp lệ: "
                f"{KINDS}. Cờ này chọn ĐƯỜNG HUẤN LUYỆN, nên một mặc định ngầm "
                f"ở đây là huấn luyện nhầm kiến trúc mà bảng vẫn đủ cột.")
        a["kind"] = kind
        if not a.get("runner"):
            raise ValueError(f"arm {aid!r} không khai `runner`.")
        if kind == "sota":
            setup = str(a.get("setup") or aid)
            if setup not in (cfg.get("sota") or {}):
                raise ValueError(
                    f"arm {aid!r} trỏ `setup: {setup}` nhưng config không có "
                    f"khối `sota.{setup}` — đó là bản chép setup paper gốc, và "
                    f"thiếu nó thì cột đối chứng được dựng bằng mặc định của "
                    f"code.")
            a["setup"] = setup
        else:
            for key in ("top_k", "n_experts"):
                if key not in a:
                    raise ValueError(
                        f"arm {aid!r} (kind=ours) không khai `{key}`. §2.4 đã "
                        f"chốt E=10·K=6 và §2.5 dùng lại con số ấy — nhưng nó "
                        f"phải được VIẾT RA, không phải rơi vào mặc định của "
                        f"`recombiner.n_experts`.")
            a["recomb"] = bool(a.get("recomb", True))
        arms.append(a)
    if only:
        arms = [a for a in arms if a["id"] in only]
    return arms


def dry_n_train(cfg: Cfg) -> int:
    """Số mẫu train của dry-run: >= batch paper lớn nhất (AdaAtt 80), >= 16."""
    from .train_sota import main_stage_spec

    return max([16] + [int(main_stage_spec(s, dict(v or {}))["batch_size"])
                       for s, v in (cfg.get("sota") or {}).items()
                       if isinstance(v, dict) and v.get("fine_tuning")])


def train_order(arms: list[dict]) -> list[dict]:
    """Candidate dễ hỏng nhất TRƯỚC — quy ước 1 của repo.

    `order` khai thẳng trong config (my_gemma = 0). Không có `order` thì `ours`
    đi trước `sota`, rồi theo id. Thứ tự này KHÔNG được phép ảnh hưởng kết quả:
    seed đặt lại trước MỖI candidate, batch hiệu dụng là hằng số trong config,
    VRAM được dọn giữa hai candidate.
    """
    return sorted(arms, key=lambda a: (int(a.get("order", 1 if a["kind"] == "ours" else 2)),
                                       str(a["id"])))


def arm_model_cfg(cfg: Cfg, arm: dict, adapter_root: Path, run_dir: Path) -> Cfg:
    """Cấu hình kiểu `models[]` để dùng lại `Runner.run()` (resume, ngắt mạch)."""
    b = dict(cfg["base"])
    aid = arm["id"]
    if arm["kind"] == "ours":
        return Cfg({
            "id": aid, "name": arm.get("name", aid), "runner": "gemma3_recomb",
            "hf_id": b["hf_id"], "dtype": b.get("dtype", "bfloat16"),
            "batch_size": int(b.get("batch_size", 2)),
            "disable_compile": b.get("disable_compile", "auto"),
            "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
            "adapter_dir": str(adapter_root / aid),
            "merge_adapter": False, "optional": False,
        })
    setup = arm["setup"]
    st = dict(cfg.get("sota_train", {}) or {})
    return Cfg({
        "id": aid, "name": arm.get("name", aid), "runner": arm["runner"],
        "ckpt": str(adapter_root / aid / "model_best.pt"),
        # Từ điển đi CÙNG checkpoint khi có (gói kéo từ Hub, hoặc vừa huấn
        # luyện); run_dir/tokenizer.json chỉ là đường lui cho checkpoint cũ.
        "tokenizer": str(adapter_root / aid / "tokenizer.json"
                         if (adapter_root / aid / "tokenizer.json").exists()
                         else run_dir / "tokenizer.json"),
        # Giải mã KHÔNG đọc `sota.*.decoding` — runner dùng `decoding_spec`
        # (khối `generate:` chung). Batch của khâu SINH: beam nhân số chuỗi lên
        # `num_beams` lần, nên con số an toàn ở đây nhỏ hơn micro_batch.
        "batch_size": max(1, int(st.get("infer_batch", st.get("micro_batch", 8)))),
        "disable_compile": True, "optional": False,
    })


def _deviations(cfg: Cfg, arms: list[dict]) -> list[str]:
    """Mọi thứ §2.5 ÁP LÊN setup gốc — đọc được, và đi thẳng vào REPORT.md.

    Điều kiện 4 và 5 sửa setup của R2Gen/AdaAtt. Đó là cái giá bắt buộc của một
    bảng so sánh công bằng, nhưng một sai lệch không ghi lại thì bảng không đọc
    được. Danh sách này được dựng từ CHÍNH config đang chạy, nên nó không thể
    lạc hậu so với thí nghiệm.
    """
    from . import imaging

    size = int((cfg.get("image_policy") or {}).get("resize_to", 512))
    # ResNet hạ mẫu 32 lần: lưới đặc trưng là (size/32)². Tính ra chứ không viết
    # cứng "16×16" — một con số viết cứng sẽ nói dối ngay lần đầu ai đó đổi
    # `resize_to`, và nó nằm trong đúng mục mà người đọc dùng để tin bảng.
    grid = max(1, size // 32)
    out = [f"ĐIỀU KIỆN 4 — mọi candidate nhận ảnh qua MỘT phép biến đổi hình "
           f"học dùng chung ({imaging.describe_unified(cfg)}). Hai baseline "
           f"quen 224px trong paper gốc (lưới 7×7); ở đây chúng nhận {size}px, "
           f"nên lưới đặc trưng ResNet là {grid}×{grid} = {grid * grid} vùng."]
    p = dict(cfg.get("pretrain", {}) or {})
    mult = float((cfg.get("train") or {}).get("pretrained_lr_mult", 0.1))
    for a in arms:
        if a["kind"] != "sota":
            continue
        ft = dict((cfg["sota"][a["setup"]].get("fine_tuning", {}) or {}))
        if "stage_1" in ft:
            out.append(
                f"ĐIỀU KIỆN 5 — {a['id']}: `fine_tuning.stage_1` của paper "
                f"(huấn luyện phần ngôn ngữ, đóng băng CNN) được THAY bằng "
                f"Stage 0 thống nhất ({p.get('n_epochs')} epoch, max_lr "
                f"{(p.get('schedule') or {}).get('max_lr')}, wd "
                f"{p.get('weight_decay')}). lr gốc của stage 1 không được dùng.")
        else:
            out.append(
                f"ĐIỀU KIỆN 5 — {a['id']}: paper gốc KHÔNG có stage pretrain "
                f"cục bộ (`strategy: {ft.get('strategy')}`). Stage 0 thống nhất "
                f"được thêm vào từ ngoài, rồi main-stage chạy đúng setup gốc.")
    tc = dict(cfg.get("train") or {})
    ftl = dict(tc.get("ft_lr") or {})
    tsch = dict(tc.get("schedule") or {})
    amult = float((cfg.get("recombiner") or {}).get("attn_lr_mult", 3.0))
    frozen = tc.get("freeze_after_pretrain") or []
    if isinstance(frozen, str):
        frozen = [frozen]
    if tc.get("freeze_pretrained"):
        frozen = ["recombinator", "cls_head"]
    out.append(
        f"ĐIỀU KIỆN 5 — phần đã qua Stage 0 của hai baseline chạy main-stage ở "
        f"{mult:g}× lr chính thức (`train.pretrained_lr_mult`). My Gemma "
        f"Stage 0 cho ma trận Attn của Recombinator lr × {amult:g} "
        f"(`recombiner.attn_lr_mult`). decoder_ft fine-tune TOÀN BỘ, lịch lr "
        f"`train.ft_lr`: lr_max toàn hệ {float(ftl.get('max_lr', 0)):.0e}, "
        f"Retriever i=1..n tăng dần theo cấp số nhân từ "
        f"{float(ftl.get('retriever_first', 0)):.0e} (i=1, kèm Learnable "
        f"Query) lên lr_max (i=n)"
        + f"; adapter (LoRA/Experts) {float(tsch.get('max_lr', 0)):.0e}; MỘT chu "
        f"kỳ one-cycle cho cả stage, đầu warm-up và đáy làm tròn xuống "
        f"(`round_down_sig` {tsch.get('round_down_sig', 0)}), đáy nhỏ nhất "
        f"{float(tsch.get('min_lr', 0)):.0e}; đóng băng: "
        f"{', '.join(frozen) or 'không'} (`train.freeze_after_pretrain`).")
    st = dict(cfg.get("sota_train", {}) or {})
    out.append(
        f"KỸ THUẬT — batch của paper được chia thành nhiều lượt forward "
        f"(`sota_train.micro_batch` = {st.get('micro_batch')}); batch HIỆU DỤNG, "
        f"số bước và đường cong lr không đổi.")
    from . import decoding_spec

    diffs = {a["id"]: decoding_spec.paper_differences(cfg, a["setup"])
             for a in arms if a["kind"] == "sota"}
    out.append(
        f"GIẢI MÃ — cả ba candidate sinh bằng MỘT đặc tả (`generate:`, "
        f"{decoding_spec.describe(cfg)}); R2Gen/AdaAtt dùng bản chép lại "
        f"`_beam_search` của transformers 4.52 (selftest đối chiếu từng token), cả "
        f"khi chọn checkpoint trên val. Khối "
        f"`sota.*.decoding` của paper không được đọc"
        + ("; lệch so với paper: " + "; ".join(
            f"{k}: {', '.join(v)}" for k, v in diffs.items() if v)
           if any(diffs.values()) else "")
        + ". Độ dài tối đa khác nhau theo từ điển: My Gemma "
        f"{decoding_spec.unified(cfg)['max_new_tokens']} subword, hai baseline "
        f"`max_seq_length − 1` từ.")
    out.append(
        "R2GEN · LỊCH — scheduler theo setup MIMIC-CXR của chính paper "
        "(`train_mimic_cxr.sh`: StepLR step 1 · gamma 0.8) thay cho setup IU "
        "X-ray (step 50 · gamma 0.1); số epoch "
        f"{int(((cfg['sota'].get('r2gen') or {}).get('fine_tuning') or {}).get('epochs', 50))} "
        "(paper: 100 IU X-ray / 30 MIMIC) và patience dừng sớm "
        f"{int(((cfg['sota'].get('r2gen') or {}).get('fine_tuning') or {}).get('early_stopping', {}).get('patience', 5))} "
        "(mặc định repo: 50) đặt vì chi phí. Từ điển giữ nguyên setup trong "
        "config (max_seq_length 60, min_word_frequency 3).")
    out.append(
        "KHÔNG ĐỔI — ngoài các mục trên, R2Gen và AdaAtt chạy optimizer, lr "
        "và tiêu chí chọn checkpoint của paper gốc; My Gemma chạy "
        "lịch của §2.3/§2.4. Đó là một khác biệt THẬT giữa ba cột và nó thuộc "
        "phần Giới hạn, không phải một lỗi cần sửa.")
    return out


def ft_artifacts(arm: dict, adapter_root: Path, run_dir: Path) -> list[Path]:
    """Những gì §2.5 ĐỌC LẠI từ đĩa cho một candidate: checkpoint, sinh, điểm."""
    aid = arm["id"]
    return [adapter_root / aid,
            run_dir / "predictions" / f"{aid}.jsonl",
            run_dir / "metrics" / f"per_sample_{aid}.csv"]


def archive_ft(arms: list[dict], adapter_root: Path, run_dir: Path,
               stamp: str) -> list[tuple[Path, Path]]:
    """`--no-resume`: ĐỔI TÊN (không xoá) mọi thứ đọc lại được của decoder_ft.

    Chỉ candidate `ours` — hai baseline không có decoder_ft. Đổi tên chứ không
    xoá: một lần gõ nhầm cờ không được phép vứt hàng giờ GPU. Trả [(cũ, mới)].
    """
    moved: list[tuple[Path, Path]] = []
    for arm in arms:
        if arm.get("kind") != "ours":
            continue
        for src in ft_artifacts(arm, adapter_root, run_dir):
            if not src.exists():
                continue
            dst = src.with_name(f"{src.name}.old_{stamp}")
            i = 1
            while dst.exists():
                dst = src.with_name(f"{src.name}.old_{stamp}_{i}")
                i += 1
            src.rename(dst)
            moved.append((src, dst))
    return moved


# ════════════════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="§2.5 — R2Gen vs AdaAtt vs My Gemma, trọn gói.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--mode", default="full", choices=list(MODES),
                    help="full | comparer_pretrain (chỉ R2Gen+AdaAtt, đẩy ckpt "
                         "lên sota_hub.repo) | infer_test (không train, ckpt "
                         "baseline kéo từ Hub)")
    ap.add_argument("--stages", default=None,
                    help="mặc định theo --mode (full: " + ",".join(STAGES) + ")")
    ap.add_argument("--arms", default=None, help="chỉ chạy các arm này (phẩy ngăn cách)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--init-from", default=None,
                    help="nguồn Stage 0 cho My Gemma: thư mục | hub:OWNER/NAME/latest "
                         "| none (đối chứng from-scratch, PHẢI nói ra)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="decoder_ft của My Gemma làm LẠI từ đầu: bỏ qua (đổi tên "
                         "sang *.old_<ts>) checkpoint ft, predictions và điểm cũ")
    ap.add_argument("--no-pull", action="store_true",
                    help="không kéo checkpoint baseline từ sota_hub.repo")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides, getattr(a, "gpu", None))
    if str(cfg.get("experiment", "")) != "2.5":
        print(f"✗ config khai `experiment: {cfg.get('experiment')!r}`, chờ đợi "
              f"'2.5'. Điểm vào này đọc `arms_s25` và `sota`; một file cấu hình "
              f"của mục khác sẽ hoặc thiếu khoá, hoặc tệ hơn — có đủ khoá và "
              f"chạy một thí nghiệm khác.")
        return 2

    dry_steps = None
    if a.dry_run:
        # Số mẫu train >= batch paper lớn nhất (AdaAtt 80): ít hơn thì main-stage
        # không bao giờ đủ một bước optimizer, và VRAM của trạng thái Adam
        # không bao giờ xuất hiện. Batch MỖI LƯỢT giữ đúng config GPU.
        n_dry = dry_n_train(cfg)
        cfg["data"]["n_test"] = 8
        cfg["data"]["n_train"] = n_dry
        cfg["data"]["n_val"] = 8
        # decoder_ft: grad_accum 2 cho nhanh; mọi chuỗi đệm tới max_seq_len ->
        # VRAM đo được là trường hợp XẤU NHẤT, không phụ thuộc mẫu nào tình cờ
        # ngắn.
        cfg["train"]["grad_accum"] = min(2, int(cfg["train"]["grad_accum"]))
        cfg["train"]["_pad_to_max"] = True
        # Stage 0 của My Gemma do `run_pretrain --dry-run` ghi ra (`*_dryrun`).
        cfg["pretrain"]["subdir"] = str(cfg["pretrain"].get("subdir",
                                                           "sup_pretrain")) + "_dryrun"
        cfg["stats"]["n_perm"] = 200
        cfg["stats"]["n_boot"] = 200
        cfg["pretrain"]["n_epochs"] = 1
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] = str(cfg["output"].get("subdir", "run")) + "_dryrun"
        cfg["output"]["adapter_dir"] = str(
            cfg["output"].get("adapter_dir", "./adapters")) + "_dryrun"
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["dry-run"]
        dry_steps = 4
        print(f"⚙ DRY-RUN: 8 ảnh test · {n_dry} mẫu train · tối đa 4 bước · không "
              f"đẩy Hub · Stage 0 từ {cfg['pretrain']['subdir']}.\n  Batch mỗi lượt"
              f" = config GPU, decoder_ft đệm tới max_seq_len -> VRAM đỉnh in ở "
              f"cuối là trường hợp xấu nhất.", flush=True)
    if a.no_push:
        cfg["output"]["push"] = False
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed_everything(int(cfg.get("seed", 42)))
    mode = a.mode
    default_stages, allowed = MODES[mode]
    stages = ([s.strip() for s in a.stages.split(",") if s.strip()]
              if a.stages else list(default_stages))
    for s in stages:
        if s not in allowed:
            print(f"✗ stage {s!r} không hợp lệ ở --mode {mode} (hợp lệ: "
                  f"{', '.join(allowed)})")
            return 2
    if mode != "full" and a.no_resume:
        print(f"✗ --no-resume chỉ dành cho decoder_ft của My Gemma (--mode "
              f"full). Làm lại hai baseline: --mode comparer_pretrain --retrain")
        return 2
    if mode == "infer_test" and a.retrain:
        print("✗ --mode infer_test không huấn luyện gì — --retrain vô nghĩa ở đây.")
        return 2
    only = set(x.strip() for x in a.arms.split(",")) if a.arms else None
    try:
        arms = enabled_arms(cfg, only)
    except ValueError as e:
        print(f"✗ {e}")
        return 2
    if not arms:
        print("✗ không arm nào được bật")
        return 2
    if mode == "comparer_pretrain":
        ours_sel = [x["id"] for x in arms if x["kind"] != "sota"]
        if only and ours_sel:
            print(f"✗ --mode comparer_pretrain chỉ huấn luyện hai baseline; "
                  f"{ours_sel} là candidate `ours` — bỏ khỏi --arms.")
            return 2
        arms = [x for x in arms if x["kind"] == "sota"]
        if not arms:
            print("✗ --mode comparer_pretrain: không baseline (kind=sota) nào "
                  "được bật.")
            return 2

    ft_arms = [x for x in arms if x["kind"] == "ours"]
    if a.no_resume:
        if "train" not in stages:
            print("✗ --no-resume làm lại decoder_ft từ đầu nên cần stage `train` "
                  "(đang --stages " + ",".join(stages) + ").\n"
                  "    Chỉ muốn chấm lại: --rescore. Chỉ muốn sinh lại: xoá "
                  "predictions/<arm>.jsonl.")
            return 2
        if not ft_arms:
            print("✗ --no-resume chỉ áp cho decoder_ft của My Gemma, nhưng "
                  "không candidate `ours` nào đang được chọn (--arms). Hai "
                  "baseline làm lại bằng --retrain.")
            return 2
        if str(a.init_from or "").lower() == "none":
            print("⚠ --no-resume + --init-from none: đây là chạy scratch, "
                  "không phải decoder_ft — vẫn làm lại từ đầu.", flush=True)

    run_id = a.run_id or str(cfg["output"].get("subdir", "2.5_sota"))
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    adapter_root = Path(str(cfg["output"].get("adapter_dir", "./adapters_2_5")))

    # comparer_pretrain ghi manifest RIÊNG: manifest.json là của run chính, và
    # một lần huấn luyện baseline không được phép ghi đè bảng kết quả của nó.
    manifest_path = run_dir / ("manifest_comparer_pretrain.json"
                               if mode == "comparer_pretrain" else "manifest.json")
    no_resume_moved: list[tuple[Path, Path]] = []
    if a.no_resume:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        no_resume_moved = archive_ft(ft_arms, adapter_root, run_dir, stamp)
        print(f"⚙ --no-resume: decoder_ft của "
              f"{', '.join(x['id'] for x in ft_arms)} làm LẠI từ đầu "
              f"(train -> sinh -> chấm).", flush=True)
        if no_resume_moved:
            for old, new in no_resume_moved:
                print(f"    {old} -> {new.name}", flush=True)
            print("    (đổi tên, KHÔNG xoá — dọn tay khi chắc chắn)", flush=True)
        else:
            print("    không có gì cũ để cất.", flush=True)
    names = {x["id"]: x.get("name", x["id"]) for x in arms}
    declared = [x["id"] for x in arms]

    # ── Hàng rào ảnh: điều kiện 4 phải đọc được TRƯỚC khi tốn một giây nào ──
    from . import imaging

    try:
        imaging.unified_policy(cfg)
    except imaging.ImagePolicyViolation as e:
        print(f"✗ DỪNG — chính sách ảnh:\n    {e}")
        return 2

    manifest = {
        "run_id": run_id, "experiment": "2.5", "mode": mode,
        "no_resume": ({"arms": [x["id"] for x in ft_arms],
                       "archived": [[str(o), str(n)] for o, n in no_resume_moved]}
                      if a.no_resume else None),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg), "env": _env_info(),
        "stages_run": stages, "arms_declared": declared, "arm_status": {},
        "deviations": _deviations(cfg, arms),
        "image_policy": imaging.describe_unified(cfg),
    }
    t_start = time.time()
    print("\n═══ §2.5 — R2Gen · AdaAtt · My Gemma: ba cặp, bootstrap BCa + "
          "permutation, Holm trên cặp ═══", flush=True)
    print("  candidate: " + " · ".join(f"{x['id']}[{x['kind']}]" for x in arms),
          flush=True)
    print(f"  ảnh (điều kiện 4): {manifest['image_policy']}", flush=True)
    print(f"⚙ run dir: {run_dir}\n⚙ checkpoint: {adapter_root}", flush=True)

    # ── Trạng thái đọc lại từ đĩa ──────────────────────────────────────────
    from . import train_sota as ts_mod

    def _trained(arm: dict) -> bool:
        d = adapter_root / arm["id"]
        if arm["kind"] == "sota":
            return ts_mod.already_trained(d)
        try:
            from . import train_recomb as tr_mod

            return bool(tr_mod.already_trained(d, True, True))
        except Exception:
            return (d / "train_stats.json").exists()

    mdir = run_dir / "metrics"
    cache = {x["id"]: {
        "trained": _trained(x),
        "n_preds": _count_preds(run_dir / "predictions" / f"{x['id']}.jsonl"),
        "scored": (mdir / f"per_sample_{x['id']}.csv").exists()} for x in arms}
    print("⚙ trạng thái đọc lại từ cache:", flush=True)
    n_test = int(cfg["data"].get("n_test") or 0)
    for aid, s in cache.items():
        print(f"    {names.get(aid, aid):<46} train {'✓' if s['trained'] else '—'}"
              f" · sinh {s['n_preds']}/{n_test} · "
              f"điểm {'✓' if s['scored'] else '—'}", flush=True)
    manifest["cache_at_start"] = cache

    # ── Checkpoint baseline từ Hub: KÉO trước khi quyết định huấn luyện ────
    def _needs_ckpt(x: dict) -> bool:
        if cache[x["id"]]["trained"]:
            return False
        if "train" in stages:
            return True
        return "infer" in stages and not (n_test and cache[x["id"]]["n_preds"] >= n_test)

    if (mode != "comparer_pretrain" and not a.no_pull and not a.dry_run
            and not a.retrain):
        from . import sota_hub

        want = [x for x in arms if x["kind"] == "sota" and _needs_ckpt(x)]
        repo = ""
        try:
            repo = sota_hub.hub_cfg(cfg)["repo"]
        except sota_hub.SotaHubError as e:
            print(f"  ⚠ {e}", flush=True)
        if want and repo:
            print(f"⚙ kéo checkpoint baseline từ {repo}: "
                  f"{', '.join(x['id'] for x in want)}", flush=True)
            manifest["sota_from_hub"] = {}
            for x in want:
                try:
                    meta = sota_hub.pull_arm(cfg, x, adapter_root / x["id"])
                except sota_hub.SotaHubError as e:
                    print(f"  ⚠ [{x['id']}] {e}", flush=True)
                    manifest["sota_from_hub"][x["id"]] = {"ok": False, "error": str(e)}
                    continue
                cache[x["id"]]["trained"] = _trained(x)
                manifest["sota_from_hub"][x["id"]] = {"ok": True, **meta}
                print(f"  ✓ [{x['id']}] {meta.get('selection')} · đẩy lúc "
                      f"{meta.get('pushed_at')}", flush=True)

    if mode == "infer_test":
        not_ready = [x for x in arms if _needs_ckpt(x)]
        if not_ready:
            for x in not_ready:
                how = ("--mode comparer_pretrain trên máy có GPU, rồi kiểm "
                       "`sota_hub.repo`" if x["kind"] == "sota"
                       else "--mode full (decoder_ft My Gemma)")
                print(f"✗ {x['id']}: không có checkpoint cục bộ lẫn trên Hub -> "
                      f"{how}", flush=True)
            print("  --mode infer_test KHÔNG huấn luyện: một cột sinh bằng "
                  "trọng số ngẫu nhiên vẫn ra tiếng Anh đọc được.", flush=True)
            return 10

    arms_to_train: list[dict] = []
    if "train" in stages:
        arms_to_train = (list(arms) if a.retrain
                         else [x for x in arms if not cache[x["id"]]["trained"]])
        if not arms_to_train:
            print("⚙ mọi candidate đã có checkpoint — BỎ QUA stage train và cả "
                  "việc chuẩn bị tập train (--retrain để làm lại)", flush=True)
    need_train = bool(arms_to_train)

    push_ckpt = ("push" in stages and mode == "comparer_pretrain"
                 and bool(cfg["output"].get("push", True)))
    pushed: dict[str, str | None] = {}
    push_ckpt_err: list[str] = []

    def _push_ckpt(arm: dict) -> None:
        """Đẩy ngay khi MỘT baseline xong — baseline thứ hai hỏng không được
        phép kéo theo vài giờ GPU của baseline thứ nhất."""
        if not push_ckpt or arm["id"] in pushed:
            return
        from . import sota_hub

        try:
            pushed[arm["id"]] = sota_hub.push_arm(
                cfg, arm, adapter_root / arm["id"], run_dir)
        except sota_hub.SotaHubError as e:
            print(f"  ✗ {e}", flush=True)
            push_ckpt_err.append(str(e))

    tracker = Tracker(cfg, run_id=(run_id if mode == "full" else f"{run_id}-{mode}"),
                      config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
        print(f"⚙ W&B  : {tracker.url}", flush=True)
    tracker.summary({"run/stages": ",".join(stages), "run/experiment": "2.5",
                     "run/arms": ",".join(declared),
                     **{f"env/{k}": v for k, v in manifest["env"].items()}})
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN — google/gemma-3-4b-it là gated repo.\n"
              "    export HF_TOKEN=hf_xxxxxxxx", flush=True)

    # ── Stage 0 của My Gemma: GIẢI QUYẾT MỘT LẦN, ở giây thứ nhất ──────────
    ours = [x for x in arms if x["kind"] == "ours"]
    if ours and need_train and any(x in arms_to_train for x in ours):
        spec = a.init_from
        if spec is None:
            p = dict(cfg.get("pretrain", {}) or {})
            spec = str(Path(cfg["output"].get("dir", "./runs"))
                       / str(p.get("subdir", "sup_pretrain")) / "ckpt" / "latest")
        if str(spec).lower() == "none":
            cfg["pretrain_init"] = None
            cfg["mode"] = "scratch"
            manifest["deviations"].append(
                "ĐỐI CHỨNG — My Gemma chạy `--init-from none`: nhánh KHÔNG qua "
                "Stage 0, trong khi hai baseline thì có. Cột này không so được "
                "với một lần chạy có pretrain.")
            print("⚙ My Gemma: --init-from none -> nhánh khởi tạo NGẪU NHIÊN "
                  "(đối chứng, đã ghi vào deviations)", flush=True)
        else:
            from . import pretrain_io

            try:
                src = pretrain_io.resolve_source(
                    spec, cache_dir=cfg["data"].get("cache_dir", "./data"))
            except Exception as e:
                print(f"\n✗ DỪNG — không lấy được checkpoint Stage 0 cho My "
                      f"Gemma:\n    {e}\n"
                      f"  Điều kiện 5 nói MỌI kiến trúc đi qua Stage 0. Hai "
                      f"baseline tự chạy nó trong `train_sota`; My Gemma thì "
                      f"nạp từ `sup_pretrain`, và nó có điểm vào riêng:\n"
                      f"      bash scripts/run_pretrain.sh --detach\n"
                      f"  Muốn cố ý chạy không pretrain thì phải NÓI RA:\n"
                      f"      --init-from none", flush=True)
                tracker.finish(1)
                return 9
            cfg["pretrain_init"] = str(src)
            cfg["mode"] = "decoder_ft"
            meta = {}
            try:
                meta = pretrain_io.read_meta(src)
            except Exception:
                pass
            manifest["pretrain_init"] = {"source": str(src), "meta": meta}
            print(f"⚙ My Gemma Stage 0: {src}"
                  + (f" · {meta.get('n_train')} mẫu · epoch {meta.get('epoch')}"
                     if meta else ""), flush=True)

    # ── data ───────────────────────────────────────────────────────────────
    from . import data as data_mod
    from . import data_train as dt_mod

    train_df = val_df = None
    if "data" in stages:
        test_df = data_mod.prepare(cfg, tracker)
        if need_train:
            try:
                train_df, val_df = dt_mod.prepare_train(cfg, test_df, tracker)
            except dt_mod.DataLeak as e:
                print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
                tracker.alert("§2.5 dừng: rò rỉ train/test", str(e), "ERROR")
                tracker.finish(1)
                return 4
    elif mode == "comparer_pretrain" and not need_train:
        test_df = None      # chỉ đẩy checkpoint đã có — không cần tập test
    else:
        test_df = data_mod.load_manifest(cfg)
        print(f"⚙ dùng lại manifest test: {len(test_df)} mẫu", flush=True)
    if test_df is not None:
        tracker.summary({"data/n_test": len(test_df)})

    # ── train ──────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    if need_train:
        _hr(f"Huấn luyện {len(arms_to_train)}/{len(arms)} candidate — "
            f"candidate NẶNG nhất trước")
        from .backbone import TrainUnusable

        if train_df is None:
            train_df, val_df = dt_mod.load_train_manifest(cfg)
        manifest["n_train"] = len(train_df)
        manifest["n_val"] = 0 if val_df is None else len(val_df)
        tracker.summary({"data/n_train": len(train_df)})

        preset = cls_pack = None
        if any(x["kind"] == "ours" for x in arms_to_train):
            from . import train_recomb as tr_mod
            from .backbone import preflight_adapters

            preset = tr_mod.resolve_preset(cfg)
            pf_err, pf_warn = preflight_adapters(
                [{"method": preset["method"], "rank": preset["rank"]}])
            e2, w2 = tr_mod.preflight_recomb(
                cfg, [x for x in arms_to_train if x["kind"] == "ours"])
            pf_err += e2
            pf_warn += w2
            for w in pf_warn:
                print(f"  {w}", flush=True)
            if pf_err:
                for e in pf_err:
                    print(f"  ✗ {e}", flush=True)
                print("\n✗ DỪNG ở preflight.", flush=True)
                manifest["preflight_errors"] = pf_err
                tracker.alert("§2.5 dừng ở preflight", "\n".join(pf_err), "ERROR")
                tracker.finish(1)
                return 2
            try:
                cls_pack = tr_mod.prepare_cls(cfg, train_df)
            except Exception as e:
                print(f"\n✗ DỪNG — nhãn phân loại không dùng được:\n    {e}",
                      flush=True)
                tracker.finish(1)
                return 5
            if cls_pack:
                manifest["labels"] = cls_pack["stats"]

        for arm in train_order(arms_to_train):
            aid = arm["id"]
            odir = adapter_root / aid
            print(f"\n── {arm.get('name', aid)} ({arm['kind']}) ──", flush=True)
            try:
                if arm["kind"] == "sota":
                    st = ts_mod.train_sota_arm(cfg, arm, train_df, val_df, odir,
                                               run_dir, tracker,
                                               max_steps_override=dry_steps)
                    from . import sota_hub

                    # Từ điển VỪA dùng đi cùng checkpoint (đè bản cũ nếu có).
                    sota_hub.bundle_tokenizer(odir, run_dir, overwrite=True)
                    if mode == "comparer_pretrain":
                        _push_ckpt(arm)
                else:
                    from . import train_recomb as tr_mod

                    st = tr_mod.train_arm(cfg, arm, preset, train_df, cls_pack,
                                          odir, tracker,
                                          max_steps_override=dry_steps)
                train_stats.append(st)
                manifest["arm_status"][aid] = {
                    "trained": True, "minutes": st.get("minutes"),
                    "batch_size": st.get("batch_size"),
                    "vram_peak_gb": st.get("vram_peak_gb")}
                tracker.summary({f"train/{aid}/minutes": st.get("minutes"),
                                 f"train/{aid}/loss": st.get("train_loss"),
                                 f"train/{aid}/trainable_params":
                                     st.get("trainable_params")})
            except TrainUnusable as e:
                print(f"\n✗ {aid} KHÔNG huấn luyện được: {e}", flush=True)
                manifest["arm_status"][aid] = {"trained": False, "unusable": str(e)}
                tracker.alert("§2.5 dừng ở train", f"{aid}: {e}", "ERROR")
                tracker.finish(1)
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
                return 3
    else:
        for arm in arms:
            p = adapter_root / arm["id"] / "train_stats.json"
            if p.exists():
                train_stats.append(json.loads(p.read_text(encoding="utf-8")))

    if mode == "comparer_pretrain":
        if push_ckpt:
            _hr(f"Đẩy checkpoint baseline lên {cfg.get('sota_hub', {}).get('repo')}")
            for arm in arms:
                if _trained(arm):
                    _push_ckpt(arm)
                else:
                    print(f"  ⊘ [{arm['id']}] chưa có checkpoint — không đẩy",
                          flush=True)
        elif "push" in stages:
            print("⊘ output.push = false (--no-push / --dry-run) — không đẩy "
                  "checkpoint", flush=True)
        manifest["sota_pushed"] = pushed
        if push_ckpt_err:
            manifest["push_error"] = push_ckpt_err
        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        tracker.finish(1 if push_ckpt_err else 0)
        print(f"\n✓ comparer_pretrain xong sau {manifest['minutes_total']} phút"
              f" · checkpoint: {adapter_root}", flush=True)
        for aid, url in pushed.items():
            print(f"  {aid:<8} {url or '(không đẩy)'}", flush=True)
        print("  Tiếp theo (máy nào cũng được): --mode infer_test hoặc --mode "
              "full — checkpoint baseline sẽ được kéo về khi thiếu.", flush=True)
        return 8 if push_ckpt_err else 0

    # ── infer ──────────────────────────────────────────────────────────────
    pred_paths = {x["id"]: run_dir / "predictions" / f"{x['id']}.jsonl" for x in arms}
    if "infer" in stages:
        _hr("Sinh báo cáo")
        from .runners import build
        from .runners.base import RunnerUnusable

        for arm in arms:
            mcfg = arm_model_cfg(cfg, arm, adapter_root, run_dir)
            print(f"\n── {arm.get('name', arm['id'])} ({mcfg['runner']}) ──",
                  flush=True)
            t0 = time.time()
            try:
                build(cfg, mcfg, tracker).run(test_df, pred_paths[arm["id"]],
                                              log_every=tracker.log_every)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
            except RunnerUnusable as e:
                # Ba cặp đòi ĐỦ ba cột. Chạy tiếp với 2/3 là lặng lẽ đổi thiết
                # kế — bảng vẫn in ra, chỉ thiếu một dòng, và không gì nói rằng
                # cặp thiếu ấy chính là cặp quan trọng nhất.
                print(f"\n✗ {arm['id']} không sinh được: {e}", flush=True)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": False, "unusable": str(e)}
                tracker.alert("§2.5 dừng: một candidate không sinh được",
                              f"{arm['id']}: {e}", "ERROR")
                tracker.finish(1)
                return 6

    # ── score ──────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model

    mdir.mkdir(parents=True, exist_ok=True)
    ref_map = dict(zip(test_df["key"], test_df["ref"]))
    scores: dict[str, pd.DataFrame] = {}
    if "score" in stages:
        _hr("Chấm điểm")
        for arm in arms:
            aid = arm["id"]
            csv_path = mdir / f"per_sample_{aid}.csv"
            if csv_path.exists() and not a.rescore:
                scores[aid] = pd.read_csv(csv_path)
                print(f"  [{aid}] dùng lại {csv_path.name} "
                      f"({len(scores[aid])} mẫu)", flush=True)
                continue
            preds = _read_preds(pred_paths[aid])
            keys = [k for k in test_df["key"] if k in preds]
            if not keys:
                print(f"  ⚠ [{aid}] không có dự đoán nào", flush=True)
                continue
            sdf, _c = score_model(cfg, keys, [preds[k] for k in keys],
                                  [ref_map[k] for k in keys], label=aid)
            sdf.to_csv(csv_path, index=False)
            scores[aid] = sdf
            tracker.summary({f"score/{aid}/{k}": float(sdf[k].mean())
                             for k in metric_columns(sdf)})
    else:
        for arm in arms:
            p = mdir / f"per_sample_{arm['id']}.csv"
            if p.exists():
                scores[arm["id"]] = pd.read_csv(p)

    missing = [i for i in declared if i not in scores]
    if missing:
        print(f"✗ §2.5 cần điểm của ĐỦ {len(declared)} candidate; thiếu "
              f"{missing}. Ba cặp là một thiết kế, không phải một danh sách "
              f"tuỳ chọn: bỏ một cột đi thì bảng vẫn in ra và không gì nói "
              f"rằng cặp thiếu là cặp nào.", flush=True)
        tracker.alert("§2.5 hỏng", f"run={run_id}: thiếu {missing}", "ERROR")
        tracker.finish(1)
        return 1

    # ── căn khối ───────────────────────────────────────────────────────────
    from .stats_multi import align_blocks

    scores = {i: scores[i] for i in declared if i in scores}
    scores, align_info = align_blocks(scores, key_col="key",
                                      order_hint=test_df["key"].tolist())
    manifest["n_eval"] = align_info["n_common"]
    manifest["align"] = align_info
    if any(v for v in align_info["n_dropped"].values()):
        print("⚙ căn khối: " + ", ".join(
            f"{k}:{v}" for k, v in align_info["n_per_arm"].items())
            + f" -> {align_info['n_common']} ảnh chung", flush=True)

    # ── stats ──────────────────────────────────────────────────────────────
    pw = pd.DataFrame()
    pw_path = mdir / "pairwise_tests.csv"
    if "stats" in stages:
        _hr("Bootstrap ghép cặp BCa + Permutation, Holm trên ba cặp")
        from .stats_multi import (compare_pairs, pair_list, resolution_warning,
                                  summary_lines)

        cols = metric_columns(scores[declared[0]])
        scfg = dict(cfg["stats"])
        pairs = pair_list(declared, scfg.get("pairs", "all"), scfg.get("reference"))
        print(f"  {len(pairs)} cặp × {len(cols)} metric × 2 kiểm định trên "
              f"{align_info['n_common']} ảnh ghép cặp "
              f"(n_perm={scfg.get('n_perm')}, n_boot={scfg.get('n_boot')}, "
              f"Holm theo `{scfg.get('adjust_scope', 'pairs')}`)", flush=True)
        w = resolution_warning(int(scfg.get("n_perm", 10000)),
                               float(scfg.get("alpha", 0.05)), len(pairs))
        if w:
            print(f"  ⚠ {w}", flush=True)
            manifest["stats_warning"] = w
        t0 = time.time()
        pw = compare_pairs(scores, cols, scfg, declared,
                           seed=int(cfg.get("seed", 42)))
        pw.to_csv(pw_path, index=False)
        n_sig = int(pw["significant"].sum())
        n_mat = int(pw["significant_and_material"].sum())
        print(f"  -> {n_sig}/{len(pw)} dòng bác H0 ở CẢ HAI kiểm định (Holm, "
              f"α={scfg.get('alpha', 0.05)}); {n_mat} trong đó có |dz| >= "
              f"{scfg.get('dz_floor', 0.1)}", flush=True)
        primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
        lines = summary_lines(pw, primary, names)
        manifest["headline"] = lines
        for ln in lines:
            print(f"  · {ln}", flush=True)
        for wmsg in [x for x in pw["dz_warning"].tolist() if x]:
            print(f"  ⚠ {wmsg}", flush=True)
        tracker.table("stats/pairwise", pw)
        tracker.summary({"stats/n_significant": n_sig, "stats/n_material": n_mat,
                         "stats/headline": " | ".join(lines)})
        print(f"✓ xong sau {time.time() - t0:.1f}s", flush=True)
    elif pw_path.exists():
        pw = pd.read_csv(pw_path)

    # ── report ─────────────────────────────────────────────────────────────
    if "report" in stages:
        _hr("Báo cáo")
        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        report_err = None
        try:
            from . import report_section25

            report_section25.write_all(cfg, run_dir, scores, names, pw,
                                       train_stats, manifest, arms=arms)
        except Exception as e:
            import traceback

            report_err = traceback.format_exc()
            print(f"\n✗ stage `report` hỏng: {e!r}", flush=True)
            print(report_err, flush=True)
            print("→ KẾT QUẢ VẪN CÒN NGUYÊN trên đĩa:\n"
                  f"    checkpoint    : {adapter_root}\n"
                  f"    predictions   : {run_dir / 'predictions'}\n"
                  f"    điểm mỗi ảnh  : {mdir}/per_sample_*.csv\n"
                  f"    kiểm định     : {pw_path}\n"
                  "  Sửa rồi chạy lại MỘT MÌNH stage báo cáo:\n"
                  "    python -m src.run_section25 --stages report", flush=True)
            manifest["report_error"] = report_err.splitlines()[-1]
            tracker.alert("§2.5: báo cáo hỏng (kết quả vẫn còn)",
                          report_err[-1500:], "ERROR")
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
        s = mdir / "summary.csv"
        if s.exists():
            tracker.table("results/summary", pd.read_csv(s))
        if (run_dir / "REPORT.md").exists():
            tracker.save_file(run_dir / "REPORT.md")
        if report_err:
            tracker.finish(1)
            return 7

    # ── push ───────────────────────────────────────────────────────────────
    push_err = None
    if "push" in stages and cfg["output"].get("push", True):
        _hr("Đẩy lên Hugging Face")
        from .push import push_results

        try:
            url = push_results(cfg, run_dir)
            if url:
                manifest["hub_url"] = url
                tracker.summary({"run/hub_url": url})
        except Exception as e:
            import traceback

            push_err = traceback.format_exc()
            print(f"\n✗ đẩy lên Hub hỏng: {e}", flush=True)
            print(f"→ Kết quả vẫn nằm nguyên ở {run_dir}. Đẩy lại RIÊNG bước "
                  f"này:\n    python -m src.run_section25 --stages push",
                  flush=True)
            manifest["push_error"] = str(e)
            tracker.alert("§2.5: đẩy Hub hỏng (kết quả vẫn còn)",
                          push_err[-1500:], "ERROR")

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    mins = (time.time() - t_start) / 60
    tracker.alert("§2.5 chạy xong",
                  f"run={run_id} · {manifest.get('n_eval')} ảnh · {mins:.0f} phút")
    tracker.finish(1 if push_err else 0)
    print(f"\n✓ HOÀN TẤT sau {mins:.1f} phút", flush=True)
    print(f"  báo cáo : {run_dir / 'REPORT.md'}", flush=True)
    if manifest.get("hub_url"):
        print(f"  Hub     : {manifest['hub_url']}", flush=True)
    if tracker.url:
        print(f"  W&B     : {tracker.url}", flush=True)
    return 8 if push_err else 0


if __name__ == "__main__":
    from .vram_monitor import run_with_monitor

    sys.exit(run_with_monitor("section25", main))

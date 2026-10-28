"""§2.3 và §2.4 — trọn một thí nghiệm trong MỘT lệnh. Chọn bằng `--exp`.

    python -m src.run_recomb --exp 2.3        # 5 candidate: baseline, QLoRA, Recombinator, MoE-LoRA và shared LoRA
    python -m src.run_recomb --exp 2.4        # 5 cấu hình (top-k, n_experts)
    python -m src.run_recomb --exp 2.4 --dry-run
    python -m src.run_recomb --exp 2.4 --stages stats,report
    python -m src.run_recomb --exp 2.3 --arms lora_recomb

Mọi stage resume được, và lần chạy lại NÓI RA nó đọc lại được những gì: ngay
sau dòng `run dir` là một bảng trạng thái đọc từ đĩa (adapter đã có chưa, đã
sinh bao nhiêu ảnh, đã chấm chưa, đã có ab_tests/REPORT chưa). Candidate nào đã
có `adapter_model.safetensors` + `train_stats.json` (+ file nhánh) thì KHÔNG
huấn luyện lại — và nếu MỌI candidate đều thế thì cả việc chuẩn bị tập train
(tải + resize 6400 ảnh) cũng bị bỏ qua, không chỉ vòng huấn luyện. `--retrain`
/ `--rescore` để làm lại.

Hai mã thoát nghĩa là "kết quả còn nguyên, chỉ bước cuối hỏng" — adapter, dự
đoán, `per_sample_*.csv` và `ab_tests.csv` đều đã ghi xong trước đó:
    7 = stage `report` hỏng  -> chạy lại `--stages report`
    8 = bước đẩy Hub hỏng    -> chạy lại `--stages push`

── HAI THÍ NGHIỆM, MỘT ĐƯỜNG CODE ──────────────────────────────────────────
Khác nhau đúng ba chỗ, và cả ba đều hiện ra trong file này:

  1. DANH SÁCH CANDIDATE. §2.3 đọc `arms` (5 candidate; `recomb` và `moe` chọn nhánh);
     §2.4 đọc `arms_moe` (5 candidate, (top_k, n_experts) phân biệt) và MỌI
     candidate đều có `recomb = True` — §2.4 luôn dùng concat.
  2. THỨ TỰ CHẠY. §2.3: MoE-LoRA trước, rồi các candidate có nhánh. §2.4: candidate NẶNG nhất trước.
  3. KIỂM ĐỊNH. §2.3: bootstrap ghép cặp + permutation trên mọi cặp. §2.4 ->
     Friedman tổng thể, rồi hậu kiểm Wilcoxon + Holm trên những metric đã qua
     cổng.

Mọi thứ còn lại — data, base, prompt, ảnh, nhánh Recombinator, ASL, batch,
lịch học, metric — đi qua CÙNG code và CÙNG khối config. Đó là nghĩa cơ học
của "§2.4 giữ nguyên vẹn mọi điều kiện thí nghiệm tại §2.3": không phải một
lời hứa trong README mà là một sự thật không có chỗ để lệch.

Hai thư mục kết quả tách rời (`output.subdir` vs `output.subdir_moe`,
`adapter_dir` vs `adapter_dir_moe`): ghi đè lên nhau là mất đúng cái bảng mà
mục kia lấy làm mốc nền, và trộn adapter hai mục trong một thư mục là mở đường
cho `already_trained()` nhận nhầm candidate rồi bỏ qua stage train.

── Vì sao MoE-LoRA chạy TRƯỚC (§2.3) ───────────────────────────────────────
MoE-LoRA là candidate nặng hơn ở cả train lẫn sinh, và nó là candidate DUY NHẤT có thể
hỏng vì lý do kiến trúc: thiếu nhãn, pan&scan lén bật, nhánh không vừa VRAM.
A thì gần như chắc chắn chạy — nó là một arm LoRA bình thường.

Chạy A trước nghĩa là trả 90 phút GPU rồi mới biết B không dựng được, và lúc đó
90 phút ấy không mua được gì: §2.3 cần ĐỦ HAI cột, một cột lẻ không định nghĩa
được phép so sánh ghép cặp. Chạy B trước thì mọi thất bại kiểu ấy xảy ra ở phút
thứ ba.

Thứ tự này KHÔNG được phép ảnh hưởng tới kết quả, và đó là điều kiện chứ không
phải hy vọng: seed và thứ tự dữ liệu được đặt lại ngay trước MỖI candidate; batch
được dò ở candidate đầu rồi khóa cho candidate tiếp theo (xem
docstring `src/train_recomb.py`). Nếu đổi thứ tự mà bảng kết quả đổi theo thì có
một chỗ rò trạng thái, và nó là lỗi.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (Cfg, add_common_args, hf_token, load_config,
                     resolved_snapshot, seed_everything)
from .tracking import Tracker

STAGES = ["data", "train", "infer", "score", "stats", "report", "push"]
EXPERIMENTS = ("2.3", "2.4", "3")
# §2.3/§2.4 đã chốt; `configs/` nay chỉ còn MỘT file (section25.yaml) và file
# ấy không có `arms`/`arms_moe`/`stats.friedman`. Điểm vào này đọc bản ghi.
DEFAULT_CONFIG = (Path(__file__).resolve().parent.parent / "configs"
                  / "recombiner.yaml")


def _hr(t: str) -> None:
    print(f"\n═══ {t} ═══", flush=True)


def _archive_stale(path: Path) -> None:
    """Move an artifact made with an obsolete label source aside for recovery."""
    if not path.exists():
        return
    stale = path.parent / "_stale"
    stale.mkdir(parents=True, exist_ok=True)
    target = stale / f"{path.stem}.{time.time_ns()}{path.suffix}"
    shutil.move(str(path), str(target))
    print(f"  ↪ artifact cũ -> {target}", flush=True)


def _read_preds(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
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
            out[str(r["key"])] = r
    return out


def _write_preds(path: Path, records: dict[str, dict]) -> None:
    """Rewrite JSONL atomically after adding side metadata; preserve reports."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
    tmp.replace(path)


def _count_preds(path: Path) -> int:
    """Số dòng JSON đọc được trong file dự đoán (0 nếu chưa có).

    Đếm bằng cách thử parse từng dòng chứ không đếm `\\n`: file bị Ctrl-C cắt
    giữa lúc ghi có dòng cuối cụt, và nói dối về số mẫu đã sinh là cách chắc
    chắn nhất để bỏ qua stage `infer` khi nó chưa xong.
    """
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


def cache_state(arms: list[dict], adapter_root: Path, run_dir: Path) -> dict:
    """Cái gì đã nằm sẵn trên đĩa — đọc TRƯỚC khi tiêu một giây GPU nào.

    Mọi stage ở đây vốn đã resume được, nhưng chúng resume LẶNG LẼ và mỗi stage
    theo cách riêng: train nhìn `adapter_model.safetensors`, infer nhìn file
    `.jsonl`, score nhìn `per_sample_*.csv`. Khi một lần chạy dài bị đứt ở stage
    cuối, câu hỏi đầu tiên là "chạy lại có phải train lại không" — và trả lời nó
    bằng cách đọc ba chỗ khác nhau trong log là lý do người ta chạy lại từ đầu
    cho chắc. Bảng này trả lời câu ấy ở giây thứ nhất.
    """
    # Import MUỘN và có đường lui: `train_recomb` kéo theo torch/peft/bitsandbytes.
    # Một lần chạy `--stages report` trên máy không GPU vẫn phải đọc được cache,
    # và nó không cần biết adapter có đủ file hay không.
    try:
        from . import train_recomb as tr_mod

        _trained = tr_mod.already_trained
    except Exception:
        def _trained(d, needs_branch=False, needs_experts=False,
                     expected_moe=None) -> bool:
            return (Path(d) / "train_stats.json").exists()

    mdir = run_dir / "metrics"
    st = {"arms": {}, "ab_tests": (mdir / "ab_tests.csv").exists(),
          "friedman": (mdir / "friedman.csv").exists(),
          "posthoc": (mdir / "posthoc.csv").exists(),
          "report": (run_dir / "REPORT.md").exists()}
    for arm in arms:
        aid = arm["id"]
        st["arms"][aid] = {
            "trained": bool(_trained(adapter_root / aid,
                                     bool(arm.get("recomb")),
                                     bool(arm.get("moe")),
                                     {k: arm[k] for k in
                                     ("top_k", "n_experts", "route_pooling",
                                      "all_linear_targets", "shared_lora",
                                      "shared_rank", "shared_alpha",
                                      "shared_warmup_mode", "shared_warmup_pct")
                                      if k in arm} or None)),
            "n_preds": _count_preds(run_dir / "predictions" / f"{aid}.jsonl"),
            "scored": (mdir / f"per_sample_{aid}.csv").exists(),
        }
    return st


def print_cache_state(st: dict, names: dict[str, str], n_test: int | None) -> None:
    print("⚙ trạng thái đọc lại từ cache:", flush=True)
    for aid, s in st["arms"].items():
        tot = f"/{n_test}" if n_test else ""
        print(f"    {names.get(aid, aid):<34} "
              f"train {'✓' if s['trained'] else '—'} · "
              f"sinh {s['n_preds']}{tot} · "
              f"điểm {'✓' if s['scored'] else '—'}", flush=True)
    flags = [k for k in ("ab_tests", "friedman", "posthoc", "report") if st.get(k)]
    print(f"    đã có sẵn: {', '.join(flags) if flags else '(chưa có gì)'}"
          f"  ·  --retrain / --rescore để làm lại", flush=True)


def _env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "peft", "bitsandbytes", "safetensors"):
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


def enabled_arms(cfg: Cfg, only: set[str] | None = None) -> list[dict]:
    """Các candidate được bật, với cờ `recomb` đã CHUẨN HOÁ và bắt buộc.

    Cờ này là toàn bộ khác biệt giữa hai cột trong bảng kết quả, nên nó không
    được phép có giá trị mặc định. Một arm quên khai (hay gõ `recombiner:` trong
    khi code đọc `recomb:`) mà lặng lẽ nhận `False` nghĩa là chạy candidate A
    hai lần, ra hai cột gần như trùng nhau, và kết luận "nhánh không có tác
    dụng" — một kết quả đọc được, hợp lý, và sai.
    """
    arms = []
    for a in cfg["arms"]:
        a = dict(a)
        if not a.get("enabled", True):
            continue
        if "recomb" in a:
            flag = a["recomb"]
        elif "recombiner" in a:
            flag = a["recombiner"]
        else:
            raise ValueError(
                f"arm {a.get('id')!r} không khai `recomb:` (true/false). Đó là "
                f"thứ DUY NHẤT phân biệt candidate A với candidate B — mặc định "
                f"ngầm ở đây là chạy A hai lần mà không ai biết.")
        a["recomb"] = bool(flag)
        arms.append(a)
    if only:
        arms = [a for a in arms if a["id"] in only]
    return arms


def moe_arms(cfg: Cfg, only: set[str] | None = None) -> list[dict]:
    """Các candidate của §2.4, với `recomb` ĐÓNG CỨNG bằng True.

    §2.4 LUÔN dùng concat, nên `recomb` ở đây không phải một khoá mà một hằng
    số: nó không nằm trong `arms_moe` để không ai tắt được nó bằng một lần gõ.
    Không nhánh thì không có Route Head, và Top-k Gate không có gì để đọc — đó
    không phải một biến thể của §2.4, nó là một thí nghiệm khác.

    `top_k` và `n_experts` thì BẮT BUỘC phải khai: chúng là hai con số §2.4
    đang so, và một candidate lặng lẽ nhận giá trị mặc định nghĩa là bảng kết
    quả có một hàng mang nhãn sai.
    """
    arms = []
    for a in cfg.get("arms_moe", []) or []:
        a = dict(a)
        if not a.get("enabled", True):
            continue
        for key in ("top_k", "n_experts"):
            if key not in a:
                raise ValueError(
                    f"candidate §2.4 {a.get('id')!r} không khai `{key}`. Hai con "
                    f"số (top_k, n_experts) là thứ DUY NHẤT phân biệt năm "
                    f"candidate — mặc định ngầm ở đây là một hàng mang nhãn sai "
                    f"trong bảng Friedman.")
        a["recomb"] = True
        a["moe"] = True
        arms.append(a)
    if not arms:
        raise ValueError("config không có `arms_moe` nào được bật.")
    if only:
        arms = [a for a in arms if a["id"] in only]
    return arms


def select_arms(cfg: Cfg, exp: str, only: set[str] | None = None) -> list[dict]:
    return moe_arms(cfg, only) if exp == "2.4" else enabled_arms(cfg, only)


def train_order(arms: list[dict], exp: str = "2.3") -> list[dict]:
    """Thứ tự CHẠY — không phải thứ tự trình bày.

    §2.3: MoE-LoRA trước, rồi các candidate có nhánh, cuối cùng baseline.

    §2.4: candidate NẶNG nhất trước — E lớn nhất, rồi K lớn nhất. Cùng lý lẽ,
    một bậc rõ hơn: `n_experts` nhân thẳng vào activation LoRA (B×T×E×r) và vào
    số tham số học được, nên `K=6·E=10` là candidate duy nhất có thể không vừa
    VRAM. Chạy nó cuối nghĩa là trả bốn lượt huấn luyện rồi mới biết bảng §2.4
    thiếu một cột — và một bảng Friedman thiếu cột thì hoặc phải bỏ hẳn một
    candidate khỏi thiết kế khối, hoặc không dùng được.

    Thứ tự này KHÔNG được phép ảnh hưởng tới kết quả, và đó là điều kiện chứ
    không phải hy vọng: seed và thứ tự dữ liệu đặt lại trước MỖI candidate bằng
    một Generator riêng, batch là hằng số trong config, VRAM được dọn giữa hai
    candidate. Đổi thứ tự mà bảng đổi theo là một chỗ rò trạng thái, và đó là lỗi.
    """
    if exp == "2.4":
        return sorted(arms, key=lambda a: (-int(a.get("n_experts", 0)),
                                           -int(a.get("top_k", 0)),
                                           str(a.get("id"))))
    return sorted(arms, key=lambda a: (
        0 if a.get("moe") else 1,
        0 if a.get("recomb") else 1))


def arm_model_cfg(cfg: Cfg, arm: dict, adapter_root: Path) -> Cfg:
    """Cấu hình kiểu `models[]` để dùng lại `Runner.run()` (resume, ngắt mạch)."""
    b = dict(cfg["base"])
    return Cfg({
        "id": arm["id"], "name": arm.get("name", arm["id"]),
        # CẢ HAI candidate dùng `gemma3_recomb`. Runner ấy tự dựng inputs_embeds
        # và candidate A đơn giản là chạy nó với `branch = None`. Cho A đi
        # `gemma3_adapter` cho "gọn" là để hai cột trong bảng đi qua hai đường
        # sinh khác nhau — tương đương về toán, không giống nhau từng phép tính.
        "runner": "gemma3" if arm.get("infer_only") else "gemma3_recomb",
        "hf_id": b["hf_id"], "dtype": b.get("dtype", "bfloat16"),
        "batch_size": int(b.get("batch_size", 4)),
        "disable_compile": b.get("disable_compile", "auto"),
        "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
        **({} if arm.get("infer_only") else
           {"adapter_dir": str(adapter_root / arm["id"])}),
        # Không hợp nhất adapter ở candidate nào: hợp nhất ở A mà không hợp nhất
        # ở B là một khác biệt nữa giữa hai cột. (Trên base 4-bit thì hợp nhất
        # vốn đã bị tắt, nhưng đây phải là một quyết định, không phải may mắn.)
        "merge_adapter": False,
        "capture_cls_head": bool(arm.get("capture_cls_head", False)),
        "moe": bool(arm.get("moe", False)),
        "optional": False,
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="§2.3 (năm candidate, có MoE-LoRA và shared LoRA) hoặc §2.4 (MoE, 5 cấu hình) — "
                    "trọn gói, chọn bằng --exp.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--exp", default=None, choices=list(EXPERIMENTS),
                    help="2.3 = năm candidate (bootstrap+permutation); "
                         "2.4 = MoE (5 cấu hình top-k/expert, Friedman). Mặc "
                         "định lấy từ khoá `experiment` trong config.")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None, help="chỉ chạy các arm này (phẩy ngăn cách)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
        help="8 ảnh, 4 bước huấn luyện, 200 lần mô phỏng, KHÔNG đẩy Hub.")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="huấn luyện lại dù đã có")
    ap.add_argument("--force", action="store_true",
                    help="chỉ Candidate D (moe_lora): ghi đè checkpoint và suy luận lại")
    ap.add_argument("--pull", action="store_true",
                    help="chỉ kéo adapter đã huấn luyện và predictions từ Hub về local rồi thoát")
    ap.add_argument("--reuse-hub-preds", action="store_true",
                    help="kéo predictions đã infer từ Hub; candidate nào Hub đã có thì "
                         "KHÔNG train/infer lại (chỉ score/stats/report), candidate "
                         "chưa có (B*) vẫn train + infer như thường")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides)

    # ── Chọn thí nghiệm ────────────────────────────────────────────────────
    # `--exp` thắng config; config chỉ là mặc định. Và `moe.enabled` được đặt
    # TỪ ĐÂY, không đọc từ file: để nó là một khoá người dùng sửa tay nghĩa là
    # có hai nguồn sự thật cho cùng một câu hỏi ("đang chạy mục nào"), và ca
    # hỏng là `--exp 2.3` với `moe.enabled: true` còn sót trong file — chạy ra
    # §2.3 với Route_Gate hoạt động, tức là không phải mục nào cả.
    exp = str(a.exp or cfg.get("experiment", "2.3")).strip()
    if exp not in EXPERIMENTS:
        print(f"✗ --exp phải là một trong {EXPERIMENTS}, nhận {exp!r}")
        return 2
    cfg["experiment"] = exp
    is_moe = exp == "2.4"
    if is_moe:
        cfg["output"]["subdir"] = str(cfg["output"].get("subdir_moe", "2.4_moe"))
        cfg["output"]["adapter_dir"] = str(
            cfg["output"].get("adapter_dir_moe", "./adapters_2_4"))
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["moe", "friedman", "2.4"]

    dry_steps = None
    if a.dry_run:
        cfg["data"]["n_test"] = 8
        cfg["data"]["n_train"] = 16
        cfg["data"]["n_val"] = 0
        cfg["stats"]["n_perm"] = 200
        cfg["stats"]["n_boot"] = 200
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] = str(cfg["output"].get("subdir", "run")) + "_dryrun"
        cfg["output"]["adapter_dir"] = str(cfg["output"].get("adapter_dir", "./adapters")) \
            + "_dryrun"
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · 4 bước · không đẩy Hub.\n"
              "  Đây là phép thử ĐƯỜNG ỐNG, không phải phép dò VRAM: nó chạy với "
              "đúng batch trong config,\n  nhưng 4 bước trên 16 mẫu ngắn không "
              "nói được gì về VRAM đỉnh của 6400 mẫu.", flush=True)
    if a.no_push:
        cfg["output"]["push"] = False
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed_everything(int(cfg.get("seed", 42)))
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    if a.force and "train" not in stages:
        print("✗ --force chỉ có tác dụng cùng stage train; hãy thêm `train` vào --stages")
        return 2
    for s in stages:
        if s not in STAGES:
            print(f"✗ stage không biết: {s!r} (hợp lệ: {', '.join(STAGES)})")
            return 2
    only = set(x.strip() for x in a.arms.split(",")) if a.arms else None
    try:
        arms = select_arms(cfg, exp, only)
    except ValueError as e:
        print(f"✗ {e}")
        return 2
    if not arms:
        print("✗ không arm nào được bật")
        return 2
    if a.force and not any(x["id"] == "moe_lora" for x in arms):
        print("✗ --force chỉ áp dụng cho Candidate D `moe_lora`; hãy thêm nó vào --arms")
        return 2
    has_moe = is_moe or any(bool(arm.get("moe", False)) for arm in arms)
    cfg.setdefault("moe", {})["enabled"] = has_moe
    arm_a = None if is_moe else next(
        (x["id"] for x in arms if not x.get("recomb")
         and not x.get("infer_only")), None)
    arm_b = None if is_moe else next(
        (x["id"] for x in arms if x.get("recomb")), None)

    run_id = a.run_id or str(cfg["output"].get("subdir", "run"))
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    adapter_root = Path(str(cfg["output"].get("adapter_dir", "./adapters")))
    pred_paths = {x["id"]: run_dir / "predictions" / f"{x['id']}.jsonl" for x in arms}
    names = {x["id"]: x.get("name", x["id"]) for x in arms}

    manifest = {
        "run_id": run_id, "experiment": exp,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg), "env": _env_info(),
        "stages_run": stages, "arms_declared": [x["id"] for x in arms],
        "arm_a": arm_a, "arm_b": arm_b, "arm_status": {},
        "moe": ({x["id"]: {"top_k": x["top_k"], "n_experts": x["n_experts"]}
                 for x in arms if x.get("moe")} if has_moe else None),
    }
    if a.pull:
        from . import resume as resume_mod

        try:
            pulled = resume_mod.pull_artifacts_from_hub(
                cfg, arms, adapter_root, pred_paths)
        except resume_mod.ScanError as e:
            print(f"✗ Không kéo được artifacts từ Hub: {e}", flush=True)
            return 5
        manifest["hub_pull"] = pulled
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        print("\n✓ Đã hoàn tất lệnh kéo Hub.", flush=True)
        for aid in pred_paths:
            print(f"  {aid}: trọng số={pulled['adapters'].get(aid, '—')} · "
                  f"dự đoán={pulled['predictions'].get(aid, '—')}", flush=True)
        missing_adapters = [aid for aid in pred_paths
                            if not any(token in str(pulled["adapters"].get(aid, ""))
                                       for token in ("đã kéo", "không cần trọng số"))]
        if missing_adapters:
            print("✗ Chưa thể suy luận các candidate này vì Hub không có đủ adapter: "
                  + ", ".join(missing_adapters), flush=True)
            return 6
        return 0
    reused: set[str] = set()
    if a.reuse_hub_preds:
        from . import resume as resume_mod

        print("\n⇣ Kéo predictions đã infer từ Hub (không kéo adapter)…", flush=True)
        try:
            got = resume_mod.pull_predictions_only(
                cfg, [x["id"] for x in arms], pred_paths)
        except resume_mod.ScanError as e:
            print(f"✗ Không kéo được predictions từ Hub: {e}", flush=True)
            return 5
        reused = {aid for aid, msg in got.items() if str(msg).startswith("đã kéo")}
        manifest["hub_reuse_predictions"] = got
        for aid in pred_paths:
            print(f"  {aid}: {got.get(aid, '—')}"
                  + ("  → dùng lại, không train/infer" if aid in reused
                     else "  → sẽ train + infer"), flush=True)
    t_start = time.time()
    hdr = ("§2.4 — MoE: 5 cấu hình (top-k, expert), Friedman + Wilcoxon/Holm"
           if is_moe else
           "Exp 3 — hậu kiểm: biến thể mạnh nhất của họ B và họ C, 2 epoch, batch hiệu dụng 16"
           if exp == "3" else
           "§2.3 — năm candidate: baseline, QLoRA, Recombinator, MoE-LoRA và MoE + shared LoRA; "
           "bootstrap ghép cặp + permutation")
    print(f"\n═══ {hdr} ═══", flush=True)
    if is_moe:
        print("  candidate: " + " · ".join(
            f"{x['id']}(k={x['top_k']},E={x['n_experts']})" for x in arms),
            flush=True)
        print("  §2.4 LUÔN dùng concat — mọi candidate đều có nhánh Recombinator.",
              flush=True)
    print(f"⚙ run dir: {run_dir}\n⚙ adapters: {adapter_root}", flush=True)

    # ── Trạng thái từ những lần chạy trước ─────────────────────────────────
    # Đọc đĩa TRƯỚC mọi thứ tốn kém. `arms_to_train` quyết định hai việc, và cả
    # hai đều đắt: có phải chuẩn bị (tải + resize) tập train hay không, và có
    # phải tạo target CheXbert từ report hay không. Cũ hơn, hai việc ấy chạy chỉ vì
    # `"train" in stages` — nghĩa là một lần chạy lại chỉ để in báo cáo vẫn trả
    # tiền cho 6400 ảnh, dù không candidate nào cần huấn luyện lại.
    arms_to_train: list[dict] = []
    cache: dict = {}
    try:
        cache = cache_state(arms, adapter_root, run_dir)
        print_cache_state(cache, names, int(cfg["data"].get("n_test") or 0) or None)
        manifest["cache_at_start"] = cache
    except Exception as e:  # đọc cache KHÔNG được phép làm hỏng lần chạy
        print(f"⚠ không đọc được trạng thái cache ({e!r}) — coi như chưa có gì",
              flush=True)
    if "train" in stages:
        trainable_arms = [x for x in arms if not x.get("infer_only")
                          and x["id"] not in reused]
        for arm in trainable_arms:
            aid = arm["id"]
            trained = bool(cache.get("arms", {}).get(aid, {}).get("trained"))
            is_candidate_d = aid == "moe_lora"
            if is_candidate_d and not trained and not a.force:
                odir = adapter_root / aid
                stats_path = odir / "train_stats.json"
                if stats_path.exists():
                    from . import moe as moe_mod

                    if moe_mod.experts_exist(odir):
                        print("✗ Checkpoint Candidate D đã có nhưng không khớp "
                              "cấu hình Route attention pooling + attention-only "
                              "Experts mới. "
                              "Để tránh ghi đè, hãy chủ động xác nhận bằng `--force`.",
                              flush=True)
                        return 2
            # Candidate D is immutable/resumable by default. `--retrain` remains
            # applicable to A/B/C; only `--force` may replace a valid D checkpoint.
            if not trained or (a.retrain and not is_candidate_d) or (
                    is_candidate_d and a.force):
                arms_to_train.append(arm)
        if a.retrain and "moe_lora" in [x["id"] for x in trainable_arms] and not a.force:
            print("⚙ giữ nguyên checkpoint Candidate D: --retrain không ghi đè D; "
                  "dùng --force nếu muốn huấn luyện lại D.", flush=True)
        if any(x.get("moe") for x in arms_to_train) and not a.dry_run:
            if "infer" not in stages:
                stages.append("infer")
                manifest["stages_run"] = stages
            print("⚙ Candidate D sẽ được lưu rồi suy luận lại; prediction cũ của D "
                  "sẽ được thay bằng kết quả mới.", flush=True)
        if not arms_to_train:
            print("⚙ mọi candidate đã có adapter + train_stats.json — BỎ QUA "
                  "stage train và cả việc chuẩn bị tập train",
                  flush=True)
    need_train_run = bool(arms_to_train)
    if bool(cfg["output"].get("push_per_arm", False)) and need_train_run \
            and "infer" not in stages:
        stages.append("infer")
        manifest["stages_run"] = stages

    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
        print(f"⚙ W&B  : {tracker.url}", flush=True)
    tracker.summary({"run/stages": ",".join(stages), "run/experiment": exp,
                     "run/arms": ",".join(manifest["arms_declared"]),
                     **{f"env/{k}": v for k, v in manifest["env"].items()}})
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN — google/gemma-3-4b-it là gated repo.\n"
              "    export HF_TOKEN=hf_xxxxxxxx", flush=True)

    # ── Nhãn phân loại ──────────────────────────────────────────────────────
    # Khi train, prepare_cls dùng CheXbert trên report gốc; khi score, cùng
    # labeler gán nhãn report gốc và report sinh.
    cls_pack = None
    need_b = any(x.get("recomb") for x in arms_to_train)

    # ── data ────────────────────────────────────────────────────────────────
    from . import data as data_mod
    from . import data_train as dt_mod

    train_df = val_df = None
    if "data" in stages:
        test_df = data_mod.prepare(cfg, tracker)
        if need_train_run:
            try:
                train_df, val_df = dt_mod.prepare_train(cfg, test_df, tracker)
            except dt_mod.DataLeak as e:
                print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
                tracker.alert(f"§{exp} dừng: rò rỉ train/test", str(e), "ERROR")
                tracker.finish(1)
                return 4
    else:
        test_df = data_mod.load_manifest(cfg)
        print(f"⚙ dùng lại manifest test: {len(test_df)} mẫu", flush=True)
    tracker.summary({"data/n_test": len(test_df)})

    # ── train → infer → push TỪNG arm ──────────────────────────────────────
    # output.push_per_arm: xong train của một arm là suy luận và đẩy NGAY, rồi
    # mới sang arm kế tiếp — một lần đứt giữa chừng không làm mất arm đã xong.
    per_arm = bool(cfg["output"].get("push_per_arm", False))
    inferred_now: set[str] = set()

    def infer_one(arm: dict):
        """Sinh báo cáo cho MỘT arm. Trả None nếu ổn, mã thoát nếu dừng."""
        from .runners import build
        from .runners.base import RunnerUnusable

        if arm["id"] in reused:
            print(f"\n── {arm.get('name', arm['id'])}: dùng predictions từ Hub, "
                  f"bỏ qua infer ──", flush=True)
            return None
        mcfg = arm_model_cfg(cfg, arm, adapter_root)
        print(f"\n── {arm.get('name', arm['id'])} ({mcfg['runner']}) ──", flush=True)
        t0 = time.time()
        try:
            build(cfg, mcfg, tracker).run(test_df, pred_paths[arm["id"]],
                                          log_every=tracker.log_every)
            manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
            inferred_now.add(arm["id"])
        except RunnerUnusable as e:
            # Dừng ngay ở CẢ HAI mục: §2.3 k nhỏ, mất một candidate là mất cả
            # phép so sánh; §2.4 Friedman đòi thiết kế khối ĐẦY ĐỦ.
            print(f"\n✗ {arm['id']} không sinh được: {e}", flush=True)
            manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                "ok": False, "unusable": str(e)}
            tracker.alert(f"§{exp} dừng: một candidate không sinh được",
                          f"{arm['id']}: {e}", "ERROR")
            tracker.finish(1)
            return 6
        return None

    def push_arm(arm: dict) -> None:
        """Đẩy adapter + predictions + train_stats của MỘT arm (không ném lỗi)."""
        if not per_arm or not cfg["output"].get("push", True):
            return
        from .push import push_artifact

        aid = arm["id"]
        push_artifact(cfg, adapter_root / aid, f"adapters/{aid}",
                      f"§{exp} adapter {aid}")
        push_artifact(cfg, pred_paths[aid], f"predictions/{aid}.jsonl",
                      f"§{exp} predictions {aid}")
        push_artifact(cfg, adapter_root / aid / "train_stats.json",
                      f"train_stats/{aid}.json", f"§{exp} train_stats {aid}")

    # Không tự kết nối hoặc kéo artifacts từ Hub. Chỉ --pull (lệnh riêng ở trên)
    # mới tải adapter/predictions; các stage thường dùng cache local hiện có.

    # ── train ───────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    if need_train_run:
        _hr(f"Huấn luyện {len(arms_to_train)}/{len(arms)} candidate — "
            + ("candidate NẶNG nhất (E lớn nhất) TRƯỚC" if is_moe
               else "MoE-LoRA trước, rồi các candidate còn lại"))
        from . import train_recomb as tr_mod
        from .backbone import TrainUnusable, preflight_adapters

        preset = tr_mod.resolve_preset(cfg)
        pf_err, pf_warn = preflight_adapters(
            [{"method": preset["method"], "rank": preset["rank"]}])
        e2, w2 = tr_mod.preflight_recomb(cfg, arms)
        pf_err += e2
        pf_warn += w2
        for w in pf_warn:
            print(f"  {w}", flush=True)
        if pf_err:
            for e in pf_err:
                print(f"  ✗ {e}", flush=True)
            print("\n✗ DỪNG ở preflight — sửa theo hướng dẫn rồi chạy lại.", flush=True)
            manifest["preflight_errors"] = pf_err
            tracker.alert(f"§{exp} dừng ở preflight", "\n".join(pf_err), "ERROR")
            tracker.finish(1)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            return 2
        print(f"  ✓ peft/bitsandbytes/GPU/nhánh sẵn sàng · preset LoRA "
              f"{preset['id']} (r={preset['rank']} alpha={preset['alpha']})",
              flush=True)
        manifest["lora_preset"] = preset

        if train_df is None:
            train_df, val_df = dt_mod.load_train_manifest(cfg)
        manifest["n_train"] = len(train_df)
        tracker.summary({"data/n_train": len(train_df)})

        # Nhãn căn theo ĐÚNG thứ tự hàng của train_df, nên chỉ dựng được sau khi
        # có train_df. Lần gọi ở trên chỉ để dò schema và chết sớm nếu hỏng.
        if need_b:
            cls_pack = tr_mod.prepare_cls(cfg, train_df)
            if cls_pack:
                manifest["labels"] = cls_pack["stats"]
                tracker.summary({"labels/n_labels": cls_pack["n_labels"],
                                 "labels/frac_missing":
                                     cls_pack["stats"].get("frac_missing")})

        tcfg = dict(cfg["train"])
        bs = int(tcfg.get("per_device_batch_size", 16))
        eff_bs = int(tcfg.get("effective_batch_size", 32))
        probe_sizes = list(tcfg.get("batch_probe_sizes", [bs]))
        print(f"  batch probe={bool(tcfg.get('batch_probe', False))} · "
              f"thứ tự {probe_sizes} · effective_batch={eff_bs}; "
              "batch được chọn ở candidate đầu sẽ dùng cho cả hai.", flush=True)
        manifest["batch"] = {"initial_batch_size": bs,
                             "probe_sizes": probe_sizes,
                             "effective_batch": eff_bs,
                             "source": "OOM backoff; shared across trainable candidates"}

        # Train target lấy trực tiếp từ report gốc qua CheXbert.
        # Bất kỳ bảng điểm/report cũ nào cũng thuộc mô hình/nhãn cũ; lưu lại để
        # khôi phục được nhưng không cho lần chạy mới đọc nhầm.
        for stale_name in ("ab_tests.csv", "summary.csv", "pairwise_tests.csv"):
            _archive_stale(run_dir / "metrics" / stale_name)
        _archive_stale(run_dir / "REPORT.md")

        for arm in train_order(arms, exp):
            if arm.get("infer_only"):
                continue
            aid = arm["id"]
            odir = adapter_root / aid
            is_b = bool(arm.get("recomb"))
            is_cached_train = tr_mod.already_trained(
                odir, is_b, bool(arm.get("moe", False)),
                {k: arm[k] for k in ("top_k", "n_experts", "route_pooling",
                                     "all_linear_targets", "shared_lora",
                                     "shared_rank", "shared_alpha",
                                     "shared_warmup_mode", "shared_warmup_pct")
                 if k in arm} or None)
            retrain_this = (a.retrain and aid != "moe_lora") or (
                aid == "moe_lora" and a.force)
            if is_cached_train and not retrain_this:
                print(f"  ✓ {aid}: đã huấn luyện ở {odir} — bỏ qua "
                      f"({'--force để làm lại D' if aid == 'moe_lora' else '--retrain để làm lại'})",
                      flush=True)
                cached = json.loads((odir / "train_stats.json").read_text(encoding="utf-8"))
                train_stats.append(cached)
                manifest["arm_status"][aid] = {"trained": "cached"}
                if is_b and bool(cfg["train"].get("batch_probe", False)):
                    selected_bs = int(cached.get("batch_size", bs))
                    cfg["train"]["per_device_batch_size"] = selected_bs
                    cfg["train"]["grad_accum"] = math.ceil(eff_bs / selected_bs)
                    cfg["train"]["batch_probe"] = False
                    cfg["train"]["batch_probe_sizes"] = [selected_bs]
                    manifest["batch"].update({
                        "batch_size": selected_bs,
                        "microbatches_per_step": math.ceil(eff_bs / selected_bs),
                        "effective_batch": eff_bs,
                        "selected_from_cached_candidate": aid})
                continue
            _archive_stale(run_dir / "predictions" / f"{aid}.jsonl")
            _archive_stale(run_dir / "metrics" / f"per_sample_{aid}.csv")
            tag = (f"MoE top-k = {arm['top_k']} · {arm['n_experts']} expert · CÓ nhánh"
                   if arm.get("moe") else
                   ("candidate B · CÓ nhánh" if is_b else "candidate A · không nhánh"))
            print(f"\n── {arm.get('name', aid)} ({tag}) ──", flush=True)
            try:
                st = tr_mod.train_arm(cfg, arm, preset, train_df,
                                      cls_pack if is_b else None, odir, tracker,
                                      max_steps_override=dry_steps)
                train_stats.append(st)
                if is_b and bool(cfg["train"].get("batch_probe", False)):
                    selected_bs = int(st["batch_size"])
                    cfg["train"]["per_device_batch_size"] = selected_bs
                    cfg["train"]["grad_accum"] = math.ceil(eff_bs / selected_bs)
                    cfg["train"]["batch_probe"] = False
                    cfg["train"]["batch_probe_sizes"] = [selected_bs]
                    manifest["batch"].update({
                        "batch_size": selected_bs,
                        "microbatches_per_step": math.ceil(eff_bs / selected_bs),
                        "effective_batch": eff_bs,
                        "selected_by_candidate": aid,
                        "probe_attempts": st.get("batch_probe_attempts")})
                manifest["arm_status"][aid] = {"trained": True,
                                               "minutes": st["minutes"],
                                               "batch_size": st["batch_size"],
                                               "vram_peak_gb": st["vram_peak_gb"]}
                tracker.summary({f"train/{aid}/minutes": st["minutes"],
                                 f"train/{aid}/loss": st["train_loss"],
                                 f"train/{aid}/vram_peak_gb": st["vram_peak_gb"],
                                 f"train/{aid}/trainable_params":
                                     st.get("trainable_params")})
                if per_arm and "infer" in stages:
                    from .backbone import free_vram
                    free_vram()
                    rc = infer_one(arm)
                    if rc is not None:
                        return rc
                    push_arm(arm)
            except TrainUnusable as e:
                print(f"\n✗ {aid} KHÔNG huấn luyện được: {e}", flush=True)
                tracker.alert(f"§{exp} dừng ở train", f"{aid}: {e}", "ERROR")
                manifest["arm_status"][aid] = {"trained": False, "unusable": str(e)}
                tracker.finish(1)
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
                return 3
            except Exception as e:
                print(f"\n✗ {aid} hỏng khi huấn luyện: {e!r}", flush=True)
                manifest["arm_status"][aid] = {"trained": False, "error": repr(e)}
                tracker.alert("Candidate hỏng khi train", f"{aid}: {e!r}", "ERROR")
                raise
    else:
        for arm in arms:
            p = adapter_root / arm["id"] / "train_stats.json"
            if p.exists():
                train_stats.append(json.loads(p.read_text(encoding="utf-8")))


    # ── predictions kéo từ Hub: phải ĐỦ mẫu, đúng bộ giải mã ───────────────
    if reused:
        from . import decoding_spec
        from . import hub_reuse as hr_mod

        keys = [str(k) for k in test_df["key"]]
        expect = decoding_spec.tag(cfg)
        warns = {}
        for arm in arms:
            aid = arm["id"]
            if aid not in reused:
                continue
            rows = hr_mod._read(pred_paths[aid])
            miss = [k for k in keys if k not in rows]
            if miss:
                print(f"✗ predictions của {aid} trên Hub thiếu {len(miss)}/{len(keys)} "
                      f"ảnh test (vd {miss[0]}). Không điền bù; chạy lại infer §2.3 "
                      f"cho {aid} rồi đẩy lên Hub.", flush=True)
                return 6
            if arm.get("moe") and not all(rows[k].get("route_experts") for k in keys):
                print(f"✗ predictions của {aid} thiếu route_experts.", flush=True)
                return 6
            try:
                w = hr_mod.check_decoding(rows, keys, expect, aid)
            except hr_mod.HubReuseError as e:
                print(f"✗ {e}", flush=True)
                return 6
            if w:
                warns[aid] = w
                print(f"  ⚠ {w}", flush=True)
        manifest["hub_reuse_decode_warnings"] = warns

    # ── infer ───────────────────────────────────────────────────────────────
    if "infer" in stages:
        _hr("Sinh báo cáo")
        for arm in arms:
            if arm["id"] in inferred_now:
                continue
            rc = infer_one(arm)
            if rc is not None:
                return rc
            if arm["id"] not in reused:
                push_arm(arm)

    # ── score ───────────────────────────────────────────────────────────────
    from .metrics import (CHEXBERT_GOLD_SOURCE, CHEXBERT_LABELS,
                          metric_columns, score_model)

    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    ref_map = dict(zip(test_df["key"], test_df["ref"]))
    scores: dict[str, pd.DataFrame] = {}

    if "score" in stages:
        _hr("Chấm điểm")
        for arm in arms:
            aid = arm["id"]
            csv_path = mdir / f"per_sample_{aid}.csv"
            if csv_path.exists() and not a.rescore:
                cached = pd.read_csv(csv_path)
                disease_counts = [f"chexbert_label_{j:02d}_{kind}"
                                  for j in range(14) for kind in ("tp", "fp", "fn")]
                counts_ready = (all(c in cached.columns for c in
                                    ("chexbert_tp", "chexbert_fp", "chexbert_fn",
                                     *disease_counts))
                                and "chexbert_gold_source" in cached.columns
                                and cached["chexbert_gold_source"].eq(
                                    CHEXBERT_GOLD_SOURCE).all())
                route_ready = not arm.get("moe") or "route_experts" in cached.columns
                radgraph_cols = ("radgraph_f1_entity", "radgraph_f1_partial",
                                 "radgraph_f1_complete")
                radgraph_ready = (not cfg.get("metrics", {}).get("radgraph", {}).get(
                    "enabled", False) or all(c in cached.columns for c in radgraph_cols))
                chexbert_ready = (not cfg.get("metrics", {}).get("chexbert", {}).get(
                    "enabled", False) or counts_ready)
                if chexbert_ready and route_ready and radgraph_ready:
                    scores[aid] = cached
                    print(f"  [{aid}] dùng lại {csv_path.name} ({len(scores[aid])} mẫu)",
                          flush=True)
                    continue
                print(f"  [{aid}] bảng điểm cũ thiếu metric/counters hiện hành; "
                      "chấm lại từ predictions ...", flush=True)
            pred_records = _read_preds(pred_paths[aid])
            keys = [k for k in test_df["key"] if str(k) in pred_records]
            if not keys:
                print(f"  ⚠ [{aid}] không có dự đoán nào", flush=True)
                continue
            route_experts = None
            if arm.get("moe"):
                missing_route = [k for k in keys
                                 if "route_experts" not in pred_records[str(k)]]
                if missing_route:
                    print(f"  [{aid}] report đã có nhưng thiếu route_experts ở "
                          f"{len(missing_route)} mẫu; chạy vision + Route Head "
                          "để bổ sung metadata, không sinh lại report ...", flush=True)
                    from .runners import build

                    route_runner = build(
                        cfg, arm_model_cfg(cfg, arm, adapter_root), tracker)
                    rows_by_key = {
                        str(row["key"]): row
                        for row in test_df.to_dict(orient="records")
                    }
                    route_rows = [rows_by_key[str(k)] for k in missing_route
                                  if str(k) in rows_by_key]
                    if len(route_rows) != len(missing_route):
                        raise ValueError(
                            f"Không tìm thấy ảnh của {len(missing_route) - len(route_rows)} "
                            f"mẫu {aid} để bổ sung route metadata.")
                    try:
                        route_runner.load()
                        bs_route = max(1, int(route_runner.mcfg.get("batch_size", 1)))
                        for start in range(0, len(route_rows), bs_route):
                            batch_routes = route_runner.route_experts_batch(
                                route_rows[start:start + bs_route])
                            for key, experts in batch_routes.items():
                                pred_records[key]["route_experts"] = experts
                            done_route = min(start + bs_route, len(route_rows))
                            if done_route == len(route_rows) or done_route % 100 < bs_route:
                                print(f"  [{aid}] route metadata {done_route}/"
                                      f"{len(route_rows)}", flush=True)
                        still_missing = [k for k in keys
                                         if "route_experts" not in pred_records[str(k)]]
                        if still_missing:
                            raise ValueError(
                                f"Route Head không trả metadata cho {len(still_missing)} "
                                f"mẫu {aid}.")
                        _write_preds(pred_paths[aid], pred_records)
                        print(f"  [{aid}] đã gắn route_experts vào predictions; "
                              "giữ nguyên report đã infer.", flush=True)
                    finally:
                        route_runner.unload()
                route_experts = [pred_records[str(k)]["route_experts"] for k in keys]
            sdf, _c = score_model(
                cfg, keys, [pred_records[str(k)].get("pred", "") for k in keys],
                [ref_map[k] for k in keys], label=aid)
            if route_experts is not None:
                sdf["route_experts"] = [json.dumps(x) for x in route_experts]
            sdf.to_csv(csv_path, index=False)
            scores[aid] = sdf
            score_summary = {f"score/{aid}/{k}": float(sdf[k].mean())
                             for k in metric_columns(sdf)}
            if all(c in sdf.columns for c in
                   ("chexbert_tp", "chexbert_fp", "chexbert_fn")):
                from .metrics import chexbert_micro_f1
                n_score = len(sdf)
                boot_n = int(cfg.get("stats", {}).get("n_boot", 4))
                boot_size = int(cfg.get("stats", {}).get(
                    "bootstrap_sample_size", n_score))
                boot_idx = np.random.default_rng(int(cfg.get("seed", 42))).integers(
                    0, n_score, size=(boot_n, boot_size))
                score_summary[f"score/{aid}/chexbert_f1"] = float(
                    chexbert_micro_f1(
                        sdf["chexbert_tp"].to_numpy(dtype=float)[boot_idx],
                        sdf["chexbert_fp"].to_numpy(dtype=float)[boot_idx],
                        sdf["chexbert_fn"].to_numpy(dtype=float)[boot_idx], axis=1
                    ).mean())
            tracker.summary(score_summary)
    else:
        for arm in arms:
            p = mdir / f"per_sample_{arm['id']}.csv"
            if p.exists():
                scores[arm["id"]] = pd.read_csv(p)

    want_ids = [x["id"] for x in arms]
    missing = [i for i in want_ids if i not in scores]
    disease_counts = [f"chexbert_label_{j:02d}_{kind}"
                      for j in range(len(CHEXBERT_LABELS))
                      for kind in ("tp", "fp", "fn")]
    if (str(cfg.get("metrics", {}).get("primary", "")) == "chexbert_f1"
            and any(not (all(c in df.columns for c in
                             ("chexbert_tp", "chexbert_fp", "chexbert_fn",
                              *disease_counts))
                         and "chexbert_gold_source" in df.columns
                         and df["chexbert_gold_source"].eq(
                             CHEXBERT_GOLD_SOURCE).all())
                    for df in scores.values())):
        raise ValueError(
            "Điểm CheXbert đã lưu theo nguồn gold cũ/schema cũ; cần gán nhãn lại "
            "report gốc + report sinh bằng f1chexbert. Chạy lại scoring từ predictions: "
            "python -m src.run_recomb --exp 2.3 --stages score,stats,report --rescore")
    for arm in arms:
        aid = arm["id"]
        if arm.get("moe") and aid in scores and "route_experts" not in scores[aid].columns:
            raise ValueError(
                f"Điểm {aid} không có route_experts để lập bảng expert × bệnh. "
                "Chạy lại infer MoE rồi score,stats,report.")
    if is_moe:
        if len(scores) < 3 or missing:
            print(f"✗ §2.4 cần điểm của ĐỦ {len(want_ids)} candidate; thiếu "
                  f"{missing}. Friedman đòi thiết kế khối đầy đủ — mọi candidate "
                  f"phải có mặt trong MỌI khối, và chạy với một tập candidate "
                  f"khác tập đã khai là đổi thí nghiệm mà bảng vẫn in ra bình "
                  f"thường.", flush=True)
            tracker.alert("§2.4 hỏng", f"run={run_id}: thiếu {missing}", "ERROR")
            tracker.finish(1)
            return 1
    elif len(scores) != len(want_ids) or missing:
        print(f"✗ cần điểm của đủ {len(want_ids)} candidate; thiếu {missing}. "
              f"Đang có {sorted(scores)}.", flush=True)
        tracker.alert("§2.3 hỏng", f"run={run_id}: {sorted(scores)}", "ERROR")
        tracker.finish(1)
        return 1

    # ── căn khối ────────────────────────────────────────────────────────────
    # Cùng một hàm, hai bản (stats_ab / stats_friedman) nội dung giống hệt nhau.
    # Dùng bản thuộc về đường kiểm định đang chạy để không tạo một đường phụ
    # thuộc chéo giữa hai module thống kê.
    if is_moe:
        from .stats_friedman import align_blocks
    else:
        from .stats_ab import align_blocks

    # Thứ tự cột = thứ tự KHAI BÁO trong config, không phải thứ tự chạy. Hạng
    # trong khối của Friedman không phụ thuộc vào thứ tự cột, nhưng bảng in ra
    # thì có — và "candidate nặng chạy trước" là một quyết định vận hành, nó
    # không được rò vào cách trình bày kết quả.
    scores = {i: scores[i] for i in want_ids if i in scores}
    scores, align_info = align_blocks(scores, key_col="key",
                                      order_hint=test_df["key"].tolist())
    manifest["n_eval"] = align_info["n_common"]
    manifest["align"] = align_info
    if any(v for v in align_info["n_dropped"].values()):
        print("⚙ căn khối: " + ", ".join(
            f"{k}:{v}" for k, v in align_info["n_per_arm"].items())
            + f" -> {align_info['n_common']} ảnh chung", flush=True)

    # ── stats ───────────────────────────────────────────────────────────────
    ab = pd.DataFrame()
    fr = pd.DataFrame()
    ab_path = mdir / "ab_tests.csv"
    fr_path = mdir / "friedman.csv"
    ph_path = mdir / "posthoc.csv"
    if is_moe:
        if "stats" in stages:
            _hr("Friedman + hậu kiểm Wilcoxon/Holm")
            from .stats_friedman import (friedman_all, negligible_warning,
                                         posthoc_all)

            cols = metric_columns(scores[want_ids[0]])
            scfg = dict(cfg["stats"])
            scfg.update(dict(scfg.get("friedman", {}) or {}))
            k = len(scores)
            n_pairs = k * (k - 1) // 2
            print(f"  kiểm định TỔNG THỂ: {len(cols)} metric × Friedman trên "
                  f"{align_info['n_common']} khối, k = {k}", flush=True)
            t0 = time.time()
            fr = friedman_all(scores, cols, scfg)
            fr.to_csv(fr_path, index=False)
            gate = dict(zip(fr["metric"], fr["reject_h0"]))
            n_open = int(fr["reject_h0"].sum())
            print(f"  -> {n_open}/{len(fr)} metric bác H0 sau Holm (α="
                  f"{scfg.get('alpha', 0.05)}). Hậu kiểm CHỈ chạy trên chúng: "
                  f"{n_open} × {n_pairs} cặp = {n_open * n_pairs} kiểm định "
                  f"(thay vì {len(cols) * n_pairs}).", flush=True)
            for wmsg in [x for x in fr.get("w_warning", []) if x]:
                print(f"  ⚠ {wmsg}", flush=True)
            ab = posthoc_all(scores, cols, scfg, gate=gate,
                             seed=int(cfg.get("seed", 42)))
            if not ab.empty:
                ab.to_csv(ph_path, index=False)
                n_sig = int(ab["significant"].sum())
                n_mat = int(ab["significant_and_material"].sum())
                print(f"  -> {n_sig}/{len(ab)} cặp có ý nghĩa (Wilcoxon + Holm "
                      f"toàn cục); {n_mat} trong đó có |r_rb| >= "
                      f"{scfg.get('rrb_floor', 0.1)}", flush=True)
                tracker.table("stats/posthoc", ab)
            else:
                n_sig = n_mat = 0
                print("  không metric nào qua cổng Friedman -> KHÔNG hậu kiểm. "
                      "Đó là một kết luận, không phải một lỗi: bằng chứng không "
                      "đủ để nói năm cấu hình khác nhau.", flush=True)
            primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
            prow = fr[fr["metric"] == primary]
            if not prow.empty:
                best = str(prow.iloc[0]["best_by_meanrank"])
                line = (f"`{primary}`: hạng trung bình tốt nhất = "
                        f"{names.get(best, best)} · Kendall W = "
                        f"{float(prow.iloc[0]['kendall_w']):.4f} · H0 "
                        f"{'BỊ BÁC' if bool(prow.iloc[0]['reject_h0']) else 'KHÔNG bị bác'}")
                manifest["headline"] = line
                print(f"  {line}", flush=True)
                _w = negligible_warning(int(prow.iloc[0]["n_blocks"]),
                                        float(prow.iloc[0]["kendall_w"]),
                                        float(scfg.get("kendall_w_floor", 0.05)))
                if _w:
                    manifest["stats_warning"] = _w
            tracker.table("stats/friedman", fr)
            tracker.summary({"stats/n_metrics_open": n_open,
                             "stats/n_significant": n_sig,
                             "stats/n_material": n_mat,
                             "stats/headline": manifest.get("headline", "")})
            print(f"✓ xong sau {time.time()-t0:.1f}s", flush=True)
        else:
            if fr_path.exists():
                fr = pd.read_csv(fr_path)
            if ph_path.exists():
                ab = pd.read_csv(ph_path)
    elif "stats" in stages:
        _hr("Bootstrap ghép cặp + Permutation")
        from .stats_multi import compare_pairs
        from .stats_ab import resolution_warning

        cols = metric_columns(scores[arm_a])
        scfg = dict(cfg["stats"])
        print(f"  {len(want_ids)} candidate · {len(cols)} metric · "
              f"{scfg.get('n_boot')} bootstrap subset × "
              f"{scfg.get('bootstrap_sample_size')} ảnh trên pool "
              f"{align_info['n_common']} (n_perm={scfg.get('n_perm')})", flush=True)
        w = resolution_warning(int(scfg.get("n_perm", 10000)),
                               float(scfg.get("alpha", 0.05)), len(cols))
        if w:
            print(f"  ⚠ {w}", flush=True)
            manifest["stats_warning"] = w
        t0 = time.time()
        ab = compare_pairs(scores, cols, scfg, arm_ids=want_ids,
                           seed=int(cfg.get("seed", 42)))
        ab.to_csv(ab_path, index=False)
        n_sig = int(ab["significant"].sum())
        n_mat = int(ab["significant_and_material"].sum())
        print(f"  -> {n_sig}/{len(ab)} metric có ý nghĩa theo p permutation + CI "
              f"(α={scfg.get('alpha', 0.05)}); {n_mat} trong đó có |dz| >= "
              f"{scfg.get('dz_floor', 0.1)}", flush=True)
        primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
        pmain = ab[ab["metric"] == primary] if not ab.empty else ab
        line = ("; ".join(
            f"{names.get(r.arm_b, r.arm_b)} vs {names.get(r.arm_a, r.arm_a)}: "
            f"Δ={r.mean_diff:.4f}, p_perm={r.p_perm:.4g}"
            for r in pmain.itertuples()) if not pmain.empty else "Không có kết quả metric chính")
        manifest["headline"] = line
        print(f"  {line}", flush=True)
        for wmsg in [x for x in ab["dz_warning"].tolist() if x]:
            print(f"  ⚠ {wmsg}", flush=True)
        tracker.table("stats/ab", ab)
        tracker.summary({"stats/n_significant": n_sig, "stats/n_material": n_mat,
                         "stats/headline": line})
        print(f"✓ xong sau {time.time()-t0:.1f}s", flush=True)
    elif ab_path.exists():
        ab = pd.read_csv(ab_path)

    # ── report ──────────────────────────────────────────────────────────────
    if "report" in stages:
        _hr("Báo cáo")
        manifest["minutes_total"] = round((time.time() - t_start) / 60, 1)
        # Stage CUỐI, sau nhiều giờ GPU, và nó chỉ định dạng lại những con số đã
        # nằm trên đĩa. Một lỗi ở đây (tên cột lệch, một khoá thiếu trong
        # train_stats) không được phép là lý do vứt cả lần chạy: mọi thứ đắt tiền
        # — adapter, predictions, per_sample_*.csv, ab_tests.csv — đã ghi xong
        # trước dòng này. Nên lỗi được ghi lại, in ra, và lần chạy kết thúc bằng
        # một mã lỗi RIÊNG (7) nghĩa là "kết quả còn nguyên, chỉ báo cáo hỏng":
        # chạy lại `--stages report` là đủ, không phải train lại.
        report_err = None
        try:
            if is_moe:
                from . import report_moe

                report_moe.write_all(cfg, run_dir, scores, names, fr, ab,
                                     train_stats, manifest, arms=arms)
            else:
                from . import report_recomb

                # `report_recomb.build_markdown` tự suy A/B từ cờ `recomb` trong
                # config — truyền thêm arm_a/arm_b vào đây là một TypeError, và nó
                # nổ ở stage CUỐI, sau khi đã trả xong tiền GPU.
                report_recomb.write_all(cfg, run_dir, scores, names, ab,
                                        train_stats, manifest)
        except Exception as e:
            import traceback

            report_err = traceback.format_exc()
            print(f"\n✗ stage `report` hỏng: {e!r}", flush=True)
            print(report_err, flush=True)
            print("→ KẾT QUẢ VẪN CÒN NGUYÊN trên đĩa:\n"
                  f"    adapters      : {adapter_root}\n"
                  f"    predictions   : {run_dir / 'predictions'}\n"
                  f"    điểm mỗi ảnh  : {mdir}/per_sample_*.csv\n"
                  f"    kiểm định     : {ab_path if not is_moe else ph_path}\n"
                  f"  Sửa rồi chạy lại MỘT MÌNH stage báo cáo — không train lại:\n"
                  f"    python -m src.run_recomb --exp {exp} --stages report",
                  flush=True)
            manifest["report_error"] = report_err.splitlines()[-1]
            tracker.alert(f"§{exp}: báo cáo hỏng (kết quả vẫn còn)",
                          report_err[-1500:], "ERROR")
            (run_dir / "manifest.json").write_text(
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

    # ── push ────────────────────────────────────────────────────────────────
    push_err = None
    if "push" in stages and cfg["output"].get("push", True):
        _hr("Đẩy lên Hugging Face")
        from .push import push_results

        # Cùng lý do với stage `report`: đây là bước CUỐI, nó chỉ chuyển những
        # file đã có trên đĩa đi nơi khác, và nó phụ thuộc vào hai thứ nằm ngoài
        # tầm với của lần chạy — mạng và quyền của token. Một 401 ở phút thứ 700
        # không được phép biến cả lần chạy thành thất bại.
        try:
            url = push_results(cfg, run_dir)
            if url:
                manifest["hub_url"] = url
                tracker.summary({"run/hub_url": url})
        except Exception as e:
            import traceback

            push_err = traceback.format_exc()
            print(f"\n✗ đẩy lên Hub hỏng: {e}", flush=True)
            print(f"→ Kết quả vẫn nằm nguyên ở {run_dir}. Đẩy lại RIÊNG bước này:\n"
                  f"    python -m src.run_recomb --exp {exp} --stages push",
                  flush=True)
            manifest["push_error"] = str(e)
            tracker.alert(f"§{exp}: đẩy Hub hỏng (kết quả vẫn còn)",
                          push_err[-1500:], "ERROR")

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    mins = (time.time() - t_start) / 60
    tracker.alert(f"§{exp} chạy xong",
                  f"run={run_id} · {manifest.get('n_eval')} ảnh · {mins:.0f} phút\n"
                  f"{manifest.get('headline', '')}")
    tracker.finish(1 if push_err else 0)
    print(f"\n✓ HOÀN TẤT sau {mins:.1f} phút", flush=True)
    print(f"  báo cáo : {run_dir / 'REPORT.md'}", flush=True)
    if manifest.get("hub_url"):
        print(f"  Hub     : {manifest['hub_url']}", flush=True)
    if tracker.url:
        print(f"  W&B     : {tracker.url}", flush=True)
    # Mã 8 = "mọi thứ xong, chỉ bước đẩy hỏng" — phân biệt với 7 (báo cáo hỏng)
    # và với 0, để một script gọi tự động biết có cần đẩy lại hay không.
    return 8 if push_err else 0


if __name__ == "__main__":
    sys.exit(main())

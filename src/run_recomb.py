"""§2.3 và §2.4 — trọn một thí nghiệm trong MỘT lệnh. Chọn bằng `--exp`.

    python -m src.run_recomb --exp 2.3        # A (LoRA) vs B (LoRA + nhánh)
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

  1. DANH SÁCH CANDIDATE. §2.3 đọc `arms` (2 candidate, cờ `recomb` phân biệt);
     §2.4 đọc `arms_moe` (5 candidate, (top_k, n_experts) phân biệt) và MỌI
     candidate đều có `recomb = True` — §2.4 luôn dùng concat.
  2. THỨ TỰ CHẠY. §2.3: B trước A. §2.4: candidate NẶNG nhất trước.
  3. KIỂM ĐỊNH. k = 2 -> bootstrap ghép cặp (BCa) + permutation. k = 5 ->
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

── Vì sao candidate B chạy TRƯỚC candidate A (§2.3) ─────────────────────────
B là candidate nặng hơn ở cả train lẫn sinh, và nó là candidate DUY NHẤT có thể
hỏng vì lý do kiến trúc: thiếu nhãn, pan&scan lén bật, nhánh không vừa VRAM.
A thì gần như chắc chắn chạy — nó là một arm LoRA bình thường.

Chạy A trước nghĩa là trả 90 phút GPU rồi mới biết B không dựng được, và lúc đó
90 phút ấy không mua được gì: §2.3 cần ĐỦ HAI cột, một cột lẻ không định nghĩa
được phép so sánh ghép cặp. Chạy B trước thì mọi thất bại kiểu ấy xảy ra ở phút
thứ ba.

Thứ tự này KHÔNG được phép ảnh hưởng tới kết quả, và đó là điều kiện chứ không
phải hy vọng: seed và thứ tự dữ liệu được đặt lại ngay trước MỖI candidate, và
batch là hằng số trong config chứ không phải thứ dò được từ VRAM còn trống (xem
docstring `src/train_recomb.py`). Nếu đổi thứ tự mà bảng kết quả đổi theo thì có
một chỗ rò trạng thái, và nó là lỗi.
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
EXPERIMENTS = ("2.3", "2.4")
# §2.3/§2.4 đã chốt; `configs/` nay chỉ còn MỘT file (section25.yaml) và file
# ấy không có `arms`/`arms_moe`/`stats.friedman`. Điểm vào này đọc bản ghi.
DEFAULT_CONFIG = (Path(__file__).resolve().parent.parent / "_archive" / "configs"
                  / "recombiner.yaml")


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


def cache_state(arms: list[dict], adapter_root: Path, run_dir: Path,
                is_moe: bool) -> dict:
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
        def _trained(d, needs_branch=False, needs_experts=False) -> bool:
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
                                     bool(arm.get("recomb")), is_moe)),
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

    §2.3: candidate B trước, A sau (xem docstring module).

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
    return sorted(arms, key=lambda a: 0 if a.get("recomb") else 1)


def arm_model_cfg(cfg: Cfg, arm: dict, adapter_root: Path) -> Cfg:
    """Cấu hình kiểu `models[]` để dùng lại `Runner.run()` (resume, ngắt mạch)."""
    b = dict(cfg["base"])
    return Cfg({
        "id": arm["id"], "name": arm.get("name", arm["id"]),
        # CẢ HAI candidate dùng `gemma3_recomb`. Runner ấy tự dựng inputs_embeds
        # và candidate A đơn giản là chạy nó với `branch = None`. Cho A đi
        # `gemma3_adapter` cho "gọn" là để hai cột trong bảng đi qua hai đường
        # sinh khác nhau — tương đương về toán, không giống nhau từng phép tính.
        "runner": "gemma3_recomb",
        "hf_id": b["hf_id"], "dtype": b.get("dtype", "bfloat16"),
        "batch_size": int(b.get("batch_size", 4)),
        "disable_compile": b.get("disable_compile", "auto"),
        "attn_implementation": b.get("attn_implementation_infer", "sdpa"),
        "adapter_dir": str(adapter_root / arm["id"]),
        # Không hợp nhất adapter ở candidate nào: hợp nhất ở A mà không hợp nhất
        # ở B là một khác biệt nữa giữa hai cột. (Trên base 4-bit thì hợp nhất
        # vốn đã bị tắt, nhưng đây phải là một quyết định, không phải may mắn.)
        "merge_adapter": False,
        "optional": False,
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="§2.3 (Recombinator A/B) hoặc §2.4 (MoE, 5 cấu hình) — "
                    "trọn gói, chọn bằng --exp.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--exp", default=None, choices=list(EXPERIMENTS),
                    help="2.3 = nhánh Recombinator (A/B, bootstrap+permutation); "
                         "2.4 = MoE (5 cấu hình top-k/expert, Friedman). Mặc "
                         "định lấy từ khoá `experiment` trong config.")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None, help="chỉ chạy các arm này (phẩy ngăn cách)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="8 ảnh, 4 bước huấn luyện, 200 lần mô phỏng, KHÔNG đẩy "
                         "Hub. KHÔNG dò batch — batch là hằng số trong config.")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--retrain", action="store_true", help="huấn luyện lại dù đã có")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.overrides, getattr(a, "hw", None))

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
    cfg.setdefault("moe", {})["enabled"] = (exp == "2.4")
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
    arm_a = None if is_moe else next(
        (x["id"] for x in arms if not x.get("recomb")), None)
    arm_b = None if is_moe else next(
        (x["id"] for x in arms if x.get("recomb")), None)

    run_id = a.run_id or str(cfg["output"].get("subdir", "run"))
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    adapter_root = Path(str(cfg["output"].get("adapter_dir", "./adapters")))
    names = {x["id"]: x.get("name", x["id"]) for x in arms}

    manifest = {
        "run_id": run_id, "experiment": exp,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg), "env": _env_info(),
        "stages_run": stages, "arms_declared": [x["id"] for x in arms],
        "arm_a": arm_a, "arm_b": arm_b, "arm_status": {},
        "moe": ({x["id"]: {"top_k": x["top_k"], "n_experts": x["n_experts"]}
                 for x in arms} if is_moe else None),
    }
    t_start = time.time()
    hdr = ("§2.4 — MoE: 5 cấu hình (top-k, expert), Friedman + Wilcoxon/Holm"
           if is_moe else
           "§2.3 — nhánh Recombinator: A/B, bootstrap ghép cặp + permutation")
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
    # phải dựng nhãn phân loại hay không. Cũ hơn, hai việc ấy chạy chỉ vì
    # `"train" in stages` — nghĩa là một lần chạy lại chỉ để in báo cáo vẫn trả
    # tiền cho 6400 ảnh, dù không candidate nào cần huấn luyện lại.
    arms_to_train: list[dict] = []
    cache: dict = {}
    try:
        cache = cache_state(arms, adapter_root, run_dir, is_moe)
        print_cache_state(cache, names, int(cfg["data"].get("n_test") or 0) or None)
        manifest["cache_at_start"] = cache
    except Exception as e:  # đọc cache KHÔNG được phép làm hỏng lần chạy
        print(f"⚠ không đọc được trạng thái cache ({e!r}) — coi như chưa có gì",
              flush=True)
    if "train" in stages:
        if a.retrain or not cache:
            arms_to_train = list(arms)
        else:
            arms_to_train = [x for x in arms
                             if not cache["arms"].get(x["id"], {}).get("trained")]
        if not arms_to_train:
            print("⚙ mọi candidate đã có adapter + train_stats.json — BỎ QUA "
                  "stage train và cả việc chuẩn bị tập train (--retrain để làm lại)",
                  flush=True)
    need_train_run = bool(arms_to_train)

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
    # Dựng TRƯỚC mọi thứ tốn kém. Nếu labels.npz có schema khác cái ta đoán thì
    # phát hiện ra ở giây thứ 30 rẻ hơn nhiều so với ở giờ thứ hai của stage
    # train — và ca tệ nhất không phải lỗi mà là nhãn lệch âm thầm.
    cls_pack = None
    need_b = any(x.get("recomb") for x in arms_to_train)
    if need_b and need_train_run:
        from . import labels_cls

        _hr("Nhãn phân loại (ClsHead)")
        # Chỉ DÒ SCHEMA ở đây. Nhãn thật phải căn theo thứ tự hàng của train_df,
        # mà train_df chưa tồn tại — nên nó được dựng ở stage train. Cái đắt cần
        # chết sớm là "labels.npz có đúng hình dạng ta nghĩ không", và câu đó
        # trả lời được ngay bây giờ, trước khi tải 3000 ảnh.
        try:
            info = labels_cls.probe(cfg)
            print(f"  labels.npz: {info.get('path')} · nhãn "
                  f"{info.get('label_array')} · key {info.get('key_array')}",
                  flush=True)
        except labels_cls.LabelsUnusable as e:
            print(f"\n✗ DỪNG — nhãn phân loại không dùng được:\n    {e}", flush=True)
            tracker.alert(f"§{exp} dừng: nhãn phân loại", str(e), "ERROR")
            tracker.finish(1)
            return 5

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

    # ── train ───────────────────────────────────────────────────────────────
    train_stats: list[dict] = []
    if need_train_run:
        _hr(f"Huấn luyện {len(arms_to_train)}/{len(arms)} candidate — "
            + ("candidate NẶNG nhất (E lớn nhất) TRƯỚC" if is_moe
               else "candidate B TRƯỚC"))
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
        bs = int(tcfg.get("per_device_batch_size", 1))
        accum = int(tcfg.get("grad_accum", 16))
        print(f"  batch {bs} × grad_accum {accum} = eff_bs {bs*accum} "
              f"[config, dùng chung cho cả hai candidate — §2.3 KHÔNG dò batch]",
              flush=True)
        manifest["batch"] = {"batch_size": bs, "grad_accum": accum,
                             "effective_batch": bs * accum, "source": "config"}

        for arm in train_order(arms, exp):
            aid = arm["id"]
            odir = adapter_root / aid
            is_b = bool(arm.get("recomb"))
            if tr_mod.already_trained(odir, is_b, is_moe) and not a.retrain:
                print(f"  ✓ {aid}: đã huấn luyện ở {odir} — bỏ qua "
                      f"(--retrain để làm lại)", flush=True)
                cached = json.loads((odir / "train_stats.json").read_text(encoding="utf-8"))
                train_stats.append(cached)
                manifest["arm_status"][aid] = {"trained": "cached"}
                continue
            tag = (f"top-k = {arm['top_k']} · {arm['n_experts']} expert · CÓ nhánh"
                   if is_moe else
                   ("candidate B · CÓ nhánh" if is_b else "candidate A · không nhánh"))
            print(f"\n── {arm.get('name', aid)} ({tag}) ──", flush=True)
            try:
                st = tr_mod.train_arm(cfg, arm, preset, train_df,
                                      cls_pack if is_b else None, odir, tracker,
                                      max_steps_override=dry_steps)
                train_stats.append(st)
                manifest["arm_status"][aid] = {"trained": True,
                                               "minutes": st["minutes"],
                                               "batch_size": st["batch_size"],
                                               "vram_peak_gb": st["vram_peak_gb"]}
                tracker.summary({f"train/{aid}/minutes": st["minutes"],
                                 f"train/{aid}/loss": st["train_loss"],
                                 f"train/{aid}/vram_peak_gb": st["vram_peak_gb"],
                                 f"train/{aid}/trainable_params":
                                     st.get("trainable_params")})
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


    # ── infer ───────────────────────────────────────────────────────────────
    pred_paths = {x["id"]: run_dir / "predictions" / f"{x['id']}.jsonl" for x in arms}
    if "infer" in stages:
        _hr("Sinh báo cáo")
        from .runners import build
        from .runners.base import RunnerUnusable

        for arm in arms:
            mcfg = arm_model_cfg(cfg, arm, adapter_root)
            print(f"\n── {arm.get('name', arm['id'])} ({mcfg['runner']}) ──", flush=True)
            t0 = time.time()
            try:
                build(cfg, mcfg, tracker).run(test_df, pred_paths[arm["id"]],
                                              log_every=tracker.log_every)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
            except RunnerUnusable as e:
                # Dừng ngay ở CẢ HAI mục, dù vì hai lý do khác nhau. §2.3: k = 2,
                # mất một candidate là mất cả phép so sánh. §2.4: Friedman đòi
                # thiết kế khối ĐẦY ĐỦ — chạy tiếp với 4/5 cột là lặng lẽ đổi
                # thí nghiệm, và bảng vẫn in ra bình thường.
                print(f"\n✗ {arm['id']} không sinh được: {e}", flush=True)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": False, "unusable": str(e)}
                tracker.alert(f"§{exp} dừng: một candidate không sinh được",
                              f"{arm['id']}: {e}", "ERROR")
                tracker.finish(1)
                return 6

    # ── score ───────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model

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
                scores[aid] = pd.read_csv(csv_path)
                print(f"  [{aid}] dùng lại {csv_path.name} ({len(scores[aid])} mẫu)",
                      flush=True)
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

    want_ids = [x["id"] for x in arms]
    missing = [i for i in want_ids if i not in scores]
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
    elif len(scores) != 2 or arm_a not in scores or arm_b not in scores:
        print(f"✗ cần ĐÚNG hai candidate có điểm (A={arm_a}, B={arm_b}); "
              f"đang có {sorted(scores)}. Phép so sánh ghép cặp không định "
              f"nghĩa được với một cột.", flush=True)
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
        from .stats_ab import compare_all, resolution_warning, summary_line

        cols = metric_columns(scores[arm_a])
        scfg = dict(cfg["stats"])
        print(f"  {len(cols)} metric × 2 kiểm định trên {align_info['n_common']} "
              f"ảnh ghép cặp (n_perm={scfg.get('n_perm')}, "
              f"n_boot={scfg.get('n_boot')})", flush=True)
        w = resolution_warning(int(scfg.get("n_perm", 10000)),
                               float(scfg.get("alpha", 0.05)), len(cols))
        if w:
            print(f"  ⚠ {w}", flush=True)
            manifest["stats_warning"] = w
        t0 = time.time()
        ab = compare_all(scores, cols, scfg, arm_a=arm_a, arm_b=arm_b,
                         seed=int(cfg.get("seed", 42)))
        ab.to_csv(ab_path, index=False)
        n_sig = int(ab["significant"].sum())
        n_mat = int(ab["significant_and_material"].sum())
        print(f"  -> {n_sig}/{len(ab)} metric bác H0 ở CẢ HAI kiểm định (Holm, "
              f"α={scfg.get('alpha', 0.05)}); {n_mat} trong đó có |dz| >= "
              f"{scfg.get('dz_floor', 0.1)}", flush=True)
        primary = str(cfg["metrics"].get("primary", "bertscore_f1"))
        line = summary_line(ab, primary, names.get(arm_a, arm_a), names.get(arm_b, arm_b))
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

"""§4 — QLoRA r4 thuần  vs  Recombinator(pretrain) + K=6·E=10 + QFormer + Cls.

    python -m src.run_section4                      # trọn gói
    python -m src.run_section4 --dry-run            # thử đường ống
    python -m src.run_section4 --stages stats,report
    python -m src.run_section4 --arms recomb_pre_k6e10

Mã thoát:
    0  xong
    1  thiếu cột điểm -> phép so sánh ghép cặp không định nghĩa được
    2  tham số sai, config chạm khoá đã khoá, checkpoint §3 không dùng được,
       hoặc preflight DỪNG
    3  một candidate không huấn luyện được
    4  rò rỉ train/test
    5  nhãn phân loại không dùng được
    6  một candidate không sinh được báo cáo
    7  stage `report` hỏng   -> chạy lại `--stages report`, kết quả còn nguyên
    8  bước đẩy Hub hỏng     -> chạy lại `--stages push`

══ §4 là một THÍ NGHIỆM SO SÁNH, không phải một lần huấn luyện ═══════════════

§3 huấn luyện MỘT model cho tử tế và dừng ở chỗ có trọng số — không điểm, không
kiểm định (README_3.md §1). §4 hỏi lại đúng câu mà §2.3 hỏi, với một candidate B
khác:

    A  qlora_r4          LoRA r=4 gắn vào decoder, KHÔNG nhánh, KHÔNG pretrain
                         -> ĐỒNG NHẤT với candidate A của §2.3
    B  recomb_pre_k6e10  A + Recombinator + QFormer(2 vector) + ClsHead (ASL)
                         + Experts K=6·E=10 + nhánh NẠP TỪ CHECKPOINT §3,
                         huấn luyện theo chế độ 2 của §3 (lr nhánh ×0.1,
                         λ-warmup TẮT)

k = 2, nên KHÔNG Friedman — cùng lý lẽ §2.3 mục 3: "kiểm định tổng thể" và
"kiểm định cặp" là cùng một câu hỏi, và chèn Friedman vào giữa chỉ thêm một
tầng Holm mà không thêm thông tin. Đường thống kê là `stats_ab` NGUYÊN VĂN.

══ Vì sao một điểm vào RIÊNG, nhưng KHÔNG một bộ điều kiện riêng ════════════

Hai ràng buộc kéo ngược chiều nhau, và §4 phải thoả CẢ HAI:

  1. §4 so được với §2.3/§2.4 -> nó phải dùng CÙNG 6400 mẫu train, CÙNG 3000
     ảnh test, cùng base, cùng lịch, cùng batch, cùng seed, cùng metric, cùng
     tham số kiểm định. README gốc đã nêu lý lẽ: với hai file cấu hình, câu ấy
     là một LỜI HỨA — một lần sửa `data.n_train` ở một bên là đủ để ba mục thôi
     so được với nhau, và không có gì báo.

  2. §4 phải NẠP được checkpoint của §3 và dùng ba khoá của chế độ 2 — thứ mà
     `run_recomb.py` KHÔNG được biết tới: selftest của §3 kiểm rằng file đó
     không nhắc `sup_pretrain`/`decoder_ft`/`pretrain_io`, và ranh giới ấy tồn
     tại để một lần chạy §3 không rơi vào nhánh code của `--exp 2.3` rồi ghi đè
     `runs/2.3_recombiner/`.

Lối ra: điểm vào riêng, ĐIỀU KIỆN ĐI VAY. `_load()` nạp `configs/recombiner.yaml`
— đúng file mà §2.3/§2.4 đọc — rồi phủ `configs/section4.yaml` lên trên, và nó
DỪNG nếu file §4 chạm vào bất kỳ khoá nào trong `LOCKED`. §4 không có cách nào
khai một `data.n_train` của riêng nó, kể cả khi ai đó gõ vào file.

`--set` thì KHÔNG bị chặn: nó là công cụ dòng lệnh, nó không sống sót qua một
lần chạy, và `manifest.json` ghi lại nguyên văn cấu hình đã giải quyết. Cấm cả
`--set` nghĩa là không dry-run được.

══ §4 KHÔNG chạy pretrain — nó NẠP checkpoint của §3 ════════════════════════

Nhánh của candidate B đến từ `runs/<§3>/pretrain/ckpt/best`, thư mục mà
`run_section3` ghi ra. §4 không có stage `pretrain`, không có khối `pretrain:`
trong config, và khai một khoá `pretrain.*` trong section4.yaml là DỪNG.

Đường dẫn KHÔNG viết cứng: `s3_ckpt_dir()` đọc `configs/section3.yaml` và dựng
`<output.dir>/<output.subdir>/pretrain/ckpt/best`. Viết cứng "runs/3_full/..."
là một LỜI HỨA rằng §3 vẫn ghi vào đúng chỗ ấy; đọc file của §3 là một sự thật.

Hệ quả phải NÓI RA, không phải giấu: §3 chạy chế độ 1 trên TOÀN BỘ train split,
còn §4 huấn luyện trên 6400 mẫu. Nhánh của candidate B vì thế đã thấy nhiều ảnh
hơn số ảnh candidate A từng thấy.

Đó KHÔNG phải rò rỉ — §3 chạy hàng rào `check_leak` với toàn bộ key của test
split, và `_resolve_pretrain` ở đây kiểm rằng checkpoint đọc ĐÚNG
`data.train_split_csv` mà §4 dùng, nên không một ảnh test nào lọt vào. Nó là
một thành phần ĐÃ PRETRAIN, cùng nghĩa với vision tower SigLIP đóng băng mà cả
hai candidate đứng trên.

Nhưng nó đổi PHẠM VI của câu trả lời: §4 đo "công thức §3 như nó được giao",
không đo "kiến trúc dưới điều kiện dữ liệu khớp nhau". Con số thật đọc từ META
được in ra và ghi vào `manifest["limitations"]`, nên báo cáo không im lặng được
về nó. `allow_pretrain_data_mismatch: false` biến nó thành DỪNG nếu ai đó muốn
ràng buộc chặt hơn.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import Cfg, add_common_args, hf_token, load_config, resolved_snapshot, \
    seed_everything
from .tracking import Tracker

# KHÔNG có stage `pretrain`: nhánh đến từ checkpoint §3, §4 không huấn luyện
# lại nó. Một stage mang tên `pretrain` mà không pretrain gì là chỗ tệ nhất để
# đặt một lời nói dối — người đọc log sẽ tin rằng §4 vừa học lại nhánh.
STAGES = ["data", "train", "infer", "score", "stats", "report", "push"]
ROOT = Path(__file__).resolve().parent.parent
BASE_CONFIG = ROOT / "configs" / "recombiner.yaml"      # điều kiện đi vay
S3_CONFIG = ROOT / "configs" / "section3.yaml"          # nơi checkpoint nằm
DEFAULT_CONFIG = ROOT / "configs" / "section4.yaml"     # phần §4 sở hữu

TRAIN_MANIFEST = "s4_trainset"
VAL_MANIFEST = "s4_valset"

# ── Khoá §4 KHÔNG được phép khai ────────────────────────────────────────────
# Mỗi mục là một khoá cấp một (cả khối) hoặc một khoá dạng chấm (một phần của
# khối). Đây là danh sách "điều kiện phải giống §2.3/§2.4", nên nó đọc được
# cạnh nhau thay vì rải rác trong code.
#
# `train` và `classifier` bị khoá THEO KHỐI nhưng chừa hai khoá: chúng là cách
# chế độ 2 của §3 khác chế độ scratch của §2.3, và chúng KHÔNG phải điều kiện
# thí nghiệm — chúng chỉ có nghĩa khi nhánh đã pretrain, tức là chỉ với
# candidate B, tức là chúng không đổi gì ở candidate A.
LOCKED: tuple[str, ...] = (
    "seed", "data", "base", "prompt", "generate", "labels",
    "recombiner", "metrics", "stats", "lora", "moe",
    "arms", "arms_moe",
    "train", "classifier",
)
LOCK_EXEMPT: tuple[str, ...] = (
    "train.pretrained_lr_mult",     # §3 §4.2 — nhánh đã hội tụ, lr ×0.1
    "train.freeze_pretrained",      # đóng băng hẳn thay vì hạ lr
    "classifier.lambda_warmup_pct_pretrained",   # §3 §4.3 — warmup TẮT
)
# §4 KHÔNG chạy chế độ 1, nên CẢ KHỐI `pretrain:` là chết ở đây. Khai bất cứ
# khoá nào dưới nó là dựng một khoá trông như điều khiển được một phase không
# tồn tại — và ca hỏng là ai đó chỉnh `pretrain.n_epochs` rồi tin rằng §4 vừa
# học lại nhánh trên 6400 mẫu, trong khi nó nạp nguyên checkpoint toàn-split
# của §3.
PRETRAIN_INERT_PREFIX = "pretrain"


def _hr(t: str) -> None:
    print(f"\n{'─' * 4} {t} {'─' * max(4, 66 - len(t))}", flush=True)


def _env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    return info


# ══ Nạp cấu hình ════════════════════════════════════════════════════════════
def _leaf_keys(d: dict, prefix: str = "") -> list[str]:
    """Chỉ những khoá LÁ, dạng chấm.

    Khoá cha không được tính là vi phạm: `train:` xuất hiện trong section4.yaml
    chỉ vì nó CHỨA `train.pretrained_lr_mult`, và khoá con ấy được miễn. Tính cả
    khoá cha nghĩa là không khai nổi một khoá miễn nào — hàng rào khi đó chặn
    đúng thứ nó phải cho qua, và cách sửa quen tay là tắt hàng rào.
    """
    out = []
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out += _leaf_keys(v, p + ".")
        else:
            out.append(p)
    return out


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def violations(s4_raw: dict) -> list[str]:
    """Khoá nào trong file §4 chạm vào điều kiện đi vay. Rỗng = hợp lệ.

    Tách khỏi `_load` để selftest gọi được mà không cần đọc đĩa của §2.3.
    """
    bad = []
    for key in _leaf_keys(s4_raw):
        if key in LOCK_EXEMPT:
            continue
        if key.split(".", 1)[0] == PRETRAIN_INERT_PREFIX:
            bad.append(
                f"{key} — §4 KHÔNG có phase pretrain. Nhánh của candidate B nạp "
                f"từ checkpoint mà §3 đã ghi ra, nên cả khối `pretrain:` là "
                f"chết ở đây: khai nó là dựng một khoá trông như điều khiển "
                f"được một phase không tồn tại. Sửa chế độ 1 thì sửa "
                f"configs/section3.yaml rồi chạy lại §3.")
            continue
        if key.split(".", 1)[0] in LOCKED:
            bad.append(
                f"{key} — thuộc điều kiện phải GIỐNG §2.3/§2.4, và nó đi vay từ "
                f"configs/recombiner.yaml. Sửa ở đó nếu thật sự muốn đổi cho cả "
                f"ba mục; sửa ở đây là làm ba bảng thôi so được với nhau mà "
                f"không có gì báo.")
    return bad


def _load(s4_path: str | Path, overrides: list[str] | None) -> tuple[Cfg, dict]:
    """(cfg đã gộp, khối §4 thô). DỪNG bằng ValueError nếu §4 chạm khoá đã khoá.

    Thứ tự cố định: recombiner.yaml TRƯỚC, section4.yaml SAU, `--set` CUỐI.
    `--set` đi cuối vì nó là công cụ dòng lệnh — nó không sống sót qua lần chạy
    và `manifest.json` ghi lại nguyên văn thứ nó tạo ra.
    """
    import yaml

    if not BASE_CONFIG.exists():
        raise FileNotFoundError(
            f"§4 đi vay điều kiện từ {BASE_CONFIG}, và file đó không có. Đây "
            f"không phải một mặc định thiếu — nó là tập 6400/3000 mà cả ba mục "
            f"đứng trên.")
    s4_path = Path(s4_path)
    if not s4_path.exists():
        raise FileNotFoundError(f"Không thấy config §4: {s4_path}")

    base_raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")) or {}
    s4_raw = yaml.safe_load(s4_path.read_text(encoding="utf-8")) or {}

    bad = violations(s4_raw)
    if bad:
        raise ValueError(
            "config §4 chạm vào điều kiện đi vay từ configs/recombiner.yaml:\n"
            + "\n".join(f"    ✗ {b}" for b in bad))

    merged = _deep_merge(base_raw, s4_raw)
    # `--set` đi qua load_config để dùng lại đúng phép ép kiểu YAML của nó.
    tmp = Path(s4_path).with_suffix(".resolved.tmp.yaml")
    try:
        tmp.write_text(yaml.safe_dump(merged, allow_unicode=True, sort_keys=False),
                       encoding="utf-8")
        cfg = load_config(tmp, overrides)
    finally:
        tmp.unlink(missing_ok=True)
    cfg["_config_path"] = f"{BASE_CONFIG} + {s4_path}"
    cfg["_config_base"] = str(BASE_CONFIG)
    cfg["_config_s4"] = str(s4_path)
    return cfg, s4_raw


# ══ Candidate ═══════════════════════════════════════════════════════════════
def s4_arms(cfg: Cfg, only: set[str] | None = None) -> list[dict]:
    """Hai candidate của §4, ba cờ đã CHUẨN HOÁ và cả ba BẮT BUỘC.

    `recomb` và `pretrain` không có mặc định ngầm vì chúng là toàn bộ khác biệt
    giữa hai cột. Một arm quên khai `pretrain:` mà lặng lẽ nhận `False` nghĩa là
    cột B mang nhãn "đã pretrain" trong khi nhánh khởi đầu từ nhiễu — và đường
    loss trông hoàn toàn bình thường trong lúc đó.

    `top_k`/`n_experts` thì đi THEO CẶP: có một mà thiếu một là DỪNG. Một
    candidate chạy `top_k` mặc định trong khi bảng ghi `E=10` là một hàng mang
    nhãn sai, và đó là loại lỗi không ai đi kiểm tra.
    """
    raw = cfg.get("arms_s4")
    if not raw:
        raise ValueError(
            "config §4 không có `arms_s4`. Khoá này CỐ Ý không tên `arms`: "
            "`arms` là hai candidate của §2.3 và nó nằm trong recombiner.yaml, "
            "nên trùng tên nghĩa là một lần nạp sai thứ tự sẽ chạy §4 với "
            "candidate của §2.3 — cùng số cột, bảng vẫn in ra bình thường.")
    arms = []
    for a in raw:
        a = dict(a)
        aid = a.get("id")
        if not a.get("enabled", True):
            continue
        for key in ("recomb", "pretrain"):
            if key not in a:
                raise ValueError(
                    f"candidate §4 {aid!r} không khai `{key}:` (true/false). Hai "
                    f"cờ này là thứ phân biệt hai cột của bảng — mặc định ngầm ở "
                    f"đây là một cột mang nhãn của một cấu hình chưa từng chạy.")
        a["recomb"] = bool(a["recomb"])
        a["pretrain"] = bool(a["pretrain"])
        has_k, has_e = "top_k" in a, "n_experts" in a
        if has_k != has_e:
            raise ValueError(
                f"candidate §4 {aid!r} khai {'top_k' if has_k else 'n_experts'} "
                f"mà thiếu khoá kia. Hai con số đi theo CẶP: một mình `top_k` "
                f"nghĩa là `n_experts` rơi vào mặc định, và hàng kết quả mang "
                f"nhãn sai.")
        if a["pretrain"] and not a["recomb"]:
            raise ValueError(
                f"candidate §4 {aid!r}: pretrain = true nhưng recomb = false. "
                f"Không nhánh thì không có gì để pretrain — chế độ 1 huấn luyện "
                f"ĐÚNG Recombinator + QFormer + ClsHead.")
        if has_k and not a["recomb"]:
            raise ValueError(
                f"candidate §4 {aid!r}: có Experts nhưng recomb = false. Không "
                f"nhánh thì không có `vec_2` để Route_Gate đọc, nên đó không "
                f"phải một biến thể — nó là một thí nghiệm khác.")
        arms.append(a)
    if only:
        arms = [a for a in arms if a["id"] in only]
    if not arms:
        raise ValueError("không candidate §4 nào được bật")
    return arms


def train_order(arms: list[dict]) -> list[dict]:
    """Candidate DỄ HỎNG NHẤT chạy trước — quy ước vận hành #1 của repo.

    Ở §4 đó là candidate B, và nó dễ hỏng theo BA đường cùng lúc: nó là
    candidate duy nhất có nhánh (§2.3), nó mang `n_experts` lớn nhất của cả
    repo — E=10 nhân thẳng vào activation LoRA và vào số tham số học được
    (§2.4) — và nó là candidate duy nhất đứng trên một checkpoint pretrain
    (§3). Chạy nó cuối là trả xong tiền GPU của A rồi mới biết bảng thiếu cột,
    và phép so sánh ghép cặp không định nghĩa được với một cột.

    Thứ tự này KHÔNG được phép ảnh hưởng tới kết quả, và đó là một điều kiện
    được cài đặt: seed đặt lại trước MỖI candidate, batch là hằng số, VRAM được
    dọn giữa hai lượt.
    """
    return sorted(arms, key=lambda a: (0 if a.get("recomb") else 1, str(a["id"])))


def arm_cfg(cfg: Cfg, arm: dict) -> Cfg:
    """Bản sao cfg đã giải quyết cho MỘT candidate.

    `moe.enabled` là một khoá CẤP CFG, không phải cấp arm (`train_recomb.
    moe_spec` đọc nó từ `cfg`). Ở §2.4 điều đó đúng vì cả năm candidate đều
    chạy MoE; ở §4 hai candidate khác nhau đúng ở chỗ ấy. Nên mỗi candidate
    nhận một BẢN SAO SÂU thay vì một lần bật/tắt trên cfg dùng chung: một lần
    quên khôi phục sẽ cho candidate A chạy qua `moe_spec` với `top_k` của B —
    và `train_arm` DỪNG ở đó, nhưng chỉ vì có một phép kiểm khác bắt được, chứ
    không vì chỗ này đúng.
    """
    c = Cfg(copy.deepcopy(dict(cfg)))
    c.setdefault("moe", {})["enabled"] = ("top_k" in arm)
    # `mode` đi vay từ recombiner.yaml mang giá trị `scratch` — đúng cho
    # candidate A, SAI cho candidate B. Nó không điều khiển gì ở đường chạy này
    # (`train_recomb` rẽ theo `cfg["pretrain_init"]`), nhưng nó đi thẳng vào
    # `manifest.json` qua `resolved_snapshot`, nên để nguyên là để manifest nói
    # rằng candidate B huấn luyện from-scratch.
    c["mode"] = "decoder_ft" if arm.get("pretrain") else "scratch"
    return c


def s3_ckpt_dir(s3_path: Path = S3_CONFIG) -> Path:
    """`ckpt/best` của §3, dựng từ CHÍNH configs/section3.yaml.

    Viết cứng "runs/3_full/pretrain/ckpt/best" là một LỜI HỨA rằng §3 vẫn ghi
    vào đúng chỗ ấy — và ngày ai đó đổi `output.subdir` của §3, §4 sẽ báo
    "chưa có checkpoint" cho một checkpoint đang nằm ngay đó. Đọc file của §3
    là một sự thật.

    `run_section3.default_init` đóng cứng đoạn "pretrain/ckpt/best" trong mã
    nguồn (không phải một khoá config), nên nó được lặp lại ở đây — và
    selftest đối chiếu hai chỗ ấy với nhau.
    """
    import yaml

    p3 = Path(s3_path)
    if not p3.exists():
        raise FileNotFoundError(
            f"§4 nạp nhánh từ checkpoint của §3, và nó định vị thư mục ấy bằng "
            f"{p3} — file đó không có.")
    raw = yaml.safe_load(p3.read_text(encoding="utf-8")) or {}
    o = dict(raw.get("output") or {})
    # KHÔNG mặc định ngầm cho `subdir`. Một giá trị lui ở đây là đúng thứ hàm
    # này tồn tại để tránh: ngày ai đó xoá khoá ấy khỏi section3.yaml, §4 sẽ đi
    # tìm một thư mục không ai khai và báo "chưa có checkpoint" — cho một
    # checkpoint có thể đang nằm ngay cạnh.
    if not o.get("subdir"):
        raise ValueError(
            f"{p3} không khai `output.subdir` — không suy ra được §3 ghi "
            f"checkpoint ở đâu.")
    return (Path(str(o.get("dir", "./runs"))) / str(o["subdir"])
            / "pretrain" / "ckpt" / "best")


# Những khoá của META quyết định nhánh ĐỌC ĐÚNG phân phối đặc trưng nó đã học.
# Lệch bất kỳ khoá nào: hình dạng vẫn khớp, checkpoint vẫn nạp được, điểm số
# vẫn ra số — và nhánh đang đọc một thứ khác thứ nó được huấn luyện trên.
SHAPE_KEYS = ("n_query", "swiglu_hidden", "n_heads", "siglip_dim", "token_type")


def _resolve_pretrain(cfg: Cfg, spec: str) -> tuple[Path | None, bool, dict]:
    """-> (thư mục ckpt | None, from_scratch, ghi chú). ValueError = DỪNG.

    Giải quyết ở GIÂY THỨ NHẤT, trước stage `data`: phát hiện "chưa chạy §3"
    sau khi đã tải và resize 3000 + 6400 ảnh là trả xong tiền rồi mới biết.
    """
    from . import pretrain_io

    if spec.lower() == "none":
        return None, True, {}

    want = spec or str(s3_ckpt_dir())
    if not spec and not pretrain_io.pretrain_exists(Path(want)):
        raise ValueError(
            f"chưa có checkpoint pretrain của §3 ở {want}.\n"
            f"    §4 KHÔNG tự pretrain — nhánh của candidate B đến từ §3. Ba cách:\n"
            f"      * chạy §3 trước      : bash scripts/run_section3.sh --mode sup_pretrain\n"
            f"      * hoặc lấy từ Hub    : --init-from hub:OWNER/NAME/best\n"
            f"      * hoặc trỏ thư mục   : --init-from <dir>\n"
            f"    Muốn cố ý chạy from-scratch thì NÓI RA: --init-from none\n"
            f"    (im lặng chạy from-scratch ở đây nghĩa là cột B mang nhãn 'đã "
            f"pretrain'\n     dựng từ một nhánh ngẫu nhiên — và không ai phát "
            f"hiện được về sau.)")
    try:
        src = pretrain_io.resolve_source(
            want, cache_dir=cfg["data"].get("cache_dir", "./data"))
        meta = pretrain_io.read_meta(src)
    except Exception as e:
        raise ValueError(f"không lấy được checkpoint pretrain ({want!r}): {e}")

    note: dict = {"source": str(src)}

    # 1. Cùng base. Vision tower khác = phân phối đặc trưng khác, và hai base
    #    cùng kích thước thì hình dạng vẫn khớp nên không gì khác bắt được.
    if meta.get("base_hf_id") and \
            str(meta["base_hf_id"]) != str(cfg["base"]["hf_id"]):
        raise ValueError(
            f"checkpoint học trên vision tower của {meta['base_hf_id']}, lần "
            f"chạy này dùng {cfg['base']['hf_id']}.")

    # 2. Cùng hình dạng nhánh. So với `spec` NGUYÊN VĂN mà checkpoint mang theo,
    #    không với thứ ai đó nhớ là đã đặt.
    mr = dict((meta.get("spec") or {}).get("recombiner") or {})
    cr = dict(cfg.get("recombiner") or {})
    for k in SHAPE_KEYS:
        if k in mr and k in cr and mr[k] != cr[k]:
            raise ValueError(
                f"checkpoint dựng với `recombiner.{k} = {mr[k]!r}`, §4 chạy với "
                f"{cr[k]!r}. Nhánh sẽ đọc một thứ khác thứ nó được huấn luyện "
                f"trên.")

    # 3. Cùng đường ảnh. Nhánh học trên `last_hidden_state` của vision tower,
    #    và một độ phân giải khác (hay pan&scan bật) cho ra đặc trưng khác —
    #    trong khi số token vào vẫn có thể khớp.
    md = dict(meta.get("data") or {})
    if md.get("image_max_side") and int(md["image_max_side"]) != \
            int(cfg["data"].get("image_max_side", 512)):
        raise ValueError(
            f"checkpoint học trên ảnh cạnh dài <= {md['image_max_side']}px, §4 "
            f"chạy {cfg['data'].get('image_max_side')}px.")
    pas_cfg = bool((cfg["base"].get("image") or {}).get("do_pan_and_scan", False))
    if md.get("do_pan_and_scan") is not None and \
            bool(md["do_pan_and_scan"]) != pas_cfg:
        raise ValueError(
            f"checkpoint học với do_pan_and_scan = {md['do_pan_and_scan']}, §4 "
            f"chạy {pas_cfg}. Mỗi mảnh phụ thêm một lượt 4096 patch, nên đây là "
            f"một đường ảnh KHÁC.")

    # 4. Hàng rào rò rỉ ĐI THEO được hay không.
    #    §3 chạy `check_leak` với toàn bộ key của test split, nên checkpoint của
    #    nó sạch — NHƯNG chỉ khi nó đọc cùng file split mà §4 đọc. Đọc file
    #    khác thì bảo đảm ấy không chuyển sang được, và triệu chứng của rò rỉ là
    #    "kết quả tốt bất ngờ": ca hỏng duy nhất người ta có xu hướng KHÔNG đi
    #    kiểm tra.
    want_csv = str(cfg["data"].get("train_split_csv"))
    got_csv = str(md.get("split_csv") or "")
    if got_csv and got_csv != want_csv:
        raise ValueError(
            f"checkpoint pretrain đọc split {got_csv!r}, §4 huấn luyện trên "
            f"{want_csv!r}.\n"
            f"    Bảo đảm 'không có ảnh test nào lọt vào nhánh' đến từ hàng rào "
            f"check_leak mà §3\n    chạy TRÊN FILE SPLIT CỦA NÓ — nó không "
            f"chuyển sang một file khác được.")
    note["split_csv"] = got_csv or want_csv

    # 5. Số mẫu — CÔNG BỐ, không chặn (trừ khi được yêu cầu chặn).
    n_pre = md.get("n_train")
    note["n_train_pretrain"] = int(n_pre) if n_pre is not None else None
    note["n_train_section4"] = int(cfg["data"]["n_train"])
    if n_pre is not None and int(n_pre) != int(cfg["data"]["n_train"]):
        if not bool(cfg.get("allow_pretrain_data_mismatch", True)):
            raise ValueError(
                f"checkpoint pretrain học trên {int(n_pre)} mẫu, §4 huấn luyện "
                f"trên {int(cfg['data']['n_train'])}, và "
                f"`allow_pretrain_data_mismatch: false`.")
        note["limitation"] = (
            f"Nhánh của candidate B được pretrain trên {int(n_pre)} ảnh (§3, "
            f"toàn bộ train split), trong khi CẢ HAI candidate chỉ huấn luyện "
            f"decoder trên {int(cfg['data']['n_train'])} mẫu. Không phải rò rỉ "
            f"— cùng file split, đã qua check_leak của §3 với toàn bộ key test "
            f"— nhưng nó có nghĩa là §4 đo CÔNG THỨC §3 NHƯ NÓ ĐƯỢC GIAO, không "
            f"đo kiến trúc dưới điều kiện dữ liệu khớp nhau.")
    note["meta"] = {k: meta.get(k) for k in
                    ("epoch", "n_epochs", "seen_samples", "val", "data",
                     "pretrained_modules", "untrained_modules", "n_labels",
                     "seed", "base_hf_id", "base_method")}
    return src, False, note


# ══ main ════════════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="§4 — QLoRA r4 thuần vs Recombinator(pretrain)+K=6·E=10, "
                    "cùng 6400/3000 của §2.3, kiểm định ghép cặp k=2.")
    add_common_args(ap)
    ap.set_defaults(config=str(DEFAULT_CONFIG))
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=None, help="chỉ chạy các arm này (phẩy ngăn)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="8 ảnh · 16 mẫu · 4 bước · 200 lần mô phỏng · KHÔNG "
                         "đẩy Hub. Thử ĐƯỜNG ỐNG, không dò VRAM.")
    ap.add_argument("--init-from", default=None,
                    help="nhánh đã pretrain cho candidate B. Mặc định: ckpt/best "
                         "của §3, đọc từ configs/section3.yaml. Ghi đè bằng "
                         "<dir> | hub:OWNER/NAME/best[@rev] | none (arm đối "
                         "chứng from-scratch, phải NÓI RA).")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args(argv)

    # ── cấu hình ────────────────────────────────────────────────────────────
    try:
        cfg, s4_raw = _load(a.config, a.overrides)
    except (ValueError, FileNotFoundError) as e:
        print(f"\n✗ DỪNG — {e}", flush=True)
        return 2
    cfg["experiment"] = "4"

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for s in stages:
        if s not in STAGES:
            print(f"✗ stage không biết: {s!r} (hợp lệ: {', '.join(STAGES)})")
            return 2
    only = set(x.strip() for x in a.arms.split(",")) if a.arms else None
    try:
        arms = s4_arms(cfg, only)
    except ValueError as e:
        print(f"\n✗ DỪNG — {e}", flush=True)
        return 2

    arm_a = next((x["id"] for x in arms if not x["recomb"]), None)
    arm_b = next((x["id"] for x in arms if x["recomb"]), None)

    # `report_recomb` suy A/B từ cờ `recomb` trong `cfg["arms"]`. §4 giữ
    # candidate ở `arms_s4` (xem docstring `s4_arms`), nên chỗ nối hai thứ ấy
    # phải là MỘT dòng, ở đây, sau khi `s4_arms` đã kiểm xong — chứ không phải
    # một khoá `arms` thứ hai trong file YAML mà ai đó có thể sửa lệch.
    cfg["arms"] = [dict(x) for x in arms]

    dry_steps = None
    if a.dry_run:
        cfg["data"]["n_test"] = 8
        cfg["data"]["n_train"] = 16
        cfg["data"]["n_val"] = 8
        cfg["stats"]["n_perm"] = 200
        cfg["stats"]["n_boot"] = 200
        cfg["output"]["push"] = False
        cfg["output"]["subdir"] = str(cfg["output"]["subdir"]) + "_dryrun"
        cfg["output"]["adapter_dir"] = str(cfg["output"]["adapter_dir"]) + "_dryrun"
        cfg.setdefault("wandb", {})["tags"] = list(
            cfg.get("wandb", {}).get("tags", []) or []) + ["dry-run"]
        dry_steps = 4
        print("⚙ DRY-RUN: 8 ảnh test · 16 mẫu train · 4 bước · không đẩy Hub.\n"
              "  Phép thử ĐƯỜNG ỐNG, không phải phép dò VRAM: nó chạy với đúng "
              "batch trong config,\n  và 4 bước trên 16 mẫu ngắn không nói được "
              "gì về VRAM đỉnh của 6400 mẫu.", flush=True)
    if a.no_push:
        cfg["output"]["push"] = False
    if a.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    seed_everything(int(cfg.get("seed", 42)))

    run_id = a.run_id or str(cfg["output"]["subdir"])
    run_dir = Path(cfg["output"].get("dir", "./runs")) / run_id
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    adapter_root = Path(str(cfg["output"]["adapter_dir"]))
    names = {x["id"]: x.get("name", x["id"]) for x in arms}
    mdir = run_dir / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id, "experiment": "4",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": resolved_snapshot(cfg), "env": _env_info(),
        "config_base": str(BASE_CONFIG), "config_s4": str(a.config),
        "stages_run": stages, "arms_declared": [x["id"] for x in arms],
        "arm_a": arm_a, "arm_b": arm_b, "arm_status": {},
        "moe": {x["id"]: {"top_k": x["top_k"], "n_experts": x["n_experts"]}
                for x in arms if "top_k" in x},
    }
    t_start = time.time()

    # ── Nguồn nhánh — giải quyết ở GIÂY THỨ NHẤT ────────────────────────────
    # Trước cả banner, và rất trước stage `data`: phát hiện "chưa chạy §3" sau
    # khi đã tải và resize 3000 + 6400 ảnh là trả xong tiền rồi mới biết.
    pre_src = None
    pre_scratch = False
    pre_note: dict = {}
    if "train" in stages and any(x["pretrain"] for x in arms):
        try:
            pre_src, pre_scratch, pre_note = _resolve_pretrain(
                cfg, str(a.init_from or "").strip())
        except ValueError as e:
            print(f"\n✗ DỪNG — {e}", flush=True)
            return 2
        manifest["pretrain_init"] = str(pre_src) if pre_src else None
        manifest["pretrain_note"] = pre_note
        if pre_note.get("limitation"):
            manifest.setdefault("limitations", []).append(pre_note["limitation"])

    print("\n═══ §4 — QLoRA r4 thuần  vs  Recombinator(pretrain §3) · K=6 · E=10 "
          "═══", flush=True)
    for x in arms:
        bits = []
        bits.append("nhánh + QFormer + ClsHead" if x["recomb"] else "KHÔNG nhánh")
        if "top_k" in x:
            bits.append(f"Experts k={x['top_k']}·E={x['n_experts']}")
        bits.append("nhánh TỪ CKPT §3" if x["pretrain"] else "không pretrain")
        print(f"  {x['id']:<18} {x.get('name','')}\n"
              f"  {'':<18} {' · '.join(bits)}", flush=True)
    print(f"⚙ điều kiện đi vay: {BASE_CONFIG.name} — "
          f"{cfg['data']['n_train']} mẫu train · {cfg['data']['n_test']} ảnh "
          f"test · seed {cfg.get('seed')}", flush=True)
    print(f"⚙ run dir : {run_dir}\n⚙ adapters: {adapter_root}", flush=True)
    if pre_scratch:
        print("⚙ nhánh   : NGẪU NHIÊN (--init-from none) — arm ĐỐI CHỨNG, "
              "KHÔNG phải candidate B của §4", flush=True)
    elif pre_src:
        m = dict(pre_note.get("meta") or {})
        v = dict(m.get("val") or {})
        print(f"⚙ nhánh   : {pre_src}\n"
              f"            epoch {m.get('epoch')}/{m.get('n_epochs')} · "
              f"{m.get('seen_samples')} lượt mẫu · val cls_f1_pos "
              f"{v.get('cls_f1_pos')}", flush=True)
        if pre_note.get("limitation"):
            n_pre = pre_note.get("n_train_pretrain")
            print(f"⚠ GIỚI HẠN — nhánh của B pretrain trên {n_pre} ảnh, cả hai "
                  f"candidate huấn luyện decoder trên\n"
                  f"  {pre_note.get('n_train_section4')} mẫu. Không rò rỉ (cùng "
                  f"split, đã qua check_leak của §3), nhưng §4 đo\n"
                  f"  CÔNG THỨC §3 NHƯ NÓ ĐƯỢC GIAO — không đo kiến trúc dưới "
                  f"điều kiện dữ liệu khớp nhau.\n"
                  f"  Câu này đi vào manifest.limitations và vào REPORT.md.",
                  flush=True)
    print(f"⚙ k = 2 -> KHÔNG Friedman. Permutation + bootstrap ghép cặp BCa "
          f"(stats_ab), Holm riêng từng họ.", flush=True)

    tracker = Tracker(cfg, run_id=run_id, config_snapshot=manifest["config"])
    if tracker.url:
        manifest["wandb_url"] = tracker.url
        print(f"⚙ W&B     : {tracker.url}", flush=True)
    tracker.summary({"run/stages": ",".join(stages), "run/experiment": "4",
                     "run/arms": ",".join(manifest["arms_declared"]),
                     **{f"env/{k}": v for k, v in manifest["env"].items()}})
    if not hf_token():
        print("⚠ Không thấy HF_TOKEN — google/gemma-3-4b-it là gated repo.\n"
              "    export HF_TOKEN=hf_xxxxxxxx", flush=True)

    from . import data as data_mod
    from . import data_train as dt_mod

    need_train = "train" in stages

    # ── nhãn: dò schema TRƯỚC mọi thứ tốn kém ───────────────────────────────
    if need_train and any(x["recomb"] for x in arms):
        from . import labels_cls

        _hr("Nhãn phân loại (ClsHead)")
        try:
            info = labels_cls.probe(cfg)
            print(f"  labels.npz: {info.get('path')} · nhãn "
                  f"{info.get('label_array')} · key {info.get('key_array')}",
                  flush=True)
        except labels_cls.LabelsUnusable as e:
            print(f"\n✗ DỪNG — nhãn phân loại không dùng được:\n    {e}", flush=True)
            tracker.alert("§4 dừng: nhãn phân loại", str(e), "ERROR")
            tracker.finish(1)
            return 5

    # ── data ────────────────────────────────────────────────────────────────
    train_df = val_df = None
    if "data" in stages:
        _hr("Dữ liệu")
        test_df = data_mod.prepare(cfg, tracker)
        if need_train:
            try:
                train_df, val_df = dt_mod.prepare_train(
                    cfg, test_df, tracker, name=TRAIN_MANIFEST,
                    val_name=VAL_MANIFEST)
            except dt_mod.DataLeak as e:
                print(f"\n✗ DỪNG — RÒ RỈ DỮ LIỆU:\n    {e}", flush=True)
                tracker.alert("§4 dừng: rò rỉ train/test", str(e), "ERROR")
                tracker.finish(1)
                return 4
    else:
        test_df = data_mod.load_manifest(cfg)
        print(f"⚙ dùng lại manifest test: {len(test_df)} mẫu", flush=True)
        if need_train:
            train_df, val_df = dt_mod.load_train_manifest(
                cfg, TRAIN_MANIFEST, VAL_MANIFEST)
            print(f"⚙ dùng lại manifest train: {len(train_df)} · val "
                  f"{len(val_df)}", flush=True)
    tracker.summary({"data/n_test": len(test_df)})
    if train_df is not None:
        manifest["n_train"] = len(train_df)
        tracker.summary({"data/n_train": len(train_df)})

    # ── nhãn căn theo ĐÚNG thứ tự hàng ──────────────────────────────────────
    # `val_df` đi qua đây nhưng §4 KHÔNG dùng nó: chọn epoch là việc của chế độ
    # 1, và chế độ 1 thuộc §3. Nó vẫn được chuẩn bị vì `data.n_val` là một điều
    # kiện ĐI VAY — bỏ nó đi là làm stage `data` của §4 khác stage `data` của
    # §2.3, tức là đúng thứ mục 1 của docstring cấm.
    cls_train = None
    if need_train and any(x["recomb"] for x in arms):
        from . import labels_cls

        lut, linfo = labels_cls.load_map(cfg, verbose=True)
        n_labels = int((cfg.get("classifier", {}) or {}).get("n_labels")
                       or linfo["n_labels"])
        y, m, lstats = labels_cls.attach(
            cfg, train_df["key"].astype(str).tolist(), lut, n_labels, "train")
        cls_train = {"y": y, "mask": m, "n_labels": n_labels,
                     "stats": dict(lstats, n_labels=n_labels,
                                   label_names=linfo.get("label_names"),
                                   class_frac=linfo.get("class_frac"))}
        manifest["labels"] = cls_train["stats"]

    # ══ Huấn luyện hai candidate ════════════════════════════════
    train_stats: list[dict] = []
    if "train" in stages:
        from . import train_recomb as tr_mod
        from .backbone import TrainUnusable, preflight_adapters

        _hr("Huấn luyện — candidate B (nhánh · E=10 · pretrain) TRƯỚC")
        preset = tr_mod.resolve_preset(cfg)
        pf_err, pf_warn = preflight_adapters(
            [{"method": preset["method"], "rank": preset["rank"]}])
        for arm in arms:
            e2, w2 = tr_mod.preflight_recomb(arm_cfg(cfg, arm), [arm])
            pf_err += e2
            pf_warn += w2
        for w in dict.fromkeys(pf_warn):
            print(f"  {w}", flush=True)
        if pf_err:
            for e in dict.fromkeys(pf_err):
                print(f"  ✗ {e}", flush=True)
            print("\n✗ DỪNG ở preflight — sửa theo hướng dẫn rồi chạy lại.",
                  flush=True)
            manifest["preflight_errors"] = list(dict.fromkeys(pf_err))
            tracker.alert("§4 dừng ở preflight", "\n".join(pf_err), "ERROR")
            tracker.finish(1)
            (run_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")
            return 2
        manifest["lora_preset"] = preset
        tcfg = dict(cfg["train"])
        bs, accum = int(tcfg["per_device_batch_size"]), int(tcfg["grad_accum"])
        print(f"  ✓ preset LoRA {preset['id']} (r={preset['rank']} "
              f"alpha={preset['alpha']}) · batch {bs} × grad_accum {accum} = "
              f"eff_bs {bs*accum}\n    [hằng số trong config, dùng chung cho CẢ "
              f"HAI candidate — §4 KHÔNG dò batch]", flush=True)
        manifest["batch"] = {"batch_size": bs, "grad_accum": accum,
                             "effective_batch": bs * accum, "source": "config"}

        for arm in train_order(arms):
            aid = arm["id"]
            odir = adapter_root / aid
            c = arm_cfg(cfg, arm)
            # `pretrain_init` là khoá mà `train_recomb._train_once` đọc. Đặt nó
            # TRÊN BẢN SAO của candidate này, không trên cfg dùng chung: đặt
            # trên cfg chung thì candidate A cũng thấy nó, và train_recomb sẽ
            # nạp một nhánh đã pretrain vào một candidate không có nhánh.
            c["pretrain_init"] = (str(pre_src) if (arm["pretrain"] and pre_src
                                                   and not pre_scratch) else None)
            if arm["pretrain"] and pre_scratch:
                c["pretrain_from_scratch_declared"] = True
            if tr_mod.already_trained(odir, arm["recomb"], "top_k" in arm) \
                    and not a.retrain:
                print(f"  ✓ {aid}: đã huấn luyện ở {odir} — bỏ qua "
                      f"(--retrain để làm lại)", flush=True)
                train_stats.append(json.loads(
                    (odir / "train_stats.json").read_text(encoding="utf-8")))
                manifest["arm_status"][aid] = {"trained": "cached"}
                continue
            bits = ["CÓ nhánh" if arm["recomb"] else "KHÔNG nhánh"]
            if "top_k" in arm:
                bits.append(f"k={arm['top_k']}·E={arm['n_experts']}")
            bits.append("nhánh TỪ PRETRAIN" if c["pretrain_init"] else "nhánh NGẪU NHIÊN")
            print(f"\n── {arm.get('name', aid)} ({' · '.join(bits)}) ──", flush=True)
            # Seed đặt lại TRƯỚC mỗi candidate: thứ tự chạy không được phép ảnh
            # hưởng tới bảng, và đó là một điều kiện được cài đặt.
            seed_everything(int(cfg.get("seed", 42)))
            try:
                st = tr_mod.train_arm(
                    c, arm, preset, train_df,
                    cls_train if arm["recomb"] else None, odir, tracker,
                    max_steps_override=dry_steps)
                train_stats.append(st)
                manifest["arm_status"][aid] = {
                    "trained": True, "minutes": st["minutes"],
                    "batch_size": st["batch_size"],
                    "vram_peak_gb": st["vram_peak_gb"],
                    "pretrain_init": st.get("pretrain_init")}
                tracker.summary({f"train/{aid}/minutes": st["minutes"],
                                 f"train/{aid}/loss": st["train_loss"],
                                 f"train/{aid}/vram_peak_gb": st["vram_peak_gb"],
                                 f"train/{aid}/trainable_params":
                                     st.get("trainable_params")})
            except TrainUnusable as e:
                print(f"\n✗ {aid} KHÔNG huấn luyện được: {e}", flush=True)
                tracker.alert("§4 dừng ở train", f"{aid}: {e}", "ERROR")
                manifest["arm_status"][aid] = {"trained": False, "unusable": str(e)}
                tracker.finish(1)
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
                return 3
    else:
        for arm in arms:
            p = adapter_root / arm["id"] / "train_stats.json"
            if p.exists():
                train_stats.append(json.loads(p.read_text(encoding="utf-8")))

    # ── infer ───────────────────────────────────────────────────────────────
    from .run_recomb import arm_model_cfg

    pred_paths = {x["id"]: run_dir / "predictions" / f"{x['id']}.jsonl" for x in arms}
    if "infer" in stages:
        _hr("Sinh báo cáo")
        from .runners import build
        from .runners.base import RunnerUnusable

        for arm in arms:
            # CẢ HAI candidate đi `gemma3_recomb` — candidate A đơn giản là nó
            # với `branch = None`. Cho A đi `gemma3_adapter` cho "gọn" là để hai
            # cột đi qua hai đường sinh khác nhau: tương đương về toán, KHÔNG
            # giống nhau từng phép tính.
            mcfg = arm_model_cfg(cfg, arm, adapter_root)
            print(f"\n── {names[arm['id']]} ({mcfg['runner']}) ──", flush=True)
            t0 = time.time()
            try:
                build(cfg, mcfg, tracker).run(test_df, pred_paths[arm["id"]],
                                              log_every=tracker.log_every)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": True, "minutes": round((time.time() - t0) / 60, 1)}
            except RunnerUnusable as e:
                # k = 2: mất một candidate là mất cả phép so sánh. Dừng ngay.
                print(f"\n✗ {arm['id']} không sinh được: {e}", flush=True)
                manifest["arm_status"].setdefault(arm["id"], {})["infer"] = {
                    "ok": False, "unusable": str(e)}
                tracker.alert("§4 dừng: một candidate không sinh được",
                              f"{arm['id']}: {e}", "ERROR")
                tracker.finish(1)
                return 6

    # ── score ───────────────────────────────────────────────────────────────
    from .metrics import metric_columns, score_model
    from .run_recomb import _read_preds

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

    # ── căn khối ────────────────────────────────────────────────────────────
    # Hiệu số ghép cặp là TOÀN BỘ phép phân tích ở k = 2, nên lệch một hàng là
    # hỏng mọi con số — và không có exception nào, chỉ có một bảng trông bình
    # thường. Căn MỘT LẦN ở đây, trước cả stats lẫn report.
    ab_path = mdir / "ab_tests.csv"
    ab = pd.DataFrame()
    if "stats" in stages or "report" in stages:
        if len(scores) != 2 or arm_a not in scores or arm_b not in scores:
            print(f"\n✗ cần ĐÚNG hai candidate có điểm (A={arm_a}, B={arm_b}); "
                  f"đang có {sorted(scores)}. Phép so sánh ghép cặp không định "
                  f"nghĩa được với một cột.", flush=True)
            tracker.alert("§4 hỏng", f"run={run_id}: {sorted(scores)}", "ERROR")
            tracker.finish(1)
            return 1
        from .stats_ab import align_blocks

        # Thứ tự cột = thứ tự KHAI BÁO, không phải thứ tự chạy: "candidate B
        # chạy trước" là một quyết định vận hành và nó không được rò vào cách
        # trình bày kết quả.
        scores = {x["id"]: scores[x["id"]] for x in arms}
        scores, align_info = align_blocks(scores, key_col="key",
                                          order_hint=test_df["key"].tolist())
        manifest["n_eval"] = align_info["n_common"]
        manifest["align"] = align_info
        if any(v for v in align_info["n_dropped"].values()):
            print("⚙ căn khối: " + ", ".join(
                f"{k}:{v}" for k, v in align_info["n_per_arm"].items())
                + f" -> {align_info['n_common']} ảnh chung", flush=True)

    # ── stats ───────────────────────────────────────────────────────────────
    if "stats" in stages:
        _hr("Bootstrap ghép cặp BCa + Permutation (k = 2, KHÔNG Friedman)")
        from .stats_ab import compare_all, resolution_warning, summary_line

        n_common = align_info["n_common"]
        cols = metric_columns(scores[arm_a])
        scfg = dict(cfg["stats"])
        print(f"  {len(cols)} metric × 2 kiểm định trên {n_common} ảnh ghép cặp "
              f"(n_perm={scfg.get('n_perm')}, n_boot={scfg.get('n_boot')})",
              flush=True)
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
        line = summary_line(ab, primary, names.get(arm_a, arm_a),
                            names.get(arm_b, arm_b))
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
        # nằm trên đĩa. Một lỗi ở đây không được phép là lý do vứt cả lần chạy.
        report_err = None
        try:
            from . import report_recomb

            report_recomb.write_all(cfg, run_dir, scores, names, ab,
                                    train_stats, manifest)
        except Exception:
            import traceback

            report_err = traceback.format_exc()
            print(f"\n✗ stage `report` hỏng:\n{report_err}", flush=True)
            print("→ KẾT QUẢ VẪN CÒN NGUYÊN trên đĩa:\n"
                  f"    adapters      : {adapter_root}\n"
                  f"    predictions   : {run_dir / 'predictions'}\n"
                  f"    điểm mỗi ảnh  : {mdir}/per_sample_*.csv\n"
                  f"    kiểm định     : {ab_path}\n"
                  "  Sửa rồi chạy lại MỘT MÌNH stage báo cáo — không train lại:\n"
                  "    python -m src.run_section4 --stages report", flush=True)
            manifest["report_error"] = report_err.splitlines()[-1]
            tracker.alert("§4: báo cáo hỏng (kết quả vẫn còn)",
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
    if "push" in stages:
        _hr("Đẩy lên Hugging Face")
        # CHỈ MỘT đích: `output.hub_repo` (repo dataset của kết quả). §4 không
        # sinh ra checkpoint pretrain nào — nhánh của nó đến từ §3, và §3 tự đẩy
        # checkpoint ấy lên repo model của nó. Đẩy lại từ đây là dựng một bản sao
        # thứ hai của cùng trọng số dưới một cái tên khác, và ngày hai bản lệch
        # nhau thì không có gì nói bản nào sinh ra bảng này.
        #
        # (`push_results` có ignore_patterns chặn *.safetensors, nên trọng số
        #  đẩy vào repo kết quả sẽ bị bỏ qua ÂM THẦM.)
        if cfg["output"].get("push", True):
            from .push import push_results

            try:
                url = push_results(cfg, run_dir)
                if url:
                    manifest.setdefault("hub", {})["results"] = url
                    tracker.summary({"run/hub_url": url})
            except Exception as e:
                push_err = str(e)
                print(f"\n✗ đẩy kết quả hỏng: {e}\n"
                      f"→ Kết quả vẫn nằm nguyên ở {run_dir}. Đẩy lại RIÊNG "
                      f"bước này:\n    python -m src.run_section4 --stages push",
                      flush=True)
                manifest["push_error"] = push_err

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    mins = (time.time() - t_start) / 60
    tracker.alert("§4 chạy xong",
                  f"run={run_id} · {manifest.get('n_eval')} ảnh · {mins:.0f} phút\n"
                  f"{manifest.get('headline', '')}")
    tracker.finish(1 if push_err else 0)
    print(f"\n✓ HOÀN TẤT sau {mins:.1f} phút", flush=True)
    print(f"  báo cáo : {run_dir / 'REPORT.md'}", flush=True)
    if tracker.url:
        print(f"  W&B     : {tracker.url}", flush=True)
    return 8 if push_err else 0


if __name__ == "__main__":
    sys.exit(main())

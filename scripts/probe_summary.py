#!/usr/bin/env python3
"""Gộp bản ghi probe trên Hub thành một bản đọc được bằng mắt (CPU, không torch).

`src/probe.py` ghi MỖI ẢNH một file JSON lên dataset repo `<hub_repo>-probe`:
bản đồ không gian 16×16, vector 14 bệnh + 4 aspect + 5 nhóm token toàn cục,
top-10 token, gate router. Một bản ghi lẻ không nói lên điều gì — nhiễu của một
ảnh lớn hơn hiệu ứng cần đo. Câu hỏi thật sự chỉ trả lời được sau khi gộp cả
tập:

    12 expert kia có THẬT SỰ phân hoá không, hay top-k chỉ là k bản sao mờ?

Script này chạy trên CPU máy cá nhân (không cần GPU, không import torch), kéo
records/*.json về, gộp lại và xuất ra:

    reports/probe/index.html        bảng nhiệt + bản đồ 16×16 + top token,
                                    một file tự chứa, mở bằng trình duyệt
    reports/probe/SUMMARY.md        bản đọc thẳng trên GitHub
    reports/probe/summary.json      số liệu máy đọc (để script khác dùng lại)
    reports/probe/*.csv             ma trận thô: route / usage / disease /
                                    aspect / từng nhóm token / similarity /
                                    top token / spatial

Chạy:

    python scripts/probe_summary.py                      # tự dò repo từ config
    python scripts/probe_summary.py --repo user/abc-probe
    python scripts/probe_summary.py --local-dir outputs/<tag>/probe_records
    python scripts/probe_summary.py --limit 200          # thử nhanh

Chạy lại thì GHI ĐÈ đúng thư mục đó — không tạo bản mới theo timestamp. Lý do:
đây là báo cáo về trạng thái hiện tại của checkpoint, không phải nhật ký; muốn
so hai thời điểm thì dùng `git diff` trên chính thư mục này, và mỗi bản ghi đã
có sẵn `train_samples` / `swa_n` để biết nó chụp lúc nào.

═══ HAI LOẠI BẢN GHI, KHÔNG ĐƯỢC TRỘN ═══

`src/probe.py` sinh ra hai loại, tự khai bằng `probe_mode`:

  * **`full`** — ảnh đó đã bật ĐỦ E expert, mỗi expert một forward riêng,
    trọng số 1.0, KHÔNG phụ thuộc router có chọn nó hay không. Đây là hồ sơ
    **vô điều kiện**.
  * **`topk`** — chỉ bật K expert mà router đã chọn. Hồ sơ **có điều kiện**.

Trung bình trên bản ghi `topk` trả lời câu "expert này làm gì TRÊN NHỮNG ẢNH
router gọi nó" — một đại lượng trộn hai thứ: expert học được gì, và router gọi
nó lúc nào. Hai expert có hồ sơ khác nhau ở đó có thể chỉ vì chúng được gọi
trên hai loại ảnh khác nhau, chứ bản thân hai hàm có khi giống hệt. Tệ hơn:
expert router chưa bao giờ chọn thì không có hàng nào cả, đúng cái expert đáng
ngờ nhất lại là cái duy nhất không đo được.

Nên MỌI bảng chuyên môn ở đây (bệnh, aspect, 5 nhóm token, cosine, bản đồ,
top token) dựng trên bản ghi `full` — cùng một tập ảnh cho cả E expert, so
trực tiếp được. Bản ghi `topk` chỉ đóng góp vào phần thống kê ROUTER (tần suất
chọn, gate, đồng xuất hiện), là chỗ chúng đúng nghĩa.

Không có bản ghi `full` nào (dữ liệu cũ, hoặc `run.probe_full_n: 0`) thì
script rơi về bản ghi `topk` và ĐÓNG DẤU cảnh báo lên đầu mọi bảng — số vẫn
hiện ra nhưng người đọc phải biết mình đang đọc một đại lượng có điều kiện.

═══ Vì sao output KHÔNG chứa một chữ nào của báo cáo ═══

Bản ghi probe có cả `report.reference` (báo cáo thật của MIMIC-CXR) lẫn
`report.generated`. MIMIC-CXR là dữ liệu PhysioNet có kiểm soát truy cập, và
`key_id` chính là đường dẫn `p10/p10000032/s50414267/...` — tức là định danh
nghiên cứu. Thư mục output này được thiết kế để ĐẨY LÊN GITHUB, nên script chỉ
xuất ra số liệu đã gộp trên toàn tập: không text báo cáo, không key_id, không
đếm theo từng ảnh. Cần xem báo cáo từng ảnh thì đọc thẳng bản ghi trên Hub
(repo private) chứ đừng đưa qua đây.

Top token thì được giữ, vì nó là token của UNEMBEDDING model — thuộc về model
chứ không thuộc về bệnh nhân nào.

═══ Bốn con số cần đọc trước ═══

**route_share** — trung bình của `softmax(route_logit)` trên n mẫu, cho ĐỦ E
expert. Đây là "router tin expert nào bao nhiêu", tính TRƯỚC khi cắt top-k, nên
expert không bao giờ lọt top-k vẫn có số. Chia đều tuyệt đối = 1/E. Đọc kèm
`effective_experts` = exp(H) — số expert hiệu dụng: 12 expert mà số này ra 3.2
nghĩa là router thực chất chỉ đang dùng khoảng ba cái.

**clone_index** — cosine trung bình giữa hồ sơ z-score của từng cặp expert,
trên toàn bộ cột (14 bệnh + 4 aspect + 5 nhóm token). z-score đã trừ trung bình
và chia độ lệch theo từng cột, nên (a) mặt bằng chung của cột bị loại — mọi
expert đều đẩy logit của mọi tên bệnh lên vì chúng cùng nằm trong miền X-quang,
cosine thô luôn ~0.9 và không phân biệt được gì; (b) nhóm token ngữ pháp không
áp đảo nhóm bệnh chỉ vì logit của `the` lớn hơn logit của `Atelectasis` cả bậc.
~1.0 = các expert là bản sao; ~0 = mỗi expert kéo về một hướng riêng; âm = phân
hoá đối lập nhau.

**balance** — entropy chuẩn hoá của tần suất router CHỌN từng expert (H/log E).
1.0 = router chia đều; thấp = vài expert ăn hết lượt chọn và phần còn lại là
tham số chết. Khác `route_share` ở chỗ nó đếm lượt lọt top-k (nhị phân), còn
`route_share` là trọng số liên tục.

**gate_max trung bình** — router có "dứt khoát" không. Gần 1/top_k nghĩa là
router chia đều trọng số cho k expert được chọn (thực chất là trung bình cộng,
không phải định tuyến).

Bốn số này KHÔNG thay cho metrics; chúng trả lời câu "MoE có hoạt động như MoE
không", còn BLEU/BERTScore vẫn nằm ở `src/report.py`.

═══ Expert không có hồ sơ được ĐỂ RIÊNG, không gộp ═══

Với bản ghi `full` thì mọi expert đều có hồ sơ, nên mục này thường rỗng. Nó chỉ
xuất hiện khi phải rơi về bản ghi `topk`: expert router chưa bao giờ chọn thì
không có vector nào. Expert như vậy KHÔNG được đưa vào bảng tổng hợp nào cả, mà
liệt kê riêng ở mục "Expert không tham gia" kèm lý do.

Lý do không phải là gọn mắt. Cho một expert vắng mặt vào với tư cách một hàng
0 thì:

  * trung bình cột bị kéo về 0 -> z-score của MỌI expert khác dịch lên, ai
    cũng trông "chuyên môn" hơn thực tế;
  * std cột phình ra -> mọi z co lại, khác biệt thật bị nén phẳng;
  * vector 0 có cosine 0 với tất cả -> `clone_index` bị pha loãng về 0, đọc ra
    thành "các expert phân hoá rõ" trong khi thực ra chỉ là thiếu dữ liệu.

Tức là một hàng rỗng làm sai đúng con số mà bảng sinh ra để trả lời. Trong code
điều này được thi hành bằng NaN (`mean_by`) chứ không phải 0: NaN không cộng
vào đâu được, nên mọi chỗ dùng nó buộc phải xử lý tường minh.

NGOẠI LỆ có chủ ý: `balance`, `usage_cv` và `route_share` vẫn tính trên ĐỦ E
expert. Chúng đo router chia việc thế nào, mà "expert này chưa từng được chọn"
chính là dữ kiện quan trọng nhất của câu đó — loại nó ra là tự xoá bằng chứng.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "reports", "probe")
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "default.yaml")
RECORD_DIR = "records"          # khớp src/probe.py::RECORD_DIR

# Tên hiển thị dự phòng cho các nhóm token, dùng khi bản ghi không kèm
# `field_titles` (bản ghi sinh bởi phiên bản probe.py cũ hơn).
FIELD_TITLE_FALLBACK = {
    "anatomy": "Giải phẫu & vị trí",
    "degree": "Mức độ & độ chắc chắn",
    "temporal": "Ngữ cảnh thời gian & so sánh",
    "grammar": "Ngữ pháp & cấu trúc báo cáo",
    "device": "Thiết bị hỗ trợ",
}

# Mỗi nhóm đo một trục khác nhau, nên câu "đọc bảng này thế nào" cũng khác.
# Dùng chung một câu cho cả năm nhóm thì người đọc lướt qua bốn cái sau.
FIELD_NOTE = {
    "anatomy": ("Expert kéo về VÙNG nào của lồng ngực. Đọc kèm bản đồ 16×16 ở "
                "cuối trang: bảng này là tên vùng trong CHỮ, bản đồ là vùng "
                "trong ẢNH. Trùng nhau thì kết luận chắc; lệch nhau thì expert "
                "đang gọi tên một vùng mà nó không nhìn tới."),
    "degree": ("Expert đẩy về phía khẳng định hay dè dặt, nặng hay nhẹ. Một "
               "expert nổi ở <code>no</code> / <code>without</code> / "
               "<code>unremarkable</code> là expert chuyên viết phim BÌNH "
               "THƯỜNG — đó là một chuyên môn thật, không phải expert yếu."),
    "temporal": ("Phần so với phim cũ: <code>stable</code>, <code>new</code>, "
                 "<code>compared</code>, <code>interval</code>. Trục này không "
                 "nằm trong bất kỳ nhãn nào của 14 finding, nên trước đây một "
                 "expert chuyên nó là hoàn toàn vô hình."),
    "grammar": ("Expert lo KHUNG và văn phong chứ không lo nội dung. Nổi ở "
                "<code>Findings</code> / <code>Impression</code> = giữ cấu "
                "trúc hai mục; nổi ở <code>is</code> / <code>are</code> / "
                "<code>the</code> = lo cú pháp câu."),
    "device": ("Ống thông, dây, máy tạo nhịp. CheXbert gộp tất cả vào đúng một "
               "nhãn <code>Support Devices</code>, nên đây là chỗ duy nhất "
               "phân biệt được chúng với nhau."),
}
FIELD_NOTE_DEFAULT = ("Ngoài danh sách bệnh: một expert có thể không chuyên "
                      "bệnh nào mà chuyên đúng phần này của bài báo cáo.")


# ═══════════════════════════════ nguồn dữ liệu ═══════════════════════════════
def resolve_repo(cli_repo, config_path):
    """Tên dataset repo probe: cờ dòng lệnh > biến môi trường > config.

    Lặp lại đúng quy tắc của `src/config.py` (probe_repo -> hub_repo + "-probe"
    -> tên mặc định theo tag) nhưng KHÔNG import src.config: file đó kéo theo cả
    torch/transformers, mà script này phải chạy được trên một máy chỉ có numpy.
    """
    if cli_repo:
        return cli_repo
    env = os.environ.get("RGEXP_PROBE_REPO")
    if env:
        return env
    try:
        import yaml
    except ImportError:
        return None
    if not os.path.isfile(config_path):
        return None
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    run = cfg.get("run") or {}
    if run.get("probe_repo"):
        return run["probe_repo"]
    hub = run.get("hub_repo")
    if not hub:
        # Khối `sweep:` cũ (model + variant + n_block) đổi tên thành `arch:`
        # và chỉ còn `model` — đọc cả hai để bản ghi của các đợt trước vẫn
        # suy ra được tên repo.
        sw = cfg.get("arch") or cfg.get("sweep") or {}
        if sw.get("variant"):
            tag = f"{sw.get('model')}-{sw['variant']}-n{sw.get('n_block')}"
        else:
            tag = f"{sw.get('model')}-cnx256"
        hub = f"RGExp123/mimic-cxr-{tag}"
    return f"{hub}-probe"


def fetch_records_dir(repo, token, revision, quiet=False):
    """Kéo records/ về cache HF rồi trả đường dẫn thư mục chứa các file JSON.

    `snapshot_download` đã lo phần "chạy lại chỉ tải phần mới": file nào hash
    không đổi thì nó lấy từ cache. Vài nghìn file nhỏ nên bật tải song song.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("✗ Thiếu huggingface_hub. Cài:  pip install huggingface_hub\n"
                 "  (hoặc chạy với --local-dir nếu đã có bản ghi trên đĩa)")
    if not quiet:
        print(f"→ tải {repo}/{RECORD_DIR} về cache HF …", flush=True)
    root = snapshot_download(
        repo_id=repo, repo_type="dataset", revision=revision, token=token,
        allow_patterns=[f"{RECORD_DIR}/*.json"], max_workers=8)
    d = os.path.join(root, RECORD_DIR)
    return d if os.path.isdir(d) else root


def load_records(directory, limit=None, quiet=False):
    """Đọc mọi *.json trong thư mục. File hỏng thì bỏ qua và đếm lại.

    Bỏ qua chứ không nổ: CommitScheduler có thể commit đúng lúc một file đang
    được ghi, và mất một bản ghi trong vài nghìn không đổi kết luận nào.
    """
    paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    if limit:
        paths = paths[:int(limit)]
    recs, bad = [], 0
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                recs.append(json.load(fh))
        except Exception:
            bad += 1
    if not quiet:
        print(f"→ đọc {len(recs)} bản ghi"
              + (f" ({bad} file hỏng, đã bỏ qua)" if bad else ""), flush=True)
    return recs, bad


def record_mode(r):
    """"full" nếu bản ghi đã bật đủ E expert, ngược lại "topk".

    Bản ghi v1 không có trường `probe_mode`. Suy ngược từ số expert có trong
    bản ghi thay vì mặc định "topk": một lần chạy cũ với top_k = n_expert cũng
    là hồ sơ vô điều kiện, chỉ là hồi đó chưa có tên gọi cho nó.
    """
    m = r.get("probe_mode")
    if m in ("full", "topk"):
        return m
    E = int(r.get("n_expert") or 0)
    return "full" if E and len(r.get("experts") or []) >= E else "topk"


# ═══════════════════════════════ phần gộp số ═════════════════════════════════
def aggregate(recs):
    """Gộp danh sách bản ghi thành các ma trận theo expert.

    Hai tầng thống kê, mẫu số KHÁC NHAU và cố ý không trộn:

      * **router** (`usage`, `route_share`, `gate`, đồng xuất hiện) — trên MỌI
        bản ghi. Câu hỏi ở đây là router chia việc thế nào, nên bản ghi `topk`
        cũng là dữ liệu hợp lệ và mẫu số phải là toàn tập.
      * **hồ sơ chuyên môn** (bệnh, aspect, 5 nhóm token, bản đồ, top token) —
        chỉ trên bản ghi `full`, nơi mọi expert đã được bật trên cùng một tập
        ảnh. Đây là điều kiện để so hai expert với nhau mà không bị lẫn với
        việc router gọi chúng trên hai loại ảnh khác nhau.

    Không có bản ghi `full` nào thì rơi về `topk` và ghi cờ `profile_source`
    để mọi chỗ hiển thị đóng dấu cảnh báo — số vẫn ra, nhưng nó là đại lượng
    CÓ ĐIỀU KIỆN và người đọc phải biết.
    """
    if not recs:
        sys.exit("✗ Không có bản ghi nào để gộp.")

    E = max(int(r.get("n_expert") or 0) for r in recs)
    K = max(int(r.get("top_k") or 0) for r in recs)
    if E < 1:
        # Không có n_expert thì suy ngược từ chỉ số expert lớn nhất từng thấy —
        # bản ghi cũ hơn schema hiện tại vẫn dùng được.
        E = 1 + max((int(it.get("expert", -1))
                     for r in recs for it in (r.get("experts") or [])), default=0)
    K = K or max((len((r.get("router") or {}).get("chosen") or []) for r in recs),
                 default=1)
    dis_names = next((r["disease_labels"] for r in recs if r.get("disease_labels")), [])
    asp_names = next((r["aspect_labels"] for r in recs if r.get("aspect_labels")), [])
    fld_names = next((r["field_labels"] for r in recs if r.get("field_labels")), {})
    fld_titles = next((r["field_titles"] for r in recs if r.get("field_titles")), {})
    fld_titles = {g: fld_titles.get(g) or FIELD_TITLE_FALLBACK.get(g, g)
                  for g in fld_names}
    fkeys = list(fld_names)
    nD, nA = len(dis_names), len(asp_names)
    grid = next((tuple(r["grid"]) for r in recs if r.get("grid")), None)
    nS = grid[0] * grid[1] if grid else 0

    modes = [record_mode(r) for r in recs]
    n_full = sum(1 for m in modes if m == "full")
    src = "full" if n_full else "topk"

    # ── tầng router: mọi bản ghi ────────────────────────────────────────────
    n_chosen = np.zeros(E)
    n_top1 = np.zeros(E)
    # Vector gate của router, cộng dồn trên MỌI bản ghi. Ô của expert không
    # lọt top-k đã là 0 sẵn, nên cùng một tổng này chia cho hai mẫu số khác
    # nhau ra hai đại lượng khác nhau: chia cho `n_chosen` = "gate trung bình
    # KHI được chọn", chia cho `n` = "đóng góp trung bình trên toàn tập".
    gate_all = np.zeros(E)
    prob_sum = np.zeros(E)                 # softmax dày, trước khi cắt top-k
    n_prob = 0
    cooc = np.zeros((E, E))                # đồng xuất hiện trong cùng top-k
    ent, gmax = [], []

    # ── tầng hồ sơ: chỉ bản ghi `full` (hoặc `topk` nếu không có full nào) ──
    dis_sum = np.zeros((E, nD))
    asp_sum = np.zeros((E, nA))
    fld_sum = {g: np.zeros((E, len(fld_names[g]))) for g in fkeys}
    n_fld = {g: np.zeros(E) for g in fkeys}
    dnorm_sum = np.zeros(E)
    sp_sum = np.zeros((E, nS)) if nS else None
    spmax_sum = np.zeros(E)
    spl2_sum = np.zeros(E)
    # Mẫu số RIÊNG cho từng đại lượng, không dùng chung một biến đếm: một
    # expert có thể xuất hiện trong 900 bản ghi mà chỉ 850 lần có `spatial`
    # (grid không vuông, hoặc bản ghi cũ thiếu trường). Chia tất cả cho cùng
    # một số là lặng lẽ kéo trung bình về 0 theo tỉ lệ phần thiếu.
    n_prof = np.zeros(E)                   # số lần expert được BẬT và đo hồ sơ
    n_dis = np.zeros(E)
    n_asp = np.zeros(E)
    n_sp = np.zeros(E)
    tok = [defaultdict(lambda: [0, 0.0, ""]) for _ in range(E)]

    n_lab_mismatch = n_topk_used = 0
    meta = defaultdict(set)
    samples = []

    for r, mode in zip(recs, modes):
        if r.get("disease_labels") and r["disease_labels"] != dis_names:
            n_lab_mismatch += 1
            continue
        # `run_tag` là tên mới của `sweep_cell` (không còn sweep). Đọc cả
        # hai để trộn được bản ghi của đợt cũ với đợt mới.
        for key in ("model", "weights"):
            if r.get(key):
                meta[key].add(str(r[key]))
        _tag = r.get("run_tag") or r.get("sweep_cell")
        if _tag:
            meta["run_tag"].add(str(_tag))
        if r.get("train_samples") is not None:
            samples.append(int(r["train_samples"]))
        router = r.get("router") or {}
        if router.get("gate"):
            g = np.asarray(router["gate"], dtype=np.float64)
            k = min(len(g), E)
            gate_all[:k] += g[:k]
        if router.get("prob"):
            p = np.asarray(router["prob"], dtype=np.float64)
            k = min(len(p), E)
            prob_sum[:k] += p[:k]
            n_prob += 1
        if router.get("entropy") is not None:
            ent.append(float(router["entropy"]))
        if router.get("gate_max") is not None:
            gmax.append(float(router["gate_max"]))
        chosen = [int(e) for e in (router.get("chosen") or []) if 0 <= int(e) < E]
        for a in chosen:
            for b in chosen:
                cooc[a, b] += 1
        if chosen:
            n_top1[chosen[0]] += 1
        for e in chosen:
            n_chosen[e] += 1

        if mode != src:
            continue                       # bản ghi topk khi đã có full: bỏ hồ sơ
        if mode == "topk":
            n_topk_used += 1
        for it in r.get("experts") or []:
            e = int(it.get("expert", -1))
            if not 0 <= e < E:
                continue
            n_prof[e] += 1
            dnorm_sum[e] += float(it.get("delta_norm") or 0.0)
            d = it.get("disease")
            if d and len(d) == nD:
                dis_sum[e] += np.asarray(d, dtype=np.float64)
                n_dis[e] += 1
            a = it.get("aspect")
            if a and len(a) == nA:
                asp_sum[e] += np.asarray(a, dtype=np.float64)
                n_asp[e] += 1
            for g, vec in (it.get("fields") or {}).items():
                if g in fld_sum and vec and len(vec) == fld_sum[g].shape[1]:
                    fld_sum[g][e] += np.asarray(vec, dtype=np.float64)
                    n_fld[g][e] += 1
            sp = it.get("spatial")
            if sp_sum is not None and sp and len(sp) == nS:
                sp_sum[e] += np.asarray(sp, dtype=np.float64)
                spmax_sum[e] += float(it.get("spatial_max") or 0.0)
                spl2_sum[e] += float(it.get("spatial_l2") or 0.0)
                n_sp[e] += 1
            for t in it.get("top_tokens") or []:
                slot = tok[e][int(t.get("id", -1))]
                slot[0] += 1
                slot[1] += float(t.get("dlogit") or 0.0)
                slot[2] = str(t.get("token", ""))

    n = len(recs) - n_lab_mismatch
    n_profile_rec = n_full if src == "full" else n_topk_used
    profile = n_dis > 0          # có hồ sơ bệnh -> vào được mọi bảng tổng hợp
    out = {
        "n_records": n,
        "n_full": n_full,
        "n_topk": n - n_full,
        "n_profile_records": n_profile_rec,
        "profile_source": src,
        "n_label_mismatch": n_lab_mismatch,
        "n_expert": E, "top_k": K,
        "disease_labels": dis_names, "aspect_labels": asp_names,
        "field_labels": fld_names, "field_titles": fld_titles,
        "field_keys": fkeys,
        "grid": list(grid) if grid else None,
        "meta": {k: sorted(v) for k, v in meta.items()},
        "train_samples": (min(samples), max(samples)) if samples else None,
        "n_chosen": n_chosen, "n_top1": n_top1, "n_prof": n_prof,
        "n_disease": n_dis, "n_aspect": n_asp, "n_spatial": n_sp,
        "n_field": n_fld,
        "profile": profile,
        "usage": n_chosen / max(n, 1),
        "route_share": (prob_sum / n_prob) if n_prob else np.full(E, np.nan),
        "n_prob_records": n_prob,
        # Mẫu số là `n_chosen` (đếm trên MỌI bản ghi), KHÔNG phải `n_prof`: ở
        # chế độ full, expert bị BẬT cả trên những ảnh gate của nó = 0, chia
        # cho số lần bật sẽ kéo gate trung bình xuống một cách giả tạo.
        "gate_mean_when_chosen": mean_by(gate_all, n_chosen),
        "gate_mean_all": gate_all / max(n, 1),
        "disease": mean_by(dis_sum, n_dis),
        "aspect": mean_by(asp_sum, n_asp) if nA else np.zeros((E, 0)),
        "fields": {g: mean_by(fld_sum[g], n_fld[g]) for g in fkeys},
        "delta_norm": mean_by(dnorm_sum, n_prof),
        "spatial": mean_by(sp_sum, n_sp) if sp_sum is not None else None,
        "spatial_max": mean_by(spmax_sum, n_sp),
        "spatial_l2": mean_by(spl2_sum, n_sp),
        "cooccurrence": cooc,
        "route_entropy": float(np.mean(ent)) if ent else None,
        "gate_max": float(np.mean(gmax)) if gmax else None,
        "tokens": tok,
    }
    out["absent"] = absent_reasons(out)
    out.update(derive(out))
    return out


def mean_by(total, count):
    """Chia tổng cho ĐÚNG số lần đại lượng đó có mặt; thiếu hẳn thì NaN.

    NaN chứ không phải 0. Số 0 trông như một phép đo ("expert này không đẩy
    bệnh nào lên") và sẽ trôi thẳng vào trung bình cột, vào z-score, vào
    cosine — làm lệch mọi expert khác. NaN thì không cộng vào đâu được, buộc
    mọi chỗ dùng nó phải xử lý tường minh: expert đó bị LOẠI khỏi bảng và
    được liệt kê riêng.
    """
    cnt = np.asarray(count, dtype=np.float64)
    if total.ndim == 2:
        cnt = cnt[:, None]
    out = np.full(total.shape, np.nan, dtype=np.float64)
    return np.divide(total, cnt, out=out, where=cnt > 0)


def absent_reasons(a):
    """Expert nào không vào được bảng tổng hợp, và vì sao.

    Ở chế độ `full` mục này thường RỖNG — mọi expert đều được bật nên đều có
    hồ sơ. Nó chỉ có nội dung khi phải rơi về bản ghi `topk`, và khi đó hai lý
    do dưới đây khác hẳn nhau nên KHÔNG gộp làm một dòng:

      * `router_never_picked` — không một snapshot nào chọn expert này. Đây là
        phát hiện về ROUTER (tham số chết / cân bằng tải hỏng), không phải về
        expert. Nó cũng là lý do duy nhất mà `balance` phải giữ nguyên mẫu số
        E: bỏ expert này ra khỏi entropy là tự xoá đúng bằng chứng đó.
      * `no_profile` — có được chọn nhưng bản ghi không kèm vector bệnh (bản
        ghi cũ hơn schema, hoặc lỗi ghi). Đây là lỗ hổng DỮ LIỆU: expert vẫn
        chạy thật, chỉ là ta không đọc được nó.
    """
    out = []
    for e in range(a["n_expert"]):
        if a["n_disease"][e] > 0:
            continue
        out.append({
            "expert": e,
            "reason": "router_never_picked" if a["n_chosen"][e] == 0 else "no_profile",
            "n_chosen": int(a["n_chosen"][e]),
        })
    return out


def zscore(M):
    """Chuẩn hoá theo CỘT: mỗi nhãn được so giữa các expert với nhau.

    Cột chứ không phải hàng, vì mặt bằng logit của từng tên chênh nhau cả bậc
    (tên hiếm gặp trong vocab luôn có Δlogit bé hơn `the`). Chuẩn theo cột thì
    ô đỏ đọc được đúng một nghĩa: "expert này đẩy nhãn này mạnh HƠN các expert
    khác", đó mới là định nghĩa của chuyên môn hoá.

    Hàng NaN (expert không góp mặt trong bản ghi nào) bị loại khỏi mean/std rồi
    trả về NaN: để nó tham gia là tự thêm một điểm giả vào mặt bằng mà mọi
    expert khác đang được đo so với.
    """
    if M.size == 0:
        return M
    ok = ~np.isnan(M).any(axis=1)
    out = np.full(M.shape, np.nan)
    if not ok.any():
        return out
    sub = M[ok]
    mu = sub.mean(axis=0, keepdims=True)
    sd = sub.std(axis=0, keepdims=True)
    out[ok] = np.divide(sub - mu, sd, out=np.zeros_like(sub), where=sd > 1e-12)
    return out


def cosine_matrix(M, ok):
    """Cosine từng cặp hàng SAU KHI trừ hàng trung bình. Trả (ma trận, tb, max).

    Trừ mặt bằng chung là bắt buộc: mọi expert đều đẩy logit của mọi tên trong
    miền X-quang lên, nên cosine thô luôn ~0.9 và không phân biệt được gì. Cái
    đáng đo là phần LỆCH so với mặt bằng.
    """
    E = M.shape[0]
    sim = np.full((E, E), np.nan)
    if M.size == 0 or ok.sum() < 2:
        return sim, 0.0, 0.0
    sub = M[ok]
    C = sub - sub.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(C, axis=1, keepdims=True)
    Cn = np.divide(C, norm, out=np.zeros_like(C), where=norm > 1e-12)
    S = Cn @ Cn.T
    idx = np.nonzero(ok)[0]
    sim[np.ix_(idx, idx)] = S
    off = S[~np.eye(len(idx), dtype=bool)]
    return sim, float(off.mean()), float(off.max())


def entropy_stats(p, E):
    """(entropy chuẩn hoá H/log E, số expert hiệu dụng exp H) của một phân phối.

    `exp(H)` dễ đọc hơn entropy trần: 12 expert mà ra 3.2 nghĩa là router đang
    dùng khoảng ba cái, còn "H = 1.16 nat" thì phải quy đổi trong đầu.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size == 0 or E < 2:
        # NaN chứ không phải (1.0, E). Bản ghi cũ không có `route_prob` thì
        # vector rỗng, và trả về 1.0 ở đây sẽ in ra "router chia đều tuyệt
        # đối" — một kết luận bịa ra từ dữ liệu không tồn tại.
        return float("nan"), float("nan")
    p = p / p.sum()
    H = float(-(p * np.log(p)).sum())
    return H / math.log(E), float(math.exp(H))


def derive(a):
    """Các chỉ số tổng và ma trận dẫn xuất — CHỈ trên expert có hồ sơ.

    `profile` là mặt nạ những expert thật sự có vector bệnh. Mọi thứ ở đây
    (z-score, cosine, clone_index, chuyên biệt hoá) đều tính trong nhóm đó và
    trả NaN cho phần còn lại. Trộn một expert vắng mặt vào không phải là "thêm
    một hàng rỗng cho đủ": nó dịch trung bình cột, bóp std, kéo cosine của mọi
    cặp khác — tức là làm sai chính con số mà bảng này sinh ra để trả lời.

    NGOẠI LỆ có chủ ý: `balance`, `usage_cv` và `route_share` vẫn tính trên ĐỦ
    E expert. Chúng đo router chia việc thế nào, mà "expert này chưa từng được
    chọn" chính là dữ kiện quan trọng nhất của câu đó — bỏ nó ra là tự làm đẹp
    số liệu.
    """
    E = a["n_expert"]
    ok = a["profile"]
    D = a["disease"]
    Z = zscore(D)
    Zasp = zscore(a["aspect"])
    Zfld = {g: zscore(M) for g, M in a["fields"].items()}

    # ── ma trận TOÀN CỤC: 14 bệnh + 4 aspect + mọi nhóm token, đã z-score ───
    # Ghép z-score chứ không ghép giá trị thô. Δlogit của `the` lớn hơn của
    # `Atelectasis` cả bậc, nên ghép thô thì cosine toàn cục thực chất chỉ là
    # cosine của nhóm ngữ pháp, bốn nhóm kia thành nhiễu làm tròn. z-score đưa
    # mọi cột về cùng thang, mỗi cột góp đúng một phần bằng nhau.
    parts, cols = [], []
    if Z.size:
        parts.append(Z)
        cols += [("disease", n) for n in a["disease_labels"]]
    if Zasp.size:
        parts.append(Zasp)
        cols += [("aspect", n) for n in a["aspect_labels"]]
    for g in a["field_keys"]:
        if Zfld[g].size:
            parts.append(Zfld[g])
            cols += [(g, n) for n in a["field_labels"][g]]
    G = np.concatenate(parts, axis=1) if parts else np.zeros((E, 0))

    sim_d, ci_d, ci_d_max = cosine_matrix(D, ok)
    sim_g, ci_g, ci_g_max = cosine_matrix(G, ok & ~np.isnan(G).any(axis=1)
                                          if G.size else ok)
    sp = a["spatial"]
    ok_sp = a["n_spatial"] > 0
    sim_sp, ci_sp, _ = (cosine_matrix(sp, ok_sp) if sp is not None
                        else (np.full((E, E), np.nan), float("nan"), 0.0))

    # ── mỗi expert giống ai nhất, và nó nổi bật tới đâu ─────────────────────
    redundancy = np.full(E, np.nan)
    twin = [None] * E
    for e in range(E):
        row = np.array(sim_g[e], dtype=np.float64)
        row[e] = np.nan
        if np.isfinite(row).any():
            j = int(np.nanargmax(row))
            redundancy[e], twin[e] = float(row[j]), j
    # Độ chuyên biệt = độ lệch chuẩn của hàng z. Cột đã chuẩn hoá theo expert,
    # nên hàng phẳng (std ~ 0) = expert bám sát mặt bằng ở MỌI nhãn, tức là
    # không chuyên gì cả; hàng gồ ghề = nó kéo mạnh vài nhãn theo hướng riêng.
    spec_std = np.array([float(np.nanstd(G[e])) if G.size and np.isfinite(G[e]).any()
                         else np.nan for e in range(E)])
    spec_peak = np.array([float(np.nanmax(np.abs(G[e])))
                          if G.size and np.isfinite(G[e]).any() else np.nan
                          for e in range(E)])

    p = a["n_chosen"] / max(a["n_chosen"].sum(), 1)
    balance, eff_pick = entropy_stats(p, E)
    prob_balance, eff_prob = entropy_stats(a["route_share"], E)

    # Router có cho gate cao đúng những expert đổi biểu diễn mạnh không?
    # Tương quan âm hoặc ~0 nghĩa là trọng số router gán không liên quan gì
    # tới lượng công expert thật sự làm.
    gs, dn = a["route_share"], a["delta_norm"]
    m = np.isfinite(gs) & np.isfinite(dn)
    corr = (float(np.corrcoef(gs[m], dn[m])[0, 1])
            if m.sum() >= 3 and np.std(gs[m]) > 1e-12 and np.std(dn[m]) > 1e-12
            else float("nan"))

    return {
        "disease_z": Z,
        "aspect_z": Zasp,
        "field_z": Zfld,
        "global_z": G,
        "global_cols": cols,
        "similarity": sim_d,
        "similarity_global": sim_g,
        "similarity_spatial": sim_sp,
        "n_profiled": int(ok.sum()),
        "clone_index": ci_g,               # chỉ số HEADLINE: trên toàn bộ cột
        "clone_index_max": ci_g_max,
        "clone_index_disease": ci_d,       # giữ lại để so với báo cáo cũ
        "clone_index_disease_max": ci_d_max,
        "clone_index_spatial": ci_sp,
        "redundancy": redundancy,
        "twin": twin,
        "spec_std": spec_std,
        "spec_peak": spec_peak,
        "balance": balance,
        "effective_experts_picked": eff_pick,
        "prob_balance": prob_balance,
        "effective_experts": eff_prob,
        "usage_cv": float(p.std() / p.mean()) if p.mean() > 0 else 0.0,
        "dead_experts": [e for e in range(E) if a["n_chosen"][e] == 0],
        "gate_effect_corr": corr,
    }


def top_tokens(a, e, k=12):
    """Top token của một expert: xếp theo TỔNG Δlogit trên cả tập.

    Xếp theo tổng chứ không theo trung bình mỗi lần xuất hiện: một token chỉ
    lọt top-10 đúng một lần với Δlogit khổng lồ là nhiễu của một ảnh, còn token
    xuất hiện ở 800/1000 ảnh với Δlogit vừa phải mới là chuyên môn. Tổng phạt
    cái đầu và thưởng cái sau; số lần xuất hiện vẫn in kèm để tự kiểm.
    """
    n = max(a["n_prof"][e], 1)
    rows = [(txt, cnt, s / n, cnt / n)
            for cnt, s, txt in a["tokens"][e].values()]
    rows.sort(key=lambda r: -r[2])
    return rows[:k]


def top_labels(vec, names, k=3, floor=0.35):
    """Tên của k giá trị z lớn nhất, bỏ những cái quá gần mặt bằng chung.

    Vector toàn NaN = expert không có hồ sơ; trả rỗng để mọi chỗ hiển thị ra
    "—" thay vì bịa một chuyên môn từ dữ liệu không tồn tại.
    """
    vec = np.asarray(vec, dtype=np.float64)
    if vec.size == 0 or np.isnan(vec).all():
        return []
    idx = np.argsort(-np.nan_to_num(vec, nan=-np.inf))[:k]
    return [(names[i], float(vec[i])) for i in idx
            if not np.isnan(vec[i]) and vec[i] > floor]


def top_global(a, e, k=4, floor=0.35):
    """k nhãn nổi nhất của expert e trên TOÀN BỘ cột, kèm tên nhóm.

    Trả [(nhóm, nhãn, z)]. Đây là câu trả lời cho "expert này là gì" khi không
    giới hạn trong 14 tên bệnh — một expert chuyên phần ngữ pháp sẽ hiện ra ở
    đây thay vì bị đọc nhầm thành "không nổi trội".
    """
    G, cols = a["global_z"], a["global_cols"]
    if not G.size or not np.isfinite(G[e]).any():
        return []
    row = np.nan_to_num(G[e], nan=-np.inf)
    return [(cols[i][0], cols[i][1], float(G[e][i]))
            for i in np.argsort(-row)[:k]
            if np.isfinite(G[e][i]) and G[e][i] > floor]


# ═════════════════════════════════ diễn giải ═════════════════════════════════
def source_caveat(a):
    """Một câu cảnh báo nếu bảng chuyên môn đang dựng trên hồ sơ CÓ ĐIỀU KIỆN."""
    if a["profile_source"] == "full":
        return ""
    return (f"⚠ Không có bản ghi `full` nào: mọi bảng chuyên môn dưới đây dựng "
            f"trên {a['n_profile_records']:,} bản ghi `topk`, tức là trung bình "
            f"CÓ ĐIỀU KIỆN — chỉ trên những ảnh router đã chọn expert đó. Hai "
            f"expert khác nhau ở đây có thể chỉ vì chúng được gọi trên hai loại "
            f"ảnh khác nhau, và expert chưa từng được chọn thì không có hàng "
            f"nào. Đặt `run.probe_full_n` (mặc định 300) rồi chạy lại "
            f"`mode=probe` để có hồ sơ vô điều kiện. ")


def verdict(a):
    """Một đoạn tiếng Việt nói thẳng bảng số kia có nghĩa gì.

    Ngưỡng ở đây là quy ước ĐỌC BẢNG, không phải kiểm định thống kê: chúng chỉ
    chia trục cosine thành ba khoảng để câu kết luận không phải viết lại mỗi
    lần. Ranh giới 0.30 / 0.70 chọn theo kinh nghiệm đọc ma trận tương đồng —
    thấy số sát ngưỡng thì đọc ma trận, đừng đọc câu chữ.
    """
    ci, bal, E, K = a["clone_index"], a["balance"], a["n_expert"], a["top_k"]
    P = a["n_profiled"]
    head = source_caveat(a)
    if P < 2:
        return "không đủ dữ liệu", head + (
            f"Chỉ {P}/{E} expert có mặt trong dữ liệu probe — không so được cặp "
            f"nào, nên clone_index không có nghĩa. Xem mục 'Expert không tham "
            f"gia' để biết vì sao, và chạy lại probe cho đủ tập trước khi đọc "
            f"bảng chuyên môn.")
    if ci > 0.70:
        lvl, msg = "cao", (
            f"Hồ sơ của {P} expert gần như trùng nhau (cosine lệch trung bình "
            f"{ci:.2f} trên {len(a['global_cols'])} nhãn). Top-{K} đang lấy về "
            f"{K} bản sao mờ của cùng một hàm — phần lớn tham số expert là "
            f"thừa. Đáng thử: giảm n_expert, nâng w_aux (ép router trải tải "
            f"rộng hơn nên mỗi expert thấy dữ liệu khác nhau hơn), hoặc giữ "
            f"σ Gumbel cao lâu hơn.")
    elif ci > 0.30:
        lvl, msg = "vừa", (
            f"Các expert phân hoá MỘT PHẦN (cosine lệch trung bình {ci:.2f} "
            f"trên {len(a['global_cols'])} nhãn): có hướng riêng nhưng vẫn "
            f"chồng lấn nhiều. Đọc ma trận z-score bên dưới để xem cặp nào "
            f"đang trùng.")
    else:
        lvl, msg = "thấp", (
            f"Các expert phân hoá RÕ (cosine lệch trung bình {ci:.2f} trên "
            f"{len(a['global_cols'])} nhãn): mỗi expert kéo hồ sơ về một hướng "
            f"khác nhau. Đây là dấu hiệu MoE đang thật sự chia việc.")

    cid = a["clone_index_disease"]
    if abs(cid - ci) > 0.15:
        msg += (f" Riêng trên 14 tên bệnh thì clone_index = {cid:.2f} — "
                + ("cao hơn hẳn: các expert trùng nhau ở phần BỆNH nhưng tách "
                   "ra ở các nhóm token khác (ngữ pháp, thời gian, giải phẫu), "
                   "tức là chúng chia việc theo VĂN PHONG chứ không theo bệnh."
                   if cid > ci else
                   "thấp hơn hẳn: chúng tách nhau ở phần bệnh nhưng dùng chung "
                   "một khuôn văn phong."))

    eff = a["effective_experts"]
    if not isnan(eff):
        msg += (f" Router phân bổ trọng số cho {eff:.1f}/{E} expert hiệu dụng "
                f"(exp của entropy `route_share`).")
    else:
        msg += (" Bản ghi không kèm `route_prob` (probe bản cũ) nên không đọc "
                "được tỉ lệ đóng góp của đủ E expert — chỉ có lượt lọt top-k.")
    if bal < 0.85:
        msg += (f" Nhưng lượt CHỌN thì lệch (balance {bal:.2f}, đều tuyệt đối "
                f"= 1.00): một phần expert hiếm khi lọt top-k, xem cột "
                f"'lượt chọn'.")
    else:
        msg += f" Lượt chọn khá đều (balance {bal:.2f})."
    if a["gate_max"] is not None and K > 0 and a["gate_max"] < 1.4 / K:
        msg += (f" Gate max trung bình {a['gate_max']:.3f} ≈ 1/top_k "
                f"({1/K:.3f}): router gần như chia đều trọng số cho {K} expert "
                f"được chọn, tức là đang trung bình cộng chứ chưa định tuyến.")
    corr = a["gate_effect_corr"]
    if not isnan(corr) and corr < 0.2:
        msg += (f" Tương quan giữa `route_share` và ‖Δh‖ chỉ {corr:+.2f}: "
                f"trọng số router gán gần như không liên quan tới lượng công "
                f"expert thật sự làm lên biểu diễn.")
    if a["dead_experts"]:
        msg += (" Expert chưa từng lọt top-k lần nào: "
                + ", ".join(f"e{e}" for e in a["dead_experts"]) + ".")
    if a["absent"]:
        never = [x["expert"] for x in a["absent"]
                 if x["reason"] == "router_never_picked"]
        nodata = [x["expert"] for x in a["absent"] if x["reason"] == "no_profile"]
        parts = []
        if never:
            parts.append("chưa từng được router chọn trong bất kỳ snapshot nào: "
                         + ", ".join(f"e{e}" for e in never))
        if nodata:
            parts.append("được chọn nhưng bản ghi không có vector bệnh: "
                         + ", ".join(f"e{e}" for e in nodata))
        msg = (f"⚠ {E - P}/{E} expert bị LOẠI khỏi mọi bảng dưới ("
               + "; ".join(parts) + f"). Các số trên chỉ nói về {P} expert còn "
               f"lại — xem mục 'Expert không tham gia'. ") + msg
    return lvl, head + msg


# ═══════════════════════════════ xuất CSV/JSON ═══════════════════════════════
def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def isnan(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (
        isinstance(v, np.floating) and math.isnan(float(v)))


def num(v, nd=5):
    """Số cho CSV; NaN -> ô rỗng chứ không phải 0 (0 là một phép đo, rỗng thì không)."""
    return "" if isnan(v) else round(float(v), nd)


def jnum(v, nd=5):
    """Số cho JSON; NaN -> null (JSON không có NaN hợp lệ, và 0 thì sai nghĩa)."""
    return None if isnan(v) else round(float(v), nd)


def dump_csvs(a, out):
    E = a["n_expert"]
    live = [e for e in range(E) if a["profile"][e]]
    dn, an = a["disease_labels"], a["aspect_labels"]

    # Bảng router giữ ĐỦ E hàng, có cột `active`: đây là bảng về router, và
    # hàng n_chosen = 0 chính là thông tin cần thấy.
    write_csv(os.path.join(out, "expert_router.csv"),
              ["expert", "active", "route_share", "usage", "n_chosen",
               "top1_share", "n_top1", "gate_mean_when_chosen", "gate_mean_all",
               "n_profile_samples", "delta_norm", "spatial_max", "spatial_l2",
               "redundancy", "twin", "spec_std", "spec_peak"],
              [[e, int(bool(a["profile"][e])), num(a["route_share"][e]),
                num(a["usage"][e]), int(a["n_chosen"][e]),
                num(a["n_top1"][e] / max(a["n_records"], 1)),
                int(a["n_top1"][e]),
                num(a["gate_mean_when_chosen"][e]), num(a["gate_mean_all"][e]),
                int(a["n_prof"][e]), num(a["delta_norm"][e]),
                num(a["spatial_max"][e]), num(a["spatial_l2"][e]),
                num(a["redundancy"][e], 4),
                ("" if a["twin"][e] is None else f"e{a['twin'][e]}"),
                num(a["spec_std"][e], 4), num(a["spec_peak"][e], 4)]
               for e in range(E)])

    write_csv(os.path.join(out, "expert_absent.csv"),
              ["expert", "reason", "n_chosen", "giải thích"],
              [[x["expert"], x["reason"], x["n_chosen"],
                "không snapshot nào chọn expert này"
                if x["reason"] == "router_never_picked"
                else "được chọn nhưng bản ghi không kèm vector bệnh"]
               for x in a["absent"]])

    # Các ma trận chuyên môn chỉ chứa expert có hồ sơ. Một hàng rỗng hay hàng
    # 0 ở đây sẽ bị mọi công cụ đọc CSV coi là số thật.
    mats = [("expert_disease_mean", a["disease"], dn),
            ("expert_disease_z", a["disease_z"], dn),
            ("expert_aspect_mean", a["aspect"], an),
            ("expert_aspect_z", a["aspect_z"], an)]
    for g in a["field_keys"]:
        names = a["field_labels"][g]
        mats += [(f"expert_field_{g}_mean", a["fields"][g], names),
                 (f"expert_field_{g}_z", a["field_z"][g], names)]
    for name, M, colnames in mats:
        if M.size:
            write_csv(os.path.join(out, name + ".csv"), ["expert"] + list(colnames),
                      [[e] + [num(v) for v in M[e]] for e in live])

    if a["global_z"].size:
        write_csv(os.path.join(out, "expert_global_z.csv"),
                  ["expert"] + [f"{g}:{n}" for g, n in a["global_cols"]],
                  [[e] + [num(v) for v in a["global_z"][e]] for e in live])

    for name, S in (("expert_similarity_global", a["similarity_global"]),
                    ("expert_similarity_disease", a["similarity"]),
                    ("expert_similarity_spatial", a["similarity_spatial"])):
        write_csv(os.path.join(out, name + ".csv"),
                  ["expert"] + [f"e{j}" for j in live],
                  [[e] + [num(S[e][j], 4) for j in live] for e in live])

    # Đồng xuất hiện là số đếm thô của router -> giữ đủ E×E.
    write_csv(os.path.join(out, "expert_cooccurrence.csv"),
              ["expert"] + [f"e{j}" for j in range(E)],
              [[e] + [int(v) for v in a["cooccurrence"][e]] for e in range(E)])

    rows = []
    for e in live:
        for rank, (txt, cnt, mean, freq) in enumerate(top_tokens(a, e, 25), 1):
            rows.append([e, rank, txt, cnt, num(freq, 4), num(mean)])
    write_csv(os.path.join(out, "expert_top_tokens.csv"),
              ["expert", "rank", "token", "n_images", "freq", "dlogit_mean"], rows)

    if a["spatial"] is not None:
        h, w = a["grid"]
        sp = [e for e in live if a["n_spatial"][e] > 0]
        write_csv(os.path.join(out, "expert_spatial_mean.csv"),
                  ["expert"] + [f"r{i}c{j}" for i in range(h) for j in range(w)],
                  [[e] + [num(v, 4) for v in a["spatial"][e]] for e in sp])


def dump_json(a, out, source):
    E = a["n_expert"]
    lvl, msg = verdict(a)
    doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "n_records": a["n_records"],
        "n_full_records": a["n_full"],
        "n_topk_records": a["n_topk"],
        "profile_source": a["profile_source"],
        "n_profile_records": a["n_profile_records"],
        "n_expert": E, "top_k": a["top_k"],
        "grid": a["grid"],
        "meta": a["meta"],
        "train_samples": a["train_samples"],
        # clone_index/gate_max nói về n_profiled expert; balance nói về đủ E.
        # Ghi cả hai mẫu số ra đây để người đọc JSON không phải đoán.
        "n_profiled": a["n_profiled"],
        "headline": {"clone_index": jnum(a["clone_index"], 4),
                     "clone_index_max": jnum(a["clone_index_max"], 4),
                     "clone_index_over_experts": a["n_profiled"],
                     "clone_index_over_labels": len(a["global_cols"]),
                     "clone_index_disease": jnum(a["clone_index_disease"], 4),
                     "clone_index_spatial": jnum(a["clone_index_spatial"], 4),
                     "balance": jnum(a["balance"], 4),
                     "balance_over": E,
                     "prob_balance": jnum(a["prob_balance"], 4),
                     "effective_experts": jnum(a["effective_experts"], 3),
                     "effective_experts_picked": jnum(a["effective_experts_picked"], 3),
                     "usage_cv": jnum(a["usage_cv"], 4),
                     "route_entropy": a["route_entropy"],
                     "gate_max": a["gate_max"],
                     "gate_effect_corr": jnum(a["gate_effect_corr"], 4),
                     "dead_experts": a["dead_experts"],
                     "level": lvl, "verdict": msg},
        "absent_experts": a["absent"],
        "disease_labels": a["disease_labels"],
        "aspect_labels": a["aspect_labels"],
        "field_labels": a["field_labels"],
        "field_titles": a["field_titles"],
        "experts": [
            {"expert": e,
             "in_summary": bool(a["profile"][e]),
             "route_share": jnum(a["route_share"][e]),
             "usage": jnum(a["usage"][e]),
             "n_chosen": int(a["n_chosen"][e]),
             "n_profile_samples": int(a["n_prof"][e]),
             "top1_share": jnum(a["n_top1"][e] / max(a["n_records"], 1)),
             "gate_mean_when_chosen": jnum(a["gate_mean_when_chosen"][e]),
             "delta_norm": jnum(a["delta_norm"][e]),
             "redundancy": jnum(a["redundancy"][e], 4),
             "closest_twin": (None if a["twin"][e] is None else int(a["twin"][e])),
             "spec_std": jnum(a["spec_std"][e], 4),
             "spec_peak": jnum(a["spec_peak"][e], 4),
             "top_labels": [{"group": g, "label": n, "z": round(z, 3)}
                            for g, n, z in top_global(a, e, 6)],
             "top_disease": top_labels(a["disease_z"][e], a["disease_labels"]),
             "top_aspect": (top_labels(a["aspect_z"][e], a["aspect_labels"])
                            if a["aspect"].size else []),
             "top_tokens": [{"token": t, "n_images": c, "dlogit_mean": round(m, 5)}
                            for t, c, m, _ in top_tokens(a, e, 12)]}
            for e in range(E)],
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    return doc


# ══════════════════════════════════ Markdown ═════════════════════════════════
BLOCKS = " ░▒▓█"


def heat_char(z):
    """z-score -> một ký tự đậm nhạt, để bảng markdown đọc được không cần màu."""
    if isnan(z) or z <= 0:
        return "·"
    return BLOCKS[min(int(z / 0.75) + 1, len(BLOCKS) - 1)]


def fmt(v, nd=2, dash="—"):
    """Số để in ra bảng; NaN -> gạch ngang, không phải 0.00."""
    return dash if isnan(v) else f"{float(v):.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def dump_md(a, out, source):
    E, K = a["n_expert"], a["top_k"]
    live = [e for e in range(E) if a["profile"][e]]
    P = a["n_profiled"]
    lvl, msg = verdict(a)
    dn = a["disease_labels"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"# Probe chuyên môn expert — {a['n_records']} ảnh", "",
         f"*Sinh bởi `scripts/probe_summary.py` lúc {ts}. "
         f"Nguồn: `{source}`. Chạy lại script sẽ ghi đè file này.*", "",
         "## Kết luận", "", f"**Mức trùng lặp: {lvl}.** {msg}", "",
         md_table(["chỉ số", "giá trị", "nghĩa"], [
             ["clone_index", fmt(a["clone_index"], 3),
              f"cosine lệch trung bình giữa các cặp trong {P} expert, trên "
              f"{len(a['global_cols'])} nhãn đã z-score — 1 = bản sao, "
              f"0 = riêng biệt"],
             ["clone_index_max", fmt(a["clone_index_max"], 3),
              "cặp giống nhau nhất"],
             ["clone_index_disease", fmt(a["clone_index_disease"], 3),
              "cùng phép đo nhưng CHỈ trên 14 tên bệnh"],
             ["clone_index_spatial", fmt(a["clone_index_spatial"], 3),
              "cùng phép đo trên bản đồ không gian — các expert có nhìn cùng "
              "một vùng phổi không"],
             ["effective_experts", fmt(a["effective_experts"], 2),
              f"exp(entropy của `route_share`) — router thực chất đang dùng "
              f"bao nhiêu trong {E} expert"],
             ["prob_balance", fmt(a["prob_balance"], 3),
              f"entropy chuẩn hoá của `route_share` trên ĐỦ {E} expert — "
              f"1 = chia đều trọng số"],
             ["balance", fmt(a["balance"], 3),
              f"như trên nhưng đếm LƯỢT lọt top-k thay vì trọng số"],
             ["usage_cv", fmt(a["usage_cv"], 3),
              "hệ số biến thiên của tần suất chọn — 0 = chia đều"],
             ["route_entropy", fmt(a["route_entropy"], 3),
              f"entropy của phân phối router (nat), tối đa = ln {E} = "
              f"{math.log(E):.2f}" if E > 1 else "—"],
             ["gate_max", fmt(a["gate_max"], 3),
              f"trọng số lớn nhất trung bình (1/top_k = {1/max(K,1):.3f})"],
             ["gate_effect_corr", fmt(a["gate_effect_corr"], 3),
              "tương quan giữa `route_share` và ‖Δh‖ — router có ưu tiên đúng "
              "expert làm nhiều việc không"],
         ]), ""]

    meta = a["meta"]
    L += ["## Ảnh chụp này là của lúc nào", "",
          md_table(["trường", "giá trị"], [
              ["model", ", ".join(meta.get("model", [])) or "—"],
              ["lần chạy", ", ".join(meta.get("run_tag", [])) or "—"],
              ["trọng số", ", ".join(meta.get("weights", [])) or "—"],
              ["n_expert × top_k", f"{E} × {K}"],
              ["bản ghi `full` (đủ {} expert)".format(E), f"{a['n_full']:,}"],
              ["bản ghi `topk` (chỉ expert được chọn)", f"{a['n_topk']:,}"],
              ["hồ sơ chuyên môn dựng trên",
               f"`{a['profile_source']}` × {a['n_profile_records']:,} bản ghi"],
              ["expert vào bảng tổng hợp", f"{P}/{E}"
               + (f" ({E-P} bị loại, xem mục dưới)" if P < E else "")],
              ["train_samples", (f"{a['train_samples'][0]:,} … {a['train_samples'][1]:,}"
                                 if a["train_samples"] else "—")],
              ["lưới patch", "×".join(map(str, a["grid"])) if a["grid"] else "—"],
              ["số bản ghi", f"{a['n_records']:,}"],
          ]), "",
          "> Bản ghi được gộp ở đây có thể đến từ nhiều thời điểm train khác "
          "nhau (probe chạy song song với train). Khoảng `train_samples` rộng "
          "nghĩa là bảng dưới trộn nhiều phiên bản checkpoint.", ""]

    if a["absent"]:
        L += [f"## Expert không tham gia ({E - P}/{E}) — ĐỂ RIÊNG, không gộp", "",
              "Những expert dưới đây bị loại khỏi mọi bảng chuyên môn phía sau "
              "(ma trận nhãn, cosine, top token, bản đồ). Gộp chúng vào với tư "
              "cách một hàng 0 sẽ dịch trung bình cột, bóp std và kéo cosine "
              "của mọi cặp còn lại — tức là làm sai chính con số cần đọc.", "",
              md_table(["expert", "lý do", "lượt được chọn", "nghĩa là"],
                       [[f"e{x['expert']}",
                         f"`{x['reason']}`", f"{x['n_chosen']:,}",
                         "không snapshot nào chọn expert này — tham số chết, "
                         "hoặc bias cân bằng tải (§3.6) chưa kéo nó vào. Chạy "
                         "probe với `probe_full_n` > 0 sẽ ép bật nó và cho nó "
                         "một hồ sơ"
                         if x["reason"] == "router_never_picked" else
                         "có chạy thật nhưng bản ghi không kèm vector bệnh — "
                         "lỗ hổng DỮ LIỆU, không phải kết luận về expert"]
                        for x in a["absent"]]), "",
              "`balance`, `usage_cv` và `route_share` ở trên vẫn tính trên đủ "
              f"{E} expert: \"chưa từng được chọn\" chính là dữ kiện quan trọng "
              "nhất của câu hỏi cân bằng tải, bỏ ra là tự làm đẹp số liệu.", ""]

    L += ["## Router phân bổ cho ai", "",
          f"Đủ {E} expert. `route_share` = trung bình softmax(route_logit) trên "
          f"{a['n_prob_records']:,} mẫu, tính TRƯỚC khi cắt top-k — nên expert "
          f"không bao giờ lọt top-k vẫn có số. Chia đều tuyệt đối = "
          f"{100/max(E,1):.1f}%. `% lọt top-k` thì chia đều = "
          f"{100*K/max(E,1):.0f}%. Hàng có `✗` không vào bảng chuyên môn.", "",
          md_table(["expert", "hồ sơ", "route_share", "% lọt top-k", "top-1",
                    "gate TB", "‖Δh‖ TB", "giống nhất", "nổi bật (z > 0.35)"],
                   [[f"e{e}", "✓" if a["profile"][e] else "✗",
                     f"{100*a['route_share'][e]:.1f}%"
                     if not isnan(a["route_share"][e]) else "—",
                     f"{100*a['usage'][e]:.1f}%",
                     f"{100*a['n_top1'][e]/max(a['n_records'],1):.1f}%",
                     fmt(a["gate_mean_when_chosen"][e], 3),
                     fmt(a["delta_norm"][e], 2),
                     ("—" if a["twin"][e] is None
                      else f"e{a['twin'][e]} ({a['redundancy'][e]:+.2f})"),
                     ", ".join(f"{n} ({z:+.1f})"
                               for _g, n, z in top_global(a, e, 3)) or "—"]
                    for e in range(E)]), ""]

    def heat_md(M, names, title, note):
        rows = [[f"e{e}"] + [f"{heat_char(M[e][j])}{M[e][j]:+.1f}"
                             for j in range(len(names))] for e in live]
        return [f"### {title}", "", note, "",
                md_table(["exp"] + list(names), rows), ""]

    L += ["## Bảng nhiệt expert × nhãn (z-score theo cột)", "",
          "Đậm = expert này đẩy nhãn đó mạnh hơn mặt bằng các expert khác. "
          "`·` = bằng hoặc dưới mặt bằng. Chuẩn theo CỘT vì mặt bằng logit của "
          "từng tên chênh nhau cả bậc. Số thô ở các file `expert_*_mean.csv`.",
          ""]
    if len(dn) and live:
        L += heat_md(a["disease_z"], dn, "14 tên bệnh (CheXbert)",
                     "Câu hỏi kinh điển: expert nào thiên về bệnh nào.")
    if a["aspect"].size and live:
        L += heat_md(a["aspect_z"], a["aspect_labels"], "4 aspect",
                     "Cùng cách đọc, trên 4 tên aspect.")
    for g in a["field_keys"]:
        if a["field_z"][g].size and live:
            note = FIELD_NOTE.get(g, FIELD_NOTE_DEFAULT)
            note = note.replace("<code>", "`").replace("</code>", "`")
            L += heat_md(a["field_z"][g], a["field_labels"][g],
                         a["field_titles"][g], note)

    L += ["## Cặp expert giống nhau nhất", "",
          "Cosine trên toàn bộ nhãn đã z-score, sau khi trừ hồ sơ trung bình "
          "của cả nhóm.", ""]
    sim = a["similarity_global"]
    pairs = sorted(((float(sim[i][j]), i, j)
                    for i in live for j in live if i < j
                    and not isnan(sim[i][j])), reverse=True)
    L += [md_table(["cặp", "cosine lệch", "đọc là"],
                   [[f"e{i} ↔ e{j}", f"{s:+.3f}",
                     "gần như trùng" if s > 0.7 else
                     "chồng lấn" if s > 0.3 else
                     "khác hướng" if s > -0.3 else "đối lập"]
                    for s, i, j in pairs[:8]]), ""]

    L += ["## Token mỗi expert đẩy lên mạnh nhất", "",
          "Xếp theo tổng Δlogit trên cả tập, nên token xuất hiện đều đặn thắng "
          "token bùng lên ở một ảnh. `n` = số ảnh mà token lọt top của expert.", ""]
    for e in live:
        toks = ", ".join(f"`{t.strip() or '␣'}`(n={c})" for t, c, _, _ in top_tokens(a, e, 10))
        L += [f"- **e{e}** — {toks or '—'}"]
    L += [""]

    sp = [e for e in live if a["n_spatial"][e] > 0] if a["spatial"] is not None else []
    if sp:
        h, w = a["grid"]
        L += [f"## Bản đồ không gian trung bình ({h}×{w})", "",
              "‖Δh‖ tại từng image token, trung bình trên mọi ảnh mà expert "
              "được bật, chuẩn hoá về [0,1] theo chính nó. Đây là câu hỏi "
              "*nhìn ĐÂU*, không phải *mạnh cỡ nào* (độ mạnh ở cột ‖Δh‖ phía "
              "trên).", "",
              "```"]
        maps = {}
        for e in sp:
            m = a["spatial"][e].reshape(h, w)
            mx = m.max() or 1.0
            maps[e] = [[BLOCKS[min(int(v / mx * (len(BLOCKS) - 1) + 0.5),
                                   len(BLOCKS) - 1)] for v in row] for row in m]
        per_row = 4
        for start in range(0, len(sp), per_row):
            grp = sp[start:start + per_row]
            L += ["   ".join(f"e{e}".ljust(w * 2) for e in grp)]
            for i in range(h):
                L += ["   ".join("".join(c * 2 for c in maps[e][i]) for e in grp)]
            L += [""]
        L += ["```", ""]

    L += ["---", "",
          "Xem bản có màu: `reports/probe/index.html`. Số liệu máy đọc: "
          "`summary.json`, `*.csv`.", "",
          "*Không có text báo cáo hay `key_id` nào trong thư mục này — bản ghi "
          "gốc chứa báo cáo MIMIC-CXR (dữ liệu PhysioNet có kiểm soát) nên chỉ "
          "số liệu đã gộp mới được xuất ra đây.*", ""]

    with open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# ════════════════════════════════════ HTML ═══════════════════════════════════
CSS = """
:root{--bg:#fbfaf8;--fg:#1e1c1a;--mut:#6b6560;--line:#e3ded7;--card:#fff;
      --pos:#c0532a;--neg:#2f6690;--acc:#7a5c3e;--warn:#8a5a1f}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--fg:#eae7e2;--mut:#9a938c;
      --line:#2e2c33;--card:#1e1d23;--pos:#e0764a;--neg:#5b9bc7;--acc:#c9a87c;
      --warn:#d3a25c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:36px 22px 80px}
h1{font-size:27px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:19px;margin:44px 0 6px;letter-spacing:-.01em}
h2:first-of-type{margin-top:32px}
h3{font-size:15.5px;margin:26px 0 4px;letter-spacing:-.01em}
p.sub{color:var(--mut);margin:0 0 6px;font-size:13.5px}
.note{color:var(--mut);font-size:13.5px;margin:2px 0 14px;max-width:76ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:16px 18px;margin:14px 0}
.verdict{border-left:3px solid var(--acc)}
.absent{border-left:3px solid var(--neg)}
.warnbox{border-left:3px solid var(--warn)}
.absent ul{margin:0 0 10px;padding-left:18px;font-size:13.5px}
.absent li+li{margin-top:4px}
tr.off td{opacity:.45}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;
     padding:11px 15px;min-width:132px}
.kpi b{display:block;font-size:23px;font-weight:600;letter-spacing:-.02em}
.kpi span{color:var(--mut);font-size:12px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:5px 8px;text-align:right;white-space:nowrap;
      border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
th{font-weight:600;color:var(--mut);font-size:11.5px;text-transform:uppercase;
   letter-spacing:.04em;vertical-align:bottom}
th.rot{height:118px;padding:0 2px;text-align:left;white-space:nowrap}
th.rot>div{transform:rotate(-60deg);transform-origin:left bottom;width:18px}
td.h{text-align:center;font-variant-numeric:tabular-nums;color:#20201f;
     border:1px solid var(--bg);border-radius:3px;min-width:52px}
/* Bảng nhiệt ít cột (4 aspect) mà kéo full width thì ô rộng bằng nửa màn
   hình và mắt không còn so được hai ô cạnh nhau — để nó tự co theo nội dung. */
table.hm{width:auto}
.bar{height:8px;border-radius:4px;background:var(--acc);display:inline-block;
     vertical-align:middle;margin-right:6px}
.bar2{background:var(--neg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(268px,1fr));gap:12px}
.ex{background:var(--card);border:1px solid var(--line);border-radius:10px;
    padding:14px 15px}
.ex h3{margin:0 0 2px;font-size:15px}
.ex .m{color:var(--mut);font-size:12px;margin-bottom:9px}
.ex ul{margin:6px 0 0;padding-left:17px;font-size:13px}
.sp{display:grid;gap:0;width:150px;margin:8px 0 4px;border:1px solid var(--line)}
.sp i{display:block;aspect-ratio:1}
.tok{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}
.tok code{background:var(--bg);border:1px solid var(--line);border-radius:5px;
          padding:1px 5px;font-size:12px}
.tag{font-size:11px;color:var(--mut);border:1px solid var(--line);
     border-radius:4px;padding:0 4px;margin-right:4px}
footer{color:var(--mut);font-size:12.5px;margin-top:46px;border-top:1px solid
       var(--line);padding-top:14px;max-width:76ch}
"""


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def col_div(z, cap=2.5):
    """z-score -> màu phân kỳ xanh↔đỏ, nền trung tính ở 0.

    Cắt ở ±2.5σ: với 12 expert, một z lớn hơn thế gần như luôn là một expert
    lệch hẳn, và để thang trải tới tận nó thì mọi ô còn lại nhạt như nhau.
    """
    if isnan(z):
        return "transparent"
    t = max(-1.0, min(1.0, float(z) / cap))
    if t >= 0:
        r, g, b = 246 - 44 * t, 243 - 128 * t, 238 - 175 * t
    else:
        u = -t
        r, g, b = 246 - 196 * u, 243 - 128 * u, 238 - 34 * u
    return f"rgb({r:.0f},{g:.0f},{b:.0f})"


SEQ_STEPS = 20


def col_seq(v):
    """[0,1] -> màu tuần tự nhạt→đậm cho bản đồ không gian."""
    t = max(0.0, min(1.0, float(v)))
    return f"rgb({250-146*t:.0f},{247-166*t:.0f},{240-146*t:.0f})"


def seq_css():
    """Thang màu bản đồ dựng sẵn thành class .q0..q19 thay vì style từng ô.

    12 expert × 256 ô = 3072 thẻ. Nhét `style='background:rgb(...)'` vào từng
    thẻ là ~120KB chỉ để lặp lại 20 màu, và mỗi lần chạy lại giá trị lệch một
    chút ở chữ số cuối là git coi như cả file đổi. Lượng tử hoá 20 mức: file
    còn ~1/3 và diff chỉ hiện những ô đổi mức thật.
    """
    return "".join(f".q{i}{{background:{col_seq(i/(SEQ_STEPS-1))}}}"
                   for i in range(SEQ_STEPS))


def seq_class(v):
    return f"q{min(int(max(0.0, min(1.0, float(v))) * SEQ_STEPS), SEQ_STEPS - 1)}"


def dump_html(a, out, source):
    E, K = a["n_expert"], a["top_k"]
    NP = a["n_profiled"]
    live = [e for e in range(E) if a["profile"][e]]
    dn, an = a["disease_labels"], a["aspect_labels"]
    lvl, msg = verdict(a)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = a["meta"]
    P = []

    P.append("<h1>Probe chuyên môn expert</h1>")
    P.append(f"<p class=sub>{a['n_records']:,} ảnh · {a['n_full']:,} bản ghi đủ "
             f"{E} expert · {NP}/{E} expert có hồ sơ · top-{K} · "
             f"{esc(', '.join(meta.get('run_tag', [])) or 'lần chạy không rõ')} · "
             f"sinh lúc {ts}</p>")
    P.append(f"<p class=sub>Nguồn: <code>{esc(source)}</code> · "
             f"trọng số: {esc(', '.join(meta.get('weights', [])) or '—')}"
             + (f" · train_samples {a['train_samples'][0]:,}–{a['train_samples'][1]:,}"
                if a["train_samples"] else "") + "</p>")

    if a["profile_source"] != "full":
        P.append(f"<div class='card warnbox'>{esc(source_caveat(a))}</div>")

    P.append(f"<div class='card verdict'><b>Mức trùng lặp: {lvl}.</b> {esc(msg)}</div>")

    kpi = [("clone_index", fmt(a["clone_index"]),
            f"{len(a['global_cols'])} nhãn × {NP} expert · 1 = bản sao"),
           ("clone_index (bệnh)", fmt(a["clone_index_disease"]),
            f"chỉ {len(dn)} tên bệnh"),
           ("effective_experts", fmt(a["effective_experts"]),
            f"exp(H của route_share) · tối đa {E}"),
           ("balance", fmt(a["balance"]), f"lượt lọt top-k · 1 = chia đều"),
           ("gate_max", fmt(a["gate_max"]), f"1/top_k = {1/max(K,1):.2f}"),
           ("gate↔‖Δh‖", fmt(a["gate_effect_corr"]),
            "router có ưu tiên đúng expert làm việc")]
    if a["dead_experts"]:
        kpi.append((f"{len(a['dead_experts'])} expert chết", "⚠",
                    "chưa từng lọt top-k"))
    if a["absent"]:
        kpi.append((f"{E - NP} expert vắng", "⚠", "không có hồ sơ"))
    P.append("<div class=kpis>" + "".join(
        f"<div class=kpi><b>{v}</b><span>{esc(k)}<br>{esc(h)}</span></div>"
        for k, v, h in kpi) + "</div>")

    # ── expert không tham gia ────────────────────────────────────────────────
    if a["absent"]:
        items = []
        for x in a["absent"]:
            why = ("không snapshot nào chọn expert này — tham số chết, hoặc "
                   "bias cân bằng tải chưa kéo nó vào. Chạy probe với "
                   "probe_full_n > 0 sẽ ép bật nó và cho nó một hồ sơ"
                   if x["reason"] == "router_never_picked" else
                   f"được chọn {x['n_chosen']:,} lần nhưng bản ghi không kèm "
                   f"vector bệnh — lỗ hổng DỮ LIỆU, không phải kết luận về expert")
            items.append(f"<li><b>e{x['expert']}</b> · <code>{x['reason']}</code>"
                         f" — {esc(why)}</li>")
        P.append(
            f"<h2>Expert không tham gia ({E - NP}/{E})</h2>"
            "<p class=note>Để RIÊNG, không gộp vào bất kỳ bảng nào phía dưới. "
            "Cho một expert vắng mặt vào với tư cách hàng 0 sẽ dịch trung bình "
            "cột, bóp std và kéo cosine của mọi cặp còn lại — tức là làm sai "
            "chính con số cần đọc.</p>"
            f"<div class='card absent'><ul>{''.join(items)}</ul>"
            f"<p class=note style='margin-bottom:0'><code>balance</code>, "
            f"<code>usage_cv</code> và <code>route_share</code> vẫn tính trên "
            f"đủ {E} expert: \"chưa từng được chọn\" chính là dữ kiện quan "
            f"trọng nhất của câu hỏi cân bằng tải, bỏ ra là tự làm đẹp số "
            f"liệu.</p></div>")

    # ── router phân bổ cho ai ────────────────────────────────────────────────
    P.append("<h2>Router phân bổ cho ai</h2>")
    P.append(f"<p class=note>Thanh nâu = <code>route_share</code>: trung bình "
             f"softmax(route_logit) trên {a['n_prob_records']:,} mẫu, tính "
             f"TRƯỚC khi cắt top-k — nên expert không bao giờ lọt top-k vẫn có "
             f"số ở đây. Chia đều tuyệt đối = {100/max(E,1):.1f}%. Thanh xanh = "
             f"tỉ lệ ảnh mà expert lọt top-k; chia đều = top_k/n_expert = "
             f"{100*K/max(E,1):.0f}%. Hàng mờ là expert không có hồ sơ.</p>")
    rows = []
    mxp = np.nanmax(a["route_share"]) if np.isfinite(a["route_share"]).any() else 1.0
    mxu = max(a["usage"].max(), 1e-9)
    for e in range(E):
        ok = bool(a["profile"][e])
        rs = a["route_share"][e]
        wp = 0 if isnan(rs) else 118 * float(rs) / max(float(mxp), 1e-9)
        wu = 118 * a["usage"][e] / mxu
        tw = ("—" if a["twin"][e] is None
              else f"e{a['twin'][e]} ({a['redundancy'][e]:+.2f})")
        lab = ", ".join(f"{n} ({z:+.1f})" for _g, n, z in top_global(a, e, 3))
        rows.append(
            f"<tr{'' if ok else ' class=off'}><td>e{e}</td>"
            f"<td>{'✓' if ok else '✗'}</td>"
            f"<td style='text-align:left'><span class=bar style='width:{wp:.0f}px'></span>"
            + ("—" if isnan(rs) else f"{100*float(rs):.1f}%") + "</td>"
            f"<td style='text-align:left'><span class='bar bar2' style='width:{wu:.0f}px'></span>"
            f"{100*a['usage'][e]:.1f}%</td>"
            f"<td>{100*a['n_top1'][e]/max(a['n_records'],1):.1f}%</td>"
            f"<td>{fmt(a['gate_mean_when_chosen'][e], 3)}</td>"
            f"<td>{fmt(a['delta_norm'][e])}</td>"
            f"<td>{fmt(a['spec_std'][e])}</td>"
            f"<td>{esc(tw)}</td>"
            f"<td style='text-align:left'>{esc(lab or '—')}</td></tr>")
    P.append("<div class=scroll><table><tr><th>expert</th><th>hồ sơ</th>"
             "<th>route_share</th><th>% lọt top-k</th>"
             "<th>là top-1</th><th>gate TB</th><th>‖Δh‖ TB</th>"
             "<th>độ chuyên biệt</th><th>giống nhất</th>"
             "<th>nổi bật (z&gt;0.35)</th></tr>" + "".join(rows) + "</table></div>")

    # ── heatmap từng nhóm nhãn ───────────────────────────────────────────────
    def heat(M, names, title, note, idx, level=3):
        head = "".join(f"<th class=rot><div>{esc(n)}</div></th>" for n in names)
        body = []
        for e in idx:
            cells = "".join(
                f"<td class=h style='background:{col_div(M[e][j])}' "
                f"title='{esc(names[j])} · z={M[e][j]:+.2f}'>"
                + ("—" if isnan(M[e][j]) else f"{M[e][j]:+.1f}") + "</td>"
                for j in range(len(names)))
            body.append(f"<tr><td>e{e}</td>{cells}</tr>")
        return (f"<h{level}>{title}</h{level}><p class=note>{note}</p>"
                f"<div class=scroll><table class=hm><tr><th>exp</th>{head}</tr>"
                + "".join(body) + "</table></div>")

    P.append("<h2>Bảng nhiệt expert × nhãn</h2>")
    P.append("<p class=note>z-score theo CỘT: đỏ = expert này đẩy nhãn đó mạnh "
             "hơn mặt bằng các expert khác, xanh = yếu hơn. Chuẩn theo cột vì "
             "mặt bằng logit của từng tên chênh nhau cả bậc — Δlogit của "
             "<code>the</code> lớn hơn của <code>Atelectasis</code> cả bậc, "
             "không chuẩn hoá thì cả bảng chỉ đọc ra được điều đó. Năm nhóm "
             "sau phần bệnh/aspect là các trục NGOÀI danh sách chẩn đoán: "
             "không đo chúng thì một expert chuyên văn phong hay chuyên phần "
             "so sánh với phim cũ sẽ hiện ra là \"không nổi trội\" — sai, chỉ "
             "vì thước đo thiếu ô. Hồ sơ dựng "
             f"trên {a['n_profile_records']:,} bản ghi "
             f"<code>{a['profile_source']}</code>. Số thô: các file "
             "<code>expert_*_mean.csv</code>.</p>")
    if len(dn) and live:
        P.append(heat(a["disease_z"], dn, "14 tên bệnh (CheXbert)",
                      "Câu hỏi kinh điển: expert nào thiên về bệnh nào.", live))
    if an and a["aspect"].size and live:
        P.append(heat(a["aspect_z"], an, "4 aspect",
                      "Cùng cách đọc, trên 4 tên aspect.", live))
    for g in a["field_keys"]:
        if a["field_z"][g].size and live:
            P.append(heat(a["field_z"][g], a["field_labels"][g],
                          esc(a["field_titles"][g]),
                          FIELD_NOTE.get(g, FIELD_NOTE_DEFAULT), live))

    # ── ma trận tương đồng ───────────────────────────────────────────────────
    def simtab(S, title, note):
        head = "".join(f"<th>e{j}</th>" for j in live)
        body = []
        for i in live:
            cells = []
            for j in live:
                if i == j:
                    cells.append("<td class=h style='background:transparent'>—</td>")
                    continue
                v = float(S[i][j])
                if isnan(v):
                    cells.append("<td class=h style='background:transparent'>—</td>")
                    continue
                # cosine nằm trong [-1, 1] còn thang màu nhận z-score cắt ở
                # ±2.5: nhân 2.5 để +1 chạm hết đầu đỏ và −1 chạm hết đầu xanh.
                cells.append(f"<td class=h style='background:{col_div(v * 2.5)}'>"
                             f"{v:+.2f}</td>")
            body.append(f"<tr><td>e{i}</td>{''.join(cells)}</tr>")
        return (f"<h3>{title}</h3><p class=note>{note}</p>"
                "<div class=scroll><table><tr><th>exp</th>" + head + "</tr>"
                + "".join(body) + "</table></div>")

    P.append("<h2>Expert giống nhau tới đâu</h2>")
    P.append("<p class=note>Cosine giữa hồ sơ của hai expert SAU KHI trừ hồ sơ "
             "trung bình của cả nhóm. Không trừ thì mọi cặp đều ~0.9 (các "
             "expert cùng sống trong miền X-quang) và ma trận vô nghĩa. "
             "+1 = trùng hướng, 0 = độc lập, −1 = đối lập.</p>")
    P.append(simtab(a["similarity_global"], "Toàn bộ nhãn (đã z-score)",
                    f"{len(a['global_cols'])} cột: bệnh + aspect + "
                    f"{len(a['field_keys'])} nhóm token. Ghép z-score chứ không "
                    "ghép giá trị thô — ghép thô thì nhóm ngữ pháp áp đảo và "
                    "cosine này thực chất chỉ là cosine của nhóm đó."))
    P.append(simtab(a["similarity"], "Chỉ 14 tên bệnh",
                    "Cùng phép đo, giới hạn ở phần bệnh. Lệch nhiều so với "
                    "bảng trên nghĩa là các expert chia việc theo trục khác "
                    "chứ không theo bệnh."))
    if a["spatial"] is not None and np.isfinite(a["similarity_spatial"]).any():
        P.append(simtab(a["similarity_spatial"], "Bản đồ không gian",
                        "Các expert có nhìn cùng một vùng phổi không. Cao ở "
                        "đây nhưng thấp ở bảng nhãn = chúng nhìn cùng chỗ "
                        "nhưng rút ra kết luận khác nhau."))

    # ── thẻ từng expert ──────────────────────────────────────────────────────
    P.append("<h2>Từng expert</h2>")
    P.append("<p class=note>Bản đồ = ‖Δh‖ trung bình tại từng image token "
             "(chuẩn hoá theo chính nó — câu hỏi là <i>nhìn ĐÂU</i>). "
             "Token = unembedding mà expert đẩy lên mạnh nhất trên cả tập.</p>")
    cards = []
    for e in live:
        c = [f"<div class=ex><h3>Expert {e}</h3>",
             f"<div class=m>"
             + ("route_share —" if isnan(a["route_share"][e])
                else f"route_share {100*float(a['route_share'][e]):.1f}%")
             + f" · lọt top-k {100*a['usage'][e]:.1f}% · "
             f"‖Δh‖ {fmt(a['delta_norm'][e])} · "
             f"{int(a['n_prof'][e]):,} mẫu hồ sơ</div>"]
        if a["spatial"] is not None and a["n_spatial"][e] > 0:
            h, w = a["grid"]
            m = a["spatial"][e]
            mx = m.max() or 1.0
            c.append(f"<div class=sp style='grid-template-columns:repeat({w},1fr)'>"
                     + "".join(f"<i class={seq_class(v/mx)}></i>" for v in m)
                     + "</div>")
        c.append("<ul>")
        tg = top_global(a, e, 5)
        c.append("<li>nổi bật: " + ("".join(
            f"<span class=tag>{esc(g)}</span>{esc(n)} ({z:+.1f})<br>"
            for g, n, z in tg) or "<i>không nổi trội ở nhãn nào</i>") + "</li>")
        td = top_labels(a["disease_z"][e], dn)
        c.append("<li>bệnh: " + (esc(", ".join(f"{n} ({z:+.1f})" for n, z in td))
                                 or "<i>không nổi trội</i>") + "</li>")
        if a["twin"][e] is not None:
            c.append(f"<li>giống nhất: e{a['twin'][e]} "
                     f"(cosine {a['redundancy'][e]:+.2f})</li>")
        c.append("</ul>")
        c.append("<div class=tok>" + "".join(
            f"<code title='{cnt} ảnh · Δlogit TB {mean:.3f}'>{esc(t.strip() or '␣')}</code>"
            for t, cnt, mean, _ in top_tokens(a, e, 10)) + "</div></div>")
        cards.append("".join(c))
    P.append("<div class=grid>" + "".join(cards) + "</div>")

    P.append("<footer>Sinh bởi <code>scripts/probe_summary.py</code> — chạy lại "
             "để cập nhật, file này bị ghi đè.<br>Thư mục này KHÔNG chứa text "
             "báo cáo hay <code>key_id</code>: bản ghi gốc dựa trên MIMIC-CXR "
             "(PhysioNet, có kiểm soát truy cập), nên chỉ số liệu đã gộp trên "
             "toàn tập mới được xuất ra đây.</footer>")

    doc = ("<!doctype html><html lang=vi><head><meta charset=utf-8>"
           "<meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Probe expert · {a['n_records']} ảnh</title>"
           f"<style>{CSS}{seq_css()}</style>"
           f"</head><body><div class=wrap>{''.join(P)}</div></body></html>")
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(doc)


# ══════════════════════════════════ vào/ra ═══════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Gộp bản ghi probe trên HF thành báo cáo HTML/Markdown/CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Chạy lại sẽ GHI ĐÈ thư mục --out. Không có text báo cáo nào "
               "được xuất ra (xem docstring đầu file).")
    ap.add_argument("--repo", help="dataset repo probe; mặc định lấy từ "
                                   "configs/default.yaml hoặc $RGEXP_PROBE_REPO")
    ap.add_argument("--revision", default=None, help="nhánh/commit trên Hub")
    ap.add_argument("--local-dir", help="đọc thẳng thư mục JSON trên đĩa, "
                                        "không đụng mạng")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"mặc định {DEFAULT_OUT}")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--limit", type=int, help="chỉ đọc N bản ghi đầu (thử nhanh)")
    ap.add_argument("--token", default=None, help="HF token; mặc định $HF_TOKEN")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.local_dir:
        src_dir, source = args.local_dir, os.path.abspath(args.local_dir)
        if not os.path.isdir(src_dir):
            sys.exit(f"✗ Không thấy thư mục {src_dir}")
    else:
        repo = resolve_repo(args.repo, args.config)
        if not repo:
            sys.exit("✗ Không xác định được repo probe. Truyền --repo user/ten-repo "
                     "hoặc đặt $RGEXP_PROBE_REPO, hoặc dùng --local-dir.")
        token = args.token or os.environ.get("HF_TOKEN")
        src_dir = fetch_records_dir(repo, token, args.revision, args.quiet)
        source = f"hf://datasets/{repo}/{RECORD_DIR}"
        if args.revision:
            source += f"@{args.revision}"

    recs, _ = load_records(src_dir, args.limit, args.quiet)
    a = aggregate(recs)
    if a["n_label_mismatch"] and not args.quiet:
        print(f"⚠ {a['n_label_mismatch']} bản ghi có danh sách nhãn khác — đã bỏ "
              f"(trộn hai lần chạy với disease_labels khác nhau?)")
    if a["profile_source"] != "full" and not args.quiet:
        print("⚠ Không có bản ghi `full` nào — bảng chuyên môn là trung bình CÓ "
              "ĐIỀU KIỆN (chỉ trên ảnh router đã chọn expert đó). Đặt "
              "run.probe_full_n rồi chạy lại mode=probe.")

    os.makedirs(args.out, exist_ok=True)
    dump_csvs(a, args.out)
    dump_json(a, args.out, source)
    dump_md(a, args.out, source)
    if not args.no_html:
        dump_html(a, args.out, source)

    if a["absent"] and not args.quiet:
        for x in a["absent"]:
            print(f"⚠ e{x['expert']} KHÔNG vào bảng tổng hợp ({x['reason']}, "
                  f"được chọn {x['n_chosen']:,} lần) — để riêng ở mục "
                  f"'Expert không tham gia'")

    if not args.quiet:
        lvl, _ = verdict(a)
        print(f"✓ {a['n_records']:,} bản ghi ({a['n_full']:,} full / "
              f"{a['n_topk']:,} topk) | {a['n_profiled']}/{a['n_expert']} "
              f"expert có hồ sơ, top-{a['top_k']} | clone_index "
              f"{fmt(a['clone_index'], 3)} ({lvl}) | balance "
              f"{fmt(a['balance'], 3)} | effective_experts "
              f"{fmt(a['effective_experts'], 2)}")
        print(f"✓ ghi đè {os.path.relpath(args.out, ROOT)}/ : "
              f"{'index.html, ' if not args.no_html else ''}SUMMARY.md, "
              f"summary.json, {len(glob.glob(os.path.join(args.out, '*.csv')))} CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

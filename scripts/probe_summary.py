#!/usr/bin/env python3
"""Gộp bản ghi probe trên Hub thành một bản đọc được bằng mắt (CPU, không torch).

`src/probe.py` ghi MỖI ẢNH một file JSON lên dataset repo `<hub_repo>-probe`:
bản đồ không gian 16×16, vector 14 bệnh + 4 aspect, top-10 token, gate router.
Một bản ghi lẻ không nói lên điều gì — nhiễu của một ảnh lớn hơn hiệu ứng cần
đo. Câu hỏi thật sự chỉ trả lời được sau khi gộp cả tập:

    12 expert kia có THẬT SỰ phân hoá không, hay top-k chỉ là k bản sao mờ?

Script này chạy trên CPU máy cá nhân (không cần GPU, không import torch), kéo
records/*.json về, gộp lại và xuất ra:

    reports/probe/index.html        bảng nhiệt + bản đồ 16×16 + top token,
                                    một file tự chứa, mở bằng trình duyệt
    reports/probe/SUMMARY.md        bản đọc thẳng trên GitHub
    reports/probe/summary.json      số liệu máy đọc (để script khác dùng lại)
    reports/probe/*.csv             ma trận thô: usage / disease / aspect /
                                    similarity / top token / spatial

Chạy:

    python scripts/probe_summary.py                      # tự dò repo từ config
    python scripts/probe_summary.py --repo user/abc-probe
    python scripts/probe_summary.py --local-dir outputs/<tag>/probe_records
    python scripts/probe_summary.py --limit 200          # thử nhanh

Chạy lại thì GHI ĐÈ đúng thư mục đó — không tạo bản mới theo timestamp. Lý do:
đây là báo cáo về trạng thái hiện tại của checkpoint, không phải nhật ký; muốn
so hai thời điểm thì dùng `git diff` trên chính thư mục này, và mỗi bản ghi đã
có sẵn `train_samples` / `swa_n` để biết nó chụp lúc nào.

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

═══ Ba con số cần đọc trước ═══

**clone_index** — cosine trung bình giữa hồ sơ bệnh của từng cặp expert, sau
khi đã trừ đi hồ sơ trung bình của cả 12. Trừ trung bình là bắt buộc: mọi
expert đều đẩy logit của mọi tên bệnh lên (chúng cùng nằm trong miền X-quang),
nên cosine thô luôn ~0.9 và không phân biệt được gì. Cái đáng đo là phần LỆCH
so với mặt bằng chung. ~1.0 = các expert là bản sao; ~0 = mỗi expert kéo về một
hướng riêng; âm = phân hoá đối lập nhau.

**balance** — entropy chuẩn hoá của tần suất router chọn từng expert
(H/log E). 1.0 = router chia đều; thấp = vài expert ăn hết lượt chọn và phần
còn lại là tham số chết.

**gate_max trung bình** — router có "dứt khoát" không. Gần 1/top_k nghĩa là
router chia đều trọng số cho k expert được chọn (thực chất là trung bình cộng,
không phải định tuyến).

Ba số này KHÔNG thay cho metrics; chúng trả lời câu "MoE có hoạt động như MoE
không", còn BLEU/BERTScore vẫn nằm ở `src/report.py`.

═══ Expert vắng mặt được ĐỂ RIÊNG, không gộp ═══

Một expert có thể không xuất hiện trong bất kỳ snapshot nào — router chưa bao
giờ chọn nó, hoặc có chọn nhưng bản ghi không kèm vector bệnh. Nó KHÔNG được
đưa vào bảng tổng hợp nào cả, mà liệt kê riêng ở mục "Expert không tham gia"
kèm lý do.

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

NGOẠI LỆ có chủ ý: `balance` và `usage_cv` vẫn tính trên ĐỦ E expert. Chúng đo
router chia việc thế nào, mà "expert này chưa từng được chọn" chính là dữ kiện
quan trọng nhất của câu đó — loại nó ra là tự xoá bằng chứng.
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
        sw = cfg.get("sweep") or {}
        tag = f"{sw.get('model')}-{sw.get('variant')}-n{sw.get('n_block')}"
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


# ═══════════════════════════════ phần gộp số ═════════════════════════════════
def aggregate(recs):
    """Gộp danh sách bản ghi thành các ma trận theo expert.

    Mọi thứ ở đây là TRUNG BÌNH TRÊN NHỮNG ẢNH MÀ EXPERT ĐÓ ĐƯỢC CHỌN, không
    phải trên toàn tập. Chia cho toàn tập sẽ trộn "expert này yếu" với "expert
    này ít được gọi" thành một con số, mà đó là hai kết luận khác hẳn nhau —
    tần suất được gọi đã nằm riêng ở cột `usage`.
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
    nD, nA = len(dis_names), len(asp_names)
    grid = next((tuple(r["grid"]) for r in recs if r.get("grid")), None)
    nS = grid[0] * grid[1] if grid else 0

    n_chosen = np.zeros(E)
    n_top1 = np.zeros(E)
    gate_sum = np.zeros(E)                 # gate thật khi expert được chọn
    gate_all = np.zeros(E)                 # gate trên MỌI ảnh (kể cả = 0)
    dis_sum = np.zeros((E, nD))
    asp_sum = np.zeros((E, nA))
    dnorm_sum = np.zeros(E)
    sp_sum = np.zeros((E, nS)) if nS else None
    spmax_sum = np.zeros(E)
    spl2_sum = np.zeros(E)
    # Mẫu số RIÊNG cho từng đại lượng, không dùng chung n_chosen: một expert có
    # thể được chọn 900 lần mà chỉ 900 lần đó có `disease` (bản ghi cũ thiếu
    # trường, hoặc grid không vuông nên không có `spatial`). Chia tất cả cho
    # n_chosen là lặng lẽ kéo trung bình về 0 theo tỉ lệ phần thiếu.
    n_dis = np.zeros(E)
    n_asp = np.zeros(E)
    n_sp = np.zeros(E)
    cooc = np.zeros((E, E))                # đồng xuất hiện trong cùng top-k
    tok = [defaultdict(lambda: [0, 0.0, ""]) for _ in range(E)]

    ent, gmax, n_lab_mismatch = [], [], 0
    meta = defaultdict(set)
    samples = []

    for r in recs:
        if r.get("disease_labels") and r["disease_labels"] != dis_names:
            n_lab_mismatch += 1
            continue
        for key in ("model", "sweep_cell", "weights"):
            if r.get(key):
                meta[key].add(str(r[key]))
        if r.get("train_samples") is not None:
            samples.append(int(r["train_samples"]))
        router = r.get("router") or {}
        if router.get("gate"):
            g = np.asarray(router["gate"], dtype=np.float64)
            k = min(len(g), E)
            gate_all[:k] += g[:k]
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

        for it in r.get("experts") or []:
            e = int(it.get("expert", -1))
            if not 0 <= e < E:
                continue
            n_chosen[e] += 1
            gate_sum[e] += float(it.get("gate") or 0.0)
            dnorm_sum[e] += float(it.get("delta_norm") or 0.0)
            d = it.get("disease")
            if d and len(d) == nD:
                dis_sum[e] += np.asarray(d, dtype=np.float64)
                n_dis[e] += 1
            a = it.get("aspect")
            if a and len(a) == nA:
                asp_sum[e] += np.asarray(a, dtype=np.float64)
                n_asp[e] += 1
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
    profile = n_dis > 0          # có hồ sơ bệnh -> vào được mọi bảng tổng hợp
    out = {
        "n_records": n,
        "n_label_mismatch": n_lab_mismatch,
        "n_expert": E, "top_k": K,
        "disease_labels": dis_names, "aspect_labels": asp_names,
        "grid": list(grid) if grid else None,
        "meta": {k: sorted(v) for k, v in meta.items()},
        "train_samples": (min(samples), max(samples)) if samples else None,
        "n_chosen": n_chosen, "n_top1": n_top1,
        "n_disease": n_dis, "n_aspect": n_asp, "n_spatial": n_sp,
        "profile": profile,
        "usage": n_chosen / max(n, 1),
        "gate_mean_when_chosen": mean_by(gate_sum, n_chosen),
        "gate_mean_all": gate_all / max(n, 1),
        "disease": mean_by(dis_sum, n_dis),
        "aspect": mean_by(asp_sum, n_asp) if nA else np.zeros((E, 0)),
        "delta_norm": mean_by(dnorm_sum, n_chosen),
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

    Hai lý do khác hẳn nhau nên KHÔNG gộp làm một dòng:

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
    """Chuẩn hoá theo CỘT: mỗi bệnh được so giữa các expert với nhau.

    Cột chứ không phải hàng, vì mặt bằng logit của từng tên bệnh chênh nhau cả
    bậc (tên hiếm gặp trong vocab luôn có Δlogit bé hơn). Chuẩn theo cột thì ô
    đỏ đọc được đúng một nghĩa: "expert này đẩy bệnh này mạnh HƠN các expert
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


def derive(a):
    """Ba chỉ số tổng và các ma trận dẫn xuất — CHỈ trên expert có góp mặt.

    `profile` là mặt nạ những expert thật sự xuất hiện trong ít nhất một bản
    ghi. Mọi thứ ở đây (z-score, cosine, clone_index) đều tính trong nhóm đó và
    trả NaN cho phần còn lại. Trộn một expert vắng mặt vào không phải là "thêm
    một hàng rỗng cho đủ": nó dịch trung bình cột, bóp std, kéo cosine của mọi
    cặp khác — tức là làm sai chính con số mà bảng này sinh ra để trả lời.

    NGOẠI LỆ có chủ ý: `balance` và `usage_cv` vẫn tính trên ĐỦ E expert. Chúng
    đo router chia việc thế nào, mà "expert này chưa từng được chọn" chính là
    dữ kiện quan trọng nhất của câu đó — bỏ nó ra là tự làm đẹp số liệu.
    """
    E = a["n_expert"]
    ok = a["profile"]
    D = a["disease"]
    Z = zscore(D)

    sim = np.full((E, E), np.nan)
    ci = ci_max = 0.0
    if ok.sum() >= 2:
        sub = D[ok]
        C = sub - sub.mean(axis=0, keepdims=True)      # trừ mặt bằng chung
        norm = np.linalg.norm(C, axis=1, keepdims=True)
        Cn = np.divide(C, norm, out=np.zeros_like(C), where=norm > 1e-12)
        S = Cn @ Cn.T
        idx = np.nonzero(ok)[0]
        sim[np.ix_(idx, idx)] = S
        off = S[~np.eye(len(idx), dtype=bool)]
        ci, ci_max = float(off.mean()), float(off.max())

    p = a["n_chosen"] / max(a["n_chosen"].sum(), 1)
    nz = p[p > 0]
    balance = float(-(nz * np.log(nz)).sum() / math.log(E)) if E > 1 else 1.0
    return {
        "disease_z": Z,
        "aspect_z": zscore(a["aspect"]),
        "similarity": sim,
        "n_profiled": int(ok.sum()),
        "clone_index": ci,
        "clone_index_max": ci_max,
        "balance": balance,
        "usage_cv": float(p.std() / p.mean()) if p.mean() > 0 else 0.0,
    }


def top_tokens(a, e, k=12):
    """Top token của một expert: xếp theo TỔNG Δlogit trên cả tập.

    Xếp theo tổng chứ không theo trung bình mỗi lần xuất hiện: một token chỉ
    lọt top-10 đúng một lần với Δlogit khổng lồ là nhiễu của một ảnh, còn token
    xuất hiện ở 800/1000 ảnh với Δlogit vừa phải mới là chuyên môn. Tổng phạt
    cái đầu và thưởng cái sau; số lần xuất hiện vẫn in kèm để tự kiểm.
    """
    n = max(a["n_chosen"][e], 1)
    rows = [(txt, cnt, s / n, cnt / n)
            for cnt, s, txt in a["tokens"][e].values()]
    rows.sort(key=lambda r: -r[2])
    return rows[:k]


def top_labels(vec, names, k=3, floor=0.35):
    """Tên của k giá trị z lớn nhất, bỏ những cái quá gần mặt bằng chung.

    Vector toàn NaN = expert không có hồ sơ; trả rỗng để mọi chỗ hiển thị ra
    "—" thay vì bịa một chuyên môn từ dữ liệu không tồn tại.
    """
    if vec.size == 0 or np.isnan(vec).all():
        return []
    idx = np.argsort(-np.nan_to_num(vec, nan=-np.inf))[:k]
    return [(names[i], float(vec[i])) for i in idx
            if not np.isnan(vec[i]) and vec[i] > floor]


# ═════════════════════════════════ diễn giải ═════════════════════════════════
def verdict(a):
    """Một đoạn tiếng Việt nói thẳng bảng số kia có nghĩa gì.

    Ngưỡng ở đây là quy ước ĐỌC BẢNG, không phải kiểm định thống kê: chúng chỉ
    chia trục cosine thành ba khoảng để câu kết luận không phải viết lại mỗi
    lần. Ranh giới 0.30 / 0.70 chọn theo kinh nghiệm đọc ma trận tương đồng —
    thấy số sát ngưỡng thì đọc ma trận, đừng đọc câu chữ.
    """
    ci, bal, E, K = a["clone_index"], a["balance"], a["n_expert"], a["top_k"]
    P = a["n_profiled"]
    if P < 2:
        return "không đủ dữ liệu", (
            f"Chỉ {P}/{E} expert có mặt trong dữ liệu probe — không so được cặp "
            f"nào, nên clone_index không có nghĩa. Xem mục 'Expert không tham "
            f"gia' để biết vì sao, và chạy lại probe cho đủ tập trước khi đọc "
            f"bảng chuyên môn.")
    if ci > 0.70:
        lvl, msg = "cao", (
            f"Hồ sơ bệnh của {P} expert gần như trùng nhau (cosine lệch trung "
            f"bình {ci:.2f}). Top-{K} đang lấy về {K} bản sao mờ của cùng một "
            f"hàm — phần lớn tham số expert là thừa. Đáng thử: tăng hệ số "
            f"separation loss, hoặc giảm n_expert.")
    elif ci > 0.30:
        lvl, msg = "vừa", (
            f"Các expert phân hoá MỘT PHẦN (cosine lệch trung bình {ci:.2f}): "
            f"có hướng riêng nhưng vẫn chồng lấn nhiều. Đọc ma trận z-score "
            f"bên dưới để xem cặp nào đang trùng.")
    else:
        lvl, msg = "thấp", (
            f"Các expert phân hoá RÕ (cosine lệch trung bình {ci:.2f}): mỗi "
            f"expert kéo hồ sơ bệnh về một hướng khác nhau. Đây là dấu hiệu MoE "
            f"đang thật sự chia việc.")
    if bal < 0.85:
        msg += (f" Nhưng router chia việc LỆCH (balance {bal:.2f}, đều tuyệt "
                f"đối = 1.00): một phần expert hiếm khi được gọi, xem cột "
                f"'lượt chọn'.")
    else:
        msg += f" Router chia lượt khá đều (balance {bal:.2f})."
    if a["gate_max"] is not None and K > 0 and a["gate_max"] < 1.4 / K:
        msg += (f" Gate max trung bình {a['gate_max']:.3f} ≈ 1/top_k "
                f"({1/K:.3f}): router gần như chia đều trọng số cho {K} expert "
                f"được chọn, tức là đang trung bình cộng chứ chưa định tuyến.")
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
    return lvl, msg


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

    # usage giữ ĐỦ E hàng, có cột `active`: đây là bảng về router, và hàng
    # n_chosen = 0 chính là thông tin cần thấy.
    write_csv(os.path.join(out, "expert_usage.csv"),
              ["expert", "active", "n_chosen", "usage", "n_top1", "top1_share",
               "n_disease", "n_spatial", "gate_mean_when_chosen", "gate_mean_all",
               "delta_norm", "spatial_max", "spatial_l2"],
              [[e, int(bool(a["profile"][e])), int(a["n_chosen"][e]),
                num(a["usage"][e]), int(a["n_top1"][e]),
                num(a["n_top1"][e] / max(a["n_records"], 1)),
                int(a["n_disease"][e]), int(a["n_spatial"][e]),
                num(a["gate_mean_when_chosen"][e]), num(a["gate_mean_all"][e]),
                num(a["delta_norm"][e]), num(a["spatial_max"][e]),
                num(a["spatial_l2"][e])]
               for e in range(E)])

    write_csv(os.path.join(out, "expert_absent.csv"),
              ["expert", "reason", "n_chosen", "giải thích"],
              [[x["expert"], x["reason"], x["n_chosen"],
                "không snapshot nào chọn expert này"
                if x["reason"] == "router_never_picked"
                else "được chọn nhưng bản ghi không kèm vector bệnh"]
               for x in a["absent"]])

    # Các ma trận chuyên môn chỉ chứa expert có góp mặt. Một hàng rỗng hay
    # hàng 0 ở đây sẽ bị mọi công cụ đọc CSV coi là số thật.
    for name, M, cols in (("expert_disease_mean", a["disease"], dn),
                          ("expert_disease_z", a["disease_z"], dn),
                          ("expert_aspect_mean", a["aspect"], an),
                          ("expert_aspect_z", a["aspect_z"], an)):
        if M.size:
            write_csv(os.path.join(out, name + ".csv"), ["expert"] + list(cols),
                      [[e] + [num(v) for v in M[e]] for e in live])

    write_csv(os.path.join(out, "expert_similarity.csv"),
              ["expert"] + [f"e{j}" for j in live],
              [[e] + [num(a["similarity"][e][j], 4) for j in live] for e in live])

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
        "n_expert": E, "top_k": a["top_k"],
        "grid": a["grid"],
        "meta": a["meta"],
        "train_samples": a["train_samples"],
        # clone_index/gate_max nói về n_profiled expert; balance nói về đủ E.
        # Ghi cả hai mẫu số ra đây để người đọc JSON không phải đoán.
        "n_profiled": a["n_profiled"],
        "headline": {"clone_index": round(a["clone_index"], 4),
                     "clone_index_max": round(a["clone_index_max"], 4),
                     "clone_index_over": a["n_profiled"],
                     "balance": round(a["balance"], 4),
                     "balance_over": E,
                     "usage_cv": round(a["usage_cv"], 4),
                     "route_entropy": a["route_entropy"],
                     "gate_max": a["gate_max"],
                     "level": lvl, "verdict": msg},
        "absent_experts": a["absent"],
        "disease_labels": a["disease_labels"],
        "aspect_labels": a["aspect_labels"],
        "experts": [
            {"expert": e,
             "in_summary": bool(a["profile"][e]),
             "n_chosen": int(a["n_chosen"][e]),
             "n_disease": int(a["n_disease"][e]),
             "usage": jnum(a["usage"][e]),
             "top1_share": jnum(a["n_top1"][e] / max(a["n_records"], 1)),
             "gate_mean_when_chosen": jnum(a["gate_mean_when_chosen"][e]),
             "delta_norm": jnum(a["delta_norm"][e]),
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
    if z <= 0:
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
    E = a["n_expert"]
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
             ["clone_index", f"{a['clone_index']:.3f}",
              f"cosine lệch trung bình giữa các cặp trong {P} expert có mặt — "
              f"1 = bản sao, 0 = riêng biệt"],
             ["clone_index_max", f"{a['clone_index_max']:.3f}",
              "cặp giống nhau nhất"],
             ["balance", f"{a['balance']:.3f}",
              f"entropy chuẩn hoá của tần suất chọn trên ĐỦ {E} expert — "
              f"1 = router chia đều"],
             ["usage_cv", f"{a['usage_cv']:.3f}",
              "hệ số biến thiên của tần suất chọn — 0 = chia đều"],
             ["route_entropy", f"{a['route_entropy']:.3f}" if a["route_entropy"] is not None else "—",
              "entropy của gate router (nat)"],
             ["gate_max", f"{a['gate_max']:.3f}" if a["gate_max"] is not None else "—",
              f"trọng số lớn nhất trung bình (1/top_k = {1/max(a['top_k'],1):.3f})"],
         ]), ""]

    meta = a["meta"]
    L += ["## Ảnh chụp này là của lúc nào", "",
          md_table(["trường", "giá trị"], [
              ["model", ", ".join(meta.get("model", [])) or "—"],
              ["ô sweep", ", ".join(meta.get("sweep_cell", [])) or "—"],
              ["trọng số", ", ".join(meta.get("weights", [])) or "—"],
              ["n_expert × top_k", f"{E} × {a['top_k']}"],
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
              "(ma trận bệnh/aspect, cosine, top token, bản đồ). Gộp chúng vào "
              "với tư cách một hàng 0 sẽ dịch trung bình cột, bóp std và kéo "
              "cosine của mọi cặp còn lại — tức là làm sai chính con số cần đọc.", "",
              md_table(["expert", "lý do", "lượt được chọn", "nghĩa là"],
                       [[f"e{x['expert']}",
                         f"`{x['reason']}`", f"{x['n_chosen']:,}",
                         "không snapshot nào chọn expert này — tham số chết, "
                         "hoặc bias cân bằng tải (§3.6) chưa kéo nó vào"
                         if x["reason"] == "router_never_picked" else
                         "có chạy thật nhưng bản ghi không kèm vector bệnh — "
                         "lỗ hổng DỮ LIỆU, không phải kết luận về expert"]
                        for x in a["absent"]]), "",
              "`balance` và `usage_cv` ở trên vẫn tính trên đủ "
              f"{E} expert: \"chưa từng được chọn\" chính là dữ kiện quan trọng "
              "nhất của câu hỏi cân bằng tải, bỏ ra là tự làm đẹp số liệu.", ""]

    L += ["## Router chọn ai", "",
          f"Đủ {E} expert. Hàng có `✗` không vào bảng tổng hợp (xem mục trên).", "",
          md_table(["expert", "vào bảng", "lượt chọn", "% ảnh", "top-1", "gate TB",
                    "‖Δh‖ TB", "chuyên môn (z > 0.35)"],
                   [[f"e{e}", "✓" if a["profile"][e] else "✗",
                     f"{int(a['n_chosen'][e]):,}",
                     f"{100*a['usage'][e]:.1f}%",
                     f"{100*a['n_top1'][e]/max(a['n_records'],1):.1f}%",
                     fmt(a["gate_mean_when_chosen"][e], 3),
                     fmt(a["delta_norm"][e], 2),
                     ", ".join(f"{n} ({z:+.1f})"
                               for n, z in top_labels(a["disease_z"][e], dn)) or "—"]
                    for e in range(E)]), ""]

    L += ["## Ma trận expert × bệnh (z-score theo cột)", "",
          f"Chỉ {P} expert có mặt trong dữ liệu. Đậm = expert này đẩy bệnh đó "
          "mạnh hơn mặt bằng các expert khác. `·` = bằng hoặc dưới mặt bằng. "
          "Số thô ở `expert_disease_mean.csv`.", "",
          md_table(["exp"] + list(dn),
                   [[f"e{e}"] + [f"{heat_char(a['disease_z'][e][j])}{a['disease_z'][e][j]:+.1f}"
                                 for j in range(len(dn))] for e in live]), ""]

    if a["aspect"].size:
        an = a["aspect_labels"]
        asp = [e for e in live if a["n_aspect"][e] > 0]
        L += ["## Ma trận expert × aspect (z-score theo cột)", "",
              md_table(["exp"] + list(an),
                       [[f"e{e}"] + [f"{heat_char(a['aspect_z'][e][j])}{a['aspect_z'][e][j]:+.1f}"
                                     for j in range(len(an))] for e in asp]), ""]

    L += ["## Cặp expert giống nhau nhất", ""]
    pairs = sorted(((float(a["similarity"][i][j]), i, j)
                    for i in live for j in live if i < j), reverse=True)
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
              "‖Δh‖ tại từng image token, trung bình trên mọi ảnh mà expert được "
              "chọn, chuẩn hoá về [0,1] theo chính nó. Đây là câu hỏi *nhìn ĐÂU*, "
              "không phải *mạnh cỡ nào* (độ mạnh ở cột ‖Δh‖ phía trên).", "",
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
      --pos:#c0532a;--neg:#2f6690;--acc:#7a5c3e}
@media (prefers-color-scheme:dark){:root{--bg:#16151a;--fg:#eae7e2;--mut:#9a938c;
      --line:#2e2c33;--card:#1e1d23;--pos:#e0764a;--neg:#5b9bc7;--acc:#c9a87c}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:15px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:36px 22px 80px}
h1{font-size:27px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:19px;margin:44px 0 6px;letter-spacing:-.01em}
h2:first-of-type{margin-top:32px}
p.sub{color:var(--mut);margin:0 0 6px;font-size:13.5px}
.note{color:var(--mut);font-size:13.5px;margin:2px 0 14px;max-width:76ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
      padding:16px 18px;margin:14px 0}
.verdict{border-left:3px solid var(--acc)}
.absent{border-left:3px solid var(--neg)}
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

    P.append(f"<h1>Probe chuyên môn expert</h1>")
    P.append(f"<p class=sub>{a['n_records']:,} ảnh · {NP}/{E} expert có mặt · "
             f"top-{K} · "
             f"{esc(', '.join(meta.get('sweep_cell', [])) or 'ô sweep không rõ')} · "
             f"sinh lúc {ts}</p>")
    P.append(f"<p class=sub>Nguồn: <code>{esc(source)}</code> · "
             f"trọng số: {esc(', '.join(meta.get('weights', [])) or '—')}"
             + (f" · train_samples {a['train_samples'][0]:,}–{a['train_samples'][1]:,}"
                if a["train_samples"] else "") + "</p>")

    P.append(f"<div class='card verdict'><b>Mức trùng lặp: {lvl}.</b> {esc(msg)}</div>")

    kpi = [("clone_index", fmt(a["clone_index"]),
            f"trên {NP} expert có mặt · 1 = bản sao"),
           ("balance", fmt(a["balance"]), f"trên đủ {E} expert · 1 = chia đều"),
           ("gate_max", fmt(a["gate_max"]), f"1/top_k = {1/max(K,1):.2f}"),
           ("route_entropy", fmt(a["route_entropy"]),
            f"tối đa = ln {E} = {math.log(E):.2f}" if E > 1 else "—")]
    if a["absent"]:
        kpi.append((f"{E - NP} expert vắng", "⚠", "bị loại khỏi bảng dưới"))
    P.append("<div class=kpis>" + "".join(
        f"<div class=kpi><b>{v}</b><span>{esc(k)}<br>{esc(h)}</span></div>"
        for k, v, h in kpi) + "</div>")

    # ── expert không tham gia ────────────────────────────────────────────────
    if a["absent"]:
        items = []
        for x in a["absent"]:
            why = ("không snapshot nào chọn expert này — tham số chết, hoặc "
                   "bias cân bằng tải chưa kéo nó vào"
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
            f"<p class=note style='margin-bottom:0'><code>balance</code> và "
            f"<code>usage_cv</code> vẫn tính trên đủ {E} expert: \"chưa từng "
            f"được chọn\" chính là dữ kiện quan trọng nhất của câu hỏi cân bằng "
            f"tải, bỏ ra là tự làm đẹp số liệu.</p></div>")

    # ── router chọn ai ───────────────────────────────────────────────────────
    P.append("<h2>Router chọn ai</h2>")
    P.append(f"<p class=note>Đủ {E} expert; hàng mờ là expert không vào bảng "
             "tổng hợp. Thanh = tỉ lệ ảnh mà expert lọt top-k. Chia đều tuyệt "
             f"đối thì mọi thanh dài bằng nhau và bằng top_k/n_expert = "
             f"{K/max(E,1)*100:.0f}%.</p>")
    rows = []
    mx = max(a["usage"].max(), 1e-9)
    for e in range(E):
        w = 128 * a["usage"][e] / mx
        ok = bool(a["profile"][e])
        rows.append(
            f"<tr{'' if ok else ' class=off'}><td>e{e}</td>"
            f"<td>{'✓' if ok else '✗'}</td>"
            f"<td style='text-align:left'><span class=bar style='width:{w:.0f}px'></span>"
            f"{100*a['usage'][e]:.1f}%</td>"
            f"<td>{int(a['n_chosen'][e]):,}</td>"
            f"<td>{100*a['n_top1'][e]/max(a['n_records'],1):.1f}%</td>"
            f"<td>{fmt(a['gate_mean_when_chosen'][e], 3)}</td>"
            f"<td>{fmt(a['delta_norm'][e])}</td>"
            f"<td style='text-align:left'>"
            + esc(", ".join(f"{n} ({z:+.1f})" for n, z in top_labels(a["disease_z"][e], dn)) or "—")
            + "</td></tr>")
    P.append("<div class=scroll><table><tr><th>expert</th><th>vào bảng</th>"
             "<th>% ảnh được chọn</th>"
             "<th>lượt</th><th>là top-1</th><th>gate TB</th><th>‖Δh‖ TB</th>"
             "<th>chuyên môn (z&gt;0.35)</th></tr>" + "".join(rows) + "</table></div>")

    # ── heatmap expert × bệnh / aspect ───────────────────────────────────────
    def heat(M, names, title, note, idx):
        head = "".join(f"<th class=rot><div>{esc(n)}</div></th>" for n in names)
        body = []
        for e in idx:
            cells = "".join(
                f"<td class=h style='background:{col_div(M[e][j])}' "
                f"title='{esc(names[j])} · z={M[e][j]:+.2f}'>{M[e][j]:+.1f}</td>"
                for j in range(len(names)))
            body.append(f"<tr><td>e{e}</td>{cells}</tr>")
        return (f"<h2>{title}</h2><p class=note>{note}</p>"
                f"<div class=scroll><table class=hm><tr><th>exp</th>{head}</tr>"
                + "".join(body) + "</table></div>")

    if len(dn) and live:
        P.append(heat(a["disease_z"], dn, "Expert × bệnh",
                      f"Chỉ {NP} expert có mặt trong dữ liệu. z-score theo CỘT: "
                      "đỏ = expert này đẩy bệnh đó mạnh hơn "
                      "mặt bằng các expert khác, xanh = yếu hơn. Chuẩn theo cột "
                      "vì mặt bằng logit của từng tên bệnh chênh nhau cả bậc. "
                      "Số thô: <code>expert_disease_mean.csv</code>.", live))
    asp = [e for e in live if a["n_aspect"][e] > 0]
    if len(an) and asp:
        P.append(heat(a["aspect_z"], an, "Expert × aspect",
                      f"Cùng cách đọc, trên {len(an)} tên aspect.", asp))

    # ── ma trận tương đồng ───────────────────────────────────────────────────
    P.append("<h2>Expert giống nhau tới đâu</h2>")
    P.append("<p class=note>Cosine giữa hồ sơ bệnh của hai expert SAU KHI trừ "
             "hồ sơ trung bình của cả nhóm. Không trừ thì mọi cặp đều ~0.9 (các "
             "expert cùng sống trong miền X-quang) và ma trận vô nghĩa. "
             "+1 = trùng hướng, 0 = độc lập, −1 = đối lập.</p>")
    head = "".join(f"<th>e{j}</th>" for j in live)
    sim = a["similarity"]
    body = []
    for i in live:
        cells = []
        for j in live:
            if i == j:
                cells.append("<td class=h style='background:transparent'>—</td>")
                continue
            v = float(sim[i][j])
            # cosine nằm trong [-1, 1] còn thang màu nhận z-score cắt ở ±2.5:
            # nhân 2.5 để +1 chạm hết đầu đỏ và −1 chạm hết đầu xanh.
            cells.append(f"<td class=h style='background:{col_div(v * 2.5)}'>"
                         f"{v:+.2f}</td>")
        body.append(f"<tr><td>e{i}</td>{''.join(cells)}</tr>")
    P.append("<div class=scroll><table><tr><th>exp</th>" + head + "</tr>"
             + "".join(body) + "</table></div>")

    # ── thẻ từng expert ──────────────────────────────────────────────────────
    P.append("<h2>Từng expert</h2>")
    P.append("<p class=note>Bản đồ = ‖Δh‖ trung bình tại từng image token "
             "(chuẩn hoá theo chính nó — câu hỏi là <i>nhìn ĐÂU</i>). "
             "Token = unembedding mà expert đẩy lên mạnh nhất trên cả tập.</p>")
    cards = []
    for e in live:
        c = [f"<div class=ex><h3>Expert {e}</h3>",
             f"<div class=m>{100*a['usage'][e]:.1f}% số ảnh · gate TB "
             f"{fmt(a['gate_mean_when_chosen'][e], 3)} · "
             f"‖Δh‖ {fmt(a['delta_norm'][e])}</div>"]
        if a["spatial"] is not None and a["n_spatial"][e] > 0:
            h, w = a["grid"]
            m = a["spatial"][e]
            mx = m.max() or 1.0
            c.append(f"<div class=sp style='grid-template-columns:repeat({w},1fr)'>"
                     + "".join(f"<i class={seq_class(v/mx)}></i>" for v in m)
                     + "</div>")
        td = top_labels(a["disease_z"][e], dn)
        ta = top_labels(a["aspect_z"][e], an) if a["aspect"].size else []
        c.append("<ul>")
        c.append("<li>bệnh: " + (esc(", ".join(f"{n} ({z:+.1f})" for n, z in td))
                                 or "<i>không nổi trội</i>") + "</li>")
        c.append("<li>aspect: " + (esc(", ".join(f"{n} ({z:+.1f})" for n, z in ta))
                                   or "<i>không nổi trội</i>") + "</li>")
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
        print(f"✓ {a['n_records']:,} bản ghi | {a['n_profiled']}/{a['n_expert']} "
              f"expert có mặt, top-{a['top_k']} | clone_index "
              f"{a['clone_index']:.3f} ({lvl}) | balance {a['balance']:.3f}")
        print(f"✓ ghi đè {os.path.relpath(args.out, ROOT)}/ : "
              f"{'index.html, ' if not args.no_html else ''}SUMMARY.md, "
              f"summary.json, {len(glob.glob(os.path.join(args.out, '*.csv')))} CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

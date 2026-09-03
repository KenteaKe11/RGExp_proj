"""§1.8 — Hiệu chuẩn hai ngưỡng `pos / unc / neg`, chạy TRƯỚC infer.

Module này tồn tại để trả lời đúng một câu hỏi: **ô nào của sup_head được
phép gán cứng là "uncertain"?** Trước bản này câu trả lời là hai hằng số
trong config (`cam_pos_thr = 0.5`, `cam_neg_thr = 0.2`) — hai con số không
ai đo, chỉ ước lượng bằng mắt từ chỗ "ASL dìm ô âm xuống thấp". Bản này đo
nó trên tập val, theo TỪNG BỆNH, và kèm một cận dưới thống kê cho độ tinh
khiết của vùng vừa tìm ra.

═══ Vì sao 0.2 / 0.5 sai HẲN kể từ khi uncertain_soft = 0.55 ═══

`asl_loss` không kéo ô uncertain về đúng `uncertain_soft`. Nhánh âm bị
γ⁻ = 4 dập, nên cực tiểu theo p của loss MỘT Ô lệch LÊN so với nhãn đã khai:

    y = 0.00 (neg)  ->  p* = 0.00   (và loss gần như PHẲNG tới p ≈ 0.4)
    y = 0.55 (unc)  ->  p* = 0.680
    y = 0.70        ->  p* = 0.751  (khớp mốc [0.70, 0.80] mà selftest chốt)
    y = 1.00 (pos)  ->  p* = 1.00

Hố uncertain nằm ở 0.68, tức HOÀN TOÀN NGOÀI khoảng (0.2, 0.5). Mọi ô
uncertain được train tử tế sẽ rơi vào nhánh `p ≥ pos_thr` và bị gán cứng
thành "pos", rồi đi thẳng vào danh sách "Present:" của lượt sinh có điều
kiện. Đây là một lỗi IM LẶNG — metrics text vẫn ra số, không có gì kêu.

`asl_prior_band` dựng lại band từ CHÍNH hình học đó: gán mỗi p cho cái hố mà
nó rẻ nhất theo "excess loss" `L(p, y) − min_p L(p, y)`. Với cấu hình đang
chốt (γ⁺ = 0, γ⁻ = 4, m = 0.02, y_unc = 0.55) hai điểm hoà là **0.544** và
**0.850**. Đó là mốc TIÊN NGHIỆM: nó nói band phải nằm ở đâu nếu model học
tới nơi tới chốn, chưa nhìn một mẫu nào. Nó KHÔNG phải band đem đi dùng —
nó là thứ để đối chiếu với band đo được. Hai cái lệch nhau nhiều là một tín
hiệu đọc được: model chưa xuống tới đáy hố, hoặc nhãn bị gập sai vai.

═══ Band đem đi dùng thì phải ĐO ═══

Với mỗi bệnh c, lấy các ô val có nhãn (BỎ `blank` — ô đó đã bị MASK khỏi
loss, model chưa từng được dạy để nói "không đề cập", nên không có cơ sở nào
để chấm nó), rồi tìm cặp (lo, hi) sao cho:

    tối đa hoá   số ô uncertain rơi vào (lo, hi)          ← độ phủ
    ràng buộc    LCB_Wilson( P(unc | p ∈ band) ) ≥ sàn    ← độ tin

`sàn = max(calib_purity, calib_lift × tỉ lệ nền của uncertain ở bệnh đó)`.
Hai vế, vì mỗi vế bắt một kiểu vô nghĩa khác nhau. Sàn tuyệt đối chặn band
"quá nửa là rác". Sàn theo tỉ lệ nền chặn một band tinh khiết GIẢ: nếu
uncertain vốn đã chiếm 45% số ô có nhãn của bệnh đó thì một band purity 50%
gần như không nói thêm được gì so với đoán mò — nó phải hơn nền một khoảng
mới đáng gọi là "vùng đáng tin".

Dùng CẬN DƯỚI Wilson chứ không dùng tỉ lệ thô: purity 5/5 = 100% trên năm ô
không phải một bằng chứng, và một band hẹp bắt trúng vài ô là đúng cái mà
phép tối ưu này sẽ tìm ra nếu không có ai chặn. LCB một phía 95% của 5/5 là
0.478 — tự nó đã loại.

═══ Chọn trên val rồi báo purity trên val là tự lừa ═══

Cặp (lo, hi) được CHỌN bằng cách tối ưu trên chính dữ liệu dùng để chấm nó,
nên purity in ra sẽ lạc quan. Cho nên mỗi bệnh đi qua hai bước:

  1. **Kiểm định** — chia đôi ô val theo tầng (mỗi nhóm neg/unc/pos chia
     50/50 bằng một seed cố định), fit trên nửa A, đo LCB trên nửa B mà
     phép fit CHƯA NHÌN THẤY. Đây mới là con số trung thực, và nó là thứ
     quyết định bệnh đó có được cấp band riêng hay không.
  2. **Triển khai** — fit lại trên TOÀN BỘ ô val để lấy cặp (lo, hi) cuối.
     Dùng hết dữ liệu cho band đem đi chạy, nhưng NIỀM TIN vào nó thì đến
     từ bước 1.

`holdout_lcb` trong bảng và trong JSON là con số bước 1. Đọc nó, đừng đọc
`purity`.

═══ Ba mức rơi, không có mức nào im lặng ═══

    source = "class"   band riêng của bệnh, đã qua kiểm định
    source = "global"  bệnh quá ít ô uncertain (< calib_min_unc) hoặc trượt
                       kiểm định -> dùng band gộp của cả 14 bệnh
    source = "none"    cả band gộp cũng không đạt sàn -> KHÔNG gán uncertain
                       cho bệnh này. lo = hi = tau_split (điểm cắt pos/neg
                       theo Youden J), tức quay về quyết định nhị phân.

"none" là một kết quả, không phải một lỗi: nó nói thẳng rằng với checkpoint
này, ở bệnh này, xác suất của sup_head không tách được uncertain ra khỏi
pos/neg đủ sạch để dám gán cứng. Thà biết điều đó trước khi infer.

═══ Band gắn với CHECKPOINT, không gắn với bài toán ═══

Đổi trọng số là band cũ hết hiệu lực — phân phối p dịch sau mỗi lần train
thêm. `calib_bands.json` vì thế ghi kèm `train_samples` + `weights`, và
`Bands.check_stale` kêu lên khi nạp một file được hiệu chuẩn cho checkpoint
khác. Rẻ: một lượt quét val bằng `router.sup_forward` (nhánh ngắn 39M tham
số, không đụng decoder) — vài chục giây cho vài nghìn ảnh.

═══ Ai đọc kết quả này ═══

    src/train.py   -> lượt sinh CÓ ĐIỀU KIỆN (condition_pass) và cột
                      cam_state của reports.csv
    src/cam.py     -> state trong khối cam của bản ghi probe
    src/probe.py   -> cam_block
    src/metrics.py -> khối meta của metrics.json

Tất cả nhận cùng một cặp vector [14], nên một ảnh chỉ có MỘT cách đọc — đúng
nguyên tắc đã có sẵn trong src/cam.py.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from .losses import aspect_roles

# Mã nhóm nội bộ. Không lưu xuống đĩa ở đâu cả — JSON ghi bằng TÊN.
NEG, UNC, POS = 0, 1, 2
GROUPS = ("neg", "unc", "pos")

# z một phía cho cận dưới Wilson.
_Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def z_for(conf):
    """Mức tin cậy -> z một phía. Mức lạ thì nội suy từ hàm ngược của Φ."""
    conf = float(conf)
    if conf in _Z:
        return _Z[conf]
    if not 0.5 < conf < 1.0:
        raise ValueError(f"calib_conf phải trong (0.5, 1), nhận {conf}")
    # Acklam/Moro rút gọn là thừa ở đây: chỉ cần một nghịch đảo đủ tốt.
    from math import erf, sqrt
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < conf:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ═════════════════════ (1) hình học của ASL — mốc tiên nghiệm ════════════════
def asl_cell_loss(p, y, gamma_pos=0.0, gamma_neg=4.0, clip=0.02, eps=1e-8):
    """ASL của MỘT ô, bản numpy. Phải khớp `src/losses.py::asl_loss`.

    Chép công thức chứ không gọi lại hàm torch vì hàm kia đã gộp mask +
    trung bình theo batch; ở đây cần đúng đường cong theo p của một ô. Hai
    bản lệch nhau là band tiên nghiệm nói về một loss không ai train — nên
    `scripts/selftest.py::test_calib` so hai bản trên một lưới p.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    y = float(y)
    p_neg = np.minimum(1.0 - p + float(clip), 1.0)
    w_pos = np.power(1.0 - p, gamma_pos) if gamma_pos > 0 else 1.0
    w_neg = (np.power(np.clip(1.0 - p_neg, 0.0, None), gamma_neg)
             if gamma_neg > 0 else 1.0)
    return -(y * w_pos * np.log(p)
             + (1.0 - y) * w_neg * np.log(np.maximum(p_neg, eps)))


def _fine_grid(n=200001):
    return np.linspace(1e-5, 1.0 - 1e-5, int(n))


def asl_well(y, gamma_pos=0.0, gamma_neg=4.0, clip=0.02, n=200001):
    """Đáy hố thế năng: argmin_p ASL(p, y). KHÔNG bằng y."""
    g = _fine_grid(n)
    return float(g[int(np.argmin(asl_cell_loss(g, y, gamma_pos, gamma_neg,
                                               clip)))])


def asl_prior_band(uncertain_soft, gamma_pos=0.0, gamma_neg=4.0, clip=0.02,
                   n=200001):
    """(lo, hi) suy TỪ LOSS, chưa nhìn dữ liệu.

    Gán mỗi p cho hố rẻ nhất theo "excess loss" — `L(p, y) − min L(·, y)` —
    tức là: nếu chỉ nhìn con số p này, nhãn nào giải thích nó với ít phần dư
    nhất? Ba hố là y ∈ {0, uncertain_soft, 1}. Vùng mà hố uncertain thắng là
    band tiên nghiệm.

    Dùng excess chứ không dùng L thô: L thô giảm đơn điệu theo y ở gần như
    mọi p, nên so trực tiếp thì y = 1 luôn thắng và band rỗng.
    """
    g = _fine_grid(n)
    d = []
    for y in (0.0, float(uncertain_soft), 1.0):
        v = asl_cell_loss(g, y, gamma_pos, gamma_neg, clip)
        d.append(v - v.min())
    lab = np.argmin(np.stack(d, axis=0), axis=0)
    hit = lab == UNC
    if not hit.any():
        return float("nan"), float("nan")
    return float(g[int(np.argmax(hit))]), float(g[len(g) - 1
                                                  - int(np.argmax(hit[::-1]))])


# ═══════════════════════ (2) cận dưới Wilson một phía ════════════════════════
def wilson_lcb(k, n, z=1.6449):
    """Cận dưới Wilson một phía cho tỉ lệ k/n.

    Wald (`p̂ ± z·√(p̂(1−p̂)/n)`) sập ở đúng chỗ cần dùng: p̂ = 1 cho bề rộng
    0, nên 5/5 thành "100%, chắc chắn". Wilson kéo tâm về 1/2 theo n và
    không bao giờ cho bề rộng 0 — 5/5 ra 0.649, 50/50 ra 0.949.
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    nn = np.maximum(n, 1.0)
    ph = k / nn
    z2 = z * z
    den = 1.0 + z2 / nn
    cen = ph + z2 / (2.0 * nn)
    rad = z * np.sqrt(np.maximum(ph * (1.0 - ph) / nn + z2 / (4.0 * nn * nn),
                                 0.0))
    out = np.clip((cen - rad) / den, 0.0, 1.0)
    out = np.where(n > 0, out, 0.0)
    return float(out) if np.ndim(out) == 0 else out


# ══════════════════════ (3) quét cặp ngưỡng trên một nhóm ô ══════════════════
def _edges(grid):
    return np.linspace(0.0, 1.0, int(grid) + 1)


def _prefix(p, edges):
    """Đếm tích luỹ theo cạnh lưới: c[k] = số ô có p < edges[k]."""
    cnt, _ = np.histogram(np.asarray(p, dtype=np.float64), bins=edges)
    return np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)


def _in_band(p, lo, hi):
    p = np.asarray(p, dtype=np.float64)
    return (p >= lo) & (p < hi)


def search_band(p, g, *, floor, z, min_cells, edges, pos_cap=0.0):
    """Cặp (lo, hi) phủ NHIỀU ô uncertain nhất mà vẫn đạt sàn purity.

    Vét cạn mọi cặp cạnh lưới bằng phép trừ tích luỹ — với `calib_grid`
    mặc định 1000 thì đó là ~500k cặp, ba ma trận 1001×1001 int64, dưới
    30MB và xong trong chớp mắt. Không có heuristic nào ở đây: band tối ưu
    theo ràng buộc là band tối ưu, không phải band mà một vòng lặp tham lam
    dừng lại ở.

    Thứ tự ưu tiên khi hoà: phủ nhiều nhất -> LCB cao nhất -> HẸP nhất. Vế
    cuối quan trọng: hai band phủ y hệt nhau thì band hẹp hơn nuốt ít ô
    pos/neg hơn ở hai rìa.

    `pos_cap > 0` thêm một ràng buộc thứ hai: band không được nuốt quá ngần
    ấy phần ô POSITIVE thật. Ô pos bị đẩy sang "unc" là một ca bệnh rút khỏi
    danh sách "Present:" của prompt — mà tối đa hoá độ phủ uncertain thì
    luôn muốn nới rìa trên xuống chỗ có ô pos. Mặc định TẮT (0.0): chi phí
    này vẫn được đo và in ra ở cột `pos→unc`, bật trần khi đã nhìn con số đó.

    Trả None nếu không cặp nào đạt sàn — chỗ gọi phải xử lý, đừng nới sàn.
    """
    pu, pp, pn = p[g == UNC], p[g == POS], p[g == NEG]
    if pu.size == 0:
        return None
    cu, cp, cn = _prefix(pu, edges), _prefix(pp, edges), _prefix(pn, edges)
    n_u = cu[None, :] - cu[:, None]
    n_o = (cp[None, :] - cp[:, None]) + (cn[None, :] - cn[:, None])
    n_t = n_u + n_o
    up = np.triu(np.ones(n_u.shape, dtype=bool), 1)
    lcb = wilson_lcb(n_u, n_t, z)
    ok = up & (n_t >= int(min_cells)) & (n_u > 0) & (lcb >= float(floor))
    if float(pos_cap) > 0 and pp.size:
        ok &= (cp[None, :] - cp[:, None]) <= float(pos_cap) * pp.size
    if not ok.any():
        return None
    score = np.where(ok, n_u, -1)
    cand = ok & (score == score.max())
    l2 = np.where(cand, lcb, -1.0)
    cand &= l2 >= l2.max() - 1e-12
    ii, jj = np.nonzero(cand)
    k = int(np.argmin(jj - ii))
    i, j = int(ii[k]), int(jj[k])
    return {"lo": float(edges[i]), "hi": float(edges[j])}


def eval_band(lo, hi, p, g, z):
    """Chấm một band ĐÃ CHO trên một bộ ô. Dùng cho cả nửa holdout."""
    inb = _in_band(p, lo, hi)
    n_u = int((inb & (g == UNC)).sum())
    n_p = int((inb & (g == POS)).sum())
    n_n = int((inb & (g == NEG)).sum())
    n_t = n_u + n_p + n_n
    tot_u = int((g == UNC).sum())
    out_hi = (p >= hi)
    out_lo = (p < lo)
    return {
        "lo": float(lo), "hi": float(hi),
        "n_band": n_t, "n_unc_in": n_u, "n_pos_in": n_p, "n_neg_in": n_n,
        "purity": round(n_u / n_t, 4) if n_t else 0.0,
        "purity_lcb": round(float(wilson_lcb(n_u, max(n_t, 1), z)), 4)
        if n_t else 0.0,
        "coverage": round(n_u / tot_u, 4) if tot_u else 0.0,
        # Hai con số này KHÔNG vào ràng buộc, nhưng phải đọc: band càng rộng
        # càng nuốt ô pos thật, mà ô pos bị đẩy sang "unc" là một ca bệnh
        # rút khỏi danh sách "Present:" của prompt.
        "pos_swallowed": round(float((inb & (g == POS)).sum()
                                     / max(int((g == POS).sum()), 1)), 4),
        "neg_swallowed": round(float((inb & (g == NEG)).sum()
                                     / max(int((g == NEG).sum()), 1)), 4),
        # Độ sạch của HAI VÙNG NGOÀI — thứ quyết định chất lượng của chính
        # danh sách present/absent, band chỉ là phần ở giữa.
        "pos_purity_out": round(float((out_hi & (g == POS)).sum()
                                      / max(int(out_hi.sum()), 1)), 4),
        "neg_purity_out": round(float((out_lo & (g == NEG)).sum()
                                      / max(int(out_lo.sum()), 1)), 4),
        "n_pos_region": int(out_hi.sum()), "n_neg_region": int(out_lo.sum()),
    }


def youden_split(p, g, edges):
    """Điểm cắt pos/neg tốt nhất khi KHÔNG có band (chỉ nhìn ô pos và neg).

    Youden J = TPR − FPR, cực đại. Đây là đường lùi cho `source = "none"`:
    bệnh nào không tách được uncertain thì vẫn phải trả lời present/absent,
    và trả lời bằng 0.5 mặc định là bỏ qua chuyện phân phối p của bệnh đó
    lệch đi đâu.
    """
    pp, pn = p[g == POS], p[g == NEG]
    if pp.size == 0 or pn.size == 0:
        return None
    cp, cn = _prefix(pp, edges), _prefix(pn, edges)
    tpr = 1.0 - cp / float(pp.size)      # tỉ lệ ô pos có p ≥ edges[k]
    fpr = 1.0 - cn / float(pn.size)
    return float(edges[int(np.argmax(tpr - fpr))])


# ═══════════════ (4) fit một nhóm ô: kiểm định rồi mới triển khai ════════════
def _stratified_half(g, rng):
    """Mặt nạ nửa A, chia 50/50 TRONG TỪNG nhóm.

    Chia ngẫu nhiên toàn cục thì ở một bệnh hiếm, nửa B có thể không còn ô
    uncertain nào và bước kiểm định trở thành một phép chia cho 0 đội lốt
    một con số.
    """
    m = np.zeros(g.shape, dtype=bool)
    for grp in (NEG, UNC, POS):
        idx = np.nonzero(g == grp)[0]
        if idx.size == 0:
            continue
        rng.shuffle(idx)
        m[idx[: idx.size // 2]] = True
    return m


def fit_one(p, g, *, purity, lift, z, min_cells, holdout, seed, edges,
            margin=0.04, pos_cap=0.0, label=""):
    """Fit band cho MỘT nhóm ô (một bệnh, hoặc gộp cả 14).

    ═══ Vì sao CHỌN ở sàn + biên nhưng CHẤM ở sàn ═══

    Phép tối ưu "phủ nhiều nhất mà vẫn đạt sàn" luôn dừng lại ĐÚNG TRÊN biên
    ràng buộc — nới thêm một nấc nữa là rớt sàn, nên band được chọn có
    purity thật xấp xỉ đúng bằng sàn. Đem một band như thế đi chấm lại trên
    nửa B ở CÙNG một sàn thì kết quả là tung đồng xu: nửa số bệnh tách tốt
    vẫn trượt vì nhiễu lấy mẫu, không vì band xấu.

    Nên `margin` đẩy sàn LÚC CHỌN lên `sàn + margin`, còn lúc CHẤM trên nửa
    B thì vẫn là `sàn`. Band đem đi dùng vì thế hơi hẹp hơn mức tối đa lý
    thuyết — đổi lại lời hứa "purity ≥ sàn, đo trên dữ liệu chưa từng tham
    gia chọn band" là một lời hứa thật, không phải một kết quả của phép chọn.

    Trả về dict luôn có các khoá `lo`/`hi`/`ok`; `ok = False` nghĩa là không
    có band nào đạt sàn và chỗ gọi phải rơi xuống mức tiếp theo.
    """
    p = np.asarray(p, dtype=np.float64)
    g = np.asarray(g, dtype=np.int64)
    n = int(p.size)
    n_unc = int((g == UNC).sum())
    base = float(n_unc / n) if n else 0.0
    floor = max(float(purity), float(lift) * base)
    floor_fit = min(floor + float(margin), 0.999)
    out = {"label": str(label), "n": n, "n_unc": n_unc,
           "n_pos": int((g == POS).sum()), "n_neg": int((g == NEG).sum()),
           "base_rate": round(base, 4), "floor": round(floor, 4),
           "floor_fit": round(floor_fit, 4),
           "ok": False, "lo": None, "hi": None, "holdout": None,
           "tau_split": youden_split(p, g, edges)}
    if n_unc == 0:
        out["note"] = "không có ô uncertain nào -> không có gì để hiệu chuẩn"
        return out

    # ── bước 1: kiểm định trên nửa CHƯA NHÌN ────────────────────────────────
    if float(holdout) > 0 and n_unc >= 4:
        rng = np.random.default_rng(int(seed))
        a = _stratified_half(g, rng)
        b = ~a
        fit_a = search_band(p[a], g[a], floor=floor_fit, z=z,
                            min_cells=max(int(min_cells) // 2, 5), edges=edges,
                            pos_cap=pos_cap)
        if fit_a is None:
            out["note"] = (f"nửa A không tìm được band nào đạt sàn chọn "
                           f"{floor_fit:.3f} -> band trên toàn bộ val (nếu "
                           f"có) chỉ là trùng hợp của phép tối ưu")
            return out
        hold = eval_band(fit_a["lo"], fit_a["hi"], p[b], g[b], z)
        hold["fit_lo"], hold["fit_hi"] = fit_a["lo"], fit_a["hi"]
        hold["passed"] = bool(hold["purity_lcb"] >= floor)
        out["holdout"] = hold
        if not hold["passed"]:
            out["note"] = (f"band của nửa A ({fit_a['lo']:.3f}, "
                           f"{fit_a['hi']:.3f}) chỉ đạt LCB "
                           f"{hold['purity_lcb']:.3f} < sàn {floor:.3f} trên "
                           f"nửa B -> KHÔNG đáng tin")
            return out

    # ── bước 2: fit lại trên TOÀN BỘ để lấy cặp đem đi dùng ─────────────────
    full = search_band(p, g, floor=floor_fit, z=z, min_cells=min_cells,
                       edges=edges, pos_cap=pos_cap)
    if full is None:
        out["note"] = (f"toàn bộ val không có band nào đạt sàn chọn "
                       f"{floor_fit:.3f}")
        return out
    out.update(eval_band(full["lo"], full["hi"], p, g, z))
    out["ok"] = True
    return out


# ═════════════════════════ (5) kết quả: lớp Bands ════════════════════════════
class Bands:
    """Hai vector [14] + toàn bộ sổ sách để đọc lại vì sao chúng là thế.

    `lo`/`hi` là thứ duy nhất pipeline dùng; `rows` là thứ người đọc dùng.
    Giữ chung một chỗ để không bao giờ có chuyện band đang chạy và bảng đang
    đọc đến từ hai lần hiệu chuẩn khác nhau.
    """

    VERSION = 1

    def __init__(self, lo, hi, rows, meta=None, glob=None):
        self.lo = np.asarray(lo, dtype=np.float64).reshape(-1)
        self.hi = np.asarray(hi, dtype=np.float64).reshape(-1)
        if self.lo.shape != self.hi.shape:
            raise ValueError(f"lo {self.lo.shape} != hi {self.hi.shape}")
        bad = np.nonzero(self.lo > self.hi)[0]
        if bad.size:
            raise ValueError(
                f"lo > hi ở lớp {bad.tolist()} — band âm là cấu hình vô "
                f"nghĩa (src/cam.py::state_of cũng sẽ nổ ở đó)")
        self.rows = list(rows or [])
        self.meta = dict(meta or {})
        self.glob = dict(glob or {})

    def __len__(self):
        return int(self.lo.size)

    def vectors(self):
        """-> (pos_thr, neg_thr) đúng quy ước của `src/cam.py::state_of`.

        state_of đọc band là khoảng MỞ (neg_thr, pos_thr), nên neg_thr = lo
        và pos_thr = hi. Trả bản sao: không ai được sửa band tại chỗ giữa
        chừng một phiên infer.
        """
        return self.hi.copy(), self.lo.copy()

    def width(self):
        return self.hi - self.lo

    @property
    def n_none(self):
        return sum(1 for r in self.rows if r.get("source") == "none")

    def as_dict(self):
        return {"version": self.VERSION, "meta": self.meta,
                "global": self.glob, "classes": self.rows}

    def save(self, path):
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)          # ghi nguyên tử: không để lại file cụt
        return path

    @classmethod
    def load(cls, path):
        with open(str(path), encoding="utf-8") as fh:
            d = json.load(fh)
        if int(d.get("version", 0)) != cls.VERSION:
            raise ValueError(f"{path}: version {d.get('version')} != "
                             f"{cls.VERSION} — hiệu chuẩn lại thay vì đoán")
        rows = d["classes"]
        return cls([r["lo"] for r in rows], [r["hi"] for r in rows], rows,
                   d.get("meta"), d.get("global"))

    def check_stale(self, train_samples, weights, warn=None):
        """Band gắn với CHECKPOINT. Đổi trọng số là band cũ hết hiệu lực."""
        msgs = []
        was = self.meta.get("train_samples")
        if was is not None and train_samples is not None \
                and int(was) != int(train_samples):
            msgs.append(f"hiệu chuẩn ở {int(was)} mẫu train, checkpoint này "
                        f"{int(train_samples)} mẫu")
        if self.meta.get("weights") and weights \
                and self.meta["weights"] != weights:
            msgs.append(f"hiệu chuẩn cho weights={self.meta['weights']}, "
                        f"đang chạy {weights}")
        if msgs and warn is not None:
            warn("calib", "calib_bands.json ĐÃ CŨ: " + "; ".join(msgs)
                 + ". Phân phối p dịch sau mỗi lần train thêm, nên band cũ "
                   "không còn giữ được purity đã hứa. Chạy lại với "
                   "--set run.calib_refit=true.")
        return not msgs

    def table(self):
        """Bảng để đọc bằng mắt trong log. Cột holdout_lcb mới là cột thật."""
        head = (f"{'bệnh':<26}{'band':>15}{'src':>7}{'nền':>7}{'phủ':>7}"
                f"{'purity':>8}{'hold':>7}{'pos→unc':>9}{'n_unc':>7}")
        out = [head, "─" * len(head)]
        for r in self.rows:
            h = r.get("holdout") or {}
            band = ("—  (nhị phân)" if r.get("source") == "none"
                    else f"{r['lo']:.3f}–{r['hi']:.3f}")
            hold = (f"{h['purity_lcb']:.3f}" if h.get("purity_lcb")
                    is not None else "—")
            out.append(
                f"{str(r['name'])[:25]:<26}{band:>15}"
                f"{str(r.get('source', '?')):>7}"
                f"{r.get('base_rate', 0):>7.3f}{r.get('coverage', 0):>7.3f}"
                f"{r.get('purity', 0):>8.3f}{hold:>7}"
                f"{r.get('pos_swallowed', 0):>9.3f}{r.get('n_unc', 0):>7d}")
        return "\n".join(out)


def default_bands(n_class, pos_thr, neg_thr, note="config"):
    """Band hằng số từ config — đường dùng khi hiệu chuẩn bị TẮT.

    Có mặt để phần còn lại của pipeline chỉ biết MỘT kiểu đối tượng, không
    phải rẽ nhánh "có calib hay không" ở bốn chỗ khác nhau.

    `note` ghi lại band này TỪ ĐÂU ra — "config" (hai hằng số cũ) hay "prior"
    (suy từ hình dạng ASL khi không quét được val). Nó đi vào meta của
    cam.npz, nên một bản ghi cũ vẫn nói được nó đã gập nhãn bằng gì.
    """
    n = int(n_class)
    rows = [{"name": f"#{i}", "lo": float(neg_thr), "hi": float(pos_thr),
             "source": note, "n": 0, "n_unc": 0} for i in range(n)]
    return Bands(np.full(n, float(neg_thr)), np.full(n, float(pos_thr)), rows,
                 {"source": note})


# ═══════════════════════ (6) fit đủ 14 bệnh + band gộp ═══════════════════════
def group_codes(Y, aspect_names):
    """Y [N, 14] chỉ số aspect -> mã nhóm [N, 14] (blank/thiếu nhãn = −1).

    Tra vai theo TÊN qua `losses.aspect_roles` — cùng một hàm mà loss dùng.
    Hardcode 0/1/2/3 ở đây là cách chắc chắn nhất để hiệu chuẩn một thứ và
    train một thứ khác mà không ai biết.
    """
    roles = aspect_roles(aspect_names)
    lut = np.full(4, -1, dtype=np.int64)
    lut[roles["negative"]] = NEG
    lut[roles["uncertain"]] = UNC
    lut[roles["positive"]] = POS
    # roles["blank"] giữ nguyên −1: ô blank đã bị MASK khỏi ASL, model chưa
    # bao giờ được dạy để nói "không đề cập" -> không có cơ sở chấm nó.
    Y = np.asarray(Y)
    return np.where(Y >= 0, lut[np.clip(Y, 0, 3)], -1)


def _global_holds(gfit, p_c, g_c, z, floor):
    """Band GỘP có còn sạch khi đem áp lên RIÊNG bệnh này không?

    Không có bước này thì một bệnh mà uncertain chỉ chiếm 3-4% số ô có nhãn
    vẫn "được" band gộp, và mọi ô rơi vào đó bị gán cứng uncertain với độ
    tinh khiết ~15% — tức sai 6 lần trên 7. Rơi về band gộp là một đường
    lùi, không phải một giấy phép: nó vẫn phải qua đúng cái sàn của bệnh đó.
    """
    ev = eval_band(gfit["lo"], gfit["hi"], p_c, g_c, z)
    return ev["n_band"] > 0 and ev["purity_lcb"] >= float(floor)


def fit_bands(P, Y, aspect_names, disease_names, *, uncertain_soft,
              asl_gamma_pos=0.0, asl_gamma_neg=4.0, asl_clip=0.02,
              purity=0.60, lift=1.5, conf=0.95, min_unc=200, min_cells=30,
              holdout=0.5, seed=0, grid=1000, margin=0.04, pos_cap=0.0,
              meta=None, log=None):
    """P [N, 14] xác suất + Y [N, 14] aspect  ->  Bands."""
    P = np.asarray(P, dtype=np.float64)
    G = group_codes(Y, aspect_names)
    if P.shape != G.shape:
        raise ValueError(f"P {P.shape} != Y {G.shape}")
    n_class = P.shape[1]
    z = z_for(conf)
    edges = _edges(grid)
    names = [str(disease_names[i]) if i < len(disease_names) else f"#{i}"
             for i in range(n_class)]

    well = asl_well(uncertain_soft, asl_gamma_pos, asl_gamma_neg, asl_clip)
    pri_lo, pri_hi = asl_prior_band(uncertain_soft, asl_gamma_pos,
                                    asl_gamma_neg, asl_clip)

    keep = G >= 0
    gfit = fit_one(P[keep], G[keep], purity=purity, lift=lift, z=z,
                   min_cells=min_cells, holdout=holdout, seed=seed,
                   edges=edges, margin=margin, pos_cap=pos_cap,
                   label="global")
    gfit["name"] = "GỘP 14 bệnh"
    if log is not None:
        log("calib", f"band GỘP: " + (
            f"({gfit['lo']:.3f}, {gfit['hi']:.3f}) purity="
            f"{gfit['purity']:.3f} (LCB holdout="
            f"{(gfit.get('holdout') or {}).get('purity_lcb', float('nan')):.3f}"
            f") trên {gfit['n']} ô" if gfit["ok"]
            else f"KHÔNG đạt sàn {gfit['floor']:.3f} — {gfit.get('note', '')}"))

    lo_v, hi_v, rows = [], [], []
    for c in range(n_class):
        k = G[:, c] >= 0
        p_c, g_c = P[k, c], G[k, c]
        row = fit_one(p_c, g_c, purity=purity, lift=lift, z=z,
                      min_cells=min_cells, holdout=holdout, seed=seed,
                      edges=edges, margin=margin, pos_cap=pos_cap,
                      label=names[c])
        row["name"] = names[c]
        row["index"] = int(c)

        if row["ok"] and row["n_unc"] >= int(min_unc):
            row["source"] = "class"
        elif gfit["ok"] and _global_holds(gfit, p_c, g_c, z, row["floor"]):
            reason = ("quá ít ô uncertain "
                      f"({row['n_unc']} < {int(min_unc)})"
                      if row["n_unc"] < int(min_unc)
                      else row.get("note", "không đạt sàn"))
            row.update(eval_band(gfit["lo"], gfit["hi"], p_c, g_c, z))
            row["source"] = "global"
            # Cột `hold` của hàng này là con số của BAND GỘP (đo trên nửa B
            # gộp), không phải của phép fit riêng đã trượt ở trên — đó mới là
            # thứ đang bảo lãnh cho band đang dùng.
            row["holdout"] = gfit.get("holdout")
            row["note"] = f"dùng band gộp — {reason}"
        else:
            tau = row["tau_split"]
            if tau is None:
                tau = gfit["tau_split"]
            if tau is None:
                tau = 0.5
            row["lo"] = row["hi"] = float(tau)
            row["source"] = "none"
            why = row.get("note") or "không đạt sàn"
            if gfit["ok"]:
                why += " (band gộp cũng không đủ sạch trên riêng bệnh này)"
            row["note"] = (f"không gán uncertain cho bệnh này — {why}; quay "
                           f"về quyết định nhị phân tại {float(tau):.3f}")
        row["well_in_band"] = bool(row["lo"] <= well <= row["hi"])
        lo_v.append(row["lo"])
        hi_v.append(row["hi"])
        rows.append(row)

    m = dict(meta or {})
    m.update({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_val_images": int(P.shape[0]),
        "n_cells_labelled": int(keep.sum()),
        "n_cells_blank": int((~keep).sum()),
        "params": {"purity": float(purity), "lift": float(lift),
                   "conf": float(conf), "z": round(float(z), 4),
                   "min_unc": int(min_unc), "min_cells": int(min_cells),
                   "holdout": float(holdout), "seed": int(seed),
                   "grid": int(grid), "margin": float(margin),
                   "max_pos_swallow": float(pos_cap)},
        "asl": {"uncertain_soft": float(uncertain_soft),
                "gamma_pos": float(asl_gamma_pos),
                "gamma_neg": float(asl_gamma_neg), "clip": float(asl_clip)},
        # Mốc tiên nghiệm: band suy TỪ LOSS, chưa nhìn dữ liệu. Lệch nhiều so
        # với band đo được là tín hiệu, không phải sai số.
        "prior": {"well": round(well, 4),
                  "band": [round(pri_lo, 4), round(pri_hi, 4)]},
        "aspect_names": [str(a) for a in aspect_names],
    })
    return Bands(lo_v, hi_v, rows, m, gfit)


# ══════════════════════ (7) quét val bằng nhánh sup_head ═════════════════════
def collect_sup_probs(router, loader, device, log=None, max_batches=0,
                      pixel_key="pixel_values", label_key="sup_labels"):
    """-> (P [N, 14] xác suất, Y [N, 14] chỉ số aspect).

    Đi qua `router.sup_forward` — nhánh NGẮN dừng ở sup_head, không đụng
    decoder Gemma, không dựng prefix. Vài chục giây cho vài nghìn ảnh, tức
    rẻ hơn một lần generate của một ảnh duy nhất.

    Loader phải là loader của tập VAL và phải có `sup_labels` (tức frame đã
    đi qua `data.attach_labels`). Hiệu chuẩn trên tập test là tự chấm bài
    mình; hiệu chuẩn trên train là đo một phân phối mà model đã thuộc lòng.
    """
    import torch

    was_train = router.training
    router.eval()
    P, Y = [], []
    try:
        with torch.no_grad():
            for i, b in enumerate(loader):
                if max_batches and i >= int(max_batches):
                    break
                if label_key not in b:
                    raise KeyError(
                        f"batch không có '{label_key}' — frame chưa qua "
                        f"attach_labels, không có nhãn thì không hiệu chuẩn "
                        f"được gì cả")
                px = b[pixel_key].to(device, non_blocking=True)
                logits, _ = router.sup_forward(px.float())
                P.append(torch.sigmoid(logits.float()).cpu().numpy())
                Y.append(b[label_key].cpu().numpy())
                if log is not None and i and i % 50 == 0:
                    log("calib", f"quét val… {sum(len(x) for x in P)} ảnh")
    finally:
        if was_train:
            router.train()
    if not P:
        raise RuntimeError("loader val rỗng — không có gì để hiệu chuẩn")
    return (np.concatenate(P, axis=0).astype(np.float64),
            np.concatenate(Y, axis=0).astype(np.int64))


# ═══════════════════ (8) một cửa cho src/train.py gọi ════════════════════════
def _report(bands, log, warn, cfg_pos, cfg_neg):
    """In bảng + ba dòng đối chiếu đáng đọc nhất."""
    m, g = bands.meta, bands.glob
    pri = m.get("prior", {})
    well = pri.get("well")
    plo, phi = (pri.get("band") or [None, None])[:2]
    log("calib", f"hố uncertain của ASL: p* = {well:.3f} "
                 f"(uncertain_soft = {m['asl']['uncertain_soft']:g} — γ⁻ = "
                 f"{m['asl']['gamma_neg']:g} kéo đáy hố LỆCH LÊN so với nhãn "
                 f"đã khai)")
    log("calib", f"band TIÊN NGHIỆM suy từ loss: ({plo:.3f}, {phi:.3f}) — "
                 f"chưa nhìn một mẫu nào")
    if not (plo <= cfg_pos and cfg_neg <= phi and cfg_neg <= well <= cfg_pos):
        warn("calib",
             f"hai hằng số cũ trong config (cam_neg_thr={cfg_neg:g}, "
             f"cam_pos_thr={cfg_pos:g}) KHÔNG chứa hố uncertain p*="
             f"{well:.3f}: với hai ngưỡng đó, mọi ô uncertain được train tới "
             f"đáy hố sẽ bị gập cứng thành '"
             f"{'pos' if well >= cfg_pos else 'neg'}' và đi thẳng vào prompt. "
             f"Từ bản này chúng chỉ còn là đường lùi khi calib_enabled=false.")
    log("calib", f"{m['n_cells_labelled']} ô có nhãn / "
                 f"{m['n_cells_blank']} ô blank (bỏ — đã MASK khỏi ASL) trên "
                 f"{m['n_val_images']} ảnh val")
    log("calib", "\n" + bands.table())
    src = {}
    for row in bands.rows:
        src[row.get("source", "?")] = src.get(row.get("source", "?"), 0) + 1
    log("calib", "nguồn band: " + ", ".join(f"{k}={v}" for k, v in
                                            sorted(src.items())))
    for row in bands.rows:
        if row.get("source") != "class" and row.get("note"):
            log("calib", f"    {row['name']}: {row['note']}")
    if src.get("none"):
        warn("calib", f"{src['none']}/{len(bands)} bệnh KHÔNG có vùng "
                      f"uncertain đáng tin — ở những bệnh đó pipeline quay về "
                      f"quyết định nhị phân. Đây là kết quả đo được, không "
                      f"phải lỗi: sup_head chưa tách được ô uncertain ra khỏi "
                      f"pos/neg đủ sạch ở checkpoint này.")
    # Giá của việc mở rộng band: mỗi ô POSITIVE rơi vào band là một ca bệnh
    # bị rút khỏi danh sách "Present:" của prompt. Không phải một lỗi — nó là
    # đúng cái đánh đổi mà tiêu chí "phủ nhiều nhất, purity ≥ sàn" chọn — nhưng
    # phải nhìn thấy con số trước khi chạy 2000 ảnh với nó.
    sw = [(r["name"], r.get("pos_swallowed", 0.0)) for r in bands.rows
          if r.get("source") in ("class", "global")
          and float(r.get("pos_swallowed", 0.0)) > 0.25]
    if sw:
        sw.sort(key=lambda t: -t[1])
        warn("calib",
             "band nuốt > 25% ô POSITIVE ở " + ", ".join(
                 f"{n} ({v:.0%})" for n, v in sw[:6])
             + (f" (+{len(sw) - 6})" if len(sw) > 6 else "")
             + ". Những ô đó chuyển từ 'Present:' sang không-nói-gì. Nếu "
               "không chấp nhận được, đặt run.calib_max_pos_swallow (vd 0.15) "
               "— band sẽ hẹp lại và độ phủ uncertain giảm theo.")

    off = [r["name"] for r in bands.rows
           if r.get("source") == "class" and not r.get("well_in_band")]
    if off:
        warn("calib", f"band ĐO ĐƯỢC không chứa hố lý thuyết p*={well:.3f} ở: "
                      f"{', '.join(off[:6])}"
                      + (f" (+{len(off) - 6})" if len(off) > 6 else "")
                      + ". Model chưa xuống tới đáy hố ở những bệnh này — "
                        "đọc kèm sup_recall và số mẫu train.")


def prepare_bands(*, cfg, router, loader, device, aspect_names, disease_names,
                  log, warn, train_samples=None, weights=None, path=None):
    """Nạp `calib_bands.json` nếu còn hợp lệ, không thì hiệu chuẩn lại.

    Trả về LUÔN LUÔN một `Bands` — phần còn lại của pipeline không phải rẽ
    nhánh. Hiệu chuẩn tắt (`calib_enabled: false`) thì đó là band hằng số từ
    hai khoá cũ trong config, và log nói rõ là đang chạy đường lùi.
    """
    r = cfg.run
    n_class = len(disease_names)
    cfg_pos, cfg_neg = float(r.cam_pos_thr), float(r.cam_neg_thr)
    path = str(path or os.path.join(r.out_dir, "calib_bands.json"))

    if not bool(getattr(r, "calib_enabled", True)):
        warn("calib", f"calib_enabled=false — chạy bằng hai hằng số cũ "
                      f"({cfg_neg:g}, {cfg_pos:g}). Không có gì bảo đảm vùng "
                      f"đó là vùng uncertain của checkpoint này.")
        return default_bands(n_class, cfg_pos, cfg_neg)

    if os.path.exists(path) and not bool(getattr(r, "calib_refit", False)):
        try:
            b = Bands.load(path)
        except Exception as exc:                     # noqa: BLE001
            warn("calib", f"{path} không đọc được ({exc}) — hiệu chuẩn lại")
        else:
            if len(b) != n_class:
                warn("calib", f"{path} có {len(b)} lớp, run này có {n_class} "
                              f"— hiệu chuẩn lại")
            else:
                b.check_stale(train_samples, weights, warn)
                log("calib", f"dùng lại {path} (hiệu chuẩn lúc "
                             f"{b.meta.get('created', '?')})")
                _report(b, log, warn, cfg_pos, cfg_neg)
                return b

    try:
        P, Y = collect_sup_probs(
            router, loader, device, log=log,
            max_batches=int(getattr(r, "calib_max_batches", 0) or 0))
    except Exception as exc:                             # noqa: BLE001
        # Không chết cả phiên infer vì hiệu chuẩn — nhưng cũng KHÔNG lặng lẽ
        # quay về hai hằng số cũ, vì chúng nằm ngoài hố uncertain. Đường lùi
        # là band TIÊN NGHIỆM suy từ chính hình dạng ASL: nó chưa nhìn dữ
        # liệu nên không có gì bảo đảm về purity, nhưng ít nhất nó bao quanh
        # đúng chỗ mà loss đang kéo ô uncertain tới.
        lo, hi = asl_prior_band(float(r.uncertain_soft),
                                float(r.asl_gamma_pos),
                                float(r.asl_gamma_neg), float(r.asl_clip))
        warn("calib", f"KHÔNG quét được tập val ({exc}). Chạy tiếp bằng band "
                      f"TIÊN NGHIỆM ({lo:.3f}, {hi:.3f}) suy từ hình dạng "
                      f"ASL — chung cho cả 14 bệnh, KHÔNG có bảo đảm purity "
                      f"nào. Mọi con số 'unc' của lần chạy này phải đọc kèm "
                      f"cảnh báo này.")
        return default_bands(n_class, hi, lo, note="prior")
    bands = fit_bands(
        P, Y, aspect_names, disease_names,
        uncertain_soft=float(r.uncertain_soft),
        asl_gamma_pos=float(r.asl_gamma_pos),
        asl_gamma_neg=float(r.asl_gamma_neg),
        asl_clip=float(r.asl_clip),
        purity=float(getattr(r, "calib_purity", 0.60)),
        lift=float(getattr(r, "calib_lift", 1.5)),
        conf=float(getattr(r, "calib_conf", 0.95)),
        min_unc=int(getattr(r, "calib_min_unc", 200)),
        min_cells=int(getattr(r, "calib_min_cells", 30)),
        holdout=float(getattr(r, "calib_holdout", 0.5)),
        seed=int(getattr(r, "calib_seed", 0)),
        grid=int(getattr(r, "calib_grid", 1000)),
        margin=float(getattr(r, "calib_margin", 0.04)),
        pos_cap=float(getattr(r, "calib_max_pos_swallow", 0.0) or 0.0),
        meta={"run_tag": getattr(cfg.arch, "tag", None),
              "mode": getattr(r, "mode", None),
              "weights": weights, "train_samples": train_samples,
              "cfg_fallback": [cfg_neg, cfg_pos]},
        log=log)
    bands.save(path)
    log("calib", f"ghi {path}")
    _report(bands, log, warn, cfg_pos, cfg_neg)
    return bands

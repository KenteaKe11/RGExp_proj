"""Grad-CAM trên nhánh router — "model nhìn ĐÂU khi nó thấy bệnh này".

Bảng metrics nói báo cáo sinh ra giống báo cáo thật đến đâu; nó không nói
model đọc được điều đó TỪ CHỖ NÀO trên phim. Một model đoán đúng "pleural
effusion" vì nhìn góc sườn hoành và một model đoán đúng vì nhớ rằng tư thế
nằm thường kèm tràn dịch cho cùng một con số BLEU. Module này tách được hai
trường hợp đó ra.

═══ Lấy từ Feat-Extractor, không phải từ SigLIP ═══

Ảnh đi vào model theo HAI đường (xem src/router.py): SigLIP của Gemma, và
Feat-Extractor ~31M tham số train từ số 0. Bản đồ ở đây lấy từ đường THỨ HAI,
vì chỉ đường đó có một cái đầu phân loại để lấy đạo hàm: `sup_head` cho ra 14
logit bệnh, mỗi logit là một đại lượng có nghĩa để hỏi "logit này tăng khi
vùng nào của feature map tăng".

    pixel_values [B,3,896,896]
        │  FeatExtractor.conv_map  (resize 512, xám, 5 bậc DownBlock,
        │                           mỗi bậc Patching về 16×16 rồi nối kênh)
        ▼
    A = fmap [B, 1984, 16, 16]        ← LỚP MỤC TIÊU của Grad-CAM
        │  from_map → self-attn → swiglu 2560 → QFormer query-0 → sup_head
        ▼
    y = sup_logits [B, 14]

    với mỗi bệnh d ĐƯỢC PHÁT HIỆN trong ảnh:
        α_k = GAP_hw( ∂y_d / ∂A_k )              [1984]
        M_d = ReLU( Σ_k α_k · A_k )              [16, 16]

Chọn `fmap` (bản nối ĐA TỈ LỆ của cả 5 bậc conv) chứ không phải `feat` sau
self-attn: sau self-attn mỗi token đã trộn cả 256 vị trí, chỉ số token không
còn tương ứng với một ô trên phim nữa. Gradient vẫn đi NGƯỢC qua self-attn và
QFormer để về tới `fmap` — nghĩa là bản đồ vẫn phản ánh trọn đường đi tới
logit, chỉ được ĐỌC ở tầng còn giữ trục không gian. Từ đợt đổi kiến trúc, 1984
kênh đó gom bằng chứng ở NĂM thang phân giải chứ không chỉ thang thô nhất, nên
bản đồ bám được cả chi tiết nhỏ (nốt, đường ống) lẫn vùng lớn.

ReLU giữ lại phần đóng góp DƯƠNG, đúng theo Grad-CAM gốc (Selvaraju et al.
2017): câu hỏi là "chỗ nào ỦNG HỘ kết luận này", không phải "chỗ nào ảnh
hưởng tới nó". Vùng phản đối một bệnh là một câu hỏi khác và cần một bản đồ
khác.

═══ CHỈ những nhãn được phát hiện ═══

Xuất đủ 14 bản đồ cho mọi ảnh là 14×256 số cho một ảnh mà 12-13 trong số đó
là bản đồ của một bệnh model tin là KHÔNG có. Grad-CAM của một logit âm sâu
vẫn ra một bản đồ trông bình thường — nó là hướng tăng của một đại lượng
không ai quan tâm — nên nó không phải dữ liệu, nó là nhiễu có hình dạng đẹp,
và nó chiếm chỗ đúng bằng phần có ích.

`detect_findings` chốt danh sách: bệnh có σ(logit) ≥ ngưỡng, nhiều nhất
`max_k` cái, xếp theo xác suất giảm dần. Không bệnh nào qua ngưỡng (ảnh model
cho là bình thường) thì vẫn lấy MỘT bệnh mạnh nhất và đánh dấu `fallback` —
"model không thấy gì" cũng là một kết luận cần soi được bản đồ, và một danh
sách rỗng thì trên giao diện không phân biệt được với "ảnh này chưa chạy CAM".

Xác suất của ĐỦ 14 bệnh thì luôn được ghi (`sup_prob`), rẻ và là thứ mà truy
vấn finding × state trong `scripts/probe_summary.py` chạy trên đó.

═══ State suy từ xác suất, không phải từ nhãn ═══

`state_of` gập một xác suất về ba trạng thái đọc được:

    p ≥ pos_thr              → "pos"
    p ≤ neg_thr              → "neg"
    ở giữa                   → "unc"

Hai ngưỡng đó KHÔNG còn là hằng số trong config. `src/calib.py` đo chúng
trên tập val trước khi infer, một cặp cho mỗi bệnh, và chỉ cấp band cho bệnh
nào qua được sàn tinh khiết có cận dưới Wilson; bệnh không qua thì lo = hi và
ở đó chỉ còn hai trạng thái. Mọi hàm dưới đây vì thế nhận `pos_thr`/`neg_thr`
là số vô hướng HOẶC vector [14] — xem `_thr_at`.

Đây là trạng thái model NGHĨ, không phải nhãn CheXbert trong labels.npz (bộ
đó có bốn ô: blank/positive/negative/uncertain). Hai thứ khác nhau và không
được trộn: `sup_head` được train trên một trục nhị phân có mask, ô `blank`
của CheXbert đã bị MASK khỏi loss (src/losses.py) nên model chưa bao giờ được
dạy để nói "không đề cập". Gọi vùng giữa là "unc" là mô tả đúng cái model
làm — nó lưỡng lự — chứ không phải khôi phục lại ô uncertain của CheXbert.

═══ Nhãn cứng quay ngược vào prompt ═══

`condition_labels` + `condition_text` gập cùng bộ xác suất đó thành hai danh
sách tên bệnh (present / absent) rồi nối vào prompt cho một lượt sinh THỨ HAI
— "gen with supervised condition". Ô `unc` không vào danh sách nào: xem
docstring của `condition_labels`. Dùng chung `pos_thr`/`neg_thr` với heatmap
là có chủ ý — một ảnh chỉ có MỘT cách đọc.

═══ Lưu ở đâu ═══

`CamStore` gom vào bộ nhớ rồi ghi `cam.npz` cạnh `reports.csv`. Mỗi ảnh ba
mảng, khoá theo `safe_name(key_id)` giống hệt tên file bản ghi probe:

    m/<key>    float16 [n_det, G, G]   bản đồ đã chuẩn hoá về [0,1]
    c/<key>    int16   [n_det]         chỉ số bệnh của từng bản đồ
    p/<key>    float16 [14]            σ(sup_logit) của ĐỦ 14 bệnh

Cộng thêm vài khoá metadata dùng chung (`__labels__`, `__grid__`, `__meta__`).
float16 vì bản đồ đã chuẩn hoá về [0,1] và người đọc nó là một thang màu 20
mức — ba chữ số thập phân là quá đủ, và fp32 chỉ nhân đôi dung lượng để lưu
những chữ số không bao giờ hiện ra. Cỡ thật: 3000 ảnh × 5 bản đồ × 256 ô ×
2B ≈ 7.7MB.

Ghi ĐÈ trọn file mỗi lần flush chứ không nối thêm: npz không có chế độ
append, và 7.7MB ghi lại mỗi 20 ảnh vẫn rẻ hơn nhiều so với một forward. Đổi
lại là phiên đứt giữa chừng vẫn còn một file đọc được, không phải một file
cụt.
"""
from __future__ import annotations

import json
import os

import numpy as np

# Ba trạng thái mà truy vấn finding × state trong probe_summary chạy trên đó.
# Thứ tự này là thứ tự hiển thị, không phải một mã số — đừng lưu chỉ số của
# nó xuống đĩa ở đâu cả.
STATES = ("pos", "unc", "neg")


def state_of(p, pos_thr=0.5, neg_thr=0.2):
    """Xác suất → "pos" / "unc" / "neg".

    Hai ngưỡng, không phải một, và chúng KHÔNG bắt buộc phải bù nhau: vùng
    giữa là chỗ model lưỡng lự, mà bề rộng của vùng đó là một lựa chọn đọc
    chứ không phải một hằng số của bài toán.

    Hai giá trị mặc định trong chữ ký (0.5 / 0.2) chỉ còn là ĐƯỜNG LÙI cho
    `calib_enabled: false`. Từ §1.8 chúng đến từ `src/calib.py`: đo trên tập
    val, một cặp cho MỖI bệnh, kèm cận dưới Wilson cho độ tinh khiết. Lý do
    phải đo chứ không đoán: đáy hố thế năng của ô uncertain KHÔNG nằm ở
    `uncertain_soft` — γ⁻ = 4 dập nhánh âm nên với y = 0.55 cực tiểu của ASL
    nằm ở p ≈ 0.68, tức NGOÀI HẲN khoảng (0.2, 0.5) mà bản trước gọi là
    "unc". Chạy bằng hai hằng số cũ là gán mọi ô uncertain thành "pos" rồi
    đẩy nó vào danh sách "Present:" của prompt, im lặng.

    `neg_thr > pos_thr` là cấu hình vô nghĩa (không còn ô nào là "pos") nên
    nó nổ ngay tại đây thay vì lặng lẽ cho ra một bảng toàn "neg".
    """
    pos_thr, neg_thr = float(pos_thr), float(neg_thr)
    if neg_thr > pos_thr:
        raise ValueError(f"cam_neg_thr={neg_thr} > cam_pos_thr={pos_thr} — "
                         f"vùng 'pos' rỗng, mọi ô sẽ là 'neg' hoặc 'unc'")
    p = float(p)
    if p >= pos_thr:
        return "pos"
    if p <= neg_thr:
        return "neg"
    return "unc"


def _thr_at(thr, i):
    """Ngưỡng của lớp `i`, chấp nhận cả số vô hướng lẫn vector [14].

    Một hàm nhỏ để mọi chỗ gọi không phải rẽ nhánh: hiệu chuẩn tắt thì
    `src/calib.py` vẫn trả về một `Bands` gồm hai vector hằng, còn code cũ
    truyền float thì vẫn chạy y như trước.
    """
    a = np.asarray(thr, dtype=np.float64).reshape(-1)
    if a.size == 0:
        raise ValueError("ngưỡng rỗng")
    return float(a[i]) if a.size > 1 else float(a[0])


def states_of(prob, pos_thr=0.5, neg_thr=0.2):
    """[14] xác suất -> [14] state, ngưỡng RIÊNG cho từng bệnh.

    Vòng lặp `state_of` thủ công ở chỗ gọi thì mỗi chỗ tự index lấy ngưỡng —
    và chỗ nào quên thì cả 14 bệnh dùng chung ngưỡng của bệnh 0 mà không có
    gì kêu lên. Một hàm, một cách index.
    """
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    return [state_of(p[i], _thr_at(pos_thr, i), _thr_at(neg_thr, i))
            for i in range(p.size)]


def condition_labels(prob, names, pos_thr=0.5, neg_thr=0.2):
    """[14] xác suất → nhãn CỨNG để điều kiện hoá lần sinh thứ hai.

    Đây là bước "feed nhãn soft ngược lại vào model": σ(sup_logit) là một
    vector mềm, còn thứ decoder đọc được trong prompt là một DANH SÁCH TÊN.
    Gập từ mềm sang cứng bằng đúng hai ngưỡng đang vẽ heatmap (`state_of`) —
    dùng ngưỡng khác thì điều kiện của câu chữ và màu của bản đồ nói hai
    chuyện khác nhau về cùng một ảnh, và không ai đối chiếu được nữa.

    ═══ Ô "unc" bị LOẠI, không bị gán 0 ═══

    `y_pred` chỉ có hai giá trị thật (1 = pos, 0 = neg); ô ở giữa để `None`
    và KHÔNG xuất hiện trong danh sách nào của prompt. Gán nó thành 0 là
    khai một điều model chưa hề nói — "không có bệnh này" — rồi bắt decoder
    viết theo. Đúng ở chỗ đó là: model lưỡng lự, mà lưỡng lự thì đã được thể
    hiện ở heatmap (bản đồ của ô unc vẫn được vẽ nếu nó qua `cam_threshold`)
    và ở `sup_prob` trong reports.csv. Điều kiện hoá chỉ nhận phần model dám
    khẳng định; phần còn lại trả về cho ảnh.

    Thứ tự trong mỗi danh sách là thứ tự CHẮC CHẮN giảm dần: pos xếp theo p
    giảm, neg xếp theo p tăng. Prompt bị cắt ở đâu thì phần mất đi cũng là
    phần model tin ít nhất.

    Trả về dict:
        y_pred  [14] 1 / 0 / None
        pos     [(tên, p)…]  giảm dần theo p
        neg     [(tên, p)…]  tăng dần theo p
        unc     [(tên, p)…]  chỉ để ghi lại, KHÔNG vào prompt
    """
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    n = min(p.size, len(names))
    y, pos, neg, unc = [], [], [], []
    for i in range(int(p.size)):
        name = str(names[i]) if i < n else f"#{i}"
        st = state_of(p[i], _thr_at(pos_thr, i), _thr_at(neg_thr, i))
        if st == "pos":
            y.append(1)
            pos.append((name, float(p[i])))
        elif st == "neg":
            y.append(0)
            neg.append((name, float(p[i])))
        else:
            y.append(None)
            unc.append((name, float(p[i])))
    pos.sort(key=lambda t: -t[1])
    neg.sort(key=lambda t: t[1])
    unc.sort(key=lambda t: -t[1])
    return {"y_pred": y, "pos": pos, "neg": neg, "unc": unc}


def condition_text(cond, max_pos=0, max_neg=0):
    """dict của `condition_labels` → đoạn thêm vào prompt (hoặc "" nếu trống).

    Hai danh sách và một câu ràng buộc, hết. Không đưa xác suất vào: con số
    trong prompt mời decoder viết lại con số ("0.82 probability of ...") — mà
    xác suất thì đã nằm trong `sup_prob`, đúng chỗ của nó.

    Câu cuối nói rõ về phần KHÔNG được liệt kê. Thiếu nó, model coi hai danh
    sách là danh sách đóng và im lặng về mọi thứ còn lại — tức là ô unc bị
    biến thành neg một cách gián tiếp, đúng cái mà `condition_labels` vừa cố
    tránh.

    `max_pos`/`max_neg` = 0 nghĩa là không cắt. 14 tên bệnh chỉ tốn ~40
    token, nên mặc định là giữ đủ.
    """
    pos = [n for n, _ in cond["pos"]][:max_pos or None]
    neg = [n for n, _ in cond["neg"]][:max_neg or None]
    if not pos and not neg:
        return ""
    lines = ["An image classifier has already read this radiograph."]
    lines.append("Present: " + (", ".join(pos) if pos else "none of the "
                                "findings it screens for") + ".")
    if neg:
        lines.append("Absent: " + ", ".join(neg) + ".")
    lines.append("Write the report consistent with those two lists. Anything "
                 "on neither list is undetermined - judge it from the image "
                 "and hedge if it is unclear. Keep the two-section format.")
    return "\n".join(lines)


def detect_findings(prob, threshold=0.5, max_k=6):
    """[14] xác suất → (chỉ số bệnh được phát hiện, có phải fallback không).

    Xếp giảm dần theo xác suất, cắt ở `threshold`, giữ nhiều nhất `max_k`.
    Không cái nào qua ngưỡng thì trả về một cái mạnh nhất kèm cờ `fallback`
    (xem docstring module).

    Trả về `list[int]` chứ không phải mảng numpy: nó đi thẳng vào JSON của
    bản ghi probe.
    """
    p = np.asarray(prob, dtype=np.float64).reshape(-1)
    if p.size == 0:
        return [], False
    order = np.argsort(-p)
    hit = [int(i) for i in order if p[i] >= float(threshold)]
    if hit:
        return hit[:max(int(max_k), 1)], False
    return [int(order[0])], True


def normalize_map(m):
    """[G,G] thô → [G,G] trong [0,1], chia theo max của CHÍNH nó.

    Cùng quy ước với `pack_map` của src/probe.py và vì cùng một lý do: câu hỏi
    của một bản đồ là "vùng nào", không phải "mạnh cỡ nào". Độ mạnh tuyệt đối
    giữa các bệnh đã nằm ở `sup_prob`, và trộn hai câu hỏi vào một thang màu
    thì bản đồ của bệnh yếu nhất ra một ô xám phẳng không đọc được gì.

    Bản đồ toàn 0 (ReLU cắt sạch — logit này giảm ở mọi hướng) trả về nguyên
    0 chứ không chia cho 0.
    """
    a = np.asarray(m, dtype=np.float32)
    mx = float(a.max()) if a.size else 0.0
    return a / mx if mx > 0 else a


def gradcam(router, pixel_values, class_ids=None, threshold=0.5, max_k=6):
    """Grad-CAM của nhánh router cho MỘT batch ảnh.

    Trả về list (một phần tử cho mỗi ảnh trong batch):

        {"prob": [14] float, "findings": [d…], "maps": [n_det, G, G] float32,
         "fallback": bool}

    `class_ids` để None -> tự chọn bằng `detect_findings`. Truyền tay một
    danh sách thì dùng đúng danh sách đó cho MỌI ảnh trong batch (dùng ở
    selftest và khi muốn so cùng một bệnh qua nhiều ảnh).

    ═══ Ba chỗ dễ sai, đã xử lý ═══

    **1. Grad phải BẬT.** Chỗ gọi (mode infer, mode probe) đang nằm trong
    `torch.no_grad()` và mọi tham số router đều `requires_grad = False`. Hàm
    này tự mở `torch.enable_grad()` và `sup_forward` tự đánh dấu feature map —
    thiếu một trong hai thì `autograd.grad` trả về None mà không báo lỗi.

    **2. Một backward cho mỗi (ảnh, bệnh), không gộp.** Gộp tổng logit của
    nhiều bệnh rồi backward một lần cho ra bản đồ của TỔNG, không phải của
    từng bệnh — mà tổng thì trộn hết mọi vùng lại và bản đồ nào cũng giống
    nhau. `retain_graph=True` giữ đồ thị của một forward duy nhất để n_det
    lần backward dùng chung; chỉ lần cuối mới thả.

    **3. Trả về numpy, không phải tensor.** Chỗ gọi lưu xuống đĩa chứ không
    tính tiếp, và giữ tensor CUDA sống thêm một vòng lặp là cách rò VRAM quen
    thuộc nhất trong file này.
    """
    import torch

    was_training = router.training
    router.eval()                       # BatchNorm: thống kê chạy, tất định
    try:
        with torch.enable_grad():
            logits, fmap = router.sup_forward(pixel_values)   # [B,14], [B,C,G,G]
            probs = torch.sigmoid(logits.detach()).float().cpu().numpy()
            B, C, G, Gw = fmap.shape
            out = []
            for b in range(B):
                if class_ids is None:
                    ids, fb = detect_findings(probs[b], threshold, max_k)
                else:
                    ids, fb = [int(c) for c in class_ids], False
                maps = np.zeros((len(ids), G, Gw), dtype=np.float32)
                for j, d in enumerate(ids):
                    last = (b == B - 1) and (j == len(ids) - 1)
                    g = torch.autograd.grad(logits[b, d], fmap,
                                            retain_graph=not last)[0]
                    # α_k = trung bình không gian của gradient trên kênh k.
                    # Chỉ hàng b: gradient của một logit của ảnh b bằng 0 ở
                    # mọi hàng khác, nhưng cắt ra rẻ hơn là nhân với 0.
                    alpha = g[b].mean(dim=(1, 2))                  # [C]
                    cam = torch.relu((alpha[:, None, None]
                                      * fmap[b]).sum(0))           # [G, G]
                    maps[j] = normalize_map(cam.detach().float().cpu().numpy())
                out.append({"prob": [float(v) for v in probs[b]],
                            "findings": ids, "maps": maps, "fallback": fb})
            del fmap, logits
        return out
    finally:
        if was_training:
            router.train()


def pack_maps(maps, nd=4):
    """[n, G, G] float → list các list phẳng đã làm tròn, cho JSON.

    Cùng khuôn với `spatial` của bản ghi probe (src/probe.py::pack_map): giá
    trị đã trong [0,1], làm tròn 4 chữ số. Không có bước này thì mỗi bản ghi
    probe phình thêm ~40KB toàn chữ số mà thang màu 20 mức không dùng tới.
    """
    a = np.asarray(maps, dtype=np.float64)
    if a.ndim == 2:
        a = a[None]
    return [[round(float(v), nd) for v in row.reshape(-1)] for row in a]


def cam_block(item, disease_names, grid, pos_thr=0.5, neg_thr=0.2):
    """Một phần tử của `gradcam` → khối `cam` nhúng vào bản ghi probe JSON.

    Ghi CẢ tên bệnh lẫn chỉ số: bản ghi trên Hub phải tự đọc được mà không cần
    tra ngược vào `disease_labels` ở cấp trên, và `scripts/probe_summary.py`
    đọc được cả bản ghi của những lần chạy có danh sách nhãn khác.
    """
    prob = [round(float(v), 5) for v in item["prob"]]
    return {
        "grid": [int(grid[0]), int(grid[1])],
        "sup_prob": prob,
        "state": states_of(prob, pos_thr, neg_thr),
        "findings": [int(d) for d in item["findings"]],
        "finding_names": [str(disease_names[d]) if d < len(disease_names)
                          else f"#{d}" for d in item["findings"]],
        "fallback": bool(item["fallback"]),
        "maps": pack_maps(item["maps"]),
    }


class CamStore:
    """Gom bản đồ của cả phiên rồi ghi `cam.npz`. Xem docstring module.

    Không có Hub, không có thread nền: file này nhỏ và nằm cạnh reports.csv,
    ai cần thì `np.load` một dòng. Bản ghi probe mới là thứ đi lên Hub.
    """

    def __init__(self, path, disease_names, grid, meta=None):
        self.path = str(path)
        self.names = [str(n) for n in disease_names]
        self.grid = [int(grid[0]), int(grid[1])]
        self.meta = dict(meta or {})
        self.maps, self.cls, self.prob = {}, {}, {}

    def add(self, key, item):
        """`key` = safe_name(key_id), `item` = một phần tử của `gradcam`."""
        self.maps[key] = np.asarray(item["maps"], dtype=np.float16)
        self.cls[key] = np.asarray(item["findings"], dtype=np.int16)
        self.prob[key] = np.asarray(item["prob"], dtype=np.float16)

    def __len__(self):
        return len(self.maps)

    def save(self):
        """Ghi đè trọn file. Trả về đường dẫn, hoặc None nếu chưa có gì."""
        if not self.maps:
            return None
        blob = {"__labels__": np.asarray(self.names, dtype=object),
                "__grid__": np.asarray(self.grid, dtype=np.int32),
                "__meta__": np.asarray(json.dumps(self.meta,
                                                  ensure_ascii=False))}
        for k, v in self.maps.items():
            blob[f"m/{k}"] = v
            blob[f"c/{k}"] = self.cls[k]
            blob[f"p/{k}"] = self.prob[k]
        tmp = self.path + ".tmp.npz"
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        # `savez` thêm đuôi .npz nếu tên chưa có — đặt sẵn để `os.replace`
        # không trượt sang một tên khác.
        np.savez_compressed(tmp, **blob)
        os.replace(tmp, self.path)
        return self.path


def load_cam(path):
    """`cam.npz` → (dict khoá → {maps, findings, prob}, labels, grid, meta).

    Đối xứng với `CamStore.save`, để script khác (và selftest) đọc lại được mà
    không phải biết quy ước tiền tố `m/` `c/` `p/`.
    """
    z = np.load(path, allow_pickle=True)
    labels = [str(x) for x in z["__labels__"].tolist()]
    grid = [int(x) for x in z["__grid__"].tolist()]
    try:
        meta = json.loads(str(z["__meta__"].item()))
    except Exception:
        meta = {}
    out = {}
    for name in z.files:
        if not name.startswith("m/"):
            continue
        k = name[2:]
        out[k] = {"maps": np.asarray(z[name], dtype=np.float32),
                  "findings": [int(v) for v in z[f"c/{k}"].tolist()],
                  "prob": [float(v) for v in z[f"p/{k}"].tolist()]}
    return out, labels, grid, meta

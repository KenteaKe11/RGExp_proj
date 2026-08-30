"""Phần supervised của loss (§1.4, §1.6) — ASL trên 14 finding.

Lịch train chỉ còn MỘT phase multi-task:

    L = CE + w_sup · ASL(sup)          (w_sup = 0.2)

CE nằm ở src/model.py (dính chặt với chuyện cắt logits của decoder). File này
giữ phần supervised: chuyển nhãn 4-aspect của CheXbert thành MỘT trục nhị
phân có mask, rồi tính Asymmetric Loss trên đó.

KHÔNG còn số hạng Aux: cân bằng tải giữa các expert do bias của router lo,
cập nhật bằng quy tắc ngoài đồ thị autograd (src/router.py::update_bias).

──────────────────────────────────────────────────────────────────────────────
VÌ SAO ĐỔI TỪ FOCAL 4 LỚP SANG ASL NHỊ PHÂN

Nhãn gốc là 14 finding × 4 aspect (blank / positive / negative / uncertain —
đầu ra chuẩn của CheXbert). Bản cũ cho sup_head đoán CẢ BỐN aspect bằng
softmax + Focal, tức là hỏi "câu báo cáo NÓI GÌ về bệnh này", trong đó lớp
đông nhất — blank, "không nhắc tới" — chiếm phần lớn bảng. Học một lớp
"không nhắc tới" là học thói quen VIẾT của bác sĩ, không phải học BỆNH; và
vì nó áp đảo về số lượng nên nó chi phối luôn gradient của cả nhánh.

Bản này hỏi đúng câu cần hỏi — "bệnh này CÓ hay KHÔNG" — trên một trục duy
nhất, và bỏ hẳn những ô không có câu trả lời:

    positive   -> 1.0
    negative   -> 0.0
    uncertain  -> 0.7      (soft label, nghiêng về positive)
    blank      -> MASK, không đóng góp gradient, không nằm trong mẫu số

`uncertain = 0.7` không phải một con số làm tròn cho đẹp. Trong báo cáo
X-quang, "possible pneumonia" / "cannot exclude effusion" là một lời KHẲNG
ĐỊNH YẾU, không phải một lời phủ định: bác sĩ đã nhìn thấy cái gì đó đủ để
viết ra. Ép nó về 1.0 là dạy model tự tin hơn người viết; ép về 0.0 là dạy
model bỏ qua đúng những ca khó. 0.7 giữ nó ở phía dương nhưng nhẹ hơn một ca
khẳng định chắc chắn.

`blank -> MASK` chứ không phải `blank -> 0.0`. Hai thứ này khác nhau về bản
chất: "báo cáo nói KHÔNG có tràn dịch" là một quan sát âm tính, còn "báo cáo
không nhắc tới tràn dịch" là KHÔNG CÓ THÔNG TIN. Gán 0.0 cho ô blank là bịa
ra một quan sát âm tính chưa ai làm, và vì blank chiếm đa số nên cái bịa đó
sẽ trở thành thứ model học thuộc nhiều nhất.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import torch

EPS = 1e-8

# ── Hyperparameter CHUẨN của ASL (Ridnik et al., ICCV 2021) ─────────────────
# Đúng default của lớp `AsymmetricLoss` bản gốc, không chỉnh gì.
ASL_GAMMA_POS = 0.0     # nhánh dương KHÔNG focus: mọi ca dương đều đáng học
ASL_GAMMA_NEG = 4.0     # nhánh âm focus mạnh: ô âm dễ gần như hết gradient
# ── DUY NHẤT một hằng số đi chệch default, và nó đi theo chuẩn Y TẾ ────────
# Default của ASL là 0.05 (5%). Ở đây 0.02, theo mốc "probably benign" của
# BI-RADS 3 — ngưỡng ≤ 2% khả năng ác tính, tức mức mà chuẩn đọc phim cho
# phép xử lý một tổn thương NHƯ THỂ nó âm tính. 5% cao gấp đôi rưỡi mốc đó:
# nó bảo model "đoán 4% là đủ âm rồi, thôi", trong khi 4% khả năng bỏ sót
# trên phim ngực không ai coi là đủ âm.
ASL_CLIP = 0.02
UNCERTAIN_SOFT = 0.7    # xem docstring trên

# Tên aspect trong labels.npz chỉ được viết một kiểu, nhưng không có gì bảo
# đảm THỨ TỰ. Nên chỉ số của từng vai được suy ra từ CHÍNH `class_names` lúc
# chạy (xem aspect_roles) chứ không hardcode 0/1/2/3 — hardcode sai thì
# uncertain thành negative và không có gì kêu lên cả.
_ROLE_WORDS = {
    "blank": ("blank", "none", "nomention", "notmentioned", "unmentioned",
              "missing", "na", "nan", "unknown", "empty", "null"),
    "positive": ("positive", "pos", "present", "yes", "abnormal"),
    "negative": ("negative", "neg", "absent", "no", "normal"),
    "uncertain": ("uncertain", "uncertainty", "unsure", "maybe", "possible",
                  "equivocal", "ambiguous"),
}


def _norm(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def aspect_roles(class_names):
    """['Blank','Positive','Negative','Uncertain'] -> {vai: chỉ số}.

    Khớp theo TÊN, không theo vị trí. Bốn vai phải khớp đúng bốn tên khác
    nhau, không thiếu không trùng — nếu không thì ném lỗi ngay lúc dựng bảng
    thay vì train xong mới phát hiện uncertain bị coi là negative.
    """
    names = [_norm(n) for n in class_names]
    if len(names) != 4:
        raise ValueError(f"cần đúng 4 aspect, labels.npz có {len(names)}: "
                         f"{list(class_names)}")
    roles = {}
    for role, words in _ROLE_WORDS.items():
        hit = [i for i, n in enumerate(names)
               if n in words or any(n.startswith(w) for w in words)]
        if len(hit) != 1:
            raise ValueError(
                f"không suy được aspect '{role}' từ class_names="
                f"{list(class_names)} (khớp {len(hit)} tên). Sửa "
                f"src/losses.py::_ROLE_WORDS cho khớp bộ nhãn này — ĐỪNG "
                f"đoán thứ tự, gán nhầm uncertain thành negative là một lỗi "
                f"im lặng.")
        roles[role] = hit[0]
    if len(set(roles.values())) != 4:
        raise ValueError(f"hai vai cùng trỏ một aspect: {roles} với "
                         f"class_names={list(class_names)}")
    return roles


def binary_target_table(class_names, uncertain_soft=UNCERTAIN_SOFT):
    """Bảng tra 4 aspect -> (đích nhị phân, có tính vào loss không).

    Trả về (target [4] float32, valid [4] bool) để `to_binary` chỉ còn là hai
    phép index — rẻ, và toàn bộ quy ước y tế nằm gọn ở một chỗ đọc được.
    """
    r = aspect_roles(class_names)
    target = torch.zeros(4, dtype=torch.float32)
    valid = torch.zeros(4, dtype=torch.bool)
    target[r["positive"]], valid[r["positive"]] = 1.0, True
    target[r["negative"]], valid[r["negative"]] = 0.0, True
    target[r["uncertain"]], valid[r["uncertain"]] = float(uncertain_soft), True
    target[r["blank"]], valid[r["blank"]] = 0.0, False      # MASK
    return target, valid


def to_binary(y, table):
    """y [B, 14] chỉ số aspect (âm = thiếu nhãn) -> (target, mask) [B, 14].

    `table` là kết quả của `binary_target_table`, đã ở đúng device.
    """
    tgt_lut, valid_lut = table
    known = y >= 0
    idx = y.clamp_min(0)
    target = tgt_lut[idx]
    mask = valid_lut[idx] & known
    return target, mask


def asl_loss(logits, target, mask, gamma_pos=ASL_GAMMA_POS,
             gamma_neg=ASL_GAMMA_NEG, clip=ASL_CLIP, eps=EPS):
    """Asymmetric Loss (Ridnik et al. 2021) trên 14 finding, có mask.

    logits [B, 14] — chưa sigmoid, một logit cho mỗi finding
    target [B, 14] — 1.0 dương | 0.0 âm | 0.7 uncertain
    mask   [B, 14] — True ở ô có nhãn; ô blank/thiếu nhãn = False

        p    = σ(logit)
        p⁻   = min(1 − p + m, 1)                 ← m = ngưỡng âm tính
        p_m  = 1 − p⁻ = max(p − m, 0)
        L    = −[ y·(1 − p)^γ⁺·log p  +  (1 − y)·p_m^γ⁻·log p⁻ ]

    Ba thứ ASL làm mà Focal đối xứng không làm được, và cả ba đều đúng thứ
    bài toán này cần:

    1. **Bất đối xứng γ.** γ⁻ = 4 dập gần hết gradient của những ô ÂM mà model
       đã đoán đúng — mà ở đây ô âm áp đảo về số lượng. γ⁺ = 0 thì ngược lại:
       KHÔNG ô dương nào bị giảm trọng số, kể cả ô model đã đoán đúng chắc
       chắn. Focal (γ chung cho cả hai phía) buộc phải chọn: γ lớn thì bỏ luôn
       các ca dương dễ, γ nhỏ thì không dập nổi biển ô âm.
    2. **Ngưỡng âm tính m.** `p⁻ = min(1 − p + m, 1)`: một ô âm mà model đã
       đoán dưới m thì `p⁻ = 1`, `log p⁻ = 0` — hết gradient HẲN, không phải
       "nhỏ dần". Đây là chỗ nhận nhãn bẩn: CheXbert gán nhầm âm cho một ca
       thật ra dương thì ô đó không kéo model đi được nữa sau khi model đã
       học đúng. m = 0.02 theo BI-RADS 3 (xem ASL_CLIP).
    3. **Nhận soft label.** y = 0.7 làm cả hai nhánh cùng sống, mỗi nhánh
       nhận đúng phần khối lượng của nó — không cần một đường code riêng cho
       uncertain.

    Lấy trung bình trên các ô CÓ NHÃN (`mask`), không trên toàn bộ B×14: chia
    cho 14 là chia phần loss của vài ô có thật cho cả những ô mình vừa tuyên
    bố là không có thông tin.
    """
    logits = logits.float()
    target = target.to(logits.dtype)
    m = mask.to(logits.dtype)

    p = torch.sigmoid(logits)
    p_neg = 1.0 - p
    if clip and clip > 0:
        # Bản gốc cộng m vào NHÁNH ÂM rồi kẹp trần 1 — tương đương
        # p_m = max(p − m, 0) ở phía dương, nhưng viết thế này thì không
        # phải nhân thêm một phép clamp nữa ở log.
        p_neg = (p_neg + clip).clamp(max=1.0)

    # Trọng số focusing tính RIÊNG cho từng nhánh, đúng công thức (5) của bài
    # báo: w⁺ = (1 − p)^γ⁺ cho nhánh dương, w⁻ = p_m^γ⁻ cho nhánh âm (với
    # p_m = 1 − p⁻ = max(p − m, 0)).
    #
    # Bản code gốc gộp hai nhánh vào MỘT trọng số (1 − pt)^γ với pt và γ đều
    # nội suy theo y. Với nhãn CỨNG (y ∈ {0,1}) hai cách cho ra số hệt nhau —
    # một nhánh luôn tắt. Với SOFT LABEL thì khác hẳn, và khác theo hướng làm
    # hỏng đúng thứ mình vừa đặt ra: γ nội suy thành 1.2 ở y = 0.7, nhánh âm
    # bị dập thêm một lần nữa, và điểm tối ưu của ô uncertain trôi từ 0.7 lên
    # ~0.9. Tức là khai báo "uncertain = 0.7" nhưng model được dạy tiến tới
    # 0.9 — con số trong config hoá ra không phải con số model học.
    # Tính riêng từng nhánh thì cực tiểu nằm ở ~0.75: vẫn nghiêng dương một
    # chút vì γ⁻ có dập nhánh âm, nhưng bám đúng nhãn đã khai.
    w_pos = torch.pow(1.0 - p, gamma_pos) if gamma_pos > 0 else 1.0
    w_neg = torch.pow((1.0 - p_neg).clamp_min(0.0), gamma_neg) \
        if gamma_neg > 0 else 1.0

    loss = (target * w_pos * torch.log(p.clamp_min(eps))
            + (1.0 - target) * w_neg * torch.log(p_neg.clamp_min(eps)))

    loss = -(loss * m)
    return loss.sum() / m.sum().clamp_min(1.0)


@torch.no_grad()
def supervised_stats(logits, target, mask, thr=0.5):
    """Vài con số để đọc nhánh supervised trong log mà không phải chạy eval.

    `sup_recall` mới là con số đáng nhìn, không phải `sup_acc`: ô dương là
    thiểu số ngay cả sau khi đã bỏ blank, nên một model đoán "âm" cho tất cả
    vẫn có acc cao mà bỏ sót sạch bệnh. Trên phim ngực, bỏ sót mới là cái sai
    đắt tiền.

    Ô uncertain (đích 0.7) tính là DƯƠNG khi chấm: đó là cách bản thân nhãn
    đang nghiêng, và tách riêng một lớp thứ ba chỉ để báo cáo thì không giúp
    đọc log nhanh hơn.
    """
    m = mask.bool()
    n = m.sum().clamp_min(1)
    pred = torch.sigmoid(logits.float()) > thr
    pos = (target >= 0.5) & m
    correct = (pred == (target >= 0.5)) & m
    tp = (pred & pos).sum().float()
    return {
        "sup_acc": float(correct.sum().float() / n),
        "sup_recall": float(tp / pos.sum().clamp_min(1)),
        "sup_precision": float(tp / (pred & m).sum().clamp_min(1)),
        "sup_pos_rate": float(pos.sum().float() / n),
        "sup_n_valid": int(m.sum()),
    }

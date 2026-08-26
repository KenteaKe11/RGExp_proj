"""Ba hàm loss của lịch train (§1.6, §1.7).

    Phase 1:  L = Focal(sup)
    Phase 2:  L = CE + λ_sup · Focal(sup) + λ_aux · Aux

CE nằm ở src/model.py (dính chặt với chuyện cắt logits của decoder), Aux nằm ở
src/router.py (dính chặt với gate). File này giữ phần supervised: bảng α theo
tần suất (nghịch đảo căn bậc 8) và Focal loss.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6


def inv_freq_alpha(cell_freq, normalize="row", power=0.125, eps=EPS):
    """α_ij = 1/(f_ij + ε)^(1/8) — nghịch đảo CĂN BẬC 8 tần suất (§1.6).

    `cell_freq` [14, 4]: tỉ lệ xuất hiện của (bệnh i, aspect j) trong TRAIN.

    Vì sao căn bậc 8 chứ không phải căn bậc hai (bản cũ). Ô hiếm nhất của bảng
    này cỡ 1e-6: 1/f cho nó cả triệu lần trọng số, mà 1/√f vẫn còn ~1_000 lần
    — một nhúm mẫu hiếm nuốt trọn gradient của cả batch và loss nhảy loạn theo
    việc batch có trúng ô đó hay không. Mũ 1/8 kéo dải α về ~5.0 lần: thứ tự
    ưu tiên giữ nguyên (ô hiếm vẫn nặng hơn ô phổ biến), nhưng không ô nào một
    mình lấn át.

    `power` là số mũ dưới mẫu: 0.125 = căn bậc 8 (mặc định §1.6),
    0.5 = căn bậc hai (hành vi cũ), 1.0 = nghịch đảo thẳng.

    normalize:
      "row"    Σ_j f_ij·α_ij = 1 cho từng bệnh — kỳ vọng α trên dữ liệu thật
               bằng 1, nên độ lớn Focal loss so sánh được với CE. MẶC ĐỊNH.
      "global" chuẩn hoá một lần trên cả bảng (giữ chênh lệch giữa các bệnh).
      "none"   để nguyên 1/f^p — với p=1/8 thì α nằm gọn trong ~[1, 5].
    """
    f = np.asarray(cell_freq, dtype=np.float64)
    assert f.shape == (14, 4), f"cell_freq phải là [14, 4], nhận {f.shape}"
    a = 1.0 / np.power(f + eps, float(power))
    if normalize == "row":
        denom = (f * a).sum(axis=1, keepdims=True)
        a = a / np.clip(denom, eps, None)
    elif normalize == "global":
        a = a / max(float((f * a).sum() / f.shape[0]), eps)
    elif normalize != "none":
        raise ValueError(f"normalize lạ: {normalize}")
    return torch.tensor(a, dtype=torch.float32)


def focal_loss(logits, target, alpha, gamma=2.0):
    """Focal loss đa lớp trên 14 bệnh × 4 aspect.

    logits [B, 14, 4] — chưa softmax
    target [B, 14]     — chỉ số aspect đúng (0..3); giá trị < 0 = bỏ qua ô đó
    alpha  [14, 4]     — bảng α theo tần suất, cùng device

        FL = −α_{i,y} · (1 − p_{i,y})^γ · log p_{i,y}

    Lấy trung bình trên các ô hợp lệ. γ=2 ("lực phạt mũ 2"): ô đã đoán đúng
    chắc chắn (p→1) gần như không còn đóng góp gradient, phần còn lại dồn vào
    ô mô hình đang sai.
    """
    B = logits.shape[0]
    logp = F.log_softmax(logits.float(), dim=-1)          # [B, 14, 4]
    valid = target >= 0
    tgt = target.clamp_min(0)
    idx = tgt.unsqueeze(-1)
    logp_t = logp.gather(-1, idx).squeeze(-1)             # [B, 14]
    p_t = logp_t.exp()
    a = alpha.to(logits.device).unsqueeze(0).expand(B, -1, -1)
    a_t = a.gather(-1, idx).squeeze(-1)
    loss = -a_t * (1.0 - p_t).pow(gamma) * logp_t
    loss = loss * valid.float()
    return loss.sum() / valid.float().sum().clamp_min(1.0)


@torch.no_grad()
def supervised_stats(logits, target):
    """Vài con số để đọc log Phase 1 mà không phải chạy eval riêng.

    `acc_nonblank` mới là con số đáng nhìn: nếu 90% ô là "không nhắc tới" thì
    một model đoán bừa "không nhắc tới" đã có acc 0.9 mà chẳng học được gì.
    """
    valid = target >= 0
    pred = logits.argmax(dim=-1)
    correct = (pred == target) & valid
    acc = correct.sum().float() / valid.sum().clamp_min(1)
    # aspect 0 theo quy ước của labels.npz là lớp phổ biến nhất (blank/negative)
    nb = valid & (target > 0)
    acc_nb = ((pred == target) & nb).sum().float() / nb.sum().clamp_min(1)
    return {"sup_acc": float(acc), "sup_acc_nonblank": float(acc_nb),
            "sup_n_nonblank": int(nb.sum())}

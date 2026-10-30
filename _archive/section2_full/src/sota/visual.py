"""Nhánh thị giác của hai baseline: ResNet-101 (R2Gen) và ResNet-152 (AdaAtt).

Cả hai paper dùng đặc trưng LƯỚI (spatial grid) lấy ở tầng conv cuối, không
phải vector sau average-pool: R2Gen coi mỗi ô lưới là một "patch" đưa vào
encoder Transformer, AdaAtt attend lên chính lưới ấy. Nên ở đây `avgpool` và
`fc` bị bỏ, và đầu ra là `(B, HW, C)` + một vector toàn cục `(B, C)` lấy bằng
trung bình trên lưới.

── ĐIỀU KIỆN 4 SỐNG Ở `src/imaging.unified_tensor` ──────────────────────────
Ảnh KHÔNG được resize ở đây. Nó đã đi qua đúng một lần biến đổi hình học dùng
chung cho cả ba candidate (letterbox 512 + pad vuông); chỗ này chỉ còn chuẩn
hoá GIÁ TRỊ pixel theo thống kê ImageNet — phép biến đổi trên giá trị, không
phải trên hình học, đúng như `image_policy.per_model_normalize_only` nói.

Hệ quả: lưới đặc trưng là 16×16 = 256 ô (512/32) chứ không phải 7×7 = 49 như
khi chạy ở 224. Đó là một SAI LỆCH so với paper gốc, nó được ghi vào
`manifest.deviations`, và nó là cái giá của việc ba candidate nhìn cùng một ảnh.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Thống kê ImageNet — đi cùng trọng số ImageNet của cả hai backbone.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_ARCH = {
    "resnet-101": ("resnet101", "ResNet101_Weights"),
    "resnet101": ("resnet101", "ResNet101_Weights"),
    "resnet-152": ("resnet152", "ResNet152_Weights"),
    "resnet152": ("resnet152", "ResNet152_Weights"),
}


class VisualExtractorUnusable(RuntimeError):
    """Không dựng được backbone — dừng, đừng huấn luyện trên đặc trưng ngẫu nhiên."""


class VisualExtractor(nn.Module):
    """ResNet cắt bỏ avgpool+fc. Trả (patch_feats (B,HW,C), global_feat (B,C)).

    `pretrained=True` trong cả hai setup gốc nghĩa là trọng số ImageNet, và đó
    là trọng số DUY NHẤT được nạp: mọi khối còn lại của hai baseline khai
    `random` trong khối `initialization`, nên không có checkpoint nào của paper
    để nạp thêm. "Nạp lại trọng số ở đúng trạng thái setup của paper gốc" ở đây
    có nghĩa đúng như thế.
    """

    def __init__(self, arch: str = "ResNet-101", pretrained: bool = True):
        super().__init__()
        key = str(arch).strip().lower()
        if key not in _ARCH:
            raise VisualExtractorUnusable(
                f"backbone không biết: {arch!r}. Hợp lệ: ResNet-101, ResNet-152.")
        fn_name, weights_name = _ARCH[key]
        try:
            import torchvision.models as tvm
        except Exception as e:  # pragma: no cover - môi trường thiếu torchvision
            raise VisualExtractorUnusable(
                "không import được torchvision — hai baseline đối chứng dùng "
                "ResNet ImageNet của torchvision.\n"
                "    pip install torchvision  (khớp bản torch đang cài)") from e

        weights = None
        if pretrained:
            try:
                weights = getattr(tvm, weights_name).IMAGENET1K_V1
            except Exception as e:
                raise VisualExtractorUnusable(
                    f"không lấy được trọng số ImageNet cho {fn_name}: {e!r}.\n"
                    "    Đây KHÔNG phải chỗ để rơi về khởi tạo ngẫu nhiên: cả hai "
                    "setup gốc khai `pretrained: true`, và một backbone ngẫu "
                    "nhiên biến cột đối chứng thành một cột khác.") from e
        net = getattr(tvm, fn_name)(weights=weights)
        self.arch = fn_name
        self.out_dim = int(net.fc.in_features)          # 2048 cho cả hai
        self.body = nn.Sequential(*list(net.children())[:-2])

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.body(pixel_values)                  # (B, C, H, W)
        b, c, h, w = fmap.shape
        patches = fmap.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        return patches, patches.mean(dim=1)

    def set_trainable(self, flag: bool) -> None:
        """Bật/tắt gradient cho CẢ backbone — hai stage của AdaAtt dùng chính nó.

        Đặt `requires_grad` chứ không `eval()`: BatchNorm của ResNet vẫn phải
        chạy ở chế độ của module cha (train/eval) để thống kê chạy khớp với
        những gì paper mô tả. Đóng băng bằng `eval()` là một quyết định KHÁC và
        nó không có trong setup nào của hai paper.
        """
        for p in self.body.parameters():
            p.requires_grad_(bool(flag))


def normalize_batch(pixel_values: torch.Tensor) -> torch.Tensor:
    """Chuẩn hoá ImageNet cho tensor [0,1] (B,3,H,W). Không đụng tới hình học."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=pixel_values.dtype,
                        device=pixel_values.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=pixel_values.dtype,
                       device=pixel_values.device).view(1, 3, 1, 1)
    return (pixel_values - mean) / std

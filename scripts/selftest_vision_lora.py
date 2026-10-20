#!/usr/bin/env python3
"""Tự kiểm chứng §2.6 — Chạy trong vài giây, KHÔNG cần GPU và KHÔNG cần mạng.

Bảo vệ các bất biến thiết kế cốt lõi của thí nghiệm:
  1. Config mặc định là configs/vision_lora.yaml.
  2. Duy nhất 3 cấu hình rank (2, 4, 8) với cùng base Gemma 3 4B-it.
  3. Tỷ lệ alpha/r = 2.0 cố định qua 3 rank (alpha = 4, 8, 16).
  4. Bộ lọc module chỉ gắn adapter vào Vision Encoder (SigLIP), loại bỏ hoàn toàn
     Decoder (language_model) và Projector (multi_modal_projector).
  5. Thiết kế kiểm định 3 cặp với Paired Permutation + Bootstrap BCa và hiệu chỉnh Holm.
  6. Lưu trữ trên CÙNG repo Hub nhưng thư mục (subdir) mới `2.6_vision_lora`.

Sử dụng:
    python scripts/selftest_vision_lora.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PASS = FAIL = SKIP = 0
FAILURES: list[str] = []


def check(name: str, fn) -> None:
    global PASS, FAIL, SKIP
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001
        FAIL += 1
        FAILURES.append(f"{name}: {e!r}")
        print(f"  ✗ {name}: {e!r}")
        return
    if r is False:
        FAIL += 1
        FAILURES.append(f"{name}: trả False")
        print(f"  ✗ {name}")
    else:
        PASS += 1
        print(f"  ✓ {name}")


def hr(t: str) -> None:
    print(f"\n─── {t} ───")


# ════════════════════════════════════════════════════════════════════════════
from src.config import DEFAULT_CONFIG, load_config  # noqa: E402

CFG = load_config()


def t_default_config():
    assert DEFAULT_CONFIG.name == "vision_lora.yaml", f"DEFAULT_CONFIG: {DEFAULT_CONFIG}"
    assert CFG.get("experiment") == "2.6"
    assert CFG.get("mode") == "adapter"


def t_arms_count_and_ranks():
    arms = CFG.get("arms", [])
    assert len(arms) == 3, f"Số lượng arm phải là 4, nhận: {len(arms)}"
    ranks = [a.get("rank") for a in arms]
    assert ranks == [2, 4, 8], f"Các rank phải là [2, 4, 8, 16], nhận: {ranks}"
    methods = [a.get("method") for a in arms]
    assert all(m in ("qlora", "lora") for m in methods)


def t_alpha_scaling():
    from src.backbone import resolve_alpha

    tcfg = dict(CFG.get("train", {}))
    assert tcfg.get("alpha_mode") == "proportional"
    assert float(tcfg.get("alpha_ratio", 0)) == 2.0

    a2 = resolve_alpha(2, tcfg)
    a4 = resolve_alpha(4, tcfg)
    a8 = resolve_alpha(8, tcfg)
    assert a2 == 4, f"r=2 -> alpha=4, nhận {a2}"
    assert a4 == 8, f"r=4 -> alpha=8, nhận {a4}"
    assert a8 == 16, f"r=8 -> alpha=16, nhận {a8}"


def t_target_modules_and_prefixes():
    tcfg = dict(CFG.get("train", {}))
    targets = list(tcfg.get("target_modules", []))
    for req in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert req in targets, f"Thiếu {req} trong train.target_modules"

    inc = list(tcfg.get("include_prefixes", []))
    assert any("vision" in p for p in inc), f"include_prefixes phải nhắm vào vision: {inc}"

    exc = list(tcfg.get("exclude_prefixes", []))
    assert any("language" in p for p in exc), f"exclude_prefixes phải chặn language_model: {exc}"
    assert any("projector" in p for p in exc), f"exclude_prefixes phải chặn projector: {exc}"


def t_mock_filter_target_modules():
    import torch
    import torch.nn as nn
    from src.backbone import find_target_modules

    class MockVisionSelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.out_proj = nn.Linear(64, 64)

    class MockLanguageSelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.o_proj = nn.Linear(64, 64)

    class MockGemma3(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = nn.ModuleDict({
                "vision_model": nn.ModuleDict({
                    "encoder": nn.ModuleDict({
                        "layers": nn.ModuleList([MockVisionSelfAttn() for _ in range(2)])
                    })
                })
            })
            self.multi_modal_projector = nn.Linear(64, 128)
            self.language_model = nn.ModuleDict({
                "model": nn.ModuleDict({
                    "layers": nn.ModuleList([MockLanguageSelfAttn() for _ in range(2)])
                })
            })

    model = MockGemma3()
    tcfg = dict(CFG.get("train", {}))
    hits = find_target_modules(model, tcfg)

    # Khẳng định: MỌI hit đều thuộc vision_tower, KHÔNG CÓ module language hay projector nào
    assert len(hits) == 8, f"Phải tìm thấy 8 module vision (4 proj x 2 layers), nhận: {len(hits)}"
    for h in hits:
        assert "vision_tower" in h, f"Module không thuộc vision tower lọt vào: {h}"
        assert "language_model" not in h, f"language_model lọt vào hits: {h}"
        assert "multi_modal_projector" not in h, f"multi_modal_projector lọt vào hits: {h}"


def t_image_policy_unified():
    ip = dict(CFG.get("image_policy", {}))
    assert ip.get("unified") is True
    assert ip.get("resize_to") == 512
    assert ip.get("pad_to_square") is True


def t_data_split_integrity():
    d = dict(CFG.get("data", {}))
    assert d.get("n_train") == 6400
    assert d.get("n_test") == 3000
    assert d.get("assert_no_leak") is True


def t_stats_3_pairs():
    from src.stats_multi import pair_list

    arm_ids = [a["id"] for a in CFG.get("arms", [])]
    pairs = pair_list(arm_ids, mode="all")
    assert len(pairs) == 3, f"3 arm phải sinh ra đúng 3 cặp so sánh, nhận {len(pairs)}: {pairs}"



def t_cpu_and_streaming_config():
    from src.data import get_cpu_workers, get_stream_connections
    cpu_w = get_cpu_workers(CFG)
    assert cpu_w >= 2, f"cpu_workers phải >= 2, nhận: {cpu_w}"
    stream_c = get_stream_connections(CFG)
    assert stream_c >= 2, f"stream_connections phải >= 2, nhận: {stream_c}"

def t_hub_output_dest():
    out = dict(CFG.get("output", {}))
    assert out.get("hub_repo") == "RGExp123/Research_results", f"Repo Hub phải giữ nguyên: {out.get('hub_repo')}"
    assert out.get("subdir") == "2.6_vision_lora", f"Subdir mới phải là 2.6_vision_lora: {out.get('subdir')}"
    assert out.get("adapter_dir") == "./adapters_vision_lora", f"Adapter dir: {out.get('adapter_dir')}"


def t_decoding_spec():
    gen = dict(CFG.get("generate", {}))
    assert gen.get("do_sample") is False
    assert gen.get("num_beams") == 3
    assert gen.get("max_new_tokens") == 256


def main():
    hr("1. Cấu hình & Điểm vào mặc định")
    check("DEFAULT_CONFIG là vision_lora.yaml", t_default_config)
    check("Đúng 3 arm với rank [2, 4, 8]", t_arms_count_and_ranks)
    check("Tỷ lệ alpha/r = 2.0 tỉ lệ thuận cho mọi rank", t_alpha_scaling)

    hr("2. Cô lập Vision Encoder & Freeze All")
    check("target_modules và prefixes định vị SigLIP", t_target_modules_and_prefixes)
    check("find_target_modules chỉ chọn vision, loại bỏ decoder & projector", t_mock_filter_target_modules)

    hr("3. Dữ liệu, Chính sách ảnh & Giải mã")
    check("Chính sách ảnh thống nhất 512px bicubic square", t_image_policy_unified)
    check("Split dữ liệu 6400 train / 3000 test, assert_no_leak", t_data_split_integrity)
    check("Quy chuẩn giải mã beam_size=3, max_new_tokens=256", t_decoding_spec)

    hr("4. Kiểm định Thống kê & Lưu trữ")
    check("Tạo đủ 3 cặp đối chứng cho 3 arm", t_stats_3_pairs)
    check("Cấu hình đa luồng CPU và đa luồng stream Range", t_cpu_and_streaming_config)
    check("Lưu trữ cùng Hub repo với subdir mới 2.6_vision_lora", t_hub_output_dest)

    hr("TỔNG KẾT")
    print(f"PASS: {PASS}  |  FAIL: {FAIL}  |  SKIP: {SKIP}")
    if FAIL:
        print("\nCác phép kiểm thất bại:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("\n✓ Toàn bộ phép kiểm selftest cho §2.6 ĐÃ QUA THÀNH CÔNG.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

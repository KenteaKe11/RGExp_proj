#!/usr/bin/env python3
"""Tự kiểm chứng §2.8 — KHÔNG cần GPU, mạng, hay Gemma.

  1. Mã hoá nhãn: 1 -> [0,1], 0 -> [1,0], -1 -> mask 0, NaN -> [0,0] mask 1 (và biến thể ignore).
  2. Đọc CSV CheXbert giả (cột tên lệch hoa/thường), khớp key nguyên văn + theo tên file,
     chặn key trùng, chặn tỉ lệ thiếu nhãn.
  3. Config: 2 lớp, ASL = mặc định của AsymmetricLoss, n_epochs/batch như đã khai.
  4. Nếu có torch: ASL mặc kiểm theo ô (ô mask 0 không sinh gradient) và khớp bản numpy tham chiếu.

    python scripts/selftest_cxr_cls.py
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import cxr_cls_labels as lab  # noqa: E402
from src.config import Cfg  # noqa: E402

OK = 0


def check(cond: bool, msg: str) -> None:
    global OK
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        sys.exit(1)
    OK += 1


def raises(fn, exc=lab.LabelsUnusable) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


cfg = Cfg(yaml.safe_load(open(ROOT / "configs" / "cxr_recomb_cls.yaml", encoding="utf-8")))

print("1. Mã hoá nhãn")
raw = np.array([[1, 0, -1, np.nan]], dtype=float)
y, m = lab.encode(raw)
check(y[0, 0].tolist() == [0, 1] and m[0, 0] == 1, "1 -> [neg=0,pos=1], hợp lệ")
check(y[0, 1].tolist() == [1, 0] and m[0, 1] == 1, "0 -> [neg=1,pos=0], hợp lệ")
check(y[0, 2].tolist() == [0, 0] and m[0, 2] == 0, "-1 (uncertain) -> mask 0")
check(y[0, 3].tolist() == [0, 0] and m[0, 3] == 1, "NaN (không nhắc) -> [0,0], hợp lệ")
_, m2 = lab.encode(raw, "zeros", "ignore")
check(m2[0].tolist() == [1, 1, 1, 0], "uncertain=zeros, unmentioned=ignore đổi mask đúng ô")
check(raises(lambda: lab.encode(raw, "bogus")), "giá trị cấu hình lạ bị chặn")
check(raises(lambda: lab.to_numeric_codes(pd.Series([2.0, 1.0]))), "mã nhãn lạ (2.0) bị chặn")
codes = lab.to_numeric_codes(pd.Series(["Positive", "negative", "Uncertain", None]))
check(np.allclose(codes[:3], [1, 0, -1]) and np.isnan(codes[3]), "nhãn dạng chữ được nhận")

print("2. Đọc CSV CheXbert giả")
rng = np.random.default_rng(0)
obs_cols = [c.lower().replace(" ", "_") for c in lab.OBSERVATIONS]          # tên cột lệch kiểu
n = 40
vals = rng.choice([1.0, 0.0, -1.0, np.nan], size=(n, 14))
df = pd.DataFrame(vals, columns=obs_cols)
df.insert(0, "img_rel", [f"files/p10/s{i}/img{i}.jpg" for i in range(n)])
split = [f"files/p10/s{i}/img{i}.jpg" for i in range(0, 30)]
cfg["labels"]["max_missing_frac"] = 0.05
with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "chexbert.csv"
    df.to_csv(f, index=False)
    lut, info = lab.build_lut(cfg, split, local=f, verbose=False)
    check(info["n_matched"] == 30 and info["n_exact"] == 30, "khớp nguyên văn 30/30")
    check(info["observations"][5].lower().startswith("edema"), "thứ tự quan sát theo OBSERVATIONS")
    yy, mm = lut[split[3]]
    check(yy.shape == (14, 2) and mm.shape == (14,), "mỗi key -> (14,2) và (14,)")
    check(np.allclose(yy[..., 1], (vals[3] == 1.0)), "kênh pos đúng với mã 1")
    check(np.all(mm[vals[3] == -1.0] == 0), "ô uncertain bị mask")

    # key trong split khác dạng: chỉ tên file
    split2 = [f"x/y/img{i}.jpg" for i in range(0, 30)]
    lut2, info2 = lab.build_lut(cfg, split2, local=f, verbose=False)
    check(info2["n_by_stem"] == 30, "khớp theo tên file khi đường dẫn khác")
    split3 = [f"img_missing_{i}.jpg" for i in range(30)]
    check(raises(lambda: lab.build_lut(cfg, split3, local=f, verbose=False)),
          "100% thiếu nhãn bị chặn bởi max_missing_frac")

    # NPZ dataset contract: explicit image key vector + 14-column label matrix.
    npz = Path(td) / "label.npz"
    np.savez(npz, img_rel=np.asarray([f"files/p10/s{i}/img{i}.jpg" for i in range(n)]),
             labels=vals)
    lut_npz, info_npz = lab.build_lut(cfg, split, local=npz, verbose=False)
    check(info_npz["n_matched"] == 30 and np.allclose(lut_npz[split[3]][0][..., 1],
          (vals[3] == 1.0)), "đọc label.npz (key + ma trận N×14) và mã hoá đúng")

    dup = pd.concat([df, df.iloc[[0]]])
    dup.to_csv(f, index=False)
    check(raises(lambda: lab.build_lut(cfg, split, local=f, verbose=False)), "key trùng trong file nhãn bị chặn")

    bad = df.drop(columns=[obs_cols[2]])
    bad.to_csv(f, index=False)
    check(raises(lambda: lab.build_lut(cfg, split, local=f, verbose=False)), "thiếu cột quan sát bị chặn")

    nokey = df.drop(columns=["img_rel"])
    nokey.to_csv(f, index=False)
    check(raises(lambda: lab.build_lut(cfg, split, local=f, verbose=False)),
          "không có cột key ⇒ dừng (không căn theo vị trí)")

y3, m3 = lab.attach(["a", "b"], {"a": (np.ones((14, 2), np.float32), np.ones(14, np.float32))}, 14)
check(m3[1].sum() == 0 and m3[0].sum() == 14, "key không có nhãn -> mask 0 toàn hàng")

print("3. Config")
check(cfg["classifier"]["n_classes"] == 2, "classifier.n_classes = 2")
label_file = str(cfg["labels"].get("file") or "").replace("\\", "/").strip("./")
check(label_file == "csv_helper/labels.npz" and cfg["labels"]["search_dir"] == "csv_helper",
      "đường dẫn nhãn labels.npz được ghim trong csv_helper; search_dir đúng")
src = (ROOT / "src" / "recombiner.py").read_text(encoding="utf-8")
m_ = re.search(r"def __init__\(self, gamma_neg=([\d.]+), gamma_pos: float = ([\d.]+),\s*clip: float = ([\d.]+)", src)
dflt = tuple(float(x) for x in m_.groups())
a = cfg["classifier"]["asl"]
check((a["gamma_neg"], a["gamma_pos"], a["clip"]) == dflt, f"ASL trong config = mặc định của AsymmetricLoss {dflt}")
check(cfg["recombiner"]["n_tokens"] == cfg["recombiner"]["n_vis"] // cfg["recombiner"]["local"]["pool_factor"] ** cfg["recombiner"]["local"]["n_blocks"],
      "n_tokens = n_vis / pool_factor^n_blocks")
check(cfg["cls_train"]["schedule"]["lr_init"] <= cfg["cls_train"]["schedule"]["max_lr"], "lr_init ≤ max_lr")

print("4. ASL theo ô (cần torch)")
try:
    import torch
except ImportError:
    print("  ⊘ không có torch — bỏ qua mục 4 (chạy lại trên máy GPU)")
else:
    from src.run_cxr_cls import make_loss

    crit = make_loss(a)
    torch.manual_seed(0)
    lg = torch.randn(3, 14, 2, requires_grad=True)
    yt = torch.zeros(3, 14, 2); yt[..., 1] = (torch.rand(3, 14) > 0.7).float(); yt[..., 0] = 1 - yt[..., 1]
    mk = torch.ones(3, 14); mk[0, 4] = 0
    crit(lg, yt, mk).backward()
    check(float(lg.grad[0, 4].abs().sum()) == 0.0, "ô mask 0 không nhận gradient")
    check(float(lg.grad[0, 5].abs().sum()) > 0.0, "ô mask 1 có gradient")
    mk0 = torch.zeros(3, 14); mk0[1] = 1
    l_only = crit(lg.detach(), yt, mk0)
    l_one = crit(lg.detach()[1:2], yt[1:2], torch.ones(1, 14))
    check(abs(float(l_only) - float(l_one)) < 1e-6, "mẫu rỗng không kéo mẫu số")

    def ref(lg, y, mk, gn=a["gamma_neg"], gp=a["gamma_pos"], clip=a["clip"]):
        p = 1 / (1 + np.exp(-lg)); xn = np.minimum(1 - p + clip, 1.0)
        pt = p * y + xn * (1 - y); g = gp * y + gn * (1 - y); w = (1 - pt) ** g
        c = -(w * (y * np.log(np.clip(p, 1e-8, None)) + (1 - y) * np.log(np.clip(xn, 1e-8, None))))
        per = (c * mk[..., None]).sum((1, 2)); v = (mk.sum(1) > 0)
        return (per * v).sum() / max(1, v.sum())

    check(abs(float(crit(lg.detach(), yt, mk)) - ref(lg.detach().numpy(), yt.numpy(), mk.numpy())) < 1e-5,
          "khớp bản numpy tham chiếu")

print(f"\nSELFTEST §2.8: {OK} kiểm tra đạt.")

"""Tự kiểm cột token/proportion (top-k token PREFIX mỗi expert chọn) và bảng tổng hợp (chỉ numpy/pandas)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import s3_route_tokens as rt   # noqa: E402
from src.config import Cfg              # noqa: E402

OK = []


def check(name, cond, extra=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  [{extra}]" if extra else ""))
    OK.append(bool(cond))


class Tok:
    def convert_ids_to_tokens(self, i):
        return {10: "▁the", 11: "a\nb", 12: "."}[int(i)]


def main() -> int:
    E = 3
    dec = rt.tokenizer_decoder(Tok(), 4)
    check("decoder: văn bản đổi ▁→␣ và \\n→⏎; token ảnh → IMG[r,c] trên lưới 2×2",
          dec(10) == "␣the" and dec(11) == "a⏎b" and dec(-1) == "IMG[0,0]" and dec(-4) == "IMG[1,1]")
    # prefix: [10, 11, IMG0, IMG1, IMG2, IMG3, 12, 10]  (token 10 xuất hiện 2 lần ⇒ gộp theo id)
    ids = np.array([10, 11, -1, -1, -1, -1, 12, 10])
    W = np.array([[.6, 0, 0], [0, 1, 0], [0, 0, 1], [.2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [.2, 0, 0]])
    c = rt.sample_columns(ids, W, dec, top_k=3)
    check("cột token: top-k theo p mỗi expert; token lặp (id 10) gộp p",
          c["token"][0] == ["␣the", "IMG[0,1]"] and c["token"][1] == ["a⏎b"] and c["token"][2] == ["IMG[0,0]"], str(c["token"]))
    check("cột proportion: E0 = (0.6+0.2)/1.0 = 0.8 cho ␣the và 0.2 cho IMG[0,1]; E1, E2 chiếm 1.0",
          c["proportion"][0] == [0.8, 0.2] and c["proportion"][1] == [1.0] and c["proportion"][2] == [1.0], str(c["proportion"]))
    check("token không được chọn (p=0) không xuất hiện; mỗi expert đúng 1 danh sách", all(len(x) == E for x in c.values()))
    c1 = rt.sample_columns(ids, W, dec, top_k=1)
    check("top_k=1 chỉ giữ token đầu", [len(x) for x in c1["token"]] == [1, 1, 1] and c1["proportion"][0] == [0.8])
    json.dumps(c)                                                      # phải ghi được vào jsonl

    # 3 ảnh: ảnh 0,1 như trên; ảnh 2 expert 0 chọn IMG[1,1] thay vì ␣the
    ids2 = ids
    W2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0]], float)
    preds = {"a": {"key": "a", **c}, "b": {"key": "b", **c}, "c": {"key": "c", **rt.sample_columns(ids2, W2, dec, 3)},
             "old": {"key": "old", "pred": "x"}}
    df, n_s = rt.aggregate_preds(preds, E)
    e0 = df[df.expert == "E0"].set_index("unit")
    check("aggregate: bỏ dòng thiếu cột; 3 ảnh", n_s == 3)
    check("aggregate E0: ␣the mean_proportion=(0.8+0.8)/3, freq=2/3; IMG[1,1] mean=1/3 freq=1/3; IMG[0,1] mean=0.4/3",
          abs(e0.loc["␣the", "mean_proportion"] - 1.6 / 3) < 1e-4 and abs(e0.loc["␣the", "freq"] - 2 / 3) < 1e-9
          and abs(e0.loc["IMG[1,1]", "mean_proportion"] - 1 / 3) < 1e-4 and abs(e0.loc["IMG[0,1]", "mean_proportion"] - 0.4 / 3) < 1e-4)
    check("aggregate: nhóm text/image theo tên", e0.loc["␣the", "group"] == "text" and e0.loc["IMG[1,1]", "group"] == "image")
    tt = rt.top_table(df, E, 5)
    r0 = tt[(tt.group == "text") & (tt.expert == "E0")].sort_values("rank")
    check("top_table: xếp giảm dần theo mean_proportion trong nhóm", r0.iloc[0].unit == "␣the")
    md = rt.to_markdown(tt, n_s, E, 5)
    check("markdown: hai bảng (văn bản + patch ảnh), đủ cột expert", md.count("####") == 2 and "| # | E0 | E1 | E2 |" in md and "IMG[1,1]" in md)
    with tempfile.TemporaryDirectory() as td:
        cfg = Cfg({"route_tokens": {"enabled": True, "top_k": 5}})
        r = rt.write(cfg, "ac_e4", preds, E, Path(td))
        check("write: ghi csv + md", r is not None and r["n_samples"] == 3 and (Path(td) / "ac_e4_route_tokens.csv").exists()
              and (Path(td) / "ac_e4_route_tokens.md").exists())
        check("write: tắt bằng route_tokens.enabled=false", rt.write(Cfg({"route_tokens": {"enabled": False}}), "x", preds, E, Path(td)) is None)
        check("write: predictions chưa có cột ⇒ None (không vỡ)", rt.write(cfg, "y", {"c": {"x": 1}}, E, Path(td)) is None)
        try:
            import pandas as pd
            from PIL import Image
            from src import s3_route_viz as rv
            Image.new("RGB", (64, 64), (120, 120, 120)).save(Path(td) / "x.jpg")
            Wm = np.random.RandomState(0).rand(16, 16, E).astype("float32") * (np.random.RandomState(1).rand(16, 16, E) > .4)
            pv = {"k1": {"route_viz": {"img": (Wm.reshape(256, E) * 1000).astype(int).tolist(), "gen_tok": ["Mild", " cardio", "megaly"], "gen_w": ((np.random.rand(3, E) > .5) * 400).astype(int).tolist()}}}
            tdf = pd.DataFrame({"key": ["k1", "k2"], "image_path": [str(Path(td) / "x.jpg")] * 2, "ref": ["Mild cardiomegaly.", "x"]})
            sv = rv.write(Cfg({}), "ac_e4", pv, tdf, E, Path(td), ["k1", "k2"])
            check("route_viz: vẽ 2 ảnh cho mẫu có route_viz, bỏ qua mẫu thiếu",
                  len(sv) == 1 and (Path(td) / "ac_e4_route_viz_1_img.png").exists() and (Path(td) / "ac_e4_route_viz_1_text.png").exists()
                  and (Path(td) / "ac_e4_route_viz.json").exists())
        except ImportError as e:
            print(f"  (bỏ qua test route_viz: {e})")
        try:
            from src.run_section3 import count_preds, need_field
            p = Path(td) / "p.jsonl"
            p.write_text("\n".join(json.dumps({"key": k, **({"token": [["a"]], "proportion": [[1.0]]} if k != "k2" else {})})
                                   for k in ("k1", "k2", "k3")) + "\n")
            check("count_preds(field): chỉ đếm dòng có cột token", count_preds(p) == 3 and count_preds(p, "token") == 2)
            moe = {"kind": "ft", "decoder": "moe"}
            check("need_field: MoE cần cột token (khi bật), baseline/dora thì không",
                  need_field(cfg, moe) == "token" and need_field(cfg, {"kind": "infer"}) is None
                  and need_field(Cfg({"route_tokens": {"enabled": False}}), moe) is None)
        except ImportError as e:
            print(f"  (bỏ qua test run_section3: {e})")
    bad = OK.count(False)
    print(f"\n{'✓ TẤT CẢ' if not bad else '✗ HỎNG'} {len(OK) - bad}/{len(OK)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Tự kiểm chứng `sup_pretrain` + `decoder_ft` — không GPU, không mạng, không model.

Chạy:  python scripts/selftest_pretrain.py        (thêm -v để xem traceback)

Cái nó bảo vệ là bốn ca hỏng cho ra một lần chạy TRÔNG HOÀN TOÀN BÌNH THƯỜNG:

  1. File checkpoint chứa SAI tập khối. Ghi cả `route_head`/`dec_proj` vào file
     pretrain thì file trông như một checkpoint đầy đủ của nhánh, và người nạp
     nó vào main-stage sẽ tưởng Route Head đã được pretrain — trong khi đó là số
     ngẫu nhiên. Thiếu khoá theo chiều ngược lại thì một phần của khối "đã pretrain"
     là ngẫu nhiên, còn META vẫn nói nó đã học.

  2. Nhóm learning-rate áp nhầm chỗ. Ở decoder_ft phần đã pretrain chạy
     pretrain max_lr / 2, KHÔNG áp lên `route_head`/`dec_proj` (còn tươi); ma
     trận Attn của Recombinator luôn × 3 (Stage 0 lẫn decoder_ft). Áp nhầm
     nghĩa là router học chậm, và bảng định tuyến sẽ đầy expert chết vì một lý
     do không liên quan tới MoE.

  3. Route Head lọt vào pretrain. Ở Stage 0 không có Experts, nên Route Head
     không được gọi và không được nhận gradient; nếu nó có gradient thì file
     pretrain nói sai về chính nó.

  4. Resume đọc sai vị trí. Ckpt ghi theo epoch phải khôi phục ĐÚNG trạng thái
     Generator + cursor: sai một chỗ thì lần chạy tiếp theo lặp lại (hoặc bỏ
     qua) một phần dữ liệu, và không có gì trong log nói vậy.

Cộng thêm: kiến trúc nhánh khớp sơ đồ (Recombinator 4096→1024→256 token,
MHA 18 head, Pooling Stack 256→64→16→4→1, dec_proj 256→64), checkpoint
pretrain nạp được sang main-stage (Route Head dựng theo E của candidate),
`assert_compatible` bắt được lệch n_labels / format cũ, λ-warmup bị TẮT ở
decoder_ft, và `pretrain_rcfg` tắt định tuyến.

KHÔNG kiểm chất lượng model. Kiểm rằng những gì hai chế độ TUYÊN BỐ làm thì
chúng có làm.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = [], []


def check(name: str, fn) -> None:
    try:
        fn()
        PASS.append(name)
    except Exception as e:
        FAIL.append((name, f"{type(e).__name__}: {e}", traceback.format_exc()))


def true(c, msg=""):
    if not c:
        raise AssertionError(msg or "sai")


def eq(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg} {a!r} != {b!r}")


def raises(exc, fn, msg=""):
    try:
        fn()
    except exc:
        return
    except Exception as e:
        raise AssertionError(
            f"{msg}: chờ {exc.__name__}, nhận {type(e).__name__}: {e}") from None
    raise AssertionError(f"{msg}: chờ {exc.__name__}, nhưng KHÔNG có lỗi nào")


# ── một nhánh nhỏ, cùng đường code với nhánh thật ───────────────────────────
D_VIS, D_MODEL, N_LABELS = 16, 24, 5
RCFG_BASE = {
    # Lưới 16×16 -> 8×8 -> 4×4: cùng CẤU TRÚC với 64×64 -> 32×32 -> 16×16 thật,
    # nhỏ đủ để chạy trên CPU.
    "siglip_dim": D_VIS, "out_dim": None, "n_vis": 256, "n_tokens": 16,
    "local": {"n_blocks": 2, "hidden": 20, "pool_factor": 4},
    "head_dim": 4, "dropout": 0.0, "pool_stages": 2,
    "route_head": {"hidden": 12, "dropout": 0.0},
    "n_dec_tokens": 4,
    "n_experts": 4, "route_noise_init": -3.0, "route_enabled": False,
}
CCFG = {"n_classes": 3, "hidden": 16, "dropout": 0.0,
        "asl": {"gamma_neg": 2.0, "gamma_pos": 0.0, "clip": 0.05, "eps": 1e-8}}


def _branch(rcfg=None, n_labels=N_LABELS):
    from src import recombiner as R

    import copy
    r = copy.deepcopy(rcfg or RCFG_BASE)
    return R.RecombinerBranch(D_VIS, D_MODEL, n_labels, r, CCFG)


# ═══ 1. Tập khối trong checkpoint ════════════════════════════════════════════
def io_tests() -> None:
    from src import pretrain_io as P

    def t_disjoint():
        overlap = set(P.PRETRAINED_MODULES) & set(P.UNTRAINED_MODULES)
        true(not overlap, f"hai danh sách giao nhau: {overlap}")
    check("PRETRAINED/UNTRAINED_MODULES rời nhau", t_disjoint)

    def t_cover():
        b = _branch()
        mods = {k.split(".", 1)[0] for k in b.state_dict()}
        known = set(P.PRETRAINED_MODULES) | set(P.UNTRAINED_MODULES)
        true(mods <= known,
             f"nhánh có khối không được khai: {sorted(mods - known)}")
        # Và ngược lại: mọi khối đã khai phải THẬT SỰ có trong nhánh — một tên
        # gõ nhầm ở PRETRAINED_MODULES thì `split_state` bỏ khối ấy ra khỏi file
        # mà không ai biết.
        for m in P.PRETRAINED_MODULES:
            true(any(k.split(".", 1)[0] == m for k in b.state_dict()),
                 f"khối {m!r} khai là được pretrain nhưng không có trong nhánh")
    check("mọi khối của nhánh đều được khai (cả hai chiều)", t_cover)

    def t_split():
        b = _branch()
        keep, drop = P.split_state(b)
        true(all(k.split(".", 1)[0] in P.PRETRAINED_MODULES for k in keep))
        true(all(k.split(".", 1)[0] in P.UNTRAINED_MODULES for k in drop))
        true(any(k.startswith("dec_proj.") for k in drop),
             "`dec_proj` phải nằm ngoài file pretrain")
        true(not any(k.startswith("dec_proj.") for k in keep))
        true(any(k.startswith("route_head.") for k in drop),
             "`route_head` phải nằm ngoài file pretrain")
        true(not any(k.startswith("route_head.") for k in keep))
        eq(len(keep) + len(drop), len(b.state_dict()), "không được mất khoá")
    check("split_state: đúng khối vào file, dec_proj/route_head ở ngoài", t_split)

    def t_unknown_module():
        import torch.nn as nn

        b = _branch()
        b.mystery = nn.Linear(2, 2)          # ai đó vừa thêm một khối
        raises(P.PretrainIncompatible, lambda: P.split_state(b),
               "khối lạ phải DỪNG, không được rơi vào nhóm nào cũng được")
    check("khối mới chưa khai -> DỪNG", t_unknown_module)

    def t_roundtrip():
        import torch

        b = _branch()
        with torch.no_grad():                 # làm cho trọng số khác mặc định
            for p in b.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        with tempfile.TemporaryDirectory() as d:
            P.save_pretrain(b, Path(d), {"epoch": 3, "val": {"cls_f1_pos": 0.42}})
            meta = P.read_meta(Path(d))
            eq(meta["format"], P.FORMAT)
            eq(sorted(meta["pretrained_modules"]), sorted(P.PRETRAINED_MODULES))
            true(all(k.split(".", 1)[0] in P.PRETRAINED_MODULES
                     for k in meta["keys"]), "META khai khoá ngoài danh sách")
            true(any(k.startswith("dec_proj.") for k in meta["keys_not_saved"]))

            b2 = _branch()
            rep = P.load_into_branch(b2, Path(d), verbose=False)
            eq(sorted(rep["fresh_modules"]), sorted(P.UNTRAINED_MODULES))
            for k, v in b.state_dict().items():
                if k.split(".", 1)[0] in P.PRETRAINED_MODULES:
                    true(torch.allclose(v, b2.state_dict()[k]),
                         f"khoá {k} không khớp sau khi nạp lại")
            # Và khối KHÔNG pretrain phải giữ giá trị khởi tạo của b2, không
            # phải của b: nếu nó khớp b thì file đã lén mang chúng theo.
            from src.recombiner import avg_pool_matrix

            w2 = b2.dec_proj.proj.weight
            true(torch.allclose(w2, avg_pool_matrix(w2.shape[1], w2.shape[0])),
                 "`dec_proj` của b2 phải giữ khởi tạo, không được lấy từ file")
            true(not torch.allclose(b.dec_proj.proj.weight, w2),
                 "`dec_proj` của b đã bị làm lệch — phải khác b2")
    check("ghi rồi nạp lại: đúng trọng số, đúng khối tươi", t_roundtrip)

    def t_truncated():
        import torch
        from safetensors.torch import load_file, save_file

        b = _branch()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            P.save_pretrain(b, d, {})
            sd = load_file(str(d / P.PRETRAIN_FILE))
            victim = next(k for k in sd if k.startswith("cls_head."))
            del sd[victim]
            save_file(sd, str(d / P.PRETRAIN_FILE))
            raises(P.PretrainIncompatible,
                   lambda: P.load_into_branch(_branch(), d, verbose=False),
                   "file thiếu khoá của khối đã pretrain phải DỪNG")
    check("file bị cắt xén -> DỪNG (không nạp một phần)", t_truncated)

    def t_extra_key():
        import torch
        from safetensors.torch import load_file, save_file

        b = _branch()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            P.save_pretrain(b, d, {})
            sd = load_file(str(d / P.PRETRAIN_FILE))
            sd["cls_head.head.fc9.weight"] = torch.zeros(2, 2)
            save_file(sd, str(d / P.PRETRAIN_FILE))
            raises(P.PretrainIncompatible,
                   lambda: P.load_into_branch(_branch(), d, verbose=False),
                   "khoá lạ trong file phải DỪNG")
    check("file có khoá lạ -> DỪNG", t_extra_key)

    def t_n_labels_mismatch():
        b = _branch()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            P.save_pretrain(b, d, {})
            other = _branch(n_labels=N_LABELS + 2)
            raises(P.PretrainIncompatible,
                   lambda: P.load_into_branch(other, d, verbose=False),
                   "lệch n_labels phải DỪNG")
    check("lệch n_labels -> DỪNG (ClsHead của bộ nhãn khác)", t_n_labels_mismatch)

    def t_old_format():
        """Checkpoint PDVR + QFormer (format /1) phải DỪNG với lý do rõ ràng."""
        import json

        b = _branch()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            P.save_pretrain(b, d, {})
            mp = d / P.PRETRAIN_META
            meta = json.loads(mp.read_text(encoding="utf-8"))
            meta["format"] = "rgexp.sup_pretrain/1"
            mp.write_text(json.dumps(meta), encoding="utf-8")
            try:
                P.load_into_branch(_branch(), d, verbose=False)
            except P.PretrainIncompatible as e:
                true("format" in str(e), f"thông báo phải nói về format: {e}")
            else:
                raise AssertionError("format /1 phải DỪNG")
    check("checkpoint format cũ (PDVR/QFormer) -> DỪNG", t_old_format)

    def t_pool_stage_mismatch():
        import copy

        b = _branch()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            P.save_pretrain(b, d, {})
            r = copy.deepcopy(RCFG_BASE)
            r["n_vis"], r["n_tokens"], r["pool_stages"] = 1024, 64, 3
            raises(P.PretrainIncompatible,
                   lambda: P.load_into_branch(_branch(r), d, verbose=False),
                   "lệch n_tokens / pool_stages phải DỪNG")
    check("lệch n_tokens/pool_stages -> DỪNG", t_pool_stage_mismatch)


# ═══ 2. Hình dạng nạp thẳng sang §2.4 ════════════════════════════════════════
def compat_tests() -> None:
    import copy

    from src import pretrain_io as P
    from src import train_recomb as T

    def t_route_head_built():
        b = _branch()
        eq(b.route_head.n_experts, RCFG_BASE["n_experts"],
           "pretrain vẫn DỰNG Route Head (để hình dạng nhánh khớp main-stage)")
        eq(b.route_enabled, False)
    check("pretrain dựng Route Head nhưng tắt định tuyến", t_route_head_built)

    def t_into_moe():
        """Nhánh main-stage (route_enabled=True, E khác) nạp được file pretrain."""
        from src.config import Cfg

        b = _branch()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            P.save_pretrain(b, d, {})
            # ĐÚNG hàm mà §2.4 dùng để dựng rcfg, không phải một bản chép tay.
            cfg = Cfg({"recombiner": copy.deepcopy(RCFG_BASE)})
            spec = {"n_experts": 6, "noise_init": -3.0}
            r24 = T.effective_rcfg(cfg, spec)
            eq(r24["route_enabled"], True)
            b24 = _branch(r24)
            rep = P.load_into_branch(b24, d, verbose=False)
            eq(sorted(rep["fresh_modules"]), sorted(P.UNTRAINED_MODULES))
            eq(b24.route_head.n_experts, 6,
               "route_head dựng theo n_experts của candidate, KHÔNG lấy từ "
               "file pretrain")
            import torch
            _t, _l, route = b24(torch.randn(2, RCFG_BASE["n_vis"], D_VIS))
            true(route is not None and tuple(route[0].shape) == (2, 6),
                 "main-stage phải trả route (B, E)")
    check("checkpoint pretrain nạp thẳng vào nhánh main-stage (E khác)", t_into_moe)


# ═══ 3. Route Head / dec_proj ngoài pretrain ════════════════════════════════
def freeze_tests() -> None:
    from src import pretrain_sup as PS
    from src.config import Cfg

    def t_rcfg_forces():
        cfg = Cfg({"recombiner": {"route_enabled": True, "n_experts": 9,
                                  "n_dec_tokens": 64}})
        r = PS.pretrain_rcfg(cfg)
        eq(r["route_enabled"], False, "pretrain không định tuyến gì")
        eq(r["n_experts"], 9, "các khoá khác giữ nguyên")
        eq(r["n_dec_tokens"], 64)
    check("pretrain_rcfg ép route_enabled=false, giữ nguyên phần còn lại",
          t_rcfg_forces)

    def t_asl_reaches_recombinator():
        """ASL chảy tới Recombinator + ClsHead; route=None ở pretrain."""
        import torch

        b = _branch().float()
        vis = torch.randn(2, RCFG_BASE["n_vis"], D_VIS)
        _tok, logits, route = b(vis)
        true(route is None, "route_enabled=false phải trả None")
        y = torch.zeros(2, N_LABELS, 3)
        y[:, :, 0] = 1.0
        b.sup_loss(logits, y, torch.ones(2)).backward()
        for name in ("recombinator", "cls_head"):
            mod = getattr(b, name)
            gs = [p.grad for p in mod.parameters() if p.grad is not None]
            true(gs and any(float(g.abs().sum()) > 0 for g in gs),
                 f"{name} phải nhận gradient từ ASL")
        for k in ("local", "global", "ffn"):
            m = getattr(b.recombinator, {"local": "local_extractors",
                                         "global": "global_extractor",
                                         "ffn": "ffn"}[k])
            gs = [p.grad for p in m.parameters() if p.grad is not None]
            true(gs and any(float(g.abs().sum()) > 0 for g in gs),
                 f"khối {k} của Recombinator phải nhận gradient")
        from src import recombiner as R
        gn = R.recombinator_grad_norms(b)
        eq(sorted(gn), sorted(f"{k}_mean_per_param" for k in R.GRAD_GROUPS))
        true(all(v > 0 for v in gn.values()), f"grad norm phải > 0: {gn}")
    check("ASL tới Recombinator (local/global/ffn) + ClsHead", t_asl_reaches_recombinator)

    def t_dec_route_no_grad():
        """`dec_proj` và `route_head` không nhận gradient khi decoder không chạy."""
        import torch

        b = _branch().float()
        vis = torch.randn(2, RCFG_BASE["n_vis"], D_VIS)
        _tok, logits, _r = b(vis)
        y = torch.zeros(2, N_LABELS, 3)
        y[:, :, 1] = 1.0
        b.sup_loss(logits, y, torch.ones(2)).backward()
        gw = b.dec_proj.proj.weight.grad
        true(gw is None or float(gw.abs().sum()) == 0.0,
             "`dec_proj` không được nhận gradient — đó là lý do nó ở ngoài file")
        for n, p in b.route_head.named_parameters():
            true(p.grad is None or float(p.grad.abs().sum()) == 0.0,
                 f"route_head.{n} không được nhận gradient ở pretrain")
    check("dec_proj / route_head không nhận gradient -> đúng là UNTRAINED",
          t_dec_route_no_grad)

    def t_trainable_list():
        from src import pretrain_io as P

        b = _branch()
        names = {n.split(".", 1)[0] for n, _ in PS.trainable_params(b)}
        eq(sorted(names), sorted(P.PRETRAINED_MODULES),
           "optimizer của pretrain phải nhận ĐÚNG PRETRAINED_MODULES")
    check("optimizer pretrain = đúng PRETRAINED_MODULES", t_trainable_list)


# ═══ 3b. Vào decoder: Linear/GELU trên trục token, không cổng ═══════════════
def dec_proj_tests() -> None:
    from src import recombiner as R

    def t_shape_and_no_gate():
        import torch

        b = _branch()
        true(not hasattr(b, "gate") and "gate" not in dict(b.named_parameters()),
             "nhánh không được còn tham số `gate`")
        tok, _l, _r = b(torch.randn(3, RCFG_BASE["n_vis"], D_VIS))
        eq(tuple(tok.shape), (3, RCFG_BASE["n_dec_tokens"], D_MODEL),
           "token ra decoder phải là (B, n_dec_tokens, d_model)")
    check("nhánh trả (B, n_dec_tokens, d) và không còn `gate`", t_shape_and_no_gate)

    def t_is_linear_gelu_on_recombinator():
        """Token ra = GELU(W · Recombinator(vis)) trên trục token."""
        import torch
        import torch.nn.functional as F

        b = _branch().eval()
        vis = torch.randn(2, RCFG_BASE["n_vis"], D_VIS)
        with torch.no_grad():
            tok, _l, _r = b(vis)
            rec = b.recombinator(vis)
            want = F.gelu(torch.einsum("oi,bid->bod", b.dec_proj.proj.weight, rec))
        true(torch.allclose(tok, want, atol=1e-6),
             "token vào decoder phải đúng bằng GELU(Linear(Recombinator))")
        eq(b.dec_proj.proj.bias, None, "dec_proj không có bias")
    check("token vào decoder = Linear/GELU trục token của Recombinator",
          t_is_linear_gelu_on_recombinator)

    def t_init_is_cluster_mean():
        import torch

        w = R.avg_pool_matrix(64, 32)          # 32 không chính phương -> cặp
        eq(tuple(w.shape), (32, 64))
        true(torch.allclose(w.sum(1), torch.ones(32)), "mỗi hàng là trung bình")
        eq(float(w[0, 0]), 0.5)
        eq(float(w[0, 1]), 0.5)
        eq(float(w[31, 63]), 0.5)
        w2 = R.avg_pool_matrix(256, 64)        # 16×16 -> 8×8: ô 2×2
        true(torch.allclose(w2.sum(1), torch.ones(64)))
        eq(sorted(w2[0].nonzero().flatten().tolist()), [0, 1, 16, 17],
           "hàng 0 phải là ô 2×2 góc trên-trái của lưới 16×16")
        eq(float(w2[0, 16]), 0.25)
        stats = _branch().dec_stats()
        eq(stats["dec_proj_rel_change"], 0.0, "mới dựng thì chưa rời khởi tạo")
    check("khởi tạo dec_proj = trung bình cụm (2×2 trên lưới)", t_init_is_cluster_mean)

    def t_grid_pool_matches_matrix():
        """avg_pool_tokens(grid) == ma trận 2×2 của avg_pool_matrix."""
        import torch

        x = torch.randn(3, 256, 5)
        got = R.avg_pool_tokens(x, 4, grid=True)
        want = torch.einsum("oi,bid->bod", R.avg_pool_matrix(256, 64), x)
        true(torch.allclose(got, want, atol=1e-6),
             "AvgPool của Local Extractor phải là pool 2×2 trên lưới ảnh")
        flat = R.avg_pool_tokens(x, 4, grid=False)
        true(not torch.allclose(got, flat),
             "pool lưới phải KHÁC gộp 4 token liên tiếp")
    check("AvgPool Local Extractor = 2×2 trên lưới, không phải 1×4",
          t_grid_pool_matches_matrix)

    def t_ce_reaches_dec_proj():
        import torch

        b = _branch()
        tok, _l, _r = b(torch.randn(2, RCFG_BASE["n_vis"], D_VIS))
        tok.pow(2).mean().backward()
        g = b.dec_proj.proj.weight.grad
        true(g is not None and float(g.abs().sum()) > 0,
             "gradient từ decoder phải tới dec_proj")
        gp = [p.grad for p in b.recombinator.parameters() if p.grad is not None]
        true(gp and any(float(x.abs().sum()) > 0 for x in gp),
             "và đi tiếp tới Recombinator — không có cổng nào chặn")
    check("gradient phía decoder tới dec_proj và Recombinator", t_ce_reaches_dec_proj)

    def t_stale_keys_stop():
        import copy

        for k, v in (("gate_init_logit", -2.5), ("gate", {"beta": 0.66}),
                     ("gate_init", 0.0)):
            r = copy.deepcopy(RCFG_BASE)
            r[k] = v
            raises(ValueError, lambda r=r: _branch(r),
                   f"recombiner.{k} còn sót phải DỪNG")
        for k, v in (("qformer", {"tasks": ["sup", "aux"]}),
                     ("prepare_pool_factor", 4), ("highway_token_dims", [512]),
                     ("swiglu_dropout", 0.05), ("n_query", 64), ("n_heads", 18)):
            r = copy.deepcopy(RCFG_BASE)
            r[k] = v
            raises(ValueError, lambda r=r: _branch(r),
                   f"recombiner.{k} (PDVR/QFormer cũ) còn sót phải DỪNG")
        r = copy.deepcopy(RCFG_BASE)
        r["n_dec_tokens"] = r["n_tokens"] + 1
        raises(ValueError, lambda: _branch(r), "n_dec_tokens > n_tokens phải DỪNG")
        r = copy.deepcopy(RCFG_BASE)
        r["n_tokens"] = 32
        raises(ValueError, lambda: _branch(r), "n_tokens khai sai phải DỪNG")
        r = copy.deepcopy(RCFG_BASE)
        r["pool_stages"] = 3
        raises(ValueError, lambda: _branch(r),
               "Pooling Stack không ra đúng 1 token phải DỪNG")

        from src import train_recomb as T
        from src.config import Cfg

        ok = Cfg({"recombiner": {}, "train": {}, "classifier": {}})
        eq(T.stale_gate_keys(ok), [])
        eq(T.stale_recomb_keys(ok), [])
        true(T.stale_recomb_keys(Cfg({"recombiner": {"qformer": {}}})),
             "recombiner.qformer còn sót phải bị preflight chặn")
        for blk, k in (("train", "gate_lr_mult"), ("classifier", "l0_lambda")):
            bad = Cfg({"recombiner": {}, "train": {}, "classifier": {}})
            bad[blk][k] = 1.0
            true(T.stale_gate_keys(bad), f"{blk}.{k} còn sót phải bị chặn")
    check("khoá của cổng / PDVR / QFormer cũ còn sót -> DỪNG", t_stale_keys_stop)

    def t_config_clean():
        from src.config import load_config
        from src import train_recomb as T

        cfg = load_config()
        eq(T.stale_gate_keys(cfg), [], "config mặc định còn khoá cổng cũ")
        eq(T.stale_recomb_keys(cfg), [], "config mặc định còn khoá PDVR/QFormer")
        r = cfg["recombiner"]
        eq(int(r["n_dec_tokens"]), 64)
        eq(int(r["n_tokens"]), 256)
    check("config: n_dec_tokens = 64, n_tokens = 256, không còn khoá cũ",
          t_config_clean)


# ═══ 3c. Kiến trúc khớp sơ đồ, ở KÍCH THƯỚC THẬT của config ═════════════════
def arch_tests() -> None:
    import copy

    from src import recombiner as R
    from src.config import load_config

    def _real(route=True):
        cfg = load_config()
        r = copy.deepcopy(dict(cfg["recombiner"]))
        r["route_enabled"] = route
        c = copy.deepcopy(dict(cfg["classifier"]))
        return R.RecombinerBranch(1152, 2560, 14, r, c).eval()

    def t_recombinator_shapes():
        import torch

        b = _real()
        rec = b.recombinator
        eq(len(rec.local_extractors), 2, "Local Extractors ×2")
        eq(rec.local_extractors[0].up.out_features, 2048)
        eq(rec.global_extractor.n_heads, 18, "MHA head_dim=64 -> 1152/64 = 18")
        eq(rec.global_extractor.attn.d_head, 64)
        x = torch.randn(1, 4096, 1152)
        shapes = []
        with torch.no_grad():
            h = x
            for blk in rec.local_extractors:
                h = blk(h)
                shapes.append(tuple(h.shape))
            g = rec.global_extractor(h, x)
            y = rec.ffn(g)
        eq(shapes, [(1, 1024, 1152), (1, 256, 1152)], "Local Extractor")
        eq(tuple(g.shape), (1, 256, 1152), "Global Extractor")
        eq(tuple(y.shape), (1, 256, 2560), "FFN -> Y [256, 2560]")
    check("Recombinator: 4096→1024→256 token, 18 head, Y (256, 2560)",
          t_recombinator_shapes)

    def t_heads_shapes():
        import torch

        b = _real()
        eq([s.n_in for s in b.cls_head.head.pool.stages], [256, 64, 16, 4],
           "Pooling Stack: 256 -> 64 -> 16 -> 4 -> 1")
        eq([s.mix.n_out for s in b.cls_head.head.pool.stages], [128, 32, 8, 2],
           "Linear N -> N/2 trong từng tầng")
        eq(b.cls_head.head.fc1.out_features, 1024)
        eq(b.cls_head.head.fc2.out_features, 14 * 3, "target = 14 * 3")
        eq(b.route_head.head.fc2.out_features, 10, "target = Experts")
        x = torch.randn(2, 4096, 1152)
        with torch.no_grad():
            tok, logits, route = b(x)
            rec = b.recombinator(x)
            pooled = b.cls_head.head.pool(rec)
        eq(tuple(pooled.shape), (2, 2560), "Pooling Stack ra [, 2560]")
        eq(tuple(logits.shape), (2, 14, 3))
        eq(tuple(route[0].shape), (2, 10))
        eq(tuple(route[1].shape), (2, 10))
        eq(tuple(tok.shape), (2, 64, 2560), "dec_proj 256 -> 64")
        true(b.cls_head.head.pool is not b.route_head.head.pool,
             "hai head có Pooling Stack RIÊNG")
    check("ClsHead/RouteHead: PoolingStack 256→1, target 14·3 / E; dec 64 token",
          t_heads_shapes)

    def t_logits_not_gelu_bounded():
        """Main Head không GELU sau lớp cuối: logit phải xuống được dưới −0.17."""
        import torch

        b = _real()
        with torch.no_grad():
            b.cls_head.head.fc2.bias.fill_(-3.0)
            _t, logits, _r = b(torch.randn(1, 4096, 1152))
        true(float(logits.min()) < -1.0,
             "logit bị chặn dưới — có GELU sau lớp cuối của Main Head")
    check("Main Head trả logit tuyến tính (không GELU cuối)",
          t_logits_not_gelu_bounded)

    def t_residuals():
        """Residual lấy từ đầu ra RMSNorm, đúng như sơ đồ."""
        import torch
        import torch.nn as nn

        b = _real()
        ge, ffn = b.recombinator.global_extractor, b.recombinator.ffn
        q = torch.randn(1, 256, 1152)
        kv = torch.randn(1, 4096, 1152)
        with torch.no_grad():
            nn.init.zeros_(ge.attn.o_proj.weight)
            eq_ = torch.allclose(ge(q, kv), ge.norm_q(q), atol=1e-5)
            true(eq_, "MHA = 0 thì Global Extractor phải trả RMSNorm(query)")
            nn.init.zeros_(ffn.fc.weight)
            nn.init.zeros_(ffn.fc.bias)
            h = ffn.norm(torch.nn.functional.gelu(ffn.up(q)))
            true(torch.allclose(ffn(q), h + torch.nn.functional.gelu(torch.zeros(())),
                                atol=1e-5),
                 "FFN: residual phải là đầu ra RMSNorm(2560)")
    check("residual của Global Extractor và FFN lấy từ RMSNorm", t_residuals)

    def t_roundtrip_branch():
        import torch

        b = _branch(dict(RCFG_BASE, route_enabled=True))
        with tempfile.TemporaryDirectory() as d:
            R.save_branch(b, Path(d))
            b2 = R.load_branch(Path(d))
            x = torch.randn(2, RCFG_BASE["n_vis"], D_VIS)
            b.eval()
            with torch.no_grad():
                o1, o2 = b(x), b2(x)
            true(torch.allclose(o1[0], o2[0]) and torch.allclose(o1[1], o2[1])
                 and torch.allclose(o1[2][0], o2[2][0]),
                 "save_branch/load_branch phải cho lại đúng nhánh")
    check("save_branch -> load_branch dựng lại đúng nhánh", t_roundtrip_branch)


# ═══ 4. Nhóm learning-rate của decoder_ft ════════════════════════════════════
class _FakePeft:
    """Đủ để `build_optimizer` chạy: một tham số requires_grad."""

    def __init__(self):
        import torch

        self._p = [torch.nn.Parameter(torch.zeros(3))]

    def parameters(self):
        return iter(self._p)


def ids_of(mod) -> set:
    return {id(p) for p in mod.parameters()}


LRS = {"main": 3e-4, "pretrain_max_lr": 2e-4, "pretrained_div": 2.0,
       "pretrained": 1e-4, "attn_mult": 3.0}


def _close(a, b, msg=""):
    if abs(float(a) - float(b)) > 1e-12 * max(1.0, abs(float(b))):
        raise AssertionError(f"{msg} {a!r} != {b!r}")


def lr_tests() -> None:
    from src import pretrain_io as P
    from src import pretrain_sup as S
    from src import recombiner as R
    from src import train_recomb as T
    from src.backbone import TrainUnusable

    TCFG = {"schedule": {"max_lr": 3e-4}, "weight_decay": 0.0,
            "branch_weight_decay": 0.01}

    def attn_ids(b):
        return {id(p) for n, p in b.named_parameters() if R.is_attn_param(n)}

    def t_attn_params_are_qkvo():
        b = _branch()
        names = [n for n, _ in b.named_parameters() if R.is_attn_param(n)]
        eq(sorted(n.rsplit(".", 2)[-2] for n in names),
           ["k_proj", "o_proj", "q_proj", "v_proj"], "Attn = Q/K/V/O")
        eq(attn_ids(b), ids_of(b.recombinator.global_extractor.attn))
    check("ma trận Attn = Q/K/V/O của Global Extractor", t_attn_params_are_qkvo)

    def t_pretrain_groups():
        b = _branch()
        pcfg = {"schedule": {"max_lr": 2e-4}, "weight_decay": 0.8}
        opt, n = S.build_optimizer(b, pcfg, {"attn_lr_mult": 3.0})
        by = {g["name"]: g for g in opt.param_groups}
        eq([g["name"] for g in opt.param_groups],
           ["branch_sup", "branch_sup_attn"], "nhóm Stage 0")
        _close(by["branch_sup"]["lr"], 2e-4, "lr chính Stage 0")
        _close(by["branch_sup_attn"]["lr"], 6e-4, "Attn Stage 0 = 3×")
        eq(by["branch_sup"]["weight_decay"], 0.8)
        eq(by["branch_sup_attn"]["weight_decay"], 0.8)
        eq({id(p) for p in by["branch_sup_attn"]["params"]}, attn_ids(b))
        want = {id(p) for nm, p in b.named_parameters()
                if nm.split(".", 1)[0] in P.PRETRAINED_MODULES}
        eq({id(p) for g in opt.param_groups for p in g["params"]}, want,
           "Stage 0 tối ưu ĐÚNG PRETRAINED_MODULES")
        raises(TrainUnusable,
               lambda: S.build_optimizer(b, pcfg, {"attn_lr_mult": 0.5}),
               "attn_lr_mult < 1 phải DỪNG")
    check("Stage 0: lr 2e-4, Attn 6e-4, wd 0.8", t_pretrain_groups)

    def t_scratch_groups():
        b = _branch()
        opt, _ = T.build_optimizer(_FakePeft(), b, TCFG, from_pretrain=False,
                                   lrs=LRS)
        by = {g["name"]: g for g in opt.param_groups}
        eq(list(by), ["lora", "branch", "branch_attn"], "nhóm của scratch")
        _close(by["branch"]["lr"], 3e-4)
        _close(by["branch_attn"]["lr"], 9e-4, "scratch: Attn = 3 × main")
        eq({id(p) for p in by["branch_attn"]["params"]}, attn_ids(b))
    check("scratch: nhánh lr chính, Attn × 3", t_scratch_groups)

    def t_ft_groups():
        b = _branch()
        opt, _ = T.build_optimizer(_FakePeft(), b, TCFG, from_pretrain=True,
                                   lrs=LRS)
        by = {g["name"]: g for g in opt.param_groups}
        eq(list(by), ["lora", "branch_pretrained", "branch_pretrained_attn",
                      "branch"], "nhóm của decoder_ft")
        _close(by["lora"]["lr"], 3e-4, "adapter lr chính")
        _close(by["branch_pretrained"]["lr"], 1e-4, "đã pretrain = 2e-4 / 2")
        _close(by["branch_pretrained_attn"]["lr"], 3e-4, "Attn = 1e-4 × 3")
        _close(by["branch"]["lr"], 3e-4, "phần còn TƯƠI lr chính")
        true(by["branch_pretrained_attn"]["lr"] > by["branch_pretrained"]["lr"],
             "Attn phải cao hơn phần đã pretrain")
    check("decoder_ft: pretrain lr / 2, Attn × 3, tươi lr chính", t_ft_groups)

    def t_ft_membership():
        """Không còn lr riêng cho Main Head: cả cls_head ở nhóm đã pretrain."""
        b = _branch()
        opt, _ = T.build_optimizer(_FakePeft(), b, TCFG, from_pretrain=True,
                                   lrs=LRS)
        by = {g["name"]: g for g in opt.param_groups}
        pre = {id(p) for nm, p in b.named_parameters()
               if nm.split(".", 1)[0] in P.PRETRAINED_MODULES}
        eq({id(p) for p in by["branch_pretrained"]["params"]},
           pre - attn_ids(b), "nhóm đã pretrain = Recombinator(−Attn) + cls_head")
        true(ids_of(b.cls_head) <= {id(p) for p in by["branch_pretrained"]["params"]},
             "Main Head + Pooling Stack chạy CÙNG lr phần đã pretrain")
        eq({id(p) for p in by["branch_pretrained_attn"]["params"]}, attn_ids(b))
        eq({id(p) for p in by["branch"]["params"]},
           ids_of(b.route_head) | ids_of(b.dec_proj),
           "nhóm tươi = route_head + dec_proj và CHỈ chúng")
    check("decoder_ft: route_head/dec_proj tươi, cls_head không có lr riêng",
          t_ft_membership)

    def t_ft_lrs_from_cfg():
        from src.config import load_config

        cfg = load_config()
        l = T.ft_lrs(cfg)
        _close(l["main"], float(cfg["train"]["schedule"]["max_lr"]))
        _close(l["pretrain_max_lr"], 2e-4, "pretrain.schedule.max_lr")
        _close(l["pretrained"], 1e-4, "decoder_ft: 2e-4 / 2")
        eq(l["attn_mult"], 3.0)
        eq(float(cfg["pretrain"]["weight_decay"]), 0.8)
        bad = load_config()
        bad["train"]["pretrained_lr_div"] = 0.5
        raises(TrainUnusable, lambda: T.ft_lrs(bad), "div < 1 phải DỪNG")
        bad = load_config()
        bad["recombiner"]["attn_lr_mult"] = 0.9
        raises(TrainUnusable, lambda: T.ft_lrs(bad), "attn < 1 phải DỪNG")
        raises(TrainUnusable,
               lambda: T.build_optimizer(_FakePeft(), _branch(), TCFG,
                                         from_pretrain=True,
                                         lrs=dict(LRS, pretrained=None)),
               "decoder_ft thiếu lr Stage 0 phải DỪNG")
    check("ft_lrs đọc config: 2e-4/2, Attn × 3, tên sai -> DỪNG",
          t_ft_lrs_from_cfg)

    def t_scheduler_keeps_ratio():
        """LambdaLR nhân CÙNG hệ số vào mọi nhóm -> tỉ lệ lr giữ nguyên."""
        from src.backbone import build_scheduler

        b = _branch()
        opt, _ = T.build_optimizer(_FakePeft(), b, TCFG, from_pretrain=True,
                                   lrs=LRS)
        sch = build_scheduler(opt, 100, {"impl": "lambda", "max_lr": 3e-4,
                                         "pct_start": 0.1, "div_factor": 25.0,
                                         "final_div_factor": 100.0})
        for _ in range(30):
            opt.step()
            sch.step()
        by = {g["name"]: g["lr"] for g in opt.param_groups}
        r = by["branch_pretrained"] / by["branch"]
        true(abs(r - 1 / 3) < 1e-9, f"tỉ lệ đã pretrain/tươi phải giữ 1/3, đang {r}")
        r = by["branch_pretrained_attn"] / by["branch_pretrained"]
        true(abs(r - 3.0) < 1e-9, f"tỉ lệ Attn phải giữ 3, đang {r}")
    check("one-cycle giữ nguyên tỉ lệ lr giữa các nhóm", t_scheduler_keeps_ratio)


# ═══ 4b. Đóng băng Main Head của Classification Head sau pretrain ═══════════
def freeze_ft_tests() -> None:
    from src import pretrain_io as P
    from src import train_recomb as T
    from src.backbone import TrainUnusable

    TCFG = {"schedule": {"max_lr": 3e-4}, "weight_decay": 0.0,
            "branch_weight_decay": 0.01, "freeze_after_pretrain": []}

    def ids(mods):
        return {id(p) for m in mods for p in m.parameters()}

    def t_config_values():
        from src.config import load_config

        tc = load_config()["train"]
        eq(float(tc["pretrained_lr_mult"]), 0.1, "pretrained_lr_mult sau pretrain")
        true("pretrained_lr_parts" not in tc,
             "train.pretrained_lr_parts đã bỏ — Main Head không có lr riêng")
        eq(float(tc["pretrained_lr_div"]), 2.0)
        eq(list(tc["freeze_after_pretrain"]), [],
           "decoder_ft không đóng băng gì")
        eq(bool(tc.get("freeze_pretrained", False)), False)
        eq(T.frozen_after_pretrain(dict(tc)), ())
    check("config: không đóng băng, lr phần đã pretrain = Stage 0 / 2",
          t_config_values)

    def t_resolve():
        eq(T.frozen_after_pretrain({}), (), "mặc định = không đóng băng")
        eq(T.frozen_after_pretrain({"freeze_after_pretrain": None}), ())
        eq(T.frozen_after_pretrain({"freeze_after_pretrain": []}), ())
        eq(T.frozen_after_pretrain({"freeze_after_pretrain": "cls_head"}),
           ("cls_head",))
        eq(sorted(T.frozen_after_pretrain({"freeze_pretrained": True})),
           sorted(P.PRETRAINED_MODULES), "freeze_pretrained ghi đè")
        for bad in (["cls_heads"], ["route_head"], ["dec_proj"],
                    ["route_head.main_head"], ["cls_head.fc2"]):
            raises(TrainUnusable,
                   lambda bad=bad: T.frozen_after_pretrain(
                       {"freeze_after_pretrain": bad}),
                   f"{bad} phải DỪNG")
    check("freeze_after_pretrain: mặc định, ghi đè, tên sai -> DỪNG", t_resolve)

    def t_main_head_parts():
        b = _branch()
        th = b.cls_head.head
        eq(ids(th.main_head_modules()), ids([th.norm, th.fc1, th.fc2]),
           "Main Head = RMSNorm + fc1 + fc2")
        true(not (ids(th.main_head_modules()) & ids([th.pool])),
             "Main Head không được gồm Pooling Stack")
        eq(ids([th.pool]) | ids(th.main_head_modules()), ids([b.cls_head]),
           "Pooling Stack + Main Head = toàn bộ cls_head")
    check("cls_head = Pooling Stack ⊔ Main Head", t_main_head_parts)

    def t_optimizer_groups():
        b = _branch(dict(RCFG_BASE, route_enabled=True))
        n = T.apply_freeze(b, T.frozen_after_pretrain(TCFG))
        eq(n, 0, "không đóng băng tham số nào")
        th = b.cls_head.head
        opt, _ = T.build_optimizer(_FakePeft(), b, TCFG, from_pretrain=True,
                                   lrs=LRS)
        by = {g["name"]: g for g in opt.param_groups}
        in_opt = set().union(*({id(p) for p in g["params"]}
                               for g in opt.param_groups))
        true(ids([b]) <= in_opt, "MỌI tham số nhánh phải nằm trong optimizer")
        true(ids(th.main_head_modules())
             <= {id(p) for p in by["branch_pretrained"]["params"]},
             "Main Head ở nhóm đã pretrain (không có lr riêng)")
        eq({id(p) for p in by["branch"]["params"]},
           ids([b.route_head, b.dec_proj]), "nhóm tươi = route_head + dec_proj")
        eq(T.learning_pretrained_parts(b),
           ["recombinator", "cls_head.pooling_stack", "cls_head.main_head"])
    check("decoder_ft: fine-tune toàn bộ, không đóng băng",
          t_optimizer_groups)

    def t_explicit_freeze_still_works():
        """Đối chứng: đóng băng tường minh Main Head -> nó rời optimizer."""
        b = _branch(dict(RCFG_BASE, route_enabled=True))
        tc = dict(TCFG, freeze_after_pretrain=["cls_head.main_head"])
        th = b.cls_head.head
        n = T.apply_freeze(b, T.frozen_after_pretrain(tc))
        eq(n, sum(p.numel() for m in th.main_head_modules() for p in m.parameters()))
        opt, _ = T.build_optimizer(_FakePeft(), b, tc, from_pretrain=True,
                                   lrs=LRS)
        in_opt = set().union(*({id(p) for p in g["params"]}
                               for g in opt.param_groups))
        true(not (ids(th.main_head_modules()) & in_opt),
             "Main Head đóng băng KHÔNG được nằm trong optimizer")
    check("freeze_after_pretrain tường minh vẫn đóng băng đúng",
          t_explicit_freeze_still_works)

    def t_frozen_main_head_behaviour():
        """Main Head đóng băng: không grad, không dropout; Pooling + Recombinator vẫn học."""
        import copy

        import torch

        from src import recombiner as R
        c = copy.deepcopy(CCFG)
        c["dropout"] = 0.5
        b = R.RecombinerBranch(D_VIS, D_MODEL, N_LABELS,
                               copy.deepcopy(RCFG_BASE), c).float()
        b.train()
        T.apply_freeze(b, ("cls_head.main_head",))
        th = b.cls_head.head
        true(all(not m.training for m in th.main_head_modules()),
             "các lớp Main Head phải ở eval mode")
        true(th.pool.training and b.recombinator.training,
             "Pooling Stack và Recombinator vẫn ở train mode")
        vis = torch.randn(2, RCFG_BASE["n_vis"], D_VIS)
        with torch.no_grad():
            rec = b.recombinator(vis)
            l1, l2 = b.cls_head(rec), b.cls_head(rec)
        true(torch.equal(l1, l2),
             "Main Head đóng băng phải không còn dropout (cùng vào -> cùng ra)")
        _t, logits, _r = b(vis)
        y = torch.zeros(2, N_LABELS, 3)
        y[:, :, 1] = 1.0
        b.sup_loss(logits, y, torch.ones(2)).backward()
        true(all(p.grad is None for m in th.main_head_modules()
                 for p in m.parameters()), "Main Head không được nhận gradient")
        for name, mod in (("Pooling Stack", th.pool), ("Recombinator", b.recombinator)):
            gs = [p.grad for p in mod.parameters() if p.grad is not None]
            true(gs and any(float(g.abs().sum()) > 0 for g in gs),
                 f"ASL vẫn phải kéo {name} qua Main Head đóng băng")
    check("Main Head đóng băng: eval, không grad; Pooling/Recombinator vẫn học",
          t_frozen_main_head_behaviour)

    def t_loop_refreezes():
        """`branch.train()` trong vòng lặp phải được theo sau bởi apply_freeze."""
        src = (ROOT / "src" / "train_recomb.py").read_text(encoding="utf-8")
        i = src.index("        branch.train()\n")
        true("apply_freeze(branch, frozen)" in src[i:i + 200],
             "sau branch.train() phải gọi lại apply_freeze")
    check("vòng lặp gọi lại apply_freeze sau branch.train()", t_loop_refreezes)


# ═══ 5. λ-warmup ở decoder_ft ════════════════════════════════════════════════
def lambda_tests() -> None:
    from src import train_recomb as T

    def t_warmup_math():
        # scratch: λ leo từ 0 trong 10% đầu
        eq(T.lambda_at(0, 100, 0.2, 0.1), 0.0)
        true(abs(T.lambda_at(5, 100, 0.2, 0.1) - 0.1) < 1e-12)
        true(abs(T.lambda_at(50, 100, 0.2, 0.1) - 0.2) < 1e-12)
        # decoder_ft: warmup = 0 -> λ đầy NGAY từ bước 0
        eq(T.lambda_at(0, 100, 0.2, 0.0), 0.2)
    check("lambda_at: warmup 0 -> λ đầy ngay bước 0", t_warmup_math)

    def t_config_key():
        import yaml

        from src.config import DEFAULT_CONFIG

        c = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        cc = c["classifier"]
        true("lambda_warmup_pct_pretrained" in cc,
             "config thiếu `classifier.lambda_warmup_pct_pretrained`")
        eq(float(cc["lambda_warmup_pct_pretrained"]), 0.0,
           "mặc định phải là 0.0 — ClsHead sau pretrain không còn ngẫu nhiên")
        true(float(cc["lambda_warmup_pct"]) > 0.0,
             "chế độ scratch vẫn phải có warmup")
    check("config khai đủ hai giá trị λ-warmup", t_config_key)


# ═══ 6. Checkpoint theo epoch + resume ═══════════════════════════════════════
def ckpt_tests() -> None:
    from src import pretrain_io as P
    from src import pretrain_sup as PS

    def t_write_read():
        import torch

        b = _branch()
        opt = torch.optim.AdamW([p for _, p in PS.trainable_params(b)], lr=1e-3)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        g = torch.Generator().manual_seed(7)
        order = torch.randperm(50, generator=g).tolist()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            st = {"global_step": 12, "cursor": 30,
                  "sample_index": order[30], "seen_samples": 30,
                  "gen_state": g.get_state(), "best": None}
            PS.write_ckpt(b, opt, sch, d, 1, st, {"epoch": 1}, keep_last=2)
            last = PS.read_last(d)
            eq(last["epoch"], 1)
            s = torch.load(Path(last["dir"]) / P.TRAINER_STATE,
                           map_location="cpu", weights_only=False)
            eq(s["cursor"], 30)
            eq(s["sample_index"], order[30],
               "chỉ số mẫu tại cursor phải được ghi đúng")
            # Generator khôi phục phải cho ra ĐÚNG hoán vị kế tiếp.
            g2 = torch.Generator()
            g2.set_state(s["gen_state"])
            eq(torch.randperm(50, generator=g2).tolist(),
               torch.randperm(50, generator=g).tolist(),
               "resume phải thấy đúng thứ tự dữ liệu của epoch kế tiếp")
            true(P.pretrain_exists(Path(last["dir"])),
                 "mỗi epoch phải có cả file trọng số lẫn META")
    check("ckpt epoch: cursor + idx mẫu + RNG khôi phục đúng", t_write_read)

    def t_keep_last_spares_best():
        import torch

        b = _branch()
        opt = torch.optim.AdamW([p for _, p in PS.trainable_params(b)], lr=1e-3)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        g = torch.Generator().manual_seed(1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for e in range(1, 6):
                st = {"global_step": e, "cursor": 0, "sample_index": 0,
                      "seen_samples": e * 10, "gen_state": g.get_state(),
                      "best": None}
                dd = PS.write_ckpt(b, opt, sch, d, e, st, {"epoch": e},
                                   keep_last=2)
                if e == 2:
                    P.copy_ckpt(dd, d / "ckpt" / "best")
            eps = sorted(p.name for p in (d / "ckpt").glob("epoch_*"))
            eq(eps, ["epoch_004", "epoch_005"], "keep_last=2 phải dọn epoch cũ")
            true(P.pretrain_exists(d / "ckpt" / "best"),
                 "ckpt/best KHÔNG được bị dọn cùng epoch_002")
            true(not (d / "ckpt" / "best" / P.TRAINER_STATE).exists(),
                 "ckpt/best không mang theo trạng thái optimizer")
    check("keep_last dọn epoch cũ nhưng CHỪA ckpt/best", t_keep_last_spares_best)

    def t_keep_last_spares_current_after_restart():
        """Run mới bắt đầu epoch 1 không được tự dọn state của chính nó."""
        import torch

        b = _branch()
        opt = torch.optim.AdamW([p for _, p in PS.trainable_params(b)], lr=1e-3)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        g = torch.Generator().manual_seed(2)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for e in (4, 5):
                st = {"global_step": e, "cursor": 0, "sample_index": None,
                      "seen_samples": e, "gen_state": g.get_state(), "best": None}
                PS.write_ckpt(b, opt, sch, d, e, st, {"epoch": e}, keep_last=2)
            st = {"global_step": 1, "cursor": 0, "sample_index": None,
                  "seen_samples": 1, "gen_state": g.get_state(), "best": None}
            PS.write_ckpt(b, opt, sch, d, 1, st, {"epoch": 1}, keep_last=2)
            last = PS.read_last(d)
            eq(last["epoch"], 1)
            true((Path(last["dir"]) / P.TRAINER_STATE).exists(),
                 "epoch hiện tại không được bị keep_last xóa sau --no-resume")
    check("keep_last chừa trainer_state của epoch mới sau khi chạy lại",
          t_keep_last_spares_current_after_restart)

    def t_schedule_guard():
        """Ckpt phải mang theo đường cong one-cycle nó nằm trên.

        `LambdaLR.state_dict()` KHÔNG lưu `total_steps`. Không có hai con số này
        thì nối tiếp với `n_epochs` khác sẽ đọc bước cũ trên một đường cong mới:
        lr LEO NGƯỢC LÊN từ đáy đúng lúc nhánh đã hội tụ, và mọi con số trong
        log vẫn hợp lệ. Đây là ca hỏng gần như chắc chắn xảy ra — "chạy thêm vài
        epoch nữa" là việc tự nhiên nhất người ta làm với một lần pretrain.
        """
        import torch

        b = _branch()
        opt = torch.optim.AdamW([p for _, p in PS.trainable_params(b)], lr=1e-3)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
        g = torch.Generator().manual_seed(3)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            st = {"global_step": 12, "cursor": 0, "sample_index": None,
                  "seen_samples": 48, "gen_state": g.get_state(), "best": None,
                  "n_epochs": 2, "steps_total": 12}
            PS.write_ckpt(b, opt, sch, d, 2, st, {"epoch": 2}, keep_last=2)
            s = torch.load(Path(PS.read_last(d)["dir"]) / P.TRAINER_STATE,
                           map_location="cpu", weights_only=False)
            eq(s["steps_total"], 12, "ckpt phải ghi steps_total")
            eq(s["n_epochs"], 2, "ckpt phải ghi n_epochs")
        src = (ROOT / "src" / "pretrain_sup.py").read_text(encoding="utf-8")
        true("allow_schedule_change" in src,
             "phải có cổng chặn nối tiếp trên một lịch khác")
        true("prev_total != steps" in src,
             "phải SO SÁNH steps_total cũ với steps_total mới")
    check("ckpt mang theo lịch one-cycle; đổi n_epochs -> phải có cổng chặn",
          t_schedule_guard)

    def t_resolve_local():
        b = _branch()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            P.save_pretrain(b, d, {})
            eq(P.resolve_source(str(d)), d)
            raises(FileNotFoundError,
                   lambda: P.resolve_source(str(d / "khong-co")),
                   "thư mục không phải ckpt phải DỪNG")
    check("resolve_source: thư mục cục bộ", t_resolve_local)


# ═══ 7. Cấu hình + ranh giới ═════════════════════════════════════════════════
def config_tests() -> None:
    import yaml

    from src.config import DEFAULT_CONFIG

    CFG = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def t_single_config():
        """MỘT file cấu hình: Stage 0 và §2.5 đọc cùng một file.

        Hai file là cách Stage 0 của My Gemma từng chạy lịch lr (div 25/100)
        khác hẳn khối `pretrain` mà điều kiện 5 công bố (div 30/10) — không
        exception, không một dòng log nào gọi tên nó.
        """
        from src import run_pretrain, run_section25

        eq(Path(run_pretrain.DEFAULT_CONFIG).resolve(),
           Path(DEFAULT_CONFIG).resolve(),
           "run_pretrain phải đọc config mặc định của repo")
        eq(Path(run_section25.DEFAULT_CONFIG).resolve(),
           Path(DEFAULT_CONFIG).resolve(),
           "run_section25 phải đọc config mặc định của repo")
        ymls = sorted(p.name for p in (ROOT / "configs").glob("*.y*ml"))
        eq(ymls, [DEFAULT_CONFIG.name],
           "configs/ chỉ được có MỘT file — file thứ hai là mở lại đường để "
           "Stage 0 và §2.5 đọc hai khối `pretrain` khác nhau")
    check("một file cấu hình cho Stage 0 và §2.5", t_single_config)

    def t_block():
        p = CFG.get("pretrain")
        true(p, "config thiếu khối `pretrain:`")
        for k in ("n_train", "n_epochs", "per_device_batch_size", "grad_accum",
                  "schedule", "hub_repo", "hub_subdir", "select_metric",
                  "keep_last_ckpt", "base_method", "subdir"):
            true(k in p, f"pretrain thiếu khoá `{k}`")
        eq(p["select_metric"], "cls_f1_pos",
           "`cls_acc` không dùng để chọn epoch: lớp `neg` áp đảo nên một head "
           "đoán `neg` cho mọi ô vẫn đạt accuracy rất cao")
    check("config có khối pretrain đầy đủ", t_block)

    def t_repo_separate():
        p, o = CFG["pretrain"], CFG["output"]
        true(p["hub_repo"] != o["hub_repo"],
             "repo pretrain phải KHÁC repo kết quả: push_results có "
             "ignore_patterns chặn *.safetensors, nên trọng số đẩy vào đó bị bỏ "
             "qua ÂM THẦM")
    check("repo pretrain tách khỏi repo kết quả", t_repo_separate)

    def t_ft_dirs():
        o = CFG["output"]
        true(o.get("ft_suffix"), "thiếu output.ft_suffix")
        true(str(o["ft_suffix"]).strip(),
             "ft_suffix rỗng = decoder_ft ghi đè lên lần chạy scratch, và "
             "already_trained() sẽ bỏ qua stage train")
    check("decoder_ft có thư mục riêng (ft_suffix)", t_ft_dirs)

    def t_effective_batch_matches():
        p, t = CFG["pretrain"], CFG["train"]
        eq(p["per_device_batch_size"] * p["grad_accum"],
           t["per_device_batch_size"] * t["grad_accum"],
           "eff_bs của hai chế độ nên bằng nhau — không bắt buộc về toán, "
           "nhưng nó làm 'một bước' có cùng nghĩa ở cả hai bảng")
    check("eff_bs của pretrain = eff_bs của main-stage", t_effective_batch_matches)

    def t_mode_key():
        eq(CFG.get("mode"), "decoder_ft",
           "mặc định của §2.5 là decoder_ft — điều kiện 5 bắt My Gemma đi qua "
           "Stage 0 trước; `scratch` chỉ bật qua `--init-from none`")
    check("mode mặc định = decoder_ft", t_mode_key)

    def t_no_decoder_import():
        """`pretrain_sup` KHÔNG được gọi decoder: đó là toàn bộ định nghĩa chế độ."""
        src = (ROOT / "src" / "pretrain_sup.py").read_text(encoding="utf-8")
        for bad in ("build_inputs_embeds", "inject_tokens", "out.loss",
                    "attach_adapter", "get_peft_model"):
            true(bad not in src,
                 f"src/pretrain_sup.py có `{bad}` — pretrain không chạy decoder "
                 f"và không gắn adapter. Có nó ở đây nghĩa là 100%% có giám sát "
                 f"đã thôi đúng.")
        true("no_grad" in src, "lượt vision phải chạy trong no_grad")
    check("pretrain_sup không chạm decoder / adapter", t_no_decoder_import)


def main() -> int:
    print("═══ selftest sup_pretrain + decoder_ft — checkpoint · lr · resume ═══\n")
    for fn in (io_tests, compat_tests, freeze_tests, dec_proj_tests, arch_tests,
               lr_tests, freeze_ft_tests,
               lambda_tests, ckpt_tests, config_tests):
        fn()
    for name in PASS:
        print(f"  ✓ {name}")
    if FAIL:
        print(f"\n✗ {len(FAIL)} phép kiểm HỎNG:\n")
        for name, msg, tb in FAIL:
            print(f"  ✗ {name}\n      {msg}")
            if "-v" in sys.argv:
                print("\n".join("      " + l for l in tb.splitlines()))
        print(f"\n{len(PASS)} qua · {len(FAIL)} hỏng")
        return 1
    print(f"\n✓ TẤT CẢ {len(PASS)} phép kiểm đều qua.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

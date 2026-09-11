#!/usr/bin/env python3
"""Tự kiểm chứng `sup_pretrain` + `decoder_ft` — không GPU, không mạng, không model.

Chạy:  python scripts/selftest_pretrain.py        (thêm -v để xem traceback)

Cái nó bảo vệ là bốn ca hỏng cho ra một lần chạy TRÔNG HOÀN TOÀN BÌNH THƯỜNG:

  1. File checkpoint chứa SAI tập khối. Ghi cả `route_gate`/`gate` vào file
     pretrain thì file trông như một checkpoint đầy đủ của nhánh, và người nạp
     nó vào §2.4 sẽ tưởng Route_Gate đã được pretrain — trong khi đó là số ngẫu
     nhiên. Thiếu khoá theo chiều ngược lại thì một phần của khối "đã pretrain"
     là ngẫu nhiên, còn META vẫn nói nó đã học.

  2. Nhóm learning-rate áp nhầm chỗ. `pretrained_lr_mult` phải áp ĐÚNG các khối
     có trong file pretrain, không áp lên `route_gate` (còn tươi) và không áp
     lên `gate`. Áp nhầm ở §2.4 nghĩa là router học chậm gấp mười, và bảng
     định tuyến sẽ đầy expert chết vì một lý do không liên quan tới MoE.

  3. `aux` không thật sự bị đóng băng ở pretrain. Nếu `vec_2` còn gradient thì
     hai query QFormer cùng học một việc, và "hai vector không biết gì về nhau"
     — quy ước của cả thiết kế — mất nghĩa. ASL vẫn giảm bình thường.

  4. Resume đọc sai vị trí. Ckpt ghi theo epoch phải khôi phục ĐÚNG trạng thái
     Generator + cursor: sai một chỗ thì lần chạy tiếp theo lặp lại (hoặc bỏ
     qua) một phần dữ liệu, và không có gì trong log nói vậy.

Cộng thêm: checkpoint pretrain nạp được sang §2.4 (hình dạng `query` (2, d) và
`route_gate` vẫn còn), `assert_compatible` bắt được lệch n_labels / đổi chỗ
task, λ-warmup bị TẮT ở decoder_ft, và `pretrain_rcfg` ép đúng ba khoá.

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
    "siglip_dim": D_VIS, "swiglu_hidden": 32, "out_dim": None,
    "n_query": 4, "n_vis": 12, "n_heads": 2, "dropout": 0.0,
    "swiglu_dropout": 0.0, "pre_norm": True, "gate_init": 0.0,
    "qformer": {"tasks": ["sup", "aux"], "freeze": ["aux"], "d_model": 12,
                "n_layers": 1, "n_heads": 2, "ffn_mult": 2, "dropout": 0.0},
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
        true("gate" in drop, "`gate` phải nằm ngoài file pretrain")
        true(any(k.startswith("route_gate.") for k in drop),
             "`route_gate` phải nằm ngoài file pretrain")
        true(not any(k.startswith("route_gate.") for k in keep))
        eq(len(keep) + len(drop), len(b.state_dict()), "không được mất khoá")
    check("split_state: đúng khối vào file, gate/route_gate ở ngoài", t_split)

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
            true(any(k == "gate" for k in meta["keys_not_saved"]))

            b2 = _branch()
            rep = P.load_into_branch(b2, Path(d), verbose=False)
            eq(sorted(rep["fresh_modules"]), sorted(P.UNTRAINED_MODULES))
            for k, v in b.state_dict().items():
                if k.split(".", 1)[0] in P.PRETRAINED_MODULES:
                    true(torch.allclose(v, b2.state_dict()[k]),
                         f"khoá {k} không khớp sau khi nạp lại")
            # Và khối KHÔNG pretrain phải giữ giá trị khởi tạo của b2, không
            # phải của b: nếu nó khớp b thì file đã lén mang chúng theo.
            true(not torch.allclose(b.gate, b2.gate) or float(b.gate.abs().sum()) == 0,
                 "`gate` của b2 không được lấy từ file")
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
            sd["cls_head.net.99.weight"] = torch.zeros(2, 2)
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

    def t_task_order():
        import copy

        b = _branch()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            P.save_pretrain(b, d, {})
            r = copy.deepcopy(RCFG_BASE)
            r["qformer"]["tasks"] = ["aux", "sup"]      # ĐỔI CHỖ
            r["qformer"]["freeze"] = ["sup"]
            swapped = _branch(r)
            raises(P.PretrainIncompatible,
                   lambda: P.load_into_branch(swapped, d, verbose=False),
                   "đổi thứ tự task phải DỪNG — shape vẫn khớp nên không có gì "
                   "khác bắt được")
    check("đổi chỗ sup/aux -> DỪNG (shape vẫn khớp!)", t_task_order)


# ═══ 2. Hình dạng nạp thẳng sang §2.4 ════════════════════════════════════════
def compat_tests() -> None:
    import copy

    from src import pretrain_io as P
    from src import train_recomb as T

    def t_query_rows():
        b = _branch()
        eq(tuple(b.qformer.query.shape), (2, RCFG_BASE["qformer"]["d_model"]),
           "QFormer của pretrain phải giữ HAI hàng query")
        eq(list(b.qformer.frozen), ["aux"], "chỉ `aux` bị đóng băng")
    check("pretrain giữ query (2, d) — nạp thẳng sang §2.4 được", t_query_rows)

    def t_into_moe():
        """Nhánh §2.4 (freeze=[], route_enabled=True) nạp được file pretrain."""
        from src.config import Cfg

        b = _branch()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            P.save_pretrain(b, d, {})
            # ĐÚNG hàm mà §2.4 dùng để dựng rcfg, không phải một bản chép tay.
            cfg = Cfg({"recombiner": copy.deepcopy(RCFG_BASE)})
            spec = {"n_experts": 6, "noise_init": -3.0}
            r24 = T.effective_rcfg(cfg, spec)
            eq(r24["qformer"]["freeze"], [], "§2.4 phải mở băng `aux`")
            eq(r24["route_enabled"], True)
            b24 = _branch(r24)
            rep = P.load_into_branch(b24, d, verbose=False)
            eq(sorted(rep["fresh_modules"]), sorted(P.UNTRAINED_MODULES))
            eq(b24.route_gate.n_experts, 6,
               "route_gate của §2.4 dựng theo n_experts của candidate, KHÔNG "
               "lấy từ file pretrain")
            eq(list(b24.qformer.frozen), [],
               "sau khi nạp, `aux` phải học được ở §2.4")
    check("checkpoint pretrain nạp thẳng vào nhánh §2.4 (E khác)", t_into_moe)


# ═══ 3. Đóng băng `aux` ở pretrain ═══════════════════════════════════════════
def freeze_tests() -> None:
    from src import pretrain_sup as PS
    from src.config import Cfg

    def t_rcfg_forces():
        cfg = Cfg({"recombiner": {"qformer": {"tasks": ["sup", "aux"],
                                              "freeze": []},
                                  "route_enabled": True, "n_experts": 9}})
        r = PS.pretrain_rcfg(cfg)
        eq(r["qformer"]["freeze"], ["aux"],
           "pretrain phải ÉP đóng băng mọi task khác `sup`, kể cả khi config "
           "nói freeze: []")
        eq(r["route_enabled"], False, "pretrain không định tuyến gì")
    check("pretrain_rcfg ép freeze=[aux] + route_enabled=false", t_rcfg_forces)

    def t_no_sup():
        cfg = Cfg({"recombiner": {"qformer": {"tasks": ["aux"]}}})
        from src.backbone import TrainUnusable
        raises(TrainUnusable, lambda: PS.pretrain_rcfg(cfg),
               "không có task `sup` thì không có gì để pretrain")
    check("thiếu task `sup` -> DỪNG", t_no_sup)

    def t_aux_no_grad():
        """Gradient của ASL KHÔNG được chảy tới hàng `aux` của query."""
        import torch

        b = _branch().float()
        vis = torch.randn(2, RCFG_BASE["n_vis"], D_VIS)
        _tok, logits, route = b(vis)
        true(route is None, "route_enabled=false phải trả None")
        y = torch.zeros(2, N_LABELS, 3)
        y[:, :, 0] = 1.0
        loss = b.sup_loss(logits, y, torch.ones(2))
        loss.backward()
        g = b.qformer.query.grad
        true(g is not None, "query phải có gradient ở hàng `sup`")
        sup_i = list(b.qformer.tasks).index("sup")
        aux_i = list(b.qformer.tasks).index("aux")
        true(float(g[sup_i].abs().sum()) > 0, "hàng `sup` phải nhận gradient")
        eq(float(g[aux_i].abs().sum()), 0.0,
           "hàng `aux` phải nhận ĐÚNG BẰNG 0 — nếu không, hai vector cùng học "
           "một việc và quy ước 'không biết gì về nhau' mất nghĩa")
    check("ASL không chảy tới query `aux`", t_aux_no_grad)

    def t_gate_route_no_grad():
        """`gate` và `route_gate` không nhận gradient khi decoder không chạy."""
        import torch

        b = _branch().float()
        vis = torch.randn(2, RCFG_BASE["n_vis"], D_VIS)
        _tok, logits, _r = b(vis)
        y = torch.zeros(2, N_LABELS, 3)
        y[:, :, 1] = 1.0
        b.sup_loss(logits, y, torch.ones(2)).backward()
        true(b.gate.grad is None or float(b.gate.grad.abs().sum()) == 0.0,
             "`gate` không được nhận gradient — đó là lý do nó ở ngoài file")
        for n, p in b.route_gate.named_parameters():
            true(p.grad is None or float(p.grad.abs().sum()) == 0.0,
                 f"route_gate.{n} không được nhận gradient ở pretrain")
    check("gate / route_gate không nhận gradient -> đúng là UNTRAINED", t_gate_route_no_grad)

    def t_trainable_list():
        from src import pretrain_io as P

        b = _branch()
        names = {n.split(".", 1)[0] for n, _ in PS.trainable_params(b)}
        eq(sorted(names), sorted(P.PRETRAINED_MODULES),
           "optimizer của pretrain phải nhận ĐÚNG PRETRAINED_MODULES")
    check("optimizer pretrain = đúng PRETRAINED_MODULES", t_trainable_list)


# ═══ 4. Nhóm learning-rate của decoder_ft ════════════════════════════════════
class _FakePeft:
    """Đủ để `build_optimizer` chạy: một tham số requires_grad."""

    def __init__(self):
        import torch

        self._p = [torch.nn.Parameter(torch.zeros(3))]

    def parameters(self):
        return iter(self._p)


def lr_tests() -> None:
    from src import pretrain_io as P
    from src import train_recomb as T

    TCFG = {"schedule": {"max_lr": 3e-4}, "weight_decay": 0.0,
            "branch_weight_decay": 0.01, "gate_lr_mult": 20.0,
            "pretrained_lr_mult": 0.1}

    def t_scratch_unchanged():
        """Chế độ scratch phải cho ra ĐÚNG ba nhóm như trước — không hồi quy."""
        b = _branch()
        opt, n = T.build_optimizer(_FakePeft(), b, TCFG, from_pretrain=False)
        names = [g["name"] for g in opt.param_groups]
        eq(names, ["lora", "branch", "gate"], "nhóm của chế độ scratch")
        for g in opt.param_groups:
            if g["name"] == "branch":
                eq(g["lr"], 3e-4, "scratch: nhánh chạy lr đầy đủ")
    check("scratch: vẫn đúng 3 nhóm, nhánh lr đầy đủ (không hồi quy)",
          t_scratch_unchanged)

    def t_ft_groups():
        b = _branch()
        opt, n = T.build_optimizer(_FakePeft(), b, TCFG, from_pretrain=True)
        names = [g["name"] for g in opt.param_groups]
        eq(names, ["lora", "branch_pretrained", "branch", "gate"],
           "decoder_ft phải tách thân nhánh làm đôi")
        by = {g["name"]: g for g in opt.param_groups}
        eq(by["branch_pretrained"]["lr"], 3e-4 * 0.1, "lr của phần đã pretrain")
        eq(by["branch"]["lr"], 3e-4, "phần còn TƯƠI chạy lr đầy đủ")
        eq(by["gate"]["lr"], 3e-4 * 20.0, "gate giữ nguyên hệ số riêng")
    check("decoder_ft: 4 nhóm, đúng 3 mức lr", t_ft_groups)

    def t_ft_membership():
        """Nhóm đã-pretrain phải chứa ĐÚNG tham số của PRETRAINED_MODULES."""
        b = _branch()
        opt, _ = T.build_optimizer(_FakePeft(), b, TCFG, from_pretrain=True)
        by = {g["name"]: g for g in opt.param_groups}
        want_pre, want_fresh, want_gate = set(), set(), set()
        for nm, p in b.named_parameters():
            if nm == "gate":
                want_gate.add(id(p))
            elif nm.split(".", 1)[0] in P.PRETRAINED_MODULES:
                want_pre.add(id(p))
            else:
                want_fresh.add(id(p))
        eq({id(p) for p in by["branch_pretrained"]["params"]}, want_pre,
           "nhóm đã-pretrain")
        eq({id(p) for p in by["branch"]["params"]}, want_fresh,
           "nhóm tươi phải là route_gate và CHỈ route_gate")
        eq({id(p) for p in by["gate"]["params"]}, want_gate)
        true(want_fresh, "phải có ít nhất route_gate ở nhóm tươi — nếu rỗng thì "
                         "`route_gate` đã lọt vào nhóm đã-pretrain")
    check("decoder_ft: route_gate ở nhóm TƯƠI, không bị hạ lr", t_ft_membership)

    def t_bad_mult():
        from src.backbone import TrainUnusable

        b = _branch()
        for bad in (0.0, 1.5, -0.1):
            raises(TrainUnusable,
                   lambda bad=bad: T.build_optimizer(
                       _FakePeft(), b, dict(TCFG, pretrained_lr_mult=bad),
                       from_pretrain=True),
                   f"pretrained_lr_mult={bad} phải DỪNG")
    check("pretrained_lr_mult ngoài (0,1] -> DỪNG", t_bad_mult)

    def t_scheduler_keeps_ratio():
        """LambdaLR nhân CÙNG hệ số vào mọi nhóm -> tỉ lệ lr giữ nguyên."""
        from src.backbone import build_scheduler

        b = _branch()
        opt, _ = T.build_optimizer(_FakePeft(), b, TCFG, from_pretrain=True)
        sch = build_scheduler(opt, 100, {"impl": "lambda", "max_lr": 3e-4,
                                         "pct_start": 0.1, "div_factor": 25.0,
                                         "final_div_factor": 100.0})
        for _ in range(30):
            opt.step()
            sch.step()
        by = {g["name"]: g["lr"] for g in opt.param_groups}
        r = by["branch_pretrained"] / by["branch"]
        true(abs(r - 0.1) < 1e-9,
             f"tỉ lệ lr phải giữ 0.1 suốt lịch, đang {r}")
    check("one-cycle giữ nguyên tỉ lệ lr giữa các nhóm", t_scheduler_keeps_ratio)


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

        c = yaml.safe_load((ROOT / "configs" / "recombiner.yaml").read_text(
            encoding="utf-8"))
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

    CFG = yaml.safe_load((ROOT / "configs" / "recombiner.yaml").read_text(
        encoding="utf-8"))

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
    check("eff_bs của pretrain = eff_bs của §2.3", t_effective_batch_matches)

    def t_mode_key():
        eq(CFG.get("mode"), "scratch",
           "mặc định phải là scratch — mọi bảng §2.3/§2.4 đã có đều sinh ra từ đó")
    check("mode mặc định = scratch", t_mode_key)

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

    def t_manifest_separate():
        src = (ROOT / "src" / "run_pretrain.py").read_text(encoding="utf-8")
        true('TRAIN_MANIFEST = "pretrainset"' in src,
             "tập pretrain phải nằm ở manifest RIÊNG — dùng chung trainset.jsonl "
             "nghĩa là §2.3 lặng lẽ đổi tập huấn luyện")
    check("tập pretrain có manifest riêng", t_manifest_separate)


def main() -> int:
    print("═══ selftest sup_pretrain + decoder_ft — checkpoint · lr · resume ═══\n")
    for fn in (io_tests, compat_tests, freeze_tests, lr_tests, lambda_tests,
               ckpt_tests, config_tests):
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

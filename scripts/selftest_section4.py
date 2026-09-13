#!/usr/bin/env python3
"""Tự kiểm chứng §4 — không GPU, không mạng, không model, không tải ảnh.

Chạy:  python scripts/selftest_section4.py        (thêm -v để xem traceback)

§4 so QLoRA r4 THUẦN với Recombinator(pretrain) + K=6·E=10 + QFormer + Cls, trên
CÙNG 6400 mẫu / 3000 ảnh của §2.3, bằng kiểm định ghép cặp k = 2.

Cái nó bảo vệ là NĂM ca hỏng cho ra một bảng TRÔNG HOÀN TOÀN BÌNH THƯỜNG:

  1. Điều kiện trôi. §4 khai một `data.n_train` của riêng nó — bảng vẫn in ra,
     vẫn có p-value, và nó lặng lẽ thôi so được với §2.3/§2.4. Mọi con số vẫn
     hợp lệ khi đọc riêng lẻ.

  2. Cột B mất pretrain. `pretrain: false` lọt vào, hoặc checkpoint không tồn
     tại và code im lặng chạy from-scratch. Cột vẫn mang nhãn "đã pretrain",
     đường loss vẫn giảm bình thường, và không ai phát hiện được về sau.

  3. Checkpoint pretrain đến từ một TẬP KHÁC. `runs/sup_pretrain/` (toàn bộ
     split, ~42k ảnh) nạp vào §4 khớp mọi hình dạng. Bảng khi đó nói "cùng 6400
     mẫu" trong khi candidate B đã thấy gấp bảy lần.

  4. Cổng không đi nổi. `gate_lr_mult` quá nhỏ so với `|gate_init_logit|` thì
     cổng đứng yên vì AdamW chứ không vì dữ liệu, candidate B trông y hệt
     candidate A, và BLEU/ROUGE không phân biệt được hai nguyên nhân. Đây là ca
     tốn nhất: vài tiếng GPU để kết luận về một phép nhân.

  5. `moe.enabled` rò giữa hai candidate. Nó là khoá CẤP CFG, còn ở §4 hai
     candidate khác nhau đúng ở chỗ ấy — một lần quên khôi phục là candidate A
     đi qua `moe_spec` với `top_k` của B.

Cộng thêm: ranh giới file theo CẢ HAI chiều (§4 không ghi vào thư mục của §2.3/
§2.4/§3, và ba mục kia không biết tới §4), k = 2 không được chạm Friedman, và
`n_perm` đủ phân giải cho Holm.
"""
from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

import yaml

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
        raise AssertionError(f"{msg} — mong {exc.__name__}, nhận {e!r}") from None
    raise AssertionError(msg or f"mong {exc.__name__}, không có exception nào")


S4_YAML = ROOT / "configs" / "section4.yaml"
BASE_YAML = ROOT / "configs" / "recombiner.yaml"


def _raw(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


# ══ 1. Điều kiện đi vay ═════════════════════════════════════════════════════
def config_boundary_tests() -> None:
    from src import run_section4 as s4

    def t_s4_clean():
        bad = s4.violations(_raw(S4_YAML))
        true(not bad,
             "configs/section4.yaml chạm vào điều kiện đi vay:\n      "
             + "\n      ".join(bad))
    check("§4: config KHÔNG khai lại điều kiện của §2.3/§2.4", t_s4_clean)

    def t_violation_detected():
        """Chiều ngược lại: hàng rào phải THẬT SỰ bắt được, không chỉ im lặng."""
        for key, sub in (("data", {"n_train": 999}),
                         ("stats", {"dz_floor": 0.0}),
                         ("metrics", {"primary": "bleu_1"}),
                         ("recombiner", {"n_query": 8}),
                         ("lora", {"preset": "qlora_r32"}),
                         ("train", {"per_device_batch_size": 8}),
                         ("seed", 7)):
            bad = s4.violations({key: sub})
            true(bad, f"khai `{key}` trong section4.yaml phải bị BẮT — nếu không "
                      f"thì §4 lặng lẽ chạy trên điều kiện khác §2.3")
    check("§4: hàng rào bắt được khoá đã khoá (chiều ngược lại)",
          t_violation_detected)

    def t_exempt_allowed():
        for key, sub in (("train", {"pretrained_lr_mult": 0.05}),
                         ("train", {"freeze_pretrained": True}),
                         ("classifier", {"lambda_warmup_pct_pretrained": 0.0})):
            eq(s4.violations({key: sub}), [],
               f"{key}.{list(sub)[0]} là khoá của chế độ 2 (§3), nó CHỈ có nghĩa "
               f"khi nhánh đã pretrain — tức là chỉ ở candidate B — nên nó không "
               f"phải điều kiện thí nghiệm và phải được phép khai:")
    check("§4: ba khoá của chế độ 2 vẫn khai được", t_exempt_allowed)

    def t_pretrain_n_train_inert():
        true(s4.violations({"pretrain": {"n_train": 6400}}),
             "`pretrain.n_train` phải bị BẮT: §4 không đọc nó (chế độ 1 nhận "
             "thẳng train_df), nên khai nó là dựng một khoá trông như điều "
             "khiển được tập pretrain mà không điều khiển gì")
        r = _raw(S4_YAML)
        true("n_train" not in (r.get("pretrain") or {}),
             "section4.yaml còn khai `pretrain.n_train`")
    check("§4: `pretrain.n_train` là khoá CHẾT, và config không khai nó",
          t_pretrain_n_train_inert)

    def t_borrowed_values():
        """Điều kiện thật sự đi vay đúng con số, không phải chỉ 'không khai'."""
        base = _raw(BASE_YAML)
        eq(int(base["data"]["n_train"]), 6400, "§2.3 phải là 6400 mẫu train:")
        eq(int(base["data"]["n_test"]), 3000, "§2.3 phải là 3000 ảnh test:")
        true(int(base["data"].get("n_val", 0)) > 0,
             "cần val để chế độ 1 chọn epoch qua select_metric")
        true(bool(base["data"].get("assert_no_leak", True)),
             "hàng rào rò rỉ phải BẬT — rò rỉ không làm bảng §4 nổ, nó làm bảng "
             "§4 đẹp lên, và đó là ca duy nhất người ta không đi kiểm tra")
    check("§4: tập đi vay đúng 6400 / 3000 / val > 0 / hàng rào bật",
          t_borrowed_values)


# ══ 2. Candidate ════════════════════════════════════════════════════════════
def arm_tests() -> None:
    from src.config import Cfg
    from src import run_section4 as s4

    A = {"id": "a", "recomb": False, "pretrain": False}
    B = {"id": "b", "recomb": True, "pretrain": True, "top_k": 6, "n_experts": 10}

    def cfg_of(arms):
        return Cfg({"arms_s4": arms, "moe": {"enabled": False},
                    "lora": {"preset": "qlora_r4"}})

    def t_declared():
        eq([x["id"] for x in s4.s4_arms(cfg_of([A, B]))], ["a", "b"],
           "hai candidate khai đúng phải qua:")
    check("§4: hai candidate hợp lệ", t_declared)

    def t_missing_flags():
        for drop in ("recomb", "pretrain"):
            bad = {k: v for k, v in B.items() if k != drop}
            raises(ValueError, lambda: s4.s4_arms(cfg_of([A, bad])),
                   f"thiếu cờ `{drop}` phải DỪNG — mặc định ngầm ở đây là một "
                   f"cột mang nhãn của một cấu hình chưa từng chạy")
    check("§4: thiếu `recomb` hoặc `pretrain` -> DỪNG", t_missing_flags)

    def t_k_e_pair():
        half = {k: v for k, v in B.items() if k != "n_experts"}
        raises(ValueError, lambda: s4.s4_arms(cfg_of([A, half])),
               "`top_k` không có `n_experts` phải DỪNG: một mình top_k nghĩa là "
               "n_experts rơi vào mặc định và hàng kết quả mang nhãn sai")
    check("§4: top_k/n_experts đi theo CẶP", t_k_e_pair)

    def t_pretrain_needs_branch():
        raises(ValueError,
               lambda: s4.s4_arms(cfg_of([{"id": "x", "recomb": False,
                                           "pretrain": True}])),
               "pretrain = true mà recomb = false phải DỪNG: không nhánh thì "
               "không có gì để pretrain")
        raises(ValueError,
               lambda: s4.s4_arms(cfg_of([{"id": "x", "recomb": False,
                                           "pretrain": False, "top_k": 2,
                                           "n_experts": 4}])),
               "Experts mà không nhánh phải DỪNG: không có vec_2 để Route_Gate đọc")
    check("§4: pretrain/Experts đòi có nhánh", t_pretrain_needs_branch)

    def t_arms_s4_not_arms():
        r = _raw(S4_YAML)
        true("arms_s4" in r,
             "§4 phải khai candidate ở `arms_s4`")
        true("arms" not in r,
             "section4.yaml không được khai `arms`: đó là hai candidate của §2.3 "
             "và nó nằm trong recombiner.yaml. Trùng tên nghĩa là một lần nạp "
             "sai thứ tự sẽ chạy §4 với candidate của §2.3 — cùng số cột, cùng "
             "kiểu kiểm định, bảng vẫn in ra bình thường.")
    check("§4: candidate ở `arms_s4`, KHÔNG trùng tên `arms` của §2.3",
          t_arms_s4_not_arms)

    def t_order_b_first():
        arms = s4.s4_arms(cfg_of([A, B]))
        eq(s4.train_order(arms)[0]["id"], "b",
           "candidate DỄ HỎNG NHẤT phải chạy trước. B dễ hỏng theo ba đường cùng "
           "lúc: có nhánh, E lớn nhất repo, và đứng trên một checkpoint. Chạy nó "
           "cuối là trả xong tiền GPU của A rồi mới biết bảng thiếu cột — và "
           "phép so sánh ghép cặp không định nghĩa được với một cột. Thứ tự:")
    check("§4: candidate B chạy TRƯỚC (quy ước vận hành #1)", t_order_b_first)

    def t_real_config_arms():
        """Hai candidate THẬT trong file phải đúng cấu hình đề bài."""
        arms = s4.s4_arms(Cfg(_raw(S4_YAML)))
        eq(len(arms), 2, "§4 có ĐÚNG hai candidate (k = 2):")
        a = next(x for x in arms if not x["recomb"])
        b = next(x for x in arms if x["recomb"])
        true(not a["pretrain"] and "top_k" not in a,
             "candidate A phải là QLoRA r4 THUẦN — không nhánh, không pretrain, "
             "không Experts. Đó là thứ làm nó ĐỒNG NHẤT với candidate A của §2.3")
        true(b["pretrain"], "candidate B phải khởi tạo nhánh từ pretrain")
        eq((b["top_k"], b["n_experts"]), (6, 10),
           "candidate B phải là K=6 · E=10 (hàng cuối của bảng ablation):")
    check("§4: A = QLoRA r4 thuần · B = pretrain + K=6·E=10", t_real_config_arms)


# ══ 3. moe.enabled không rò giữa hai candidate ══════════════════════════════
def isolation_tests() -> None:
    from src.config import Cfg
    from src import run_section4 as s4

    base = Cfg({"moe": {"enabled": False, "expert": {}, "router": {}, "aux": {}},
                "train": {"alpha_mode": "proportional", "alpha_ratio": 2.0},
                "lora": {"preset": "qlora_r4",
                         "presets": {"qlora_r4": {"method": "qlora", "rank": 4}}}})
    A = {"id": "a", "recomb": False, "pretrain": False}
    B = {"id": "b", "recomb": True, "pretrain": True, "top_k": 6, "n_experts": 10}

    def t_per_arm_moe():
        ca, cb = s4.arm_cfg(base, A), s4.arm_cfg(base, B)
        eq(ca["moe"]["enabled"], False, "candidate A phải TẮT MoE:")
        eq(cb["moe"]["enabled"], True, "candidate B phải BẬT MoE:")
        eq(base["moe"]["enabled"], False,
           "cfg DÙNG CHUNG không được bị sửa — `arm_cfg` phải trả BẢN SAO SÂU. "
           "Một lần quên khôi phục là candidate A đi qua moe_spec với top_k của B:")
    check("§4: `moe.enabled` tách theo từng candidate, cfg chung không đổi",
          t_per_arm_moe)

    def t_deep_copy():
        cb = s4.arm_cfg(base, B)
        cb["moe"]["expert"]["rank"] = 999
        true((base["moe"].get("expert") or {}).get("rank") != 999,
             "`arm_cfg` phải sao SÂU: sao nông thì sửa một khối lồng ở candidate "
             "này đổi luôn candidate kia, và không có gì báo")
    check("§4: `arm_cfg` sao SÂU, không sao nông", t_deep_copy)

    def t_moe_spec_b():
        from src.train_recomb import moe_spec, resolve_preset

        cb = s4.arm_cfg(base, B)
        preset = resolve_preset(cb)
        spec = moe_spec(cb, B, preset)
        eq(spec["top_k"], 6, "spec của B phải mang top_k = 6:")
        eq(spec["n_experts"], 10, "spec của B phải mang n_experts = 10:")
        eq(spec["rank"], preset["rank"],
           "mỗi expert phải có ĐÚNG sức chứa của adapter §2.3 (rank từ "
           "lora.preset). Ghi số riêng vào moe.expert.rank là đưa một biến thứ "
           "ba vào bảng:")
        eq(moe_spec(s4.arm_cfg(base, A), A, preset), None,
           "candidate A phải KHÔNG có spec MoE:")
    check("§4: moe_spec của B = (k=6, E=10, rank của qlora_r4)", t_moe_spec_b)

    def t_k_lt_e():
        b = _raw(S4_YAML)["arms_s4"]
        for a in b:
            if "top_k" in a:
                true(1 <= int(a["top_k"]) < int(a["n_experts"]),
                     f"{a['id']}: cần 1 <= top_k < n_experts. top_k = n_experts "
                     f"là một mô hình DÀY ĐẶC mang tên MoE — mọi expert luôn "
                     f"được chọn nên Route_Gate không định tuyến gì")
    check("§4: 1 <= top_k < n_experts", t_k_lt_e)


# ══ 4. Cổng phải đi nổi ═════════════════════════════════════════════════════
def gate_tests() -> None:
    """Ca hỏng tốn nhất của cả repo: vài tiếng GPU để kết luận về AdamW.

    AdamW CHUẨN HOÁ bước cập nhật — mỗi bước nó dịch một tham số đi khoảng `lr`,
    bất kể gradient lớn hay nhỏ. Nên quãng đường cổng đi được trong cả lịch là
    một phép nhân đọc thẳng từ config, và nó đo bằng đơn vị LOGIT:

        steps × max_lr × gate_lr_mult × ~0.6      (0.6 = trung bình one-cycle)

    Nếu quãng đường < |gate_init_logit| thì cổng đứng yên vì CƠ HỌC chứ không vì
    dữ liệu, candidate B trông y hệt candidate A, và bảng BLEU/ROUGE không phân
    biệt được hai nguyên nhân ấy.
    """
    base = _raw(BASE_YAML)

    def travel(cfg: dict) -> tuple[float, float, int]:
        d, t = cfg["data"], cfg["train"]
        steps = (int(d["n_train"]) // (int(t["per_device_batch_size"])
                                       * int(t["grad_accum"]))) \
            * int(t.get("n_epochs", 1))
        dist = steps * float(t["schedule"]["max_lr"]) \
            * float(t["gate_lr_mult"]) * 0.6
        need = abs(float(cfg["recombiner"]["gate_init_logit"]))
        return dist, need, steps

    def t_gate_can_open():
        dist, need, steps = travel(base)
        true(dist >= need,
             f"cổng KHÔNG đi nổi: {steps} bước × max_lr "
             f"{base['train']['schedule']['max_lr']} × gate_lr_mult "
             f"{base['train']['gate_lr_mult']} × 0.6 = {dist:.2f} logit, trong "
             f"khi cần >= |gate_init_logit| = {need:.2f}.\n"
             f"      Cổng sẽ đứng yên vì AdamW chứ không vì dữ liệu — candidate "
             f"B trông y hệt candidate A,\n      và BLEU/ROUGE không phân biệt "
             f"được hai nguyên nhân. Nâng train.gate_lr_mult lên "
             f">= {need / max(dist / float(base['train']['gate_lr_mult']), 1e-9):.0f}.")
    check("§4: cổng có ĐỦ quãng đường để mở (phép nhân, không phải hy vọng)",
          t_gate_can_open)

    def t_invariant_bites():
        """Chiều ngược lại: bất đẳng thức phải THẬT SỰ bắt được config xấu."""
        import copy as _c

        bad = _c.deepcopy(base)
        bad["train"]["gate_lr_mult"] = 20.0        # giá trị đời trước
        dist, need, _ = travel(bad)
        true(dist < need,
             "với gate_lr_mult = 20 thì quãng đường phải KHÔNG đủ — nếu phép "
             "kiểm này qua thì bất đẳng thức đang không bắt được gì")
    check("§4: bất đẳng thức cổng bắt được gate_lr_mult = 20 (đời trước)",
          t_invariant_bites)

    def t_gate_is_per_token():
        import torch

        from src.recombiner import RecombinerBranch

        r = dict(base["recombiner"])
        r.update({"swiglu_hidden": 32, "n_heads": 4, "n_vis": 4096,
                  "qformer": {"tasks": ["sup", "aux"], "freeze": ["aux"],
                              "d_model": 12, "n_heads": 4, "n_layers": 1,
                              "ffn_mult": 2}})
        b = RecombinerBranch(16, 24, 5, r, {"n_classes": 3, "hidden": 16,
                                            "asl": {"gamma_neg": 2.0,
                                                    "gamma_pos": 0.0,
                                                    "clip": 0.05}})
        n_q = int(base["recombiner"]["n_query"])
        eq(tuple(b.gate.shape), (n_q,),
           f"cổng phải MỞ THEO CHIỀU TOKEN, hình dạng (n_query,) = ({n_q},). "
           f"Hình dạng (d_model,) là cổng theo KÊNH — nó mở 'một phần biên độ "
           f"của mọi token' thay vì 'một số token trọn vẹn', và "
           f"`gate_open_tokens` khi đó là một con số vô nghĩa. Nhận:")
        g = b.gate_value().detach()
        true(float(g.min()) > 0.0 and float(g.max()) < 1.0,
             "cổng phải là sigmoid, chặn ngặt trong (0, 1)")
        with torch.no_grad():
            b.gate.data = torch.tensor(
                [20.0 if i < 3 else -20.0 for i in range(n_q)])
        eq(b.gate_stats()["gate_open_tokens"], 3,
           "gate_open_tokens phải ĐẾM token có cổng >= 0.5:")
    check("§4: cổng là sigmoid THEO CHIỀU TOKEN, (n_query,)", t_gate_is_per_token)


# ══ 5. Kiểm định: k = 2, và nó phải ĐỌC được ════════════════════════════════
def stats_tests() -> None:
    base = _raw(BASE_YAML)
    s = base["stats"]
    m = base["metrics"]

    def t_no_friedman_in_s4():
        src = (ROOT / "src" / "run_section4.py").read_text(encoding="utf-8")
        true("stats_friedman" not in src,
             "run_section4.py nhắc tới `stats_friedman`. k = 2 thì 'kiểm định "
             "tổng thể' và 'kiểm định cặp' là CÙNG một câu hỏi, và chèn Friedman "
             "vào giữa chỉ thêm một tầng Holm mà không thêm thông tin. Nếu đường "
             "§4 chạm vào Friedman thì 'k = 2 không dùng Friedman' thành một câu "
             "trong README chứ không phải một sự thật của code.")
        true("stats_ab" in src, "run_section4.py phải dùng `stats_ab`")
    check("§4: k = 2 -> KHÔNG chạm stats_friedman", t_no_friedman_in_s4)

    def t_resolution():
        n_metrics = len(m["bleu_orders"]) + len(m["rouge_types"]) \
            + len(((m.get("bertscore") or {}).get("keep")) or [])
        floor = 1.0 / (int(s["n_perm"]) + 1)
        thr = float(s["alpha"]) / max(n_metrics, 1)
        true(floor <= thr,
             f"n_perm = {s['n_perm']} cho p nhỏ nhất = {floor:.2e}, trong khi "
             f"Holm trên {n_metrics} metric đòi < {thr:.2e}. Không kiểm định nào "
             f"sống sót được, và bảng sẽ toàn dấu ✗ — trông như một kết luận về "
             f"khoa học trong khi nó là một kết luận về n_perm.")
    check("§4: n_perm đủ phân giải cho Holm", t_resolution)

    def t_dz_floor_binds():
        """Ở n = 3000, `dz_floor` là ràng buộc DUY NHẤT còn hiệu lực.

        t = dz·sqrt(n). Với n = 3000 thì sqrt(n) ~ 54.8, nên một hiệu số nhỏ tới
        mức dz ~ 0.05 đã có |t| ~ 2.7 và sống sót Holm. `dz_floor` = 0.10 mới là
        thứ quyết định cột `verdict`, và bảng phải được đọc theo đúng thứ tự đó.
        """
        import math

        n = int(base["data"]["n_test"])
        n_metrics = len(m["bleu_orders"]) + len(m["rouge_types"]) \
            + len(((m.get("bertscore") or {}).get("keep")) or [])
        dz_sig = 2.5 / math.sqrt(n)          # ~ngưỡng Holm khắt nhất, xấp xỉ z
        dz_floor = float(s["dz_floor"])
        true(dz_floor > dz_sig,
             f"dz_floor = {dz_floor} KHÔNG lớn hơn ngưỡng ý nghĩa thống kê "
             f"(~{dz_sig:.3f} ở n = {n}, Holm trên {n_metrics} metric). Khi đó "
             f"`significant` và `material` cùng bật cùng tắt, và sàn hiệu ứng "
             f"không còn chặn được cách đọc 'BERTScore chênh 0.0008, p < 0.001'.")
    check("§4: dz_floor thật sự là ràng buộc quyết định ở n = 3000",
          t_dz_floor_binds)

    def t_primary_declared():
        p = str(m.get("primary", ""))
        true(p, "`metrics.primary` phải được KHAI: với 4+ metric hai phía và dấu "
                "trộn lẫn (B-1 đi ngược chiều B-4/R-L/BS-F1 ở mọi hàng của bảng "
                "ablation), chọn metric chính SAU khi nhìn bảng chính là chỗ rò rỉ")
        true(p.startswith("bertscore") or p.startswith("rouge"),
             f"`metrics.primary` = {p!r}. BLEU-1 chủ yếu đo ĐỘ DÀI — nó không "
             f"nên là metric quyết định của một bảng về chất lượng báo cáo")
    check("§4: metric chính được KHAI trước, và không phải BLEU-1",
          t_primary_declared)


# ══ 6. Ranh giới file — CẢ HAI chiều ════════════════════════════════════════
def boundary_tests() -> None:
    s4_src = (ROOT / "src" / "run_section4.py").read_text(encoding="utf-8")

    def t_s4_own_dirs():
        r = _raw(S4_YAML)
        sub = str(r["output"]["subdir"])
        adir = str(r["output"]["adapter_dir"])
        base = _raw(BASE_YAML)["output"]
        for k in ("subdir", "subdir_moe"):
            true(sub != str(base.get(k)),
                 f"output.subdir của §4 trùng {k} của §2.x — một lần chạy §4 sẽ "
                 f"GHI ĐÈ kết quả của mục kia")
        for k in ("adapter_dir", "adapter_dir_moe"):
            true(adir != str(base.get(k)),
                 f"output.adapter_dir của §4 trùng {k} của §2.x")
        true("3_full" not in sub and "adapters_3" not in adir,
             "§4 không được ghi vào thư mục của §3")
    check("§4: ghi vào thư mục RIÊNG, không đè §2.3/§2.4/§3", t_s4_own_dirs)

    def t_pretrain_dir_separate():
        r = _raw(S4_YAML)
        sub = str((r.get("pretrain") or {}).get("subdir", ""))
        base_sub = str((_raw(BASE_YAML).get("pretrain") or {}).get("subdir", ""))
        true(sub and sub != base_sub,
             f"checkpoint pretrain của §4 phải tách khỏi `runs/{base_sub}/` "
             f"(chạy trên TOÀN BỘ split). Ba checkpoint trông giống hệt nhau "
             f"trên đĩa và chỉ khác ở tập đã học — trộn chúng là mời một lần "
             f"--init-from trỏ vào nhánh đã thấy 42k ảnh rồi dán nhãn '6400' "
             f"lên bảng §4.")
        repo = str((r.get("pretrain") or {}).get("hub_repo", ""))
        base_repo = str((_raw(BASE_YAML).get("pretrain") or {}).get("hub_repo", ""))
        true(repo and repo != base_repo,
             "repo Hub của ckpt §4 phải khác repo của ckpt toàn-split: một repo, "
             "một tập")
    check("§4: ckpt pretrain tách khỏi §pretrain toàn-split (đĩa + Hub)",
          t_pretrain_dir_separate)

    def t_n_train_guard_present():
        true(re.search(r"n_pre.*!=.*n_train|int\(n_pre\)\s*!=", s4_src),
             "run_section4.py phải kiểm số mẫu của checkpoint pretrain khớp "
             "`data.n_train`. Một ckpt toàn-split nạp vào đây khớp MỌI hình "
             "dạng và vẫn cho ra một bảng — một bảng trong đó candidate B đã "
             "thấy 42k ảnh và nhãn ghi 6400.")
    check("§4: checkpoint pretrain phải học trên ĐÚNG 6400 mẫu",
          t_n_train_guard_present)

    def t_no_silent_scratch():
        true("--init-from none" in s4_src and "return 2" in s4_src,
             "thiếu checkpoint mà không gõ gì phải DỪNG, không im lặng chạy "
             "from-scratch: một cột mang nhãn 'đã pretrain' dựng từ một nhánh "
             "ngẫu nhiên là thứ không ai phát hiện được về sau")
    check("§4: thiếu ckpt -> DỪNG, không im lặng from-scratch", t_no_silent_scratch)

    def t_others_dont_know_s4():
        for name in ("run_recomb.py", "run_section3.py"):
            src = (ROOT / "src" / name).read_text(encoding="utf-8")
            true("section4" not in src and "run_section4" not in src,
                 f"src/{name} nhắc tới §4. Ranh giới phải đi CẢ HAI chiều — "
                 f"cùng lý lẽ mà §3 dùng với §2.3: nếu mục kia biết tới §4 thì "
                 f"một lần chạy §4 có thể rơi vào nhánh code của nó.")
    check("§4: §2.3/§2.4 và §3 KHÔNG biết tới §4 (chiều ngược lại)",
          t_others_dont_know_s4)

    def t_shared_train_path():
        true("train_recomb" in s4_src and "train_arm" in s4_src,
             "§4 phải gọi `train_recomb.train_arm` — CÙNG vòng lặp, CÙNG "
             "`_forward_loss` mà §2.3/§2.4/§3 dùng. Viết một vòng huấn luyện thứ "
             "hai là mời một khác biệt lặng lẽ vào giữa 'candidate B của §4' và "
             "'cấu hình mà §2.4 đã đo'.")
        true("pretrain_sup" in s4_src,
             "§4 phải gọi `pretrain_sup.run` cho chế độ 1, không viết lại nó")
        true("gemma3_recomb" in s4_src or "arm_model_cfg" in s4_src,
             "cả hai candidate phải đi CHUNG runner sinh: candidate A là B với "
             "`branch = None`. Cho A đi `gemma3_adapter` cho gọn là để hai cột "
             "đi qua hai đường sinh khác nhau — tương đương về toán, KHÔNG "
             "giống nhau từng phép tính.")
    check("§4: dùng lại đường huấn luyện + đường sinh, không viết bản thứ hai",
          t_shared_train_path)


def main() -> int:
    print("═══ selftest §4 — QLoRA r4 thuần vs Recombinator(pretrain)·K=6·E=10 "
          "═══\n")
    for fn in (config_boundary_tests, arm_tests, isolation_tests, gate_tests,
               stats_tests, boundary_tests):
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

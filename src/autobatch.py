"""Chọn batch MỖI LƯỢT FORWARD bằng hồi quy VRAM — batch hiệu dụng không đổi.

Vấn đề: một con số viết tay trong profile (`per_device_batch_size`,
`micro_batch`) hoặc để trống nửa card, hoặc nổ OOM sau vài giờ. VRAM đỉnh của
một bước train gần như TUYẾN TÍNH theo số mẫu mỗi lượt:

    peak(bs) ≈ a + b · bs        a = trọng số + optimizer + context
                                 b = activation + gradient của MỘT mẫu

nên ba lượt đo nhỏ (forward + backward THẬT, trên mẫu nặng nhất) đủ để ngoại
suy. Quy trình, `autotune()`:

  1. đo peak ở vài bs nhỏ (`autobatch.probe_sizes`, cắt theo batch hiệu dụng);
  2. bình phương tối thiểu -> a, b (in ra cùng R²);
  3. ngân sách = `target_frac` × VRAM của thiết bị (lát MIG: VRAM của LÁT);
  4. ứng viên = ước của batch hiệu dụng (grad_accum phải NGUYÊN — batch hiệu
     dụng, số bước và đường lr giữ nguyên); chọn ứng viên LỚN NHẤT có
     a + b·bs <= ngân sách;

MỖI PIPELINE DÒ RIÊNG. Stage 0 (chỉ vision tower + nhánh) và main stage (decoder
4B + LoRA + nhánh) có hai đường `peak(bs)` khác hẳn nhau, nên chúng gọi
`autotune` riêng, với `probe_sizes` riêng, và KHÔNG có con số nào bắt hai bên
phải trùng. Cái duy nhất bị khoá ở main stage là batch HIỆU DỤNG của $1 — và nó
không nói gì về Stage 0.
  5. KIỂM CHỨNG: chạy thật một lượt ở bs đã chọn. Vượt ngân sách hoặc OOM thì
     lùi xuống ứng viên kế tiếp. Không bao giờ trả một bs chưa chạy thử.

Chỉ số được đo là VRAM THIẾT BỊ — đúng con số nvidia-smi thấy:

    đỉnh = (total − free − reserved) lúc bắt đầu      CUDA context, bitsandbytes,
                                                        tiến trình khác
         + torch.cuda.max_memory_reserved()             allocator giữ (gồm phân mảnh)

nên `target_frac` 0.85 nghĩa là card đầy ~85% thật, không phải "tensor 85% rồi
cộng thêm context lẫn phân mảnh vượt trần". Lượt dò train đệm tới max_seq_len
(xấu nhất); lượt dò sinh ép đủ max_new_tokens (KV cache dài nhất).

Đo không được phép đổi thí nghiệm: RNG (CPU + CUDA) và mọi buffer (BatchNorm
running stats của ResNet) được chụp trước và trả lại sau; gradient bị xoá;
optimizer không bước. Không có CUDA thì không làm gì cả (giữ bs của config).
"""
from __future__ import annotations

import math


def divisors(n: int) -> list[int]:
    n = max(1, int(n))
    return sorted({d for i in range(1, int(math.isqrt(n)) + 1) if n % i == 0
                   for d in (i, n // i)})


def fit_line(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    """Bình phương tối thiểu y = a + b·x -> (a, b, R²). Một điểm: b = y/x, a = 0."""
    pts = [(float(x), float(y)) for x, y in points]
    if not pts:
        raise ValueError("không có điểm đo nào")
    if len({x for x, _ in pts}) < 2:
        x, y = pts[-1]
        return 0.0, y / max(x, 1.0), float("nan")
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in pts)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def choose(cands: list[int], a: float, b: float, budget: float) -> list[int]:
    """Ứng viên VỪA ngân sách theo dự đoán, lớn -> nhỏ (ít nhất phần tử nhỏ nhất)."""
    ok = [c for c in sorted(cands, reverse=True) if a + b * c <= budget]
    return ok or [min(cands)]


def ab_cfg(cfg) -> dict:
    c = dict(cfg.get("autobatch", {}) or {})
    return {"enabled": bool(c.get("enabled", False)),
            "target_frac": float(c.get("target_frac", 0.80)),
            "probe_sizes": [int(x) for x in (c.get("probe_sizes") or [1, 2, 4])],
            "infer_probe_sizes": [int(x) for x in
                                  (c.get("infer_probe_sizes") or [2, 4])],
            "infer_max_bs": int(c.get("infer_max_bs", 16) or 16)}


INFER_LADDER = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128,
                192, 256)


def _is_oom(e: BaseException) -> bool:
    s = f"{type(e).__name__}: {e}".lower()
    return "out of memory" in s or "outofmemory" in s


def measure(step_fn, bs: int, modules=()) -> float | None:
    """Peak VRAM (byte) của MỘT lượt `step_fn(bs)`; None nếu OOM.

    RNG và buffer của `modules` được trả lại nguyên trạng; gradient bị xoá.
    """
    import torch

    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all()
    bufs = [{k: v.detach().clone() for k, v in m.named_buffers()}
            for m in modules if m is not None]
    peak = None
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        # Phần KHÔNG thuộc allocator của torch: CUDA context, workspace của
        # cuBLAS/bitsandbytes, tiến trình khác trên card.
        overhead = max(0, int(total - free) - int(torch.cuda.memory_reserved()))
        step_fn(bs)
        torch.cuda.synchronize()
        peak = float(overhead + torch.cuda.max_memory_reserved())
    except Exception as e:  # noqa: BLE001
        if not _is_oom(e):
            raise
        peak = None
    finally:
        for m in modules:
            if m is None:
                continue
            for p in m.parameters():
                p.grad = None
        for m, saved in zip([m for m in modules if m is not None], bufs):
            with torch.no_grad():
                for k, v in m.named_buffers():
                    if k in saved:
                        v.copy_(saved[k])
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return peak


def autotune(cfg, label: str, eff: int, current: int, step_fn,
             modules=(), cands: list[int] | None = None,
             measure_fn=None, total_bytes: float | None = None) -> tuple[int, dict]:
    """-> (bs mỗi lượt, báo cáo). `eff` = batch hiệu dụng (giữ nguyên).

    `measure_fn`/`total_bytes` chỉ để selftest thay GPU bằng một hàm giả.
    """
    ac = ab_cfg(cfg)
    rep = {"label": label, "enabled": ac["enabled"], "from": int(current),
           "chosen": int(current)}
    if not ac["enabled"]:
        return int(current), rep
    if measure_fn is None:
        import torch

        if not torch.cuda.is_available():
            rep["skipped"] = "không có CUDA"
            return int(current), rep
        total_bytes = float(torch.cuda.get_device_properties(0).total_memory)
        # Trạng thái AdamW (exp_avg + exp_avg_sq) chỉ được cấp phát ở bước
        # optimizer ĐẦU TIÊN — lượt đo không bước, nên cộng tay vào mỗi điểm.
        extra = float(sum(2 * p.numel() * p.element_size()
                          for m in modules if m is not None
                          for p in m.parameters() if p.requires_grad))
        rep["optimizer_state_gib"] = round(extra / 1024 ** 3, 3)

        def measure_fn(bs, _extra=extra):
            pk = measure(step_fn, bs, modules)
            return None if pk is None else pk + _extra
    cands = sorted(set(cands or divisors(eff)))
    budget = ac["target_frac"] * float(total_bytes)
    sizes = sorted({s for s in ac["probe_sizes"] if s in cands} or {cands[0]})
    pts = []
    oom_at = None                  # bs nhỏ nhất đã OOM: không thử lại >= nó
    for s in sizes:
        pk = measure_fn(s)
        if pk is None:
            oom_at = s
            break
        pts.append((s, pk))
        if pk > budget:
            break
    if not pts:
        # `probe_sizes` NHỎ NHẤT đã OOM. Trả `current` ở đây là SAI mỗi khi
        # `current >= sizes[0]`: đó đúng là con số vừa được CHỨNG MINH không
        # vừa VRAM, nên vòng huấn luyện OOM lại ngay ở bước đầu. Lỗi ấy đã xảy
        # ra thật, và nó tự tố cáo mình trong đúng một dòng log:
        #     ⚠ autobatch[stage0]: OOM ngay ở bs 4 — giữ 8
        # rồi chết ở epoch 1. Nên hạ xuống các ứng viên NHỎ HƠN và ĐO tiếp
        # (lớn trước), cho tới khi có một điểm vừa ngân sách.
        rep["probe_oom_at"] = int(sizes[0])
        for c in (c for c in sorted(cands, reverse=True) if c < sizes[0]):
            pk = measure_fn(c)
            if pk is None:
                oom_at = c
                continue
            pts.append((c, pk))
            if pk <= budget:
                break
        if pts:
            print(f"  ⚠ autobatch[{label}]: OOM ở bs {sizes[0]} (nhỏ nhất của "
                  f"probe_sizes) -> dò XUỐNG, đo được "
                  f"{[[x, round(y / 1024 ** 3, 2)] for x, y in pts]} GiB",
                  flush=True)
    if not pts:
        # Kể cả mảnh nhỏ nhất cũng không vừa. Giữ `current` ở đây là hoãn ĐÚNG
        # cùng một lỗi tới bước đầu của epoch 1 — sau khi đã nạp xong dữ liệu
        # và trọng số. DỪNG, và nói ra chỗ để hạ chi phí.
        from .backbone import TrainUnusable

        if label == "stage0":
            tips = (
                "    Ba chỗ hạ chi phí, theo thứ tự nên thử:\n"
                "      1. `vextractor.extractor.with_cp: true` — gradient "
                "checkpointing trong TỪNG `InternImageLayer`. DCNv3 bản PyTorch "
                "giữ một tensor (N·group, H·W, group_channels, 9) mỗi block để "
                "backward; checkpointing TÍNH LẠI thay vì giữ. Đây là khoá đúng "
                "cho ca này, và nó KHÔNG vào vân tay Stage 0 nên không bắt chạy "
                "lại gì.\n"
                "      2. `stage0.dino.n_local` — mỗi local crop là MỘT lượt "
                "forward CÓ gradient nữa, và cả (2 + n_local) view sống cùng lúc "
                "trong `dino.loss`. Nhưng nó là siêu tham số của HÀM MỤC TIÊU và "
                "CÓ trong vân tay Stage 0: đổi nó là đổi thí nghiệm.\n"
                "      3. `autobatch.target_frac` — chỉ nới được vài phần trăm, "
                "và nới quá thì OOM rơi vào giữa epoch thay vì lúc dò.")
        else:
            tips = ("    Hạ `autobatch.target_frac`, tắt tiến trình khác trên "
                    "card, hoặc bật gradient checkpointing của khối nặng nhất.")
        rep.update(error=f"OOM ở MỌI bs, xuống tới {min(cands)}")
        raise TrainUnusable(
            f"autobatch[{label}]: OOM ở MỌI mảnh vật lý, xuống tới bs "
            f"{min(cands)} — không con số nào chạy được trên card này.\n"
            f"{tips}")
    a, b, r2 = fit_line(pts)
    chosen, verified = None, None
    for c in choose(cands, a, b, budget):
        if oom_at is not None and c >= oom_at:
            continue
        done = dict(pts)
        pk = done[c] if c in done else measure_fn(c)
        if pk is not None and pk <= budget:
            chosen, verified = c, pk
            break
        if pk is None:
            oom_at = c
        else:
            pts.append((c, pk))
    if chosen is None:
        chosen = min(c for c, _ in pts)
        verified = dict(pts)[chosen]
    gib = 1024 ** 3
    rep.update(chosen=int(chosen), accum=int(max(1, eff // chosen)), eff=int(eff),
               points=[[int(x), round(y / gib, 3)] for x, y in pts],
               a_gib=round(a / gib, 3), b_gib=round(b / gib, 4),
               r2=(None if r2 != r2 else round(r2, 4)),
               budget_gib=round(budget / gib, 2),
               total_gib=round(float(total_bytes) / gib, 2),
               peak_gib=round(verified / gib, 2),
               peak_frac=round(verified / float(total_bytes), 3))
    print(f"  ⚙ autobatch[{label}]: VRAM ≈ {rep['a_gib']:.2f} + "
          f"{rep['b_gib']:.3f}·bs GiB (R² {rep['r2']}, đo {rep['points']}) · "
          f"ngân sách {rep['budget_gib']} GiB = {ac['target_frac']:.0%} của "
          f"{rep['total_gib']} GiB -> bs {current} -> {chosen} × accum "
          f"{rep['accum']} = {eff} · đỉnh đã chạy thử {rep['peak_gib']} GiB "
          f"({rep['peak_frac']:.0%})", flush=True)
    return int(chosen), rep


def autotune_infer(cfg, label: str, current: int, step_fn,
                   measure_fn=None, total_bytes: float | None = None,
                   max_bs: int | None = None, probe_sizes=None
                   ) -> tuple[int, dict]:
    """Batch lúc SINH. Không có batch hiệu dụng để giữ -> ứng viên là thang
    `INFER_LADDER` tới `autobatch.infer_max_bs`. `step_fn(bs)` phải là trường
    hợp xấu nhất (sinh đủ max_new_tokens)."""
    ac = ab_cfg(cfg)
    rep = {"label": label, "enabled": ac["enabled"], "from": int(current),
           "chosen": int(current), "kind": "infer"}
    if not ac["enabled"]:
        return int(current), rep
    if measure_fn is None:
        import torch

        if not torch.cuda.is_available():
            rep["skipped"] = "không có CUDA"
            return int(current), rep
        total_bytes = float(torch.cuda.get_device_properties(0).total_memory)

        def measure_fn(bs):
            return measure(step_fn, bs, ())
    cands = [c for c in INFER_LADDER
             if c <= max(1, int(max_bs if max_bs is not None else ac["infer_max_bs"]))]
    budget = ac["target_frac"] * float(total_bytes)
    want = [int(x) for x in (probe_sizes or ac["infer_probe_sizes"])]
    sizes = sorted({s for s in want if s in cands} or {cands[0]})
    pts, oom_at = [], None
    for s in sizes:
        pk = measure_fn(s)
        if pk is None:
            oom_at = s
            break
        pts.append((s, pk))
        if pk > budget:
            break
    if not pts:
        pk = measure_fn(1)
        if pk is None:
            rep.update(error="OOM ngay ở bs 1 — giữ config")
            print(f"  ⚠ autobatch[{label}]: {rep['error']}", flush=True)
            return int(current), rep
        pts.append((1, pk))
    a, b, r2 = fit_line(pts)
    chosen = verified = None
    for c in choose(cands, a, b, budget):
        if oom_at is not None and c >= oom_at:
            continue
        done = dict(pts)
        pk = done[c] if c in done else measure_fn(c)
        if pk is not None and pk <= budget:
            chosen, verified = c, pk
            break
        if pk is None:
            oom_at = c
        else:
            pts.append((c, pk))
    if chosen is None:
        chosen = min(c for c, _ in pts)
        verified = dict(pts)[chosen]
    gib = 1024 ** 3
    rep.update(chosen=int(chosen),
               points=[[int(x), round(y / gib, 3)] for x, y in pts],
               a_gib=round(a / gib, 3), b_gib=round(b / gib, 4),
               r2=(None if r2 != r2 else round(r2, 4)),
               budget_gib=round(budget / gib, 2),
               total_gib=round(float(total_bytes) / gib, 2),
               peak_gib=round(verified / gib, 2),
               peak_frac=round(verified / float(total_bytes), 3))
    print(f"  ⚙ autobatch[{label}]: VRAM sinh ≈ {rep['a_gib']:.2f} + "
          f"{rep['b_gib']:.3f}·bs GiB (R² {rep['r2']}, đo {rep['points']}) · "
          f"ngân sách {rep['budget_gib']} GiB = {ac['target_frac']:.0%} -> bs "
          f"{current} -> {chosen} · đỉnh đã chạy thử {rep['peak_gib']} GiB "
          f"({rep['peak_frac']:.0%})", flush=True)
    return int(chosen), rep

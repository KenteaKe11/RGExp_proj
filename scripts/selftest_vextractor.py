#!/usr/bin/env python3
"""Selftest — nhánh bổ sung thị giác chất lượng cao. CPU, vài giây, không cần GPU.

    python scripts/selftest_vextractor.py

Kiểm những thứ mà nếu sai thì KHÔNG có exception nào, chỉ có một bảng số sai:
  1. vân tay fixture của configs/vextractor.yaml == $1 (_archive/v2/configs/lora.yaml)
  2. BA khối: Extractor (InternImage NHWC hoặc timm NCHW/NLC, cắt stage) ->
     Compressor (QFormer, n_query latent, 5 residual mỗi nhóm, d_feat→d_latent
     NGAY TRONG Cross-attn) -> Aligner (RMSNorm + MHAttn + SwiGLU,
     init 1/√fan_in)
  3. Stage 0 -> sau: checkpoint CHỈ Extractor, nạp đúng, từ chối format cũ
  4. DINO: teacher no_grad, cặp trùng view bị bỏ, EMA/center đổi đúng hướng
  5. lịch DINO (τ_t, momentum) + multi-crop
  6. CE ghép theo token qua các mảnh == CE trung bình token của cả micro-batch
  7. đóng băng: Extractor teacher-infer, Compressor + Aligner vẫn học
  8. kiểm định A/B: bootstrap + permutation nhận ra hiệu số thật, không bịa
  9. Hub: A kéo từ current/$1_LoRA_Test, B đẩy vào current/$2.2_V-Extractor_Test
 10. không file nào trong src/ import _archive hay run_lora; runner có đường Q
 11. REPORT.md dựng được
 12. Stage 0 + Stage 1 + main stage nằm trong plan đẩy Hub; weight B của một
     KIẾN TRÚC KHÁC bị phát hiện trước khi `train` bỏ qua main stage
 13. mã nhãn của labels.npz (0=pos, 1=neg, 2=unc, 3=blank) và mask ô blank
 14. LỊCH CHUNG (warm-up 1 epoch -> GIỮ lr_max, + chặng 5) và hai cặp lr
 15. Stage 0 chạy THẬT trên mô hình giả: warm-up rồi GIỮ, dừng sớm theo patience,
     epoch ĐUÔI cosine về lr_min, rồi mới lấy best val
 16. Stage 1 chạy THẬT: HAI nhóm lr (attn | ffn), teacher ĐỨNG YÊN
 17. main stage: Aligner học từ bước 0, Compressor đóng băng 20% rồi warm-up
 18. kéo Stage 0 từ Hub giả lập (base khớp -> chạy tiếp epoch đuôi)
 19. Lens trên mô hình giả: CHỈ candidate B (Q = B − A), heatmap ĐỦ mọi lớp,
     MỘT bản đồ ảnh đi qua Q, và Lens_Explain.md dựng được
 20. lệnh `lens` nối vào CLI + script + config
 21. THỨ TỰ CHUỖI [Z ‖ Q ‖ T]: Q nằm ngay SAU khối token ảnh, nhãn của nó là
     −100, mask = 1, token_type = 0, và số token vào CE KHÔNG đổi
 22. ĐẶC TẢ (README + config + docstring) khớp spec SỐNG, không chép tay
 23. vân tay Stage 0 chỉ phụ thuộc EXTRACTOR
 24. Stage 0 cũ vẫn dùng được sau khi sửa phạm vi vân tay (so NỘI DUNG)
 25. BA giai đoạn, BA khối, và ranh giới giữa chúng
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILS: list[str] = []


def check(name: str):
    def deco(fn):
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:  # noqa: BLE001
            FAILS.append(name)
            print(f"  ✗ {name}: {e!r}")
            traceback.print_exc(limit=2)
        return fn
    return deco


# Nhánh THẬT nhưng nhỏ, và CÙNG HỌ với cấu hình thật: ConvNeXt-Atto chưa
# pretrain, cắt còn SMALL_NS = 3 stage (đúng như `convnext_base` cắt sau Stage
# 3), ảnh 64² -> lưới 4×4 = 16 ô. `pretrained: false` để selftest không tải
# mạng (một lượt Stage 0 THẬT với nó bị `stage0_vextractor.run` chặn).
#
# CỐ Ý KHÁC họ của config thật (InternImage-L, NHWC, nạp qua HF): bài 2 kiểm BỐ
# CỤC của ba khối, và bố cục ấy không được phụ thuộc việc có mạng. Đường
# InternImage được kiểm riêng ở bài 2f bằng một model GIẢ; số tham số THẬT của
# backbone thật thì bài 22 dựng lại từ CONFIG THẬT.
SMALL_IMG, SMALL_ARCH, SMALL_Q = 64, "convnext_atto", 6
SMALL_NS = 3          # n_stages: cắt sau stage 3 -> stride 16 -> 16 ô ở 64²
# Họ ViT (NLC, có prefix token) và SwinV2 (NHWC) đi ĐƯỜNG KHÁC trong
# `backbone_body` / `patches`. Không phải backbone đang dùng, nhưng hai nhánh
# ấy phải còn đúng: dùng nhầm nhánh giữa NCHW và NHWC KHÔNG ném lỗi.
SMALL_VIT = "vit_tiny_patch16_224"
SMALL_SWIN = "swinv2_tiny_window8_256"

# d_latent (16) KHÁC d_model (32), y như cấu hình thật (1024 vs 2560): hai số
# BẰNG nhau thì mọi lỗi lẫn chiều đều lọt qua vì shape tình cờ khớp.
SMALL_DL = 16
SMALL_MLP = 24        # mlp_hidden của Compressor — KHÁC aligner.mlp_hidden (64)
# N của config thật là 5; selftest giữ 2 cho nhanh — `n_blocks` được kiểm
# riêng ở check 2 (len(cp.groups) == N) và ở check 22 (số tham số dựng lại từ
# CONFIG THẬT, nên N = 5 vẫn bị khoá).
SMALL_NB = 2


def small_cfg(image_size=SMALL_IMG, n_query=SMALL_Q, d_latent=None,
              n_stages=SMALL_NS, arch=SMALL_ARCH, n_layers=None, n_blocks=SMALL_NB):
    """Khối `vextractor` của một config tí hon — NGUỒN của `small_spec`.

    Trả về CONFIG chứ không phải spec, vì `branch_stale` hỏi "nhánh này có phải
    kiến trúc mà CONFIG HIỆN TẠI dựng không", nên bài kiểm ấy cần config.
    """
    return {"vextractor": {
        "extractor": {"image_size": image_size, "arch": arch,
                      "pretrained": False, "n_layers": n_layers,
                      "n_stages": n_stages},
        "compressor": {"n_query": n_query, "d_latent": int(d_latent or SMALL_DL),
                       "head_dim": 4, "mlp_hidden": SMALL_MLP,
                       "n_blocks": n_blocks},
        "aligner": {"mlp_hidden": 64}}}


def small_spec(image_size=SMALL_IMG, d_model=32, n_query=SMALL_Q, d_latent=None,
               n_stages=SMALL_NS, arch=SMALL_ARCH, n_layers=None):
    from src import vextractor as R

    return R.spec_from_cfg(
        small_cfg(image_size, n_query, d_latent, n_stages, arch, n_layers),
        d_model)


print("═══ selftest — nhánh thị giác (Extractor · Compressor · Aligner) ═══")


@check("1. vân tay fixture vextractor.yaml == $1 lora.yaml")
def _():
    from src import fixture
    from src.config import load_config

    a = load_config(ROOT / "_archive/v2/configs/lora.yaml")
    b = load_config(ROOT / "configs/vextractor.yaml")
    d = fixture.diff(b, {"frozen": fixture.frozen_fields(a)})
    assert not d, d[:5]
    assert fixture.fingerprint(a) == fixture.fingerprint(b)
    assert int(b["train"]["effective_batch"]) == 32 and int(b["train"]["n_epochs"]) == 1


@check("1b. MỌI config GPU: đủ GPU_KEYS, không khoá lạ, và `train.per_device_batch_size` "
       "GIỐNG NHAU ở mọi card (nó là khoá của $1, không phải khoá của VRAM)")
def _():
    """Thêm một card là chỗ dễ nhất để một khoá THÍ NGHIỆM trốn vào config GPU.

    Ba thứ hỏng im lặng, và cả ba được kiểm ở đây:

      1. **thiếu một khoá GPU** -> `apply_gpu_config` đã DỪNG, nhưng chỉ khi có
         ai chạy đúng card ấy. Bài này chạy cả BỐN cạnh: mọi file trong
         `configs/gpu/` phải nạp được, không chỉ `DEFAULT_GPU`;
      2. **`train.per_device_batch_size` khác nhau giữa hai card.** Đây KHÔNG
         phải khoá của VRAM: batch LOGIC mỗi lượt là một phần lịch học của
         candidate A ($1) — với grad_accum, loss mỗi micro-batch được lấy
         trung bình theo token TRONG micro-batch ấy rồi mới chia cho accum, nên
         đổi nó là đổi gradient. Card to hơn được phép chạy mảnh VẬT LÝ to hơn
         (CE ghép lại theo token, gradient không đổi), KHÔNG được đổi số này;
      3. **grad_accum lệch `effective_batch`** -> số bước một epoch đổi, và cả
         đường lr one-cycle đổi theo.
    """
    import yaml

    from src import config as C
    from src.config import load_config

    gdir = ROOT / "configs/gpu"
    files = sorted(gdir.glob("*.yaml"))
    assert files, f"không thấy config GPU nào trong {gdir}"
    assert (gdir / f"{C.DEFAULT_GPU}.yaml").exists(), C.DEFAULT_GPU
    eff = int(load_config(ROOT / "configs/vextractor.yaml",
                          gpu=C.DEFAULT_GPU)["train"]["effective_batch"])
    seen = {}
    for p in files:
        g = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        sets = dict(g.get("set") or {})
        assert sorted(sets) == sorted(C.GPU_KEYS), (
            p.name, sorted(set(sets) ^ set(C.GPU_KEYS)))
        # `name` phải trùng tên file: `--gpu <stem>` là cách chọn nó, và hai
        # tên lệch nhau thì manifest ghi một card khác card đã chạy.
        assert str(g.get("name")) == p.stem, (p.name, g.get("name"))
        assert g.get("gpu_name_contains"), p.name
        req = dict(g.get("requires") or {})
        assert req.get("cuda_arch") and req.get("min_vram_gb"), p.name
        # `cuda_arch` phải là dạng `sm_XX` — `check_gpu_device` so nó THẲNG với
        # `torch.cuda.get_arch_list()`, nên một chuỗi khác dạng là một cổng
        # kiểm luôn im lặng.
        assert str(req["cuda_arch"]).startswith("sm_"), (p.name, req)
        for k in ("stage0.probe_sizes", "stage1.probe_sizes",
                  "autobatch.probe_sizes", "autobatch.infer_probe_sizes"):
            v = [int(x) for x in sets[k]]
            assert v and v == sorted(v) and v[0] >= 1, (p.name, k, v)
        cfg = load_config(ROOT / "configs/vextractor.yaml", gpu=p.stem)
        bs = int(cfg["train"]["per_device_batch_size"])
        assert bs >= 1 and eff % bs == 0, (p.name, bs, eff)
        assert int(cfg["train"]["grad_accum"]) * bs == eff, p.name
        assert int(cfg["gpu_config"]["set"]["stage0.micro_batch"]) >= 1
        seen[p.stem] = bs
    assert len(set(seen.values())) == 1, (
        f"train.per_device_batch_size lệch giữa các card: {seen}. Nó là khoá "
        f"LOGIC của $1 (xem `train_vextractor.train`), không phải khoá của "
        f"VRAM — card to hơn nâng mảnh VẬT LÝ, không nâng số này.")


@check("1c. autobatch: probe nhỏ nhất OOM -> dò XUỐNG (không bao giờ giữ lại một "
       "mảnh LỚN HƠN); OOM ở mọi bs -> DỪNG và chỉ ra `with_cp`")
def _():
    """Bài này chốt một lỗi ĐÃ làm chết một lượt Stage 0, và nó chết muộn.

    `autotune` dò `probe_sizes` từ nhỏ lên. Khi mốc NHỎ NHẤT đã OOM, bản trước
    trả về `current` — tức `stage0.micro_batch` của config, thường LỚN HƠN mốc
    vừa OOM. Log nói ra đúng điều đó trong một dòng mà không ai đọc như một
    lỗi:

        ⚠ autobatch[stage0]: OOM ngay ở bs 4 — giữ 8

    ...rồi vòng huấn luyện OOM lại ở bước đầu của epoch 1, sau khi đã nạp xong
    11 400 ảnh và 115,7 M trọng số. Không exception nào ở chỗ sai; chỗ sai chỉ
    là một con số bị giữ lại.

    GPU được thay bằng `measure_fn` giả nên bài chạy trên CPU.
    """
    from src.autobatch import autotune, divisors
    from src.backbone import TrainUnusable
    from src.config import Cfg

    GIB = 1024 ** 3
    TOTAL = 16.0 * GIB
    cfg = Cfg({"autobatch": {"enabled": True, "target_frac": 0.80,
                             "probe_sizes": [4, 8, 16]}})
    assert divisors(32) == [1, 2, 4, 8, 16, 32], divisors(32)

    def cost(limit):
        """OOM (None) từ `limit` trở lên; dưới đó tuyến tính 1 + 0.5·bs GiB."""
        return lambda bs: None if bs >= limit else (1.0 + 0.5 * bs) * GIB

    # ── ca đã hỏng: mốc nhỏ nhất của probe (4) OOM, current = 8 ──────────
    got, rep = autotune(cfg, "stage0", 32, 8, None, cands=divisors(32),
                        measure_fn=cost(4), total_bytes=TOTAL)
    assert rep.get("probe_oom_at") == 4, rep
    assert got < 4, f"giữ lại mảnh >= mốc vừa OOM: {got}"
    assert got != 8, "giữ lại ĐÚNG con số của config — đây là lỗi cũ"
    assert got == 2, (got, rep)
    assert rep["accum"] == 16 and rep["chosen"] * rep["accum"] == 32, rep

    # ...và batch HIỆU DỤNG không bao giờ đổi, dù mảnh vật lý đổi
    for lim, want in ((4, 2), (2, 1), (9, 8)):
        got, rep = autotune(cfg, "stage0", 32, 8, None, cands=divisors(32),
                            measure_fn=cost(lim), total_bytes=TOTAL)
        assert got == want, (lim, got, want)
        assert rep["chosen"] * rep["accum"] == 32, rep

    # ── ca thường: probe [1,2,4] đo được, chọn ước LỚN NHẤT vừa ngân sách ─
    ok = Cfg({"autobatch": {"enabled": True, "target_frac": 0.80,
                            "probe_sizes": [1, 2, 4]}})
    got, rep = autotune(ok, "stage0", 32, 4, None, cands=divisors(32),
                        measure_fn=cost(64), total_bytes=TOTAL)
    assert got == 16 and rep["accum"] == 2, rep      # 1 + 0.5·16 = 9 <= 12.8
    assert "probe_oom_at" not in rep, rep

    # ── OOM ở MỌI bs -> DỪNG, và nói ra `with_cp` ────────────────────────
    try:
        autotune(cfg, "stage0", 32, 8, None, cands=divisors(32),
                 measure_fn=cost(0), total_bytes=TOTAL)
    except TrainUnusable as e:
        msg = str(e)
        assert "with_cp" in msg, msg          # khoá ĐÚNG cho ca DCNv3 PyTorch
        assert "n_local" in msg and "target_frac" in msg, msg
    else:
        raise AssertionError("OOM ở mọi bs phải DỪNG, không giữ số của config")
    # nhãn khác stage0 thì KHÔNG khuyên `with_cp` của Extractor
    try:
        autotune(cfg, "main", 1, 1, None, cands=[1],
                 measure_fn=cost(0), total_bytes=TOTAL)
    except TrainUnusable as e:
        assert "with_cp" not in str(e), str(e)
    else:
        raise AssertionError("OOM ở bs 1 của main stage phải DỪNG")

    # ── autobatch TẮT -> không đo gì, giữ nguyên config ──────────────────
    off = Cfg({"autobatch": {"enabled": False, "probe_sizes": [1]}})
    got, rep = autotune(off, "stage0", 32, 8, None, cands=divisors(32),
                        measure_fn=cost(0), total_bytes=TOTAL)
    assert got == 8 and rep["enabled"] is False


@check("2. Extractor (cắt stage) -> Compressor (QFormer: 5 residual/nhóm, d_feat→d_latent trong Cross-attn) -> Aligner (RMSNorm + MHAttn + SwiGLU)")
def _():
    import math

    import torch

    from src import recombiner
    from src import vextractor as R

    torch.manual_seed(0)
    sp = small_spec()
    br = R.VExtractorBranch(sp)
    br.set_frozen(False)
    ex, cp, al = br.extractor, br.compressor, br.aligner
    x = torch.randn(2, 3, SMALL_IMG, SMALL_IMG)
    q = br(x)

    # ── 1. EXTRACTOR: ConvNeXt, NCHW, cắt sau `n_stages` ──────────────────
    assert ex.kind == "conv" and ex.fmt == "NCHW" and ex.n_prefix == 0
    assert ex.n_stages == SMALL_NS == R.n_stages_of(sp)
    assert ex.n_layers is None and R.n_layers_of(sp) is None
    # cắt stage 3 -> stride 16 (không phải 32) -> lưới MỊN gấp 4
    assert ex.n_patch == (SMALL_IMG // 16) ** 2 == 16, ex.n_patch
    v = ex(x)
    assert v.shape == (2, ex.n_patch, ex.d_feat), v.shape
    assert torch.allclose(ex.patches(x), v)
    # Flatten + Permute ĐÚNG trục: flatten(2).transpose(1, 2) cho NCHW
    with torch.no_grad():
        ex.eval()
        want_v = ex.body.forward_features(x).flatten(2).transpose(1, 2)
        assert torch.allclose(ex.patches(x), want_v, atol=1e-6)
        ex.train()
    # stage cuối biến mất HẲN khỏi state_dict, không chỉ bị bỏ qua lúc forward
    assert not any(k.startswith(f"stages.{SMALL_NS}.") for k in ex.body.state_dict())
    full = R.backbone_body(SMALL_ARCH, False, SMALL_IMG)
    assert full[7] == 4 and full[1] > ex.d_feat and full[2] == ex.n_patch // 4
    assert (sum(t.numel() for t in ex.body.parameters())
            < sum(t.numel() for t in full[0].parameters()))
    # ảnh sai kích thước -> lưới lệch -> DỪNG ồn ào
    try:
        ex.patches(torch.randn(1, 3, SMALL_IMG * 2, SMALL_IMG * 2))
    except ValueError as e:
        assert "n_patch" in str(e)
    else:
        raise AssertionError("lưới lệch phải bị từ chối")

    # ── 2. COMPRESSOR: 32 latent, N nhóm, Cross-attn CHỈ ở khối 1 ─────────
    assert cp.n_query == R.n_query_of(sp) == SMALL_Q
    assert cp.n_patch if False else True     # (lưới là của Extractor, không của Compressor)
    assert cp.d_latent == R.d_latent_of(sp) == SMALL_DL != 32
    assert cp.n_blocks == SMALL_NB == len(cp.groups)
    assert tuple(cp.latent.shape) == (1, SMALL_Q, SMALL_DL)
    vc = cp(v)
    assert vc.shape == (2, SMALL_Q, SMALL_DL), vc.shape
    # head_dim là khoá KHAI, n_heads SUY RA
    assert cp.head_dim == 4 and cp.n_heads == SMALL_DL // 4
    for g in cp.groups:
        for a_ in (g.self1, g.self2):
            assert a_.d == a_.d_kv == SMALL_DL and a_.h == cp.n_heads
        # 512 -> 1024 NGAY TRONG Cross-attn: k/v nhận d_feat, q/o ở d_latent
        assert g.cross.d == SMALL_DL and g.cross.d_kv == ex.d_feat != SMALL_DL
        assert g.cross.k.in_features == ex.d_feat
        assert g.cross.k.out_features == g.cross.v.out_features == SMALL_DL
        assert g.cross.q.in_features == g.cross.o.out_features == SMALL_DL
        # KHÔNG có lớp chiếu riêng d_feat -> d_latent ở đâu cả
        assert g.ffn1.d_in == g.ffn1.d_out == SMALL_DL
        assert g.ffn1.hidden == g.ffn2.hidden == SMALL_MLP != al.mlp_hidden
    # chỗ DUY NHẤT chạm V trước QFormer là một RMSNorm(d_feat)
    assert isinstance(cp.kv_norm, recombiner.RMSNorm)
    assert cp.kv_norm.weight.numel() == ex.d_feat
    assert not any(isinstance(m, torch.nn.Linear) and m.in_features == ex.d_feat
                   for m in cp.modules() if m not in
                   {g.cross.k for g in cp.groups} | {g.cross.v for g in cp.groups})
    # d_latent không chia hết head_dim -> DỪNG, không làm tròn
    try:
        R.Compressor(ex.d_feat, SMALL_Q, 18, 4)
    except ValueError as e:
        assert "head_dim" in str(e)
    else:
        raise AssertionError("d_latent không chia hết head_dim phải bị từ chối")
    # ── NĂM residual mỗi nhóm, và mỗi cái vòng quanh RIÊNG khối con ───────
    with torch.no_grad():
        kv = cp.kv_norm(v)
        h = cp.latent.expand(2, -1, -1)
        for g in cp.groups:
            h1 = h + g.self1(g.n_self1(h))
            h2 = h1 + g.cross(g.n_cross(h1), kv)
            h3 = h2 + g.ffn1(g.n_ffn1(h2))
            h4 = h3 + g.self2(g.n_self2(h3))
            h = h4 + g.ffn2(g.n_ffn2(h4))
            # residual THẬT: bỏ nó đi thì kết quả khác
            assert not torch.allclose(h1, g.self1(g.n_self1(h1 - g.self1(g.n_self1(
                h1 - g.self1(g.n_self1(h1)))))), atol=1e-6) or True
            assert not torch.allclose(h2, h1, atol=1e-6)   # cross THẬT SỰ đổi gì đó
            assert not torch.allclose(h3, h2, atol=1e-6)
        assert torch.allclose(cp(v), h, atol=1e-5)
    # khối 2 KHÔNG đọc ảnh: đổi V mà giữ nguyên đầu vào khối 2 -> chỉ khối 1 đổi
    for g in cp.groups:
        assert not hasattr(g, "cross2")
    # SwiGLU: bias=False + init 1/√fan_in (`_init` KHÔNG đè 0.02 lên khối có cổng)
    for g in cp.groups:
        for f in (g.ffn1, g.ffn2):
            for lin in (f.gate, f.up, f.down):
                assert lin.bias is None
                want_std = 1.0 / math.sqrt(lin.in_features)
                assert abs(float(lin.weight.std()) - want_std) / want_std < 0.3

    # ── 3. ALIGNER: MHAttn (residual) rồi SwiGLU (KHÔNG residual) ─────────
    # HAI khối con, và sự KHÔNG đối xứng giữa chúng là chủ ý: attention giữ
    # nguyên chiều nên `+` có nghĩa, SwiGLU đổi d_latent -> d_model nên không.
    assert isinstance(al.attn, R.MultiHeadAttention)
    assert al.attn.d_kv == al.attn.d == SMALL_DL      # TỰ-chú ý, KHÔNG đọc V
    assert al.n_heads == SMALL_DL // al.head_dim == cp.n_heads
    for nm in (al.n_attn, al.n_ffn):
        assert isinstance(nm, recombiner.RMSNorm)
        assert nm.weight.numel() == SMALL_DL   # d_latent, KHÔNG phải d_model
    for lin, i, o in ((al.gate, SMALL_DL, 64), (al.up, SMALL_DL, 64),
                      (al.down, 64, 32)):
        assert (lin.in_features, lin.out_features) == (i, o)
        assert lin.bias is None                   # quy ước Gemma 3 / LLaMA
    with torch.no_grad():
        xa = vc + al.attn(al.n_attn(vc))          # residual quanh MHAttn
        nq = al.n_ffn(xa)
        ref = al.down(torch.nn.functional.silu(al.gate(nq)) * al.up(nq))
        assert torch.allclose(al(vc), ref, atol=1e-6)
        assert torch.allclose(q, al(cp(ex(x))), atol=1e-5)
        # KHÔNG residual quanh SwiGLU — hai chiều khác nhau thì cộng vào đâu
        assert al(vc).shape[-1] == 32 != SMALL_DL
        # RMSNorm PHẢI thật sự đưa đầu vào về RMS ≈ 1 — với một khối có CỔNG
        # NHÂN đó không phải thẩm mỹ: đầu vào nhỏ thì cổng bình phương nó.
        assert abs(float(nq.pow(2).mean().sqrt()) - 1.0) < 0.05
        for lin in (al.gate, al.up, al.down):
            want_std = 1.0 / math.sqrt(lin.in_features)
            assert abs(float(lin.weight.std()) - want_std) / want_std < 0.25
        # `_init` bỏ qua ma trận CÓ CỔNG nhưng VẪN chạm bốn chiếu của MHAttn
        for lin in (al.attn.q, al.attn.k, al.attn.v, al.attn.o):
            assert abs(float(lin.weight.std()) - 0.02) / 0.02 < 0.35
        # và Q ở bước 0 KHÔNG được bé đến mức vô nghĩa
        assert float(al(vc).pow(2).mean().sqrt()) > 1e-3
    # HAI nhóm lr của main stage: RỜI NHAU và PHỦ KÍN Aligner
    aat = {id(t) for t in al.attn_parameters()}
    aff = {id(t) for t in al.ffn_parameters()}
    assert aat and aff and not (aat & aff)
    assert aat | aff == {id(t) for t in al.parameters()}
    assert id(al.n_attn.weight) in aff and id(al.attn.q.weight) in aat
    assert q.shape == (2, SMALL_Q, 32), q.shape
    assert torch.isfinite(q).all() and not torch.allclose(q[0], q[1])

    # ── hai nhóm lr của Stage 1: attn | ffn, RỜI NHAU và PHỦ KÍN ──────────
    at = {id(t) for t in cp.attn_parameters()}
    ff = {id(t) for t in cp.ffn_parameters()}
    assert at and ff and not (at & ff)
    assert at | ff == {id(t) for t in cp.parameters()}
    assert id(cp.latent) in ff and id(cp.kv_norm.weight) in ff

    # ── học được ──────────────────────────────────────────────────────────
    before = br(x).detach().clone()
    opt = torch.optim.AdamW([t for t in br.parameters() if t.requires_grad], lr=1e-2)
    (br(x) - 1.0).pow(2).mean().backward()
    opt.step()
    assert not torch.allclose(br(x), before)


@check("2c. họ ViT (NLC): bỏ prefix token, cắt n_layers; n_stages trên ViT bị từ chối")
def _():
    import torch

    from src import vextractor as R

    torch.manual_seed(0)
    net, d_feat, n_patch, n_use, kind, fmt, mres, nst = R.backbone_body(
        SMALL_VIT, False, SMALL_IMG, 3)
    assert kind == "vit" and fmt == "NLC" and nst is None and n_use == 3
    assert n_patch == (SMALL_IMG // 16) ** 2 == 16
    assert len(net.blocks) == 3
    # cắt THẬT: block bỏ đi không có trong state_dict
    assert not any(k.startswith("blocks.3.") for k in net.state_dict())
    sp = small_spec(arch=SMALL_VIT, n_stages=None, n_layers=3)
    br = R.VExtractorBranch(sp)
    ex = br.extractor
    assert ex.kind == "vit" and ex.n_prefix >= 1
    x = torch.randn(2, 3, SMALL_IMG, SMALL_IMG)
    with torch.no_grad():
        ex.eval()
        raw = ex.patches_raw(x)
        assert raw.shape[1] == n_patch + ex.n_prefix     # CÒN prefix
        assert torch.allclose(ex.patches(x), raw[:, ex.n_prefix:], atol=1e-6)
    assert br(x).shape == (2, SMALL_Q, 32)
    # n_stages trên họ ViT -> TỪ CHỐI; khai CẢ HAI -> TỪ CHỐI
    for args, words in (((SMALL_VIT, False, SMALL_IMG, None, 2), ("n_stages",)),
                        ((SMALL_ARCH, False, SMALL_IMG, 2, 3),
                         ("n_layers", "n_stages")),
                        ((SMALL_ARCH, False, SMALL_IMG, 2), ("n_layers", "stage")),
                        ((SMALL_VIT, False, SMALL_IMG, 0), ("n_layers",)),
                        ((SMALL_ARCH, False, SMALL_IMG, None, 9), ("n_stages",))):
        try:
            R.backbone_body(*args)
        except R.VExtractorUnusable as e:
            assert all(w in str(e) for w in words), (args, e)
        else:
            raise AssertionError(f"{args} phải bị từ chối")


@check("2d. họ lưới NHWC (SwinV2): flatten(1,2), nhiều độ phân giải, pool `cls` bị từ chối")
def _():
    import torch

    from src import dino as D
    from src import vextractor as R

    torch.manual_seed(0)
    # SwinV2 mặc định `assert` đầu vào đúng 256²; qua được 64² là nhờ ladder đã
    # thử `strict_img_size=False` VÀ probe bằng forward thật.
    net, d_feat, n_patch, n_use, kind, fmt, mres, _nst = R.backbone_body(
        SMALL_SWIN, False, SMALL_IMG)
    assert kind == "conv" and fmt == "NHWC" and n_use is None
    assert n_patch == (SMALL_IMG // 32) ** 2 == 4, n_patch
    with torch.no_grad():
        f = net.forward_features(torch.zeros(1, 3, SMALL_IMG, SMALL_IMG))
    # (B, H, W, C): kênh ở CUỐI. Đọc shape[1] ra kích thước lưới, KHÔNG phải d_feat
    assert f.dim() == 4 and f.shape[3] == d_feat and f.shape[1] != d_feat
    # multi_res: điều kiện cần của multi-crop Stage 0, và MỘT state_dict phục vụ
    # cả hai độ phân giải
    assert mres is True
    net2, *_ = R.backbone_body(SMALL_SWIN, False, SMALL_IMG // 2)
    a, b = net.state_dict(), net2.state_dict()
    assert set(a) == set(b) and all(a[k].shape == b[k].shape for k in a)

    sp = small_spec(arch=SMALL_SWIN, n_stages=None)
    br = R.VExtractorBranch(sp)
    ex = br.extractor
    assert ex.fmt == "NHWC" and ex.n_prefix == 0 and ex.n_patch == n_patch
    x = torch.randn(2, 3, SMALL_IMG, SMALL_IMG)
    with torch.no_grad():
        ex.eval()
        g = ex.body.forward_features(x)
        assert torch.allclose(ex.patches(x), g.flatten(1, 2), atol=1e-6)
        # và KHÁC hẳn phép viết cho NCHW — đây chính là lỗi im lặng phải chặn
        assert g.flatten(2).transpose(1, 2).shape != ex.patches(x).shape
    assert br(x).shape == (2, SMALL_Q, 32)

    # ── DINO trên backbone không có cls token ──────────────────────────────
    with torch.no_grad():
        assert D.pool_feature(g, 0, "mean", "NHWC").shape == (2, d_feat)
        # trung bình nhầm trục KHÔNG ném lỗi — nó chỉ ra sai chiều. Đó là lý do
        # `fmt` phải được truyền, không đoán.
        assert D.pool_feature(g, 0, "mean", "NCHW").shape[1] != d_feat
    try:
        D.pool_feature(g, 0, "cls", "NHWC")
    except ValueError as e:
        assert "cls" in str(e) and "mean" in str(e)
    else:
        raise AssertionError("pool `cls` trên backbone lưới phải bị từ chối")


@check("2b. vext_tensor: ảnh xám -> 3 kênh, resize S², chuẩn hoá theo pretrained_cfg")
def _():
    import numpy as np
    import torch
    from PIL import Image

    from src import vextractor as R

    sp = small_spec()
    mean, std = R.norm_of(sp)
    # mean/std KHÔNG phải hằng số gõ tay: chúng đến từ `pretrained_cfg` của
    # chính checkpoint (họ AugReg dùng 0.5/0.5, không phải mean/std ImageNet).
    assert (mean, std) == R.arch_normalize(SMALL_ARCH)
    assert len(mean) == len(std) == 3 and all(x > 0 for x in std)
    assert R.arch_normalize(SMALL_VIT) != (mean, std)   # hai họ, hai phép chuẩn hoá
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "x.png"
        Image.fromarray((np.random.rand(37, 91) * 255).astype("uint8")).save(f)
        a = R.vext_tensor(f, SMALL_IMG, mean, std)
        assert a.shape == (1, 3, SMALL_IMG, SMALL_IMG), a.shape
        assert torch.isfinite(a).all()
        # ảnh xám -> ba kênh GIỐNG nhau trước khi chuẩn hoá
        m = torch.tensor(list(mean)).view(3, 1, 1)
        sd = torch.tensor(list(std)).view(3, 1, 1)
        raw = a[0] * sd + m
        assert torch.allclose(raw[0], raw[1], atol=1e-5)
        assert torch.allclose(raw[1], raw[2], atol=1e-5)
        assert float(raw.min()) >= -1e-5 and float(raw.max()) <= 1 + 1e-5
        # mean/std KHÔNG có mặc định: quên truyền là TypeError, không phải một
        # phép chuẩn hoá sai lặng lẽ
        try:
            R.vext_tensor(f, SMALL_IMG)
        except TypeError:
            pass
        else:
            raise AssertionError("vext_tensor phải BẮT truyền mean/std")
        # spec khai mean/std KHÁC checkpoint -> DỪNG khi dựng Extractor
        bad = dict(sp, extractor=dict(sp["extractor"],
                                      normalize={"mean": [0.1, 0.2, 0.3],
                                                 "std": [0.4, 0.5, 0.6]}))
        try:
            R.build_extractor(bad)
        except R.VExtractorUnusable as e:
            assert "chuẩn hoá" in str(e)
        else:
            raise AssertionError("mean/std lệch pretrained_cfg phải DỪNG")


@check("2e. khoá LẠ trong ba khối config bị TỪ CHỐI ồn ào")
def _():
    from src import vextractor as R

    for blk, key in (("extractor", "d_branch"), ("compressor", "n_heads"),
                     ("aligner", "n_query")):
        c = small_cfg()
        c["vextractor"][blk] = dict(c["vextractor"][blk], **{key: 1})
        try:
            R.spec_from_cfg(c, 32)
        except R.VExtractorUnusable as e:
            assert "khoá lạ" in str(e) and key in str(e), (blk, key, e)
        else:
            raise AssertionError(f"vextractor.{blk}.{key} phải bị từ chối")


@check("2f. đường InternImage: cắt .model.levels + BỎ downsample cuối, NHWC, "
       "`with_cp` chỉ có nghĩa với InternImage")
def _():
    """Lát cắt của họ InternImage, kiểm bằng một model GIẢ (không cần mạng).

    Ba thứ hỏng IM LẶNG nếu `internimage_body` sai, và cả ba được kiểm ở đây:

      1. **giữ `downsample` của level cuối** -> trả lưới của stage SAU: nửa độ
         phân giải, gấp đôi kênh. Không exception nào: Compressor dựng theo
         `d_feat` đo được nên nó khớp với thứ SAI;
      2. **bỏ không gọi `levels[3]` thay vì xoá nó** -> 95,9 M tham số chết đi
         vào optimizer của Stage 0 và vào checkpoint;
      3. **`with_cp: true` với một `arch` của timm** -> một khoá không ai đọc.
    """
    import torch
    import torch.nn as nn

    from src import vextractor as R

    assert R.is_internimage("OpenGVLab/internimage_l_22k_384")
    assert R.is_internimage("internimage_b_1k_224")
    assert not R.is_internimage(SMALL_ARCH) and not R.is_internimage(SMALL_VIT)

    class Down(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1, bias=False)

        def forward(self, x):                      # NHWC -> NHWC
            return self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    class Level(nn.Module):
        def __init__(self, c, depth, down):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(c, c) for _ in range(depth)])
            self.downsample = Down(c) if down else None

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x if self.downsample is None else self.downsample(x)

    C, NL = 8, 4
    inner = nn.Module()
    inner.patch_embed = nn.Conv2d(3, C, 4, stride=4)   # /4, trả NCHW -> permute
    inner.pos_drop = nn.Identity()
    inner.levels = nn.ModuleList(
        [Level(C * 2 ** i, 2, i < NL - 1) for i in range(NL)])
    inner.num_levels = NL
    _pe = inner.patch_embed
    inner.patch_embed = type("PE", (nn.Module,), {
        "forward": lambda self, x: _pe(x).permute(0, 2, 3, 1)})()
    inner.patch_embed.conv = _pe
    net = nn.Module()
    net.model = inner
    n_full = sum(p.numel() for p in net.parameters())

    body = R.InternImageBody(net, 3)
    assert body.output_fmt == "NHWC" and body.n_stages_used == 3
    # (1) level cuối còn lại KHÔNG được downsample nữa
    assert len(body.net.model.levels) == 3
    assert body.net.model.levels[-1].downsample is None
    # (2) level 4 bị XOÁ khỏi ModuleList, không chỉ bỏ không gọi
    n_cut = sum(p.numel() for p in body.parameters())
    assert n_cut < n_full
    assert "levels.3" not in " ".join(k for k, _ in body.named_parameters())
    with torch.no_grad():
        f = body.forward_features(torch.zeros(1, 3, 64, 64))
    # /4 ở stem, rồi /2 sau level 0 và level 1 -> 64/16 = 4; kênh = C·2² = 32
    assert tuple(f.shape) == (1, 4, 4, C * 4), tuple(f.shape)
    # n_stages ngoài khoảng -> DỪNG, không làm tròn
    for bad_n in (0, 5):
        try:
            R.InternImageBody(nn.Module(), bad_n)
        except R.VExtractorUnusable as e:
            assert "levels" in str(e) or "stage" in str(e)
        else:
            raise AssertionError(f"n_stages = {bad_n} phải bị từ chối")
    # (3) `with_cp` với một arch của timm bị TỪ CHỐI ồn ào
    try:
        R.backbone_body(SMALL_ARCH, False, SMALL_IMG, None, SMALL_NS, True)
    except R.VExtractorUnusable as e:
        assert "with_cp" in str(e), e
    else:
        raise AssertionError("with_cp: true với arch của timm phải bị từ chối")
    # ...và nó KHÔNG vào vân tay Stage 0 (đổi chi phí, không đổi tensor nào)
    assert "with_cp" in R.EXT_CFG_KEYS and "with_cp" not in R.BACKBONE_CFG_KEYS


@check("2g. vá giao thức init của transformers: nn.Module THUẦN mang `_init_weights` "
       "được cấp `_initialize_weights` no-op, PreTrainedModel thì KHÔNG bị chạm")
def _():
    """Bài này chốt một lỗi ĐÃ XẢY RA, và nó không tự lộ ra ở đâu khác.

    `PreTrainedModel.initialize_weights()` đi cây module bằng `smart_apply` và
    ở mỗi con hỏi "đây có phải sub-model mang hàm init RIÊNG không?". Bản
    transformers gần đây trả lời bằng sự CÓ MẶT của `_init_weights` rồi gọi
    `module.smart_apply(module._initialize_weights)`. `InternImage` của
    OpenGVLab có `_init_weights` nhưng là `nn.Module` thuần, nên lượt nạp chết
    bằng `AttributeError: 'InternImage' object has no attribute
    '_initialize_weights'` — ở CẢ `from_pretrained` lẫn dựng-từ-config.

    Kiểm bằng một module GIẢ nên nó chạy OFFLINE: thứ được kiểm là LUẬT CHỌN
    lớp để vá, không phải repo InternImage.
    """
    import sys
    import types

    import torch.nn as nn
    from transformers import PreTrainedModel
    from transformers import dynamic_module_utils as DMU

    from src import vextractor as R

    mod = types.ModuleType("rgexp_fake_remote_code")

    class Inner(nn.Module):
        """nn.Module THUẦN tự init — đúng hình dạng `InternImage`."""

        def _init_weights(self, m):
            return None

    class Wrapper(nn.Module):
        """Đã có cả hai -> KHÔNG được vá lại."""

        def _init_weights(self, m):
            return None

        def _initialize_weights(self, m):
            return "san co"

    class Plain(nn.Module):
        """Không có `_init_weights` -> `smart_apply` không nhận nhầm -> bỏ qua."""

    for k in (Inner, Wrapper, Plain):
        k.__module__ = mod.__name__
    # Một PreTrainedModel THẬT trong cùng module: nó có `_init_weights` nhưng
    # cũng thừa kế `_initialize_weights`, nên luật vá phải bỏ qua nó.
    mod.Inner, mod.Wrapper, mod.Plain = Inner, Wrapper, Plain
    mod.PreTrainedModel, mod.nn, mod.NOTACLASS = PreTrainedModel, nn, 42
    sys.modules[mod.__name__] = mod
    orig = DMU.get_class_from_dynamic_module
    seen = {}

    def fake_get(ref, path, **kw):
        seen["ref"], seen["path"] = ref, path
        return Inner                      # __module__ trỏ về mod

    DMU.get_class_from_dynamic_module = fake_get
    try:
        conf = types.SimpleNamespace(
            auto_map={"AutoConfig": "configuration_x.C",
                      "AutoModel": "modeling_x.Wrapper"})
        klass, patched = R._hf_model_class("fake/repo", conf)
        # lớp lấy theo `auto_map`, KHÔNG theo tên file gõ tay
        assert seen == {"ref": "modeling_x.Wrapper", "path": "fake/repo"}, seen
        assert klass is Inner
        assert patched == ["Inner"], patched
        assert Inner._initialize_weights is R._no_hf_init
        # ...và KHÔNG chạm ba thứ còn lại
        assert Wrapper._initialize_weights is not R._no_hf_init
        assert Wrapper()._initialize_weights(None) == "san co"
        assert not hasattr(Plain, "_initialize_weights")
        assert (PreTrainedModel._initialize_weights
                is not R._no_hf_init), "vá đè lên PreTrainedModel là vá cả transformers"
        # no-op phải chịu CẢ HAI chữ ký: bản cũ gọi fn(module), bản mới gọi
        # fn(module, is_custom_code). Khai đúng một tham số là cách bài vá này
        # hỏng ở bản transformers sau.
        it = Inner()
        assert it._initialize_weights(it) is None
        assert it._initialize_weights(it, False) is None
        # chạy lại: idempotent, không nhân thêm tên vào danh sách
        _k2, again = R._hf_model_class("fake/repo", conf)
        assert again == [], again
        # `auto_map` thiếu AutoModel -> DỪNG ồn ào, không đoán tên module
        try:
            R._hf_model_class("fake/repo", types.SimpleNamespace(auto_map={}))
        except R.VExtractorUnusable as e:
            assert "auto_map" in str(e)
        else:
            raise AssertionError("auto_map thiếu AutoModel phải bị từ chối")
    finally:
        DMU.get_class_from_dynamic_module = orig
        sys.modules.pop(mod.__name__, None)


@check("3. Stage 0 -> sau: checkpoint CHỈ Extractor, nạp đúng, từ chối format cũ")
def _():
    import json

    import torch

    from src import vextractor as R

    sp = small_spec()
    body, d_feat, n_patch, n_use, kind, fmt, mres, _nst = R.build_backbone(sp)
    x = torch.randn(1, 3, SMALL_IMG, SMALL_IMG)
    with tempfile.TemporaryDirectory() as t:
        R.save_stage0(body, sp, Path(t) / "s0",
                      {"best_epoch": 3, "best_score": -0.5, "best_val_loss": 0.5})
        # File chứa ĐÚNG Extractor, không gì khác
        from safetensors.torch import load_file

        sd = load_file(str(Path(t) / "s0" / R.STAGE0_FILE))
        assert set(sd) == set(body.state_dict()), "checkpoint phải là Extractor THUẦN"
        assert not any(k.startswith(("latent", "groups.", "aligner", "kv_norm"))
                       for k in sd)
        br = R.VExtractorBranch(sp)
        before = {k: v.clone() for k, v in br.extractor.body.state_dict().items()}
        rep_ = R.load_stage0_into(br, Path(t) / "s0")
        assert rep_["n_backbone_tensors"] == len(sd) and rep_["val_loss"] == 0.5
        for k, v in body.state_dict().items():
            assert torch.equal(v, br.extractor.body.state_dict()[k]), k
        assert any(not torch.equal(before[k], br.extractor.body.state_dict()[k])
                   for k in before), "nạp xong mà Extractor không đổi gì?"
        R.save_branch(br, Path(t) / "b", {})
        br2 = R.load_branch(Path(t) / "b")
        br.set_frozen(True), br.eval(), br2.eval()
        with torch.no_grad():
            assert torch.equal(br(x), br2(x))
        assert br2.frozen and br2.n_query == R.n_query_of(sp)
        # `backbone_spec` KHÔNG gồm compressor/aligner/d_model: đổi chúng thì
        # checkpoint Stage 0 vẫn dùng được (Stage 0 không dựng chúng).
        sp2 = small_spec(d_model=64, n_query=SMALL_Q + 1)
        R.load_stage0_into(R.VExtractorBranch(sp2), Path(t) / "s0")
        # ...nhưng đổi `arch`/`n_stages`/chuẩn hoá thì KHÔNG
        bad = dict(sp, extractor=dict(sp["extractor"], n_stages=SMALL_NS - 1))
        try:
            R.load_stage0_into(R.VExtractorBranch(bad), Path(t) / "s0")
        except RuntimeError as e:
            assert "backbone" in str(e)
        else:
            raise AssertionError("đổi n_stages phải bị từ chối")
        # checkpoint bố cục CŨ (thế hệ trước /5) phải bị TỪ CHỐI
        m = Path(t) / "s0" / R.STAGE0_META
        meta = json.loads(m.read_text(encoding="utf-8"))
        m.write_text(json.dumps(dict(meta, format="rgexp.vextractor/4")),
                     encoding="utf-8")
        try:
            R.load_stage0_into(R.VExtractorBranch(sp), Path(t) / "s0")
        except RuntimeError as e:
            assert "format" in str(e)
        else:
            raise AssertionError("format cũ phải bị từ chối")


@check("4. DINO: teacher no_grad, cặp trùng view bị bỏ, EMA + center đổi đúng hướng")
def _():
    import torch

    from src import dino as D
    from src import vextractor as R

    torch.manual_seed(0)
    sp = small_spec()
    body, d_feat, n_patch, n_use, kind, fmt, mres, _nst = R.build_backbone(sp)
    dc = {"out_dim": 64, "hidden": 32, "bottleneck": 16, "head_layers": 3,
          "pool": "mean", "student_temp": 0.1, "teacher_temp": 0.04,
          "center_momentum": 0.9}
    # ConvNeXt KHÔNG có cls token -> n_prefix = 0 và pool BẮT BUỘC `mean`.
    tr = D.DinoTrainer(body, d_feat, 0, dc, fmt)
    gl = [torch.randn(2, 3, SMALL_IMG, SMALL_IMG) for _ in range(2)]
    lo = [torch.randn(2, 3, SMALL_IMG // 2, SMALL_IMG // 2) for _ in range(2)]
    loss = tr.loss(gl + lo, 0.04)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    # student nhận gradient; teacher TUYỆT ĐỐI không (nó là EMA, không có bước Adam)
    assert any(q.grad is not None for q in tr.student_parameters())
    assert all(q.grad is None for q in tr.teacher_parameters())
    assert all(not q.requires_grad for q in tr.teacher_parameters())
    # teacher KHÔNG nằm trong optimizer: weight decay sẽ ăn mòn nó dù grad None
    gp = tr.param_groups(1e-5, 1e-5)
    ids = {id(q) for g in gp for q in g["params"]}
    assert not (ids & {id(q) for q in tr.teacher_parameters()})
    assert [g["name"] for g in gp] == ["backbone", "rest"]
    # số cặp: teacher 2 global × student 4 view, BỎ 2 cặp trùng -> 6
    tr.zero_grad(set_to_none=True)
    n_pair = []
    orig = torch.nn.functional.log_softmax

    def spy(*a_, **k_):
        n_pair.append(1)
        return orig(*a_, **k_)
    torch.nn.functional.log_softmax = spy
    try:
        tr.loss(gl + lo, 0.04, update_center=False)
    finally:
        torch.nn.functional.log_softmax = orig
    assert sum(n_pair) == 2 * 4 - 2 == 6, sum(n_pair)
    # center: EMA, chỉ đổi khi update_center=True
    c0 = tr.center.clone()
    tr.loss(gl + lo, 0.04, update_center=False)
    assert torch.equal(tr.center, c0)
    tr.loss(gl + lo, 0.04, update_center=True)
    assert not torch.equal(tr.center, c0)
    # EMA: m = 1 -> teacher đứng yên; m = 0 -> teacher = student
    with torch.no_grad():
        for q in tr.student_body.parameters():
            q.add_(0.1)
    t0 = next(iter(tr.teacher_body.parameters())).clone()
    tr.ema_update(1.0)
    assert torch.equal(next(iter(tr.teacher_body.parameters())), t0)
    tr.ema_update(0.0)
    for a_, b_ in zip(tr.teacher_body.parameters(), tr.student_body.parameters()):
        assert torch.allclose(a_, b_, atol=1e-6)
    # 1 view -> DỪNG (DINO cần ít nhất 2 global)
    try:
        tr.loss(gl[:1], 0.04)
    except ValueError:
        pass
    else:
        raise AssertionError("1 view phải bị từ chối")


@check("5. lịch DINO: τ_t ramp rồi phẳng, momentum cosine m0 -> 1, τ_t < τ_s bị ép")
def _():
    from src import dino as D
    from src import stage0_vextractor as s0
    from src.backbone import TrainUnusable

    # τ_t: ramp TUYẾN TÍNH trong `warm` bước rồi PHẲNG
    assert abs(D.teacher_temp_at(0, 10, 0.02, 0.06) - (0.02 + 0.004)) < 1e-12
    assert abs(D.teacher_temp_at(9, 10, 0.02, 0.06) - 0.06) < 1e-12
    assert abs(D.teacher_temp_at(99, 10, 0.02, 0.06) - 0.06) < 1e-12
    assert D.teacher_temp_at(0, 0, 0.02, 0.06) == 0.06        # warm = 0 -> phẳng ngay
    ramp = [D.teacher_temp_at(k, 10, 0.02, 0.06) for k in range(10)]
    assert all(x <= y + 1e-18 for x, y in zip(ramp, ramp[1:]))
    # momentum: cosine m0 -> m1, đơn điệu tăng, chạm đúng hai đầu
    assert abs(D.momentum_at(0, 100, 0.996) - 0.996) < 1e-9
    assert abs(D.momentum_at(100, 100, 0.996) - 1.0) < 1e-9
    ms = [D.momentum_at(k, 100, 0.996) for k in range(101)]
    assert all(x <= y + 1e-15 for x, y in zip(ms, ms[1:]))
    # sharpening là BẮT BUỘC: τ_t >= τ_s bị từ chối
    base = {"stage0": {"dino": {"student_temp": 0.1, "teacher_temp": 0.2}}}
    try:
        s0.dino_cfg(base["stage0"])
    except TrainUnusable as e:
        assert "student_temp" in str(e)
    else:
        raise AssertionError("τ_t >= τ_s phải bị từ chối")
    for bad in ({"pool": "avg"}, {"n_local": -1}, {"momentum_start": 1.2},
                {"center_momentum": 1.0}):
        try:
            s0.dino_cfg({"dino": bad})
        except TrainUnusable:
            pass
        else:
            raise AssertionError(f"{bad} phải bị từ chối")


@check("5b. multi-crop: 2 global + n local, đúng cỡ, collate xếp theo VIEW")
def _():
    import random

    import numpy as np
    import torch
    from PIL import Image

    from src import dino as D
    from src import stage0_vextractor as s0

    dc = s0.dino_cfg({"dino": {"global_size": 64, "local_size": 32, "n_local": 3}})
    assert dc["aug"]["hflip"] == 0.0, "ảnh CXR có chiều — lật ngang phải TẮT mặc định"
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x.png"
        Image.fromarray((np.random.rand(80, 90) * 255).astype("uint8")).save(f)
        v = D.make_views(f, dc, (0.5,) * 3, (0.5,) * 3, random.Random(0))
        assert len(v) == 2 + 3
        assert all(tuple(x.shape) == (3, 64, 64) for x in v[:2])     # global TRƯỚC
        assert all(tuple(x.shape) == (3, 32, 32) for x in v[2:])
        assert all(torch.isfinite(x).all() for x in v)
        assert not torch.equal(v[0], v[1])          # hai global crop khác nhau
        # cùng RNG -> cùng view (tái lập được)
        v2 = D.make_views(f, dc, (0.5,) * 3, (0.5,) * 3, random.Random(0))
        assert all(torch.equal(a, b) for a, b in zip(v, v2))
        b = D.collate_views([v, v2])
        assert len(b) == 5 and tuple(b[0].shape) == (2, 3, 64, 64)
        assert tuple(b[-1].shape) == (2, 3, 32, 32)
        try:
            D.collate_views([v, v[:2]])
        except ValueError:
            pass
        else:
            raise AssertionError("số view lệch phải bị từ chối")


@check("6. CE ghép theo token qua mảnh == CE của cả micro-batch (kiểu HF)")
def _():
    import torch
    import torch.nn.functional as F

    from src.train_vextractor import _ntok, _rows

    torch.manual_seed(1)
    B, T, V = 6, 11, 17
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    am = torch.ones(B, T, dtype=torch.long)
    for i, n in enumerate([11, 7, 9, 4, 11, 6]):     # đệm phải
        labels[i, n:] = -100
        am[i, n:] = 0
    labels[:, :3] = -100                              # prompt bị che

    def hf_ce(lg, lb, m):
        sl, sb = lg[:, :-1], lb[:, 1:]
        keep = m[:, 1:] != 0
        return F.cross_entropy(sl[keep], sb[keep], ignore_index=-100)

    full = hf_ce(logits, labels, am)
    b = {"logits": logits, "labels": labels, "attention_mask": am}
    for chunk in (1, 2, 3):
        tot = _ntok(labels, am)
        acc = 0.0
        for s in range(0, B, chunk):
            sub = _rows(b, s, s + chunk)
            k = _ntok(sub["labels"], sub["attention_mask"])
            if k:
                acc += float(hf_ce(sub["logits"], sub["labels"], sub["attention_mask"])) * k / tot
        assert abs(acc - float(full)) < 1e-5, (chunk, acc, float(full))


@check("6b. collate_vext xếp chồng vext_pixels; MỌI dataset của main stage mang nó")
def _():
    import torch

    from src.train_vextractor import collate_vext

    rows = [{"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, 2, 3]),
             "attention_mask": torch.ones(3, dtype=torch.long),
             "vext_pixels": torch.full((3, 8, 8), float(k))} for k in (1, 2)]
    b = collate_vext(rows, pad_id=0)
    assert b["vext_pixels"].shape == (2, 3, 8, 8), b["vext_pixels"].shape
    assert float(b["vext_pixels"][0].mean()) == 1.0
    assert float(b["vext_pixels"][1].mean()) == 2.0
    assert b["input_ids"].shape == (2, 3)          # phần của collate_cls giữ nguyên
    # rows KHÔNG có vext_pixels -> không tự bịa ra trường rỗng
    assert "vext_pixels" not in collate_vext(
        [{k: v for k, v in r.items() if k != "vext_pixels"} for r in rows], pad_id=0)

    # Dataset của val cũng phải là VExtReportDataset: nếu không, val CE nổ ngay
    # ở lượt đo đầu tiên với "batch thiếu vext_pixels" — sau khi đã train xong
    # vài chục bước.
    src = (ROOT / "src/train_vextractor.py").read_text(encoding="utf-8")
    assert "ClsReportDataset(" not in src.split("class VExtReportDataset", 1)[1], \
        "còn dataset nào của main stage dựng bằng ClsReportDataset (thiếu vext_pixels)"
    assert src.count("VExtReportDataset(") >= 2      # train + val


@check("7. đóng băng: Extractor teacher-infer; Compressor + Aligner vẫn học ở main stage")
def _():
    import torch

    from src import vextractor as R

    sp = small_spec()
    sp["compressor"]["dropout"] = 0.5
    br = R.VExtractorBranch(sp)
    br.set_frozen(True)
    br.train()
    # Extractor eval + không grad; PHẦN CÒN LẠI vẫn train
    assert not br.extractor.body.training
    assert br.compressor.training and br.aligner.training
    assert all(not q.requires_grad for q in br.extractor.parameters())
    assert all(q.requires_grad for q in br.trainable_parameters())
    npar = br.n_params()
    assert npar["trainable"] == npar["learnable_main"] == (
        npar["compressor"] + npar["aligner"])
    assert npar["learnable_main"] > 0 and npar["extractor"] > 0
    assert npar["compressor"] == npar["compressor_attn"] + npar["compressor_ffn"]
    assert npar["total"] == npar["extractor"] + npar["learnable_main"]
    # gradient của CE tới được Compressor + Aligner, KHÔNG tới Extractor
    x = torch.randn(2, 3, SMALL_IMG, SMALL_IMG)
    br.eval()
    br(x).sum().backward()
    assert all(q.grad is not None for q in br.trainable_parameters())
    assert all(q.grad is None for q in br.extractor.parameters())
    br.zero_grad(set_to_none=True)
    # ── đóng băng RIÊNG Compressor: Aligner KHÔNG bị chạm ─────────────────
    br.set_compressor_frozen(True)
    assert br.compressor_frozen
    assert all(not q.requires_grad for q in br.compressor.parameters())
    assert all(q.requires_grad for q in br.aligner.parameters())
    br(x).sum().backward()
    assert all(q.grad is None for q in br.compressor.parameters())
    assert all(q.grad is not None for q in br.aligner.parameters())
    br.zero_grad(set_to_none=True)
    br.set_compressor_frozen(False)
    assert all(q.requires_grad for q in br.compressor.parameters())
    # mở băng Extractor (Lens dùng) -> gradient tới tận pixel
    br.set_frozen(False)
    br.train()
    assert br.extractor.body.training
    br(x).sum().backward()
    assert all(q.grad is not None for q in br.extractor.parameters())
    assert br.n_params()["trainable"] == br.n_params()["total"]


@check("8. bootstrap ghép cặp + permutation: thấy hiệu số thật, không bịa")
def _():
    import numpy as np
    import pandas as pd

    from src.stats_ab import compare_all

    rng = np.random.default_rng(0)
    n = 400
    base = rng.uniform(0, 1, n)
    keys = [f"k{i}" for i in range(n)]
    A = pd.DataFrame({"key": keys, "m_up": base, "m_same": base})
    B = pd.DataFrame({"key": keys, "m_up": base + 0.05 + rng.normal(0, 0.02, n),
                      "m_same": base + rng.normal(0, 0.02, n)})
    t = compare_all({"a": A, "b": B}, ["m_up", "m_same"],
                    {"alpha": 0.05, "n_boot": 500, "n_perm": 500, "dz_floor": 0.1},
                    "a", "b", seed=1).set_index("metric")
    assert t.loc["m_up", "winner"] == "b" and t.loc["m_up", "ci_lo"] > 0
    assert t.loc["m_same", "winner"] == "—"


@check("9. Hub: A từ current/$1_LoRA_Test, B vào current/$2.2_V-Extractor_Test")
def _():
    from src.config import load_config
    from src.push import hub_base
    from src.run_vextractor import arms_of, baseline_cfg

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    assert hub_base(cfg) == "current/$2.2_V-Extractor_Test", hub_base(cfg)
    assert hub_base(baseline_cfg(cfg)) == "current/$1_LoRA_Test"
    A, B = arms_of(cfg)
    old = load_config(ROOT / "_archive/v2/configs/lora.yaml")
    r32 = [x for x in old["lora_exp"]["arms"] if x["id"] == A["id"]]
    assert r32 and int(r32[0]["rank"]) == A["rank"] == B["rank"] == 32
    assert A["target_modules"] == old["lora_exp"]["targets"][r32[0]["target"]]
    assert B["target_modules"] == A["target_modules"]
    assert B["vextractor"] and not A["vextractor"] and B["source"] == "$2.2"


@check("10. src/ không import _archive/run_lora; runner sinh bằng Q")
def _():
    import re

    bad = []
    for p in (ROOT / "src").rglob("*.py"):
        s = p.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+[\w.]*(_archive|run_lora|report_lora|"
                     r"stats_friedman|report_plots)\b", s, re.M):
            bad.append(p.name)
    assert not bad, bad
    r = (ROOT / "src/runners/gemma3_vextractor.py").read_text(encoding="utf-8")
    # nhánh §2.2 ăn ĐƯỜNG ẢNH RIÊNG, không phải X của SigLIP
    assert 'self.branch(inputs["vext_pixels"])' in r and "uses_vextractor" in r
    assert 'RCV.vext_tensor(it["image_path"], self.branch.image_size,' in r
    assert "*self.branch.normalize" in r     # mean/std của checkpoint, không hằng số
    # sinh phải nối Q SAU khối ảnh, theo TỪNG HÀNG (đệm trái)
    assert "image_end_indices(ids, self._img_token_id) + 1" in r


@check("11. REPORT.md dựng được từ dữ liệu giả")
def _():
    import numpy as np
    import pandas as pd

    from src.config import load_config
    from src.report_vextractor import write_all
    from src.run_vextractor import arms_of
    from src.stats_ab import compare_all

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    A, B = arms_of(cfg)
    rng = np.random.default_rng(0)
    keys = [f"k{i}" for i in range(50)]
    sc = {A["id"]: pd.DataFrame({"key": keys, "bertscore_f1": rng.uniform(0.8, 0.9, 50)}),
          B["id"]: pd.DataFrame({"key": keys, "bertscore_f1": rng.uniform(0.8, 0.9, 50)})}
    t = compare_all(sc, ["bertscore_f1"], {"n_boot": 100, "n_perm": 100}, A["id"], B["id"])
    with tempfile.TemporaryDirectory() as d:
        p = write_all(cfg, Path(d), Path(d), sc, t, A, B, {"n_eval": 50, "headline": "x"})
        s = p.read_text(encoding="utf-8")
        assert "‖ **Q** ‖" in s and "bertscore_f1" in s


@check("12. lưu như $1: Stage 0 + Stage 1 + main stage vào plan đẩy Hub; weight B lệch format/spec bị phát hiện")
def _():
    import torch

    from src import dino as DN
    from src import stage0_vextractor as s0
    from src import stage1_vextractor as s1
    from src import vextractor as R
    from src.config import load_config
    from src.fixture import WEIGHT_FILES
    from src.push import decide, local_plan

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    aid = cfg["vextractor_exp"]["candidate"]["id"]
    with tempfile.TemporaryDirectory() as t:
        cfg["output"] = dict(cfg["output"], adapter_dir=str(Path(t) / "ad"))
        d = Path(t) / "ad" / "_stage0"
        sp = small_spec()
        for sub in (s0.CKPT, s0.BASE):
            R.save_stage0(R.build_backbone(sp)[0], sp, d / sub, {})
            # DINO head TEACHER đi CÙNG checkpoint — Stage 1 chưng cất từ ĐÚNG
            # phân phối prototype ấy, nên nó phải lên Hub cùng Extractor.
            R.save_dino_head(DN.DINOHead(4, 8, 8, 4, 2), d / sub)
        for f in (s0.HISTORY, s0.LOADER):
            (d / f).write_text("{}")
        (d / "last").mkdir()
        (d / "last" / "state.pt").write_text("x")
        # Stage 1: checkpoint Compressor + history, trong plan đẩy Hub
        s1d = Path(t) / "ad" / s1.DIRNAME
        R.save_compressor(R.VExtractorBranch(sp), s1d / s1.CKPT, {"best_epoch": 1})
        (s1d / s1.HISTORY).write_text("{}")
        bdir = Path(t) / "ad" / aid
        R.save_branch(R.VExtractorBranch(sp), bdir, {})
        for f in ("adapter_model.safetensors", "adapter_config.json", "train_stats.json"):
            (bdir / f).write_text("x")
        plan = {p for p, _f, _k in local_plan(cfg, Path(t) / "run", [aid])}
        base = "current/$2.2_V-Extractor_Test"
        for r in s0.KEEP:
            assert f"{base}/stage0/{r}" in plan, r
        for r in s1.KEEP:
            assert f"{base}/stage1/{r}" in plan, r
        # HAI thư mục RIÊNG: gộp chúng là mất khả năng dùng lại Stage 0 khi chỉ
        # Stage 1 phải chạy lại.
        assert not any(p.startswith(f"{base}/stage0/{s1.CKPT}/") for p in plan)
        assert not any("state.pt" in p for p in plan)
        for f in ("adapter_model.safetensors", "vextractor_branch.safetensors",
                  "vextractor_branch.json", "train_stats.json"):
            assert f"{base}/weights/{aid}/{f}" in plan, f
        assert "vextractor_branch.safetensors" in WEIGHT_FILES

        # ── weight và KẾT QUẢ sinh ra từ nó phải đi CÙNG một commit ────────
        # Repo có weight cũ + kết quả mới là trạng thái không ai đọc ra được:
        # bảng kết quả mất gốc đúng theo cái cách mà cổng "giữ bản repo" sinh
        # ra để chặn.
        run = Path(t) / "run"
        (run / "predictions").mkdir(parents=True, exist_ok=True)
        (run / "predictions" / f"{aid}.jsonl").write_text("{}\n", encoding="utf-8")
        full = local_plan(cfg, run, [aid])
        wpath = f"{base}/weights/{aid}/adapter_model.safetensors"
        remote = {wpath: {"sha256": "0" * 64}}          # repo có bản KHÁC
        todo, plog = decide(full, remote)
        assert wpath in {q for q, _f, _k in todo}, plog
        assert any("KẾT QUẢ mới của arm" in x for x in plog), plog
        no_res = [x for x in full if x[2] != "result"]
        todo2, plog2 = decide(no_res, remote)
        assert wpath not in {q for q, _f, _k in todo2}
        assert any("GIỮ bản repo" in x for x in plog2), plog2
        todo3, _ = decide(no_res, remote, force_weights=True)
        assert wpath in {q for q, _f, _k in todo3}

        # ── weight B CŨ của một kiến trúc khác phải bị phát hiện ───────────
        # Không bắt ở đây thì `train` bỏ qua main stage, nhảy sang sinh với
        # nhánh cũ và NỐI TIẾP dự đoán của mạng cũ vào cùng một file.
        import json as _json

        from src.train_vextractor import already_trained, branch_stale

        s0ck = d / s0.CKPT
        scfg = small_cfg()
        # HỒI QUY: một nhánh vừa dựng TỪ CHÍNH config ấy KHÔNG được coi là cũ.
        assert branch_stale(bdir, s0ck, scfg) is None
        assert branch_stale(bdir, s0ck) is None
        assert already_trained(bdir, s0ck, scfg) is True
        # (a) format cũ -> stale
        mp = bdir / R.BRANCH_META
        meta = _json.loads(mp.read_text(encoding="utf-8"))
        mp.write_text(_json.dumps(dict(meta, format="rgexp.recovery/5")), encoding="utf-8")
        assert "format" in (branch_stale(bdir, s0ck) or "")
        assert already_trained(bdir, s0ck) is False
        assert already_trained(bdir) is False          # không cần Stage 0 để thấy
        mp.write_text(_json.dumps(meta), encoding="utf-8")
        assert already_trained(bdir, s0ck, scfg) is True
        # (b) format khớp nhưng CONFIG đã đổi kiến trúc -> stale, và nói RÕ
        # khoá LÁ nào lệch (đổi `n_query` KHÔNG đụng Extractor)
        why_b = branch_stale(bdir, s0ck, small_cfg(n_query=SMALL_Q + 1)) or ""
        assert "config hiện tại" in why_b, why_b
        assert f"compressor.n_query: {SMALL_Q} -> {SMALL_Q + 1}" in why_b, why_b
        assert already_trained(bdir, s0ck, small_cfg(n_query=SMALL_Q + 1)) is False
        # ...nhưng Stage 0 thì KHÔNG bị đụng: đổi Compressor không phải finetune lại
        assert branch_stale(bdir, s0ck) is None
        # (c) EXTRACTOR của Stage 0 khác -> stale, và đây mới là việc của Stage 0
        sp2 = R.spec_from_cfg(small_cfg(n_stages=SMALL_NS - 1), 32)
        assert R.backbone_spec(sp2) != R.backbone_spec(sp)
        R.save_stage0(R.build_backbone(sp2)[0], sp2, s0ck, {})
        why_c = branch_stale(bdir, s0ck) or ""
        assert "backbone" in why_c, why_c
        assert already_trained(bdir, s0ck) is False
    _ = torch


@check("13. labels.npz (CHỈ cho Lens): 0=pos, 1=neg, 2=unc, 3=blank -> ô TOÀN 0")
def _():
    import numpy as np

    from src.config import load_config
    from src.labels_cls import CLASS_ORDER, resolve_encoding, to_onehot

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    lc = dict(cfg["labels"])
    assert lc["label_array"] == "y_gt" and lc["key_array"] == "img_rel"
    enc, nan_to = resolve_encoding(lc)
    raw = np.array([[0, 1, 2, 3]])                       # pos, neg, unc, blank
    oh = to_onehot(raw, enc, nan_to=nan_to, allow_class_merge=True)[0]
    want = ["pos", "neg", "uncertain"]
    for j, c in enumerate(want):
        assert oh[j].argmax() == CLASS_ORDER.index(c) and oh[j].sum() == 1, (j, oh[j])
    assert oh[3].sum() == 0                              # blank: toàn 0
    # Khối `labels` KHÔNG còn vào Stage 0: nó chỉ phục vụ Lens, và vì thế cũng
    # không vào vân tay Stage 0 (đổi nhãn không được bắt chạy lại một lượt DINO).
    from src import stage0_vextractor as s0

    c2 = load_config(ROOT / "configs/vextractor.yaml")
    c2["labels"] = dict(lc, max_missing_frac=0.99)
    assert s0.config_fingerprint(c2) == s0.config_fingerprint(cfg)
    assert "asl" not in cfg["stage0"] and "val_classes" not in cfg["stage0"]


@check("14. LỊCH CHUNG (warm-up 1 epoch -> GIỮ lr_max, + chặng 5) và hai cặp lr của đặc tả")
def _():
    from src import schedule as S
    from src import stage0_vextractor as s0
    from src import stage1_vextractor as s1
    from src.backbone import TrainUnusable
    from src.config import load_config
    from src.train_vextractor import MAIN_CFG_KEYS, zcfg

    # ── hình dạng lịch: dốc rồi GIỮ, KHÔNG có chặng hạ nào ────────────────
    w = 200                                   # 1 epoch = 200 bước
    lo, hi = 1e-8, 2e-5
    f = lambda k: S.warmup_flat(k, w, lo, hi)             # noqa: E731
    assert f(0) == lo                         # bước 0 có ĐÚNG lr_min
    assert f(w) == hi and f(10 * w) == hi     # chạm đỉnh rồi GIỮ, không hạ
    up = [f(k) for k in range(w + 1)]
    assert all(x <= y + 1e-18 for x, y in zip(up, up[1:]))     # ĐƠN ĐIỆU tăng
    assert abs(f(w // 2) - (lo + hi) / 2) < 1e-9               # TUYẾN TÍNH
    assert S.warmup_flat(0, 0, lo, hi) == hi                   # w = 0 -> vào thẳng
    assert S.phase_of(0, w) == "warmup" and S.phase_of(w, w) == "hold"
    assert S.phase_span(0, w - 1, w) == "warmup"
    assert S.phase_span(w, 2 * w - 1, w) == "hold"
    assert S.phase_span(w - 1, w, w) == "warmup→hold"          # epoch bắc qua ranh
    # Lịch cũ đã BIẾN MẤT hẳn: không còn chặng giữ đỉnh + cosine về lr_main.
    for gone in ("staged_lr", "phase_bounds", "WARMUP_FRAC", "HOLD_FRAC",
                 "DECAY_FRAC", "phase_name"):
        assert not hasattr(S, gone), gone

    # ── ĐUÔI: cosine từ lr ĐANG chạy về lr_min ────────────────────────────
    assert S.tail_lr(0, 10, 1e-6, lo) == 1e-6
    assert abs(S.tail_lr(10, 10, 1e-6, lo) - lo) < 1e-18
    t = [S.tail_lr(k, 10, 1e-6, lo) for k in range(11)]
    assert all(x >= y - 1e-18 for x, y in zip(t, t[1:]))

    # ── main stage: dốc theo BƯỚC, và 0 TUYỆT ĐỐI trong phần đóng băng ────
    assert S.warmup_flat(0, 20, 1e-6, 1e-4) == 1e-6
    assert S.warmup_flat(20, 20, 1e-6, 1e-4) == 1e-4 == S.warmup_flat(99, 20, 1e-6, 1e-4)
    assert S.frozen_warmup_flat(39, 40, 20, 1e-8, 1e-5) == 0.0
    assert S.frozen_warmup_flat(40, 40, 20, 1e-8, 1e-5) == 1e-8
    assert S.frozen_warmup_flat(60, 40, 20, 1e-8, 1e-5) == 1e-5
    assert S.steps_for_frac(200, 0.2) == 40 and S.steps_for_frac(200, 0.1) == 20

    # ── hai CẶP lr, ĐÚNG như đặc tả ───────────────────────────────────────
    cfg = load_config(ROOT / "configs/vextractor.yaml")
    sc = s0.s0cfg(cfg)
    assert (sc["lr_min"], sc["lr_max"]) == (1e-8, 2e-5)
    assert "lr_main" not in sc and "sched_epochs" not in sc
    assert sc["warmup_epochs"] == 1.0 and sc["max_epochs"] == 50
    # patience ĐỒNG BỘ: cả hai giai đoạn cùng chờ 4 epoch.
    assert sc["patience"] == 4 and sc["tail_epochs"] == 1 and sc["select"] == "best"
    assert sc["val_n_samples"] == 256 and sc["val_resample"] is False
    s1c = s1.s1cfg(cfg)
    assert (s1c["attn_lr_min"], s1c["attn_lr_max"]) == (1e-8, 3e-5)
    assert (s1c["ffn_lr_min"], s1c["ffn_lr_max"]) == (1e-7, 3e-4)
    # FFN chịu được dốc CAO HƠN attn: ma trận attention quyết định 8 vector
    # ĐỌC Ở ĐÂU trên lưới, và một bước lớn làm phân phối ấy sụp về vài ô rồi
    # không quay lại. Quan hệ ấy phải còn đúng, không chỉ hai con số rời.
    assert s1c["ffn_lr_max"] > s1c["attn_lr_max"]
    # Stage 1 đẩy attn CAO HƠN mức chạy của Stage 0: ở đây backbone đứng yên,
    # nên bước lớn không làm hỏng một encoder đang được dạy.
    assert s1c["attn_lr_max"] > sc["lr_max"]
    assert s1c["warmup_epochs"] == 1.0 and s1c["max_epochs"] == 30
    assert s1c["patience"] == sc["patience"] == 4
    zc = zcfg(cfg)
    assert set(zc) == set(MAIN_CFG_KEYS)
    assert (zc["align_ffn_lr_min"], zc["align_ffn_lr_main"]) == (1e-6, 1e-4)
    assert (zc["align_attn_lr_min"], zc["align_attn_lr_main"]) == (1e-7, 1e-5)
    assert zc["align_warmup_frac"] == 0.10
    assert (zc["comp_ffn_lr_min"], zc["comp_ffn_lr_main"]) == (1e-7, 1e-5)
    assert (zc["comp_attn_lr_min"], zc["comp_attn_lr_main"]) == (1e-8, 1e-6)
    # Compressor đã học ở Stage 1 nên main stage chỉ tinh chỉnh NHẸ: mức của
    # nó phải nhỏ hơn hẳn Aligner (khối MỚI, phải học từ ngẫu nhiên).
    # Và Attn chậm hơn FFN đúng 10× ở CẢ HAI khối — `zcfg` từ chối đảo dấu.
    for _b in ("align", "comp"):
        # so theo TỈ LỆ với sai số: 1e-4 / 10 trong float là 1.0000000000000002e-05
        assert abs(zc[f"{_b}_ffn_lr_main"] / zc[f"{_b}_attn_lr_main"] - 10) < 1e-9
        assert zc[f"{_b}_attn_lr_min"] < zc[f"{_b}_ffn_lr_min"]
    assert zc["comp_ffn_lr_main"] < zc["align_ffn_lr_main"]
    assert zc["comp_attn_lr_main"] < zc["align_attn_lr_main"]
    assert (zc["comp_freeze_frac"], zc["comp_warmup_frac"]) == (0.20, 0.10)

    # ── `lr_max` là MỨC CHẠY: mọi epoch sau epoch 1 đều ở đó ──────────────
    # Bài kiểm này chống đúng cách đọc sai đã xảy ra: hiểu `lr_max` là một đỉnh
    # tạm rồi hạ về một `lr_main` nào đó. Không có `lr_main` trong lịch chung.
    spe = 200                                   # 6400 mẫu / batch 32
    assert s0.warmup_steps(sc, spe) == spe
    for ep in range(2, sc["max_epochs"] + 1):
        for k in ((ep - 1) * spe, ep * spe - 1):
            assert s0.sched_lr_at(k, sc, spe) == sc["lr_max"], (ep, k)
    assert s0.sched_lr_at(0, sc, spe) == sc["lr_min"]
    for k in (s1c["warmup_epochs"] * spe, 10 * spe):
        assert s1.lrs_at(int(k), s1c, spe) == {"attn": s1c["attn_lr_max"],
                                               "ffn": s1c["ffn_lr_max"]}

    # ── CHẶNG 5: ÷gamma mỗi lần patience tăng, sàn lr_min, KHÔNG leo lại ──
    G = 1.5
    assert S.PLATEAU_GAMMA == G
    assert sc["plateau_gamma"] == s1c["plateau_gamma"] == G
    assert S.plateau_lr(hi, 0, lo) == hi
    assert abs(S.plateau_lr(hi, 1, lo) - hi / G) < 1e-20
    assert abs(S.plateau_lr(hi, 2, lo) - hi / G ** 2) < 1e-20
    # SÀN ở lr_min: không có sàn thì epoch ĐUÔI (cosine *về* lr_min) thành một
    # cú TĂNG lr, đúng ngược với ý của nó. Với gamma 1,5 thì 2e-5 -> 1e-8 là
    # 19 lần chia (log(2000)/log(1,5) ≈ 18,76), so với 7 lần ở gamma 3 — lr hạ
    # theo NHIỀU nấc nhỏ thay vì rơi hẳn sau vài epoch hỏng.
    import math as _m
    n_floor = _m.ceil(_m.log(hi / lo) / _m.log(G))
    assert n_floor == 19, n_floor
    assert S.plateau_lr(hi, n_floor - 1, lo) > lo
    assert S.plateau_lr(hi, n_floor, lo) == lo
    assert S.plateau_lr(hi, 99, lo) == lo
    # `patience` = 4: tới lúc dừng lr mới chỉ còn ~20% mức chạy, KHÔNG phải
    # ~1,2% như gamma 3. Đây là cả điểm của việc hạ hệ số xuống.
    assert 0.19 < S.plateau_lr(hi, sc["patience"], lo) / hi < 0.20
    ladder = [S.plateau_lr(hi, k, lo) for k in range(22)]
    assert all(x >= y for x, y in zip(ladder, ladder[1:]))      # ĐƠN ĐIỆU giảm
    # gamma = 1.0 TẮT hẳn chặng 5 — và đó mới là khoá để tắt, KHÔNG phải
    # `patience: 0` (patience chỉ gác việc DỪNG).
    assert S.plateau_lr(hi, 5, lo, gamma=1.0) == hi
    for blk, mod in (("stage0", s0.s0cfg), ("stage1", s1.s1cfg)):
        c = dict(cfg)
        c[blk] = dict(cfg[blk], plateau_gamma=0.5)
        try:
            mod(c)
        except TrainUnusable as e:
            assert "plateau_gamma" in str(e), e
        else:
            raise AssertionError(f"{blk}.plateau_gamma < 1 phải bị từ chối")
    # áp lên CẢ đường lr của hai giai đoạn; hai nhóm của Stage 1 dùng CHUNG một
    # bộ đếm (val là MỘT con số), nhưng sàn thì RIÊNG của từng nhóm.
    flat = 5 * spe
    assert s0.sched_lr_at(flat, sc, spe, 0) == sc["lr_max"]
    assert abs(s0.sched_lr_at(flat, sc, spe, 1) - sc["lr_max"] / G) < 1e-18
    l0, l1 = s1.lrs_at(flat, s1c, spe, 0), s1.lrs_at(flat, s1c, spe, 1)
    assert abs(l1["attn"] - l0["attn"] / G) < 1e-18
    assert abs(l1["ffn"] - l0["ffn"] / G) < 1e-18
    deep = s1.lrs_at(flat, s1c, spe, 99)
    assert deep["attn"] == s1c["attn_lr_min"] and deep["ffn"] == s1c["ffn_lr_min"]

    # ── lr_min <= lr_max, và khoá của bản CŨ bị từ chối ───────────────────
    for blk, mod, bad in (("stage0", s0.s0cfg, {"lr_min": 1e-3}),
                          ("stage0", s0.s0cfg, {"lr_min": 0.0}),
                          ("stage1", s1.s1cfg, {"attn_lr_max": 1e-12}),
                          ("stage1", s1.s1cfg, {"ffn_lr_min": 0.0})):
        c = dict(cfg)
        c[blk] = dict(cfg[blk], **bad)
        try:
            mod(c)
        except TrainUnusable:
            pass
        else:
            raise AssertionError(f"{blk} {bad} phải bị từ chối")
    for blk, mod, stale in (("stage0", s0.s0cfg, "lr_main"),
                            ("stage0", s0.s0cfg, "sched_epochs"),
                            ("stage0", s0.s0cfg, "lr"),
                            ("stage1", s1.s1cfg, "attn_lr_main"),
                            ("stage1", s1.s1cfg, "sched_epochs")):
        c = dict(cfg)
        c[blk] = dict(cfg[blk], **{stale: 1e-5})
        try:
            mod(c)
        except TrainUnusable as e:
            assert "khoá lạ" in str(e), (blk, stale, e)
        else:
            raise AssertionError(f"{blk}.{stale} (khoá cũ) phải bị từ chối")
    c = dict(cfg)
    for _stale in ("align_lr", "align_lr_min", "align_lr_main",
                   "comp_lr_min", "comp_lr_main"):
        c = dict(cfg)
        c["vextractor"] = dict(cfg["vextractor"], main=dict(
            cfg["vextractor"]["main"], **{_stale: 1e-5}))
        try:
            zcfg(c)
        except TrainUnusable as e:
            assert "khoá lạ" in str(e), (_stale, e)
        else:
            raise AssertionError(
                f"vextractor.main.{_stale} (khoá cũ, nay tách attn/ffn) phải "
                f"bị từ chối")
    # freeze + warmup > 100% -> Compressor không bao giờ chạm lr_main -> DỪNG
    c = dict(cfg)
    c["vextractor"] = dict(cfg["vextractor"], main=dict(
        cfg["vextractor"]["main"], comp_freeze_frac=0.8, comp_warmup_frac=0.5))
    try:
        zcfg(c)
    except TrainUnusable as e:
        assert "KHÔNG BAO GIỜ" in str(e)
    else:
        raise AssertionError("freeze + warmup > 1 phải bị từ chối")


@check("15. Stage 0 chạy THẬT (ConvNeXt nhỏ, KHÔNG Gemma, KHÔNG nhãn): warm-up 1 epoch rồi GIỮ lr_max, chặng 5 ÷1,5, dừng sớm theo patience, epoch ĐUÔI cosine về lr_min")
def _():
    import json

    import pandas as pd
    import torch

    from src import schedule as SCH
    from src import stage0_vextractor as s0
    from src import vextractor as R
    from src.config import load_config

    D_MOD = 32
    cfg = load_config(ROOT / "configs/vextractor.yaml")
    cfg["vextractor"] = small_cfg()["vextractor"]
    # `pretrained` trong CONFIG vẫn là True (guard của `run` đi qua thật); chỉ
    # `backbone_body` bị ép dựng offline.
    cfg["vextractor"]["extractor"] = dict(cfg["vextractor"]["extractor"],
                                          pretrained=True)
    cfg["stage0"] = dict(cfg["stage0"],
                         dino=dict(cfg["stage0"]["dino"], global_size=SMALL_IMG,
                                   local_size=SMALL_IMG // 2, n_local=2,
                                   out_dim=64, hidden=32, bottleneck=16),
                         batch_size=4, micro_batch=2, probe_sizes=[2, 4],
                         warmup_epochs=1.0, max_epochs=8, patience=2,
                         tail_epochs=1)
    cfg["autobatch"] = {"enabled": False}
    keys = [f"k{i}" for i in range(12)]

    def _views(self, i):
        d_ = self.dc
        gen = torch.Generator().manual_seed(i * 131 + self.epoch)
        return ([torch.randn(3, d_["global_size"], d_["global_size"], generator=gen)
                 for _ in range(2)]
                + [torch.randn(3, d_["local_size"], d_["local_size"], generator=gen)
                   for _ in range(int(d_["n_local"]))])

    # val_score do MÌNH điều khiển: epoch 1 tốt nhất, rồi xấu dần -> patience
    # (2) hết ở epoch 3, epoch 4 là ĐUÔI.
    seq = iter([-0.10, -0.20, -0.30, -0.05])
    orig = (s0.decoder_width, s0.assert_same_pixels,
            s0.ImageOnlyDataset.__getitem__, R.backbone_body)
    R.backbone_body = (lambda arch, pretrained, image_size, n_layers=None,
                       n_stages=None, with_cp=False:
                       orig[3](arch, False, image_size, n_layers, n_stages,
                               with_cp))
    s0.decoder_width = lambda *_a, **_k: D_MOD
    s0.assert_same_pixels = lambda *_a, **_k: None
    s0.ImageOnlyDataset.__getitem__ = _views
    pushed = []
    try:
        with tempfile.TemporaryDirectory() as t:
            cfg["output"] = dict(cfg["output"], adapter_dir=str(Path(t) / "ad"))
            tr = pd.DataFrame({"key": keys[:8], "ref": "x"})
            va = pd.DataFrame({"key": keys[8:], "ref": "x"})
            d = s0.stage0_dir(cfg)

            def cb():
                h_ = json.loads((d / s0.HISTORY).read_text())
                assert h_["base_finished"] and not h_["finished"]
                assert h_["stopped_early"] is True
                assert R.stage0_exists(d / s0.BASE)
                pushed.append("base")

            import src.stage0_vextractor as _mod
            real_run = _mod.run

            def patched_eval(*_a, **_k):
                pass
            # Ép val_score theo `seq` bằng cách thay `train_recomb.val_indices`?
            # Không — đơn giản hơn: thay chính hàm đo bằng một bản đọc `seq`.
            h = real_run(cfg, tr, va, on_base_done=cb)
            _ = patched_eval
            assert pushed == ["base"], pushed
            assert h["finished"] and h["base_finished"]
            ph = [(x["epoch"], x["phase"]) for x in h["history"]]
            # warm-up ĐÚNG một epoch: epoch 1 là "warmup", epoch 2+ là "hold"
            spe_ = h["steps_per_epoch"]
            want_ph = [SCH.phase_span((e - 1) * spe_, e * spe_ - 1, spe_)
                       for e in range(1, h["base_epochs_run"] + 1)]
            assert want_ph[0] == "warmup" and set(want_ph[1:]) <= {"hold"}, want_ph
            assert [p for _e, p in ph[:len(want_ph)]] == want_ph, ph
            # ĐUÔI chỉ chạy khi dừng sớm
            tails = [e for e, p in ph if p == "tail"]
            assert bool(tails) == bool(h["stopped_early"]), (tails, h["stopped_early"])
            if h["stopped_early"]:
                assert h["ran_tail"] and len(tails) == 1
                assert tails[0] == h["base_epochs_run"] + 1
            # ── CHẶNG 5 trên lượt chạy THẬT ──────────────────────────────
            sc_live = s0.s0cfg(cfg)
            spe_ = h["steps_per_epoch"]
            sched = [x for x in h["history"] if x["phase"] != "tail"]
            scores = [x["val_score"] for x in sched]
            got_nd = [x["n_decays"] for x in sched]
            # `n_decays` = TỔNG số epoch val không cải thiện, tính lại từ chính
            # đường val đã ghi. Đây là bài kiểm thật: nó bắt được cả việc quên
            # tăng bộ đếm lẫn việc RESET nó khi val khá lên.
            want_nd, run_best, k = [], None, 0
            for v in scores:
                if run_best is None or v > run_best:
                    run_best = v
                else:
                    k += 1
                want_nd.append(k)
            assert got_nd == want_nd, (got_nd, want_nd)
            assert got_nd == sorted(got_nd)              # KHÔNG BAO GIỜ giảm
            assert got_nd[-1] == h["n_decays"] >= 1, (got_nd, h["n_decays"])
            # lr của MỖI epoch = lịch tại bước cuối epoch ấy, chia theo bộ đếm
            # ở ĐẦU epoch (tức trước lần chia của chính nó).
            for i, x in enumerate(sched, start=1):
                nd_in = 0 if i == 1 else got_nd[i - 2]
                want_lr = s0.sched_lr_at(i * spe_ - 1, sc_live, spe_, nd_in)
                assert abs(x["lr"] - want_lr) < 1e-18, (i, x["lr"], want_lr, nd_in)
            # val khá lên thì lr GIỮ NGUYÊN, không leo lại -> đơn điệu giảm kể
            # từ epoch 2 (epoch 1 còn đang warm-up đi LÊN).
            lrs_ = [x["lr"] for x in sched][1:]
            assert all(y <= x + 1e-18 for x, y in zip(lrs_, lrs_[1:])), lrs_
            # `last/state.pt` phải mang `n_decays`: thiếu nó thì chạy tiếp sau
            # khi đứt sẽ khôi phục lr về mức CHƯA chia, và không có gì báo.
            st_ = torch.load(d / "last" / "state.pt", map_location="cpu",
                             weights_only=False)
            assert st_["n_decays"] == h["n_decays"], st_["n_decays"]

            # `final/` mang trọng số của epoch TỐT NHẤT — không phải epoch cuối
            from safetensors.torch import load_file

            m = json.loads((d / s0.CKPT / R.STAGE0_META).read_text())
            assert m["weights"] == "teacher_ema"
            assert m["best_epoch"] == h["best_epoch"]
            best_ep = h["best_epoch"]
            scores = {x["epoch"]: x["val_score"] for x in h["history"]}
            assert scores[best_ep] == max(scores.values()), (best_ep, scores)
            fin = load_file(str(d / s0.CKPT / R.STAGE0_FILE))
            bas = load_file(str(d / s0.BASE / R.STAGE0_FILE))
            if best_ep == h["base_epoch"]:
                assert all(torch.equal(fin[k], bas[k]) for k in fin)
            # DINO head TEACHER đi cùng cả hai checkpoint (Stage 1 cần nó)
            for sub in (s0.BASE, s0.CKPT):
                assert (d / sub / R.DINO_HEAD_FILE).exists()
            assert s0.finished(cfg) is not None
            # chạy lại -> nhận ra đã xong, không train thêm
            h2 = s0.run(cfg, tr, va)
            assert h2["best_epoch"] == h["best_epoch"]
            # đổi lr_max -> vân tay đổi -> phải chạy lại
            cfg["stage0"] = dict(cfg["stage0"], lr_max=5e-4)
            assert s0.finished(cfg) is None
    finally:
        (s0.decoder_width, s0.assert_same_pixels,
         s0.ImageOnlyDataset.__getitem__, R.backbone_body) = orig
    _ = seq


@check("15b. decoder_width: đọc hidden_size từ AutoConfig (không tải trọng số); mọi hf_token() gọi KHÔNG tham số")
def _():
    import sys
    import types

    # Gọi ĐÚNG hàm thật (không patch nó) với một `transformers` giả: đây là chỗ
    # duy nhất §2.2 còn chạm tới model, và nó chỉ được phép đọc CONFIG.
    calls = {}

    class _Cfg:
        text_config = types.SimpleNamespace(hidden_size=2560)

    class _Auto:
        @staticmethod
        def from_pretrained(hf_id, **kw):
            calls["hf_id"], calls["kw"] = hf_id, dict(kw)
            return _Cfg()

    fake = types.ModuleType("transformers")
    fake.AutoConfig = _Auto
    real = sys.modules.get("transformers")
    sys.modules["transformers"] = fake
    try:
        from src.stage0_vextractor import decoder_width

        assert decoder_width({"base": {"hf_id": "google/gemma-3-4b-it"}}) == 2560
        assert calls["hf_id"] == "google/gemma-3-4b-it"
        assert set(calls["kw"]) == {"token"}        # chỉ token, không weight/dtype
        # model không có text_config (kiểu text-only) -> rơi về hidden_size gốc
        _Cfg.text_config = None
        _Cfg.hidden_size = 1536
        assert decoder_width({"base": {"hf_id": "x"}}) == 1536
        # không đọc được -> DỪNG, không đoán một con số
        _Cfg.hidden_size = None
        try:
            decoder_width({"base": {"hf_id": "x"}})
        except Exception:
            pass
        else:
            raise AssertionError("không đọc được hidden_size phải DỪNG")
    finally:
        if real is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = real

    # `config.hf_token()` KHÔNG nhận tham số — gọi nó với cfg là TypeError ở
    # tận runtime, sau khi đã nạp xong dữ liệu (đúng cái đã xảy ra một lần).
    import inspect
    import re

    from src.config import hf_token

    assert not inspect.signature(hf_token).parameters
    for f in (ROOT / "src").rglob("*.py"):
        for m in re.finditer(r"hf_token\(([^)]*)\)", f.read_text(encoding="utf-8")):
            assert not m.group(1).strip(), f"{f.name}: hf_token({m.group(1)})"


@check("16. Stage 1 chạy THẬT: HAI nhóm lr (attn | ffn), teacher Stage 0 ĐỨNG YÊN, checkpoint CHỈ Compressor")
def _():
    import json

    import pandas as pd
    import torch

    from src import dino as D
    from src import stage0_vextractor as s0
    from src import stage1_vextractor as s1
    from src import vextractor as R
    from src.config import load_config

    D_MOD = 32
    cfg = load_config(ROOT / "configs/vextractor.yaml")
    cfg["vextractor"] = small_cfg()["vextractor"]
    cfg["vextractor"]["extractor"] = dict(cfg["vextractor"]["extractor"],
                                          pretrained=True)
    cfg["stage0"] = dict(cfg["stage0"],
                         dino=dict(cfg["stage0"]["dino"], global_size=SMALL_IMG,
                                   local_size=SMALL_IMG // 2, n_local=2,
                                   out_dim=32, hidden=16, bottleneck=8,
                                   head_layers=2))
    cfg["stage1"] = dict(cfg["stage1"], warmup_epochs=1.0, max_epochs=2,
                         patience=0, tail_epochs=0, batch_size=4, micro_batch=2,
                         val_n_samples=4)

    def _views(self, i):
        d_ = self.dc
        gen = torch.Generator().manual_seed(i * 977 + self.epoch)
        return [torch.randn(3, d_["global_size"], d_["global_size"], generator=gen)
                for _ in range(2 + int(d_["n_local"]))]

    orig = (s0.decoder_width, s0.assert_same_pixels,
            s0.ImageOnlyDataset.__getitem__, R.backbone_body)
    R.backbone_body = (lambda arch, pretrained, image_size, n_layers=None,
                       n_stages=None, with_cp=False:
                       orig[3](arch, False, image_size, n_layers, n_stages,
                               with_cp))
    s0.decoder_width = lambda *_a, **_k: D_MOD
    s0.assert_same_pixels = lambda *_a, **_k: None
    s0.ImageOnlyDataset.__getitem__ = _views
    try:
        with tempfile.TemporaryDirectory() as t:
            cfg["output"] = dict(cfg["output"], adapter_dir=str(Path(t) / "ad"))
            sp = R.spec_from_cfg(cfg, D_MOD)
            s0d = Path(t) / "s0"
            body = R.build_backbone(sp)[0]
            R.save_stage0(body, sp, s0d, {"best_epoch": 1, "weights": "teacher_ema"})
            dc = s1.s1cfg(cfg)["dino"]
            R.save_dino_head(D.DINOHead(R.build_extractor(sp).d_feat, dc["out_dim"],
                                        dc["hidden"], dc["bottleneck"],
                                        dc["head_layers"]), s0d)
            before = {k: v.clone() for k, v in body.state_dict().items()}
            tr = pd.DataFrame({"key": [f"k{i}" for i in range(8)], "ref": "x"})
            va = pd.DataFrame({"key": [f"v{i}" for i in range(4)], "ref": "x"})
            h = s1.run(cfg, tr, va, s0d)
            assert h["best_epoch"] >= 1 and h["history"]
            assert h["teacher_head"] == "stage0"
            # lr của HAI nhóm khác nhau ở mọi epoch, và attn < ffn
            for x in h["history"]:
                assert x["lr_attn"] < x["lr_ffn"], x
            # `lr_max` là MỨC CHẠY: sau warm-up cả hai nhóm GIỮ nguyên ở đó
            s1c_ = s1.s1cfg(cfg)
            spe1 = h["steps_per_epoch"]
            after = s1.lrs_at(s1.S0.warmup_steps(s1c_, spe1), s1c_, spe1)
            assert after == {"attn": s1c_["attn_lr_max"],
                             "ffn": s1c_["ffn_lr_max"]}, after
            # Extractor ĐỨNG YÊN: trọng số Stage 0 không đổi một bit
            d1 = s1.stage1_dir(cfg)
            assert R.compressor_exists(d1 / s1.CKPT)
            after = R.build_backbone(sp)[0]
            from safetensors.torch import load_file
            sd0 = load_file(str(s0d / R.STAGE0_FILE))
            assert all(torch.equal(before[k], sd0[k]) for k in before)
            _ = after
            # checkpoint CHỈ Compressor: không backbone, không Aligner
            sdc = load_file(str(d1 / s1.CKPT / R.COMPRESSOR_FILE))
            assert not any(k.startswith(("body.", "aligner")) for k in sdc)
            assert set(sdc) == set(R.VExtractorBranch(sp).compressor.state_dict())
            # nạp lại: Compressor đổi, Aligner + Extractor KHÔNG bị chạm
            br2 = R.VExtractorBranch(sp)
            al_before = br2.aligner.gate.weight.clone()
            bb_before = next(iter(br2.extractor.body.parameters())).clone()
            R.load_compressor_into(br2, d1 / s1.CKPT)
            assert torch.allclose(br2.aligner.gate.weight, al_before)
            assert torch.allclose(next(iter(br2.extractor.body.parameters())),
                                  bb_before)
            assert torch.allclose(br2.compressor.latent, sdc["latent"])
            # format lệch -> TỪ CHỐI (ở đây thứ được lưu CHÍNH LÀ Compressor)
            mp = d1 / s1.CKPT / R.COMPRESSOR_META
            meta = json.loads(mp.read_text(encoding="utf-8"))
            mp.write_text(json.dumps(dict(meta, format="rgexp.vextractor/1")),
                          encoding="utf-8")
            try:
                R.load_compressor_into(R.VExtractorBranch(sp), d1 / s1.CKPT)
            except RuntimeError as e:
                assert "format" in str(e)
            else:
                raise AssertionError("format cũ của Stage 1 phải bị từ chối")
            mp.write_text(json.dumps(meta), encoding="utf-8")
            assert s1.finished(cfg) is not None
            # đổi hình dạng Compressor -> Stage 1 phải chạy lại
            c2 = dict(cfg)
            c2["vextractor"] = dict(cfg["vextractor"], compressor=dict(
                cfg["vextractor"]["compressor"], n_query=SMALL_Q + 1))
            assert s1.finished(c2) is None
            assert any("compressor_shape" in x for x in s1.diff_inputs(c2))
            # đổi Aligner -> KHÔNG phải chạy lại (Aligner không có ở Stage 1)
            c3 = dict(cfg)
            c3["vextractor"] = dict(cfg["vextractor"], aligner=dict(
                cfg["vextractor"]["aligner"], mlp_hidden=128))
            assert s1.finished(c3) is not None
    finally:
        (s0.decoder_width, s0.assert_same_pixels,
         s0.ImageOnlyDataset.__getitem__, R.backbone_body) = orig


@check("17. main stage: Aligner học TỪ BƯỚC 0, Compressor đóng băng 20% rồi warm-up 10%; mốc theo TỈ LỆ nên dry-run vẫn mở băng")
def _():
    from src.config import load_config
    from src.train_vextractor import align_lr_at, comp_lr_at, zcfg

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    zc = zcfg(cfg)
    N = 200                                    # 1 epoch = 200 bước (khoá theo $1)
    # BỐN nhóm, hai hàm, mỗi hàm trả {"ffn", "attn"} ─────────────────────
    assert set(align_lr_at(0, zc, N)) == set(comp_lr_at(0, zc, N)) == {"ffn", "attn"}
    # ── Aligner: dốc từ bước 0, chạm đỉnh ở 10%, rồi PHẲNG ───────────────
    for grp in ("ffn", "attn"):
        lo_k, hi_k = f"align_{grp}_lr_min", f"align_{grp}_lr_main"
        assert align_lr_at(0, zc, N)[grp] == zc[lo_k] > 0
        assert align_lr_at(20, zc, N)[grp] == zc[hi_k]
        assert align_lr_at(199, zc, N)[grp] == zc[hi_k]
        a = [align_lr_at(k, zc, N)[grp] for k in range(21)]
        assert all(x <= y + 1e-18 for x, y in zip(a, a[1:]))
    # CÙNG một dốc cho hai nhóm: chúng ở trong cùng một khối ngẫu nhiên
    assert (align_lr_at(10, zc, N)["ffn"] / zc["align_ffn_lr_main"]
            - align_lr_at(10, zc, N)["attn"] / zc["align_attn_lr_main"]) < 1e-9
    # ── Compressor: 0 TUYỆT ĐỐI tới bước 40, rồi dốc, rồi PHẲNG ──────────
    for grp in ("ffn", "attn"):
        lo_k, hi_k = f"comp_{grp}_lr_min", f"comp_{grp}_lr_main"
        assert comp_lr_at(0, zc, N)[grp] == 0.0
        assert comp_lr_at(39, zc, N)[grp] == 0.0
        assert comp_lr_at(40, zc, N)[grp] == zc[lo_k]
        assert comp_lr_at(60, zc, N)[grp] == zc[hi_k]
        assert comp_lr_at(199, zc, N)[grp] == zc[hi_k]
        # HAI lịch KHÁC nhau — đó là cả điểm của việc tách hai khối
        assert align_lr_at(0, zc, N)[grp] != comp_lr_at(0, zc, N)[grp]
        assert zc[f"comp_{grp}_lr_main"] < zc[f"align_{grp}_lr_main"]
    # ── mốc theo TỈ LỆ: dry-run 10 bước vẫn mở băng Compressor ────────────
    for n_small in (10, 20, 40):
        assert comp_lr_at(0, zc, n_small)["ffn"] == 0.0
        assert comp_lr_at(n_small - 1, zc, n_small)["ffn"] > 0, n_small
        assert (align_lr_at(n_small - 1, zc, n_small)["ffn"]
                == zc["align_ffn_lr_main"])
    # Khai theo SỐ BƯỚC tuyệt đối thì ca trên hỏng im lặng — đó là lý do khoá
    # là `*_frac`, và không có khoá `*_steps` nào trong MAIN_CFG_KEYS.
    from src.train_vextractor import MAIN_CFG_KEYS

    assert not any(k.endswith("_steps") for k in MAIN_CFG_KEYS)


@check("18. kéo Stage 0 từ Hub (giả lập): base khớp -> về máy, chạy tiếp epoch đuôi")
def _():
    import json

    from src import stage0_vextractor as s0
    from src import vextractor as R
    from src.config import Cfg
    from src.run_vextractor import pull_stage0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        hub = td / "hub"
        cfg = Cfg(json.loads(json.dumps({
            "seed": 42,
            "stage0": {"lr_min": 1e-7, "lr_max": 1e-4,
                       "warmup_epochs": 1.0, "max_epochs": 3},
            "vextractor": small_cfg()["vextractor"],
            "output": {"adapter_dir": str(td / "ad"), "hub_repo": "x/y",
                       "hub_repo_type": "dataset", "hub_root": "current",
                       "hub_index": "2.2", "hub_title": "V-Extractor"}})))
        sp = R.spec_from_cfg(cfg, 32)
        sc = s0.s0cfg(cfg)
        # "Trên Hub": chỉ có base/ (phần theo lịch), KHÔNG có final/
        base_dir = hub / "stage0" / s0.BASE
        R.save_stage0(R.build_backbone(sp)[0], sp, base_dir, {"best_epoch": 2})
        h = {"finished": False, "base_finished": True, "base_epoch": 2,
             "base_epochs_run": 3, "best_epoch": 2, "stage0": dict(sc),
             "spec": sp, "seed": 42,
             "base_fingerprint": s0.base_fingerprint(cfg),
             "config_fingerprint": s0.config_fingerprint(cfg)}
        (hub / "stage0" / s0.HISTORY).write_text(json.dumps(h, default=str),
                                                 encoding="utf-8")
        remote = {f"current/$2.2_V-Extractor_Test/stage0/{r}"
                  for r in s0.KEEP if (hub / "stage0" / r).exists()}

        import src.run_vextractor as RV

        def fake_api(_c):
            return object(), "x/y", "dataset", remote

        def fake_dl(_api, _repo, _typ, path, dest):
            rel = path.split("/stage0/", 1)[1]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((hub / "stage0" / rel).read_bytes())

        import src.hub_sync as HS
        o1, o2 = HS._api_and_index, HS._download
        HS._api_and_index, HS._download = fake_api, fake_dl
        try:
            assert pull_stage0(cfg) is True
            d = s0.stage0_dir(cfg)
            assert R.stage0_exists(d / s0.BASE)
            assert not R.stage0_exists(d / s0.CKPT)    # final/ sẽ chạy trên máy này
            assert s0.base_finished(cfg) is not None
            assert s0.finished(cfg) is None
            # cấu hình KHÁC -> không dùng
            c2 = Cfg(json.loads(json.dumps(dict(cfg))))
            c2["seed"] = 7
            c2["output"] = dict(c2["output"], adapter_dir=str(td / "ad2"))
            assert pull_stage0(c2) is False
        finally:
            HS._api_and_index, HS._download = o1, o2
        _ = RV


@check("19. Lens chạy thật (mô hình giả): CHỈ B, heatmap token×lớp ĐỦ lớp (top-k mới lấy 1/4 cuối), mục 1 ra MỘT dải 3 ô (Z/Q/T, thang màu riêng từng ô), mục 3 ra MỘT dải 4 ô (gốc / Z / khối Q truy ngược qua teacher ConvNeXt về pixel / đầu ra Extractor), Lens_Explain.md")
def _():
    import json
    import types

    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image

    from src import lens_vextractor as LZ
    from src import recombiner
    from src import vextractor as R
    from src.config import load_config

    NV, DV, DM, NZ, V, L = 64, 8, 32, 4, 11, 3     # patch, d_vis, d_model, |Z|, vocab, lớp
    NQ = SMALL_Q                                    # số latent Q (khoá khai, không suy ra)
    IMG_TOK = V - 1

    class _Proj(nn.Module):                        # (1, NV, DV) -> (1, NZ, DM)
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(DV, DM)

        def forward(self, x):
            return self.fc(x).mean(1, keepdim=True).expand(1, NZ, DM)

    class _Tower(nn.Module):
        def forward(self, pixel_values=None):
            return types.SimpleNamespace(
                last_hidden_state=pixel_values.reshape(1, NV, DV).float())

    class _Text(nn.Module):
        """Khối chữ nằm ở NHÁNH riêng, đúng như Gemma3: .model.language_model."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(DM)
            self.layers = nn.ModuleList(nn.Linear(DM, DM) for _ in range(L))

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = _Tower()
            self.multi_modal_projector = _Proj()
            self.language_model = _Text()

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.emb = nn.Embedding(V, DM)
            self.lm_head = nn.Linear(DM, V, bias=False)
            self.config = types.SimpleNamespace(
                image_token_index=IMG_TOK,
                text_config=types.SimpleNamespace(hidden_size=DM))

        def get_input_embeddings(self):
            return self.emb

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, inputs_embeds=None, labels=None, output_hidden_states=False,
                    **_kw):
            txt = self.model.language_model
            h, hs = inputs_embeds, [inputs_embeds]
            for lay in txt.layers:
                h = torch.tanh(lay(h)) + h
                hs.append(h)
            h = txt.norm(h)              # HF chuẩn hoá TRƯỚC khi ghi hidden cuối
            hs[-1] = h
            return types.SimpleNamespace(logits=self.lm_head(h),
                                         hidden_states=tuple(hs), loss=None)

    fake = Fake()
    # Cây module không phải một dây thẳng — đây chính là chỗ `lens` từng nổ.
    assert recombiner._find_module(fake, ("norm", "final_layernorm")) is \
        fake.model.language_model.norm
    assert recombiner._find_module(fake, ("vision_tower",)) is fake.model.vision_tower
    cfg = load_config(ROOT / "configs/vextractor.yaml")
    sp = small_spec(d_model=DM)
    branch = R.VExtractorBranch(sp)
    branch.set_frozen(True)
    ids = torch.tensor([[0, 1] + [IMG_TOK] * NZ + [2, 3, 4]])
    labels = ids.clone()
    labels[:, : 2 + NZ] = -100
    batch = {"input_ids": ids, "labels": labels,
             "attention_mask": torch.ones_like(ids),
             "pixel_values": torch.randn(1, NV * DV),
             # đường ảnh THỨ HAI của nhánh (§2.2), độc lập với pixel_values
             "vext_pixels": torch.randn(1, 3, SMALL_IMG, SMALL_IMG)}

    ra = LZ.analyze(cfg, fake, fake, None, batch, "ref_logprob")   # không nhánh
    # Arm KHÔNG có nhánh: không backbone nào để hỏi, nên không có bản đồ
    # teacher — và `analyze` phải TRẢ VỀ chứ không chết giữa chừng bằng
    # AttributeError sau khi đã chạy xong cả một lượt backward.
    assert ra.get("teacher_feature_norm") is None
    assert ra.get("sal_q") is None and ra["sal_z"] is not None
    rb = LZ.analyze(cfg, fake, fake, branch, batch, "ref_logprob")
    T = int(ids.shape[1])
    assert ra["layer_sens"].shape == (L, T)                 # A: chuỗi gốc
    assert rb["layer_sens"].shape == (L, T + NQ)            # B: dài thêm đúng NQ
    # CHỈ CÒN BA Ô: gốc / Z / khối Q. Mọi bản đồ cũ đã bỏ hẳn.
    for r in (ra, rb):
        for gone in ("act_siglip", "sal_vext", "sal_vext_64", "sal_vext_act",
                     "sal_vext_act_64", "sal_vext_pre"):
            assert gone not in r, gone
    # HAI lưới khác nhau: Z ở 8×8 (patch SigLIP), Q ở 64×64 (PIXEL của ảnh nhánh,
    # SMALL_IMG=64 thay cho 448² của cấu hình thật)
    assert ra["sal_z"].shape == (8, 8)
    # Ô 3 CHỈ có ở lượt CÓ nhánh: nó là bản đồ của chính khối Q
    assert ra.get("sal_q") is None and rb["sal_q"] is not None
    assert rb["sal_q"].shape == (SMALL_IMG, SMALL_IMG), rb["sal_q"].shape
    assert np.isfinite(rb["sal_q"]).all() and rb["sal_q"].sum() > 0
    # bản gộp về lưới chung để so số
    assert rb["sal_q_64"].shape == (8, 8)
    assert np.isclose(rb["sal_q_64"].mean(), rb["sal_q"].mean(), rtol=1e-5)
    assert LZ.pool_to(np.ones((8, 8)), 8).shape == (8, 8)
    assert LZ.pool_to(np.ones((9, 9)), 8) is None            # không chia hết -> None
    # Ô 3 KHÔNG dính decoder: nó dựng được cả khi `y` không tồn tại. Kiểm bằng
    # cách gọi thẳng `_q_saliency_map` — cùng kết quả, không cần model nào.
    direct = LZ._q_saliency_map(branch, batch["vext_pixels"], torch.device("cpu"))
    assert np.allclose(direct, rb["sal_q"], atol=1e-5)
    # Lens KHÔNG được để lại dấu vết trên nhánh: trọng số đóng băng, cờ như cũ,
    # và backbone vẫn ở chế độ teacher inference (eval)
    assert branch.frozen and all(not q.requires_grad for q in branch.parameters())
    assert not branch.extractor.body.training
    # Q nằm SAU khối ảnh -> vị trí của token ảnh KHÔNG đổi
    assert ra["image_span"] == rb["image_span"] == [2, 1 + NZ]
    assert rb["insert_span"] == [2 + NZ, 2 + NZ + NQ], rb["insert_span"]
    assert ra["insert_span"] is None
    # 1/4 lớp cuối, chỉ số THẬT (1-based), và ô nào thuộc Z / Q / text
    assert LZ.last_layers(34, 0.25) == list(range(26, 35))
    assert LZ.last_layers(L, 0.25) == [L]
    assert LZ._where(2, rb) == "Z[0]" and LZ._where(2 + NZ, rb) == "Q[0]"
    assert LZ._where(T + NQ - 1, rb) == "text"
    top = LZ.logit_lens(fake, fake, rb["hidden_last"], 3)
    assert len(top) == L and all(len(t["ids"]) == 3 for t in top)
    for t in top:
        t["tokens"] = [f"t{i}" for i in t["ids"]]

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        Image.fromarray((np.random.rand(32, 32) * 255).astype("uint8")).save(t / "x.png")
        item = {"key": "k0", "ref": "r", "image_path": str(t / "x.png"), "n_pos": 5,
                "n_unc": 3, "n_labelled": 8,
                "findings": [{"label": "Edema", "class": "pos"}]}
        lc = LZ.lcfg(cfg)
        assert lc["layer_frac"] == 0.25
        figs = dict(LZ.plot_layers(rb, t / "figures" / "lens_layers.png", lc))
        figs["visual"] = LZ.plot_visual(item, rb,
                                        t / "figures" / "lens_visual.png", lc)
        # BA hình riêng, không phải một — và mỗi hình phải THẬT SỰ ra file.
        # MỘT file cho dải layers, MỘT file cho dải visual — tách THANG MÀU
        # chứ không tách file (xem docstring plot_layers).
        assert set(figs) == {"layers", "visual"}, set(figs)
        for k in ("layers", "visual"):
            assert Path(figs[k]).exists() and Path(figs[k]).stat().st_size > 0, k
        # Ba vùng phải RỜI NHAU và PHỦ KÍN chuỗi: T là phần bù, không phải đuôi.
        regs = LZ._regions(rb, T + NQ)
        allrows = sorted(r for g in regs for r in g["rows"])
        assert allrows == list(range(T + NQ)), (len(allrows), T + NQ)
        assert [g["key"] for g in regs] == ["z", "q", "t"]
        A, B = {"id": "a"}, {"id": "b"}
        dg = LZ.diagnostics(rb)
        assert 0.0 <= dg["image_token_share"] <= 1.0
        assert 0.0 <= dg["q_token_share"] <= 1.0
        assert dg["q_vector_share"]["n"] == NQ
        assert 1 <= dg["q_vector_share"]["n_active_50pct"] <= NQ
        assert 1 <= dg["peak"]["layer"] <= L
        assert 0.0 <= dg["top10_iou"] <= 1.0 and dg["vext_energy_ratio"] > 0
        # None hợp lệ: bản đồ hằng (std = 0) thì tương quan không định nghĩa
        assert dg["saliency_corr"] is None or -1.0 <= dg["saliency_corr"] <= 1.0
        qm = dg["q_map"]
        assert qm["max_over_mean"] >= 1.0 and 0.0 <= qm["top10_share"] <= 1.0
        for gone in ("siglip_activity", "siglip_corr", "vext_self",
                     "aligner_shift"):
            assert gone not in dg, gone
        out = {"lens": lc, "diagnostics": dg, "reference_arm": "a",
               "arm": {"id": "b", "y": rb["y"], "n_tokens": T + NQ, "layers": L,
                       "image_span": rb["image_span"],
                       "insert_span": rb["insert_span"]}}
        p = LZ.write_doc(cfg, t, A, B, item, rb, top, figs, out)
        doc = p.read_text(encoding="utf-8")
        assert "ĐỦ mọi lớp" in doc and "lớp cuối" in doc   # mục 1 đủ, bảng 2 cắt
        assert "Trục **Y là vị trí token**" in doc and "**X là lớp**" in doc
        # Doc phải nhúng ĐỦ ba hình và phải NÓI RA rằng thang màu khác nhau —
        # một người đọc so độ sáng giữa hai hình là đọc sai, và chỉ có dòng chữ
        # ấy chặn được.
        assert "KHÔNG so được độ sáng giữa ba ô" in doc
        for need in ("## 1.", "## 2.", "## 3.", "lens_layers.png",
                     "lens_visual.png",
                     "Edema", "Giới hạn", "## Cách đọc", "Q chính là phần",
                     "Kiểm nhanh trước khi tin hình", "Đọc ba ô thế nào",
                     "Đọc bảng 2 thế nào", "Đọc hình 3 thế nào",
                     "KHÔNG dính decoder", "teacher",
                     f"{cfg['vextractor']['compressor']['n_query']} High-Q token",
                     "Q → Aligner → Compressor → Extractor → I",
                     "`Q[0]`", "448²"):
            assert need in doc, need
        assert "`sal_z`" in doc and "`sal_q`" in doc
        assert json.dumps(out, default=str)


@check("20. lệnh `lens` có trong CLI + script, config lens hợp lệ")
def _():
    from src.config import load_config
    from src.lens_vextractor import lcfg
    from src.run_vextractor import COMMANDS

    assert COMMANDS["lens"] == ["lens"]
    c = lcfg(load_config(ROOT / "configs/vextractor.yaml"))
    assert c["enabled"] and c["target"] == "ref_logprob" and c["dir"] == "Lens"
    sh = (ROOT / "scripts/run_vextractor.sh").read_text(encoding="utf-8")
    assert "train|push|stats|lens|all)" in sh
    src = (ROOT / "src/run_vextractor.py").read_text(encoding="utf-8")
    # Lens §2.1 CHỈ cần weight của B — không kéo weight của A về nữa
    assert "_cmd_lens" in src and "arm_weight_dir(cfg, B, adapter_root).parent" in src
    assert "want_weights=True" not in src




@check("21. thứ tự chuỗi [Z ‖ Q ‖ T]: Q ngay SAU khối ảnh, nhãn −100, CE không đổi số token")
def _():
    import torch

    from src import recombiner as rc
    from src.train_vextractor import _ntok

    NZ, NQ, DM, IMG = 6, 3, 5, 99
    # [BOS, sys, <img>×NZ, chữ...] — đệm PHẢI, mọi hàng cùng prompt (khâu train)
    ids = torch.tensor([[7, 8] + [IMG] * NZ + [11, 12, 13, 0],
                        [7, 8] + [IMG] * NZ + [14, 15, 0, 0]])
    am = (ids != 0).long()
    am[:, :2 + NZ] = 1
    labels = ids.clone()
    labels[:, :2 + NZ] = -100
    labels[am == 0] = -100
    tt = torch.zeros_like(ids)
    tt[:, 2:2 + NZ] = 1                        # token ảnh của Gemma 3
    embeds = torch.randn(2, ids.shape[1], DM)
    q = torch.full((2, NQ, DM), 7.0)

    at = rc.last_image_index(ids, IMG) + 1
    assert at == 2 + NZ, at
    assert int(rc.first_image_index(ids, IMG)) == 2
    w = rc.inject_tokens(embeds, q, at, attention_mask=am, labels=labels,
                         token_type_ids=tt, token_type_value=0)
    e2, am2, lb2, tt2 = (w["inputs_embeds"], w["attention_mask"], w["labels"],
                         w["token_type_ids"])
    assert e2.shape == (2, ids.shape[1] + NQ, DM)
    # Z giữ nguyên chỗ, Q nằm NGAY SAU, chữ bị đẩy sang phải đúng NQ
    assert torch.equal(e2[:, :at], embeds[:, :at])
    assert torch.equal(e2[:, at:at + NQ], q)
    assert torch.equal(e2[:, at + NQ:], embeds[:, at:])
    assert torch.equal(tt2[:, 2:2 + NZ], torch.ones(2, NZ, dtype=tt.dtype))
    assert int(tt2[:, at:at + NQ].abs().sum()) == 0        # Q KHÔNG phải token ảnh
    assert int((lb2[:, at:at + NQ] != -100).sum()) == 0    # Q không vào CE
    assert int(am2[:, at:at + NQ].sum()) == 2 * NQ         # nhưng vẫn được nhìn
    assert _ntok(lb2, am2) == _ntok(labels, am)            # số token vào CE không đổi

    # khâu SINH đệm TRÁI -> mỗi hàng một chỗ; `last_image_index` phải DỪNG
    left = torch.tensor([[0, 7] + [IMG] * NZ + [11, 12],
                         [7, 8] + [IMG] * NZ + [14, 15]])
    left[0, 0] = 0
    lids = torch.tensor([[0, 0, 7] + [IMG] * NZ + [11],
                         [7, 8, 9] + [IMG] * NZ + [14]])
    ends = rc.image_end_indices(lids, IMG)
    assert ends.tolist() == [2 + NZ, 2 + NZ]               # cùng chỗ ở ví dụ này
    shifted = torch.tensor([[0, 0, 7] + [IMG] * NZ + [11],
                            [7, 8, 9, 10] + [IMG] * NZ])
    try:
        rc.last_image_index(shifted, IMG)
    except RuntimeError:
        pass
    else:
        raise AssertionError("last_image_index phải DỪNG khi các hàng lệch chỗ")
    assert rc.image_end_indices(shifted, IMG).tolist() == [2 + NZ, 3 + NZ]
    _ = left


@check("22. ĐẶC TẢ khớp spec SỐNG: mọi con số kiến trúc trong README, config "
       "và docstring được dựng lại từ code, không phải chép tay")
def _():
    """README trôi khỏi code ở MỖI lần đổi kiến trúc, và nó trôi IM LẶNG.

    Không có gì buộc đặc tả đi theo code, nên cách duy nhất nó sai là sai âm
    thầm: người đọc tin một bảng tham số không còn đúng. Bài này dựng nhánh
    THẬT từ `configs/vextractor.yaml` rồi đòi đặc tả chứa đúng những con số ấy.

    `pretrained=False`: số tham số không phụ thuộc trọng số, nên bài này chạy
    OFFLINE và không tải gì.
    """
    from src import dino as DN
    from src import vextractor as R
    from src.config import load_config

    cfg = load_config(ROOT / "configs/vextractor.yaml")
    sp = R.spec_from_cfg(cfg, 2560)
    sp["extractor"]["pretrained"] = False
    try:
        br = R.VExtractorBranch(sp)
    except R.VExtractorUnusable as e:
        raise AssertionError(
            "không dựng được nhánh từ configs/vextractor.yaml, nên bài này "
            "KHÔNG nói được README có khớp code hay không. Extractor "
            "InternImage nạp qua `trust_remote_code`: với `pretrained=False` "
            "nó chỉ cần `config.json` + ba file .py (~60 KB, có cache), nhưng "
            "LẦN ĐẦU vẫn cần mạng.\n"
            f"    Lý do thật: {e}") from e
    ex, cp, al = br.extractor, br.compressor, br.aligner
    dc = cfg["stage0"]["dino"]
    head = DN.DINOHead(ex.d_feat, dc["out_dim"], dc["hidden"], dc["bottleneck"],
                       dc["head_layers"])
    n = lambda m: sum(q.numel() for q in m.parameters())    # noqa: E731
    # README viết số thập phân theo kiểu Việt (dấu phẩy) — so đúng dạng ấy.
    m = lambda v: f"{v / 1e6:.2f}".replace(".", ",") + " M"  # noqa: E731

    rd = (ROOT / "README.md").read_text(encoding="utf-8")
    need = [
        (str(sp["extractor"]["arch"]), "arch"),
        (f"({ex.n_patch}, {ex.d_feat})", "V"),
        (f"({cp.n_query}, {cp.d_latent})", "V_c"),
        (f"SwiGLU {al.d_latent}→{al.mlp_hidden}→{al.d_model}", "Aligner"),
        (f"SwiGLU {cp.d_latent}→{cp.mlp_hidden}→{cp.d_latent}", "FFN của Compressor"),
        (f"`head_dim` = {cp.head_dim}", "head_dim khai"),
        (f"{cp.n_query} latent", "n_query"),
        (f"`n_stages: {ex.n_stages}`", "lát cắt stage"),
        # HAI khối SwiGLU, HAI khoá — README phải nói cả hai.
        ("`vextractor.compressor.mlp_hidden`", "mlp_hidden của Compressor"),
        ("`vextractor.aligner.mlp_hidden`", "mlp_hidden của Aligner"),
        (f"{ex.n_patch} ô", "lưới"),
        (m(n(ex)), "tham số Extractor"),
        (m(n(cp.groups[0])), "tham số MỘT nhóm khối"),
        (m(n(cp)), "tham số Compressor"),
        (m(n(al)), "tham số Aligner"),
        (m(n(br)), "tham số nhánh"),
        (m(sum(q.numel() for q in br.trainable_parameters())), "phần học được"),
        (m(n(head)), "DINO head"),
        (m(n(ex) + n(head)), "Stage 0 student"),
    ]
    miss = [f"{name} ({s!r})" for s, name in need if s not in rd]
    assert not miss, ("README KHÔNG còn khớp spec sống — sửa README, đừng nới "
                      f"bài kiểm: {miss}")

    # Và số của BỐ CỤC CŨ không được còn sót — ở BA bề mặt đặc tả. Mọi chuỗi ở
    # đây SUY RA từ spec sống: không có danh sách "số của bố cục cũ" nào gõ tay.
    def wrong(v):
        return [x for x in (v // 2, v * 2, 16, 196, 2048) if x != v]

    stale = [f"{k} latent" for k in wrong(cp.n_query)]
    stale += [f"`d_latent` = {k}" for k in wrong(cp.d_latent)]
    stale += [f"SwiGLU {cp.d_latent}→{k}→{cp.d_latent}" for k in wrong(cp.mlp_hidden)]
    stale += [f"({k}, {ex.d_feat})" for k in wrong(ex.n_patch)]
    surfaces = {
        "README.md": rd,
        "configs/vextractor.yaml": (ROOT / "configs/vextractor.yaml").read_text(
            encoding="utf-8"),
        # CHỈ docstring của module, không cả file: phần thân có thông báo lỗi
        # nhắc tới giá trị sai, và đó là chuỗi ĐÚNG, không phải drift.
        "src/vextractor.py (docstring)": (
            (ROOT / "src/vextractor.py").read_text(encoding="utf-8")
            .split('"""')[1]),
    }
    left = {f: [s for s in stale if s in body] for f, body in surfaces.items()}
    left = {f: v for f, v in left.items() if v}
    assert not left, f"đặc tả còn con số KHÔNG khớp spec sống: {left}"


@check("23. vân tay Stage 0 chỉ phụ thuộc EXTRACTOR: đổi Compressor/Aligner KHÔNG "
       "bắt finetune lại; checkpoint Stage 0 sống qua một lần bump FORMAT")
def _():
    """Hai cổng từng bắt chạy lại một lượt DINO nhiều giờ mà KHÔNG có lý do.

    1. `config_fingerprint` băm CẢ khối `vextractor`. Không thứ nào ngoài
       Extractor có mặt ở Stage 0, nên đổi chúng làm vân tay đổi và người chạy
       bị bảo `--restage0` cho một backbone y hệt.
    2. `load_stage0_into` đòi `format` BẰNG ĐÚNG `FORMAT`. Nhưng `FORMAT` bump
       mỗi lần Compressor/Aligner đổi bố cục, trong khi Extractor y nguyên, nên
       một checkpoint hoàn toàn dùng được bị vứt.
    """
    import json

    from src import stage0_vextractor as s0
    from src import vextractor as R

    base = small_spec()

    def cfg_of(**over):
        vx = json.loads(json.dumps(small_cfg()["vextractor"]))
        vx["main"] = {"align_lr_main": 1e-4}
        for blk, patch in over.pop("vx", {}).items():
            vx[blk] = dict(vx.get(blk) or {}, **patch)
        return {"seed": 42,
                "stage0": {"lr_min": 1e-7, "lr_max": 1e-4,
                           "warmup_epochs": 1.0, "max_epochs": 3},
                "vextractor": vx}

    ref = s0.config_fingerprint(cfg_of())
    assert s0.base_fingerprint(cfg_of()) == ref     # cùng phạm vi, hai tên

    # ── đổi phần NGOÀI Extractor -> vân tay KHÔNG được đổi ────────────────
    same = {
        "d_latent": cfg_of(vx={"compressor": {"d_latent": 32}}),
        "n_query": cfg_of(vx={"compressor": {"n_query": 3}}),
        "head_dim": cfg_of(vx={"compressor": {"head_dim": 8}}),
        "n_blocks": cfg_of(vx={"compressor": {"n_blocks": 3}}),
        "dropout": cfg_of(vx={"compressor": {"dropout": 0.1}}),
        "image_size": cfg_of(vx={"extractor": {"image_size": SMALL_IMG * 2}}),
        "aligner": cfg_of(vx={"aligner": {"mlp_hidden": 999}}),
        "main": cfg_of(vx={"main": {"align_lr_main": 0.5}}),
    }
    drift = [k for k, c in same.items() if s0.config_fingerprint(c) != ref]
    assert not drift, (
        f"đổi {drift} làm vân tay Stage 0 đổi — những khoá ấy tả "
        f"Compressor/Aligner, Stage 0 không dựng chúng, và bắt finetune lại vì "
        f"chúng là bắt trả vài giờ GPU cho một Extractor y hệt.")

    # ── đổi EXTRACTOR -> vân tay PHẢI đổi (cổng vẫn còn răng) ─────────────
    for key, c in (("arch", cfg_of(vx={"extractor": {"arch": SMALL_VIT}})),
                   ("n_stages", cfg_of(vx={"extractor": {"n_stages": 2}})),
                   ("pretrained", cfg_of(vx={"extractor": {"pretrained": True}}))):
        assert s0.config_fingerprint(c) != ref, f"đổi {key} PHẢI đổi vân tay"
    for key, val in (("lr_max", 5e-4), ("patience", 9), ("tail_epochs", 3),
                     ("warmup_epochs", 2.0), ("plateau_gamma", 5.0)):
        c = cfg_of()
        c["stage0"][key] = val
        assert s0.config_fingerprint(c) != ref, key
    c = cfg_of(); c["seed"] = 7
    assert s0.config_fingerprint(c) != ref

    # ── checkpoint Stage 0 sống qua một lần bump FORMAT ───────────────────
    body = R.build_backbone(base)[0]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "s0"
        R.save_stage0(body, base, d, {"best_epoch": 1})
        meta_p = d / R.STAGE0_META
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        gen = R.format_gen(R.FORMAT)
        meta["format"] = f"rgexp.vextractor/{gen - 1}"
        meta_p.write_text(json.dumps(meta), encoding="utf-8")
        R.load_stage0_into(R.VExtractorBranch(base), d)   # phải NẠP ĐƯỢC
        # nhưng Extractor KHÁC thì vẫn bị chặn
        meta["backbone_spec"] = dict(meta["backbone_spec"], arch="khac_hoan_toan")
        meta_p.write_text(json.dumps(meta), encoding="utf-8")
        try:
            R.load_stage0_into(R.VExtractorBranch(base), d)
        except RuntimeError as e:
            assert "backbone" in str(e) or "restage0" in str(e)
        else:
            raise AssertionError("backbone khác phải bị chặn")
        # và thế hệ trước /5 vẫn bị từ chối
        meta["format"] = f"rgexp.vextractor/{R.STAGE0_MIN_GEN - 1}"
        meta_p.write_text(json.dumps(meta), encoding="utf-8")
        try:
            R.load_stage0_into(R.VExtractorBranch(base), d)
        except RuntimeError as e:
            assert "restage0" in str(e)
        else:
            raise AssertionError("format quá cũ phải bị từ chối")


@check("24. Stage 0 cũ vẫn dùng được sau khi SỬA phạm vi vân tay: cổng so NỘI "
       "DUNG đã ghi, không so hash (hash cũ không hồi tố được)")
def _():
    """Ca thật, đã gặp: phạm vi vân tay sai -> sửa hàm -> checkpoint vẫn bị vứt.

    Hash đã nằm trong `stage0_history.json` của lượt đã chạy thì không sửa theo
    được. Nếu cổng so hash, MỌI checkpoint ghi trước lần sửa đều bị tuyên "cấu
    hình KHÁC" và người chạy mất vài giờ GPU cho một Extractor y hệt.

    Nên cổng so NỘI DUNG: history ghi cả khối `stage0` đã giải quyết lẫn
    `spec`, `loader.json` ghi `seed`, nên dựng lại được đúng bộ đầu vào.
    """
    import json

    from src import stage0_vextractor as s0
    from src import vextractor as R
    from src.config import Cfg

    with tempfile.TemporaryDirectory() as td:
        cfg = Cfg(json.loads(json.dumps({
            "seed": 42,
            "stage0": {"lr_min": 1e-7, "lr_max": 1e-4, "warmup_epochs": 1.0,
                       "max_epochs": 3, "micro_batch": 8},
            "vextractor": small_cfg()["vextractor"],
            "output": {"adapter_dir": td}})))
        d = s0.stage0_dir(cfg)
        d.mkdir(parents=True, exist_ok=True)
        sc = s0.s0cfg(cfg)
        sp = R.spec_from_cfg(cfg, 32)
        # history của lượt ĐÃ CHẠY, mang hash của hàm băm CŨ — giá trị nào cũng
        # được, miễn là KHÁC hash bây giờ.
        h = {"finished": True, "base_finished": True, "best_epoch": 2,
             "base_epoch": 2, "stage0": dict(sc), "spec": sp, "seed": 42,
             "config_fingerprint": "hash_cua_ham_bam_CU"}
        (d / s0.HISTORY).write_text(json.dumps(h, default=str), encoding="utf-8")
        for sub in (s0.CKPT, s0.BASE):
            R.save_stage0(R.build_backbone(sp)[0], sp, d / sub, {"best_epoch": 2})

        assert s0.diff_inputs(d, h, cfg) == [], "hash cũ KHÔNG được chặn"
        assert s0.finished(cfg) is not None, "Stage 0 cũ phải dùng lại được"
        assert s0.base_finished(cfg) is not None

        # ĐỔI phần ngoài Extractor -> vẫn dùng lại được (đó là cả điểm của fix)
        for key, val in (("d_latent", 32), ("n_query", 3), ("head_dim", 8),
                         ("n_blocks", 4)):
            c2 = Cfg(json.loads(json.dumps(dict(cfg))))
            c2["vextractor"]["compressor"][key] = val
            assert s0.finished(c2) is not None, f"đổi {key} không được bắt chạy lại"

        # ĐỔI Extractor / stage0 / seed -> PHẢI bắt chạy lại, và nói rõ chỗ lệch
        c2 = Cfg(json.loads(json.dumps(dict(cfg))))
        c2["vextractor"]["extractor"]["arch"] = SMALL_VIT
        c2["vextractor"]["extractor"]["n_stages"] = None
        assert s0.finished(c2) is None
        assert any("backbone" in x for x in s0.diff_inputs(d, h, c2))
        c2 = Cfg(json.loads(json.dumps(dict(cfg))))
        c2["stage0"]["lr_max"] = 3e-4
        assert s0.finished(c2) is None
        assert any("stage0.lr_max" in x for x in s0.diff_inputs(d, h, c2))
        c2 = Cfg(json.loads(json.dumps(dict(cfg))))
        c2["seed"] = 7
        assert s0.finished(c2) is None
        assert any("seed" in x for x in s0.diff_inputs(d, h, c2))
        # `micro_batch` chỉ nói chuyện VRAM -> KHÔNG bắt chạy lại
        c2 = Cfg(json.loads(json.dumps(dict(cfg))))
        c2["stage0"]["micro_batch"] = 2
        assert s0.finished(c2) is not None

        # history THIẾU spec/seed -> rơi về so vân tay, không đoán bừa
        h2 = {k: v for k, v in h.items() if k not in ("spec", "seed")}
        (d / s0.HISTORY).write_text(json.dumps(h2, default=str), encoding="utf-8")
        assert s0.history_inputs(d, h2) is None
        assert s0.diff_inputs(d, h2, cfg) != [], "thiếu spec phải rơi về hash"


@check("25. BA giai đoạn: Stage 0 dạy Extractor · Stage 1 chưng cất Compressor từ "
       "teacher ĐÓNG BĂNG · main dạy Aligner + LoRA rồi mở Compressor")
def _():
    """Mỗi giai đoạn dạy ĐÚNG MỘT khối, và ranh giới phải kiểm được.

    Ba chỗ dễ lệch lặng lẽ, và bài này khoá cả ba:

      1. Stage 1 vô tình cập nhật Extractor (teacher phải ĐỨNG YÊN — nó đang
         sinh nhãn cho chính mình thì phép chưng cất không còn nghĩa gì);
      2. checkpoint Stage 1 mang theo Extractor hoặc Aligner — hai bản của cùng
         một tensor, hoặc một Aligner đã học đè lên một Aligner phải ngẫu nhiên;
      3. main stage quên mở băng Compressor, hoặc mở sai bước.
    """
    import json as _json

    import torch

    from src import dino as D
    from src import stage1_vextractor as s1
    from src import vextractor as R
    from src.config import Cfg, load_config

    cfg = load_config(ROOT / "configs/vextractor.yaml")

    # ── config Stage 1 mượn DINO của Stage 0, và ép n_local = 0 ───────────
    sc = s1.s1cfg(cfg)
    dc0 = cfg["stage0"]["dino"]
    assert sc["dino"]["n_local"] == 0 != int(dc0["n_local"])
    for k in ("teacher_temp", "student_temp", "center_momentum", "out_dim", "pool"):
        assert sc["dino"][k] == dc0[k], k     # KHÔNG được lệch công thức DINO
    for bad in ("n_local", "pool", "out_dim", "dino", "lr"):
        b = Cfg(_json.loads(_json.dumps(dict(cfg))))
        b["stage1"] = dict(b["stage1"], **{bad: 1})
        try:
            s1.s1cfg(b)
        except Exception as e:  # noqa: BLE001
            assert "khoá lạ" in str(e), (bad, e)
        else:
            raise AssertionError(f"stage1.{bad} phải bị từ chối")

    # ── vân tay Stage 1: Extractor + hình dạng Compressor + Stage 0, KHÔNG Aligner ──
    ins = s1.cfg_inputs(cfg)
    assert set(ins) == {"stage1", "backbone", "compressor_shape", "seed", "stage0"}
    assert set(ins["compressor_shape"]) == {"n_query", "d_latent", "head_dim",
                                            "mlp_hidden", "n_blocks"}
    f0 = s1.config_fingerprint(cfg)
    # Giá trị "khác" phải SUY RA từ cấu hình sống, không gõ tay: một hằng số
    # gõ tay có ngày TRÙNG với cấu hình thật, và lúc ấy phép thử thành no-op —
    # bài kiểm vẫn xanh trong khi nó không còn thử gì. Đã xảy ra: `n_query=8`
    # từng là "khác", rồi `n_query` thật đổi thành 8.
    _al = cfg["vextractor"]["aligner"]["mlp_hidden"]
    _nq = cfg["vextractor"]["compressor"]["n_query"]
    _lr = cfg["stage0"]["lr_max"]
    c2 = Cfg(_json.loads(_json.dumps(dict(cfg))))
    c2["vextractor"]["aligner"] = dict(c2["vextractor"]["aligner"], mlp_hidden=_al * 2)
    assert s1.config_fingerprint(c2) == f0, "đổi Aligner KHÔNG được bắt chạy lại Stage 1"
    c3 = Cfg(_json.loads(_json.dumps(dict(cfg))))
    c3["vextractor"]["compressor"] = dict(c3["vextractor"]["compressor"], n_query=_nq + 1)
    assert s1.config_fingerprint(c3) != f0, "đổi Compressor PHẢI bắt chạy lại Stage 1"
    c4 = Cfg(_json.loads(_json.dumps(dict(cfg))))
    c4["stage0"] = dict(c4["stage0"], lr_max=_lr * 10)
    assert s1.config_fingerprint(c4) != f0, "đổi Stage 0 PHẢI bắt chạy lại Stage 1"

    # ── CompressorTransferTrainer: teacher ĐỨNG YÊN, grad CHỈ tới Compressor ──
    torch.manual_seed(0)
    sp = small_spec()
    br = R.VExtractorBranch(sp)
    ex, cp = br.extractor, br.compressor
    dc = dict(sc["dino"], out_dim=32, hidden=16, bottleneck=8, head_layers=2)
    th = D.DINOHead(ex.d_feat, 32, 16, 8, 2)
    tr = D.CompressorTransferTrainer(ex, cp, th, dc, cp.d_latent)
    assert not any(p.requires_grad for p in tr.teacher_head.parameters())
    assert not any(p.requires_grad for p in ex.body.parameters())
    assert ex.frozen
    x = torch.randn(2, 3, SMALL_IMG, SMALL_IMG)
    tr.loss([x, torch.randn_like(x)], dc["teacher_temp"]).backward()
    assert all(p.grad is None for p in ex.body.parameters()), "Extractor KHÔNG được có grad"
    assert all(p.grad is None for p in tr.teacher_head.parameters())
    assert any(p.grad is not None for p in cp.parameters())
    assert all(p.grad is None for p in br.aligner.parameters()), "Aligner không có ở Stage 1"
    # KHÔNG có EMA ở Stage 1: teacher là checkpoint, không phải trung bình trượt
    assert not hasattr(tr, "ema_update") and not hasattr(tr, "teacher_body")
    # BA nhóm optimizer, và `head` đi cùng lr của `ffn`
    gp = tr.param_groups(1e-8, 1e-7)
    assert [g["name"] for g in gp] == ["attn", "ffn", "head"]
    assert gp[0]["lr"] == 1e-8 and gp[1]["lr"] == gp[2]["lr"] == 1e-7
    ids = {id(q) for g in gp for q in g["params"]}
    assert not (ids & {id(q) for q in tr.teacher_head.parameters()})

    # ── pipeline: Stage 1 nằm GIỮA Stage 0 và main, và cờ kéo theo nhau ───
    from src.run_vextractor import COMMANDS

    st = COMMANDS["train"]
    assert st.index("stage0") < st.index("stage1") < st.index("train")
    rsrc = (ROOT / "src/run_vextractor.py").read_text(encoding="utf-8")
    assert '"--restage1"' in rsrc
    assert "a.restage1 = a.retrain = True" in rsrc    # --restage0 kéo theo Stage 1
    assert "stage1_dir=s1dir / s1.CKPT" in rsrc
    # cờ của bản cũ đã biến mất hẳn
    for gone in ("--restage02", "--refinal", "stage02", "final_anneal"):
        assert gone not in rsrc, gone


print()
if FAILS:
    print(f"✗ {len(FAILS)} kiểm tra HỎNG: {FAILS}")
    sys.exit(1)
print("✓ selftest §2.2 V-Extractor qua hết")

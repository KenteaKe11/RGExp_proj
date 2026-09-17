"""Runner sinh báo cáo cho CẢ HAI candidate của §2.3.

Vì sao một runner cho cả hai, thay vì để candidate A dùng `gemma3_adapter`:
cùng lý lẽ với `train_recomb._forward_loss`. Ở đây chuỗi vào decoder được dựng
tay (`build_inputs_embeds`) thay vì để `Gemma3ForConditionalGeneration` tự ghép
từ `input_ids` + `pixel_values`. Hai đường tương đương về toán nhưng không giống
nhau từng phép tính, và khi hiệu số đo được là vài phần trăm BLEU thì "gần như
tương đương" không đủ. Candidate A ở đây = candidate B với `branch = None`, và
đó là một câu kiểm tra được bằng cách đọc.

Ba chỗ phải đúng, không chỗ nào tự lộ ra nếu sai:

  1. Base 4-bit lúc train thì phải 4-bit lúc sinh. Gắn adapter QDoRA lên base
     bf16 vẫn nạp được và vẫn ra văn bản đọc được — chỉ là model được chấm
     không phải model đã huấn luyện. Nguồn sự thật là `train_stats.json`, không
     phải config hiện tại (config có thể đã bị `--set` sửa giữa hai stage).
  2. Candidate B mà thiếu file nhánh thì DỪNG. Nạp adapter đúng với một nhánh
     ngẫu nhiên vẫn sinh ra tiếng Anh trôi chảy, và cột số ấy trông y như kết
     quả thật.
  3. Đệm BÊN TRÁI khi sinh (ngược với khâu train). Hệ quả: vị trí token ảnh
     khác nhau theo từng hàng, nên chỗ chèn token nhánh cũng phải theo từng hàng —
     xem `recombiner.image_start_indices`.

── §2.4: cùng runner, thêm Experts ─────────────────────────────────────────
Candidate của §2.4 KHÔNG có adapter peft — nó có E bộ LoRA nằm trong
`MoELoraLinear` và một file `moe_experts.safetensors`. Runner dựng lại đúng
cấu hình ấy rồi nạp trọng số vào, và ba chỗ nữa phải đúng:

  4. Router ở chế độ `eval` -> KHÔNG có nhiễu Gauss. Nhiễu là cơ chế thăm dò
     lúc học; để nó chạy lúc sinh là để cùng một ảnh ra hai báo cáo khác nhau ở
     hai lần chạy, và §2.5 chấm bằng beam search tất định chính vì muốn con số
     lặp lại được.
  5. Trọng số định tuyến phải nằm trong `RoutingContext` TRƯỚC `generate`, và
     bị xoá NGAY SAU. `strict=True` biến "quên set" thành một exception thay vì
     một model lặng lẽ chạy nền trần — vẫn ra tiếng Anh đọc được.
  6. Beam search (`generate.num_beams`, §2.5 = 3) nhân mỗi ảnh thành k hàng
     liền nhau, nên trọng số định tuyến được đặt với `repeat = k` và nhân bản
     theo đúng thứ tự đó. `num_return_sequences` phải là 1.
     `RoutingContext.get` kiểm số hàng và DỪNG nếu lệch.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .. import imaging, moe, recombiner
from ..config import hf_token
from ..prompts import chat_messages
from .base import RunnerUnusable
from .gemma3_adapter import Gemma3AdapterRunner


class Gemma3RecombRunner(Gemma3AdapterRunner):
    name = "gemma3_recomb"

    branch = None
    _gemma = None
    _img_token_id = None
    _tt_value = 0
    _router = None
    _route_ctx = None

    def load(self) -> None:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        adapter_dir = Path(str(self.mcfg["adapter_dir"]))
        stats_path = adapter_dir / "train_stats.json"
        if not adapter_dir.exists():
            raise RunnerUnusable(
                f"không thấy adapter ở {adapter_dir}. Chạy stage `train` trước, "
                f"hoặc `--stages train,infer,...`.")
        if not stats_path.exists():
            raise RunnerUnusable(
                f"{adapter_dir} thiếu train_stats.json — adapter ghi dở. Xoá thư "
                f"mục đó rồi chạy lại stage `train`.")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

        quantized = bool(stats.get("quantized", False))
        uses_branch = bool(stats.get("uses_recombiner", False))
        uses_moe = bool(stats.get("uses_moe", False))
        hf_id = self.mcfg["hf_id"]
        tok = hf_token()
        dt = self.torch_dtype()
        from ..backbone import device_map_for, quant_config

        kw = dict(token=tok, torch_dtype=dt,
                  device_map=device_map_for(self.cfg, for_training=False),
                  attn_implementation=str(self.mcfg.get("attn_implementation", "sdpa")))
        if quantized:
            kw["quantization_config"] = quant_config(dict(self.cfg.get("train", {})))
            print(f"  nạp {hf_id} ở 4-bit (preset {stats.get('lora_preset')} huấn "
                  f"luyện trên base lượng tử)", flush=True)
        else:
            print(f"  nạp {hf_id} ở {dt}", flush=True)

        self.processor = AutoProcessor.from_pretrained(hf_id, token=tok)
        # padding_side=left cho khâu SINH (ngược với khâu train). Với padding
        # phải, pad token nằm SAU prompt và model sinh tiếp từ pad -> đầu ra bắt
        # đầu bằng rác. Lỗi này không ném exception, nó chỉ làm điểm tụt.
        self.processor.tokenizer.padding_side = "left"
        self._apply_image_policy()

        # Chính sách ảnh lúc sinh phải TRÙNG lúc huấn luyện.
        want = imaging.policy(self.cfg)
        for key, now in (("image_max_side",
                          int((self.cfg.get("data", {}) or {}).get("image_max_side", 512))),
                         ("do_pan_and_scan", bool(want["do_pan_and_scan"]))):
            was = stats.get(key)
            if was is not None and was != now:
                raise RunnerUnusable(
                    f"candidate này huấn luyện với {key}={was}, còn cấu hình lúc "
                    f"sinh là {now}. Sinh với ảnh khác lúc train là chấm một model "
                    f"khác model đã huấn luyện.\n"
                    f"    Sửa config cho khớp, hoặc xoá {adapter_dir} rồi huấn "
                    f"luyện lại candidate này.")

        base = Gemma3ForConditionalGeneration.from_pretrained(hf_id, **kw)
        if uses_moe:
            self.model = self._attach_moe(base, stats, adapter_dir).eval()
        else:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
        self._gemma = recombiner.unwrap_gemma(self.model)
        self._img_token_id = recombiner.image_token_id(self._gemma)
        self._tt_value = int(stats.get("token_type_value") or 0)

        if uses_branch:
            if not recombiner.branch_exists(adapter_dir):
                raise RunnerUnusable(
                    f"train_stats.json nói candidate này CÓ nhánh Recombinator, "
                    f"nhưng {adapter_dir} không có file nhánh.\n"
                    f"    Sinh tiếp mà không có nhánh là chấm candidate A rồi dán "
                    f"nhãn candidate B lên kết quả — hai cột trong bảng §2.3 khi "
                    f"đó là cùng một model. Huấn luyện lại candidate này.")
            dev = next(p for p in self.model.parameters()
                       if p.device.type != "meta").device
            self.branch = recombiner.load_branch(
                adapter_dir,
                rcfg=dict(self.cfg.get("recombiner", {}) or {}),
                ccfg=dict(self.cfg.get("classifier", {}) or {}),
                device=dev, dtype=recombiner.float_dtype(self._gemma))
            n = self.branch.n_params()
            d = self.branch.dec_stats()
            print(f"  ✓ nhánh: {self.branch.describe()} · {n['total']:,} tham "
                  f"số · route {'BẬT' if n['route_enabled'] else 'TẮT'} · "
                  f"dec_proj rời khởi tạo {d['dec_proj_rel_change']:.3f}",
                  flush=True)
        else:
            self.branch = None
            print("  candidate A: không nạp nhánh Recombinator", flush=True)

        print(f"  ✓ {self.mcfg['id']}: {hf_id} + {adapter_dir.name} "
              f"(preset={stats.get('lora_preset')}, r={stats.get('rank')}, "
              f"alpha={stats.get('alpha')}, dora={stats.get('dora')}, "
              f"4bit={quantized}, recombinator={uses_branch}, moe={uses_moe})",
              flush=True)

    # ── §2.4 ───────────────────────────────────────────────────────────────
    def _attach_moe(self, base, stats: dict, adapter_dir: Path):
        """Dựng lại Experts + Router THEO `train_stats.json`, rồi nạp trọng số.

        Hình dạng lấy từ file thống kê của lần huấn luyện, KHÔNG từ config hiện
        tại — cùng lý lẽ với `recombiner.load_branch`. Config có thể đã bị
        `--set` sửa giữa stage train và stage infer, và khi đó tốt nhất là
        `load_experts` báo lệch shape; tệ nhất là chỉ lệch `top_k` (không đổi
        hình dạng tensor nào) và candidate `K=2·E=8` được chấm bằng `K=6`.
        """
        from ..backbone import find_target_modules

        m = dict(stats.get("moe") or {})
        n_e = int(m.get("n_experts") or stats.get("n_experts") or 0)
        k = int(m.get("top_k") or stats.get("top_k") or 0)
        if n_e < 1 or k < 1:
            raise RunnerUnusable(
                f"{adapter_dir}/train_stats.json nói candidate này CÓ MoE nhưng "
                f"không ghi top_k/n_experts. Huấn luyện lại candidate này.")
        targets = find_target_modules(base, dict(self.cfg.get("train", {}) or {}))
        ctx, st = moe.attach_experts(
            base, targets, n_e, int(m["rank"]), float(m["alpha"]),
            0.0)                      # dropout = 0 khi sinh, luôn luôn
        meta = moe.load_experts(base, adapter_dir)
        router = moe.Router(n_experts=n_e, top_k=k,
                            clip_c=float(m.get("logit_clip_c", 1.2)),
                            noise=bool(m.get("noise", True)),
                            renormalize=bool(m.get("renormalize_topk", True)))
        # eval() -> Router.training = False -> KHÔNG có nhiễu Gauss. Đây là chỗ
        # tính chất ấy được bảo đảm, và selftest kiểm nó.
        router.eval()
        self._router, self._route_ctx = router, ctx
        print(f"  ✓ Experts: {n_e} × LoRA(r={m['rank']}) trên "
              f"{st['n_modules']} ma trận Attn · top-k = {k} · "
              f"{st['expert_params']:,} tham số · nạp {meta.get('n_tensors')} "
              f"tensor từ {adapter_dir.name}", flush=True)
        print(f"  ✓ Router: soft clip C={m.get('logit_clip_c', 1.2)} -> softmax "
              f"-> top-{k} · nhiễu TẮT khi sinh (beam search phải lặp lại được)"
              f" · trọng số × {self._n_beams()} theo beam", flush=True)
        if int(self.gen_kwargs().get("num_return_sequences", 1) or 1) != 1:
            raise RunnerUnusable(
                "generate.num_return_sequences != 1 với MoE: mỗi ảnh chỉ được "
                "trả một báo cáo, và định tuyến chỉ nhân bản theo num_beams.")
        return base

    def _n_beams(self) -> int:
        return max(1, int(self.gen_kwargs().get("num_beams", 1) or 1))

    def generate_batch(self, batch: list[dict]) -> list[str]:
        """Sinh cho một batch, đi ĐÚNG con đường của khâu huấn luyện."""
        import torch

        msgs = [chat_messages(self.cfg, Image.open(it["image_path"]).convert("RGB"))
                for it in batch]
        inputs = self.processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True, **self.img_kw,
        ).to(self.model.device)

        ids = inputs["input_ids"]
        with torch.inference_mode():
            vis, aligned = recombiner.encode_vision(
                self._gemma,
                inputs["pixel_values"].to(
                    dtype=recombiner.float_dtype(self._gemma, torch.float32)))
            embeds = recombiner.build_inputs_embeds(self._gemma, ids, aligned)
            kw = {"inputs_embeds": embeds,
                  "attention_mask": inputs.get("attention_mask"),
                  "token_type_ids": inputs.get("token_type_ids")}
            if self.branch is not None:
                tokens, _logits, route = self.branch(
                    vis.to(next(self.branch.parameters()).dtype))
                if self._router is not None:
                    if route is None:
                        raise RunnerUnusable(
                            "candidate này có Experts nhưng nhánh trả route = "
                            "None — nhánh đã ghi ở chế độ §2.3 (route_enabled = "
                            "false). Huấn luyện lại candidate này.")
                    # generate nhân mỗi ảnh thành num_beams hàng liền nhau.
                    self._route_ctx.set(
                        self._router(route[0], route[1])["weights"],
                        repeat=self._n_beams())
                # Đệm TRÁI -> mỗi hàng có vị trí token ảnh riêng.
                at = recombiner.image_start_indices(ids, self._img_token_id)
                kw.update(recombiner.inject_tokens(
                    embeds, tokens, at,
                    attention_mask=inputs.get("attention_mask"),
                    token_type_ids=inputs.get("token_type_ids"),
                    token_type_value=self._tt_value))
            kw = {k: v for k, v in kw.items() if v is not None}
            in_len = int(kw["inputs_embeds"].shape[1])
            try:
                out = self.model.generate(**kw, **self.gen_kwargs())
            finally:
                # Xoá NGAY: batch là hằng số, nên một tensor sót lại từ batch
                # trước có đúng số hàng của batch sau và phép kiểm shape KHÔNG
                # bắt được — vài ảnh sẽ được định tuyến bằng expert của ảnh khác.
                if self._route_ctx is not None:
                    self._route_ctx.clear()

        # Với `inputs_embeds` (và không `input_ids`), transformers trả về CHỈ
        # phần sinh mới. Nhưng hành vi đó đã đổi giữa các bản, và đoán sai theo
        # hướng nào cũng tệ: giữ nguyên khi lẽ ra phải cắt thì mọi "báo cáo" mở
        # đầu bằng nguyên văn prompt và BLEU bị thổi lên; cắt khi lẽ ra không
        # thì mất phần đầu báo cáo. Nên KIỂM theo độ dài thay vì tin một bản.
        gen = out[:, in_len:] if out.shape[1] > in_len else out
        return [t.strip() for t in
                self.processor.batch_decode(gen, skip_special_tokens=True)]

    def unload(self) -> None:
        self.branch = None
        self._gemma = None
        self._router = None
        self._route_ctx = None
        super().unload()

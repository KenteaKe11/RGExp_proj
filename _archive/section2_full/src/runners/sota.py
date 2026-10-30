"""Runner sinh cho hai baseline đối chứng — R2Gen và AdaAtt.

MỘT file cho cả hai, có chủ ý: chúng khác nhau đúng một dòng (tên mô hình đưa
cho `sota.build`), và mọi thứ còn lại — nạp checkpoint, nạp từ điển, dựng tensor
ảnh theo `image_policy`, gọi beam search dùng chung — phải giống hệt nhau.

Giải mã đọc `decoding_spec.sota_kwargs(cfg)` — CÙNG khối `generate:` mà My
Gemma đưa cho `transformers.generate`, không phải `sota.*.decoding` của paper.
Hai file nghĩa là hai chỗ để lệch, và một lệch ở đây (ví dụ một bên quên
`model.eval()`) cho ra hai cột số trông bình thường.

Ảnh đi qua `imaging.unified_tensor` — ĐÚNG hàm mà `train_sota` dùng. Sinh bằng
một phép tiền xử lý khác phép đã huấn luyện là loại lỗi làm điểm tụt đều trên
mọi ảnh mà không có gì báo.
"""
from __future__ import annotations

from pathlib import Path

from .base import Runner, RunnerUnusable


class SotaRunner(Runner):
    """Nạp `model_best.pt` + `tokenizer.json`, sinh bằng đặc tả giải mã chung."""

    name = "sota"
    setup = ""

    def load(self) -> None:
        import torch

        from ..sota import build as build_sota
        from ..sota.tokenizer import WordTokenizer

        ckpt = Path(str(self.mcfg.get("ckpt", "")))
        if not ckpt.exists():
            raise RunnerUnusable(
                f"không thấy checkpoint {ckpt}. Candidate này chưa huấn luyện "
                f"xong — chạy `--stages train` trước. (Sinh bằng trọng số ngẫu "
                f"nhiên vẫn ra tiếng Anh đọc được và một cột điểm trông thật.)")
        tok_path = Path(str(self.mcfg.get("tokenizer", "")))
        if not tok_path.exists():
            raise RunnerUnusable(
                f"không thấy từ điển {tok_path}. Checkpoint và từ điển phải đi "
                f"cùng nhau: id của từ do từ điển quyết định, nên một checkpoint "
                f"giải mã bằng từ điển khác sinh ra văn bản trôi chảy và sai "
                f"nghĩa hoàn toàn.")
        self.tok = WordTokenizer.load(tok_path)
        from ..decoding_spec import DecodingSpecError, sota_kwargs

        try:
            self.dcfg = sota_kwargs(self.cfg)
        except DecodingSpecError as e:
            raise RunnerUnusable(str(e)) from e

        scfg = dict(self.cfg.get("sota", {}) or {})
        model = build_sota(self.setup, len(self.tok), scfg)
        state = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(
            state.get("model", state), strict=False)
        if missing or unexpected:
            # `strict=False` để một khoá phụ (buffer của bản khác) không giết cả
            # lần chạy — nhưng THIẾU trọng số thật thì phải dừng: nó nghĩa là
            # một phần mô hình đang ở trạng thái ngẫu nhiên.
            real = [k for k in missing if not k.endswith("memory_init")
                    and ".pe" not in k]
            if real:
                raise RunnerUnusable(
                    f"checkpoint thiếu {len(real)} trọng số, ví dụ {real[:3]}. "
                    f"Nạp tiếp nghĩa là một phần mô hình chạy bằng khởi tạo "
                    f"ngẫu nhiên.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(device).eval()
        self.device = device
        ep = state.get("epoch")
        mon = state.get("monitor")
        sc = state.get("score")
        print(f"  ✓ {self.setup}: nạp {ckpt.name}"
              + (f" (epoch {ep}" + (f", {mon}={sc:.4f}" if sc is not None else "")
                 + ")" if ep is not None else "")
              + f" · từ điển {len(self.tok)} từ · giải mã chung: beam "
              f"{self.dcfg['num_beams']}, lp {self.dcfg['length_penalty']:g}, "
              f"tối đa {self.tok.max_seq_length - 1} token", flush=True)

    def generate_batch(self, batch: list[dict]) -> list[str]:
        import torch

        from .. import imaging

        px = imaging.unified_tensor([b["image_path"] for b in batch], self.cfg)
        px = px.to(self.device)
        with torch.no_grad():
            return self.model.generate(px, self.tok, self.dcfg)

    def configure_compile(self) -> None:
        """Không có gì để compile.

        Đường sinh ở đây là vòng `for` python trên các bước beam, không phải
        `generate()` của transformers — thứ mà `Runner.configure_compile` sinh
        ra để thuần hoá. Gọi bản gốc chỉ tổ in một dòng về gcc không liên quan
        tới hai cột này.
        """
        return


class R2GenRunner(SotaRunner):
    name = "r2gen"
    setup = "r2gen"


class AdaAttRunner(SotaRunner):
    name = "adaatt"
    setup = "adaatt"

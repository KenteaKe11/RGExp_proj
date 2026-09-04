"""LLaVA-Med v1.5 7B — chạy Ở MỘT VENV KHÁC, giao tiếp qua JSONL.

Vì sao không nạp thẳng trong tiến trình này: microsoft/llava-med-v1.5-mistral-7b
khai `model_type: llava_mistral` — transformers không biết kiến trúc đó, phải có
package `llava` của repo microsoft/LLaVA-Med, mà repo đó ghim transformers 4.36.x
trong khi Gemma 3 đòi >= 4.50. Hai ràng buộc loại trừ nhau trong cùng một
sys.path.

Nên: một venv riêng (.venv-llavamed, dựng bằng --system-site-packages để mượn
torch), một tiến trình con, và một file JSONL đúng schema của các runner khác.
Khâu chấm điểm không cần biết sự sắp xếp này.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ..prompts import flat_prompt
from .base import Runner

ROOT = Path(__file__).resolve().parent.parent.parent
WORKER = ROOT / "scripts" / "infer_llavamed.py"


class LlavaMedRunner(Runner):
    name = "llavamed"

    def _venv_python(self) -> Path | None:
        venv = ROOT / str(self.mcfg.get("venv", ".venv-llavamed"))
        for cand in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
            if cand.exists():
                return cand
        return None

    def load(self) -> None:  # model được nạp trong tiến trình con
        return None

    def generate_batch(self, batch: list[dict]) -> list[str]:
        raise NotImplementedError("LlavaMedRunner ghi đè run(), không dùng generate_batch")

    def run(self, df: pd.DataFrame, out_path: Path, log_every: int = 20) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        py = self._venv_python()
        if py is None:
            msg = (f"không thấy venv {self.mcfg.get('venv', '.venv-llavamed')} "
                   f"— chạy `bash scripts/setup.sh` (hoặc INSTALL_LLAVAMED=1)")
            if self.mcfg.get("optional", False):
                print(f"  ⚠ bỏ qua {self.mcfg['id']}: {msg}", flush=True)
                return out_path
            raise RuntimeError(msg)

        done: set[str] = set()
        if out_path.exists():
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done.add(json.loads(line)["key"])
                    except Exception:
                        continue
        todo = [r for r in df.to_dict("records") if r["key"] not in done]
        if not todo:
            print(f"  ✓ {self.name}: đã đủ {len(done)} mẫu, bỏ qua", flush=True)
            return out_path

        tasks = out_path.with_name(out_path.stem + "_tasks.jsonl")
        with open(tasks, "w", encoding="utf-8") as f:
            for r in todo:
                f.write(json.dumps({"key": r["key"], "image_path": r["image_path"]},
                                   ensure_ascii=False) + "\n")

        g = self.gen_kwargs()
        cmd = [
            str(py), str(WORKER),
            "--tasks", str(tasks),
            "--out", str(out_path),
            "--model-path", str(self.mcfg["hf_id"]),
            "--model-id", str(self.mcfg["id"]),
            "--prompt", flat_prompt(self.cfg),
            "--conv-mode", str(self.mcfg.get("conv_mode", "mistral_instruct")),
            "--max-new-tokens", str(g.get("max_new_tokens", 256)),
            "--dtype", str(self.mcfg.get("dtype", "bfloat16")),
            "--log-every", str(max(1, log_every)),
        ]
        print(f"  → tiến trình con: {py}", flush=True)
        print(f"    {len(todo)} mẫu cần sinh", flush=True)
        # stdout của tiến trình con đi thẳng ra console cha (không capture):
        # đây là khâu chạy lâu nhất, và một thanh tiến độ bị nuốt mất nghĩa là
        # ngồi nhìn màn hình đen vài giờ không biết nó còn sống hay không.
        rc = subprocess.call(cmd, cwd=str(ROOT), stdout=sys.stdout, stderr=sys.stderr)
        if rc != 0:
            msg = f"tiến trình LLaVA-Med thoát với mã {rc}"
            if self.mcfg.get("optional", False):
                n = sum(1 for _ in open(out_path, encoding="utf-8")) if out_path.exists() else 0
                print(f"  ⚠ {msg} — giữ {n} mẫu đã sinh, báo cáo sẽ ghi rõ tình trạng",
                      flush=True)
                return out_path
            raise RuntimeError(msg)
        print(f"✓ {self.name}: xong -> {out_path}", flush=True)
        return out_path

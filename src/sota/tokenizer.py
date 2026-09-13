"""Từ điển mức TỪ dùng chung cho R2Gen và AdaAtt.

Cả hai paper gốc đều dùng từ điển mức từ dựng TỪ TẬP HUẤN LUYỆN, không phải một
tokenizer con từ đã pretrain. Giữ đúng như vậy là một phần của điều kiện 1
("chỉ nạp lại trọng số ở đúng trạng thái setup của paper gốc"): thay nó bằng
BPE của Gemma là cho hai baseline một lợi thế mà paper của chúng không có, và
bảng so sánh khi ấy đo một thứ khác.

MỘT từ điển cho CẢ HAI. `sota.r2gen.tokenizer` là khối duy nhất khai tham số
(`max_seq_length: 60`, `min_word_frequency: 3`); AdaAtt không khai gì về từ
điển trong setup gốc, nên nó dùng lại đúng từ điển ấy thay vì dựng một bản thứ
hai với ngưỡng khác. Hai từ điển khác nhau nghĩa là hai không gian nhãn khác
nhau, và khi đó "cùng 6400 mẫu train" không còn là cùng một bài toán.

Từ điển được ghi ra `<run_dir>/tokenizer.json` và ĐỌC LẠI ở mọi stage sau. Dựng
lại từ train_df mỗi lần là mời một lỗi im lặng: chỉ cần thứ tự hàng đổi (hay
`n_train` đổi) là id của từ đổi theo, và một checkpoint cũ nạp vào từ điển mới
sinh ra tiếng Anh đọc được với nghĩa hoàn toàn khác.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# Chuẩn hoá nhẹ, giống tiền xử lý của R2Gen: hạ chữ, bỏ ký tự không phải chữ/số
# (giữ dấu chấm câu tách câu), gộp khoảng trắng. KHÔNG stem, KHÔNG bỏ stopword —
# đây là văn bản đích của mô hình sinh, không phải đầu vào của một bộ truy hồi.
_CLEAN = re.compile(r"[^a-z0-9.,;:?!/\-\s]+")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"([.,;:?!])")

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


def clean(text: str) -> str:
    t = str(text or "").lower()
    t = _CLEAN.sub(" ", t)
    t = _PUNCT.sub(r" \1 ", t)
    return _WS.sub(" ", t).strip()


def tokenize(text: str) -> list[str]:
    return clean(text).split()


class WordTokenizer:
    """id 0 = <pad>, 1 = <bos>, 2 = <eos>, 3 = <unk>.

    `<pad>` phải là 0 vì cả hai mô hình dùng `ignore_index=0` ở CrossEntropy và
    `padding_idx=0` ở lớp nhúng — đặt nó ở một id khác là để loss đếm cả phần
    đệm, và loss vẫn giảm đẹp trong khi mô hình học cách sinh ra khoảng trắng.
    """

    def __init__(self, itos: list[str], max_seq_length: int = 60,
                 min_word_frequency: int = 3):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.max_seq_length = int(max_seq_length)
        self.min_word_frequency = int(min_word_frequency)

    # ── dựng / ghi / đọc ───────────────────────────────────────────────────
    @classmethod
    def build(cls, texts, max_seq_length: int = 60,
              min_word_frequency: int = 3) -> "WordTokenizer":
        cnt: Counter = Counter()
        for t in texts:
            cnt.update(tokenize(t))
        # sort theo (-tần suất, từ): tần suất bằng nhau thì thứ tự chữ cái quyết
        # định, nên từ điển KHÔNG phụ thuộc thứ tự hàng của train_df.
        kept = sorted((w for w, c in cnt.items() if c >= int(min_word_frequency)),
                      key=lambda w: (-cnt[w], w))
        return cls(SPECIALS + kept, max_seq_length, min_word_frequency)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "itos": self.itos,
            "max_seq_length": self.max_seq_length,
            "min_word_frequency": self.min_word_frequency,
        }, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["itos"], d.get("max_seq_length", 60),
                   d.get("min_word_frequency", 3))

    # ── mã hoá / giải mã ───────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def unk_id(self) -> int:
        return 3

    def encode(self, text: str) -> list[int]:
        """<bos> … <eos>, CẮT về đúng max_seq_length kể cả hai token đặc biệt.

        `max_seq_length: 60` của R2Gen là độ dài CHUỖI, không phải số từ cộng
        thêm hai token. Hiểu nhầm chỗ này làm chuỗi dài ra 2 token ở mọi mẫu và
        một phần báo cáo dài bị cắt ở chỗ khác với paper.
        """
        ids = [self.bos_id]
        for w in tokenize(text)[: max(0, self.max_seq_length - 2)]:
            ids.append(self.stoi.get(w, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in (self.pad_id, self.bos_id):
                continue
            if i == self.eos_id:
                break
            out.append(self.itos[i] if 0 <= i < len(self.itos) else UNK)
        # Dán lại dấu câu vào từ đứng trước để văn bản sinh ra đọc được như một
        # báo cáo, không phải một chuỗi token. Khâu chấm điểm chuẩn hoá lại lần
        # nữa (metrics.normalize), nên đây chỉ để người đọc.
        s = " ".join(out)
        return _WS.sub(" ", re.sub(r"\s+([.,;:?!])", r"\1", s)).strip()

    def stats(self) -> dict:
        return {"vocab_size": len(self.itos),
                "max_seq_length": self.max_seq_length,
                "min_word_frequency": self.min_word_frequency,
                "n_specials": len(SPECIALS)}


def build_or_load(run_dir: Path, texts, tcfg: dict) -> WordTokenizer:
    """Đọc `<run_dir>/tokenizer.json` nếu có, không thì dựng rồi ghi.

    Thứ tự này (đọc trước, dựng sau) là thứ giữ cho stage `infer` chạy lại sau
    nhiều ngày vẫn giải mã đúng checkpoint đã huấn luyện.
    """
    path = Path(run_dir) / "tokenizer.json"
    if path.exists():
        return WordTokenizer.load(path)
    tok = WordTokenizer.build(
        texts,
        max_seq_length=int(tcfg.get("max_seq_length", 60)),
        min_word_frequency=int(tcfg.get("min_word_frequency", 3)))
    tok.save(path)
    print(f"  ⚙ từ điển: {len(tok)} từ (ngưỡng tần suất "
          f"{tok.min_word_frequency}, dài tối đa {tok.max_seq_length}) -> {path}",
          flush=True)
    return tok

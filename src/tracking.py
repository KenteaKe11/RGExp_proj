"""Theo dõi lần chạy bằng Weights & Biases.

Vì sao cần: một lần chạy §2.1 là nhiều giờ trên máy thuê. Nhìn nó qua tmux
buộc phải giữ một phiên ssh mở và ngồi cạnh máy; mất mạng là mất luôn tầm nhìn
vào tiến độ, và nếu quá trình chết ở giờ thứ ba thì không có gì để đọc lại. Đẩy
tiến độ lên W&B thì xem được từ điện thoại, và số liệu ở lại sau khi phiên kết
thúc.

Lớp này là VỎ BỌC, không phải phụ thuộc cứng: thiếu `wandb`, thiếu API key, hay
init hỏng thì mọi lời gọi thành no-op và pipeline chạy y nguyên. Một công cụ
theo dõi không bao giờ được phép là lý do làm hỏng thứ nó đang theo dõi.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class Tracker:
    def __init__(self, cfg: dict | None = None, run_id: str = "run",
                 config_snapshot: dict | None = None):
        self.run = None
        self.wandb = None
        self._url = None
        wcfg = dict((cfg or {}).get("wandb", {}) or {})
        self.cfg = wcfg
        self._log_every = int(wcfg.get("log_every", 20))

        mode = str(wcfg.get("mode", "online")).lower()
        if not wcfg.get("enabled", True):
            print("⊘ W&B: tắt (wandb.enabled=false)", flush=True)
            return
        if mode == "disabled":
            print("⊘ W&B: tắt (wandb.mode=disabled)", flush=True)
            return
        try:
            import wandb
        except ImportError:
            print("⚠ W&B: chưa cài `wandb` — chạy tiếp, không theo dõi.\n"
                  "    pip install wandb", flush=True)
            return

        has_key = bool(os.environ.get("WANDB_API_KEY")) or \
            Path(os.path.expanduser("~/.netrc")).exists()
        if mode == "online" and not has_key:
            # Chuyển sang offline thay vì tắt: dữ liệu vẫn được ghi xuống đĩa và
            # `wandb sync` đẩy lên sau được. Mất tiền GPU rồi mới phát hiện
            # không có key thì ít nhất số liệu không mất theo.
            print("⚠ W&B: không thấy WANDB_API_KEY -> ghi OFFLINE vào ./wandb.\n"
                  "    Đẩy lên sau bằng:  wandb sync wandb/latest-run\n"
                  "    Hoặc đặt trước:    export WANDB_API_KEY=xxxxxxxx", flush=True)
            mode = "offline"

        # MỖI LẦN GỌI = MỘT RUN MỚI, gom lại bằng `group`.
        #
        # Bản trước dùng id ổn định + resume="allow" để lần chạy lại nối tiếp
        # đúng run cũ. Nghe hợp lý, nhưng thực tế của việc dò lỗi là chạy đi
        # chạy lại nhiều lần với cấu hình khác nhau — và resume thì trộn số
        # liệu của lần HỎNG với lần TỐT vào cùng một biểu đồ, không tách ra
        # được nữa. Group cho cả hai thứ: mỗi lần chạy một run riêng để so, mà
        # vẫn nằm chung một chỗ trên dashboard.
        stamp = datetime.now().strftime("%m-%d %H:%M")
        kw: dict = {}
        if wcfg.get("resume_same_run", False):
            kw = {"id": hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:16],
                  "resume": "allow"}
        try:
            self.run = wandb.init(
                project=str(wcfg.get("project", "rgexp-basemodel")),
                entity=wcfg.get("entity") or None,
                name=wcfg.get("run_name") or f"{run_id} · {stamp}",
                group=str(wcfg.get("group") or run_id),
                mode=mode,
                config=config_snapshot or {},
                tags=list(wcfg.get("tags", []) or []),
                notes=str(wcfg.get("notes", "") or ""),
                **kw,
            )
            self.wandb = wandb
            self._url = getattr(self.run, "url", None)
            print(f"✓ W&B: {self._url or f'(offline) mode={mode}'}", flush=True)
        except Exception as e:
            print(f"⚠ W&B init hỏng ({e!r}) — chạy tiếp, không theo dõi.", flush=True)
            self.run = None

    # ── trạng thái ──────────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self.run is not None

    @property
    def url(self) -> str | None:
        return self._url

    @property
    def log_every(self) -> int:
        return self._log_every

    # ── ghi ─────────────────────────────────────────────────────────────────
    def log(self, data: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.log(data)
        except Exception:
            pass   # theo dõi hỏng không được làm hỏng lần chạy

    def summary(self, data: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            for k, v in data.items():
                self.run.summary[k] = v
        except Exception:
            pass

    def table(self, name: str, df, max_rows: int = 500) -> None:
        if self.run is None or df is None or len(df) == 0:
            return
        try:
            self.run.log({name: self.wandb.Table(dataframe=df.head(max_rows))})
        except Exception:
            pass

    def samples(self, name: str, rows: list[dict]) -> None:
        """Bảng văn bản: xem tận mắt model viết ra cái gì, ngay khi đang chạy.

        Quan trọng hơn vẻ ngoài của nó: BLEU 0.02 có thể là "model viết sai" mà
        cũng có thể là "model trả lời bằng một câu từ chối" hay "prompt bị dính
        vào output". Ba ca đó cần ba cách sửa khác hẳn nhau, và chỉ nhìn số thì
        không phân biệt được.
        """
        if self.run is None or not rows:
            return
        try:
            cols = list(rows[0].keys())
            t = self.wandb.Table(columns=cols)
            for r in rows:
                t.add_data(*[r.get(c) for c in cols])
            self.run.log({name: t})
        except Exception:
            pass

    def save_file(self, path: Path | str) -> None:
        if self.run is None:
            return
        try:
            p = Path(path)
            if p.exists():
                self.wandb.save(str(p), base_path=str(p.parent), policy="now")
        except Exception:
            pass

    def artifact(self, name: str, path: Path | str, kind: str = "results") -> None:
        if self.run is None:
            return
        try:
            art = self.wandb.Artifact(name=name, type=kind)
            p = Path(path)
            if p.is_dir():
                art.add_dir(str(p))
            else:
                art.add_file(str(p))
            self.run.log_artifact(art)
        except Exception:
            pass

    def alert(self, title: str, text: str, level: str = "INFO") -> None:
        """Báo động đẩy về email/điện thoại — cái tmux không làm được."""
        if self.run is None:
            return
        try:
            lv = getattr(self.wandb.AlertLevel, level, None)
            self.wandb.alert(title=title, text=text, level=lv)
        except Exception:
            pass

    def finish(self, exit_code: int = 0) -> None:
        if self.run is None:
            return
        try:
            self.run.finish(exit_code=exit_code)
        except Exception:
            pass


class NullTracker(Tracker):
    """Dùng trong selftest và khi chạy stage lẻ — không đụng tới wandb."""

    def __init__(self):  # noqa: D107  (cố ý không gọi super)
        self.run = None
        self.wandb = None
        self._url = None
        self.cfg = {}
        self._log_every = 20

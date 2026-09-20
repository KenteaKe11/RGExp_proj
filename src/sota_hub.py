"""Checkpoint của hai baseline §2.5 (R2Gen, AdaAtt) trên Hugging Face Hub.

`--mode comparer_pretrain` huấn luyện RIÊNG hai baseline rồi đẩy checkpoint lên
một repo MODEL (`sota_hub.repo`). Mọi lần chạy khác (mặc định, `infer_test`)
thấy thiếu checkpoint cục bộ thì KÉO về từ đó trước khi quyết định có phải
huấn luyện lại hay không. Như vậy hai baseline — vài giờ GPU, không đổi giữa
các lần sửa My Gemma — chỉ phải huấn luyện MỘT lần, trên bất kỳ máy nào.

── MỖI CANDIDATE LÀ MỘT GÓI BỐN FILE, Ở `<subdir>/<arm_id>/` ────────────────
    model_best.pt      trọng số đã chọn (runner nạp đúng file này)
    train_stats.json   `already_trained()` đòi nó; báo cáo đọc nó
    tokenizer.json     từ điển ĐÃ DÙNG để huấn luyện checkpoint này
    hub_meta.json      sha256 của ba file trên + chính sách ảnh lúc huấn luyện

Từ điển đi CÙNG checkpoint, không nằm chung ở run_dir: id của từ do từ điển
quyết định, và một checkpoint giải mã bằng từ điển khác sinh ra văn bản trôi
chảy mà sai nghĩa hoàn toàn — không có gì báo lỗi.

`last.pt` và `stage0.pt` KHÔNG lên: chúng chỉ phục vụ việc chạy tiếp của chính
lần huấn luyện ấy, và mỗi file nặng ngang `model_best.pt`.

Kéo về thì KIỂM hai thứ trước khi chép vào `adapters_2_5/<arm_id>/`:
  * sha256 khớp `hub_meta.json` — một lần đẩy đứt giữa chừng để lại checkpoint
    của lần này đi với từ điển của lần trước;
  * `image_policy` lúc huấn luyện == `image_policy` hiện tại (điều kiện 4) — một
    ResNet học ở 512px đem sinh ở 384px vẫn ra báo cáo đọc được, chỉ tụt điểm
    đều trên mọi ảnh.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

CKPT_BEST = "model_best.pt"
STATS = "train_stats.json"
TOKENIZER = "tokenizer.json"
META = "hub_meta.json"
BUNDLE = (CKPT_BEST, STATS, TOKENIZER)


class SotaHubError(RuntimeError):
    """Không đẩy/kéo được — checkpoint cục bộ (nếu có) vẫn còn nguyên."""


def hub_cfg(cfg) -> dict:
    h = dict(cfg.get("sota_hub", {}) or {})
    repo = str(h.get("repo", "") or "").strip()
    if repo and "/" not in repo:
        raise SotaHubError(f"sota_hub.repo phải có dạng owner/name, nhận {repo!r}")
    return {"repo": repo,
            "private": bool(h.get("private", True)),
            "subdir": str(h.get("subdir", "") or "").strip("/"),
            "revision": (str(h["revision"]) if h.get("revision") else None)}


def path_in_repo(cfg, aid: str) -> str:
    sub = hub_cfg(cfg)["subdir"]
    return f"{sub}/{aid}" if sub else str(aid)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_policy(cfg) -> dict:
    return json.loads(json.dumps(dict(cfg.get("image_policy", {}) or {}),
                                 default=str))


def bundle_tokenizer(odir: Path, run_dir: Path, overwrite: bool = False) -> Path:
    """Chép `run_dir/tokenizer.json` vào thư mục checkpoint.

    `overwrite=True` ngay sau khi VỪA huấn luyện (từ điển của run_dir chính là
    từ điển đã dùng). Ngược lại chỉ chép khi thư mục checkpoint chưa có — một
    gói kéo từ Hub đã mang từ điển của riêng nó.
    """
    odir, src = Path(odir), Path(run_dir) / TOKENIZER
    dst = odir / TOKENIZER
    if dst.exists() and not overwrite:
        return dst
    if not src.exists():
        raise SotaHubError(
            f"không thấy {src} để đóng gói cùng checkpoint {odir.name}. "
            f"Checkpoint không kèm từ điển thì không giải mã đúng được.")
    odir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def write_meta(cfg, arm: dict, odir: Path) -> dict:
    odir = Path(odir)
    stats = {}
    try:
        stats = json.loads((odir / STATS).read_text(encoding="utf-8"))
    except Exception:
        pass
    meta = {
        "format": "rgexp.sota_ckpt/1",
        "arm_id": arm["id"],
        "setup": arm.get("setup", arm["id"]),
        "files": {f: _sha256(odir / f) for f in BUNDLE},
        "image_policy": _image_policy(cfg),
        "n_train": int((cfg.get("data", {}) or {}).get("n_train", 0) or 0),
        "selection": stats.get("selection"),
        "vocab_size": stats.get("vocab_size"),
        "seed": cfg.get("seed"),
        "pushed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (odir / META).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return meta


def _card(cfg, arm: dict, meta: dict) -> str:
    return (f"# {arm.get('name', arm['id'])} — checkpoint §2.5\n\n"
            f"Baseline đối chứng của RGExp §2.5, huấn luyện bằng "
            f"`python -m src.run_section25 --mode comparer_pretrain`.\n\n"
            f"- setup: `{meta.get('setup')}` · chọn checkpoint: "
            f"`{meta.get('selection')}` · từ điển {meta.get('vocab_size')} từ\n"
            f"- image_policy: `{json.dumps(meta.get('image_policy'))}`\n"
            f"- đẩy lúc {meta.get('pushed_at')}\n\n"
            f"Chạy §2.5 với `sota_hub.repo` trỏ về repo này thì checkpoint được "
            f"kéo về tự động khi thiếu cục bộ.\n")


def push_arm(cfg, arm: dict, odir: Path, run_dir: Path) -> str | None:
    """Đẩy gói bốn file của MỘT baseline lên `sota_hub.repo/<subdir>/<id>/`."""
    from .config import hf_token

    h = hub_cfg(cfg)
    aid = arm["id"]
    if not h["repo"]:
        print(f"  ⊘ [{aid}] sota_hub.repo trống — không đẩy checkpoint", flush=True)
        return None
    odir = Path(odir)
    missing = [f for f in (CKPT_BEST, STATS) if not (odir / f).exists()]
    if missing:
        raise SotaHubError(f"[{aid}] {odir} thiếu {missing} — chưa huấn luyện xong, "
                           f"không có gì để đẩy.")
    tok = hf_token()
    if not tok:
        print(f"  ⚠ [{aid}] không có HF_TOKEN — bỏ qua đẩy. Checkpoint vẫn ở {odir}",
              flush=True)
        return None
    bundle_tokenizer(odir, run_dir)
    meta = write_meta(cfg, arm, odir)
    (odir / "README.md").write_text(_card(cfg, arm, meta), encoding="utf-8")

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SotaHubError(f"thiếu huggingface_hub ({e})") from e
    api = HfApi(token=tok)
    dest = path_in_repo(cfg, aid)
    try:
        api.create_repo(repo_id=h["repo"], repo_type="model",
                        private=h["private"], exist_ok=True)
        print(f"  → [{aid}] đẩy {odir} -> {h['repo']}/{dest} (model)", flush=True)
        api.upload_folder(
            folder_path=str(odir), repo_id=h["repo"], repo_type="model",
            path_in_repo=dest,
            allow_patterns=list(BUNDLE) + [META, "README.md"],
            commit_message=f"§2.5 {aid}: {meta.get('selection')}")
    except Exception as e:
        raise SotaHubError(
            f"[{aid}] đẩy lên {h['repo']} hỏng: {e!r}\n"
            f"    Token phải có quyền WRITE vào `{h['repo'].split('/')[0]}`. "
            f"Checkpoint vẫn ở {odir}; đẩy lại: --mode comparer_pretrain "
            f"--stages push") from e
    url = f"https://huggingface.co/{h['repo']}/tree/main/{dest}"
    print(f"  ✓ [{aid}] đã đẩy: {url}", flush=True)
    return url


def pull_arm(cfg, arm: dict, odir: Path) -> dict:
    """Kéo gói của MỘT baseline về `odir`. Trả `hub_meta`. Ném SotaHubError."""
    from .config import hf_token

    h = hub_cfg(cfg)
    aid = arm["id"]
    if not h["repo"]:
        raise SotaHubError("sota_hub.repo trống — không có chỗ để kéo về")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SotaHubError(f"thiếu huggingface_hub ({e})") from e
    src_sub = path_in_repo(cfg, aid)
    try:
        local = snapshot_download(
            repo_id=h["repo"], repo_type="model", revision=h["revision"],
            token=hf_token(),
            cache_dir=str((cfg.get("data", {}) or {}).get("cache_dir") or "") or None,
            allow_patterns=[f"{src_sub}/{f}" for f in BUNDLE + (META,)])
    except Exception as e:
        raise SotaHubError(f"không tải được {h['repo']}/{src_sub}: {e!r}") from e
    src = Path(local) / src_sub
    return install_bundle(cfg, arm, src, odir)


def install_bundle(cfg, arm: dict, src: Path, odir: Path) -> dict:
    """Kiểm gói ở `src` rồi chép vào `odir`. Tách riêng để selftest không cần mạng."""
    aid = arm["id"]
    src, odir = Path(src), Path(odir)
    missing = [f for f in BUNDLE + (META,) if not (src / f).exists()]
    if missing:
        raise SotaHubError(
            f"[{aid}] gói trên Hub thiếu {missing}. Đẩy lại từ máy đã huấn "
            f"luyện: --mode comparer_pretrain --arms {aid}")
    meta = json.loads((src / META).read_text(encoding="utf-8"))
    bad = [f for f in BUNDLE if _sha256(src / f) != (meta.get("files") or {}).get(f)]
    if bad:
        raise SotaHubError(
            f"[{aid}] sha256 không khớp hub_meta.json ở {bad} — gói bị trộn giữa "
            f"hai lần đẩy. Đẩy lại: --mode comparer_pretrain --arms {aid}")
    want, got = _image_policy(cfg), meta.get("image_policy")
    if got != want:
        raise SotaHubError(
            f"[{aid}] checkpoint trên Hub huấn luyện với image_policy {got}, "
            f"config hiện tại là {want}. Sinh bằng một phép tiền xử lý khác "
            f"phép đã học là điểm tụt đều mà không có gì báo (điều kiện 4). "
            f"Huấn luyện lại baseline: --mode comparer_pretrain --retrain")
    want_n = int((cfg.get("data", {}) or {}).get("n_train", 0) or 0)
    got_n = meta.get("n_train")
    if got_n is not None and int(got_n) != want_n:
        raise SotaHubError(
            f"[{aid}] checkpoint trên Hub huấn luyện trên {got_n} mẫu, config "
            f"hiện tại là data.n_train={want_n} (điều kiện 2: CÙNG tập train cho "
            f"cả ba candidate). Kiểm `sota_hub.subdir`, hoặc huấn luyện lại: "
            f"--mode comparer_pretrain --retrain")
    odir.mkdir(parents=True, exist_ok=True)
    # `train_stats.json` chép SAU CÙNG: `already_trained()` đọc nó, nên một lần
    # chép đứt giữa chừng không được phép để lại một gói trông như đã xong.
    for f in (TOKENIZER, CKPT_BEST, META, STATS):
        tmp = odir / f"{f}.part"
        shutil.copyfile(src / f, tmp)
        tmp.replace(odir / f)
    return meta

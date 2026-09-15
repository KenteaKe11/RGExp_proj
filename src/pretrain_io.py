"""Ghi / nạp / đẩy checkpoint của chế độ `sup_pretrain` — và ĐÚNG những khối đã
thật sự được pretrain, không hơn.

── Vì sao file này tồn tại thay vì dùng lại `recombiner.save_branch` ────────
`save_branch` ghi TOÀN BỘ `state_dict()` của nhánh: recombinator, qformer,
cls_head, route_gate, gate. Ở chế độ pretrain, hai khoá cuối KHÔNG nhận một
gradient nào:

  * `gate` là cổng vào DECODER, và pretrain không chạy decoder. Nó ở lại đúng
    giá trị khởi tạo (0.0).
  * `route_gate` đọc `vec_2` của QFormer, mà `vec_2` bị đóng băng ở pretrain
    (`qformer.freeze = [aux]`) — query detach, đầu ra detach. Nó ở lại đúng
    giá trị `_init` + `init_noise`.

Ghi chúng vào file pretrain thì file ấy nói một câu không đúng: nó trông như
một checkpoint đầy đủ của nhánh. Ca hỏng cụ thể: ai đó nạp nó vào §2.4, thấy
`route_gate.*` có trong file, và kết luận rằng Route_Gate đã được pretrain —
trong khi nó là số ngẫu nhiên của một lần `torch.manual_seed` nào đó. Không có
exception nào, và đường loss của §2.4 trông hoàn toàn bình thường.

Nên: file pretrain chứa ĐÚNG `PRETRAINED_MODULES`, META khai luôn cả danh sách
khối KHÔNG được pretrain, và `load_into_branch` đối chiếu tập khoá theo CẢ HAI
chiều (thiếu VÀ thừa) trước khi nạp. Khối nào không có trong file thì được nói
ra bằng một dòng log ở lúc nạp, chứ không im lặng dùng giá trị khởi tạo.

── Vì sao HÌNH DẠNG nhánh vẫn là hình dạng của §2.3/§2.4 ───────────────────
Pretrain dựng `RecombinerBranch` với `qformer.tasks = [sup, aux]`,
`freeze = [aux]`, `route_enabled = false` — tức là đúng cấu hình §2.3. Chỉ MỘT
vector có gradient (`sup`), nên "QFormer 1 vector" là đúng theo nghĩa đường
thông tin; nhưng `query` vẫn là (2, d_model) và `route_gate` vẫn được dựng, nên
checkpoint NẠP THẲNG sang §2.4 được mà không cần một hàm mở rộng shape nào.

Một hàm mở rộng shape là chỗ tệ nhất để đặt một lỗi: nó chạy đúng ở mọi phép
kiểm hình dạng, và nếu nó chèn hàng `aux` vào SAI vị trí thì §2.4 định tuyến
bằng vector phân loại và phân loại bằng vector định tuyến — cả hai loss vẫn
giảm.

── Repo lưu trữ ────────────────────────────────────────────────────────────
`pretrain.hub_repo` là một repo MODEL riêng, không phải `output.hub_repo`
(dataset kết quả). Hai lý do, cả hai cơ học:

  1. `push.push_results` có `ignore_patterns` chặn `*.safetensors` — trọng số
     đẩy vào đó sẽ bị bỏ qua ÂM THẦM.
  2. Repo kết quả được ghi đè mỗi lần chạy lại một stage report; checkpoint
     pretrain thì phải sống lâu hơn mọi lần chạy §2.3/§2.4 dùng nó.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# Những khối THẬT SỰ nhận gradient ở chế độ sup_pretrain. Danh sách này là
# nguồn sự thật cho cả ghi lẫn nạp lẫn nhóm learning-rate của decoder_ft —
# `train_recomb.build_optimizer` đọc CHÍNH hằng số này, nên "nhóm lr giảm áp
# đúng vào phần đã pretrain" kiểm được bằng cách đọc chứ không phải bằng cách
# tin.
PRETRAINED_MODULES: tuple[str, ...] = ("pdvr", "qformer", "cls_head")

# Những khối có trong nhánh nhưng KHÔNG nhận gradient ở pretrain. Khai tường
# minh (thay vì "phần còn lại") để `save_pretrain` dừng khi nhánh mọc thêm một
# khối thứ sáu mà không ai quyết định nó thuộc nhóm nào.
UNTRAINED_MODULES: tuple[str, ...] = ("route_gate", "gate")

PRETRAIN_FILE = "sup_pretrain.safetensors"
PRETRAIN_META = "sup_pretrain.json"
TRAINER_STATE = "trainer_state.pt"      # chỉ để resume, KHÔNG bao giờ đẩy Hub
FORMAT = "rgexp.sup_pretrain/1"


class PretrainIncompatible(RuntimeError):
    """Checkpoint pretrain không khớp nhánh đang dựng. Dừng, đừng nạp một phần."""


# ══ Tách state_dict ═════════════════════════════════════════════════════════
def _module_of(key: str) -> str:
    """Tên khối của một khoá state_dict. `gate` là Parameter trần -> chính nó."""
    return key.split(".", 1)[0]


def split_state(branch) -> tuple[dict, dict]:
    """-> (phần được pretrain, phần KHÔNG được pretrain), theo tên khối.

    Dừng nếu gặp một khối không nằm trong cả hai danh sách: nhánh vừa mọc thêm
    một thứ và chưa ai quyết định nó có được pretrain hay không. Mặc định ngầm
    ở đây (rơi vào nhóm nào cũng được) là cách một khối mới lặng lẽ bị bỏ khỏi
    file pretrain — hoặc lặng lẽ được dán nhãn "đã pretrain".
    """
    keep: dict = {}
    drop: dict = {}
    unknown: set[str] = set()
    for k, v in branch.state_dict().items():
        m = _module_of(k)
        if m in PRETRAINED_MODULES:
            keep[k] = v
        elif m in UNTRAINED_MODULES:
            drop[k] = v
        else:
            unknown.add(m)
    if unknown:
        raise PretrainIncompatible(
            f"nhánh có khối {sorted(unknown)} không nằm trong "
            f"PRETRAINED_MODULES {list(PRETRAINED_MODULES)} lẫn "
            f"UNTRAINED_MODULES {list(UNTRAINED_MODULES)}.\n"
            f"    Ai đó vừa thêm một khối vào RecombinerBranch. Quyết định nó "
            f"có nhận gradient ở sup_pretrain hay không, rồi khai vào đúng một "
            f"trong hai danh sách ở src/pretrain_io.py — im lặng cho nó rơi vào "
            f"nhóm nào cũng được nghĩa là file pretrain nói sai về chính nó.")
    if not keep:
        raise PretrainIncompatible(
            "không có tham số nào thuộc PRETRAINED_MODULES — nhánh dựng hỏng.")
    return keep, drop


# ══ Ghi ═════════════════════════════════════════════════════════════════════
def save_pretrain(branch, out_dir: Path, extra: dict | None = None) -> Path:
    """Ghi CHỈ phần đã pretrain + META. Trả đường dẫn file trọng số."""
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep, drop = split_state(branch)
    sd = {k: v.detach().to("cpu").contiguous() for k, v in keep.items()}
    path = out_dir / PRETRAIN_FILE
    save_file(sd, str(path))

    q = branch.qformer
    meta = {
        "format": FORMAT,
        "pretrained_modules": list(PRETRAINED_MODULES),
        "untrained_modules": list(UNTRAINED_MODULES),
        # Khai cả những khoá KHÔNG đi vào file, và số lượng, để một lần đọc META
        # trả lời được "cái gì trong đây đã học và cái gì chưa" mà không cần mở
        # file safetensors.
        "keys": sorted(sd),
        "n_keys": len(sd),
        "keys_not_saved": sorted(drop),
        "n_params_saved": int(sum(v.numel() for v in sd.values())),
        # `spec` là NGUYÊN VĂN hai khối config đã dựng ra nhánh — cùng cơ chế
        # với recombiner.save_branch, và là thứ `assert_compatible` đối chiếu.
        "spec": branch.spec,
        "qformer": {"tasks": list(q.tasks), "frozen": list(q.frozen),
                    "d_model": int(q.d_model)},
        "n_query": int(branch.n_query),
        "d_vis": int(branch.d_vis),
        "d_model": int(branch.d_model),
        "n_labels": int(getattr(branch.cls_head, "n_labels", 0) or 0),
        "n_classes": int(getattr(branch.cls_head, "n_classes", 0) or 0),
        "route_enabled": bool(branch.route_enabled),
        "n_experts": int(branch.route_gate.n_experts),
    }
    meta.update(dict(extra or {}))
    (out_dir / PRETRAIN_META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")
    return path


def pretrain_exists(d: Path) -> bool:
    d = Path(d)
    return (d / PRETRAIN_FILE).exists() and (d / PRETRAIN_META).exists()


def read_meta(d: Path) -> dict:
    d = Path(d)
    if not pretrain_exists(d):
        raise FileNotFoundError(
            f"{d} thiếu {PRETRAIN_FILE} hoặc {PRETRAIN_META} — đây không phải "
            f"một checkpoint sup_pretrain.")
    return json.loads((d / PRETRAIN_META).read_text(encoding="utf-8"))


def copy_ckpt(src: Path, dst: Path, with_trainer_state: bool = False) -> Path:
    """Chép một ckpt sang chỗ khác (dùng cho `best/`). Mặc định BỎ trainer_state.

    `best/` là thứ đem đi tái sử dụng, và trạng thái optimizer của một lần chạy
    pretrain không có nghĩa gì ở lần chạy decoder_ft — nó chỉ làm repo nặng lên
    vài trăm MB và mời người ta nạp nhầm.
    """
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    names = [PRETRAIN_FILE, PRETRAIN_META]
    if with_trainer_state:
        names.append(TRAINER_STATE)
    for n in names:
        p = src / n
        if p.exists():
            shutil.copy2(p, dst / n)
    return dst


# ══ Kiểm tương thích ════════════════════════════════════════════════════════
def assert_compatible(meta: dict, branch, where: str = "") -> None:
    """Checkpoint này có dựng ra ĐÚNG nhánh đang có trong bộ nhớ không?

    Bốn con số phải khớp, và cả bốn đều là loại "lệch thì `load_state_dict` vẫn
    có thể chạy":

      * `d_vis`   — vision tower khác -> Recombinator chiếu sai chiều;
      * `d_model` — decoder khác -> 32 token nối vào sai chiều;
      * `n_labels`/`n_classes` — ClsHead khác hình dạng; lệch là nạp một head
        học trên một bộ bệnh lý KHÁC, và ASL vẫn giảm bình thường vì nó không
        biết gì về tên nhãn;
      * `qformer.d_model` — vec_1/vec_2 khác chiều.

    Riêng `qformer.tasks` phải khớp THỨ TỰ, không chỉ tập hợp: `query` là một
    Parameter (n_task, d) và chỉ số hàng được suy ra từ vị trí trong `tasks`.
    Đổi thứ tự là phân loại bằng vector định tuyến, và không có gì nổ.
    """
    tag = f" ({where})" if where else ""
    q_now = branch.qformer
    want = {
        "d_vis": int(branch.d_vis),
        "d_model": int(branch.d_model),
        "n_labels": int(getattr(branch.cls_head, "n_labels", 0) or 0),
        "n_classes": int(getattr(branch.cls_head, "n_classes", 0) or 0),
    }
    for k, v in want.items():
        got = meta.get(k)
        if got is None:
            continue
        if int(got) != int(v):
            raise PretrainIncompatible(
                f"checkpoint pretrain{tag}: {k} = {got}, nhánh đang dựng cần "
                f"{v}.\n"
                f"    Nạp tiếp là chạy với một phần trọng số ngẫu nhiên (hoặc "
                f"tệ hơn: một ClsHead đã học trên một bộ nhãn khác). Pretrain "
                f"lại, hoặc sửa config cho khớp checkpoint.")
    mq = dict(meta.get("qformer") or {})
    if mq.get("d_model") is not None and int(mq["d_model"]) != int(q_now.d_model):
        raise PretrainIncompatible(
            f"checkpoint pretrain{tag}: qformer.d_model = {mq['d_model']}, "
            f"nhánh cần {q_now.d_model}.")
    if mq.get("tasks") and list(mq["tasks"]) != list(q_now.tasks):
        raise PretrainIncompatible(
            f"checkpoint pretrain{tag}: qformer.tasks = {list(mq['tasks'])}, "
            f"nhánh đang dựng {list(q_now.tasks)}.\n"
            f"    THỨ TỰ cũng phải khớp, không chỉ tập hợp: `query` là một "
            f"Parameter (n_task, d) và chỉ số hàng lấy từ vị trí trong danh "
            f"sách. Đổi chỗ `sup` với `aux` là phân loại bằng vector định tuyến "
            f"và định tuyến bằng vector phân loại — cả hai loss vẫn giảm.")
    fmt = str(meta.get("format", ""))
    if fmt and fmt != FORMAT:
        raise PretrainIncompatible(
            f"checkpoint pretrain{tag} có format {fmt!r}, bản này đọc {FORMAT!r}.")


def load_into_branch(branch, src_dir: Path, verbose: bool = True) -> dict:
    """Nạp phần đã pretrain vào `branch`. Trả báo cáo (khối nào nạp, khối nào tươi).

    Đối chiếu tập khoá theo CẢ HAI chiều trước khi nạp:

      * khoá trong file mà nhánh không có -> file thuộc một kiến trúc khác;
      * khoá của một PRETRAINED_MODULE mà file không có -> file bị cắt xén, và
        nạp tiếp nghĩa là một phần của khối "đã pretrain" thật ra là ngẫu nhiên.

    `strict=False` của `load_state_dict` một mình là KHÔNG đủ ở đây: nó trả về
    hai danh sách rồi để người gọi tự quyết, và cách viết quen tay là bỏ qua
    chúng. Ở đây chúng là điều kiện dừng.
    """
    from safetensors.torch import load_file

    src_dir = Path(src_dir)
    meta = read_meta(src_dir)
    assert_compatible(meta, branch, where=str(src_dir))

    sd = load_file(str(src_dir / PRETRAIN_FILE))
    own = dict(branch.state_dict())

    extra = sorted(set(sd) - set(own))
    if extra:
        raise PretrainIncompatible(
            f"{src_dir / PRETRAIN_FILE} có {len(extra)} khoá nhánh hiện tại "
            f"không có: {extra[:5]}.\n"
            f"    File này do một bản src/recombiner.py khác ghi ra. Nạp tiếp "
            f"là nạp một phần và bỏ qua phần kia — im lặng.")
    want = sorted(k for k in own if _module_of(k) in PRETRAINED_MODULES)
    missing = sorted(set(want) - set(sd))
    if missing:
        raise PretrainIncompatible(
            f"{src_dir / PRETRAIN_FILE} thiếu {len(missing)} khoá của các khối "
            f"ĐÃ khai là được pretrain: {missing[:5]}.\n"
            f"    Nghĩa là một phần của khối 'đã pretrain' sẽ là số ngẫu nhiên, "
            f"trong khi META nói nó đã học. Pretrain lại.")
    for k in want:
        if tuple(sd[k].shape) != tuple(own[k].shape):
            raise PretrainIncompatible(
                f"khoá {k}: checkpoint {tuple(sd[k].shape)} vs nhánh "
                f"{tuple(own[k].shape)}.")

    miss, unexp = branch.load_state_dict(sd, strict=False)
    # `miss` ở đây ĐÚNG BẰNG các khoá của UNTRAINED_MODULES — đã kiểm ở trên là
    # không thiếu khoá nào của PRETRAINED_MODULES. Giữ phép kiểm này để nếu ai
    # đổi PRETRAINED_MODULES mà quên chỗ khác thì nó nổ ở đây.
    bad = [k for k in miss if _module_of(k) in PRETRAINED_MODULES]
    if bad or unexp:
        raise PretrainIncompatible(
            f"nạp pretrain hỏng: thiếu {bad[:5]} · thừa {list(unexp)[:5]}")

    fresh = sorted({_module_of(k) for k in miss})
    report = {
        "source": str(src_dir),
        "loaded_modules": list(PRETRAINED_MODULES),
        "fresh_modules": fresh,
        "n_keys_loaded": len(want),
        "n_params_loaded": int(sum(own[k].numel() for k in want)),
        "pretrain_epoch": meta.get("epoch"),
        "pretrain_val": meta.get("val"),
        "pretrain_run_id": meta.get("run_id"),
    }
    if verbose:
        print(f"  ← nạp pretrain từ {src_dir}", flush=True)
        print(f"    khối ĐÃ pretrain  : {', '.join(PRETRAINED_MODULES)} "
              f"({report['n_params_loaded']:,} tham số, {len(want)} khoá)",
              flush=True)
        print(f"    khối còn TƯƠI     : {', '.join(fresh) or '—'} "
              f"(khởi tạo lại, không có trong file — đúng như META khai)",
              flush=True)
        v = meta.get("val") or {}
        if v:
            print(f"    checkpoint: epoch {meta.get('epoch')} · val "
                  f"cls_f1_pos {v.get('cls_f1_pos')} · asl {v.get('asl')}",
                  flush=True)
    return report


# ══ Nguồn: thư mục cục bộ hoặc Hub ══════════════════════════════════════════
def resolve_source(spec: str, cache_dir: str | Path | None = None) -> Path:
    """`--init-from` -> một thư mục cục bộ có sẵn hai file.

    Ba dạng nhận vào:
        /đường/dẫn/ckpt/latest        thư mục cục bộ (SWA cuối)
        hub:OWNER/NAME                repo model, gốc repo
        hub:OWNER/NAME/SUBDIR@rev     thư mục con + revision

    Tải về bằng `allow_patterns` chỉ hai file: kéo cả repo về nghĩa là kéo luôn
    mọi epoch đã đẩy lên, và ở một repo pretrain chạy vài chục epoch thì đó là
    vài chục GB cho một lần chạy chỉ cần `best/`.
    """
    s = str(spec).strip()
    if not s:
        raise ValueError("nguồn pretrain rỗng")
    if not s.startswith("hub:"):
        p = Path(s).expanduser()
        if not pretrain_exists(p):
            raise FileNotFoundError(
                f"{p} không phải checkpoint sup_pretrain (thiếu {PRETRAIN_FILE} "
                f"hoặc {PRETRAIN_META}).\n"
                f"    Trỏ vào thư mục ckpt cụ thể, ví dụ "
                f"runs/sup_pretrain/ckpt/latest — không phải thư mục run.")
        return p

    from huggingface_hub import snapshot_download

    from .config import hf_token

    body = s[4:]
    rev = None
    if "@" in body:
        body, rev = body.split("@", 1)
    parts = [x for x in body.split("/") if x]
    if len(parts) < 2:
        raise ValueError(f"`hub:` cần dạng OWNER/NAME[/SUBDIR], nhận {s!r}")
    repo_id = "/".join(parts[:2])
    subdir = "/".join(parts[2:])
    pat = f"{subdir}/*" if subdir else "*"
    local = snapshot_download(
        repo_id=repo_id, repo_type="model", revision=rev, token=hf_token(),
        cache_dir=str(cache_dir) if cache_dir else None,
        allow_patterns=[f"{subdir}/{PRETRAIN_FILE}" if subdir else PRETRAIN_FILE,
                        f"{subdir}/{PRETRAIN_META}" if subdir else PRETRAIN_META,
                        pat.replace("*", "README.md")])
    d = Path(local) / subdir if subdir else Path(local)
    if not pretrain_exists(d):
        raise FileNotFoundError(
            f"tải {repo_id} về {d} nhưng không thấy {PRETRAIN_FILE}/"
            f"{PRETRAIN_META}. Kiểm `pretrain.hub_subdir` của lần đẩy.")
    return d


# ══ Ckpt MỚI NHẤT trên Hub (cho `--resume`) ═════════════════════════════════
def hub_candidates(repo_id: str, revision: str | None = None,
                   token: str | None = None) -> list[dict]:
    """Mọi thư mục trong repo MODEL có ĐỦ hai file của một ckpt sup_pretrain.

    Chỉ tải META (vài KB mỗi cái), không tải trọng số: mục đích của hàm này là
    TRẢ LỜI "cái nào mới nhất", và câu trả lời nằm trong META.

    Một `{PRETRAIN_META}` không có `{PRETRAIN_FILE}` đi kèm thì KHÔNG tính là
    ứng viên. Ca đó có thật: một lần `upload_folder` đứt giữa chừng để lại file
    JSON nhỏ (đi trước, không qua LFS) mà không có file trọng số — và một hàm
    chọn theo tên thư mục sẽ chọn đúng cái đó rồi hỏng ở bước nạp, sau khi đã
    tải xong base và dựng xong nhánh.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    files = set(api.list_repo_files(repo_id=repo_id, repo_type="model",
                                    revision=revision))
    out: list[dict] = []
    for f in sorted(files):
        if f.rsplit("/", 1)[-1] != PRETRAIN_META:
            continue
        sub = f[: -len(PRETRAIN_META)].strip("/")
        weight = f"{sub}/{PRETRAIN_FILE}" if sub else PRETRAIN_FILE
        if weight not in files:
            out.append({"subdir": sub, "meta": None,
                        "error": f"thiếu {weight}"})
            continue
        try:
            p = hf_hub_download(repo_id=repo_id, filename=f, repo_type="model",
                                revision=revision, token=token)
            out.append({"subdir": sub, "error": None,
                        "meta": json.loads(Path(p).read_text(encoding="utf-8"))})
        except Exception as e:                      # noqa: BLE001
            out.append({"subdir": sub, "meta": None, "error": repr(e)})
    return out


def _rank(c: dict) -> tuple:
    """Khoá sắp xếp "mới nhất": epoch, rồi global_step, rồi seen_samples.

    Theo `epoch` chứ KHÔNG theo commit gần nhất, và KHÔNG theo `hub_subdir`:

      * commit gần nhất là thời điểm ĐẨY, không phải thời điểm HUẤN LUYỆN. Chạy
        lại `--stages push` cho một ckpt cũ là một commit mới mang một epoch cũ.
      * `hub_subdir` (mặc định `best`) là một TÊN. `best/` giữ epoch có val tốt
        nhất, và với `select_metric = cls_f1_pos` thì đó thường KHÔNG phải epoch
        cuối. Lấy `best/` khi cạnh nó có `epoch_012/` là lùi lại vài epoch mà
        không có dòng log nào nói vậy.
    """
    m = dict(c.get("meta") or {})

    def _i(k: str) -> int:
        try:
            return int(m.get(k))
        except (TypeError, ValueError):
            return -1

    return (_i("epoch"), _i("global_step"), _i("seen_samples"), c["subdir"])


def latest_on_hub(cfg, revision: str | None = None,
                  cache_dir: str | Path | None = None, verbose: bool = True
                  ) -> tuple[Path, dict, str]:
    """Ckpt MỚI NHẤT trên `pretrain.hub_repo` -> (thư mục cục bộ, META, subdir).

    In RA TẤT CẢ ứng viên tìm thấy và đánh dấu cái được chọn. Một lần `--resume`
    lấy nhầm epoch là một lần chạy mấy tiếng cho ra một checkpoint không ai giải
    thích được, nên việc "nó chọn cái nào" phải đọc được từ log chứ không phải
    suy ra từ quy ước đặt tên.
    """
    from .config import hf_token

    p = dict(cfg.get("pretrain", {}) or {})
    repo_id = str(p.get("hub_repo", "")).strip()
    if not repo_id or "/" not in repo_id:
        raise ValueError(
            f"pretrain.hub_repo = {repo_id!r} — cần dạng owner/name thì mới "
            f"biết lấy checkpoint ở đâu. Hoặc chỉ định thẳng nguồn: "
            f"--resume hub:OWNER/NAME/SUBDIR")
    tok = hf_token()
    try:
        cands = hub_candidates(repo_id, revision, tok)
    except Exception as e:
        raise RuntimeError(
            f"không đọc được repo model {repo_id}: {e!r}\n"
            f"    `pretrain.hub_private` mặc định là true, nên cần HF_TOKEN có "
            f"quyền ĐỌC `{repo_id.split('/')[0]}`:\n"
            f"      export HF_TOKEN=hf_xxxxxxxx") from e

    ok = [c for c in cands if c["meta"]]
    if not ok:
        broken = "\n".join(f"      {c['subdir'] or '.'}: {c['error']}"
                           for c in cands)
        raise FileNotFoundError(
            f"{repo_id} không có thư mục nào chứa đủ {PRETRAIN_FILE} + "
            f"{PRETRAIN_META}.\n"
            + (f"    Có {len(cands)} thư mục gần giống nhưng không dùng được:\n"
               f"{broken}\n" if cands else "")
            + f"    Chưa đẩy lần nào? Chạy `python -m src.run_pretrain "
              f"--stages push` trước, hoặc trỏ thẳng vào một thư mục ckpt cục "
              f"bộ: --resume runs/sup_pretrain/ckpt/latest")

    ok.sort(key=_rank, reverse=True)
    win = ok[0]
    sub = str(win["subdir"] or "")
    if verbose:
        print(f"  ⌁ {repo_id}: {len(ok)} checkpoint trên Hub"
              + (f" (rev {revision})" if revision else ""), flush=True)
        for c in ok:
            m = dict(c["meta"])
            v = dict(m.get("val") or {})
            print(f"    {'→' if c is win else ' '} {(c['subdir'] or '.'):<14} "
                  f"epoch {m.get('epoch')}/{m.get('n_epochs')} · bước "
                  f"{m.get('global_step')} · {m.get('seen_samples')} lượt mẫu "
                  f"· val f1_pos {v.get('cls_f1_pos')}", flush=True)
        if len(ok) > 1:
            print("      (chọn theo `epoch` trong META, không theo tên thư mục "
                  "và không theo commit gần nhất)", flush=True)

    spec = f"hub:{repo_id}" + (f"/{sub}" if sub else "")
    if revision:
        spec += f"@{revision}"
    return resolve_source(spec, cache_dir=cache_dir), dict(win["meta"]), sub


# ══ Đẩy lên Hub ═════════════════════════════════════════════════════════════
def _model_card(meta: dict, cfg) -> str:
    v = dict(meta.get("val") or {})
    t = dict(meta.get("train") or {})
    d = dict(meta.get("data") or {})
    rows = "\n".join(
        f"| `{m}` | {'CÓ' if m in meta.get('pretrained_modules', []) else 'KHÔNG'} |"
        for m in list(meta.get("pretrained_modules", []))
        + list(meta.get("untrained_modules", [])))
    # Nguồn gốc phải nằm TRONG card, không chỉ trong log của lần chạy đã trôi
    # mất: một checkpoint `--resume` trông y hệt một checkpoint chạy liền mạch
    # ở mọi con số trên Hub, mà đường lr của nó thì không giống — và "8 epoch"
    # trong một bảng kết quả có nghĩa khác hẳn nếu nó là 3 + 5 trên hai chu kỳ.
    r = dict(meta.get("resumed_from") or {})
    lineage = "" if not r else f"""
## Nguồn khởi tạo — đây là một **warm restart**

Checkpoint này KHÔNG bắt đầu từ trọng số ngẫu nhiên. Nó nối tiếp
`{r.get("spec")}` (epoch {r.get("epoch")}, val `cls_f1_pos` {r.get("val_f1_pos")})
qua `--resume`, rồi chạy thêm {meta.get("n_epochs")} epoch trên một chu kỳ
one-cycle **MỚI** (`max_lr` {(meta.get("schedule") or {}).get("max_lr")}).

Trạng thái optimizer/scheduler/RNG của lần chạy trước **không** được nối tiếp —
`trainer_state.pt` không bao giờ lên Hub. lr của lần chạy này bắt đầu lại từ đầu
chu kỳ, nên `epoch` ở bảng dưới đếm từ 1 của lần chạy NÀY: tổng số epoch mà
trọng số đã đi qua là {r.get("epoch")} + {meta.get("n_epochs")}, trên hai đường
cong khác nhau. Con số ấy phải vào phần Giới hạn nếu checkpoint này ra bảng.
"""
    return f"""---
library_name: rgexp
tags: [recombiner, qformer, siglip, asl, chest-xray, supervised-pretraining]
---

# RGExp — sup_pretrain (Recombinator + QFormer(vec_1) + ClsHead)

Checkpoint của chế độ **`sup_pretrain`**: nhánh thị giác có giám sát được huấn
luyện 100% bằng Sup Loss (ASL trên nhãn bệnh lý), **không có decoder, không có
CE**. Dùng để khởi tạo nhánh cho §2.3 / §2.4 ở chế độ `decoder_ft`.

| khối | có trong file này |
|---|---|
{rows}

`route_gate` và `gate` **không** nằm trong file: ở pretrain chúng không nhận một
gradient nào (`gate` là cổng vào decoder, mà decoder không chạy; `route_gate`
đọc `vec_2`, mà `vec_2` bị đóng băng). Ghi chúng vào đây sẽ làm file trông như
một checkpoint đầy đủ của nhánh, và người nạp sẽ tưởng Route_Gate đã được
pretrain.
{lineage}
## Số liệu

| | |
|---|---|
| epoch | {meta.get("epoch")} / {meta.get("n_epochs")} |
| bước tối ưu hoá | {meta.get("global_step")} |
| mẫu đã đi qua | {meta.get("seen_samples")} |
| train ASL (epoch cuối) | {t.get("asl")} |
| val ASL | {v.get("asl")} |
| val `cls_f1_pos` | {v.get("cls_f1_pos")} |
| val `cls_acc` | {v.get("cls_acc")} |

## Điều kiện

| | |
|---|---|
| base (vision tower) | `{(cfg.get("base", {}) or {}).get("hf_id")}` |
| n_pretrain | {d.get("n_train")} (split `{d.get("split_csv")}`) |
| n_val | {d.get("n_val")} |
| ảnh | cạnh dài ≤ {d.get("image_max_side")}px · processor {d.get("processor_size")}² · pan&scan {d.get("do_pan_and_scan")} |
| nhãn | {meta.get("n_labels")} bệnh lý × {meta.get("n_classes")} lớp |
| seed | {meta.get("seed")} |

## Nạp lại

```bash
    python -m src.run_recomb --exp 2.3 --mode decoder_ft \\
    --init-from hub:{(cfg.get("pretrain", {}) or {}).get("hub_repo")}/{(cfg.get("pretrain", {}) or {}).get("hub_subdir", "latest")}
```

## Giới hạn

Vision tower **đóng băng** và chạy trong `no_grad`: file này chỉ chứa nhánh đọc
đặc trưng SigLIP, không chứa một tham số nào của base. Nó chỉ dùng lại được với
đúng base, đúng `image_max_side` và đúng chính sách pan&scan ghi ở bảng trên —
đổi một trong ba là nhánh đọc một phân phối đặc trưng khác phân phối nó đã học.
"""


def push_folder(repo_id: str, folder: Path, path_in_repo: str = "",
                private: bool = True, commit_message: str = "",
                ignore_patterns: list[str] | None = None) -> str:
    """Đẩy một thư mục lên repo MODEL. Dùng cho cả checkpoint pretrain lẫn model §3.

    Tách ra khỏi `src/push.push_results` chứ không dùng lại nó: hàm ấy đẩy vào
    một repo DATASET và có `ignore_patterns` chặn `*.safetensors` (kết quả §2.x
    là bảng và văn bản, không phải trọng số). Gọi nó với một thư mục trọng số
    thì nó chạy trơn tru và đẩy lên đúng... không có gì.
    """
    from huggingface_hub import HfApi

    from .config import hf_token

    api = HfApi(token=hf_token())
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private,
                        exist_ok=True)
    except Exception as e:
        raise RuntimeError(
            f"không tạo/mở được model repo {repo_id}: {e!r}\n"
            f"    Token phải có quyền WRITE, và tài khoản phải ghi được vào "
            f"`{repo_id.split('/')[0]}`.") from e
    api.upload_folder(
        folder_path=str(folder), repo_id=repo_id, repo_type="model",
        path_in_repo=path_in_repo.strip("/") or ".",
        commit_message=commit_message or f"upload {Path(folder).name}",
        ignore_patterns=list(ignore_patterns or []) + ["*.part", "__pycache__/*"])
    return f"https://huggingface.co/{repo_id}"


def push_pretrain(cfg, ckpt_dir: Path, extra_files: list[Path] | None = None
                  ) -> str | None:
    """Đẩy CHỈ các khối đã pretrain lên `pretrain.hub_repo` (repo MODEL).

    Không đẩy `trainer_state.pt` (trạng thái optimizer — vài trăm MB, vô nghĩa
    với người tái sử dụng) và không đẩy ckpt của từng epoch trừ khi được gọi
    riêng cho thư mục ấy.
    """
    from .config import hf_token

    p = dict(cfg.get("pretrain", {}) or {})
    repo_id = str(p.get("hub_repo", "")).strip()
    if not repo_id:
        print("⊘ pretrain.hub_repo trống — không đẩy", flush=True)
        return None
    if "/" not in repo_id:
        raise ValueError(f"pretrain.hub_repo phải có dạng owner/name, "
                         f"nhận {repo_id!r}")
    ckpt_dir = Path(ckpt_dir)
    if not pretrain_exists(ckpt_dir):
        raise FileNotFoundError(
            f"{ckpt_dir} chưa có {PRETRAIN_FILE}/{PRETRAIN_META} — chưa có gì "
            f"để đẩy. Chạy stage `train` trước.")
    tok = hf_token()
    if not tok:
        print(f"⚠ không có HF_TOKEN — bỏ qua đẩy. Checkpoint vẫn ở {ckpt_dir}\n"
              f"    export HF_TOKEN=hf_xxxxxxxx  &&  "
              f"python -m src.run_pretrain --stages push", flush=True)
        return None

    meta = read_meta(ckpt_dir)
    (ckpt_dir / "README.md").write_text(_model_card(meta, cfg), encoding="utf-8")
    for f in (extra_files or []):
        f = Path(f)
        if f.exists() and f.parent != ckpt_dir:
            shutil.copy2(f, ckpt_dir / f.name)

    subdir = str(p.get("hub_subdir", "latest")).strip("/")
    print(f"→ đẩy {ckpt_dir} -> {repo_id}/{subdir or '.'} (model)", flush=True)
    url = push_folder(
        repo_id, ckpt_dir, subdir, bool(p.get("hub_private", True)),
        commit_message=f"sup_pretrain epoch {meta.get('epoch')} · val f1_pos "
                       f"{(meta.get('val') or {}).get('cls_f1_pos')}",
        # Trạng thái optimizer KHÔNG lên: nó chỉ phục vụ resume của chính lần
        # chạy pretrain này, và nó nặng gấp ~3 lần phần trọng số.
        ignore_patterns=[TRAINER_STATE])
    print(f"✓ đã đẩy: {url}  (thư mục `{subdir or '.'}`)", flush=True)
    return url

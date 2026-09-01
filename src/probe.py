"""Probe chuyên môn của từng expert (`run.mode = "probe"`).

Mode này KHÔNG train gì cả: nó lấy một checkpoint đã chạy và hỏi một câu khác
hẳn — **mỗi expert trong 12 cái kia thực ra đã học được cái gì?** Bảng sweep
chỉ trả lời "kiến trúc nào cho BLEU cao hơn"; nó không nói expert 7 chuyên về
tràn dịch màng phổi hay chỉ là một bản sao mờ của expert 3. Không trả lời được
câu đó thì MoE chỉ là một cách tiêu tham số.

Với MỖI ảnh của tập test, probe làm ba việc:

  1. Sinh báo cáo MỘT lần với gate thật (đúng như `mode=test` vẫn làm) — đây
     là kết quả để chấm metrics, không dính gì tới probe.
  2. Một forward NỀN với gate = 0 (không expert nào chạy) — mốc để trừ.
  3. Thêm một loạt forward, mỗi lần bật ĐÚNG MỘT expert với trọng số 1.0.
     Hiệu số so với nền chính là **dấu vân tay** của expert đó trên đúng ảnh
     này. Bật bao nhiêu expert thì tuỳ `run.probe_full_n`:

       * `probe_full_n` ảnh ĐẦU của tập test -> bật ĐỦ E expert (E = 12), kể
         cả những expert router không chọn. Đây là hồ sơ **vô điều kiện**: mọi
         expert được đo trên đúng cùng một tập ảnh nên so trực tiếp được, và
         expert chưa bao giờ được router gọi vẫn có hồ sơ.
       * các ảnh còn lại -> chỉ bật K expert router đã chọn (K = top_k = 4),
         như trước. Rẻ hơn, và vẫn đủ để thống kê router trên trọn tập test.

     Ảnh nào thuộc nhóm nào lấy theo THỨ TỰ GỐC của tập test, nên phiên sau
     resume vẫn ra đúng tập ảnh đó. Bản ghi tự khai bằng `probe_mode`.

Bốn thứ đọc ra từ mỗi dấu vân tay:

  * **Bản đồ không gian 16×16** — ‖Δh‖ tại từng image token, reshape về lưới
    patch của SigLIP. Chồng lên ảnh X-quang thì thấy expert đổi biểu diễn ở
    VÙNG NÀO của phổi.
  * **Vector 14 bệnh + vector 4 aspect** — Δlogit trên đúng những dòng
    unembedding ứng với tên 14 finding của CheXbert và 4 tên aspect. Xếp K
    expert lại được ma trận K×14 và K×4: đây là "bảng chuyên môn" đọc được
    bằng mắt.
  * **Năm nhóm token toàn cục** (`PROBE_FIELDS`) — cùng phép đo Δlogit nhưng
    trên các nhóm từ ngoài danh sách bệnh: giải phẫu & vị trí, mức độ & độ
    chắc chắn, ngữ cảnh thời gian & so sánh, ngữ pháp & cấu trúc báo cáo,
    thiết bị hỗ trợ. Một expert có thể không chuyên bệnh nào mà chuyên phần
    KHÁC của bài báo cáo — chỗ viết "Findings:", chỗ viết "stable compared to
    prior", chỗ đặt tên vùng phổi. Không có những cột này thì expert như vậy
    hiện ra là "không nổi trội", một kết luận sai do thước đo thiếu ô.

  * **Top-10 token unembedding** — 10 token mà expert đẩy logit lên mạnh nhất
    trên toàn vocab 262k. Không giới hạn trong danh sách nào, nên nó bắt được
    cả những chuyên môn mà cả năm nhóm trên đều không nghĩ tới.

═══ Hai quyết định thiết kế cần biết trước khi đọc số ═══

**1. Teacher-forcing, không generate lại.** Forward probe cho ăn prompt + ảnh
+ ĐÚNG báo cáo tham chiếu, y hệt một step train. Lý do: lúc generate, chuỗi
chỉ có prompt nên đúng MỘT vị trí (token cuối) dự đoán được gì đó, và đọc
unembedding ở một vị trí là đọc nhiễu. Teacher-forcing cho ~100 vị trí báo cáo
để lấy trung bình, mà vẫn chỉ tốn ĐÚNG một forward — rẻ hơn generate khoảng
150 lần. Gate thì vẫn là gate thật: `router_mask` che target nên router chỉ
nhìn prompt + ảnh, đúng thứ nó nhìn lúc infer.

**2. Bật expert với trọng số 1.0, không phải g_e của router.** Đo "expert này
là gì", không phải "expert này đóng góp bao nhiêu vào mẫu này". Trọng số thật
vẫn được ghi trong `gate` của từng bản ghi, muốn quy đổi lại thì nhân sau.

═══ Vì sao KHÔNG có ma trận 14×4 ═══

Nhãn là 14 bệnh × 4 aspect, nên phản xạ đầu tiên là đòi một ma trận 14×4 cho
mỗi expert. Nhưng logit-lens không sinh ra được ma trận đó: cái đọc được là
Δlogit trên từng DÒNG unembedding, mà tên bệnh và tên aspect là hai nhóm dòng
rời nhau. Ghép lại chỉ có thể là `M[d,a] = u[d] + v[a]` — một ma trận hạng 1
đội lốt 14×4, mọi hàng giống hệt nhau sai khác một hằng số. Nó KHÔNG chứa
thêm một bit thông tin nào so với hai vector rời, chỉ khiến người đọc tưởng có
tương tác bệnh×aspect. Nên ở đây là 14 và 4 để riêng.

Muốn tương tác thật thì phải có nhãn per-token (biết vị trí nào trong báo cáo
đang nói về aspect nào) — không có trong dữ liệu này.

═══ Đẩy lên Hub ═══

`CommitScheduler` của huggingface_hub: mỗi ảnh xong 6 expert là ghi ngay một
file JSON vào thư mục cục bộ, một thread nền gom lại và commit mỗi
`probe_commit_every` phút. Đúng nghĩa "xong ảnh nào có ngay ảnh đó", nhưng
3000 commit riêng lẻ thì HF chặn giữa chừng và lịch sử repo không đọc được.

`key_id` của mỗi bản ghi là `img_rel` nguyên bản — cùng khoá mà split và
`labels.npz` dùng, nên nối lại với nhãn hay với `reports.csv` không cần bảng
tra nào. Tên FILE là khoá đã thay `/` bằng `__` (khoá gốc có dấu / thì thành
thư mục con trên Hub).

Chạy lại được: lúc khởi động, danh sách file đã có trên Hub được đọc về và
những ảnh đó bị bỏ qua. Phiên Kaggle 12h đứt giữa chừng thì phiên sau chạy
tiếp, không làm lại từ đầu.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

RECORD_DIR = "records"          # đường dẫn trong dataset repo

# Bản ghi v1 chỉ có top-k expert và hai vector bệnh/aspect; v2 có ĐỦ E expert
# (khi `probe_mode == "full"`), vector `prob` dày của router, và 5 nhóm token
# `fields`. `scripts/probe_summary.py` đọc được cả hai và tách rõ trong báo cáo
# — nên KHÔNG được xoá số này khi đổi schema lần sau, chỉ tăng nó lên.
SCHEMA_VERSION = 2

# ═══════════════════ 5 nhóm token khảo sát ngoài danh sách bệnh ══════════════
#
# 14 finding của CheXbert trả lời "expert này thiên về BỆNH nào". Nhưng một
# expert hoàn toàn có thể không chuyên bệnh nào cả mà chuyên phần khác của bài
# báo cáo: chỗ viết "Findings:", chỗ viết "stable compared to prior", chỗ đặt
# tên vùng phổi, chỗ mô tả ống thông. Không đo những cột đó thì expert như vậy
# hiện ra là "không nổi trội" — một kết luận SAI, chỉ vì thước đo không có ô
# nào cho nó.
#
# Mỗi tên trong danh sách -> một tập dòng unembedding (name_row_ids) -> một
# cột trong bảng nhiệt. Chọn tên theo hai tiêu chí: (1) xuất hiện thật trong
# báo cáo X-quang ngực, (2) phân biệt được với nhau — thêm ba từ đồng nghĩa
# của cùng một khái niệm chỉ làm bảng rộng ra chứ không thêm thông tin.
PROBE_FIELDS = {
    "anatomy": ("Giải phẫu & vị trí", [
        "lung", "lungs", "pleural", "cardiac", "mediastinal", "hilar",
        "diaphragm", "costophrenic", "apex", "base", "lobe",
        "left", "right", "bilateral", "upper", "lower", "retrocardiac",
        "airspace",
    ]),
    "degree": ("Mức độ & độ chắc chắn", [
        "mild", "moderate", "severe", "minimal", "small", "large", "trace",
        "extensive", "focal", "diffuse", "patchy",
        "likely", "possible", "probable", "suspected", "questionable",
        "definite", "no", "without", "unremarkable",
    ]),
    "temporal": ("Ngữ cảnh thời gian & so sánh", [
        "stable", "unchanged", "new", "increased", "decreased", "improved",
        "worsened", "resolved", "persistent", "prior", "previous",
        "compared", "comparison", "since", "interval", "again",
    ]),
    "grammar": ("Ngữ pháp & cấu trúc báo cáo", [
        "Findings", "Impression", "is", "are", "was", "were", "there",
        "the", "of", "and", "with", "seen", "noted", "demonstrates",
        "shows", "evidence", "normal", "otherwise", ".", ":",
    ]),
    "device": ("Thiết bị hỗ trợ", [
        "tube", "catheter", "line", "pacemaker", "sternotomy", "wire",
        "port", "PICC", "endotracheal", "tracheostomy", "drain", "clip",
        "valve", "leads",
    ]),
}


# ═══════════════════════════ tiện ích thuần (test được) ══════════════════════
def grid_shape(n_patch):
    """Số image token -> (h, w) của lưới patch. None nếu không phải số chính phương.

    SigLIP của Gemma 3 chia ảnh 896² thành lưới vuông (256 = 16×16). Model khác
    (Qwen2.5-VL phân giải động) thì số patch không vuông — lúc đó bản đồ không
    gian không dựng được và probe bỏ qua phần đó thay vì reshape bừa.
    """
    if n_patch <= 0:
        return None
    s = int(round(math.sqrt(n_patch)))
    return (s, s) if s * s == n_patch else None


def name_row_ids(tokenizer, names):
    """Tên -> list token id, để lấy các DÒNG unembedding tương ứng.

    Tên nhiều token ("Pleural Effusion") thì lấy TRUNG BÌNH Δlogit của các
    token của nó. Thô, nhưng đúng hướng: dòng unembedding của mỗi mảnh tên đều
    được đẩy lên khi model sắp viết ra cái tên ấy.

    Thêm tiền tố khoảng trắng: hầu hết tokenizer BPE coi "▁effusion" (giữa
    câu) và "effusion" (đầu câu) là hai token khác nhau, mà trong báo cáo thì
    gần như luôn là dạng giữa câu.
    """
    out = []
    for name in names:
        ids = set()
        for form in (" " + str(name).strip(), str(name).strip(),
                     " " + str(name).strip().lower(), str(name).strip().lower()):
            try:
                ids.update(tokenizer.encode(form, add_special_tokens=False))
            except Exception:
                pass
        out.append(sorted(ids))
    return out


def field_row_ids(tokenizer, fields=None):
    """{nhóm: (tên hiển thị, [tên])} -> {nhóm: [[token id]]} + danh sách tên hụt.

    Tách riêng khỏi `name_row_ids` chỉ để gom phần cảnh báo: một tên không
    tokenize được sẽ cho cột hằng 0 trong bảng nhiệt, mà 0 ở đó trông y hệt
    "expert không đẩy từ này lên" — hai chuyện khác hẳn nhau. Trả tên hụt ra
    ngoài để `run_probe` in cảnh báo MỘT lần thay vì để người đọc bảng tự đoán.
    """
    fields = PROBE_FIELDS if fields is None else fields
    ids, missing = {}, []
    for key, (_title, names) in fields.items():
        groups = name_row_ids(tokenizer, names)
        ids[key] = groups
        missing += [f"{key}/{names[i]}" for i, g in enumerate(groups) if not g]
    return ids, missing


def rows_mean(dlogit, id_groups):
    """Δlogit toàn vocab -> một số cho mỗi nhóm token (trung bình trong nhóm)."""
    return [float(dlogit[g].mean()) if g else 0.0 for g in id_groups]


def top_tokens(dlogit, tokenizer, k=10):
    """k token có Δlogit lớn nhất, đã decode ra chữ."""
    k = min(int(k), dlogit.numel())
    val, idx = torch.topk(dlogit, k)
    out = []
    for v, i in zip(val.tolist(), idx.tolist()):
        try:
            s = tokenizer.decode([i])
        except Exception:
            s = f"<id {i}>"
        out.append({"id": int(i), "token": s, "dlogit": round(float(v), 5)})
    return out


def safe_name(key):
    """img_rel -> tên file dùng được trên mọi hệ (khoá GỐC vẫn nằm trong JSON)."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(key))


def pack_map(vals):
    """[n] float -> (list đã chuẩn hoá về [0,1], max thật).

    Chuẩn hoá từng bản đồ theo max của CHÍNH NÓ: câu hỏi của bản đồ là "expert
    này nhìn ĐÂU", không phải "expert này mạnh cỡ nào" — độ mạnh đi riêng ở
    `spatial_max` / `spatial_l2` để so giữa các expert. Làm tròn 4 chữ số vì
    3000 ảnh × 6 expert × 256 ô mà in đủ float là ~120MB toàn chữ số vô nghĩa.
    """
    a = np.asarray(vals, dtype=np.float64)
    mx = float(a.max()) if a.size else 0.0
    norm = (a / mx) if mx > 0 else a
    return [round(float(v), 4) for v in norm], mx


def unembedding(model):
    """Ma trận unembedding [V, D] — dò qua nhiều lớp bọc.

    Model ở đây là PeftModel bọc Gemma3ForConditionalGeneration, và
    `get_output_embeddings` có bản không xuyên qua được lớp bọc. Nên thử lần
    lượt: chính nó -> base_model -> base_model.model, rồi mới tới việc tìm một
    module tên `lm_head`. Không thấy thì NỔ chứ không lặng lẽ bỏ qua: thiếu nó
    thì top-10 token và cả hai vector bệnh/aspect đều không tồn tại, tức probe
    chẳng còn gì để trả về.
    """
    seen = []
    for obj in (model, getattr(model, "base_model", None),
                getattr(getattr(model, "base_model", None), "model", None)):
        if obj is None:
            continue
        seen.append(type(obj).__name__)
        try:
            emb = obj.get_output_embeddings()
        except Exception:
            emb = None
        w = getattr(emb, "weight", None)
        if w is not None and w.dim() == 2:
            return w
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if name.endswith("lm_head") and w is not None and w.dim() == 2:
            return w
    raise RuntimeError(
        f"không tìm thấy ma trận unembedding (đã thử {seen} và mọi module tên "
        f"lm_head) — probe không có gì để chiếu Δh lên")


# ═══════════════════════════ luồng đẩy lên Hub ═══════════════════════════════
class HubStream:
    """Ghi bản ghi ra đĩa ngay, commit gộp theo lịch nền.

    Không có Hub (thiếu token / push tắt) thì vẫn ghi cục bộ và chạy tiếp —
    probe vẫn có giá trị khi chỉ nằm trên đĩa, không đáng để nổ giữa chừng.
    """

    def __init__(self, repo_id, token, local_dir, every_min=1.0, log=None,
                 warn=None, private=True):
        self.dir = local_dir
        self.log, self.warn = log, warn
        self.repo_id, self.token = repo_id, token
        self.sched = None
        os.makedirs(self.dir, exist_ok=True)
        if not (repo_id and token):
            if warn:
                warn("probe", "không có hub_repo/HF_TOKEN — bản ghi chỉ nằm "
                              f"ở {local_dir}")
            return
        try:
            from huggingface_hub import CommitScheduler, create_repo
            create_repo(repo_id, token=token, repo_type="dataset",
                        private=private, exist_ok=True)
            self.sched = CommitScheduler(
                repo_id=repo_id, repo_type="dataset", folder_path=self.dir,
                path_in_repo=RECORD_DIR, every=every_min, token=token,
                squash_history=False)
            if log:
                log("probe", f"CommitScheduler -> {repo_id}/{RECORD_DIR} "
                             f"(gộp mỗi {every_min*60:.0f}s)")
        except Exception as e:
            if warn:
                warn("probe", f"không bật được CommitScheduler ({e}) — bản ghi "
                              f"chỉ nằm ở {local_dir}")

    def done_ids(self):
        """Khoá đã probe xong: hợp của những gì có trên Hub và trên đĩa.

        Đọc cả hai vì hai thứ hỏng theo hai kiểu: Hub mất khi repo bị xoá/đổi
        tên, đĩa mất khi phiên Kaggle kết thúc.
        """
        got = {p[:-5] for p in os.listdir(self.dir) if p.endswith(".json")}
        if self.repo_id and self.token:
            try:
                from huggingface_hub import list_repo_files
                pre = RECORD_DIR + "/"
                got |= {os.path.basename(f)[:-5]
                        for f in list_repo_files(self.repo_id, token=self.token,
                                                 repo_type="dataset")
                        if f.startswith(pre) and f.endswith(".json")}
            except Exception as e:
                if self.warn:
                    self.warn("probe", f"không đọc được danh sách file trên Hub "
                                       f"({e}) — chỉ bỏ qua theo bản đĩa")
        return got

    def write(self, name, record):
        path = os.path.join(self.dir, f"{name}.json")
        tmp = path + ".tmp"
        # Ghi tạm rồi đổi tên: CommitScheduler quét thư mục theo lịch riêng và
        # sẽ vớ phải file đang viết dở nếu ghi thẳng.
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)

    def close(self):
        if self.sched is not None:
            try:
                self.sched.trigger().result()      # commit nốt phần còn lại
                self.sched.stop()
                if self.log:
                    self.log("probe", "đã commit nốt phần còn lại và dừng "
                                      "scheduler")
            except Exception as e:
                if self.warn:
                    self.warn("probe", f"commit cuối thất bại ({e})")


def reports_from_records(local_dir, key_col):
    """Dựng lại các dòng của reports.csv từ những bản ghi đã có trên đĩa.

    Phiên Kaggle đứt thì reports.csv mất theo /kaggle/temp, nhưng bản ghi thì
    còn (và còn cả trên Hub). Báo cáo sinh ra nằm SẴN trong mỗi bản ghi, nên
    dựng lại rẻ hơn nhiều so với generate lại 2000 ảnh chỉ để có metrics.
    """
    rows = []
    for name in sorted(os.listdir(local_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(local_dir, name), encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception:
            continue
        rp = rec.get("report")
        if rp:
            rows.append({key_col: rec["key_id"], **rp})
    return rows


# ═════════════════════════════ vòng probe chính ══════════════════════════════
@torch.no_grad()
def run_probe(*, cfg, model, bridge, tokenizer, collate, df, generate_records,
              disease_names, aspect_names, img_token_id, hf_token,
              train_state, log, warn, event):
    """Sinh báo cáo + probe top-k expert cho từng ảnh, đẩy lên Hub theo luồng.

    Trả về DataFrame kiểu reports.csv (gồm cả những dòng dựng lại từ bản ghi
    cũ) để `src/train.py` chấm metrics như mọi mode khác.
    """
    import pandas as pd

    from .model import make_loader, split_batch, to_device

    dev = torch.device("cuda:0")
    f, r = cfg.fixed, cfg.run
    key_col = cfg.data.image_column
    E, K = f.n_expert, f.top_k

    local = os.path.join(r.out_dir, "probe_records")
    stream = HubStream(r.probe_repo, hf_token, local,
                       every_min=float(r.probe_commit_every),
                       log=log, warn=warn, private=r.hub_private)
    done = stream.done_ids()
    todo = df[~df[key_col].map(lambda k: safe_name(k) in done)] \
        .reset_index(drop=True)
    log("probe", f"{len(df)} ảnh test | đã có {len(df) - len(todo)} bản ghi "
                 f"| còn {len(todo)}")
    event("probe_start", n_total=len(df), n_done=len(df) - len(todo),
          n_todo=len(todo), repo=r.probe_repo)

    # Dòng unembedding của 14 tên bệnh + 4 tên aspect + 5 nhóm token toàn cục.
    # Tính MỘT lần.
    dis_ids = name_row_ids(tokenizer, disease_names)
    asp_ids = name_row_ids(tokenizer, aspect_names)
    fld_ids, fld_missing = field_row_ids(tokenizer)
    empty = [disease_names[i] for i, g in enumerate(dis_ids) if not g]
    if empty:
        warn("probe", f"không tokenize được tên bệnh {empty} — cột tương ứng "
                      f"trong ma trận K×14 sẽ luôn = 0")
    if fld_missing:
        warn("probe", f"không tokenize được {len(fld_missing)} tên nhóm "
                      f"({', '.join(fld_missing[:8])}"
                      f"{' …' if len(fld_missing) > 8 else ''}) — cột tương "
                      f"ứng trong bảng nhiệt sẽ luôn = 0")

    # ── ảnh nào chạy ĐỦ E expert ────────────────────────────────────────────
    # `probe_full_n` ảnh ĐẦU của tập test (theo thứ tự gốc của `df`, KHÔNG
    # theo `todo`) chạy 1 + E forward: mọi expert đều bị bật một lần, kể cả
    # cái router không chọn. Lấy theo df gốc để phiên sau resume vẫn ra đúng
    # tập ảnh đó — lấy theo `todo` thì mỗi lần chạy lại tập full lại trượt đi.
    #
    # Vì sao cần: bảng "expert × nhãn" tính trên các ảnh mà expert ĐƯỢC CHỌN là
    # một đại lượng CÓ ĐIỀU KIỆN — nó trộn "expert này học được gì" với "router
    # gọi nó vào lúc nào", và expert chưa bao giờ được chọn thì không có hàng
    # nào cả. Ép chạy đủ E trên cùng một tập ảnh cho ta hồ sơ VÔ ĐIỀU KIỆN:
    # mọi expert đo trên đúng cùng những ảnh đó, so được trực tiếp với nhau.
    #
    # Giá: 1 + E forward/ảnh thay vì 1 + K (E=12, K=4 -> 13 so với 5). Nên nó
    # giới hạn ở N ảnh, phần còn lại của tập test vẫn chạy top-k để metrics
    # báo cáo phủ trọn tập.
    full_n = r.probe_full_n
    if full_n is None:
        full_keys = set(df[key_col].astype(str))
    else:
        full_keys = set(df[key_col].astype(str).iloc[:max(int(full_n), 0)])
    n_full_todo = int(todo[key_col].astype(str).isin(full_keys).sum())
    log("probe", f"chạy đủ {E} expert trên {len(full_keys)} ảnh "
                 f"({n_full_todo} cái còn phải làm, {1+E} forward/ảnh); "
                 f"{len(df) - len(full_keys)} ảnh còn lại chạy top-{K} "
                 f"({1+K} forward/ảnh)")
    W = unembedding(model)                 # [V, D], Gemma buộc chung với embed
    log("probe", f"unembedding {tuple(W.shape)} | {len(disease_names)} bệnh × "
                 f"{len(aspect_names)} aspect | top-{r.probe_top_tokens} token")

    grid, warned_grid = None, []

    def forward_with(core_fwd, mask, gate_vec):
        """Một forward, ép gate = `gate_vec`. Trả (hidden cuối [N,D], gate THẬT).

        `bridge.out` vẫn giữ gate router thật sự tính ra — `force_gate` chỉ
        thay cái đi vào MoE, không thay cái router nghĩ. Nhờ vậy một lần
        forward nền cũng đủ để biết router chọn 6 expert nào.
        """
        bridge.begin("route", mask=mask,
                     pixels=core_fwd.get("pixel_values"))
        bridge.force_gate = gate_vec
        try:
            # logits_to_keep=1: vocab 262k × ~1000 vị trí là ~1GB logits mà ta
            # không dùng tới — Δlogit được tính TAY từ hidden state ở dưới.
            out = model(**core_fwd, output_hidden_states=True, use_cache=False,
                        logits_to_keep=1)
            h = out.hidden_states[-1][0].detach()
            gate = bridge.out[1][0].detach().float() if bridge.out else None
            stats = bridge.out[2] if bridge.out else {}
        finally:
            bridge.end()
        del out
        return h, gate, stats

    rows, n_new, t0 = reports_from_records(local, key_col), 0, time.time()
    loader = make_loader(todo, collate, cfg, bs=1) if len(todo) else []
    for i, batch in enumerate(loader):
        row = todo.iloc[i]
        key = row[key_col]

        # ── (1) báo cáo thật, gate thật, generate như mode=test ──────────────
        recs, _ = generate_records([row["img_abs_path"]])
        rec = recs[0]

        # ── (2) forward nền: teacher-forcing prompt + ảnh + báo cáo tham chiếu
        core, extra = split_batch(to_device(batch, dev))
        mask = extra.get("router_mask")
        labels = core["labels"][0]
        ids = core["input_ids"][0]
        core_fwd = {k: v for k, v in core.items() if k != "labels"}

        # Vị trí h dự đoán token báo cáo là vị trí NGAY TRƯỚC token đó.
        rep = (labels != -100).nonzero(as_tuple=True)[0]
        pred = torch.unique(torch.clamp(rep - 1, min=0))
        img = (ids == img_token_id).nonzero(as_tuple=True)[0] \
            if img_token_id is not None else ids.new_empty(0)
        if grid is None and img.numel():
            grid = grid_shape(int(img.numel()))
            if grid is None and not warned_grid:
                warned_grid.append(1)
                warn("probe", f"{int(img.numel())} image token không phải số "
                              f"chính phương — bỏ bản đồ không gian, vẫn giữ "
                              f"ma trận bệnh/aspect và top token")
        if pred.numel() == 0:
            warn("probe", f"{key}: không có vị trí báo cáo nào (target rỗng?) "
                          f"— bỏ qua ảnh này")
            continue

        zeros = torch.zeros(1, E, device=dev, dtype=torch.float32)
        h_base, gate, stats = forward_with(core_fwd, mask, zeros)
        if gate is None:
            raise RuntimeError("router không chạy trong forward probe — hook "
                               "RouterBridge không được gọi")
        chosen = [int(e) for e in torch.argsort(gate, descending=True).tolist()
                  if float(gate[e]) > 0][:K]
        # softmax ĐẦY ĐỦ trên E expert, trước khi cắt top-k (src/router.py).
        # `gate` đã bị top-k làm rỗng E-k ô nên không đọc được "router tin
        # expert nào bao nhiêu"; `prob` thì có số cho cả expert không được chọn.
        prob = [round(float(v), 6) for v in (stats.get("route_prob") or [])]

        # ── (3) bật lần lượt TỪNG expert, trọng số 1.0 ───────────────────────
        full = str(key) in full_keys
        targets = list(range(E)) if full else chosen
        experts = []
        for e in targets:
            oh = torch.zeros(1, E, device=dev, dtype=torch.float32)
            oh[0, e] = 1.0
            h_e, _, _ = forward_with(core_fwd, mask, oh)
            dh = (h_e - h_base).float()                     # [N, D]
            del h_e

            item = {"expert": e,
                    "gate": round(float(gate[e]), 5),   # 0 nếu không lọt top-k
                    "prob": prob[e] if e < len(prob) else None,
                    "in_topk": bool(e in chosen)}
            if img.numel() and grid is not None:
                sp = dh[img].norm(dim=-1)
                vals, mx = pack_map(sp.cpu().numpy())
                item.update(spatial=vals, spatial_max=round(mx, 5),
                            spatial_l2=round(float(sp.norm()), 5))

            dh_rep = dh[pred].mean(0)
            dlogit = (W @ dh_rep.to(W.dtype)).float()       # [V]
            item.update(
                disease=[round(v, 5) for v in rows_mean(dlogit, dis_ids)],
                aspect=[round(v, 5) for v in rows_mean(dlogit, asp_ids)],
                fields={g: [round(v, 5) for v in rows_mean(dlogit, ids)]
                        for g, ids in fld_ids.items()},
                top_tokens=top_tokens(dlogit, tokenizer, r.probe_top_tokens),
                delta_norm=round(float(dh_rep.norm()), 5))
            experts.append(item)
            del dh, dlogit, dh_rep

        del h_base
        report = {"reference": str(row[f.target_column]).strip(), **rec}
        record = {
            "key_id": str(key),
            "schema_version": SCHEMA_VERSION,
            # "full" = đã bật đủ E expert trên ảnh này -> hồ sơ VÔ ĐIỀU KIỆN,
            # so trực tiếp được giữa các expert. "topk" = chỉ có expert router
            # đã chọn -> chỉ dùng được cho thống kê router. summary tách hai
            # loại này ra, KHÔNG trộn vào cùng một trung bình.
            "probe_mode": "full" if full else "topk",
            "sweep_cell": cfg.sweep.tag,
            "model": f.model_id,
            "n_expert": E, "top_k": K,
            "train_epoch": train_state.get("epoch"),
            "train_samples": train_state.get("samples"),
            "swa_n": train_state.get("swa_n"),
            "weights": train_state.get("weights_src"),
            "grid": list(grid) if grid else None,
            "disease_labels": list(disease_names),
            "aspect_labels": list(aspect_names),
            "field_labels": {g: list(names)
                             for g, (_t, names) in PROBE_FIELDS.items()},
            "field_titles": {g: t for g, (t, _n) in PROBE_FIELDS.items()},
            "router": {"gate": [round(float(v), 5) for v in gate.tolist()],
                       "prob": prob,
                       "chosen": chosen,
                       "entropy": round(float(stats.get("route_entropy", 0)), 5),
                       "gate_max": round(float(stats.get("gate_max", 0)), 5)},
            "report": report,
            "experts": experts,
        }
        stream.write(safe_name(key), record)
        rows.append({key_col: str(key), **report})
        n_new += 1

        if n_new == 1 or n_new % max(int(r.probe_log_every), 1) == 0:
            rate = n_new / max(time.time() - t0, 1e-9)
            left = len(todo) - (i + 1)
            log("probe", f"{i+1}/{len(todo)} | {rate*60:.1f} ảnh/phút | "
                         f"ETA {left/max(rate,1e-9)/3600:.1f}h | {key}")
            event("probe_progress", done=i + 1, todo=len(todo),
                  rate_per_min=round(rate * 60, 3))
        if n_new % max(int(r.probe_flush_every), 1) == 0:
            pd.DataFrame(rows).to_csv(
                os.path.join(r.out_dir, "reports.csv"), index=False)

    stream.close()
    res = pd.DataFrame(rows)
    if len(res):
        res.to_csv(os.path.join(r.out_dir, "reports.csv"), index=False)
    log("probe", f"XONG {len(res)} bản ghi ({n_new} mới trong phiên này)")
    event("probe_done", n_records=len(res), n_new=n_new)
    return res

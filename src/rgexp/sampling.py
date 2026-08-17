"""Multi-label stratified sampling cho MIMIC-CXR (CheXpert labels)."""

import os
import numpy as np
import pandas as pd


def _iterative_split_two_way(labels: np.ndarray, fraction_a: float, random_state: int):
    """Chia 2 phần giữ phân phối multi-label, ưu tiên nhãn hiếm nhất trước."""
    rng = np.random.RandomState(random_state)
    n_samples, n_labels = labels.shape

    target_a = int(round(fraction_a * n_samples))
    target_b = n_samples - target_a

    desired_a = np.array([
        int(round(fraction_a * labels[:, c].sum())) for c in range(n_labels)
    ])
    remaining_a = desired_a.copy()
    remaining_b = labels.sum(axis=0).astype(int) - desired_a

    idx_a, idx_b = [], []
    remaining_idx = set(range(n_samples))

    while remaining_idx:
        idx_list = list(remaining_idx)
        label_counts_remaining = labels[idx_list].sum(axis=0)

        active_labels = [c for c in range(n_labels) if label_counts_remaining[c] > 0]
        if not active_labels:
            rng.shuffle(idx_list)
            for i in idx_list:
                (idx_a if len(idx_a) < target_a else idx_b).append(i)
            remaining_idx.clear()
            break

        c = min(active_labels, key=lambda c: label_counts_remaining[c])
        candidates = [i for i in idx_list if labels[i, c] == 1]
        rng.shuffle(candidates)

        for i in candidates:
            if i not in remaining_idx:
                continue
            want_a, want_b = remaining_a[c], remaining_b[c]
            if want_a == 0 and want_b == 0:
                go_to_a = (target_a - len(idx_a)) >= (target_b - len(idx_b))
            else:
                go_to_a = want_a >= want_b

            room_a = target_a - len(idx_a)
            room_b = target_b - len(idx_b)

            if go_to_a and room_a > 0:
                idx_a.append(i)
                remaining_a = np.maximum(remaining_a - labels[i], 0)
            elif room_b > 0:
                idx_b.append(i)
                remaining_b = np.maximum(remaining_b - labels[i], 0)
            elif room_a > 0:
                idx_a.append(i)
                remaining_a = np.maximum(remaining_a - labels[i], 0)
            else:
                break   # cả hai bên đã đầy

            remaining_idx.discard(i)

    return np.array(idx_a, dtype=int), np.array(idx_b, dtype=int)


def load_label_matrix(labels_npz_path: str, source: str = "y_gt",
                      uncertain_as_positive: bool = False):
    """Đọc labels.npz -> ma trận nhị phân (n_samples, n_labels)."""
    data = np.load(labels_npz_path, allow_pickle=True)
    assert source in data.files, f"'{source}' không có trong labels.npz. Có: {data.files}"

    y = data[source]
    class_names = list(data["class_names"])
    codes_counted_as_1 = {class_names.index("positive")}
    if uncertain_as_positive and "uncertain" in class_names:
        codes_counted_as_1.add(class_names.index("uncertain"))

    label_names = list(data["disease_labels"])
    labels = np.isin(y, list(codes_counted_as_1)).astype(int)
    img_rel = data["img_rel"] if "img_rel" in data.files else None

    return labels, label_names, img_rel


def attach_labels_to_df(df: pd.DataFrame, labels: np.ndarray, img_rel: np.ndarray,
                        df_img_col: str = "img_rel"):
    """Join nhãn vào df theo img_rel (join khoá, không dựa vào thứ tự dòng)."""
    assert img_rel is not None, "labels.npz không có mảng 'img_rel' để join an toàn."

    label_df = pd.DataFrame({"__img_rel__": img_rel, "__row__": np.arange(len(img_rel))})
    merged = df.merge(label_df, left_on=df_img_col, right_on="__img_rel__", how="inner")

    n_dropped = len(df) - len(merged)
    if n_dropped > 0:
        print(f"[attach_labels_to_df] CẢNH BÁO: {n_dropped} dòng không tìm thấy nhãn "
              f"khớp theo '{df_img_col}' và đã bị loại.")

    labels_aligned = labels[merged["__row__"].to_numpy()]
    df_aligned = merged.drop(columns=["__img_rel__", "__row__"]).reset_index(drop=True)
    return df_aligned, labels_aligned


def check_label_alignment(df: pd.DataFrame, labels: np.ndarray):
    assert len(df) == labels.shape[0], (
        f"Mismatch: df có {len(df)} dòng nhưng label matrix có {labels.shape[0]} dòng."
    )


def stratified_prefix(df, labels, n_samples, random_state=42):
    """Lấy n_samples đầu của thứ tự stratified — cân bằng ở mọi n."""
    check_label_alignment(df, labels)
    if n_samples >= len(df):
        return df.reset_index(drop=True), labels, np.arange(len(df))

    idx = stratified_order(labels, random_state)[:n_samples]
    return df.iloc[idx].reset_index(drop=True), labels[idx], idx


def stratified_train_val_test_split(df: pd.DataFrame, labels: np.ndarray,
                                    val_frac: float = 0.15, test_frac: float = 0.15,
                                    random_state: int = 42):
    check_label_alignment(df, labels)

    test_idx, trainval_idx = _iterative_split_two_way(labels, test_frac, random_state)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    labels_test = labels[test_idx]

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    labels_trainval = labels[trainval_idx]

    val_frac_adjusted = val_frac / (1 - test_frac)
    val_idx, train_idx = _iterative_split_two_way(labels_trainval, val_frac_adjusted,
                                                  random_state)

    return {
        "train": (df_trainval.iloc[train_idx].reset_index(drop=True), labels_trainval[train_idx]),
        "val":   (df_trainval.iloc[val_idx].reset_index(drop=True),   labels_trainval[val_idx]),
        "test":  (df_test, labels_test),
    }


def persist_splits(splits: dict, label_names: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for split_name, (d, l) in splits.items():
        csv_path = os.path.join(out_dir, f"{split_name}_split.csv")
        npz_path = os.path.join(out_dir, f"{split_name}_labels.npz")
        d.to_csv(csv_path, index=False)
        np.savez(npz_path, labels=l, label_names=np.array(label_names, dtype=object))
        manifest[split_name] = {"csv": csv_path, "npz": npz_path, "n": len(d)}
        print(f"  [{split_name}] n={len(d):6d}  -> {csv_path}")
    return manifest


def load_splits(out_dir: str, split_names=("train", "val", "test")):
    """Trả về None nếu thiếu bất kỳ file nào -> caller sẽ tạo split mới."""
    result = {}
    for name in split_names:
        csv_path = os.path.join(out_dir, f"{name}_split.csv")
        npz_path = os.path.join(out_dir, f"{name}_labels.npz")
        if not (os.path.exists(csv_path) and os.path.exists(npz_path)):
            return None
        d = pd.read_csv(csv_path)
        data = np.load(npz_path, allow_pickle=True)
        result[name] = (d, data["labels"])
    return result


def report_label_balance(labels_full: np.ndarray, labels_subset: np.ndarray,
                         label_names: list):
    rows = [{
        "label": name,
        "pct_full (%)":   round(labels_full[:, i].mean() * 100, 2),
        "pct_subset (%)": round(labels_subset[:, i].mean() * 100, 2),
        "abs_diff (pp)":  round(abs(labels_full[:, i].mean() - labels_subset[:, i].mean()) * 100, 2),
    } for i, name in enumerate(label_names)]
    return (pd.DataFrame(rows)
            .sort_values("pct_full (%)", ascending=False)
            .reset_index(drop=True))


def _label_signature(row: np.ndarray) -> tuple:
    """Khoá nhóm = tổ hợp nhãn. Hàng toàn 0 (No Finding) thành một nhóm riêng."""
    return tuple(row.tolist())


def stratified_order(labels: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Trả về permutation sao cho MỌI prefix đều giữ xấp xỉ phân phối nhãn gốc.

    Cơ chế: gom mẫu theo tổ hợp nhãn, rồi ở mỗi vị trí lấy từ nhóm đang "thiếu"
    nhiều nhất so với hạn ngạch của nó (largest-remainder). Sai lệch của bất kỳ
    prefix nào so với tỉ lệ gốc bị chặn ở dưới 1 mẫu cho mỗi nhóm.
    """
    rng = np.random.RandomState(random_state)
    n = len(labels)

    groups = {}
    for i in range(n):
        groups.setdefault(_label_signature(labels[i]), []).append(i)

    keys = sorted(groups.keys())
    for k in keys:
        rng.shuffle(groups[k])

    sizes = np.array([len(groups[k]) for k in keys], dtype=float)
    quota = sizes / n              # tỉ lệ mục tiêu của từng nhóm
    taken = np.zeros(len(keys))
    cursor = {k: 0 for k in keys}

    order = []
    for pos in range(n):
        # deficit = số mẫu nhóm g "đáng lẽ" đã có ở vị trí này, trừ đi số đã lấy
        deficit = quota * (pos + 1) - taken
        deficit[taken >= sizes] = -np.inf      # nhóm đã cạn
        g = int(np.argmax(deficit))

        k = keys[g]
        order.append(groups[k][cursor[k]])
        cursor[k] += 1
        taken[g] += 1

    return np.array(order, dtype=int)
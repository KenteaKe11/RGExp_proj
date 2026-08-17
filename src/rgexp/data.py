"""
Load dataset, split cố định, resolve đường dẫn ảnh, Dataset/collate cho LoRA —
port từ Section 9 và phần dữ liệu của Section 12.

`text_gt` / `aug_gt` là ground truth: target khi train và tham chiếu khi eval.
Chúng **không bao giờ** được đưa vào prompt lúc inference.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

# torch chỉ cần cho Dataset/DataLoader (train & infer). Import mềm để các script
# không cần GPU — prepare_splits, e2e local — chạy được trên máy chưa cài torch.
try:
    from torch.utils.data import DataLoader, Dataset, Subset
    _HAS_TORCH = True
except ImportError:                                  # pragma: no cover
    _HAS_TORCH = False
    Dataset = object                                 # type: ignore[assignment]
    DataLoader = Subset = None                       # type: ignore[assignment]

_NO_TORCH_MSG = ("Cần torch cho bước này (Dataset/DataLoader). "
                 "Cài bằng: pip install -r src/requirements.txt")

from .config import BATCH, DATA, PATHS, TRAIN_CFG
from .sampling import (
    attach_labels_to_df,
    load_label_matrix,
    load_splits,
    persist_splits,
    report_label_balance,
    stratified_order,
    stratified_prefix,
    stratified_train_val_test_split,
)


# ---- Paths -------------------------------------------------------------------
def resolve_image_path(rel_path: str) -> str:
    return os.path.join(PATHS.IMAGE_ROOT, rel_path)


def with_abs_path(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["img_abs_path"] = df[DATA.IMAGE_COLUMN].apply(resolve_image_path)
    return df


def assert_images_available(df: pd.DataFrame, n_probe: int = 5) -> None:
    """Fail sớm và rõ ràng nếu IMAGE_ROOT chưa trỏ đúng chỗ."""
    missing = [p for p in df["img_abs_path"].head(n_probe) if not os.path.exists(p)]
    assert not missing, (
        f"Không tìm thấy ảnh dưới IMAGE_ROOT={PATHS.IMAGE_ROOT}.\n"
        f"Ví dụ thiếu: {missing[0]}\n"
        "Set biến môi trường RGEXP_IMAGE_ROOT trỏ tới thư mục chứa 'files/p10/...'."
    )


# ---- Splits ------------------------------------------------------------------
def load_or_create_splits(verbose: bool = True):
    """
    Split được tạo **một lần** rồi persist. Nếu file đã có thì load nguyên vẹn,
    không bao giờ re-randomize — tránh rò rỉ train sang test giữa các lần chạy.
    """
    df_full = pd.read_csv(PATHS.CSV_PATH)
    labels_full, label_names, img_rel = load_label_matrix(PATHS.LABELS_NPZ, source="y_gt")
    df_full, labels_full = attach_labels_to_df(df_full, labels_full, img_rel,
                                               df_img_col=DATA.IMAGE_COLUMN)

    splits = load_splits(PATHS.SPLIT_DIR)
    if splits is None:
        if verbose:
            print(f"Chưa có split cố định — tạo mới (random_state={DATA.RANDOM_SEED}).")
        splits = stratified_train_val_test_split(
            df_full, labels_full,
            val_frac=DATA.VAL_FRAC, test_frac=DATA.TEST_FRAC,
            random_state=DATA.RANDOM_SEED,
        )
        persist_splits(splits, label_names, PATHS.SPLIT_DIR)
    elif verbose:
        print(f"Đã có split cố định tại {PATHS.SPLIT_DIR} — load lại, KHÔNG tạo mới.")

    df_train, labels_train = splits["train"]
    df_val, labels_val = splits["val"]
    df_test, labels_test = splits["test"]
    if verbose:
        print(f"train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    return SimpleNamespace(
        df_full=df_full, labels_full=labels_full, label_names=label_names,
        df_train=df_train, labels_train=labels_train,
        df_val=df_val, labels_val=labels_val,
        df_test=df_test, labels_test=labels_test,
    )


def build_infer_frame(data, n_samples=DATA.N_SAMPLES, verbose: bool = True):
    """Inference chỉ chạy trên test split (tuỳ chọn lấy prefix stratified n mẫu)."""
    df_infer, labels_infer = data.df_test, data.labels_test
    if n_samples is not None and n_samples < len(data.df_test):
        df_infer, labels_infer, _ = stratified_prefix(
            data.df_test, data.labels_test, n_samples=n_samples,
            random_state=DATA.RANDOM_SEED,
        )
        if verbose:
            print(f"\nStratified subset n={n_samples} từ test split "
                  f"(n_test={len(data.df_test)}):")
            print(report_label_balance(data.labels_test, labels_infer,
                                       data.label_names).to_string(index=False))

    df_infer = with_abs_path(df_infer)
    if verbose:
        print(f"\nInference trên {len(df_infer)} ảnh "
              f"(test split, n_test_full={len(data.df_test)}).")
    return df_infer, labels_infer


def build_lora_frames(data, verbose: bool = True):
    """Train/val cho LoRA, có thể cắt bớt bằng TRAIN_CFG.N_TRAIN/VAL_SAMPLES."""
    df_lora_train, labels_lora_train = data.df_train, data.labels_train
    df_lora_val, labels_lora_val = data.df_val, data.labels_val

    if TRAIN_CFG.N_TRAIN_SAMPLES is not None:
        df_lora_train, labels_lora_train, _ = stratified_prefix(
            data.df_train, data.labels_train, n_samples=TRAIN_CFG.N_TRAIN_SAMPLES,
            random_state=DATA.RANDOM_SEED,
        )
    if TRAIN_CFG.N_VAL_SAMPLES is not None:
        df_lora_val, labels_lora_val, _ = stratified_prefix(
            data.df_val, data.labels_val, n_samples=TRAIN_CFG.N_VAL_SAMPLES,
            random_state=DATA.RANDOM_SEED,
        )

    df_lora_train = with_abs_path(df_lora_train)
    df_lora_val = with_abs_path(df_lora_val)

    if verbose:
        print(f"[lora data] train: {len(df_lora_train)}/{len(data.df_train)}")
        print(f"[lora data] val  : {len(df_lora_val)}/{len(data.df_val)}")

    return df_lora_train, labels_lora_train, df_lora_val, labels_lora_val


# ---- Dataset / collate -------------------------------------------------------
class CXRLoraDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        assert _HAS_TORCH, _NO_TORCH_MSG
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["img_abs_path"]).convert("RGB")
        target = str(row["text_gt"]) if pd.notna(row.get("text_gt")) else ""
        return image, target


def make_collate_fn(processor, text_tokenizer, prompt_text: str):
    """Mask toàn bộ token prompt bằng -100; loss chỉ tính trên phần target."""

    def collate_fn(batch):
        images, targets = zip(*batch)
        image_lists = [[im] for im in images]

        prompt_inputs = processor(text=[prompt_text] * len(images), images=image_lists,
                                  padding=True, return_tensors="pt")
        full_texts = [prompt_text + t + text_tokenizer.eos_token for t in targets]
        full_inputs = processor(text=full_texts, images=image_lists,
                                padding=True, return_tensors="pt")

        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_inputs["input_ids"].shape[-1]] = -100
        labels[labels == text_tokenizer.pad_token_id] = -100
        full_inputs["labels"] = labels
        return full_inputs

    return collate_fn


def make_epoch_permutation(labels: np.ndarray, epoch: int) -> list:
    """Thứ tự stratified, tất định theo (seed, epoch) — nền tảng của resume."""
    return stratified_order(labels, random_state=DATA.RANDOM_SEED + epoch).tolist()


def make_train_loader(dataset, indices, collate_fn):
    assert _HAS_TORCH, _NO_TORCH_MSG
    num_workers = TRAIN_CFG.NUM_WORKERS
    kwargs = dict(
        batch_size=BATCH.TRAIN_BATCH_SIZE,
        shuffle=False,                 # thứ tự do permutation quyết định, không shuffle
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(Subset(dataset, indices), **kwargs)


def make_val_loader(df_val, collate_fn):
    assert _HAS_TORCH, _NO_TORCH_MSG
    return DataLoader(
        CXRLoraDataset(df_val),
        batch_size=BATCH.TRAIN_BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    )


def warn_if_subset_too_small(labels_full: np.ndarray, n_samples: int,
                             label_names: list) -> bool:
    p = labels_full.mean(axis=0)
    p_min = p[p > 0].min()
    rare = label_names[int(np.argmin(np.where(p > 0, p, np.inf)))]
    n_needed = int(np.ceil(2 / p_min))

    if n_samples < n_needed:
        print(f"[stratify] CẢNH BÁO: n={n_samples} quá nhỏ để bảo toàn tỉ lệ nhãn.")
        print(f"           Nhãn hiếm nhất '{rare}' chiếm {p_min * 100:.2f}% "
              f"-> cần n ≳ {n_needed} mới có kỳ vọng ≥2 mẫu dương.")
        print("           Subset vẫn cân bằng ở mức tốt nhất có thể, nhưng nhiều "
              "nhãn sẽ vắng mặt hoàn toàn.")
    return n_samples >= n_needed

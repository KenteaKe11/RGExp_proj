"""
DEPRECATED shim — logic đã chuyển vào `rgexp/sampling.py`.

Giữ lại để code/notebook cũ `from stratified_sampling import ...` vẫn chạy.
Code mới nên dùng: `from rgexp.sampling import ...`
"""

from rgexp.sampling import (  # noqa: F401
    attach_labels_to_df,
    check_label_alignment,
    load_label_matrix,
    load_splits,
    persist_splits,
    report_label_balance,
    stratified_order,
    stratified_prefix,
    stratified_train_val_test_split,
)

__all__ = [
    "attach_labels_to_df",
    "check_label_alignment",
    "load_label_matrix",
    "load_splits",
    "persist_splits",
    "report_label_balance",
    "stratified_order",
    "stratified_prefix",
    "stratified_train_val_test_split",
]

"""
Inference loop + guard — port từ Section 13.

Mỗi batch BATCH.INFER_BATCH_SIZE ảnh được generate một lần như batch thật, rồi
từng output đi qua guard Section 10; chỉ mẫu FAIL mới sinh lại.

Kết quả ghi ra PATHS.OUTPUT_CSV sau MỖI batch — watchdog kill giữa chừng chỉ mất
một batch đang dở.
"""

from __future__ import annotations

import pandas as pd

from .config import BATCH, PATHS
from .hardware import VRAM_KILL_FLAG
from .tracking import wandb_log


def run_inference(generator, df_infer: pd.DataFrame,
                  output_csv: str = PATHS.OUTPUT_CSV) -> pd.DataFrame:
    results = []

    for start in range(0, len(df_infer), BATCH.INFER_BATCH_SIZE):
        if VRAM_KILL_FLAG.is_set():
            print("[infer] watchdog trip — dừng vòng lặp.")
            break

        chunk = df_infer.iloc[start:start + BATCH.INFER_BATCH_SIZE]
        records = generator.generate_with_guard(chunk["img_abs_path"].tolist())

        for (_, row), rec in zip(chunk.iterrows(), records):
            results.append({
                "subject_id": row["subject_id"],
                "study_id":   row["study_id"],
                "image_id":   row["image_id"],
                "img_rel":    row["img_rel"],
                "view":       row.get("view"),
                **rec,
            })

        pd.DataFrame(results).to_csv(output_csv, index=False)
        wandb_log({"infer/n_done": len(results)})

    results_df = pd.DataFrame(results)
    n_total = len(results_df)
    n_passed = int(results_df["guard_passed"].sum()) if n_total else 0
    avg_attempts = results_df["attempts"].mean() if n_total else float("nan")
    print(f"[infer] {n_total} báo cáo | guard pass {n_passed}/{n_total} | "
          f"trung bình {avg_attempts:.2f} lần thử")
    print(f"[infer] CSV: {output_csv}")
    return results_df

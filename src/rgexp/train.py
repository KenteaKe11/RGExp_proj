"""
Training loop — step-based, resume theo sample index (port Section 12).

Supervised fine-tune: input = [ảnh + PROMPT.TEXT], target = `text_gt`. Loss là
causal-LM loss tính CHỈ trên phần target (token prompt bị mask -100).

Đếm step
--------
1 step = 1 optimizer.step() = TRAIN_BATCH_SIZE * GRAD_ACCUM_STEPS samples.
MAX_STEPS=None -> total_steps = steps_per_epoch * NUM_EPOCHS, tính từ kích thước
train split thật.

Resume
------
Thứ tự sample do một permutation tất định theo (seed, epoch) quyết định, không
dùng `shuffle=True`. Nhờ vậy `next_sample_idx` trong `train_state.json` đủ để tái
tạo chính xác vị trí đang dở.

Mỗi checkpoint lưu 3 thứ: adapter weights, `optimizer.pt` (optimizer + scheduler),
và `train_state.json` (step / sample index / epoch / seed).

Sample index chính xác đến mức step, không đến mức micro-batch: nếu bị kill giữa
vòng grad-accum, các micro-batch của step dở đó sẽ train lại — đúng về ngữ nghĩa
vì `optimizer.step()` chưa chạy. Mất mát thực tế tối đa SAVE_EVERY_N_STEPS-1 step.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .config import BATCH, DATA, PATHS, TRAIN_CFG
from .data import (
    CXRLoraDataset,
    make_epoch_permutation,
    make_train_loader,
    make_val_loader,
    warn_if_subset_too_small,
)
from .hardware import VRAM_KILL_FLAG
from .tracking import push_to_hub_dir, wandb_log


def run_training(model, processor, text_tokenizer, device, collate_fn,
                 df_train, labels_train, df_val, label_names,
                 resume_dir: str | None, hf_token: str):
    samples_per_step = BATCH.TRAIN_BATCH_SIZE * TRAIN_CFG.GRAD_ACCUM_STEPS

    train_ds = CXRLoraDataset(df_train)
    val_loader = make_val_loader(df_val, collate_fn)

    # ---- Khôi phục state ------------------------------------------------------
    start_step, start_sample, start_epoch = 0, 0, 0
    if resume_dir:
        with open(os.path.join(resume_dir, "train_state.json")) as f:
            state = json.load(f)
        start_step = state["global_step"]
        start_sample = state["next_sample_idx"]
        start_epoch = state["epoch"]

        assert state["n_train"] == len(train_ds), (
            f"Train split đổi kích thước ({state['n_train']} -> {len(train_ds)}); "
            "checkpoint không dùng lại được. Đặt TRAIN_CFG.RESUME=False."
        )
        assert state["random_seed"] == DATA.RANDOM_SEED, (
            "Seed đổi so với lúc lưu checkpoint — permutation sẽ khác. "
            "Đặt TRAIN_CFG.RESUME=False."
        )
        print(f"[resume] step={start_step} | sample={start_sample}/{len(train_ds)} "
              f"| epoch={start_epoch}")
    else:
        print("[resume] không có checkpoint — train từ đầu.")

    # ---- Tổng số step ---------------------------------------------------------
    steps_per_epoch = len(train_ds) // samples_per_step
    assert steps_per_epoch > 0, (
        f"Train split quá nhỏ: {len(train_ds)} samples < {samples_per_step} samples/step."
    )

    if TRAIN_CFG.MAX_STEPS is None:
        total_steps = steps_per_epoch * TRAIN_CFG.NUM_EPOCHS
        print(f"[steps] MAX_STEPS=None -> {TRAIN_CFG.NUM_EPOCHS} epoch = {total_steps} step")
    else:
        total_steps = TRAIN_CFG.MAX_STEPS
        n_consumed = min(total_steps * samples_per_step, len(train_ds))
        print(f"[steps] MAX_STEPS={total_steps} (≈{total_steps / steps_per_epoch:.2f} epoch)")
        print(f"[stratify] train trên {n_consumed}/{len(train_ds)} mẫu đầu "
              f"của thứ tự stratified.")
        warn_if_subset_too_small(labels_train, n_consumed, label_names)

    print(f"[steps] {steps_per_epoch} step/epoch | {samples_per_step} samples/step "
          f"| n_train={len(train_ds)}")

    assert total_steps > start_step, (
        f"Đã train đủ {start_step}/{total_steps} step. Tăng MAX_STEPS/NUM_EPOCHS, "
        "hoặc đặt TRAIN_CFG.RESUME=False để train lại từ đầu."
    )

    # ---- Optimizer / scheduler ------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=TRAIN_CFG.LR,
                      weight_decay=TRAIN_CFG.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAIN_CFG.WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    if start_step > 0:
        opt_path = os.path.join(resume_dir, "optimizer.pt")
        if os.path.exists(opt_path):
            ckpt = torch.load(opt_path, map_location="cpu")
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            print("[resume] đã load optimizer + scheduler state.")
        else:
            for _ in range(start_step):
                scheduler.step()
            print("[resume] CẢNH BÁO: thiếu optimizer.pt — momentum AdamW bị reset, "
                  "chỉ khôi phục được LR schedule.")

    # ---- Helpers --------------------------------------------------------------
    @torch.no_grad()
    def run_val_loss(max_batches=TRAIN_CFG.EVAL_MAX_BATCHES):
        model.eval()
        losses = []
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            losses.append(model(**batch).loss.item())
        model.train()
        return float(np.mean(losses)) if losses else float("nan")

    def save_checkpoint(ckpt_dir, global_step, next_sample_idx, epoch):
        """Lưu đủ để resume: adapter + optimizer/scheduler + vị trí dữ liệu."""
        os.makedirs(ckpt_dir, exist_ok=True)

        model.save_pretrained(ckpt_dir)
        torch.save(
            {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            os.path.join(ckpt_dir, "optimizer.pt"),
        )
        with open(os.path.join(ckpt_dir, "train_state.json"), "w") as f:
            json.dump({
                "global_step":      global_step,
                "next_sample_idx":  next_sample_idx,
                "epoch":            epoch,
                "n_train":          len(train_ds),
                "samples_per_step": samples_per_step,
                "random_seed":      DATA.RANDOM_SEED,
                "total_steps":      total_steps,
            }, f, indent=2)

        return ckpt_dir

    # ---- Loop -----------------------------------------------------------------
    epoch = start_epoch
    sample_cursor = start_sample
    global_step = start_step
    n_saves = 0

    epoch_perm = make_epoch_permutation(labels_train, epoch)
    train_iter = iter(make_train_loader(train_ds, epoch_perm[sample_cursor:], collate_fn))

    model.train()
    optimizer.zero_grad()
    start_time = time.time()

    while global_step < total_steps:
        if VRAM_KILL_FLAG.is_set():
            print("[train] watchdog trip — dừng vòng lặp.")
            break

        accum_loss = 0.0
        for _ in range(TRAIN_CFG.GRAD_ACCUM_STEPS):
            try:
                batch = next(train_iter)
            except StopIteration:
                # Hết epoch: permutation mới, cursor về 0
                epoch += 1
                sample_cursor = 0
                epoch_perm = make_epoch_permutation(labels_train, epoch)
                train_iter = iter(make_train_loader(train_ds, epoch_perm, collate_fn))
                batch = next(train_iter)

            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / TRAIN_CFG.GRAD_ACCUM_STEPS
            loss.backward()

            accum_loss += loss.item()
            sample_cursor += BATCH.TRAIN_BATCH_SIZE

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params,
                                                   TRAIN_CFG.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        elapsed = time.time() - start_time
        wandb_log({
            "train/loss":           accum_loss,
            "train/lr":             scheduler.get_last_lr()[0],
            "train/grad_norm":      float(grad_norm),
            "train/epoch":          epoch,
            "train/sample_cursor":  sample_cursor,
            "train/epoch_progress": sample_cursor / len(train_ds),
            "train/samples_seen":   (global_step - start_step) * samples_per_step,
            "train/steps_per_sec":  (global_step - start_step) / max(elapsed, 1e-6),
            "train/elapsed_sec":    elapsed,
        }, step=global_step)

        if TRAIN_CFG.EVAL_EVERY_N_STEPS and global_step % TRAIN_CFG.EVAL_EVERY_N_STEPS == 0:
            wandb_log({"val/loss": run_val_loss()}, step=global_step)

        if TRAIN_CFG.SAVE_EVERY_N_STEPS and global_step % TRAIN_CFG.SAVE_EVERY_N_STEPS == 0:
            save_checkpoint(PATHS.CKPT_STAGING_DIR, global_step, sample_cursor, epoch)
            n_saves += 1

            # 'last/' phải được push MỖI lần save — đây là state duy nhất sống sót
            # qua session, bỏ qua một lần push là mất đúng chừng đó tiến độ.
            push_to_hub_dir(PATHS.CKPT_STAGING_DIR, "last",
                            f"last @ step {global_step} | epoch {epoch} | "
                            f"sample {sample_cursor}/{len(train_ds)}", hf_token)

            if n_saves % TRAIN_CFG.PUSH_EVERY_N_SAVES == 0:
                archive = save_checkpoint(
                    os.path.join(PATHS.CKPT_ARCHIVE_DIR, f"step_{global_step}"),
                    global_step, sample_cursor, epoch,
                )
                push_to_hub_dir(archive, f"step_{global_step}",
                                f"archive @ step {global_step}", hf_token)

    save_checkpoint(PATHS.CKPT_STAGING_DIR, global_step, sample_cursor, epoch)
    push_to_hub_dir(PATHS.CKPT_STAGING_DIR, "last",
                    f"final | step {global_step} | epoch {epoch}", hf_token)

    print(f"[train] xong tại step {global_step}/{total_steps} | epoch {epoch} | "
          f"sample {sample_cursor}/{len(train_ds)}")
    model.eval()
    return {"global_step": global_step, "total_steps": total_steps, "epoch": epoch,
            "sample_cursor": sample_cursor,
            "checkpoint_dir": PATHS.CKPT_STAGING_DIR}

# train_ldct.py
import os
import json
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ldct_model import UNet
from ldct_npy_dataset import LDCTPatchNPYDataset

# ================== 配置区域 ==================
PREPARED_ROOT = r"E:\LDCT\prepared_2p5d_3slice_1mm3mm_hu_-160_240"
SPLIT_DIR = r"E:\LDCT\splits"
SAVE_DIR = r"E:\LDCT\experiments\baseline_1mm_only_2p5d_3slice_sigmoid_l1"

os.makedirs(SAVE_DIR, exist_ok=True)

# HU window（仅用于实验记录，不参与训练计算）
CLIP_MIN = -160.0
CLIP_MAX = 240.0

BATCH_SIZE = 8
NUM_WORKERS = 0
LR = 1e-4
NUM_EPOCHS = 50

EARLY_STOP_PATIENCE = 8
MIN_DELTA = 1e-5

torch.backends.cudnn.benchmark = True


# ================== 工具函数 ==================
def load_patient_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


# ================== PSNR ==================
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float()
    target = target.float()
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(10 * np.log10(1.0 / mse))


# ================== 训练 / 验证 ==================
def train_one_epoch(model, loader, optimizer, scaler, l1_fn, device):
    model.train()
    running_l1 = 0.0

    for qd, fd in tqdm(loader, desc="Training", ncols=120):
        qd = qd.to(device, non_blocking=True)
        fd = fd.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(qd)
            loss = l1_fn(out, fd)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_l1 += loss.item()

    n = len(loader)
    return running_l1 / n


@torch.no_grad()
def validate(model, loader, l1_fn, device):
    model.eval()
    running_l1 = 0.0
    running_psnr = 0.0

    for qd, fd in tqdm(loader, desc="Validating", ncols=120):
        qd = qd.to(device, non_blocking=True)
        fd = fd.to(device, non_blocking=True)

        out = model(qd)
        out = out.float()
        fd = fd.float()

        l1 = l1_fn(out, fd)
        psnr = compute_psnr(out, fd)

        running_l1 += l1.item()
        running_psnr += psnr

    n = len(loader)
    avg_l1 = running_l1 / n
    avg_psnr = running_psnr / n
    return avg_l1, avg_psnr


# ================== 主函数 ==================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    config = {
        "prepared_root": PREPARED_ROOT,
        "split_dir": SPLIT_DIR,
        "save_dir": SAVE_DIR,
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "lr": LR,
        "num_epochs": NUM_EPOCHS,
        "model_in_channels": 3,
        "model_out_channels": 1,
        "base_ch": 64,
        "train_thickness": "1mm",
        "val_thickness": "1mm",
        "input_mode": "2.5D_3slice",
        "loss": "L1",
        "activation": "sigmoid",
        "deep_supervision": False,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "min_delta": MIN_DELTA,
        "early_stop_monitor": "val_l1",
        "scheduler_monitor": "val_l1",
    }
    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    train_dataset = LDCTPatchNPYDataset(
        PREPARED_ROOT,
        split="train",
        thickness="1mm",
    )
    val_dataset = LDCTPatchNPYDataset(
        PREPARED_ROOT,
        split="val",
        thickness="1mm",
    )

    sample_qd, sample_fd = train_dataset[0]
    print(f"[DEBUG] single sample qd shape: {sample_qd.shape}")
    print(f"[DEBUG] single sample fd shape: {sample_fd.shape}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    first_qd, first_fd = next(iter(train_loader))
    print(f"[DEBUG] train batch qd shape: {first_qd.shape}")
    print(f"[DEBUG] train batch fd shape: {first_fd.shape}")

    model = UNet(in_channels=3, out_channels=1, base_ch=64).to(device)
    l1_loss_fn = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=1e-6,
    )

    best_val_l1 = float("inf")
    best_val_psnr = -1.0
    best_epoch = -1

    history = []
    no_improve_epochs = 0

    history_path = os.path.join(SAVE_DIR, "train_history.json")
    ckpt_path = os.path.join(SAVE_DIR, "best_unet.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        train_l1 = train_one_epoch(
            model, train_loader, optimizer, scaler,
            l1_loss_fn, device
        )
        val_l1, val_psnr = validate(
            model, val_loader,
            l1_loss_fn, device
        )

        scheduler.step(val_l1)

        epoch_time = (time.time() - start_time) / 60.0
        current_lr = optimizer.param_groups[0]["lr"]

        improved = val_l1 < (best_val_l1 - MIN_DELTA)
        if improved:
            current_best_val_l1 = val_l1
        else:
            current_best_val_l1 = best_val_l1

        print(
            f"[Epoch {epoch:03d}] "
            f"Train L1: {train_l1:.6f} | "
            f"Val L1: {val_l1:.6f}, Val PSNR: {val_psnr:.4f} | "
            f"Best Val L1: {current_best_val_l1 if current_best_val_l1 < 1e9 else -1:.6f} | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.2f} min"
        )

        history.append({
            "epoch": epoch,
            "train_l1": train_l1,
            "val_l1": val_l1,
            "val_psnr": val_psnr,
            "best_val_l1_so_far": None if current_best_val_l1 == float("inf") else current_best_val_l1,
            "lr": current_lr,
            "epoch_time_min": epoch_time,
        })

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        if improved:
            best_val_l1 = val_l1
            best_val_psnr = val_psnr
            best_epoch = epoch
            no_improve_epochs = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_l1": best_val_l1,
                    "best_val_psnr": best_val_psnr,
                    "clip_min": CLIP_MIN,
                    "clip_max": CLIP_MAX,
                    "model_in_channels": 3,
                    "input_mode": "2.5D_3slice",
                    "train_thickness": "1mm",
                    "val_thickness": "1mm",
                    "loss": "L1",
                    "activation": "sigmoid",
                    "deep_supervision": False,
                },
                ckpt_path,
            )
            print(
                f"[INFO] 保存新的 best 模型到 {ckpt_path} | "
                f"Val L1 = {best_val_l1:.6f}, Val PSNR = {best_val_psnr:.4f}"
            )
        else:
            no_improve_epochs += 1
            print(
                f"[INFO] Val L1 未提升，连续 {no_improve_epochs}/{EARLY_STOP_PATIENCE} 个 epoch | "
                f"best_epoch={best_epoch}, best_val_l1={best_val_l1:.6f}"
            )

            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print("[INFO] 触发 Early Stopping，停止训练。")
                break

    print(
        f"[INFO] 训练结束。Best epoch = {best_epoch}, "
        f"Best Val L1 = {best_val_l1:.6f}, Best Val PSNR = {best_val_psnr:.4f}"
    )
    print("[INFO] 日志已写入 train_history.json")


if __name__ == "__main__":
    main()
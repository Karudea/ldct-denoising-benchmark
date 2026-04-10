import os
import json
import time
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ldct_model import UNet
from ldct_npy_dataset import LDCTPatchNPYDataset


# ================== 统一路径配置 ==================
def get_runtime_paths() -> Dict[str, Path]:
    """
    本地 Windows:
        默认 E:\\LDCT

    Vast / 云端 Linux:
        export IS_CLOUD=1
        默认:
            /workspace/data
            /workspace/experiments
    """
    is_cloud = os.getenv("IS_CLOUD", "0") == "1"

    if is_cloud:
        data_root = Path("/workspace/data")
        exp_root = Path("/workspace/experiments")
    else:
        data_root = Path(r"E:\LDCT")
        exp_root = data_root / "experiments"

    return {
        "data_root": data_root,
        "prepared_root": data_root / "prepared_2p5d_5slice_1mm3mm_hu_-160_240",
        "split_dir": data_root / "splits",
        "exp_root": exp_root,
    }


PATHS = get_runtime_paths()
PREPARED_ROOT = PATHS["prepared_root"]
SPLIT_DIR = PATHS["split_dir"]
EXP_ROOT = PATHS["exp_root"]

# ================== 实验配置 ==================
EXP_NAME = "C4_mixed_cond_th_2p5d_5slice_residual_nosig_l1_trainall_valall"
SAVE_DIR = EXP_ROOT / EXP_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# HU window（仅用于实验记录，不参与训练计算）
CLIP_MIN = -160.0
CLIP_MAX = 240.0

BATCH_SIZE = 8
NUM_WORKERS = 0
LR = 1e-4
NUM_EPOCHS = 50

SEEDS = [0]
DETERMINISTIC = False

EARLY_STOP_PATIENCE = 8
MIN_DELTA = 1e-5

# ===== C4 mixed cond thickness =====
TRAIN_THICKNESS = "all"
VAL_THICKNESS = "all"
COND_THICKNESS = True

# ===== loss / output =====
USE_SIGMOID = False
RESIDUAL_LEARNING = True
LOSS_NAME = "L1"

torch.backends.cudnn.benchmark = True


def set_seed(seed: int, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    在归一化空间 [0,1] 上计算 PSNR
    """
    pred = pred.float()
    target = target.float()
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(10 * np.log10(1.0 / mse))


def get_output_activation_name(use_sigmoid: bool) -> str:
    return "sigmoid" if use_sigmoid else "none"


def create_grad_scaler(device: torch.device):
    """
    兼容新旧 PyTorch
    """
    if device.type != "cuda":
        return None

    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: torch.device):
    """
    兼容新旧 PyTorch autocast
    """
    if device.type != "cuda":
        return torch.cuda.amp.autocast(enabled=False)

    try:
        return torch.amp.autocast("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def check_paths():
    if not PREPARED_ROOT.exists():
        raise FileNotFoundError(f"[ERROR] PREPARED_ROOT 不存在: {PREPARED_ROOT}")

    if not SPLIT_DIR.exists():
        print(f"[WARN] SPLIT_DIR 不存在: {SPLIT_DIR}（当前训练脚本未直接使用 split 文件，可暂时忽略）")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)


def create_scheduler(optimizer):
    """
    兼容旧版 PyTorch：不传 verbose
    """
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )


def infer_model_in_channels(dataset: LDCTPatchNPYDataset) -> int:
    sample = dataset[0]
    if isinstance(sample, tuple):
        x = sample[0]
    else:
        x = sample
    if x.ndim != 3:
        raise RuntimeError(f"[ERROR] 样本输入维度异常: {x.shape}")
    return int(x.shape[0])


def train_one_epoch(model, loader, optimizer, scaler, l1_fn, device):
    model.train()
    running_l1 = 0.0

    for qd, fd in tqdm(loader, desc="Training", ncols=120):
        qd = qd.to(device, non_blocking=True)
        fd = fd.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device):
            out = model(qd)
            loss = l1_fn(out, fd)

        if device.type == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_l1 += loss.item()

    n = len(loader)
    if n == 0:
        raise RuntimeError("[ERROR] train_loader 为空，无法训练。")
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
    if n == 0:
        raise RuntimeError("[ERROR] val_loader 为空，无法验证。")
    return running_l1 / n, running_psnr / n


def main():
    check_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_activation = get_output_activation_name(USE_SIGMOID)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] PREPARED_ROOT = {PREPARED_ROOT}")
    print(f"[INFO] SAVE_DIR = {SAVE_DIR}")
    print(f"[INFO] TRAIN_THICKNESS = {TRAIN_THICKNESS}")
    print(f"[INFO] VAL_THICKNESS = {VAL_THICKNESS}")
    print(f"[INFO] COND_THICKNESS = {COND_THICKNESS}")
    print(f"[INFO] USE_SIGMOID = {USE_SIGMOID}")
    print(f"[INFO] OUTPUT_ACTIVATION = {output_activation}")
    print(f"[INFO] RESIDUAL_LEARNING = {RESIDUAL_LEARNING}")

    summary_path = SAVE_DIR / f"summary_seeds_{TRAIN_THICKNESS}_{VAL_THICKNESS}.json"
    all_seed_results = []

    for seed in SEEDS:
        print("\n" + "=" * 90)
        print(f"[INFO] Running seed = {seed}")
        print("=" * 90)

        set_seed(seed, deterministic=DETERMINISTIC)

        seed_dir = SAVE_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        history_path = seed_dir / f"train_history_{EXP_NAME}_seed{seed}.json"
        ckpt_path = seed_dir / f"best_{EXP_NAME}_seed{seed}.pth"

        train_dataset = LDCTPatchNPYDataset(
            root=PREPARED_ROOT,
            split="train",
            thickness=TRAIN_THICKNESS,
            cond_thickness=COND_THICKNESS,
        )
        val_dataset = LDCTPatchNPYDataset(
            root=PREPARED_ROOT,
            split="val",
            thickness=VAL_THICKNESS,
            cond_thickness=COND_THICKNESS,
        )

        if len(train_dataset) == 0:
            raise RuntimeError("[ERROR] train_dataset 长度为 0，请检查数据目录结构。")
        if len(val_dataset) == 0:
            raise RuntimeError("[ERROR] val_dataset 长度为 0，请检查数据目录结构。")

        model_in_channels = infer_model_in_channels(train_dataset)

        sample_qd, sample_fd = train_dataset[0]
        print(f"[DEBUG] single sample qd shape: {sample_qd.shape}")
        print(f"[DEBUG] single sample fd shape: {sample_fd.shape}")
        print(f"[INFO] MODEL_IN_CHANNELS = {model_in_channels}")

        with open(seed_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "exp_name": EXP_NAME,
                    "prepared_root": str(PREPARED_ROOT),
                    "split_dir": str(SPLIT_DIR),
                    "save_dir": str(seed_dir),
                    "clip_min": CLIP_MIN,
                    "clip_max": CLIP_MAX,
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "lr": LR,
                    "epochs": NUM_EPOCHS,
                    "train_thickness": TRAIN_THICKNESS,
                    "val_thickness": VAL_THICKNESS,
                    "cond_thickness": COND_THICKNESS,
                    "model_in_channels": model_in_channels,
                    "model_out_channels": 1,
                    "base_ch": 64,
                    "input_mode": "2.5D_5slice_cond_th",
                    "loss": LOSS_NAME,
                    "output_activation": output_activation,
                    "use_sigmoid": USE_SIGMOID,
                    "residual_learning": RESIDUAL_LEARNING,
                    "seed": seed,
                    "deterministic": DETERMINISTIC,
                    "early_stop_patience": EARLY_STOP_PATIENCE,
                    "min_delta": MIN_DELTA,
                    "early_stop_monitor": "val_l1",
                    "scheduler_monitor": "val_l1",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
        )

        first_qd, first_fd = next(iter(train_loader))
        print(f"[DEBUG] train batch qd shape: {first_qd.shape}")
        print(f"[DEBUG] train batch fd shape: {first_fd.shape}")

        model = UNet(
            in_channels=model_in_channels,
            out_channels=1,
            base_ch=64,
            use_sigmoid=USE_SIGMOID,
        ).to(device)

        l1_loss_fn = nn.L1Loss()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scaler = create_grad_scaler(device)
        scheduler = create_scheduler(optimizer)

        best_val_l1 = float("inf")
        best_val_psnr = -1.0
        best_epoch = -1

        history = []
        no_improve_epochs = 0
        prev_lr = optimizer.param_groups[0]["lr"]

        for epoch in range(1, NUM_EPOCHS + 1):
            start_time = time.time()

            train_l1 = train_one_epoch(
                model, train_loader, optimizer, scaler, l1_loss_fn, device
            )
            val_l1, val_psnr = validate(
                model, val_loader, l1_loss_fn, device
            )

            scheduler.step(val_l1)

            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]

            if current_lr != prev_lr:
                print(f"[INFO] LR changed: {prev_lr:.2e} -> {current_lr:.2e}")
                prev_lr = current_lr

            print(
                f"[Seed {seed}] [Epoch {epoch:03d}] "
                f"Train L1: {train_l1:.6f} | "
                f"Val L1: {val_l1:.6f}, Val PSNR: {val_psnr:.4f} | "
                f"Best Val L1: {best_val_l1 if best_val_l1 < 1e9 else -1:.6f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_time/60:.2f} min"
            )

            history.append({
                "epoch": epoch,
                "train_l1": train_l1,
                "val_l1": val_l1,
                "val_psnr": val_psnr,
                "best_val_l1_so_far": None if best_val_l1 == float("inf") else best_val_l1,
                "lr": current_lr,
                "epoch_time_min": epoch_time / 60.0,
            })

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            improved = val_l1 < (best_val_l1 - MIN_DELTA)

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
                        "seed": seed,
                        "deterministic": DETERMINISTIC,
                        "train_thickness": TRAIN_THICKNESS,
                        "val_thickness": VAL_THICKNESS,
                        "cond_thickness": COND_THICKNESS,
                        "model_in_channels": model_in_channels,
                        "exp_name": EXP_NAME,
                        "output_activation": output_activation,
                        "use_sigmoid": USE_SIGMOID,
                        "residual_learning": RESIDUAL_LEARNING,
                        "loss": LOSS_NAME,
                        "input_mode": "2.5D_5slice_cond_th",
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

        seed_result = {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_l1": best_val_l1,
            "best_val_psnr": best_val_psnr,
            "seed_dir": str(seed_dir),
            "checkpoint_path": str(ckpt_path),
            "history_path": str(history_path),
        }
        all_seed_results.append(seed_result)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_seed_results, f, indent=2, ensure_ascii=False)

        print(
            f"[INFO] Seed {seed} 完成："
            f"best_epoch={best_epoch}, "
            f"best_val_l1={best_val_l1:.6f}, "
            f"best_val_psnr={best_val_psnr:.4f}"
        )

    l1s = [x["best_val_l1"] for x in all_seed_results]
    psnrs = [x["best_val_psnr"] for x in all_seed_results]

    l1_mean = float(np.mean(l1s))
    psnr_mean = float(np.mean(psnrs))

    l1_std = float(np.std(l1s, ddof=1)) if len(l1s) > 1 else 0.0
    psnr_std = float(np.std(psnrs, ddof=1)) if len(psnrs) > 1 else 0.0

    print("\n" + "=" * 90)
    print(f"[SUMMARY] SAVE_DIR = {SAVE_DIR}")
    print(f"Val L1 : mean={l1_mean:.6f}, std={l1_std:.6f}, values={l1s}")
    print(f"PSNR   : mean={psnr_mean:.4f}, std={psnr_std:.4f}, values={psnrs}")
    print(f"[INFO] Summary saved to: {summary_path}")
    print("=" * 90)

    print("[INFO] 训练结束，日志已写入各 seed 的 history 文件")


if __name__ == "__main__":
    main()
# main_custom.py
import os
import json
import random
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from CTformer import CTformer
from loader_custom import get_loader
from ctformer_cond_adapter import CTformerCondAdapter


def parse_args():
    parser = argparse.ArgumentParser()

    # 路径参数
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    # 数据设置
    parser.add_argument("--train_thickness", type=str, default="1mm", choices=["1mm", "3mm", "all"])
    parser.add_argument("--val_thickness", type=str, default="1mm", choices=["1mm", "3mm", "all"])
    parser.add_argument("--cond_thickness", action="store_true",
                        help="启用 thickness condition: 输入变为 [LDCT, thickness_map] 两通道")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # 模型参数
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    # smoke test / 小样本快速测试
    parser.add_argument("--train_max_samples", type=int, default=None)
    parser.add_argument("--val_max_samples", type=int, default=None)

    # 随机种子
    parser.add_argument("--seed", type=int, default=42)

    # Early Stopping
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-5)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_base_model(img_size: int):
    model = CTformer(
        img_size=img_size,
        tokens_type="performer",
        embed_dim=64,
        depth=1,
        num_heads=8,
        kernel=4,
        stride=4,
        mlp_ratio=2.0,
        token_dim=64
    )
    return model


def build_model(img_size: int, cond_thickness: bool = False):
    base_model = build_base_model(img_size)

    if cond_thickness:
        model = CTformerCondAdapter(base_model)
    else:
        model = base_model

    return model


def save_args(args, save_dir: str):
    args_path = os.path.join(save_dir, "train_args.json")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"[INFO] Train args saved to: {args_path}")


def save_history(history: dict, save_dir: str):
    path = os.path.join(save_dir, "train_history.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Train history saved to: {path}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for inp, tgt in pbar:
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(inp)
        loss = criterion(pred, tgt)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inp.size(0)
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Val", leave=False)
    for inp, tgt in pbar:
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        pred = model(inp)
        loss = criterion(pred, tgt)

        running_loss += loss.item() * inp.size(0)
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] CUDA is not available, fallback to CPU.")

    print("=" * 90)
    print("[INFO] Training configuration:")
    print(f"  data_root        : {args.data_root}")
    print(f"  save_dir         : {args.save_dir}")
    print(f"  train_thickness  : {args.train_thickness}")
    print(f"  val_thickness    : {args.val_thickness}")
    print(f"  cond_thickness   : {args.cond_thickness}")
    print(f"  batch_size       : {args.batch_size}")
    print(f"  num_workers      : {args.num_workers}")
    print(f"  epochs           : {args.epochs}")
    print(f"  lr               : {args.lr}")
    print(f"  weight_decay     : {args.weight_decay}")
    print(f"  img_size         : {args.img_size}")
    print(f"  seed             : {args.seed}")
    print("=" * 90)

    save_args(args, args.save_dir)

    train_loader = get_loader(
        root=args.data_root,
        split="train",
        thickness=args.train_thickness,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        crop_size=args.img_size,
        random_crop=True,
        max_samples=args.train_max_samples,
        cond_thickness=args.cond_thickness,
    )

    val_loader = get_loader(
        root=args.data_root,
        split="val",
        thickness=args.val_thickness,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        crop_size=args.img_size,
        random_crop=False,
        max_samples=args.val_max_samples,
        cond_thickness=args.cond_thickness,
    )

    model = build_model(
        img_size=args.img_size,
        cond_thickness=args.cond_thickness
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    best_val = float("inf")
    early_stop_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
        "best_val": None,
    }

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={current_lr:.8f}"
        )

        # 始终保存最后一轮
        last_ckpt = os.path.join(args.save_dir, "last.pth")
        torch.save(model.state_dict(), last_ckpt)

        # 判断是否有“有效提升”
        if val_loss < (best_val - args.min_delta):
            best_val = val_loss
            early_stop_counter = 0

            best_ckpt = os.path.join(args.save_dir, "best.pth")
            torch.save(model.state_dict(), best_ckpt)
            print(f"[INFO] Best model saved to: {best_ckpt}")
        else:
            early_stop_counter += 1
            print(
                f"[INFO] No significant improvement. "
                f"EarlyStop counter: {early_stop_counter}/{args.early_stop_patience}"
            )

        history["best_val"] = best_val
        save_history(history, args.save_dir)

        if early_stop_counter >= args.early_stop_patience:
            print(f"[INFO] Early stopping triggered at epoch {epoch}.")
            break

    print(f"[INFO] Training finished. Best val loss = {best_val:.6f}")


if __name__ == "__main__":
    main()
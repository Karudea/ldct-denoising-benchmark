import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ldct_model import UNet
from ldct_npy_dataset import LDCTPatchNPYDataset


# ================== 统一路径配置 ==================
def get_runtime_paths() -> Dict[str, Path]:
    """
    通过环境变量 IS_CLOUD 控制本地 / 云端路径

    本地 Windows:
        不设置 IS_CLOUD，默认使用 E:\\LDCT

    云端 Linux:
        export IS_CLOUD=1
        默认路径:
            /workspace/data
            /workspace/experiments
            /workspace/eval_results
    """
    is_cloud = os.getenv("IS_CLOUD", "0") == "1"

    if is_cloud:
        data_root = Path("/workspace/data")
        exp_root = Path("/workspace/experiments")
        eval_root = Path("/workspace/eval_results")
    else:
        data_root = Path(r"E:\LDCT")
        exp_root = data_root / "experiments"
        eval_root = data_root / "eval_results"

    return {
        "data_root": data_root,
        "prepared_root": data_root / "prepared_1mm3mm_hu_-160_240",
        "split_dir": data_root / "splits",
        "exp_root": exp_root,
        "eval_root": eval_root,
    }


PATHS = get_runtime_paths()
PREPARED_ROOT = PATHS["prepared_root"]
SPLIT_DIR = PATHS["split_dir"]
EXP_ROOT = PATHS["exp_root"]
EVAL_ROOT = PATHS["eval_root"]


# ================== 配置区（你只改这里） ==================
EXP_NAME = "A3_C4_residual_sigmoid_trainall_valall"
SEED = 0

# 评估哪个 split
EVAL_SPLIT = "test"   # "train" | "val" | "test"

# 分别统计 1mm 和 3mm
EVAL_THICKNESSES = ["1mm", "3mm"]

# 与训练保持一致（仅用于记录）
TRAIN_THICKNESS = "all"
VAL_THICKNESS = "all"
COND_THICKNESS = True
MODEL_IN_CHANNELS = 2 if COND_THICKNESS else 1

# A3: residual + sigmoid
USE_SIGMOID = True
RESIDUAL_LEARNING = True
MODEL_PREDICTION_TYPE = "input_plus_residual"
MODEL_OUTPUT_ACTIVATION = "sigmoid"

# DataLoader
BATCH_SIZE = 8
NUM_WORKERS = 0

# 归一化窗口（与训练保持一致）
CLIP_MIN = -160.0
CLIP_MAX = 240.0

# 指标空间
# "norm": 在 [0,1] 空间算 MAE/RMSE/NRMSE/HFEN
# "hu"  : 先反归一化到 HU 再算 MAE/RMSE/NRMSE/HFEN
# PSNR / SSIM 仍在 [0,1] 空间计算
METRIC_SPACE_FOR_MAE_RMSE_NRMSE_HFEN = "hu"

# 预测后处理
CLIP_PRED_BEFORE_METRIC = True

# HFEN
HFEN_KERNEL_SIZE = 15
HFEN_SIGMA = 1.5

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint 路径
CKPT_PATH = EXP_ROOT / EXP_NAME / f"seed_{SEED}" / f"best_{EXP_NAME}_seed{SEED}.pth"

# 输出目录：注意这里不再包含单个 thickness
SAVE_ROOT = EVAL_ROOT / f"{EXP_NAME}_seed{SEED}" / EVAL_SPLIT


# ================== 基础工具函数 ==================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_to_hu(img: np.ndarray, clip_min=CLIP_MIN, clip_max=CLIP_MAX) -> np.ndarray:
    return (img * (clip_max - clip_min) + clip_min).astype(np.float32)


def create_log_kernel(kernel_size=15, sigma=1.5, device="cpu"):
    ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    norm2 = xx ** 2 + yy ** 2
    sigma2 = sigma ** 2

    gauss = torch.exp(-norm2 / (2 * sigma2))
    gauss = gauss / gauss.sum()

    log_kernel = ((norm2 - 2 * sigma2) / (sigma2 ** 2)) * gauss
    log_kernel = log_kernel - log_kernel.mean()
    return log_kernel.unsqueeze(0).unsqueeze(0)


def compute_hfen(pred: np.ndarray, gt: np.ndarray, kernel_size=15, sigma=1.5) -> float:
    kernel = create_log_kernel(kernel_size=kernel_size, sigma=sigma, device="cpu")

    pred_t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    pred_f = F.conv2d(pred_t, kernel, padding=kernel_size // 2)
    gt_f = F.conv2d(gt_t, kernel, padding=kernel_size // 2)

    num = torch.norm(pred_f - gt_f, p=2).item()
    den = torch.norm(gt_f, p=2).item() + 1e-12
    return float(num / den)


# ================== SSIM ==================
class SSIMMetric(nn.Module):
    """
    返回 SSIM 值（不是 loss）
    输入: [N,1,H,W]，范围默认 [0,1]
    """
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

        sigma = 1.5
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window_1d = g.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float()
        window = window_2d.unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        self.eps = 1e-8

    def forward(self, img1, img2):
        img1 = img1.float()
        img2 = img2.float()
        window = self.window.to(dtype=img1.dtype, device=img1.device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2) - mu1_mu2

        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        denominator = torch.clamp(denominator, min=self.eps)

        ssim_map = numerator / denominator
        ssim_val = ssim_map.mean()
        ssim_val = torch.clamp(ssim_val, 0.0, 1.0)
        return ssim_val


# ================== 指标 ==================
def compute_psnr_from_torch(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    """
    pred/target: [N,1,H,W] or [N,H,W]
    返回 batch 内每个样本的 PSNR
    """
    if pred.ndim == 4:
        pred = pred.squeeze(1)
    if target.ndim == 4:
        target = target.squeeze(1)

    pred = pred.float()
    target = target.float()

    batch_vals = []
    for i in range(pred.shape[0]):
        mse = torch.mean((pred[i] - target[i]) ** 2).item()
        if mse <= 1e-12:
            batch_vals.append(99.0)
        else:
            batch_vals.append(float(10.0 * np.log10(1.0 / mse)))
    return batch_vals


@torch.no_grad()
def compute_ssim_batch(pred: torch.Tensor, target: torch.Tensor, ssim_metric: SSIMMetric) -> List[float]:
    vals = []
    for i in range(pred.shape[0]):
        val = ssim_metric(pred[i:i + 1], target[i:i + 1]).item()
        vals.append(float(val))
    return vals


def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt), dtype=np.float64))


def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2, dtype=np.float64)))


def compute_nrmse(pred: np.ndarray, gt: np.ndarray, mode: str = "range") -> float:
    rmse = compute_rmse(pred, gt)
    if mode == "range":
        denom = float(gt.max() - gt.min())
    elif mode == "mean":
        denom = float(np.mean(np.abs(gt)))
    else:
        raise ValueError("mode must be 'range' or 'mean'")
    if abs(denom) < 1e-12:
        return 0.0
    return float(rmse / denom)


def summarize_metrics(records: List[dict], keys: List[str]):
    summary = {}
    for k in keys:
        vals = [r[k] for r in records]
        summary[k] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)) if vals else None,
            "max": float(np.max(vals)) if vals else None,
        }
    return summary


# ================== 模型 & 数据 ==================
def build_model_and_load_ckpt(ckpt_path: Path, device: torch.device):
    model = UNet(
        in_channels=MODEL_IN_CHANNELS,
        out_channels=1,
        base_ch=64,
        use_sigmoid=USE_SIGMOID,
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()

    print(f"[INFO] Checkpoint loaded from: {ckpt_path}")
    return model


def build_eval_loader(eval_thickness: str) -> Tuple[LDCTPatchNPYDataset, DataLoader]:
    dataset = LDCTPatchNPYDataset(
        root=PREPARED_ROOT,
        split=EVAL_SPLIT,
        thickness=eval_thickness,
        cond_thickness=COND_THICKNESS,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"[ERROR] eval dataset 长度为 0，请检查: "
            f"split={EVAL_SPLIT}, thickness={eval_thickness}, root={PREPARED_ROOT}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return dataset, loader


# ================== 主评估逻辑 ==================
@torch.no_grad()
def evaluate(model, loader, ssim_metric, device, eval_thickness: str):
    all_records = []

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {EVAL_SPLIT}-{eval_thickness}", ncols=120)):
        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise RuntimeError(
                "[ERROR] 当前脚本期望 DataLoader 返回 (qd, fd)。"
                "如果你后续改了 Dataset 返回格式，需要同步修改 eval 脚本。"
            )

        qd, fd = batch
        qd = qd.to(device, non_blocking=True)
        fd = fd.to(device, non_blocking=True)

        out = model(qd)
        out = out.float()
        fd = fd.float()

        if CLIP_PRED_BEFORE_METRIC:
            out = torch.clamp(out, 0.0, 1.0)

        psnr_list = compute_psnr_from_torch(out, fd)
        ssim_list = compute_ssim_batch(out, fd, ssim_metric)

        out_np = out.squeeze(1).cpu().numpy().astype(np.float32)
        fd_np = fd.squeeze(1).cpu().numpy().astype(np.float32)

        for i in range(out_np.shape[0]):
            pred_norm = out_np[i]
            gt_norm = fd_np[i]

            if METRIC_SPACE_FOR_MAE_RMSE_NRMSE_HFEN == "hu":
                pred_eval = norm_to_hu(pred_norm)
                gt_eval = norm_to_hu(gt_norm)
            elif METRIC_SPACE_FOR_MAE_RMSE_NRMSE_HFEN == "norm":
                pred_eval = pred_norm
                gt_eval = gt_norm
            else:
                raise ValueError("METRIC_SPACE_FOR_MAE_RMSE_NRMSE_HFEN must be 'hu' or 'norm'")

            mae = compute_mae(pred_eval, gt_eval)
            rmse = compute_rmse(pred_eval, gt_eval)
            nrmse = compute_nrmse(pred_eval, gt_eval, mode="range")
            hfen = compute_hfen(
                pred_eval,
                gt_eval,
                kernel_size=HFEN_KERNEL_SIZE,
                sigma=HFEN_SIGMA,
            )

            rec = {
                "global_patch_idx": batch_idx * BATCH_SIZE + i,
                "thickness": eval_thickness,
                "pred_min_after_postprocess": float(pred_norm.min()),
                "pred_max_after_postprocess": float(pred_norm.max()),
                "gt_min": float(gt_norm.min()),
                "gt_max": float(gt_norm.max()),
                "PSNR": float(psnr_list[i]),
                "SSIM": float(ssim_list[i]),
                "MAE": float(mae),
                "RMSE": float(rmse),
                "NRMSE": float(nrmse),
                "HFEN": float(hfen),
            }
            all_records.append(rec)

    return all_records


def evaluate_one_thickness(model, ssim_metric, device, eval_thickness: str):
    _, loader = build_eval_loader(eval_thickness)
    records = evaluate(model, loader, ssim_metric, device, eval_thickness)

    metric_keys = ["PSNR", "SSIM", "MAE", "RMSE", "NRMSE", "HFEN"]
    summary = summarize_metrics(records, metric_keys)

    save_dir = SAVE_ROOT / eval_thickness
    ensure_dir(save_dir)

    records_path = save_dir / f"patch_level_metrics_{EVAL_SPLIT}_{eval_thickness}.json"
    summary_path = save_dir / f"summary_{EVAL_SPLIT}_{eval_thickness}.json"

    summary_json = {
        "exp_name": EXP_NAME,
        "seed": SEED,
        "checkpoint_path": str(CKPT_PATH),
        "eval_split": EVAL_SPLIT,
        "eval_thickness": eval_thickness,
        "train_thickness": TRAIN_THICKNESS,
        "val_thickness": VAL_THICKNESS,
        "cond_thickness": COND_THICKNESS,
        "model_in_channels": MODEL_IN_CHANNELS,
        "use_sigmoid": USE_SIGMOID,
        "residual_learning": RESIDUAL_LEARNING,
        "model_output_activation": MODEL_OUTPUT_ACTIVATION,
        "model_prediction_type": MODEL_PREDICTION_TYPE,
        "metric_space_for_MAE_RMSE_NRMSE_HFEN": METRIC_SPACE_FOR_MAE_RMSE_NRMSE_HFEN,
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "clip_pred_before_metric": CLIP_PRED_BEFORE_METRIC,
        "hfen_kernel_size": HFEN_KERNEL_SIZE,
        "hfen_sigma": HFEN_SIGMA,
        "num_patches": len(records),
        "patch_level_summary": summary,
    }

    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    return records, summary


def main():
    set_seed(SEED)

    if not PREPARED_ROOT.exists():
        raise FileNotFoundError(f"[ERROR] PREPARED_ROOT 不存在: {PREPARED_ROOT}")

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"[ERROR] CKPT_PATH 不存在: {CKPT_PATH}")

    ensure_dir(SAVE_ROOT)

    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] PREPARED_ROOT = {PREPARED_ROOT}")
    print(f"[INFO] SPLIT_DIR = {SPLIT_DIR}")
    print(f"[INFO] CKPT_PATH = {CKPT_PATH}")
    print(f"[INFO] SAVE_ROOT = {SAVE_ROOT}")
    print(f"[INFO] EVAL_SPLIT = {EVAL_SPLIT}")
    print(f"[INFO] EVAL_THICKNESSES = {EVAL_THICKNESSES}")
    print(f"[INFO] TRAIN_THICKNESS = {TRAIN_THICKNESS}")
    print(f"[INFO] VAL_THICKNESS = {VAL_THICKNESS}")
    print(f"[INFO] COND_THICKNESS = {COND_THICKNESS}")
    print(f"[INFO] MODEL_IN_CHANNELS = {MODEL_IN_CHANNELS}")
    print(f"[INFO] USE_SIGMOID = {USE_SIGMOID}")
    print(f"[INFO] RESIDUAL_LEARNING = {RESIDUAL_LEARNING}")
    print(f"[INFO] MODEL_OUTPUT_ACTIVATION = {MODEL_OUTPUT_ACTIVATION}")
    print(f"[INFO] MODEL_PREDICTION_TYPE = {MODEL_PREDICTION_TYPE}")
    print(f"[INFO] METRIC_SPACE_FOR_MAE_RMSE_NRMSE_HFEN = {METRIC_SPACE_FOR_MAE_RMSE_NRMSE_HFEN}")

    model = build_model_and_load_ckpt(CKPT_PATH, DEVICE)
    ssim_metric = SSIMMetric(window_size=11).to(DEVICE)

    all_records_merged = []
    per_thickness_summary = {}

    for eval_thickness in EVAL_THICKNESSES:
        print("\n" + "-" * 90)
        print(f"[INFO] Evaluating thickness = {eval_thickness}")
        print("-" * 90)

        records, summary = evaluate_one_thickness(
            model=model,
            ssim_metric=ssim_metric,
            device=DEVICE,
            eval_thickness=eval_thickness,
        )

        all_records_merged.extend(records)
        per_thickness_summary[eval_thickness] = {
            "num_patches": len(records),
            "patch_level_summary": summary,
        }

    metric_keys = ["PSNR", "SSIM", "MAE", "RMSE", "NRMSE", "HFEN"]
    merged_summary = summarize_metrics(all_records_merged, metric_keys)

    merged_summary_json = {
        "exp_name": EXP_NAME,
        "seed": SEED,
        "checkpoint_path": str(CKPT_PATH),
        "eval_split": EVAL_SPLIT,
        "eval_thicknesses": EVAL_THICKNESSES,
        "train_thickness": TRAIN_THICKNESS,
        "val_thickness": VAL_THICKNESS,
        "cond_thickness": COND_THICKNESS,
        "model_in_channels": MODEL_IN_CHANNELS,
        "use_sigmoid": USE_SIGMOID,
        "residual_learning": RESIDUAL_LEARNING,
        "model_output_activation": MODEL_OUTPUT_ACTIVATION,
        "model_prediction_type": MODEL_PREDICTION_TYPE,
        "metric_space_for_MAE_RMSE_NRMSE_HFEN": METRIC_SPACE_FOR_MAE_RMSE_NRMSE_HFEN,
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "clip_pred_before_metric": CLIP_PRED_BEFORE_METRIC,
        "hfen_kernel_size": HFEN_KERNEL_SIZE,
        "hfen_sigma": HFEN_SIGMA,
        "num_patches_total": len(all_records_merged),
        "per_thickness_summary": per_thickness_summary,
        "merged_patch_level_summary": merged_summary,
    }

    merged_records_path = SAVE_ROOT / f"patch_level_metrics_{EVAL_SPLIT}_merged.json"
    merged_summary_path = SAVE_ROOT / f"summary_{EVAL_SPLIT}_merged.json"

    with open(merged_records_path, "w", encoding="utf-8") as f:
        json.dump(all_records_merged, f, indent=2, ensure_ascii=False)

    with open(merged_summary_path, "w", encoding="utf-8") as f:
        json.dump(merged_summary_json, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("[INFO] Evaluation finished.")
    print(f"[INFO] merged patch metrics saved to: {merged_records_path}")
    print(f"[INFO] merged summary saved to     : {merged_summary_path}")
    print("=" * 90)

    for th in EVAL_THICKNESSES:
        print(f"\n[RESULT] Thickness = {th}")
        th_summary = per_thickness_summary[th]["patch_level_summary"]
        for k, v in th_summary.items():
            print(
                f"{k:>6s} | "
                f"mean={v['mean']:.6f} | "
                f"std={v['std']:.6f} | "
                f"min={v['min']:.6f} | "
                f"max={v['max']:.6f}"
            )

    print(f"\n[RESULT] Merged ({'+'.join(EVAL_THICKNESSES)})")
    for k, v in merged_summary.items():
        print(
            f"{k:>6s} | "
            f"mean={v['mean']:.6f} | "
            f"std={v['std']:.6f} | "
            f"min={v['min']:.6f} | "
            f"max={v['max']:.6f}"
        )


if __name__ == "__main__":
    main()
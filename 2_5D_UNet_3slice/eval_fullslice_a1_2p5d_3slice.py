import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm

from ldct_model import UNet


# =========================
# 0. 配置区（你只改这里）
# =========================

# ---- 实验命名（和 train 保持一致）----
EXP_NAME = "baseline_1mm_only_2p5d_3slice_sigmoid_l1"
SEED = 0

# ---- 数据路径 ----
TRAIN_DATA_ROOT = r"E:\LDCT\Training_Image_Data\1mm B30"
TEST_QD_ROOT = r"E:\LDCT\Testing_Image_Data\1mm B30\QD_1mm"
SPLIT_JSON = r"E:\LDCT\splits\patient_splits.json"

# ---- 模型路径 ----
CKPT_PATH = rf"E:\LDCT\experiments\{EXP_NAME}\best_unet.pth"

# ---- 输出目录 ----
SAVE_ROOT = rf"E:\LDCT\eval_results\{EXP_NAME}_seed{SEED}"

# ---- 评估设置 ----
EVAL_SPLIT = "test"          # "train" | "val" | "test" | "external_test"
EVAL_THICKNESS = "1mm"       # 你的训练设定是 1mm-only，这里建议优先评估 1mm
PATCH_SIZE = 256

# ---- 与训练对应的 thickness 信息（仅用于记录）----
TRAIN_THICKNESS = "1mm"
VAL_THICKNESS = "1mm"

# ---- 输入模式记录 ----
INPUT_MODE = "2.5D_3slice"
MODEL_IN_CHANNELS = 3

# ---- 归一化窗口（必须和训练保持一致）----
CLIP_MIN = -160.0
CLIP_MAX = 240.0

# ---- 指标空间 ----
# "norm"：在 [0,1] 归一化空间算
# "hu"  ：先反归一化回 HU 再算 RMSE/MAE/NRMSE/HFEN；PSNR/SSIM 仍默认在 norm 空间
METRIC_SPACE = "hu"

# ---- 预测后处理 ----
# 当前模型有 sigmoid，理论上输出应在 [0,1]
# 这里保留 clip，作为保险
CLIP_PRED_BEFORE_METRIC = True
SAVE_RAW_PRED = True

# ---- 可视化 ----
SAVE_VIS = True
VIS_PER_PATIENT_MAX = 3
DIFF_VMAX_HU = 80.0
ROI_SIZE = 128
AUTO_ROI_BY_ERROR = True

# ---- 设备 ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


FD_ROOTS = {
    "1mm": os.path.join(TRAIN_DATA_ROOT, "full_1mm"),
    "3mm": os.path.join(TRAIN_DATA_ROOT, "full_3mm"),
}
QD_ROOTS = {
    "1mm": os.path.join(TRAIN_DATA_ROOT, "quarter_1mm"),
    "3mm": os.path.join(TRAIN_DATA_ROOT, "quarter_3mm"),
}

Z_TOL_MM = 1e-2


# =========================
# 1. 基础工具函数
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_splits(split_json_path: str):
    with open(split_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    for k in ["train", "val", "test", "external_test"]:
        splits.setdefault(k, [])
    return splits


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def get_z_position(ds) -> Optional[float]:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        z = safe_float(ipp[2])
        if z is not None:
            return z
    sl = getattr(ds, "SliceLocation", None)
    return safe_float(sl)


def get_instance_number(ds) -> int:
    try:
        return int(getattr(ds, "InstanceNumber", 0))
    except Exception:
        return 0


def list_dicom_files(folder: Path):
    if not folder.exists():
        return []
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in [".dcm", ".ima", ""]
    ]


def load_dicom_series(folder: Path) -> List:
    files = list_dicom_files(folder)
    if not files:
        print(f"⚠️ 空目录: {folder}")
        return []

    series = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            series.append(ds)
        except Exception as e:
            print(f"⚠️ 读取失败: {f}, err={e}")

    if not series:
        return []

    zs = [get_z_position(d) for d in series]
    if all(z is not None for z in zs):
        series.sort(key=lambda d: get_z_position(d))
    else:
        series.sort(key=get_instance_number)

    return series


def dicom_to_hu(ds):
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return img * slope + intercept


def hu_to_norm(img, clip_min=CLIP_MIN, clip_max=CLIP_MAX):
    img = np.clip(img, clip_min, clip_max)
    return ((img - clip_min) / (clip_max - clip_min)).astype(np.float32)


def norm_to_hu(img, clip_min=CLIP_MIN, clip_max=CLIP_MAX):
    return (img * (clip_max - clip_min) + clip_min).astype(np.float32)


def center_crop_to_multiple(img: np.ndarray, multiple: int):
    h, w = img.shape
    nh = (h // multiple) * multiple
    nw = (w // multiple) * multiple
    sh = (h - nh) // 2
    sw = (w - nw) // 2
    cropped = img[sh:sh + nh, sw:sw + nw]
    return cropped, (sh, sw, nh, nw)


def quantize_z(z: float, tol: float) -> float:
    return round(z / tol) * tol


def match_fd_qd_pairs(fd_series: List, qd_series: List) -> List[Tuple]:
    if not fd_series or not qd_series:
        return []

    fd_z = [get_z_position(d) for d in fd_series]
    qd_z = [get_z_position(d) for d in qd_series]

    if all(z is not None for z in fd_z) and all(z is not None for z in qd_z):
        fd_map: Dict[float, object] = {}
        for d in fd_series:
            z = quantize_z(get_z_position(d), Z_TOL_MM)
            fd_map.setdefault(z, d)

        pairs = []
        miss = 0
        for q in qd_series:
            z = quantize_z(get_z_position(q), Z_TOL_MM)
            f = fd_map.get(z, None)
            if f is None:
                miss += 1
                continue
            pairs.append((f, q))

        if pairs:
            if miss > 0:
                print(f"⚠️ z 匹配有缺失：QD 有 {miss} 张在 FD 中找不到对应 z（已跳过）")
            return pairs

        print("⚠️ z 匹配失败：退化为 InstanceNumber 对齐")

    print("⚠️ DICOM 缺少完整 z 信息，退化为 InstanceNumber 对齐（可能存在错配风险）")
    fd_sorted = sorted(fd_series, key=get_instance_number)
    qd_sorted = sorted(qd_series, key=get_instance_number)
    min_n = min(len(fd_sorted), len(qd_sorted))
    return [(fd_sorted[i], qd_sorted[i]) for i in range(min_n)]


# =========================
# 2. 2.5D patch 切分 / 回拼
# =========================

def image3d_to_patches(img3: np.ndarray, patch_size: int):
    """
    img3: (C, H, W), 例如 2.5D 三切片输入 (3, H, W)

    return:
      patches: (N, C, patch_size, patch_size)
      coords : [(y, x), ...]
      hw     : (H, W)
    """
    assert img3.ndim == 3, f"img3 shape must be (C,H,W), got {img3.shape}"
    c, h, w = img3.shape

    patches = []
    coords = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            p = img3[:, y:y + patch_size, x:x + patch_size]
            if p.shape == (c, patch_size, patch_size):
                patches.append(p)
                coords.append((y, x))

    patches = np.stack(patches, axis=0).astype(np.float32)  # (N,C,H,W)
    return patches, coords, (h, w)


def stitch_patches(patches: np.ndarray, coords: List[Tuple[int, int]], out_hw: Tuple[int, int], patch_size: int):
    """
    直接无重叠拼接
    patches: (N, patch_size, patch_size)
    """
    h, w = out_hw
    out = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for p, (y, x) in zip(patches, coords):
        out[y:y + patch_size, x:x + patch_size] += p
        count[y:y + patch_size, x:x + patch_size] += 1.0

    count[count == 0] = 1.0
    out = out / count
    return out


def build_2p5d_stack_from_list(slice_list: List[np.ndarray], center_idx: int) -> np.ndarray:
    """
    slice_list: [ (H,W), (H,W), ... ]，元素已裁剪、已归一化
    返回: (3,H,W)
    边界采用 replicate
    """
    n = len(slice_list)
    assert n > 0, "slice_list is empty"

    left_idx = max(center_idx - 1, 0)
    right_idx = min(center_idx + 1, n - 1)

    left = slice_list[left_idx]
    center = slice_list[center_idx]
    right = slice_list[right_idx]

    stack = np.stack([left, center, right], axis=0).astype(np.float32)
    return stack


@torch.no_grad()
def predict_full_slice_raw_2p5d(model: nn.Module, qd_stack_norm: np.ndarray, patch_size: int, device: str, batch_size: int = 8):
    """
    qd_stack_norm: (3, H, W)，已裁剪并归一化到 [0,1]
    return: pred_norm_raw (H, W)，不做 clip
    """
    patches, coords, out_hw = image3d_to_patches(qd_stack_norm, patch_size)
    preds = []

    for i in range(0, len(patches), batch_size):
        batch = patches[i:i + batch_size]                      # (N,3,H,W)
        batch_t = torch.from_numpy(batch).to(device)           # (N,3,H,W)
        out = model(batch_t)                                   # (N,1,H,W)
        out = out.squeeze(1).cpu().numpy().astype(np.float32)  # (N,H,W)
        preds.append(out)

    preds = np.concatenate(preds, axis=0)
    pred_full_raw = stitch_patches(preds, coords, out_hw, patch_size)
    return pred_full_raw


def postprocess_prediction_for_metric(pred_norm_raw: np.ndarray) -> np.ndarray:
    if CLIP_PRED_BEFORE_METRIC:
        return np.clip(pred_norm_raw, 0.0, 1.0).astype(np.float32)
    return pred_norm_raw.astype(np.float32)


# =========================
# 3. 指标
# =========================

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


def compute_psnr_norm(pred_norm: np.ndarray, gt_norm: np.ndarray) -> float:
    mse = np.mean((pred_norm - gt_norm) ** 2, dtype=np.float64)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim_norm(pred_norm: np.ndarray, gt_norm: np.ndarray, ssim_metric: SSIMMetric, device: str) -> float:
    a = torch.from_numpy(pred_norm).unsqueeze(0).unsqueeze(0).to(device)
    b = torch.from_numpy(gt_norm).unsqueeze(0).unsqueeze(0).to(device)
    val = ssim_metric(a, b).item()
    return float(val)


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


def compute_hfen(pred: np.ndarray, gt: np.ndarray, device: str = "cpu", kernel_size: int = 15, sigma: float = 1.5) -> float:
    kernel = create_log_kernel(kernel_size=kernel_size, sigma=sigma, device=device)

    pred_t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    pred_f = F.conv2d(pred_t, kernel, padding=kernel_size // 2)
    gt_f = F.conv2d(gt_t, kernel, padding=kernel_size // 2)

    num = torch.norm(pred_f - gt_f, p=2).item()
    den = torch.norm(gt_f, p=2).item() + 1e-12
    return float(num / den)


def evaluate_pair(pred_norm_eval: np.ndarray, gt_norm: np.ndarray, ssim_metric: SSIMMetric, metric_space: str, device: str):
    psnr = compute_psnr_norm(pred_norm_eval, gt_norm)
    ssim = compute_ssim_norm(pred_norm_eval, gt_norm, ssim_metric, device)

    if metric_space == "hu":
        pred_eval = norm_to_hu(pred_norm_eval)
        gt_eval = norm_to_hu(gt_norm)
    elif metric_space == "norm":
        pred_eval = pred_norm_eval
        gt_eval = gt_norm
    else:
        raise ValueError("metric_space must be 'hu' or 'norm'")

    mae = compute_mae(pred_eval, gt_eval)
    rmse = compute_rmse(pred_eval, gt_eval)
    nrmse = compute_nrmse(pred_eval, gt_eval, mode="range")
    hfen = compute_hfen(pred_eval, gt_eval, device="cpu")

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "HFEN": hfen,
    }


# =========================
# 4. 结构核查可视化
# =========================

def find_roi_by_error(gt_hu: np.ndarray, pred_hu: np.ndarray, roi_size: int = 128):
    h, w = gt_hu.shape
    diff = np.abs(pred_hu - gt_hu)

    if h < roi_size or w < roi_size:
        return 0, 0, min(h, roi_size), min(w, roi_size)

    step = max(roi_size // 4, 16)
    best_score = -1.0
    best_xy = (0, 0)

    for y in range(0, h - roi_size + 1, step):
        for x in range(0, w - roi_size + 1, step):
            score = diff[y:y + roi_size, x:x + roi_size].mean()
            if score > best_score:
                best_score = score
                best_xy = (y, x)

    return best_xy[0], best_xy[1], roi_size, roi_size


def get_center_roi(h: int, w: int, roi_size: int = 128):
    rh = min(h, roi_size)
    rw = min(w, roi_size)
    y = max((h - rh) // 2, 0)
    x = max((w - rw) // 2, 0)
    return y, x, rh, rw


def save_visualization(
    save_path: Path,
    qd_hu_center: np.ndarray,
    gt_hu: np.ndarray,
    pred_hu: np.ndarray,
    patient_id: str,
    slice_idx: int,
    metrics: dict,
):
    diff_hu = pred_hu - gt_hu
    h, w = gt_hu.shape

    if AUTO_ROI_BY_ERROR:
        ry, rx, rh, rw = find_roi_by_error(gt_hu, pred_hu, ROI_SIZE)
    else:
        ry, rx, rh, rw = get_center_roi(h, w, ROI_SIZE)

    gt_roi = gt_hu[ry:ry + rh, rx:rx + rw]
    pred_roi = pred_hu[ry:ry + rh, rx:rx + rw]
    diff_roi = diff_hu[ry:ry + rh, rx:rx + rw]

    vmin = CLIP_MIN
    vmax = CLIP_MAX

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    axes[0, 0].imshow(qd_hu_center, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("QD input (center slice)")
    axes[0, 0].add_patch(plt.Rectangle((rx, ry), rw, rh, fill=False, edgecolor="yellow", linewidth=1.5))
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_hu, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("FD / GT")
    axes[0, 1].add_patch(plt.Rectangle((rx, ry), rw, rh, fill=False, edgecolor="yellow", linewidth=1.5))
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pred_hu, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("Prediction")
    axes[0, 2].add_patch(plt.Rectangle((rx, ry), rw, rh, fill=False, edgecolor="yellow", linewidth=1.5))
    axes[0, 2].axis("off")

    im = axes[0, 3].imshow(diff_hu, cmap="bwr", vmin=-DIFF_VMAX_HU, vmax=DIFF_VMAX_HU)
    axes[0, 3].set_title("Diff (Pred - GT)")
    axes[0, 3].axis("off")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(gt_roi, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("GT ROI")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_roi, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Pred ROI")
    axes[1, 1].axis("off")

    im2 = axes[1, 2].imshow(diff_roi, cmap="bwr", vmin=-DIFF_VMAX_HU, vmax=DIFF_VMAX_HU)
    axes[1, 2].set_title("Diff ROI")
    axes[1, 2].axis("off")
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    axes[1, 3].axis("off")
    text = (
        f"Patient: {patient_id}\n"
        f"Slice: {slice_idx}\n"
        f"PSNR : {metrics['PSNR']:.4f}\n"
        f"SSIM : {metrics['SSIM']:.6f}\n"
        f"MAE  : {metrics['MAE']:.4f}\n"
        f"RMSE : {metrics['RMSE']:.4f}\n"
        f"NRMSE: {metrics['NRMSE']:.6f}\n"
        f"HFEN : {metrics['HFEN']:.6f}\n"
    )
    axes[1, 3].text(0.02, 0.98, text, va="top", ha="left", fontsize=12)

    fig.suptitle(f"{patient_id} | slice {slice_idx}", fontsize=14)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================
# 5. external_test 纯推理导出
# =========================

@torch.no_grad()
def run_external_test(model: nn.Module, patient_ids: List[str], device: str):
    out_dir = Path(SAVE_ROOT) / "external_test" / "pred_npy"
    ensure_dir(out_dir)

    for patient_id in tqdm(patient_ids, desc="External inference", ncols=120):
        qd_dir = Path(TEST_QD_ROOT) / patient_id / "quarter_1mm"
        qd_series = load_dicom_series(qd_dir)
        if not qd_series:
            print(f"❌ external {patient_id}: 空")
            continue

        patient_out = out_dir / patient_id
        ensure_dir(patient_out)

        qd_crop_hu_list = []
        qd_norm_list = []

        for ds in qd_series:
            qd_hu = dicom_to_hu(ds)
            qd_crop_hu, _ = center_crop_to_multiple(qd_hu, PATCH_SIZE)
            qd_norm = hu_to_norm(qd_crop_hu)
            qd_crop_hu_list.append(qd_crop_hu)
            qd_norm_list.append(qd_norm)

        for i in range(len(qd_norm_list)):
            qd_stack_norm = build_2p5d_stack_from_list(qd_norm_list, i)

            pred_norm_raw = predict_full_slice_raw_2p5d(model, qd_stack_norm, PATCH_SIZE, device, batch_size=8)
            pred_norm_eval = postprocess_prediction_for_metric(pred_norm_raw)
            pred_hu_eval = norm_to_hu(pred_norm_eval)

            if SAVE_RAW_PRED:
                np.save(patient_out / f"{patient_id}_s{i:03d}_pred_norm_raw.npy", pred_norm_raw.astype(np.float32))

            np.save(patient_out / f"{patient_id}_s{i:03d}_pred_norm_eval.npy", pred_norm_eval.astype(np.float32))
            np.save(patient_out / f"{patient_id}_s{i:03d}_pred_hu_eval.npy", pred_hu_eval.astype(np.float32))

    print(f"[INFO] external_test 推理完成，结果保存在: {out_dir}")


# =========================
# 6. 主评估流程
# =========================

def build_model_and_load_ckpt(ckpt_path: str, device: str):
    model = UNet(in_channels=3, out_channels=1, base_ch=64).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"[INFO] Checkpoint loaded from: {ckpt_path}")
    return model


def summarize_metrics(records: List[dict]):
    keys = ["PSNR", "SSIM", "MAE", "RMSE", "NRMSE", "HFEN"]
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


def main():
    ensure_dir(Path(SAVE_ROOT))

    model = build_model_and_load_ckpt(CKPT_PATH, DEVICE)
    ssim_metric = SSIMMetric(window_size=11).to(DEVICE)

    splits = load_splits(SPLIT_JSON)
    patient_ids = splits.get(EVAL_SPLIT, [])

    if EVAL_SPLIT == "external_test":
        run_external_test(model, patient_ids, DEVICE)
        return

    assert EVAL_SPLIT in ["train", "val", "test"]
    assert EVAL_THICKNESS in ["1mm", "3mm"]

    result_dir = Path(SAVE_ROOT) / EVAL_SPLIT / EVAL_THICKNESS
    vis_dir = result_dir / "vis"
    ensure_dir(result_dir)
    ensure_dir(vis_dir)

    all_slice_records = []
    patient_level_records = []

    for patient_id in tqdm(patient_ids, desc=f"Evaluating {EVAL_SPLIT}-{EVAL_THICKNESS}", ncols=120):
        fd_dir = Path(FD_ROOTS[EVAL_THICKNESS]) / patient_id / f"full_{EVAL_THICKNESS}"
        qd_dir = Path(QD_ROOTS[EVAL_THICKNESS]) / patient_id / f"quarter_{EVAL_THICKNESS}"

        fd_series = load_dicom_series(fd_dir)
        qd_series = load_dicom_series(qd_dir)

        if not fd_series or not qd_series:
            print(f"❌ {patient_id} {EVAL_THICKNESS}: FD/QD 缺失，跳过")
            continue

        pairs = match_fd_qd_pairs(fd_series, qd_series)
        if not pairs:
            print(f"❌ {patient_id} {EVAL_THICKNESS}: 无法配对切片，跳过")
            continue

        # 先把配对后的序列全部转成 crop + norm，后面按 index 构造 2.5D 输入
        fd_crop_hu_list = []
        fd_norm_list = []
        qd_crop_hu_list = []
        qd_norm_list = []

        for fd_ds, qd_ds in pairs:
            fd_hu = dicom_to_hu(fd_ds)
            qd_hu = dicom_to_hu(qd_ds)

            fd_crop_hu, _ = center_crop_to_multiple(fd_hu, PATCH_SIZE)
            qd_crop_hu, _ = center_crop_to_multiple(qd_hu, PATCH_SIZE)

            fd_norm = hu_to_norm(fd_crop_hu)
            qd_norm = hu_to_norm(qd_crop_hu)

            fd_crop_hu_list.append(fd_crop_hu)
            fd_norm_list.append(fd_norm)
            qd_crop_hu_list.append(qd_crop_hu)
            qd_norm_list.append(qd_norm)

        patient_slice_records = []
        vis_saved = 0

        for i in range(len(qd_norm_list)):
            fd_norm = fd_norm_list[i]
            fd_crop_hu = fd_crop_hu_list[i]
            qd_crop_hu_center = qd_crop_hu_list[i]

            # 构造 2.5D 三切片输入
            qd_stack_norm = build_2p5d_stack_from_list(qd_norm_list, i)  # (3,H,W)

            pred_norm_raw = predict_full_slice_raw_2p5d(model, qd_stack_norm, PATCH_SIZE, DEVICE, batch_size=8)
            pred_norm_eval = postprocess_prediction_for_metric(pred_norm_raw)

            metrics = evaluate_pair(pred_norm_eval, fd_norm, ssim_metric, METRIC_SPACE, DEVICE)

            slice_record = {
                "patient_id": patient_id,
                "slice_idx": i,
                "thickness": EVAL_THICKNESS,
                "input_mode": INPUT_MODE,
                "model_in_channels": MODEL_IN_CHANNELS,
                "pred_min_raw": float(np.min(pred_norm_raw)),
                "pred_max_raw": float(np.max(pred_norm_raw)),
                "pred_min_eval": float(np.min(pred_norm_eval)),
                "pred_max_eval": float(np.max(pred_norm_eval)),
                **metrics
            }
            all_slice_records.append(slice_record)
            patient_slice_records.append(slice_record)

            if SAVE_VIS and vis_saved < VIS_PER_PATIENT_MAX:
                pred_hu_eval = norm_to_hu(pred_norm_eval)
                save_path = vis_dir / f"{patient_id}_s{i:03d}.png"
                save_visualization(
                    save_path=save_path,
                    qd_hu_center=qd_crop_hu_center,
                    gt_hu=fd_crop_hu,
                    pred_hu=pred_hu_eval,
                    patient_id=patient_id,
                    slice_idx=i,
                    metrics=metrics,
                )
                vis_saved += 1

        if patient_slice_records:
            patient_summary = {
                "patient_id": patient_id,
                "num_slices": len(patient_slice_records),
            }
            for key in ["PSNR", "SSIM", "MAE", "RMSE", "NRMSE", "HFEN"]:
                vals = [x[key] for x in patient_slice_records]
                patient_summary[key] = float(np.mean(vals))
            patient_level_records.append(patient_summary)

    slice_json = result_dir / f"slice_metrics_{EXP_NAME}_seed{SEED}.json"
    with open(slice_json, "w", encoding="utf-8") as f:
        json.dump(all_slice_records, f, indent=2, ensure_ascii=False)

    patient_json = result_dir / f"patient_metrics_{EXP_NAME}_seed{SEED}.json"
    with open(patient_json, "w", encoding="utf-8") as f:
        json.dump(patient_level_records, f, indent=2, ensure_ascii=False)

    summary = {
        "exp_name": EXP_NAME,
        "seed": SEED,
        "eval_split": EVAL_SPLIT,
        "eval_thickness": EVAL_THICKNESS,
        "train_thickness": TRAIN_THICKNESS,
        "val_thickness": VAL_THICKNESS,
        "metric_space_for_MAE_RMSE_NRMSE_HFEN": METRIC_SPACE,
        "num_patients": len(patient_level_records),
        "num_slices": len(all_slice_records),
        "slice_level_summary": summarize_metrics(all_slice_records),
        "patient_level_summary": summarize_metrics(patient_level_records),
        "ckpt_path": CKPT_PATH,
        "clip_min": CLIP_MIN,
        "clip_max": CLIP_MAX,
        "patch_size": PATCH_SIZE,
        "loss": "L1",
        "model_output_activation": "sigmoid",
        "clip_pred_before_metric": CLIP_PRED_BEFORE_METRIC,
        "save_raw_pred": SAVE_RAW_PRED,
        "input_mode": INPUT_MODE,
        "model_in_channels": MODEL_IN_CHANNELS,
        "deep_supervision": False,
    }

    summary_json = result_dir / f"summary_{EXP_NAME}_seed{SEED}.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(f"[INFO] Evaluation finished")
    print(f"[INFO] exp_name={EXP_NAME}")
    print(f"[INFO] split={EVAL_SPLIT}, thickness={EVAL_THICKNESS}")
    print(f"[INFO] train_thickness={TRAIN_THICKNESS}, val_thickness={VAL_THICKNESS}")
    print(f"[INFO] input_mode={INPUT_MODE}, model_in_channels={MODEL_IN_CHANNELS}")
    print(f"[INFO] slices={summary['num_slices']}, patients={summary['num_patients']}")
    print(f"[INFO] summary saved to: {summary_json}")

    s = summary["slice_level_summary"]
    print(
        f"[SLICE] PSNR={s['PSNR']['mean']:.4f}±{s['PSNR']['std']:.4f} | "
        f"SSIM={s['SSIM']['mean']:.6f}±{s['SSIM']['std']:.6f} | "
        f"MAE={s['MAE']['mean']:.4f}±{s['MAE']['std']:.4f} | "
        f"RMSE={s['RMSE']['mean']:.4f}±{s['RMSE']['std']:.4f} | "
        f"NRMSE={s['NRMSE']['mean']:.6f}±{s['NRMSE']['std']:.6f} | "
        f"HFEN={s['HFEN']['mean']:.6f}±{s['HFEN']['std']:.6f}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
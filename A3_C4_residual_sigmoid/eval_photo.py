import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pydicom
import piq
import cv2

from matplotlib.patches import Rectangle, Circle

from ldct_model import UNet


# =========================
# 0. 配置区（你只改这里）
# =========================

FD_ROOTS = {
    "1mm": r"E:\LDCT\Training_Image_Data\1mm B30\full_1mm",
    "3mm": r"E:\LDCT\Training_Image_Data\1mm B30\full_3mm",
}
QD_ROOTS = {
    "1mm": r"E:\LDCT\Training_Image_Data\1mm B30\quarter_1mm",
    "3mm": r"E:\LDCT\Training_Image_Data\1mm B30\quarter_3mm",
}

# A3 checkpoint
CKPT_PATH = r"E:\LDCT\experiments\A3_C4_residual_sigmoid_trainall_valall\seed_0\best_A3_C4_residual_sigmoid_trainall_valall_seed0.pth"
SAVE_ROOT = r"E:\LDCT\eval_results\A3_browser_index_target_cases"

PATCH_SIZE = 256
INFER_BATCH_SIZE = 8

# ===== 无拼接缝关键参数 =====
PATCH_STRIDE = 128
GAUSSIAN_SIGMA_RATIO = 0.125
WEIGHT_EPS = 1e-6

CLIP_MIN = -160.0
CLIP_MAX = 240.0

METRIC_SPACE = "hu"   # "hu" | "norm"
CLIP_PRED_TO_01 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THICKNESS_TO_ID = {"1mm": 0, "3mm": 1}

TARGET_CASES = [
    {
        "patient_id": "L506",
        "thickness": "1mm",
        "s_idx": 61,
        "name": "L506_1mm_FD_s0061",
    },
    {
        "patient_id": "L506",
        "thickness": "3mm",
        "s_idx": 34,
        "name": "L506_3mm_FD_s0034",
    },
]

# =========================
# 参考画布与标注配置
# =========================

# ROI 仍沿用旧逻辑（不变）
ROI_REF_W = 495.0
ROI_REF_H = 495.0
ROI_REF = {
    "left": 75.0,
    "top": 187.0,
    "right": 301.0,
    "bottom": 188.0,
}

# 1mm：三个圆，基于 495x495
REF_1MM_W = 495.0
REF_1MM_H = 495.0
POINTS_REF_XY_1MM = [
    (74.0, 371.0),
    (58.0, 438.0),
    (97.0, 434.0),
]
CIRCLE_RADIUS_1MM = 12.0

# 3mm：一个正方形，基于 496x496
REF_3MM_W = 496.0
REF_3MM_H = 496.0
SQUARE_3MM = {
    "left": 102.0,
    "top": 214.0,
    "size": 80.0,
}

# ROI 放大图在参考画布中的大小：156x156（保持不变）
ROI_INSET_REF_SIZE = 156.0
ROI_INSET_SCALE = ROI_INSET_REF_SIZE / ROI_REF_W


# =========================
# 1. 基础工具函数
# =========================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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
        if p.is_file() and (p.suffix.lower() in [".dcm", ".ima"] or p.suffix == "")
    ]


def load_dicom_series_instance_sorted(folder: Path):
    files = list_dicom_files(folder)
    if not files:
        print(f"⚠️ 空目录: {folder}")
        return []

    series = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            _ = ds.pixel_array
            ds._source_path = str(f)
            series.append(ds)
        except Exception as e:
            print(f"⚠️ 读取失败: {f} | {e}")

    if not series:
        return []

    series.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
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


def hu_window_to_u8(img_hu: np.ndarray, clip_min: float, clip_max: float):
    x = np.clip(img_hu, clip_min, clip_max)
    x = (x - clip_min) / (clip_max - clip_min + 1e-8)
    x = (x * 255.0).round().astype(np.uint8)
    return x


def export_single_like_browser(ds, save_path):
    img_hu = dicom_to_hu(ds)
    img_hu, _ = center_crop_to_multiple(img_hu, PATCH_SIZE)
    img_u8 = hu_window_to_u8(img_hu, CLIP_MIN, CLIP_MAX)
    cv2.imwrite(str(save_path), img_u8)


def compute_msssim_norm(pred_norm: np.ndarray, gt_norm: np.ndarray, device: str) -> float:
    pred_t = torch.from_numpy(pred_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    val = piq.multi_scale_ssim(
        pred_t,
        gt_t,
        data_range=1.0,
        reduction="mean"
    )
    return float(val.item())


def compute_vif_norm(pred_norm: np.ndarray, gt_norm: np.ndarray, device: str) -> float:
    pred_t = torch.from_numpy(pred_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    val = piq.vif_p(
        pred_t,
        gt_t,
        data_range=1.0,
        reduction="mean"
    )
    return float(val.item())


# =========================
# 2. 通过浏览图编号精确取图
# =========================
def get_browser_index_pair(patient_id: str, thickness: str, s_idx: int):
    fd_folder = Path(FD_ROOTS[thickness]) / patient_id / f"full_{thickness}"
    qd_folder = Path(QD_ROOTS[thickness]) / patient_id / f"quarter_{thickness}"

    fd_series = load_dicom_series_instance_sorted(fd_folder)
    qd_series = load_dicom_series_instance_sorted(qd_folder)

    if not fd_series:
        raise RuntimeError(f"FD 目录无有效 DICOM: {fd_folder}")
    if not qd_series:
        raise RuntimeError(f"QD 目录无有效 DICOM: {qd_folder}")

    if s_idx < 0 or s_idx >= len(fd_series):
        raise IndexError(f"FD s_idx={s_idx} 越界，FD 当前共有 {len(fd_series)} 张")
    if s_idx < 0 or s_idx >= len(qd_series):
        raise IndexError(f"QD s_idx={s_idx} 越界，QD 当前共有 {len(qd_series)} 张")

    fd_ds = fd_series[s_idx]
    qd_ds = qd_series[s_idx]
    return fd_ds, qd_ds


# =========================
# 3. Patch 切分 / 回拼（无明显拼接缝版）
# =========================
def generate_sliding_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length < patch_size:
        raise ValueError(f"length={length} < patch_size={patch_size}")

    positions = list(range(0, length - patch_size + 1, stride))
    if not positions:
        positions = [0]

    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)

    return positions


def image_to_patches_overlap(img: np.ndarray, patch_size: int, stride: int):
    h, w = img.shape
    ys = generate_sliding_positions(h, patch_size, stride)
    xs = generate_sliding_positions(w, patch_size, stride)

    patches = []
    coords = []
    for y in ys:
        for x in xs:
            p = img[y:y + patch_size, x:x + patch_size]
            if p.shape != (patch_size, patch_size):
                raise RuntimeError(f"Patch shape error at {(y, x)}: {p.shape}")
            patches.append(p)
            coords.append((y, x))

    patches = np.stack(patches, axis=0).astype(np.float32)
    return patches, coords, (h, w)


def create_gaussian_weight_map(
    patch_size: int,
    sigma_ratio: float = 0.125,
    eps: float = 1e-6
) -> np.ndarray:
    ax = np.arange(patch_size, dtype=np.float32) - (patch_size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    sigma = patch_size * sigma_ratio
    weight = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    weight = weight.astype(np.float32)
    weight = np.maximum(weight, eps)
    weight = weight / (weight.max() + 1e-12)
    return weight


def stitch_patches_overlap_weighted(
    patches: np.ndarray,
    coords: List[Tuple[int, int]],
    out_hw: Tuple[int, int],
    patch_size: int,
    weight_map: np.ndarray,
    eps: float = 1e-6,
):
    h, w = out_hw
    out = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)

    for p, (y, x) in zip(patches, coords):
        out[y:y + patch_size, x:x + patch_size] += p * weight_map
        count[y:y + patch_size, x:x + patch_size] += weight_map

    out = out / np.maximum(count, eps)
    return out


# =========================
# 4. thickness 条件推理（无明显拼接缝版）
# =========================
def build_thickness_map_numpy(thickness: str, h: int, w: int) -> np.ndarray:
    th_id = THICKNESS_TO_ID[thickness]
    return np.full((h, w), fill_value=float(th_id), dtype=np.float32)


@torch.no_grad()
def predict_full_slice(
    model: nn.Module,
    qd_norm: np.ndarray,
    patch_size: int,
    device: str,
    infer_batch_size: int,
    cond_thickness: bool,
    thickness: str,
    residual_learning: bool = False,
    stride: int = PATCH_STRIDE,
    gaussian_sigma_ratio: float = GAUSSIAN_SIGMA_RATIO,
):
    patches, coords, out_hw = image_to_patches_overlap(
        qd_norm,
        patch_size=patch_size,
        stride=stride
    )

    if cond_thickness:
        h, w = qd_norm.shape
        th_map = build_thickness_map_numpy(thickness, h, w)
        th_patches, _, _ = image_to_patches_overlap(
            th_map,
            patch_size=patch_size,
            stride=stride
        )
    else:
        th_patches = None

    preds = []
    for i in range(0, len(patches), infer_batch_size):
        q_batch = patches[i:i + infer_batch_size]
        q_batch_t = torch.from_numpy(q_batch).unsqueeze(1).to(device)

        if cond_thickness:
            th_batch = th_patches[i:i + infer_batch_size]
            th_batch_t = torch.from_numpy(th_batch).unsqueeze(1).to(device)
            x = torch.cat([q_batch_t, th_batch_t], dim=1)
        else:
            x = q_batch_t

        out = model(x)

        # 这里不额外做 residual 回加：
        # 因为训练脚本中是直接 out = model(qd)，然后与 fd 做 L1；
        # 若 residual 已在模型内部实现，这里会自然生效。
        out = out.squeeze(1).cpu().numpy().astype(np.float32)
        preds.append(out)

    preds = np.concatenate(preds, axis=0)

    weight_map = create_gaussian_weight_map(
        patch_size=patch_size,
        sigma_ratio=gaussian_sigma_ratio,
        eps=WEIGHT_EPS
    )

    pred_full = stitch_patches_overlap_weighted(
        patches=preds,
        coords=coords,
        out_hw=out_hw,
        patch_size=patch_size,
        weight_map=weight_map,
        eps=WEIGHT_EPS,
    )

    if CLIP_PRED_TO_01:
        pred_full = np.clip(pred_full, 0.0, 1.0)

    return pred_full


# =========================
# 5. 指标
# =========================
class SSIMMetric(nn.Module):
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
    return float(ssim_metric(a, b).item())


def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt), dtype=np.float64))


def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2), dtype=np.float64))


def compute_nrmse(pred: np.ndarray, gt: np.ndarray) -> float:
    rmse = compute_rmse(pred, gt)
    denom = float(gt.max() - gt.min())
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


def evaluate_pair(pred_norm: np.ndarray, gt_norm: np.ndarray, ssim_metric: SSIMMetric, metric_space: str, device: str):
    psnr = compute_psnr_norm(pred_norm, gt_norm)
    ssim = compute_ssim_norm(pred_norm, gt_norm, ssim_metric, device)
    ms_ssim = compute_msssim_norm(pred_norm, gt_norm, device)
    vif = compute_vif_norm(pred_norm, gt_norm, device)

    if metric_space == "hu":
        pred_eval = norm_to_hu(pred_norm)
        gt_eval = norm_to_hu(gt_norm)
    elif metric_space == "norm":
        pred_eval = pred_norm
        gt_eval = gt_norm
    else:
        raise ValueError("metric_space must be 'hu' or 'norm'")

    mae = compute_mae(pred_eval, gt_eval)
    rmse = compute_rmse(pred_eval, gt_eval)
    nrmse = compute_nrmse(pred_eval, gt_eval)
    hfen = compute_hfen(pred_eval, gt_eval, device="cpu")

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "MS-SSIM": ms_ssim,
        "VIF": vif,
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "HFEN": hfen,
    }


# =========================
# 6. ROI / 标注换算
# =========================
def margins_to_roi_xywh(ref_w: float, ref_h: float, left: float, top: float, right: float, bottom: float):
    x0 = float(left)
    y0 = float(top)
    x1 = float(ref_w - right)
    y1 = float(ref_h - bottom)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI 参数非法")
    return x0, y0, x1 - x0, y1 - y0


def build_scaled_roi(cur_h: int, cur_w: int):
    rx_ref, ry_ref, rw_ref, rh_ref = margins_to_roi_xywh(
        ROI_REF_W, ROI_REF_H,
        ROI_REF["left"], ROI_REF["top"], ROI_REF["right"], ROI_REF["bottom"]
    )

    x0 = rx_ref / ROI_REF_W * cur_w
    y0 = ry_ref / ROI_REF_H * cur_h
    rw = rw_ref / ROI_REF_W * cur_w
    rh = rh_ref / ROI_REF_H * cur_h

    x0 = int(round(x0))
    y0 = int(round(y0))
    rw = int(round(rw))
    rh = int(round(rh))

    x0 = max(0, min(x0, cur_w - 1))
    y0 = max(0, min(y0, cur_h - 1))
    rw = max(1, min(rw, cur_w - x0))
    rh = max(1, min(rh, cur_h - y0))
    return x0, y0, rw, rh


def build_annotations_for_thickness(thickness: str, cur_h: int, cur_w: int) -> List[Dict[str, Any]]:
    annos = []

    if thickness == "1mm":
        sx = cur_w / REF_1MM_W
        sy = cur_h / REF_1MM_H
        r = CIRCLE_RADIUS_1MM * (sx + sy) / 2.0

        for x_ref, y_ref in POINTS_REF_XY_1MM:
            annos.append({
                "type": "circle",
                "cx": x_ref * sx,
                "cy": y_ref * sy,
                "r": r,
            })

    elif thickness == "3mm":
        sx = cur_w / REF_3MM_W
        sy = cur_h / REF_3MM_H

        x = SQUARE_3MM["left"] * sx
        y = SQUARE_3MM["top"] * sy
        w = SQUARE_3MM["size"] * sx
        h = SQUARE_3MM["size"] * sy

        annos.append({
            "type": "rect",
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

    else:
        raise ValueError(f"Unsupported thickness: {thickness}")

    return annos


def build_roi_local_annotations(annotations, roi_x: int, roi_y: int, roi_w: int, roi_h: int):
    local_annos = []
    for a in annotations:
        if a["type"] == "circle":
            cx = a["cx"] - roi_x
            cy = a["cy"] - roi_y
            local_annos.append({
                "type": "circle",
                "cx": cx,
                "cy": cy,
                "r": a["r"],
            })

        elif a["type"] == "rect":
            x = a["x"] - roi_x
            y = a["y"] - roi_y
            local_annos.append({
                "type": "rect",
                "x": x,
                "y": y,
                "w": a["w"],
                "h": a["h"],
            })
    return local_annos


def map_local_annos_to_inset(local_annos, roi_w: int, roi_h: int, inset_x: int, inset_y: int, inset_w: int, inset_h: int):
    sx = inset_w / max(roi_w, 1)
    sy = inset_h / max(roi_h, 1)

    mapped = []
    for a in local_annos:
        if a["type"] == "circle":
            mapped.append({
                "type": "circle",
                "cx": inset_x + a["cx"] * sx,
                "cy": inset_y + a["cy"] * sy,
                "r": a["r"] * (sx + sy) / 2.0,
            })
        elif a["type"] == "rect":
            mapped.append({
                "type": "rect",
                "x": inset_x + a["x"] * sx,
                "y": inset_y + a["y"] * sy,
                "w": a["w"] * sx,
                "h": a["h"] * sy,
            })
    return mapped


# =========================
# 7. 画布合成 + 可视化
# =========================
def gray_to_rgb_uint8(img_hu: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    img_u8 = hu_window_to_u8(img_hu, clip_min, clip_max)
    return np.stack([img_u8, img_u8, img_u8], axis=-1)


def paste_roi_inset_on_canvas(
    canvas_rgb: np.ndarray,
    roi_hu: np.ndarray,
    inset_size_px: int
) -> Tuple[int, int, int, int]:
    h, w, _ = canvas_rgb.shape

    roi_rgb = gray_to_rgb_uint8(roi_hu, CLIP_MIN, CLIP_MAX)
    roi_resized = cv2.resize(
        roi_rgb,
        (inset_size_px, inset_size_px),
        interpolation=cv2.INTER_CUBIC
    )

    x0 = 0
    y0 = h - inset_size_px
    x1 = x0 + inset_size_px
    y1 = y0 + inset_size_px

    canvas_rgb[y0:y1, x0:x1] = roi_resized
    return x0, y0, inset_size_px, inset_size_px


def draw_annotations(ax, annotations, color="red", linewidth=2.2, zorder=100):
    for a in annotations:
        if a["type"] == "circle":
            cx, cy = a["cx"], a["cy"]
            r = a["r"]

            outer = Circle(
                (cx, cy),
                radius=r,
                fill=False,
                edgecolor="white",
                linewidth=linewidth + 1.2,
                zorder=zorder
            )
            inner = Circle(
                (cx, cy),
                radius=r,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                zorder=zorder + 1
            )
            ax.add_patch(outer)
            ax.add_patch(inner)

        elif a["type"] == "rect":
            x, y, w, h = a["x"], a["y"], a["w"], a["h"]

            outer = Rectangle(
                (x, y), w, h,
                fill=False,
                edgecolor="white",
                linewidth=linewidth + 1.2,
                zorder=zorder
            )
            inner = Rectangle(
                (x, y), w, h,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                zorder=zorder + 1
            )
            ax.add_patch(outer)
            ax.add_patch(inner)


def save_single_slice_roi_vis(
    save_path: Path,
    pred_hu: np.ndarray,
    thickness: str,
    metrics: Dict[str, float] = None,
):
    h, w = pred_hu.shape

    rx, ry, rw, rh = build_scaled_roi(h, w)
    pred_roi = pred_hu[ry:ry + rh, rx:rx + rw]

    canvas_rgb = gray_to_rgb_uint8(pred_hu, CLIP_MIN, CLIP_MAX)

    inset_size_px = max(1, int(round(w * ROI_INSET_SCALE)))
    inset_x, inset_y, inset_w, inset_h = paste_roi_inset_on_canvas(
        canvas_rgb=canvas_rgb,
        roi_hu=pred_roi,
        inset_size_px=inset_size_px
    )

    annotations_main = build_annotations_for_thickness(thickness, h, w)
    annotations_roi_local = build_roi_local_annotations(annotations_main, rx, ry, rw, rh)
    annotations_inset = map_local_annos_to_inset(
        annotations_roi_local, rw, rh, inset_x, inset_y, inset_w, inset_h
    )

    fig = plt.figure(figsize=(8, 8), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(canvas_rgb, origin="upper")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_axis_off()

    if thickness == "1mm":
        roi_rect = Rectangle(
            (rx, ry),
            rw,
            rh,
            fill=False,
            edgecolor="yellow",
            linewidth=2.2,
            zorder=20
        )
        ax.add_patch(roi_rect)

        inset_border = Rectangle(
            (inset_x, inset_y),
            inset_w,
            inset_h,
            fill=False,
            edgecolor="white",
            linewidth=1.8,
            zorder=30
        )
        ax.add_patch(inset_border)

        draw_annotations(
            ax,
            annotations_inset,
            color="red",
            linewidth=2.0,
            zorder=40
        )

    elif thickness == "3mm":
        pass

    else:
        raise ValueError(f"Unsupported thickness: {thickness}")

    draw_annotations(
        ax,
        annotations_main,
        color="red",
        linewidth=2.4,
        zorder=100
    )

    if metrics is None:
        metrics = {}

    left_text = "\n".join([
        f"PSNR: {metrics.get('PSNR', 'N/A'):.4f}" if isinstance(metrics.get("PSNR", None), (int, float)) else f"PSNR: {metrics.get('PSNR', 'N/A')}",
        f"SSIM: {metrics.get('SSIM', 'N/A'):.4f}" if isinstance(metrics.get("SSIM", None), (int, float)) else f"SSIM: {metrics.get('SSIM', 'N/A')}",
        f"RMSE: {metrics.get('RMSE', 'N/A'):.4f}" if isinstance(metrics.get("RMSE", None), (int, float)) else f"RMSE: {metrics.get('RMSE', 'N/A')}",
    ])

    right_text = "\n".join([
        f"VIF: {metrics.get('VIF', 'N/A'):.4f}" if isinstance(metrics.get("VIF", None), (int, float)) else f"VIF: {metrics.get('VIF', 'N/A')}",
        f"MS-SSIM: {metrics.get('MS-SSIM', 'N/A'):.4f}" if isinstance(metrics.get("MS-SSIM", None), (int, float)) else f"MS-SSIM: {metrics.get('MS-SSIM', 'N/A')}",
    ])

    ax.text(
        10, 18,
        left_text,
        color="white",
        fontsize=16,
        ha="left",
        va="top",
        zorder=200,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="black",
            edgecolor="none",
            alpha=0.65
        )
    )

    ax.text(
        w - 10, 18,
        right_text,
        color="white",
        fontsize=16,
        ha="right",
        va="top",
        zorder=200,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="black",
            edgecolor="none",
            alpha=0.65
        )
    )

    fig.savefig(
        str(save_path),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
        facecolor="black",
        edgecolor="none"
    )
    plt.close(fig)

    print(f"[DEBUG] image size: w={w}, h={h}")
    print(f"[DEBUG] thickness: {thickness}")
    print(f"[DEBUG] scaled ROI: x={rx}, y={ry}, w={rw}, h={rh}")
    print(f"[DEBUG] pasted inset: x={inset_x}, y={inset_y}, w={inset_w}, h={inset_h}")
    print(f"[DEBUG] annotations_main: {annotations_main}")


# =========================
# 8. 模型加载
# =========================
def build_model_and_load_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    cond_thickness = bool(ckpt.get("cond_thickness", False))
    model_in_channels = int(ckpt.get("model_in_channels", 2 if cond_thickness else 1))
    residual_learning = bool(ckpt.get("residual_learning", False))

    try:
        use_sigmoid = bool(ckpt.get("use_sigmoid", False))
        model = UNet(
            in_channels=model_in_channels,
            out_channels=1,
            base_ch=64,
            use_sigmoid=use_sigmoid,
        ).to(device)
    except TypeError:
        model = UNet(
            in_channels=model_in_channels,
            out_channels=1,
            base_ch=64,
        ).to(device)
        use_sigmoid = None

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    print(f"[INFO] Checkpoint loaded from: {ckpt_path}")
    print(f"[INFO] cond_thickness={cond_thickness}, model_in_channels={model_in_channels}")
    print(f"[INFO] use_sigmoid={use_sigmoid}")
    print(f"[INFO] residual_learning={residual_learning}")
    print(f"[INFO] exp_name={ckpt.get('exp_name', '')}")

    return model, ckpt, cond_thickness, model_in_channels, use_sigmoid, residual_learning


# =========================
# 9. 主流程：严格按浏览图编号推理
# =========================
@torch.no_grad()
def run_target_cases(model: nn.Module, device: str, cond_thickness: bool, residual_learning: bool):
    out_dir = Path(SAVE_ROOT) / "target_cases"
    ensure_dir(out_dir)

    ssim_metric = SSIMMetric(window_size=11).to(device)
    all_results = []

    for case in TARGET_CASES:
        patient_id = case["patient_id"]
        thickness = case["thickness"]
        s_idx = case["s_idx"]
        case_name = case["name"]

        print("\n" + "=" * 100)
        print(f"[INFO] Processing case: {case_name}")

        fd_ds, qd_ds = get_browser_index_pair(
            patient_id=patient_id,
            thickness=thickness,
            s_idx=s_idx,
        )

        print(f"[INFO] browser slice = {thickness} FD_s{s_idx:04d} / QD_s{s_idx:04d}")
        print(f"[INFO] FD source path: {getattr(fd_ds, '_source_path', 'unknown')}")
        print(f"[INFO] QD source path: {getattr(qd_ds, '_source_path', 'unknown')}")
        print(f"[INFO] FD InstanceNumber: {get_instance_number(fd_ds)}")
        print(f"[INFO] QD InstanceNumber: {get_instance_number(qd_ds)}")

        export_single_like_browser(fd_ds, out_dir / f"{case_name}_fd_browser_check.png")
        export_single_like_browser(qd_ds, out_dir / f"{case_name}_qd_browser_check.png")

        fd_hu = dicom_to_hu(fd_ds)
        qd_hu = dicom_to_hu(qd_ds)

        fd_crop_hu, crop_info = center_crop_to_multiple(fd_hu, PATCH_SIZE)
        qd_crop_hu, _ = center_crop_to_multiple(qd_hu, PATCH_SIZE)

        fd_norm = hu_to_norm(fd_crop_hu)
        qd_norm = hu_to_norm(qd_crop_hu)

        pred_norm = predict_full_slice(
            model=model,
            qd_norm=qd_norm,
            patch_size=PATCH_SIZE,
            device=device,
            infer_batch_size=INFER_BATCH_SIZE,
            cond_thickness=cond_thickness,
            thickness=thickness,
            residual_learning=residual_learning,
        )

        pred_hu = norm_to_hu(pred_norm)
        metrics = evaluate_pair(pred_norm, fd_norm, ssim_metric, METRIC_SPACE, device)

        np.save(out_dir / f"{case_name}_qd_hu.npy", qd_crop_hu.astype(np.float32))
        np.save(out_dir / f"{case_name}_fd_hu.npy", fd_crop_hu.astype(np.float32))
        np.save(out_dir / f"{case_name}_pred_hu.npy", pred_hu.astype(np.float32))
        np.save(out_dir / f"{case_name}_pred_norm.npy", pred_norm.astype(np.float32))

        save_single_slice_roi_vis(
            save_path=out_dir / f"{case_name}_roi_vis.png",
            pred_hu=pred_hu,
            thickness=thickness,
            metrics=metrics,
        )

        scaled_roi = build_scaled_roi(pred_hu.shape[0], pred_hu.shape[1])
        annotations_main = build_annotations_for_thickness(thickness, pred_hu.shape[0], pred_hu.shape[1])

        result_item = {
            "case_name": case_name,
            "patient_id": patient_id,
            "thickness": thickness,
            "browser_s_idx": s_idx,
            "fd_source_path": getattr(fd_ds, "_source_path", ""),
            "qd_source_path": getattr(qd_ds, "_source_path", ""),
            "fd_instance_number": get_instance_number(fd_ds),
            "qd_instance_number": get_instance_number(qd_ds),
            "crop_info": {
                "sh": int(crop_info[0]),
                "sw": int(crop_info[1]),
                "nh": int(crop_info[2]),
                "nw": int(crop_info[3]),
            },
            "scaled_roi_xywh": {
                "x": int(scaled_roi[0]),
                "y": int(scaled_roi[1]),
                "w": int(scaled_roi[2]),
                "h": int(scaled_roi[3]),
            },
            "annotations_main": annotations_main,
            "stitch_mode": "overlap_gaussian_weighted",
            "patch_size": int(PATCH_SIZE),
            "patch_stride": int(PATCH_STRIDE),
            "gaussian_sigma_ratio": float(GAUSSIAN_SIGMA_RATIO),
            "cond_thickness": bool(cond_thickness),
            "residual_learning": bool(residual_learning),
            "metrics": metrics,
        }
        all_results.append(result_item)

        print(f"✅ 完成: {case_name}")
        print(f"    可视化: {out_dir / f'{case_name}_roi_vis.png'}")

    with open(out_dir / "target_case_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(f"[INFO] finished -> {out_dir}")


def main():
    ensure_dir(Path(SAVE_ROOT))
    model, ckpt, cond_thickness, model_in_channels, use_sigmoid, residual_learning = build_model_and_load_ckpt(CKPT_PATH, DEVICE)
    run_target_cases(model, DEVICE, cond_thickness, residual_learning)


if __name__ == "__main__":
    main()
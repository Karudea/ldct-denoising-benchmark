import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pydicom

from ldct_model import UNet


# =========================
# 0. 配置区（按你的 make_splits 来）
# =========================
DATA_ROOT = r"E:\LDCT\Training_Image_Data\1mm B30"

FD_ROOTS = {
    "1mm": os.path.join(DATA_ROOT, "full_1mm"),
    "3mm": os.path.join(DATA_ROOT, "full_3mm"),
}
QD_ROOTS = {
    "1mm": os.path.join(DATA_ROOT, "quarter_1mm"),
    "3mm": os.path.join(DATA_ROOT, "quarter_3mm"),
}

SPLIT_JSON = r"E:\LDCT\splits\patient_splits.json"

CKPT_PATH = r"E:\LDCT\experiments\A3_C4_residual_sigmoid_trainall_valall\seed_0\best_A3_C4_residual_sigmoid_trainall_valall_seed0.pth"
SAVE_ROOT = r"E:\LDCT\eval_results\A3_full_error_analysis"

PATCH_SIZE = 256
PATCH_STRIDE = 128
INFER_BATCH_SIZE = 8

GAUSSIAN_SIGMA_RATIO = 0.125
WEIGHT_EPS = 1e-6

CLIP_MIN = -160.0
CLIP_MAX = 240.0
CLIP_PRED_TO_01 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THICKNESS_TO_ID = {"1mm": 0, "3mm": 1}

# 画图参数
HIST_BINS = 100
HIST_MAX_ERROR_HU = 100.0
SAVE_ALL_ERROR_NPY = True

# CDF 下采样，避免内存爆炸
CDF_MAX_POINTS = 200000
CDF_RANDOM_SEED = 2025


# =========================
# 1. 工具函数
# =========================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_split_patients(split_json_path: str, split_name: str = "test") -> List[str]:
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    if split_name not in split_info:
        raise KeyError(f"{split_name} not found in {split_json_path}")

    patients = split_info[split_name]
    if not isinstance(patients, list):
        raise ValueError(f"{split_name} must be a list in {split_json_path}")

    return patients


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
        return []

    series = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            _ = ds.pixel_array
            ds._source_path = str(f)
            series.append(ds)
        except Exception as e:
            print(f"[WARN] failed to read: {f} | {e}")

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


# =========================
# 2. patch inference
# =========================
def generate_sliding_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length < patch_size:
        raise ValueError(f"length={length} < patch_size={patch_size}")

    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if not positions:
        positions = [0]
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
    weight = np.maximum(weight.astype(np.float32), eps)
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
    stride: int = PATCH_STRIDE,
    gaussian_sigma_ratio: float = GAUSSIAN_SIGMA_RATIO,
):
    patches, coords, out_hw = image_to_patches_overlap(
        qd_norm, patch_size=patch_size, stride=stride
    )

    if cond_thickness:
        h, w = qd_norm.shape
        th_map = build_thickness_map_numpy(thickness, h, w)
        th_patches, _, _ = image_to_patches_overlap(
            th_map, patch_size=patch_size, stride=stride
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
        out = out.squeeze(1).cpu().numpy().astype(np.float32)
        preds.append(out)

    preds = np.concatenate(preds, axis=0)

    weight_map = create_gaussian_weight_map(
        patch_size=patch_size,
        sigma_ratio=gaussian_sigma_ratio,
        eps=WEIGHT_EPS,
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
# 3. 误差统计
# =========================
def compute_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt), dtype=np.float64))


def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2), dtype=np.float64))


def summarize_abs_errors(abs_errors: np.ndarray) -> Dict[str, float]:
    abs_errors = np.asarray(abs_errors, dtype=np.float32).reshape(-1)
    mse = float(np.mean(abs_errors ** 2, dtype=np.float64))
    rmse = float(np.sqrt(mse))
    return {
        "num_pixels": int(abs_errors.size),
        "mae_from_abs_errors": float(np.mean(abs_errors, dtype=np.float64)),
        "rmse_from_abs_errors": rmse,
        "mse_from_abs_errors": mse,
        "median_ae": float(np.median(abs_errors)),
        "std_ae": float(np.std(abs_errors, dtype=np.float64)),
        "min_ae": float(np.min(abs_errors)),
        "max_ae": float(np.max(abs_errors)),
        "p90_ae": float(np.percentile(abs_errors, 90)),
        "p95_ae": float(np.percentile(abs_errors, 95)),
        "p99_ae": float(np.percentile(abs_errors, 99)),
    }


def plot_error_histogram(abs_errors: np.ndarray, save_path: Path, bins=100, max_error_hu=100.0):
    errs = np.asarray(abs_errors, dtype=np.float32).reshape(-1)
    errs_plot = errs[errs <= max_error_hu] if max_error_hu is not None else errs

    plt.figure(figsize=(8, 5))
    plt.hist(errs_plot, bins=bins)
    plt.xlabel("Absolute Error (HU)")
    plt.ylabel("Pixel Count")
    plt.title("Error Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()


def plot_error_histogram_log(abs_errors: np.ndarray, save_path: Path, bins=100, max_error_hu=100.0):
    errs = np.asarray(abs_errors, dtype=np.float32).reshape(-1)
    errs_plot = errs[errs <= max_error_hu] if max_error_hu is not None else errs

    plt.figure(figsize=(8, 5))
    plt.hist(errs_plot, bins=bins)
    plt.yscale("log")
    plt.xlabel("Absolute Error (HU)")
    plt.ylabel("Pixel Count (log)")
    plt.title("Error Distribution (log scale)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()


def plot_error_cdf(
    abs_errors: np.ndarray,
    save_path: Path,
    max_points: int = 200000,
    seed: int = 2025,
):
    errs = np.asarray(abs_errors, dtype=np.float32).reshape(-1)

    if errs.size == 0:
        return

    if errs.size > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(errs.size, size=max_points, replace=False)
        errs = errs[idx]

    errs = np.sort(errs)
    y = np.arange(1, len(errs) + 1, dtype=np.float32) / len(errs)

    plt.figure(figsize=(8, 5))
    plt.plot(errs, y, linewidth=1.2)
    plt.xlabel("Absolute Error (HU)")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Absolute Errors")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()


# =========================
# 4. 模型加载
# =========================
def build_model_and_load_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    cond_thickness = bool(ckpt.get("cond_thickness", False))
    model_in_channels = int(ckpt.get("model_in_channels", 2 if cond_thickness else 1))

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

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print(f"[INFO] loaded checkpoint: {ckpt_path}")
    print(f"[INFO] cond_thickness={cond_thickness}, model_in_channels={model_in_channels}")
    return model, cond_thickness


# =========================
# 5. 主流程
# =========================
@torch.no_grad()
def run_full_error_analysis():
    save_root = Path(SAVE_ROOT)
    ensure_dir(save_root)

    test_patients = load_split_patients(SPLIT_JSON, split_name="test")
    print(f"[INFO] test patients from split = {test_patients}")

    model, cond_thickness = build_model_and_load_ckpt(CKPT_PATH, DEVICE)

    for thickness in ["1mm", "3mm"]:
        print("\n" + "=" * 100)
        print(f"[INFO] Running error analysis for {thickness}")

        fd_root = Path(FD_ROOTS[thickness])
        qd_root = Path(QD_ROOTS[thickness])

        out_dir = save_root / thickness
        ensure_dir(out_dir)

        all_abs_errors = []
        per_slice_records = []
        total_slices = 0
        valid_patients = 0

        for patient_id in test_patients:
            fd_folder = fd_root / patient_id / f"full_{thickness}"
            qd_folder = qd_root / patient_id / f"quarter_{thickness}"

            fd_series = load_dicom_series_instance_sorted(fd_folder)
            qd_series = load_dicom_series_instance_sorted(qd_folder)

            if len(fd_series) == 0 or len(qd_series) == 0:
                print(f"[WARN] skip empty patient: {patient_id}")
                continue

            valid_patients += 1

            if len(fd_series) != len(qd_series):
                print(f"[WARN] length mismatch: {patient_id} | FD={len(fd_series)} QD={len(qd_series)}")
                n = min(len(fd_series), len(qd_series))
                fd_series = fd_series[:n]
                qd_series = qd_series[:n]

            for s_idx, (fd_ds, qd_ds) in enumerate(zip(fd_series, qd_series)):
                fd_hu = dicom_to_hu(fd_ds)
                qd_hu = dicom_to_hu(qd_ds)

                fd_crop_hu, crop_info = center_crop_to_multiple(fd_hu, PATCH_SIZE)
                qd_crop_hu, _ = center_crop_to_multiple(qd_hu, PATCH_SIZE)

                # 与主指标一致：GT先clip到窗口，再映射回HU
                fd_norm = hu_to_norm(fd_crop_hu)
                qd_norm = hu_to_norm(qd_crop_hu)

                pred_norm = predict_full_slice(
                    model=model,
                    qd_norm=qd_norm,
                    patch_size=PATCH_SIZE,
                    device=DEVICE,
                    infer_batch_size=INFER_BATCH_SIZE,
                    cond_thickness=cond_thickness,
                    thickness=thickness,
                )

                pred_eval_hu = norm_to_hu(pred_norm)
                gt_eval_hu = norm_to_hu(fd_norm)

                abs_err_hu = np.abs(pred_eval_hu - gt_eval_hu).astype(np.float32)
                all_abs_errors.append(abs_err_hu.reshape(-1))

                per_slice_records.append({
                    "patient_id": patient_id,
                    "slice_index": int(s_idx),
                    "instance_number_fd": int(getattr(fd_ds, "InstanceNumber", 0)),
                    "instance_number_qd": int(getattr(qd_ds, "InstanceNumber", 0)),
                    "crop_info": {
                        "sh": int(crop_info[0]),
                        "sw": int(crop_info[1]),
                        "nh": int(crop_info[2]),
                        "nw": int(crop_info[3]),
                    },
                    "mae": compute_mae(pred_eval_hu, gt_eval_hu),
                    "rmse": compute_rmse(pred_eval_hu, gt_eval_hu),
                })

                total_slices += 1
                if total_slices % 50 == 0:
                    print(f"[INFO] {thickness}: processed {total_slices} slices")

        if len(all_abs_errors) == 0:
            print(f"[WARN] no valid slices found for {thickness}")
            continue

        all_abs_errors = np.concatenate(all_abs_errors, axis=0)
        summary = summarize_abs_errors(all_abs_errors)
        summary["thickness"] = thickness
        summary["num_slices"] = total_slices
        summary["num_patients"] = valid_patients
        summary["split_json"] = SPLIT_JSON
        summary["test_patients"] = test_patients

        if SAVE_ALL_ERROR_NPY:
            np.save(out_dir / f"all_abs_errors_{thickness}.npy", all_abs_errors.astype(np.float32))

        with open(out_dir / f"summary_{thickness}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(out_dir / f"per_slice_metrics_{thickness}.json", "w", encoding="utf-8") as f:
            json.dump(per_slice_records, f, indent=2, ensure_ascii=False)

        plot_error_histogram(
            all_abs_errors,
            save_path=out_dir / f"hist_{thickness}.png",
            bins=HIST_BINS,
            max_error_hu=HIST_MAX_ERROR_HU,
        )

        plot_error_histogram_log(
            all_abs_errors,
            save_path=out_dir / f"hist_log_{thickness}.png",
            bins=HIST_BINS,
            max_error_hu=HIST_MAX_ERROR_HU,
        )

        plot_error_cdf(
            all_abs_errors,
            save_path=out_dir / f"cdf_{thickness}.png",
            max_points=CDF_MAX_POINTS,
            seed=CDF_RANDOM_SEED,
        )

        print(f"\n[RESULT] {thickness}")
        print(f"  num_slices = {summary['num_slices']}")
        print(f"  MAE        = {summary['mae_from_abs_errors']:.6f}")
        print(f"  RMSE       = {summary['rmse_from_abs_errors']:.6f}")
        print(f"  MSE        = {summary['mse_from_abs_errors']:.6f}")
        print(f"  Median AE  = {summary['median_ae']:.6f}")
        print(f"  P90 AE     = {summary['p90_ae']:.6f}")
        print(f"  P95 AE     = {summary['p95_ae']:.6f}")
        print(f"  P99 AE     = {summary['p99_ae']:.6f}")
        print(f"  Max AE     = {summary['max_ae']:.6f}")

    print("\n" + "=" * 100)
    print(f"[INFO] done -> {save_root}")


if __name__ == "__main__":
    run_full_error_analysis()
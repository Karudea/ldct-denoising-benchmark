import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pydicom

# ======================
# 0. 配置
# ======================

DATA_ROOT = r"E:\LDCT\Training_Image_Data\1mm B30"
TEST_QD_ROOT = r"E:\LDCT\Testing_Image_Data\1mm B30\QD_1mm"

SPLIT_JSON = r"E:\LDCT\splits\patient_splits.json"

# 建议单独输出到新目录，避免覆盖你原来的 2D baseline prepared
OUT_ROOT = r"E:\LDCT\prepared_2p5d_3slice_1mm3mm_hu_-160_240"

PATCH_SIZE = 256
CLIP_MIN = -160
CLIP_MAX = 240

THICKNESSES = ["1mm", "3mm"]

FD_ROOTS = {
    "1mm": os.path.join(DATA_ROOT, "full_1mm"),
    "3mm": os.path.join(DATA_ROOT, "full_3mm"),
}

QD_ROOTS = {
    "1mm": os.path.join(DATA_ROOT, "quarter_1mm"),
    "3mm": os.path.join(DATA_ROOT, "quarter_3mm"),
}

# slice 匹配容差（单位：mm）
Z_TOL_MM = 1e-2  # 0.01mm

# ====== 2.5D 配置 ======
# 可选 3 或 5
NUM_INPUT_SLICES = 3

# 边界处理方式：
# "replicate"：s-1 不存在时，用 s 自己替代；推荐，样本数不减少
# "drop"：边界不生成样本；更严格，但样本数会减少
BOUNDARY_MODE = "replicate"

assert NUM_INPUT_SLICES in [3, 5]
assert BOUNDARY_MODE in ["replicate", "drop"]


# ======================
# 1. 工具函数
# ======================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_splits(split_json_path: str):
    with open(split_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    for k in ["train", "val", "test", "external_test"]:
        splits.setdefault(k, [])
    return splits


def list_dicom_files(folder: Path):
    if not folder.exists():
        return []
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in [".dcm", ".ima", ""]
    ]


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def get_z_position(ds) -> Optional[float]:
    """
    优先从 ImagePositionPatient 取 z；若没有，再尝试 SliceLocation。
    """
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        z = safe_float(ipp[2])
        if z is not None:
            return z

    sl = getattr(ds, "SliceLocation", None)
    z = safe_float(sl)
    return z


def get_instance_number(ds) -> int:
    try:
        return int(getattr(ds, "InstanceNumber", 0))
    except Exception:
        return 0


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


def hu_to_norm(img):
    img = np.clip(img, CLIP_MIN, CLIP_MAX)
    return (img - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)


def center_crop_to_multiple(img: np.ndarray, multiple: int):
    h, w = img.shape
    nh = (h // multiple) * multiple
    nw = (w // multiple) * multiple
    sh = (h - nh) // 2
    sw = (w - nw) // 2
    return img[sh:sh + nh, sw:sw + nw]


def extract_patches(img: np.ndarray, patch_size: int):
    patches = []
    h, w = img.shape
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            p = img[y:y + patch_size, x:x + patch_size]
            if p.shape == (patch_size, patch_size):
                patches.append(p)
    return patches


def quantize_z(z: float, tol: float) -> float:
    return round(z / tol) * tol


def match_fd_qd_pairs(fd_series: List, qd_series: List) -> List[Tuple]:
    """
    FD/QD 切片配对：优先按 z 匹配；失败则退化为 InstanceNumber 对齐
    返回: [(fd_ds, qd_ds), ...]
    """
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

        if not pairs:
            print("⚠️ z 匹配失败：无可配对切片，将退化为 InstanceNumber 对齐")
        else:
            if miss > 0:
                print(f"⚠️ z 匹配有缺失：QD 有 {miss} 张在 FD 中找不到对应 z（已跳过）")
            return pairs

    print("⚠️ DICOM 缺少完整 z 信息，退化为 InstanceNumber 对齐（可能存在错配风险）")

    fd_sorted = sorted(fd_series, key=get_instance_number)
    qd_sorted = sorted(qd_series, key=get_instance_number)

    min_n = min(len(fd_sorted), len(qd_sorted))
    return [(fd_sorted[i], qd_sorted[i]) for i in range(min_n)]


def resolve_neighbor_indices(center_idx: int, total_slices: int, num_input_slices: int, boundary_mode: str):
    """
    以 center_idx 为中心，返回 2.5D 输入所需的相邻 slice 下标列表
    例如 num_input_slices=3 -> [i-1, i, i+1]
    """
    radius = num_input_slices // 2
    indices = []

    for j in range(center_idx - radius, center_idx + radius + 1):
        if 0 <= j < total_slices:
            indices.append(j)
        else:
            if boundary_mode == "drop":
                return None
            # replicate
            jj = min(max(j, 0), total_slices - 1)
            indices.append(jj)

    return indices


# ======================
# 2. 将配对后的整张 slice 预处理为 patch 列表
# ======================

def preprocess_paired_slices_to_patch_lists(
    pairs: List[Tuple],
    patient_id: str,
    thickness: str,
):
    """
    把每个 paired slice 处理成 patch 列表，便于后面构造 2.5D 输入
    返回：
        qd_patch_lists: List[List[np.ndarray]]
        fd_patch_lists: List[List[np.ndarray]]
    其中：
        qd_patch_lists[i][j] = 第 i 张 slice 的第 j 个 qd patch
    """
    qd_patch_lists = []
    fd_patch_lists = []

    expected_patch_num = None

    for i, (fd_ds, qd_ds) in enumerate(pairs):
        fd = dicom_to_hu(fd_ds)
        qd = dicom_to_hu(qd_ds)

        fd = center_crop_to_multiple(fd, PATCH_SIZE)
        qd = center_crop_to_multiple(qd, PATCH_SIZE)

        # 保险：FD / QD 裁剪后尺寸必须一致
        if fd.shape != qd.shape:
            print(f"⚠️ {patient_id} {thickness} slice#{i}: FD/QD 裁剪后尺寸不一致，跳过")
            continue

        fd = hu_to_norm(fd)
        qd = hu_to_norm(qd)

        fd_patches = extract_patches(fd, PATCH_SIZE)
        qd_patches = extract_patches(qd, PATCH_SIZE)

        if len(fd_patches) != len(qd_patches) or len(fd_patches) == 0:
            print(f"⚠️ {patient_id} {thickness} slice#{i}: patch 数不一致或为空，跳过")
            continue

        if expected_patch_num is None:
            expected_patch_num = len(fd_patches)
        else:
            if len(fd_patches) != expected_patch_num:
                print(
                    f"⚠️ {patient_id} {thickness} slice#{i}: patch 数={len(fd_patches)} "
                    f"与前面 slice 的 patch 数={expected_patch_num} 不一致，跳过"
                )
                continue

        qd_patch_lists.append([p.astype(np.float32) for p in qd_patches])
        fd_patch_lists.append([p.astype(np.float32) for p in fd_patches])

    return qd_patch_lists, fd_patch_lists


# ======================
# 3. 处理 paired 病人（1mm / 3mm）
# ======================

def process_paired_patient(
    patient_id: str,
    thickness: str,
    split_name: str,
):
    fd_dir = Path(FD_ROOTS[thickness]) / patient_id / f"full_{thickness}"
    qd_dir = Path(QD_ROOTS[thickness]) / patient_id / f"quarter_{thickness}"

    fd_series = load_dicom_series(fd_dir)
    qd_series = load_dicom_series(qd_dir)

    if not fd_series or not qd_series:
        print(f"❌ {patient_id} {thickness}: FD/QD 缺失，跳过")
        return

    pairs = match_fd_qd_pairs(fd_series, qd_series)
    if not pairs:
        print(f"❌ {patient_id} {thickness}: 无法配对切片，跳过")
        return

    qd_patch_lists, fd_patch_lists = preprocess_paired_slices_to_patch_lists(
        pairs, patient_id, thickness
    )

    if not qd_patch_lists or not fd_patch_lists:
        print(f"❌ {patient_id} {thickness}: 预处理后无有效 slice，跳过")
        return

    assert len(qd_patch_lists) == len(fd_patch_lists)
    num_valid_slices = len(qd_patch_lists)
    num_patches_per_slice = len(qd_patch_lists[0])

    out_in = Path(OUT_ROOT) / split_name / thickness / "inputs"
    out_gt = Path(OUT_ROOT) / split_name / thickness / "targets"
    ensure_dir(out_in)
    ensure_dir(out_gt)

    total = 0
    dropped_center_slices = 0

    for center_i in range(num_valid_slices):
        neighbor_indices = resolve_neighbor_indices(
            center_i,
            num_valid_slices,
            NUM_INPUT_SLICES,
            BOUNDARY_MODE,
        )

        if neighbor_indices is None:
            dropped_center_slices += 1
            continue

        for patch_j in range(num_patches_per_slice):
            # stack 相邻 slice 的同一 patch 位置
            stacked_input = np.stack(
                [qd_patch_lists[s_idx][patch_j] for s_idx in neighbor_indices],
                axis=0,  # (C,H,W)
            ).astype(np.float32)

            center_target = fd_patch_lists[center_i][patch_j].astype(np.float32)  # (H,W)

            name = f"{patient_id}_{thickness}_s{center_i:03d}_p{patch_j:02d}.npy"
            np.save(out_in / name, stacked_input)
            np.save(out_gt / name, center_target)

            total += 1

    print(
        f"✅ {split_name} | {patient_id} | {thickness} | "
        f"valid_slices={num_valid_slices} | patches_per_slice={num_patches_per_slice} | "
        f"saved_patches={total} | dropped_center_slices={dropped_center_slices}"
    )


# ======================
# 4. external_test（QD only，1mm）
# ======================

def preprocess_qd_only_slices_to_patch_lists(qd_series: List, patient_id: str):
    qd_patch_lists = []
    expected_patch_num = None

    for i, ds in enumerate(qd_series):
        qd = dicom_to_hu(ds)
        qd = center_crop_to_multiple(qd, PATCH_SIZE)
        qd = hu_to_norm(qd)

        qd_patches = extract_patches(qd, PATCH_SIZE)
        if len(qd_patches) == 0:
            print(f"⚠️ external {patient_id} slice#{i}: patch 为空，跳过")
            continue

        if expected_patch_num is None:
            expected_patch_num = len(qd_patches)
        else:
            if len(qd_patches) != expected_patch_num:
                print(
                    f"⚠️ external {patient_id} slice#{i}: patch 数={len(qd_patches)} "
                    f"与前面 slice 的 patch 数={expected_patch_num} 不一致，跳过"
                )
                continue

        qd_patch_lists.append([p.astype(np.float32) for p in qd_patches])

    return qd_patch_lists


def process_external_patient(patient_id: str):
    qd_dir = Path(TEST_QD_ROOT) / patient_id / "quarter_1mm"
    qd_series = load_dicom_series(qd_dir)

    if not qd_series:
        print(f"❌ external {patient_id}: 空")
        return

    qd_patch_lists = preprocess_qd_only_slices_to_patch_lists(qd_series, patient_id)
    if not qd_patch_lists:
        print(f"❌ external {patient_id}: 预处理后无有效 slice")
        return

    num_valid_slices = len(qd_patch_lists)
    num_patches_per_slice = len(qd_patch_lists[0])

    out_in = Path(OUT_ROOT) / "external_test" / "1mm" / "inputs"
    ensure_dir(out_in)

    total = 0
    dropped_center_slices = 0

    for center_i in range(num_valid_slices):
        neighbor_indices = resolve_neighbor_indices(
            center_i,
            num_valid_slices,
            NUM_INPUT_SLICES,
            BOUNDARY_MODE,
        )

        if neighbor_indices is None:
            dropped_center_slices += 1
            continue

        for patch_j in range(num_patches_per_slice):
            stacked_input = np.stack(
                [qd_patch_lists[s_idx][patch_j] for s_idx in neighbor_indices],
                axis=0,  # (C,H,W)
            ).astype(np.float32)

            name = f"{patient_id}_1mm_s{center_i:03d}_p{patch_j:02d}.npy"
            np.save(out_in / name, stacked_input)
            total += 1

    print(
        f"✅ external_test | {patient_id} | "
        f"valid_slices={num_valid_slices} | patches_per_slice={num_patches_per_slice} | "
        f"saved_patches={total} | dropped_center_slices={dropped_center_slices}"
    )


# ======================
# 5. 主流程
# ======================

def main():
    print("=" * 80)
    print(f"[INFO] Preparing 2.5D dataset")
    print(f"[INFO] OUT_ROOT = {OUT_ROOT}")
    print(f"[INFO] NUM_INPUT_SLICES = {NUM_INPUT_SLICES}")
    print(f"[INFO] BOUNDARY_MODE = {BOUNDARY_MODE}")
    print("=" * 80)

    splits = load_splits(SPLIT_JSON)

    main_patients = set(splits.get("train", [])) | set(splits.get("val", [])) | set(splits.get("test", []))
    ext_patients = set(splits.get("external_test", []))
    assert main_patients.isdisjoint(ext_patients), \
        f"external_test 与 train/val/test 病人有重叠：{sorted(main_patients & ext_patients)}"

    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    os.makedirs(OUT_ROOT, exist_ok=True)

    # train / val / test
    for split in ["train", "val", "test"]:
        patients = splits.get(split, [])
        if not patients:
            continue
        print(f"\n==== {split.upper()} ({len(patients)}) ====")
        for pid in patients:
            for th in THICKNESSES:
                process_paired_patient(pid, th, split)

    # external_test（QD-only，1mm）
    ext = splits.get("external_test", [])
    if ext:
        print(f"\n==== EXTERNAL_TEST ({len(ext)}) ====")
        for pid in ext:
            process_external_patient(pid)

    print("\n🎉 2.5D dataset preparation finished.")


if __name__ == "__main__":
    main()
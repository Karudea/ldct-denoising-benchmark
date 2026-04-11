# repatch_256_to_64.py
from pathlib import Path
import numpy as np

SRC_ROOT = Path(r"E:\LDCT\prepared_1mm3mm_hu_-160_240")
DST_ROOT = Path(r"E:\LDCT\prepared_ctformer_64_hu_-160_240")

SPLITS = ["train", "val", "test"]
THICKNESSES = ["1mm", "3mm"]
PATCH_SIZE = 64


def split_patch(arr, patch_size=64):
    h, w = arr.shape
    assert h % patch_size == 0 and w % patch_size == 0, f"shape={arr.shape} 不能整除 {patch_size}"
    patches = []
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patches.append((i, j, arr[i:i+patch_size, j:j+patch_size]))
    return patches


def main():
    for split in SPLITS:
        for th in THICKNESSES:
            in_dir = SRC_ROOT / split / th / "inputs"
            tg_dir = SRC_ROOT / split / th / "targets"

            out_in_dir = DST_ROOT / split / th / "inputs"
            out_tg_dir = DST_ROOT / split / th / "targets"
            out_in_dir.mkdir(parents=True, exist_ok=True)
            out_tg_dir.mkdir(parents=True, exist_ok=True)

            input_files = sorted(in_dir.glob("*.npy"))
            print(f"[INFO] Processing {split}/{th} | files={len(input_files)}")

            for f in input_files:
                x = np.load(f).astype(np.float32)
                y = np.load(tg_dir / f.name).astype(np.float32)

                x_patches = split_patch(x, PATCH_SIZE)
                y_patches = split_patch(y, PATCH_SIZE)

                assert len(x_patches) == len(y_patches)

                stem = f.stem
                for k, ((ix, jx, xp), (iy, jy, yp)) in enumerate(zip(x_patches, y_patches)):
                    assert ix == iy and jx == jy
                    out_name = f"{stem}_p{k:02d}.npy"
                    np.save(out_in_dir / out_name, xp)
                    np.save(out_tg_dir / out_name, yp)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
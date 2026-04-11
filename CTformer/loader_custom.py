# loader_custom.py
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random


class LDCTPatchNPYDataset(Dataset):
    """
    适配 CTformer 的 LDCT patch 数据集
    使用原始较大 patch（如 256x256），在训练/验证阶段在线裁剪出 64x64 子 patch。

    支持：
    - thickness = 1mm / 3mm / all
    - cond_thickness = False / True

    目录结构：
    root/
      train/
        1mm/inputs/*.npy
        1mm/targets/*.npy
        3mm/inputs/*.npy
        3mm/targets/*.npy
      val/
      test/
      external_test/
    """

    def __init__(
        self,
        root: str,
        split: str,
        thickness: str = "all",
        crop_size: int = 64,
        random_crop: bool = True,
        max_samples: Optional[int] = None,
        cond_thickness: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.thickness = thickness
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.max_samples = max_samples
        self.cond_thickness = cond_thickness

        assert split in ["train", "val", "test", "external_test"]
        assert thickness in ["1mm", "3mm", "all"]

        self.input_files: List[Path] = []
        self.target_files: List[Path] = []
        self.thickness_ids: List[float] = []  # 1mm -> 0.0, 3mm -> 1.0

        thickness_list = ["1mm", "3mm"] if thickness == "all" else [thickness]

        for th in thickness_list:
            input_dir = self.root / split / th / "inputs"
            target_dir = self.root / split / th / "targets"

            if not input_dir.exists():
                raise FileNotFoundError(f"[ERROR] 输入目录不存在: {input_dir}")

            if split != "external_test" and not target_dir.exists():
                raise FileNotFoundError(f"[ERROR] 标签目录不存在: {target_dir}")

            files = sorted(input_dir.glob("*.npy"))
            if len(files) == 0:
                raise RuntimeError(f"[ERROR] 目录为空: {input_dir}")

            th_id = 0.0 if th == "1mm" else 1.0

            self.input_files.extend(files)
            self.thickness_ids.extend([th_id] * len(files))

            if split != "external_test":
                self.target_files.extend([target_dir / f.name for f in files])

        if split != "external_test":
            for tgt_path in self.target_files:
                if not tgt_path.exists():
                    raise FileNotFoundError(f"[ERROR] 找不到对应标签文件: {tgt_path}")

        if self.max_samples is not None:
            self.input_files = self.input_files[:self.max_samples]
            self.thickness_ids = self.thickness_ids[:self.max_samples]
            if split != "external_test":
                self.target_files = self.target_files[:self.max_samples]

        print(
            f"[INFO] Dataset loaded | split={split} | thickness={thickness} "
            f"| cond_thickness={cond_thickness} | samples={len(self.input_files)} "
            f"| crop_size={crop_size}"
        )

    def __len__(self):
        return len(self.input_files)

    def _load_npy(self, path: Path) -> np.ndarray:
        arr = np.load(path).astype(np.float32)

        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim != 2:
            raise ValueError(f"[ERROR] npy 维度异常: {path} | shape={arr.shape}")

        return arr

    def _crop_array(self, arr: np.ndarray, top: int, left: int, cs: int) -> np.ndarray:
        return arr[top:top + cs, left:left + cs]

    def _get_crop_coords(self, h: int, w: int, cs: int):
        if h < cs or w < cs:
            raise ValueError(
                f"[ERROR] patch 尺寸小于 crop_size: input shape=({h}, {w}), crop_size={cs}"
            )

        if self.random_crop and self.split == "train":
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
        else:
            top = (h - cs) // 2
            left = (w - cs) // 2

        return top, left

    def __getitem__(self, idx):
        inp = self._load_npy(self.input_files[idx])
        th_id = self.thickness_ids[idx]

        h, w = inp.shape
        cs = self.crop_size
        top, left = self._get_crop_coords(h, w, cs)

        inp = self._crop_array(inp, top, left, cs)

        if self.cond_thickness:
            th_map = np.full_like(inp, fill_value=th_id, dtype=np.float32)
            inp = np.stack([inp, th_map], axis=0)  # (2, H, W)
        else:
            inp = np.expand_dims(inp, axis=0)      # (1, H, W)

        inp = torch.from_numpy(inp)

        if self.split == "external_test":
            return inp, self.input_files[idx].name

        tgt = self._load_npy(self.target_files[idx])
        tgt = self._crop_array(tgt, top, left, cs)
        tgt = torch.from_numpy(tgt).unsqueeze(0)   # (1, H, W)

        return inp, tgt


def get_loader(
    root: str,
    split: str,
    thickness: str = "all",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    crop_size: int = 64,
    random_crop: bool = True,
    max_samples: Optional[int] = None,
    cond_thickness: bool = False,
):
    dataset = LDCTPatchNPYDataset(
        root=root,
        split=split,
        thickness=thickness,
        crop_size=crop_size,
        random_crop=random_crop,
        max_samples=max_samples,
        cond_thickness=cond_thickness,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(shuffle if split == "train" else False),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader
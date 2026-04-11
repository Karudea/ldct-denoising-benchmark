# ldct_npy_dataset.py
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


class LDCTPatchNPYDataset(Dataset):
    """
    支持 thickness 维度 + cond thickness 的 LDCT patch 数据集

    root/
      train/
        1mm/inputs/*.npy
        1mm/targets/*.npy
        3mm/inputs/*.npy
        3mm/targets/*.npy
      val/
        1mm/...
        3mm/...
      test/
        1mm/...
        3mm/...
      external_test/
        1mm/inputs/*.npy
        3mm/inputs/*.npy

    返回：
      cond_thickness=False:
        - train/val/test: inp, tgt
        - external_test:  inp, filename

      cond_thickness=True:
        - train/val/test: inp_with_th, tgt
        - external_test:  inp_with_th, filename

    其中：
      - 原图通道固定在第 0 通道
      - thickness map 通道在第 1 通道
      - 1mm -> 0.0
      - 3mm -> 1.0
    """

    def __init__(
        self,
        root: str,
        split: str,
        thickness: str = "all",      # "1mm" | "3mm" | "all"
        cond_thickness: bool = False
    ):
        self.root = Path(root)
        self.split = split
        self.thickness = thickness
        self.cond_thickness = cond_thickness

        assert thickness in ["1mm", "3mm", "all"], \
            f"thickness must be one of ['1mm', '3mm', 'all'], got {thickness}"

        self.input_files: List[Path] = []
        self.target_files: List[Path] = []
        self.thickness_ids: List[float] = []

        if thickness == "all":
            thickness_list = ["1mm", "3mm"]
        else:
            thickness_list = [thickness]

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
            if len(self.input_files) != len(self.target_files):
                raise RuntimeError(
                    f"[ERROR] 输入和标签数量不一致: "
                    f"inputs={len(self.input_files)}, targets={len(self.target_files)}"
                )

        if len(self.input_files) != len(self.thickness_ids):
            raise RuntimeError(
                f"[ERROR] 输入和 thickness_ids 数量不一致: "
                f"inputs={len(self.input_files)}, thickness_ids={len(self.thickness_ids)}"
            )

        print(
            f"[INFO] Dataset loaded | split={split} | thickness={thickness} "
            f"| cond_thickness={cond_thickness} | patches={len(self.input_files)}"
        )

    def __len__(self):
        return len(self.input_files)

    def _append_thickness_map(self, inp: torch.Tensor, th_id: float) -> torch.Tensor:
        """
        inp: (1, H, W)
        return:
            cond_thickness=False -> (1, H, W)
            cond_thickness=True  -> (2, H, W)
        """
        if not self.cond_thickness:
            return inp

        _, h, w = inp.shape
        th_map = torch.full((1, h, w), fill_value=th_id, dtype=torch.float32)
        inp = torch.cat([inp, th_map], dim=0)
        return inp

    def __getitem__(self, idx: int):
        inp = np.load(self.input_files[idx]).astype(np.float32)
        inp = torch.from_numpy(inp).unsqueeze(0)  # (1, H, W)

        th_id = self.thickness_ids[idx]
        inp = self._append_thickness_map(inp, th_id)

        if self.split == "external_test":
            return inp, self.input_files[idx].name

        tgt = np.load(self.target_files[idx]).astype(np.float32)
        tgt = torch.from_numpy(tgt).unsqueeze(0)  # (1, H, W)

        return inp, tgt
# ldct_npy_dataset.py
from pathlib import Path
from typing import List, Union
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
        1mm/inputs/*.npy
        1mm/targets/*.npy
        3mm/inputs/*.npy
        3mm/targets/*.npy
      test/
        1mm/inputs/*.npy
        1mm/targets/*.npy
        3mm/inputs/*.npy
        3mm/targets/*.npy
      external_test/
        1mm/inputs/*.npy
        3mm/inputs/*.npy
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        thickness: str = "all",          # "1mm" | "3mm" | "all"
        cond_thickness: bool = False,    # 是否拼接 thickness condition map
    ):
        self.root = Path(root)
        self.split = split
        self.thickness = thickness
        self.cond_thickness = cond_thickness

        assert thickness in ["1mm", "3mm", "all"]
        assert split in ["train", "val", "test", "external_test"]

        self.input_files: List[Path] = []
        self.target_files: List[Path] = []

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

            self.input_files.extend(files)

            if split != "external_test":
                target_candidates = [target_dir / f.name for f in files]
                for t in target_candidates:
                    if not t.exists():
                        raise FileNotFoundError(f"[ERROR] 标签文件不存在: {t}")
                self.target_files.extend(target_candidates)

        if self.split != "external_test" and len(self.input_files) != len(self.target_files):
            raise RuntimeError(
                f"[ERROR] 输入与标签数量不一致: "
                f"inputs={len(self.input_files)}, targets={len(self.target_files)}"
            )

        print(
            f"[INFO] Dataset loaded | split={split} | thickness={thickness} "
            f"| cond_thickness={cond_thickness} | patches={len(self.input_files)}"
        )

    def __len__(self):
        return len(self.input_files)

    @staticmethod
    def _thickness_to_scalar(th_name: str) -> float:
        if th_name == "1mm":
            return 0.0
        elif th_name == "3mm":
            return 1.0
        raise ValueError(f"[ERROR] 未知 thickness: {th_name}")

    def _build_input_tensor(self, input_path: Path) -> torch.Tensor:
        inp = np.load(input_path).astype(np.float32)
        inp = torch.from_numpy(inp).unsqueeze(0)  # (1, H, W)

        if not self.cond_thickness:
            return inp

        # input_path = root / split / th / inputs / xxx.npy
        th_name = input_path.parent.parent.name
        th_scalar = self._thickness_to_scalar(th_name)

        cond_map = torch.full_like(inp, fill_value=th_scalar)  # (1, H, W)
        inp = torch.cat([inp, cond_map], dim=0)  # (2, H, W)
        return inp

    def __getitem__(self, idx):
        input_path = self.input_files[idx]
        inp = self._build_input_tensor(input_path)

        if self.split == "external_test":
            return inp, input_path.name

        tgt = np.load(self.target_files[idx]).astype(np.float32)
        tgt = torch.from_numpy(tgt).unsqueeze(0)  # (1, H, W)

        return inp, tgt
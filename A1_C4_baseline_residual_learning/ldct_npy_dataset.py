# ldct_npy_dataset.py
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class LDCTPatchNPYDataset(Dataset):
    """
    支持 thickness 维度的 LDCT patch 数据集

    root/
      train/
        1mm/inputs/*.npy
        1mm/targets/*.npy
        3mm/inputs/*.npy
        3mm/targets/*.npy
      val/
        1mm/...
      test/
        ...
      external_test/
        1mm/inputs/*.npy
        3mm/inputs/*.npy

    参数
    ----
    root : str | Path
        prepared 数据根目录
    split : str
        "train" | "val" | "test" | "external_test"
    thickness : str
        "1mm" | "3mm" | "all"
    cond_thickness : bool
        是否启用 thickness condition channel
        - False: 输入 shape = (1, H, W)
        - True : 输入 shape = (2, H, W)，第二通道为 thickness map
    """

    VALID_SPLITS = {"train", "val", "test", "external_test"}
    VALID_THICKNESS = {"1mm", "3mm", "all"}

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        thickness: str = "all",
        cond_thickness: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.thickness = thickness
        self.cond_thickness = cond_thickness

        if self.split not in self.VALID_SPLITS:
            raise ValueError(f"[ERROR] 非法 split: {self.split}, 可选: {self.VALID_SPLITS}")
        if self.thickness not in self.VALID_THICKNESS:
            raise ValueError(f"[ERROR] 非法 thickness: {self.thickness}, 可选: {self.VALID_THICKNESS}")

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
                missing_targets = [str(p) for p in target_candidates if not p.exists()]
                if missing_targets:
                    preview = "\n".join(missing_targets[:10])
                    raise FileNotFoundError(
                        f"[ERROR] 存在缺失标签文件，最多展示前 10 个：\n{preview}"
                    )
                self.target_files.extend(target_candidates)

        if self.split != "external_test" and len(self.input_files) != len(self.target_files):
            raise RuntimeError(
                f"[ERROR] 输入和标签数量不一致: "
                f"inputs={len(self.input_files)}, targets={len(self.target_files)}"
            )

        print(
            f"[INFO] Dataset loaded | split={self.split} | thickness={self.thickness} "
            f"| cond_thickness={self.cond_thickness} | patches={len(self.input_files)}"
        )

    def __len__(self):
        return len(self.input_files)

    @staticmethod
    def _thickness_to_scalar(thickness_name: str) -> float:
        """
        1mm -> 0.0
        3mm -> 1.0
        """
        if thickness_name == "1mm":
            return 0.0
        if thickness_name == "3mm":
            return 1.0
        raise ValueError(f"[ERROR] 未知 thickness_name: {thickness_name}")

    def _build_input_tensor(self, inp_np: np.ndarray, input_path: Path) -> torch.Tensor:
        """
        构造输入张量：
        - 不带 cond: (1, H, W)
        - 带 cond : (2, H, W)
        """
        if inp_np.ndim != 2:
            raise ValueError(f"[ERROR] 期望输入 npy 为 2D array, 实际 shape={inp_np.shape}, file={input_path}")

        inp_tensor = torch.from_numpy(inp_np.astype(np.float32)).unsqueeze(0)  # (1, H, W)

        if not self.cond_thickness:
            return inp_tensor

        # input_path 结构: root/split/thickness/inputs/xxx.npy
        thickness_name = input_path.parent.parent.name
        th_scalar = self._thickness_to_scalar(thickness_name)

        cond_map = torch.full_like(inp_tensor, fill_value=th_scalar)  # (1, H, W)
        inp_tensor = torch.cat([inp_tensor, cond_map], dim=0)         # (2, H, W)

        return inp_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, str]]:
        input_path = self.input_files[idx]

        inp_np = np.load(input_path).astype(np.float32)
        inp_tensor = self._build_input_tensor(inp_np, input_path)

        if self.split == "external_test":
            return inp_tensor, input_path.name

        target_path = self.target_files[idx]
        tgt_np = np.load(target_path).astype(np.float32)

        if tgt_np.ndim != 2:
            raise ValueError(f"[ERROR] 期望 target npy 为 2D array, 实际 shape={tgt_np.shape}, file={target_path}")

        tgt_tensor = torch.from_numpy(tgt_np).unsqueeze(0)  # (1, H, W)
        return inp_tensor, tgt_tensor
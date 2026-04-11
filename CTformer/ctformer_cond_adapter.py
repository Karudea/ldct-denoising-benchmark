# ctformer_cond_adapter.py
import torch
import torch.nn as nn


class CTformerCondAdapter(nn.Module):
    """
    给原始单通道 CTformer 增加 thickness condition 的包装器。
    输入:
        x: (B, 2, H, W)
           第0通道 = LDCT patch
           第1通道 = thickness map (1mm=0, 3mm=1)
    处理:
        先通过 1x1 conv 融合成 1 通道，再送给原始 CTformer
    输出:
        与原始 CTformer 一致，形状 (B, 1, H, W)
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.input_adapter = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_as_identity_like()

    def _init_as_identity_like(self):
        """
        初始化为：
        output ≈ 第0通道（原始 LDCT）
        thickness 通道初始权重为 0
        这样刚开始训练时，不会因为新增条件分支导致输入分布剧烈变化
        """
        with torch.no_grad():
            self.input_adapter.weight.zero_()
            self.input_adapter.bias.zero_()
            self.input_adapter.weight[0, 0, 0, 0] = 1.0  # 保留原始图像通道
            self.input_adapter.weight[0, 1, 0, 0] = 0.0  # thickness 通道从 0 开始学

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"[ERROR] 输入必须是 4D tensor, got shape={tuple(x.shape)}")
        if x.shape[1] != 2:
            raise ValueError(f"[ERROR] CTformerCondAdapter 期望输入通道=2, got {x.shape[1]}")

        x = self.input_adapter(x)   # (B, 1, H, W)
        out = self.base_model(x)
        return out
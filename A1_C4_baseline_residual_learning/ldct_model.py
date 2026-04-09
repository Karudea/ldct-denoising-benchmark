# ldct_model.py
import torch
import torch.nn as nn
import numpy as np


# ===== HU 归一化 / 反归一化 =====
def hu_to_norm(
    hu_img: np.ndarray,
    clip_min: float = -1000.0,
    clip_max: float = 2000.0
) -> np.ndarray:
    """
    将 HU 图像裁剪到 [clip_min, clip_max]，再线性归一化到 [0, 1]
    """
    hu = np.clip(hu_img, clip_min, clip_max)
    norm = (hu - clip_min) / (clip_max - clip_min)
    return norm.astype(np.float32)


def norm_to_hu(
    norm_img: np.ndarray,
    clip_min: float = -1000.0,
    clip_max: float = 2000.0
) -> np.ndarray:
    """
    将 [0, 1] 归一化图像反变换回 HU 空间（近似）
    """
    hu = norm_img * (clip_max - clip_min) + clip_min
    return hu.astype(np.float32)


# ===== U-Net 模型定义 =====
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    简单 4 层 U-Net，用于 LDCT -> NDCT 去噪 / 重建
    输入输出大小：
      input : (N, in_channels, H, W)
      output: (N, 1, H, W)

    当前版本：
    - 不使用 sigmoid
    - 引入 residual learning
    - 网络学习 residual，然后输出 = input_image + residual

    注意：
    若启用 cond thickness，建议：
      in_channels = 2
    其中：
      x[:, 0:1, :, :] 为原始图像
      x[:, 1:2, :, :] 为 thickness condition map
    """

    def __init__(self, in_channels=1, out_channels=1, base_ch=64, use_residual=True):
        super().__init__()
        self.use_residual = use_residual

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        # Output: 预测 residual
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # residual 学习时，只对原始图像通道做 identity
        if self.use_residual:
            identity = x[:, 0:1, :, :]
        else:
            identity = None

        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        # Bottleneck
        x5 = self.bottleneck(self.pool(x4))

        # Decoder
        d4 = self.up4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        residual = self.out_conv(d1)

        if self.use_residual:
            out = identity + residual
        else:
            out = residual

        return out
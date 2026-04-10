# ldct_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ===== HU 归一化 / 反归一化 =====
def hu_to_norm(
    hu_img: np.ndarray,
    clip_min: float = -1000.0,
    clip_max: float = 2000.0
) -> np.ndarray:
    hu = np.clip(hu_img, clip_min, clip_max)
    norm = (hu - clip_min) / (clip_max - clip_min)
    return norm.astype(np.float32)


def norm_to_hu(
    norm_img: np.ndarray,
    clip_min: float = -1000.0,
    clip_max: float = 2000.0
) -> np.ndarray:
    hu = norm_img * (clip_max - clip_min) + clip_min
    return hu.astype(np.float32)


# ===== 基础模块 =====
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


# ===== U-Net（Deep Supervision + Sigmoid + Residual）=====
class UNet(nn.Module):
    """
    Deep supervision version
    - 支持 in_channels=1 或 2（C4 mixed cond thickness 时为 2）
    - 使用 sigmoid + residual（logit residual）输出
    - residual 基于输入图像主通道（x[:, 0:1, ...]）
    - forward 返回:
        deep_supervision=False -> Tensor
        deep_supervision=True  -> dict(main, aux_d2, aux_d3)
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_ch=64,
        deep_supervision=True,
        use_residual=True,
        eps=1e-6,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_residual = use_residual
        self.eps = eps

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

        # Final output head（输出 residual logit）
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

        # Deep supervision heads（也输出 residual logit）
        if self.deep_supervision:
            self.ds_head_d2 = nn.Conv2d(base_ch * 2, out_channels, kernel_size=1)
            self.ds_head_d3 = nn.Conv2d(base_ch * 4, out_channels, kernel_size=1)

    def _safe_logit(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 [0,1] 图像安全映射到 logit 域，避免 0/1 导致 inf
        """
        x = torch.clamp(x, self.eps, 1.0 - self.eps)
        return torch.log(x / (1.0 - x))

    def _apply_sigmoid_residual(self, residual_logit: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        """
        residual_logit + logit(identity) -> sigmoid
        identity: 原始输入图像主通道 (B,1,H,W)
        """
        if not self.use_residual:
            return torch.sigmoid(residual_logit)

        identity_logit = self._safe_logit(identity)
        out = torch.sigmoid(residual_logit + identity_logit)
        return out

    def forward(self, x):
        input_size = x.shape[-2:]  # (H, W)

        # residual 的 identity 只取图像主通道，不包含 thickness map
        identity = x[:, 0:1, :, :]

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

        # ===== main output =====
        main_residual_logit = self.out_conv(d1)
        out = self._apply_sigmoid_residual(main_residual_logit, identity)

        if not self.deep_supervision:
            return out

        # ===== aux outputs =====
        aux_d2_residual_logit = self.ds_head_d2(d2)
        aux_d3_residual_logit = self.ds_head_d3(d3)

        identity_d2 = F.interpolate(identity, size=aux_d2_residual_logit.shape[-2:], mode="bilinear", align_corners=False)
        identity_d3 = F.interpolate(identity, size=aux_d3_residual_logit.shape[-2:], mode="bilinear", align_corners=False)

        out_d2 = self._apply_sigmoid_residual(aux_d2_residual_logit, identity_d2)
        out_d3 = self._apply_sigmoid_residual(aux_d3_residual_logit, identity_d3)

        out_d2_up = F.interpolate(out_d2, size=input_size, mode="bilinear", align_corners=False)
        out_d3_up = F.interpolate(out_d3, size=input_size, mode="bilinear", align_corners=False)

        return {
            "main": out,
            "aux_d2": out_d2_up,
            "aux_d3": out_d3_up,
        }
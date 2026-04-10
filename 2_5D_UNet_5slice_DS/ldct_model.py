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
    支持：
    - 2D / 2.5D 输入
    - cond thickness
    - residual learning
    - no sigmoid
    - deep supervision

    DS 输出：
    - main_out
    - aux_d2_out
    - aux_d3_out
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_ch=64,
        use_sigmoid=False,
        deep_supervision=False,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.deep_supervision = deep_supervision

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

        # Main output
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=1)

        # Aux heads
        if self.deep_supervision:
            self.aux_d2 = nn.Conv2d(base_ch * 2, out_channels, kernel_size=1)
            self.aux_d3 = nn.Conv2d(base_ch * 4, out_channels, kernel_size=1)

    @staticmethod
    def _select_identity(x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]

        if c == 1:
            return x[:, 0:1, :, :]
        elif c == 2:
            return x[:, 0:1, :, :]
        elif c == 3:
            return x[:, 1:2, :, :]
        elif c == 4:
            return x[:, 1:2, :, :]
        elif c == 5:
            return x[:, 2:3, :, :]
        elif c == 6:
            return x[:, 2:3, :, :]
        else:
            raise ValueError(f"[ERROR] 不支持的输入通道数: {c}")

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x

    def forward(self, x):
        identity = self._select_identity(x)

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

        main_res = self.out_conv(d1)
        main_out = identity + main_res
        main_out = self._apply_activation(main_out)

        if not self.deep_supervision:
            return main_out

        aux_d2_res = self.aux_d2(d2)  # 1/2 resolution
        aux_d3_res = self.aux_d3(d3)  # 1/4 resolution

        aux_d2_res = F.interpolate(
            aux_d2_res,
            size=identity.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        aux_d3_res = F.interpolate(
            aux_d3_res,
            size=identity.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        aux_d2_out = identity + aux_d2_res
        aux_d3_out = identity + aux_d3_res

        aux_d2_out = self._apply_activation(aux_d2_out)
        aux_d3_out = self._apply_activation(aux_d3_out)

        return main_out, aux_d2_out, aux_d3_out
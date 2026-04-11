# redcnn_model.py
import torch
import torch.nn as nn


class REDCNN(nn.Module):
    """
    RED-CNN for LDCT denoising / reconstruction

    Input:
        (N, C, H, W)
        - C=1: 普通单通道输入
        - C=2: [LDCT image, thickness map]

    Output:
        (N, 1, H, W)

    Notes:
    - symmetric skip connections
    - global residual connection
    - no sigmoid output
    - 当 C=2 时，全局残差仅加回第 1 个通道（原图）
    """

    def __init__(self, in_channels=2, out_channels=1, num_features=96):
        super().__init__()

        # ===== Encoder =====
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv2d(num_features, num_features, kernel_size=5, stride=1, padding=2)

        # ===== Decoder =====
        self.deconv1 = nn.ConvTranspose2d(num_features, num_features, kernel_size=5, stride=1, padding=2)
        self.deconv2 = nn.ConvTranspose2d(num_features, num_features, kernel_size=5, stride=1, padding=2)
        self.deconv3 = nn.ConvTranspose2d(num_features, num_features, kernel_size=5, stride=1, padding=2)
        self.deconv4 = nn.ConvTranspose2d(num_features, num_features, kernel_size=5, stride=1, padding=2)
        self.deconv5 = nn.ConvTranspose2d(num_features, out_channels, kernel_size=5, stride=1, padding=2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 全局残差只加回原图通道，不加 thickness map
        residual_global = x[:, :1, :, :]

        # ===== Encoder =====
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(x4))

        # ===== Decoder + symmetric skip connections =====
        y1 = self.relu(self.deconv1(x5) + x4)
        y2 = self.relu(self.deconv2(y1) + x3)
        y3 = self.relu(self.deconv3(y2) + x2)
        y4 = self.relu(self.deconv4(y3) + x1)
        y5 = self.deconv5(y4)

        # ===== Global residual =====
        out = y5 + residual_global
        return out


if __name__ == "__main__":
    model = REDCNN(in_channels=2, out_channels=1, num_features=96)
    x = torch.randn(2, 2, 256, 256)
    y = model(x)
    print("input shape :", x.shape)
    print("output shape:", y.shape)
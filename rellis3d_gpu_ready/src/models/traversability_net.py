"""Self-contained LightweightTraversabilityNet architecture for RELLIS-3D GPU training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBlock(nn.Module):
    """Convolution + BatchNorm + Hardswish block with optional residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Hardswish(inplace=True)
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)

class LightweightTraversabilityNet(nn.Module):
    """Edge-optimized encoder-decoder network with MobileNetV3-Small backbone."""

    def __init__(
        self,
        num_classes: int = 3,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout_prob: float = 0.20
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mobilenet = models.mobilenet_v3_small(weights=weights)
        self.enc1 = mobilenet.features[:2]   # 16 channels, stride /2
        self.enc2 = mobilenet.features[2:4]  # 24 channels, stride /4
        self.enc3 = mobilenet.features[4:9]  # 48 channels, stride /8
        self.enc4 = mobilenet.features[9:12] # 96 channels, stride /16
        enc_channels = [16, 24, 48, 96]

        # Decoder with Skip Connections (Feature Pyramid Upsampling)
        self.dec3 = ConvBlock(enc_channels[3] + enc_channels[2], 64)
        self.dec2 = ConvBlock(64 + enc_channels[1], 48)
        self.dec1 = ConvBlock(48 + enc_channels[0], 32)

        # Final 3-Class Prediction Head
        self.head = nn.Sequential(
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Hardswish(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:] # (H, W)

        # Encoder stages
        f1 = self.enc1(x)   # /2
        f2 = self.enc2(f1)  # /4
        f3 = self.enc3(f2)  # /8
        f4 = self.enc4(f3)  # /16

        # Decoder with skip connections
        up3 = F.interpolate(f4, size=f3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([up3, f3], dim=1))

        up2 = F.interpolate(d3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, f2], dim=1))

        up1 = F.interpolate(d2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, f1], dim=1))

        # Prediction head and upsampling to full input resolution
        logits_low = self.head(d1)
        logits = F.interpolate(logits_low, size=input_size, mode="bilinear", align_corners=False)
        return logits

import torch
import torch.nn as nn


class Block(nn.Module):
    """One residual block: (B, in_channels, H, W) -> (B, out_channels, H / stride, W / stride)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        # The shortcut turns x into the residual branch's output shape, as cheaply as it can.
        if in_channels != out_channels:  # new channels need learned weights
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        elif stride != 1:  # same channels, smaller: keep every other pixel, as AdaFace does
            self.shortcut = nn.MaxPool2d(kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        return out


class IResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # keep 112x112 resolution for the input image (stride=1, padding=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        # 4 stages of 2 blocks; each stage's first block halves H and W: 112 -> 56 -> 28 -> 14 -> 7.
        self.body = nn.Sequential(
            Block(64, 64, stride=2),
            Block(64, 64, stride=1),
            Block(64, 128, stride=2),
            Block(128, 128, stride=1),
            Block(128, 256, stride=2),
            Block(256, 256, stride=1),
            Block(256, 512, stride=2),
            Block(512, 512, stride=1),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(p=0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False),
        )

    def forward(self, x):
        # Define the forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.body(x)  # runs the 8 blocks in order
        x = self.output(x)
        return x

import torch.nn as nn


class ResNet18Block(nn.Module):
    """Single ResNet-18 BasicBlock with optional internal downsample projection.

    Takes TWO inputs to avoid any tensor slicing inside forward() — slicing
    generates a memref.subview whose symbolic dimensions become anonymous affine
    symbols that crash ConvertToSingleProducerSingleConsumer's StoreLoad pass.

    forward(x, identity):
      x        — pre-padded input for the conv chain (H+4, W+4), so both 3×3
                 padding=0 convs can run without generating MLIR pad ops.
      identity — already-cropped / already-downsampled tensor at the output
                 spatial resolution (H, W), added directly to the conv output.

    Dimension trace (stride=1, e.g. 58×58 in, 54×54 out):
      conv1 (3×3, stride=1, pad=0): H+4 → H+2
      bn1 + relu
      conv2 (3×3, stride=1, pad=0): H+2 → H
      identity input:               (C_in,  H, W)   — added as-is (no downsample)

    Dimension trace (stride=2, e.g. 58×58 in, 26×26 out):
      conv1 (3×3, stride=2, pad=0): H+4 → (H+1)//2+1  ≈ H//2+1
      bn1 + relu
      conv2 (3×3, stride=1, pad=0): H//2+1 → H//2-1
      identity input:               (C_in, H//2-1, W//2-1) → 1×1 conv → (C_out, …)

    Used for all 8 BasicBlocks across ResNet-18's four stages:
      Layer 1: (64 →64,  stride=1) × 2  — no downsample
      Layer 2: (64 →128, stride=2) × 1  — with 1×1 channel-change conv
               (128→128, stride=1) × 1  — no downsample
      Layer 3: (128→256, stride=2) × 1  — with 1×1 channel-change conv
               (256→256, stride=1) × 1  — no downsample
      Layer 4: (256→512, stride=2) × 1  — with 1×1 channel-change conv
               (512→512, stride=1) × 1  — no downsample
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, identity):
        # x:        pre-padded input  (no slicing needed — avoids memref.subview)
        # identity: pre-sized tensor at the output spatial resolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

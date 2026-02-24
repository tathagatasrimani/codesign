import torch.nn as nn


class ResNet50Bottleneck(nn.Module):
    """Single ResNet-50 Bottleneck block (1×1 → 3×3 → 1×1) with optional downsample.

    Takes TWO inputs to avoid tensor slicing or padding ops inside forward() —
    slicing generates a memref.subview and internal F.pad generates tensor.pad,
    both of which cause crashes in the StreamHLS affine loop passes.

    forward(x, identity):
      x        — pre-padded by (1,1,1,1) externally so that conv2 (3×3, no pad)
                 produces the correct output spatial resolution.
      identity — tensor at the output spatial resolution added after the conv chain.
                 If in_channels != out_channels, a 1×1 downsample conv is applied.

    Dimension trace (stride=1, e.g. x=(H+2,W+2), out=(H,W)):
      conv1 (1×1, stride=1): (H+2) × (W+2) → (H+2) × (W+2)
      bn1 + relu
      conv2 (3×3, stride=1): (H+2) → H
      bn2 + relu
      conv3 (1×1, stride=1): H × W → H × W
      identity added as-is (no downsample when in_channels == out_channels)

    Dimension trace (stride=2, e.g. x=(H+2,W+2) from 56×56 in, out=28×28):
      conv1 (1×1, stride=1): (H+2) × (W+2) → (H+2) × (W+2)
      bn1 + relu
      conv2 (3×3, stride=2): (H+2) → H//2
      bn2 + relu
      conv3 (1×1, stride=1): H//2 × W//2 → H//2 × W//2
      identity: 1×1 conv (stride=2) → (out_channels, H//2, W//2)

    Used for all Bottleneck blocks across ResNet-50's four stages:
      Layer 1: (64→256,  stride=1) × 1  — with 1×1 channel-change conv (layer1_down)
               (256→256, stride=1) × 2  — no downsample             (layer1_block)
      Layer 2: (256→512,  stride=2) × 1  — with 1×1 channel-change conv (layer2_down)
               (512→512,  stride=1) × 3  — no downsample             (layer2_block)
      Layer 3: (512→1024, stride=2) × 1  — with 1×1 channel-change conv (layer3_down)
               (1024→1024, stride=1) × 5 — no downsample             (layer3_block)
      Layer 4: (1024→2048, stride=2) × 1 — with 1×1 channel-change conv (layer4_down)
               (2048→2048, stride=1) × 2 — no downsample             (layer4_block)
    """

    def __init__(self, in_channels, bottleneck, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, kernel_size=3,
                               stride=stride, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.conv3 = nn.Conv2d(bottleneck, out_channels, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, identity):
        # x:        pre-padded by (1,1,1,1) — no internal pad ops needed
        # identity: tensor at the output spatial resolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)  # 3×3, no padding — uses pre-padded x via conv1
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return self.relu(out)

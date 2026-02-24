import os
import sys
import torch.nn as nn
import torch.nn.functional as F

# Add parent vitis directory so sibling sub-block packages are importable.
_here = os.path.dirname(os.path.abspath(__file__))
_vitis = os.path.dirname(_here)
if _vitis not in sys.path:
    sys.path.insert(0, _vitis)

from ResNet18Stem.ResNet18Stem import ResNet18Stem
from ResNet50Bottleneck.ResNet50Bottleneck import ResNet50Bottleneck
from ResNet50Classifier.ResNet50Classifier import ResNet50Classifier


class ResNet50Inference(nn.Module):
    """ResNet-50 image classification inference.

    Each individual Bottleneck block is a separate sub-module so that StreamHLS
    can optimize each one independently.

    Sub-block attributes matched to block_types in the system YAML:
      stem         -> ResNet18Stem (reused; same 7×7 conv + 2× maxpool)
      layer1_down  -> ResNet50Bottleneck(64,   64,  256, stride=1)  called 1×  (with proj)
      layer1_block -> ResNet50Bottleneck(256,  64,  256, stride=1)  called 2×
      layer2_down  -> ResNet50Bottleneck(256,  128, 512, stride=2)  called 1×  (with proj)
      layer2_block -> ResNet50Bottleneck(512,  128, 512, stride=1)  called 3×
      layer3_down  -> ResNet50Bottleneck(512,  256, 1024, stride=2) called 1×  (with proj)
      layer3_block -> ResNet50Bottleneck(1024, 256, 1024, stride=1) called 5×
      layer4_down  -> ResNet50Bottleneck(1024, 512, 2048, stride=2) called 1×  (with proj)
      layer4_block -> ResNet50Bottleneck(2048, 512, 2048, stride=1) called 2×
      classifier   -> ResNet50Classifier
    """

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.stem = ResNet18Stem()

        # Layer 1: first block expands 64→256, then two plain blocks
        self.layer1_down  = ResNet50Bottleneck(64,   64,  256, stride=1)
        self.layer1_block = ResNet50Bottleneck(256,  64,  256, stride=1)

        # Layer 2: first block downsizes spatially and expands 256→512, then three plain
        self.layer2_down  = ResNet50Bottleneck(256,  128, 512, stride=2)
        self.layer2_block = ResNet50Bottleneck(512,  128, 512, stride=1)

        # Layer 3: first block downsizes and expands 512→1024, then five plain
        self.layer3_down  = ResNet50Bottleneck(512,  256, 1024, stride=2)
        self.layer3_block = ResNet50Bottleneck(1024, 256, 1024, stride=1)

        # Layer 4: first block downsizes and expands 1024→2048, then two plain
        self.layer4_down  = ResNet50Bottleneck(1024, 512, 2048, stride=2)
        self.layer4_block = ResNet50Bottleneck(2048, 512, 2048, stride=1)

        self.classifier = ResNet50Classifier(num_classes=num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            (batch, num_classes)
        """
        # Stem: pre-pad 224→230 (3 per side) so the 7×7 conv runs with pad=0
        x = self.stem(F.pad(x, (3, 3, 3, 3)))                              # → (batch, 64, 56, 56)

        # Layer 1 – first block expands channels 64→256, stride=1 (56→56)
        # Pre-pad by (1,1,1,1) so conv2 (3×3, no pad) produces correct output size
        x = self.layer1_down(F.pad(x, (1, 1, 1, 1)), x)                    # 58→56, 64→256
        x = self.layer1_block(F.pad(x, (1, 1, 1, 1)), x)                   # 58→56, 256→256
        x = self.layer1_block(F.pad(x, (1, 1, 1, 1)), x)                   # 58→56, 256→256

        # Layer 2 – first block downsizes (256→512, 56→28), then three plain (512→512, 28→28)
        x = self.layer2_down(F.pad(x, (1, 1, 1, 1)), x)                    # 58→28, 256→512
        x = self.layer2_block(F.pad(x, (1, 1, 1, 1)), x)                   # 30→28, 512→512
        x = self.layer2_block(F.pad(x, (1, 1, 1, 1)), x)                   # 30→28, 512→512
        x = self.layer2_block(F.pad(x, (1, 1, 1, 1)), x)                   # 30→28, 512→512

        # Layer 3 – first block downsizes (512→1024, 28→14), then five plain (1024→1024, 14→14)
        x = self.layer3_down(F.pad(x, (1, 1, 1, 1)), x)                    # 30→14, 512→1024
        x = self.layer3_block(F.pad(x, (1, 1, 1, 1)), x)                   # 16→14, 1024→1024
        x = self.layer3_block(F.pad(x, (1, 1, 1, 1)), x)                   # 16→14, 1024→1024
        x = self.layer3_block(F.pad(x, (1, 1, 1, 1)), x)                   # 16→14, 1024→1024
        x = self.layer3_block(F.pad(x, (1, 1, 1, 1)), x)                   # 16→14, 1024→1024
        x = self.layer3_block(F.pad(x, (1, 1, 1, 1)), x)                   # 16→14, 1024→1024

        # Layer 4 – first block downsizes (1024→2048, 14→7), then two plain (2048→2048, 7→7)
        x = self.layer4_down(F.pad(x, (1, 1, 1, 1)), x)                    # 16→7, 1024→2048
        x = self.layer4_block(F.pad(x, (1, 1, 1, 1)), x)                   # 9→7, 2048→2048
        x = self.layer4_block(F.pad(x, (1, 1, 1, 1)), x)                   # 9→7, 2048→2048

        x = self.classifier(x)
        return x

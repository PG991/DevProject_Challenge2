import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    """3×3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride,
        padding=1, bias=False
    )

class BasicBlock(nn.Module):
    """Basic ResNet block without SE, with optional Dropout"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.3):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            # Adjust dimensions with 1×1 convolution
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AudioResNet14(nn.Module):
    """
    ResNet-14 variant for 1-channel log-Mel-spectrogram input.
    Uses 2 blocks per layer (total conv layers: 1 + 2*4 = 9, plus final FC → ≈14 layers).
    """
    def __init__(self, num_classes=50, dropout=0.3):
        super().__init__()
        # Initial stem: one conv layer with 3×7 kernel (spans 3 Mel bins × 7 time frames),
        # stride (1,2) reduces time dimension by 2 but keeps Mel dimension roughly intact.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 7),
            stride=(1, 2),
            padding=(1, 3),
            bias=False
        )
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Now build four layers, each with 2 BasicBlocks
        self.in_planes = 64
        self.layer1 = self._make_layer(planes=64,  blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(planes=128, blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(planes=256, blocks=2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(planes=512, blocks=2, stride=2, dropout=dropout)

        # Global avg-pool and classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

        # Weight initialization (Kaiming for convs, constant for BatchNorm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride, dropout):
        """
        Create a layer consisting of `blocks` BasicBlock modules.
        `stride` applies to the first block to downsample spatially/time-freq.
        """
        layers = []
        # First block may downsample if stride != 1
        layers.append(BasicBlock(self.in_planes, planes, stride=stride, dropout=dropout))
        self.in_planes = planes * BasicBlock.expansion
        # The remaining blocks have stride=1
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 1, n_mels=64, time_steps]
        x = self.conv1(x)      # → [B, 64, 64, T/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # → [B, 64, 32, T/4]

        x = self.layer1(x)     # → [B, 64, 32, T/4]
        x = self.layer2(x)     # → [B,128, 16, T/8]
        x = self.layer3(x)     # → [B,256,  8, T/16]
        x = self.layer4(x)     # → [B,512,  4, T/32]

        x = self.avgpool(x)    # → [B,512,1,1]
        x = torch.flatten(x, 1)  # → [B,512]
        x = self.fc(x)         # → [B,50]
        return x

# Factory alias, so `config.model_constructor = "resnet14()"` works
def resnet14(num_classes=50, dropout=0.3):
    return AudioResNet14(num_classes=num_classes, dropout=dropout)

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # erster 3×3-Conv
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # zweiter 3×3-Conv
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # falls wir die Residual-Verbindung skalieren müssen
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AudioResNet18(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.in_channels = 16
        # Initial-Layer (1-Kanal → 16 Kanäle)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # die vier ResNet-Blöcke mit jeweils 2 BasicBlocks
        self.layer1 = self._make_layer(16,  2, stride=1)
        self.layer2 = self._make_layer(32,  2, stride=2)
        self.layer3 = self._make_layer(64,  2, stride=2)
        self.layer4 = self._make_layer(128, 2, stride=2)

        # globales Pooling + Klassifikations-Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(128 * BasicBlock.expansion, n_classes)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        # Residual-Shortcut anpassen, falls nötig
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        # erster Block mit evtl. Downsample
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        # weitere Blocks
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 1, n_mels, time_steps]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)            # [B, C, 1, 1]
        x = torch.flatten(x, 1)        # [B, C]
        x = self.fc(x)                 # [B, n_classes]
        return x

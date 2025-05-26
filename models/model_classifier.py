import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(p=0.2)
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
        out = self.dropout(out)   # NEU: Dropout am Block-Ende
        return out

class AudioResNet18(nn.Module):
    def __init__(self, n_classes: int = 50, zero_init_residual: bool = False):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.inplanes = 64

        # Stem: 1-Kanal → 64 Kanäle
        self.conv1   = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = norm_layer(self.inplanes)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-Blöcke: jeweils 2 BasicBlocks
        self.layer1 = self._make_layer(BasicBlock,  64, blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, blocks=2, stride=2)

        # Klassifikationskopf
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout  = nn.Dropout(p=0.5)
        self.fc      = nn.Linear(512 * BasicBlock.expansion, n_classes)

        # Gewichtsinitialisierung
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # optional: Zero-Init des letzten BatchNorm in jedem BasicBlock
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None

        # Shortcut anpassen, falls sich Spatial-Size oder Kanalzahl ändert
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

        x = self.avgpool(x)          # [B, 512, 1, 1]
        x = torch.flatten(x, 1)      # [B, 512]
        x = self.dropout(x)          # [B, 512]
        x = self.fc(x)               # [B, 50]
        return x

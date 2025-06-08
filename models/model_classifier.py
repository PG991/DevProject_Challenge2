import torch
import torch.nn as nn

# A custom ResNet18-like CNN for audio spectrograms
# - Designed for single-channel (mel spectrogram) input
# - Consists of 4 stages of residual blocks
# - Uses batch normalization and dropout (p=0.25) for regularization
# - No pretrained weights, all layers are randomly initialized

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # Standard 3x3 convolution with padding
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    # 1x1 convolution for dimension matching in shortcut paths
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    # Standard ResNet basic block:
    # Two 3x3 convolutional layers with batch norm and ReLU
    # Optional downsampling if input/output dimensions differ
    # Adds input (identity) to the output (residual connection)

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
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

class AudioResNet18(nn.Module): # ResNet18-style network for single-channel spectrogram input
    def __init__(self, n_classes: int = 50, zero_init_residual: bool = False):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.inplanes = 64

        # Initial stem (Conv + BN + ReLU + MaxPool)
        self.conv1   = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = norm_layer(self.inplanes)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four stages of residual blocks
        self.layer1 = self._make_layer(BasicBlock,  64, blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, blocks=2, stride=2)

        # Classification head -> global pooling, dropout, fully connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout  = nn.Dropout(p=0.25)
        self.fc      = nn.Linear(512 * BasicBlock.expansion, n_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Optionally zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # Create a stage with several residual blocks
        # Handles shortcut connection if input and output dims differ

        norm_layer = nn.BatchNorm2d
        downsample = None

        # Downsample if input size or channels change
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
        # Input [batch, 1, n_mels, time_steps] (e.g., a mel spectrogram)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)          # Global average pooling
        x = torch.flatten(x, 1)      # Flatten to [batch, 512]
        x = self.dropout(x)          # Dropout for regularization
        x = self.fc(x)               # Final linear layer for class prediction
        return x
import torch
import torch.nn as nn
import torchaudio.transforms as T

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_p=0.3):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(drop_p)
        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(drop_p)
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu2(out)

class AudioResNet18(nn.Module):
    """ResNet-18 für 1-kanalige Mel-Spektrogramme mit Input-Normalisierung"""

    def __init__(self, n_classes, drop_p=0.3):
        super().__init__()
        self.in_planes = 64
        # GPU-fähige Mel-Pipeline
        self.mel_transform = T.MelSpectrogram(sample_rate=44100, n_fft=2048,
                                             hop_length=512, n_mels=128, power=2.0)
        self.db_transform  = T.AmplitudeToDB()
        # ResNet-Stem
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # ResNet-18 Layers
        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1, drop_p=drop_p)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, drop_p=drop_p)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, drop_p=drop_p)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, drop_p=drop_p)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_p)
        self.fc      = nn.Linear(512 * BasicBlock.expansion, n_classes)

    def _make_layer(self, block, planes, blocks, stride, drop_p):
        layers = [block(self.in_planes, planes, stride, drop_p)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, 1, drop_p))
        return nn.Sequential(*layers)

    def forward(self, wave):
        """
        wave: Tensor (B,1,Time) auf GPU
        """
        # 1) Mel-Spektrogramm und dB
        spec = self.mel_transform(wave)         # (B, n_mels, T')
        spec_db = self.db_transform(spec)       # (B, n_mels, T')
        # 2) Per-Sample Normalisierung
        mean = spec_db.mean(dim=[1,2], keepdim=True)
        std  = spec_db.std(dim=[1,2], keepdim=True)
        spec_db = (spec_db - mean) / (std + 1e-6)
        # 3) Kanal-Dimension
        x = spec_db.unsqueeze(1)                # (B,1,n_mels,T')
        # 4) ResNet-Forward
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        # 5) Klassifier
        x = self.avgpool(x); x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

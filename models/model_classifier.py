import torch
import torch.nn as nn
import torchaudio.transforms as T
from dataset.SpecAugment import SpecAugment


def conv3x3(in_planes, out_planes, stride=1):
    """3×3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """Basic Residual Block as in ResNet-18"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_p=0.2):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d(drop_p)

        self.conv2 = conv3x3(planes, planes)
        self.bn2   = nn.BatchNorm2d(planes)
        self.drop2 = nn.Dropout2d(drop_p)

        # shortcut connection
        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class AudioResNet18(nn.Module):
    """From-scratch ResNet-18 for 1-channel audio spectrograms"""
    def __init__(self, n_classes, drop_p=0.2):
        super().__init__()
        # 1) GPU-based feature extraction
        self.mel_transform = T.MelSpectrogram(
            sample_rate=44100, n_fft=2048, hop_length=512,
            n_mels=128, power=2.0
        )
        self.db_transform  = T.AmplitudeToDB()
        self.spec_augment  = SpecAugment(
            time_mask_param=30, freq_mask_param=15, num_masks=2
        )

        # 2) ResNet-18 stem
        self.in_planes = 64
        self.conv1  = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        self.pool1  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 3) Residual layers
        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1, drop_p=drop_p)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2, drop_p=drop_p)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2, drop_p=drop_p)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2, drop_p=drop_p)

        # 4) Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.drop_fc = nn.Dropout(drop_p)
        self.fc      = nn.Linear(512 * BasicBlock.expansion, n_classes)

    def _make_layer(self, block, planes, blocks, stride, drop_p):
        layers = [block(self.in_planes, planes, stride, drop_p)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, 1, drop_p))
        return nn.Sequential(*layers)

    def forward(self, wave):
        """
        wave: Tensor (B,1,Time)
        returns logits: Tensor (B, n_classes)
        """
        # remove channel dim for mel transform
        wave = wave.squeeze(1)  # -> (B, Time)
        # Mel + dB on GPU
        spec    = self.mel_transform(wave)      # -> (B,128,T')
        spec_db = self.db_transform(spec)       # -> (B,128,T')
        # add channel dim
        x = spec_db.unsqueeze(1)                # -> (B,1,128,T')
        # per-sample norm over channel,freq,time
        mean = x.mean(dim=[1,2,3], keepdim=True)
        std  = x.std(dim=[1,2,3], keepdim=True)
        x = (x - mean) / (std + 1e-6)
        # SpecAugment
        x = self.spec_augment(x)

        # ResNet forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop_fc(x)
        return self.fc(x)

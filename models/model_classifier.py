import torch
import torch.nn as nn
import torch.nn.functional as F


# class AudioMLP(nn.Module):
#     def __init__(self, n_steps, n_mels, hidden1_size, hidden2_size, output_size, time_reduce=1, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.time_reduce = time_reduce
#         # optimized for GPU, faster than x.reshape(*x.shape[:-1], -1, 2).mean(-1)
#         self.pool = nn.AvgPool1d(kernel_size=time_reduce, stride=time_reduce)  # Non-overlapping averaging

#         self.fc1 = nn.Linear(n_steps * n_mels, hidden1_size)
#         self.fc2 = nn.Linear(hidden1_size, hidden2_size)
#         self.fc3 = nn.Linear(hidden2_size, output_size)
#         self.dropout = nn.Dropout(0.3)

#     def forward(self, x):
#         # reduce time dimension
#         shape = x.shape
#         x = x.reshape(-1, 1, x.shape[-1])
#         x = self.pool(x)  # (4096, 1, 431//n)
#         x = x.reshape(shape[0], shape[1], shape[2], -1)

#         # 2D to 1D
#         x = nn.Flatten()(x)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# class SEBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super().__init__()
#         # Squeeze
#         self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
#         # Excitation
#         self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

#     def forward(self, x):
#         # x: (B, C, H, W)
#         b, c, _, _ = x.size()
#         # Global Average Pooling -> (B, C)
#         y = x.mean(dim=[2,3])
#         # Bottleneck
#         y = F.relu(self.fc1(y))
#         # Skalierungsfaktoren
#         y = torch.sigmoid(self.fc2(y))  # (B, C)
#         # Reshape und multiplizieren
#         return x * y.view(b, c, 1, 1)



class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.down = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.down = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)
    

class AudioResNet(nn.Module):
    def __init__(self, n_mels, n_steps, n_classes):
        super().__init__()
        # initial layer
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        # build 4 Stages: [64,128,256,512] channels
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        # global pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout  = nn.Dropout(0.3)
        self.fc      = nn.Linear(512, n_classes)

    def _make_layer(self, in_c, out_c, blocks, stride): #69,5
        layers = [ResidualBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c))
        return nn.Sequential(*layers)


    def forward(self, x):
        # x: (B,1,n_mels,n_steps)
        x = self.stem(x)       # -> (B,64,n_mels/2,n_steps/2)
        x = self.layer1(x)     # -> (B,64,...)
        x = self.layer2(x)     # -> (B,128,...)
        x = self.layer3(x)     # -> (B,256,...)
        x = self.layer4(x)     # -> (B,512,...)
        x = self.dropout(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)
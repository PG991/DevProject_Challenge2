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

class AudioCNN(nn.Module):
    def __init__(self, n_mels, n_steps, n_classes):
        super().__init__()
        # Input: (B, 1, n_mels, n_steps)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)   # -> (16, n_mels/2, n_steps/2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)   # -> (32, n_mels/4, n_steps/4)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)   # -> (64, n_mels/8, n_steps/8)

        # Feature-Map-Größen berechnen
        fm_h = n_mels // 8
        fm_w = n_steps // 8
        self.fc1   = nn.Linear(64 * fm_h * fm_w, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: (B, 1, n_mels, n_steps)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


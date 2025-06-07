import torch
import torchaudio.transforms as T
import random

class SpecAugment(torch.nn.Module):
    def __init__(self, time_mask_param=15, freq_mask_param=7, num_masks=1):
        super().__init__()
        # Parameters for the masking strength and number of masks
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks

    def forward(self, x):
        x = x.clone() # a copy so we don't modify the input in-place
        for _ in range(self.num_masks):
            # Apply time masking
            x = T.TimeMasking(time_mask_param=self.time_mask_param)(x)
            # Apply frequency masking
            x = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)(x)
        return x
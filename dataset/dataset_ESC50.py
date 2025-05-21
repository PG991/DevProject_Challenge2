import torch
from torch.utils import data
import torchaudio.transforms as T
from dataset.SpecAugment import SpecAugment
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import os
import sys
from functools import partial
import numpy as np
import librosa

import config
from . import transforms

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_extract_zip(url: str, file_path: str):
    #import wget
    import zipfile
    root = os.path.dirname(file_path)
    # wget.download(url, out=file_path, bar=download_progress)
    download_file(url=url, fname=file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)


# create this bar_progress method which is invoked automatically from wget
def download_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


class ESC50(data.Dataset):

    def __init__(self, root, test_folds=frozenset((1,)), subset="train", global_mean_std=(0.0, 0.0), download=False):
        audio = 'ESC-50-master/audio'
        root = os.path.normpath(root)
        audio = os.path.join(root, audio)
        if subset in {"train", "test", "val"}:
            self.subset = subset
        else:
            raise ValueError
        # path = path.split(os.sep)
        if not os.path.exists(audio) and download:
            os.makedirs(root, exist_ok=True)
            file_name = 'master.zip'
            file_path = os.path.join(root, file_name)
            url = f'https://github.com/karoldvl/ESC-50/archive/{file_name}'
            download_extract_zip(url, file_path)

        self.root = audio
        # getting name of all files inside the all the train_folds
        temp = sorted(os.listdir(self.root))
        folds = {int(v.split('-')[0]) for v in temp}
        self.test_folds = set(test_folds)
        self.train_folds = folds - test_folds
        train_files = [f for f in temp if int(f.split('-')[0]) in self.train_folds]
        test_files = [f for f in temp if int(f.split('-')[0]) in test_folds]
        # sanity check
        assert set(temp) == (set(train_files) | set(test_files))
        if subset == "test":
            self.file_names = test_files
        else:
            if config.val_size:
                train_files, val_files = train_test_split(train_files, test_size=config.val_size, random_state=0)
            if subset == "train":
                self.file_names = train_files
            else:
                self.file_names = val_files
        # the number of samples in the wave (=length) required for spectrogram
        out_len = int(((config.sr * 5) // config.hop_length) * config.hop_length)
        train = self.subset == "train"
        
        if train:
            # augment training data with transformations that include randomness
            # transforms can be applied on wave and spectral representation
            self.wave_transforms = transforms.Compose(
                torch.Tensor,
                transforms.RandomScale(max_scale=1.25),
                transforms.RandomPadding(out_len=out_len),
                transforms.RandomCrop(out_len=out_len)
            )

            self.spec_transforms = transforms.Compose(
                # to Tensor and prepend singleton dim
                #lambda x: torch.Tensor(x).unsqueeze(0),
                # lambda non-pickleable, problem on windows, replace with partial function
                torch.Tensor,
                partial(torch.unsqueeze, dim=0),
            )

        else:
            # for testing transforms are applied deterministically to support reproducible scores
            self.wave_transforms = transforms.Compose(
                torch.Tensor,
                # disable randomness
                transforms.RandomPadding(out_len=out_len, train=False),
                transforms.RandomCrop(out_len=out_len, train=False)
            )
            self.spec_transforms = transforms.Compose(
                torch.Tensor,
                partial(torch.unsqueeze, dim=0),
            )
        self.global_mean = global_mean_std[0]
        self.global_std = global_mean_std[1]
        self.n_mfcc = config.n_mfcc if hasattr(config, "n_mfcc") else None

######################################################################################################
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sr,
            n_fft=2048,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,        # power=2.0 für Energie, entspricht librosa
        )
        self.db_transform  = T.AmplitudeToDB()
######################################################################################################
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        # 1) Dateiname und Pfad
        file_name = self.file_names[index]
        path = os.path.join(self.root, file_name)

        # 2) Roh-Audio laden
        wave, rate = librosa.load(path, sr=config.sr)  # wave: np.ndarray

        # 3) Label extrahieren
        class_id = int(file_name.split('-')[-1].split('.')[0])

        # 4) Dimensionalität sicherstellen: (1, Time)
        if wave.ndim == 1:
            wave = wave[np.newaxis, :]  # → np.ndarray (1, Time)

        # 5) Normierung
        if np.abs(wave).max() > 1.0:
            wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
        wave = wave * 32768.0

        # 6) Stille abschneiden
        nz = wave.nonzero()[1]
        start, end = nz.min(), nz.max()
        wave = wave[:, start : end+1]

        # 7) Augmentierung auf NumPy-Ebene, liefert bereits torch.Tensor
        wave_tensor = self.wave_transforms(wave)  # → torch.Tensor (1, Time)

        # 8) Sicherstellen, dass es FloatTensor ist
        wave_tensor = wave_tensor.float()

        return file_name, wave_tensor, class_id

def get_global_stats(data_path):
    res = []
    for i in range(1, 6):
        train_set = ESC50(subset="train", test_folds={i}, root=data_path, download=True)
        a = torch.concatenate([v[1] for v in tqdm(train_set)])
        res.append((a.mean(), a.std()))
    return np.array(res)

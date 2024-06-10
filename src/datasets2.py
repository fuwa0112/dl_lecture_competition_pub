import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
import numpy as np
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, sample_rate, new_sample_rate, low_cut, high_cut, baseline_window):
    # リサンプリング
    num_samples = int(len(data) * float(new_sample_rate) / sample_rate)
    data = signal.resample(data, num_samples)
    
    # フィルタリング
    nyquist = 0.5 * new_sample_rate
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = signal.butter(1, [low, high], btype="band")
    data = signal.filtfilt(b, a, data)
    
    # ベースライン補正
    baseline = np.mean(data[baseline_window[0]:baseline_window[1]], axis=0)
    data = data - baseline
    
    return data

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", new_sample_rate: int = 128, low_cut: float = 0.5, high_cut: float = 40.0, baseline_window: Tuple[int, int] = (0, 50)) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        print(f"Loading {split}_X.pt...")
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        print(f"{split}_X.pt loaded successfully.")

        print(f"Loading {split}_subject_idxs.pt...")
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        print(f"{split}_subject_idxs.pt loaded successfully.")
        
        if split in ["train", "val"]:
            print(f"Loading {split}_y.pt...")
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
            print(f"{split}_y.pt loaded successfully.")
        
        # 前処理
        # sample_rate = 1200  # 元のサンプリングレート（仮定）
        # self.X = np.array([preprocess_data(x, sample_rate, new_sample_rate, low_cut, high_cut, baseline_window) for x in self.X])
        
        # スケーリング
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X.reshape(-1, self.X.shape[-1])).reshape(self.X.shape)

        # Tensorに変換
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.subject_idxs = torch.tensor(self.subject_idxs, dtype=torch.long)
        if hasattr(self, 'y'):
            self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from scipy.signal import butter, lfilter

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
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

        # Apply preprocessing
        self.X = self.preprocess(self.X)

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
    
    def preprocess(self, X: torch.Tensor) -> torch.Tensor:
        # Normalize data
        X = (X - X.mean(dim=-1, keepdim=True)) / X.std(dim=-1, keepdim=True)
        
        # Apply bandpass filter
        X = self.apply_bandpass_filter(X, lowcut=0.1, highcut=30, fs=1000, order=5)
        
        return X

    def apply_bandpass_filter(self, X: torch.Tensor, lowcut: float, highcut: float, fs: float, order: int) -> torch.Tensor:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        
        def butter_bandpass_filter(data):
            return lfilter(b, a, data)

        X_np = X.numpy()
        X_filtered = np.apply_along_axis(butter_bandpass_filter, -1, X_np)
        return torch.from_numpy(X_filtered)

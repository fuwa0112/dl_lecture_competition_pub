import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class BasicConvClassifier3(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        num_blocks: int = 4,
        kernel_size: int = 5,
        num_subjects: int = 4,  # 被験者数を指定
        subject_emb_dim: int = 32  # 被験者の埋め込み次元を指定
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(*[
            ConvBlock(in_channels if i == 0 else hid_dim, hid_dim, kernel_size=kernel_size)
            for i in range(num_blocks)
        ])

        self.subject_embedding = nn.Embedding(num_subjects, subject_emb_dim)
        
        self.batchnorm = nn.BatchNorm1d(hid_dim)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim + subject_emb_dim, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        X = self.batchnorm(X)  # Apply batch normalization
        
        subject_emb = self.subject_embedding(subject_idxs)
        subject_emb = subject_emb.unsqueeze(-1).expand(-1, -1, X.shape[-1])  # Expand embeddings to match X's dimensions
        X = torch.cat([X, subject_emb], dim=1)  # Concatenate along the channel dimension
        
        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.2,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)

# Usage example
model = BasicConvClassifier3(num_classes=10, seq_len=100, in_channels=64, hid_dim=256, num_blocks=6, kernel_size=5)

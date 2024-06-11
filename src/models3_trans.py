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
        num_subjects: int = 4,
        subject_emb_dim: int = 32,
        nhead: int = 8,
        num_layers: int = 1,
        ff_dim: int = 512,
        dropout_prob: float = 0.5,
        weight_decay: float = 1e-5
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(*[
            ConvBlock(in_channels if i == 0 else hid_dim, hid_dim, kernel_size=kernel_size, p_drop=dropout_prob)
            for i in range(num_blocks)
        ])

        self.subject_embedding = nn.Embedding(num_subjects, subject_emb_dim)
        
        self.batchnorm = nn.BatchNorm1d(hid_dim)
        
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, hid_dim))
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim, 
            nhead=nhead, 
            dim_feedforward=ff_dim, 
            dropout=dropout_prob
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hid_dim + subject_emb_dim),
            nn.Linear(hid_dim + subject_emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        X = self.batchnorm(X)
        
        X = X.permute(0, 2, 1)  # Change shape from (b, d, t) to (b, t, d)
        b, t, _ = X.shape
        X += self.position_embedding[:, :t, :]
        
        X = X.permute(1, 0, 2)  # Transformer expects shape (t, b, d)
        X = self.transformer_encoder(X)
        X = X.permute(1, 0, 2)  # Back to shape (b, t, d)
        
        X = X.mean(dim=1)  # Global average pooling
        
        subject_emb = self.subject_embedding(subject_idxs)  # Shape: (b, subject_emb_dim)
        X = torch.cat([X, subject_emb], dim=-1)  # Concatenate along the feature dimension
        
        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.5,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.layernorm0 = nn.LayerNorm(out_dim)
        self.layernorm1 = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))
        X = self.layernorm0(X.permute(0, 2, 1)).permute(0, 2, 1)

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))
        X = self.layernorm1(X.permute(0, 2, 1)).permute(0, 2, 1)

        return self.dropout(X)

# Usage example
model = BasicConvClassifier3(num_classes=10, seq_len=100, in_channels=64, hid_dim=128, num_blocks=4, kernel_size=5, dropout_prob=0.5)

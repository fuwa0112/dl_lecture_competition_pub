import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class BasicConvClassifier6(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        num_subjects: int = 4,
        subject_emb_dim: int = 32
    ) -> None:
        super().__init__()

        self.embedding = nn.Linear(in_channels, hid_dim)
        
        self.subject_embedding = nn.Embedding(num_subjects, subject_emb_dim)
        
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, hid_dim))
        
        transformer_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.LayerNorm(hid_dim + subject_emb_dim),
            nn.Linear(hid_dim + subject_emb_dim, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        X = X.transpose(1, 2)  # Change shape from (b, c, t) to (b, t, c)
        X = self.embedding(X)  # Shape: (b, t, hid_dim)
        b, t, _ = X.shape
        
        X += self.position_embedding[:, :t, :]
        
        X = X.permute(1, 0, 2)  # Transformer expects shape (t, b, hid_dim)
        X = self.transformer_encoder(X)
        X = X.permute(1, 0, 2)  # Back to shape (b, t, hid_dim)
        
        X = X.mean(dim=1)  # Global average pooling
        
        subject_emb = self.subject_embedding(subject_idxs)  # Shape: (b, subject_emb_dim)
        X = torch.cat([X, subject_emb], dim=-1)  # Concatenate along the feature dimension
        
        return self.head(X)

# Usage example
model = BasicConvClassifier6(num_classes=10, seq_len=100, in_channels=64, hid_dim=256, num_heads=8, num_layers=6)

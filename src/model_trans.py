class LSTMConvClassifier(nn.Module):
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
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 2,
        transformer_layers: int = 2,
        num_heads: int = 8,
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
        self.layernorm = nn.LayerNorm(hid_dim + subject_emb_dim)

        self.lstm = nn.LSTM(
            input_size=hid_dim + subject_emb_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_prob
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=lstm_hidden_dim,
                nhead=num_heads,
                dropout=dropout_prob
            ),
            num_layers=transformer_layers
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(lstm_hidden_dim, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        X = self.blocks(X)
        X = self.batchnorm(X)
        
        subject_emb = self.subject_embedding(subject_idxs)
        subject_emb = subject_emb.unsqueeze(-1).expand(-1, -1, X.shape[-1])
        X = torch.cat([X, subject_emb], dim=1)
        
        X = self.layernorm(X.permute(0, 2, 1)).permute(0, 2, 1)
        X, _ = self.lstm(X.permute(0, 2, 1))
        X = self.transformer(X.permute(1, 0, 2)).permute(1, 0, 2)
        X = self.dropout(X)
        return self.head(X.permute(0, 2, 1))

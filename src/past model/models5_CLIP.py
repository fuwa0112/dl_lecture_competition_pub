import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from PIL import Image
import os
from transformers import CLIPModel, CLIPProcessor

class BasicConvClassifier4(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128,
        lstm_dim: int = 256,
        num_blocks: int = 4,
        kernel_size: int = 5,
        num_subjects: int = 4,  # 被験者数を指定
        subject_emb_dim: int = 32  # 被験者の埋め込み次元を指定
    ) -> None:
        super().__init__()

        # CLIPモデルのロード
        self.clip_model = CLIPModel.from_pretrained("data/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("data/clip-vit-base-patch32")

        self.blocks = nn.Sequential(*[
            ConvBlock(in_channels if i == 0 else hid_dim, hid_dim, kernel_size=kernel_size)
            for i in range(num_blocks)
        ])

        self.lstm = nn.LSTM(input_size=hid_dim, hidden_size=lstm_dim, batch_first=True, bidirectional=True)

        self.subject_embedding = nn.Embedding(num_subjects, subject_emb_dim)
        
        self.batchnorm = nn.BatchNorm1d(lstm_dim * 2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(lstm_dim * 2 + subject_emb_dim, num_classes),
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        # CLIPの画像エンコーダを使用して脳波データの特徴を抽出
        X = self.clip_model.vision_model(pixel_values=X).last_hidden_state

        X = self.blocks(X)
        X = X.transpose(1, 2)  # Change shape from (b, c, t) to (b, t, c) for LSTM
        X, _ = self.lstm(X)
        X = X.transpose(1, 2)  # Change shape back from (b, t, c) to (b, c, t) for pooling
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
        p_drop: float = 0.1,
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

def extract_clip_features(image_folder):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    features = []
    for img_file in os.listdir(image_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(os.path.join(image_folder, img_file))
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.get_image_features(**inputs)
            features.append(outputs)
    return torch.cat(features)

# Usage example
image_features = extract_clip_features("/path/to/Image/folder")
model = BasicConvClassifier4(num_classes=10, seq_len=100, in_channels=image_features.shape[1], hid_dim=256, lstm_dim=512, num_blocks=6, kernel_size=5)

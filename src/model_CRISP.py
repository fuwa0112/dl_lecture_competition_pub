import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class MEGPretrainedEncoder(nn.Module):
    def __init__(self, vision_encoder, in_channels, hidden_dim):
        super(MEGPretrainedEncoder, self).__init__()
        self.vision_encoder = vision_encoder
        self.conv1d = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1d(x)
        x = self.vision_encoder(x)
        return x

class MEGClassifier(nn.Module):
    def __init__(self, pretrained_encoder, num_classes):
        super(MEGClassifier, self).__init__()
        self.pretrained_encoder = pretrained_encoder
        self.fc = nn.Linear(768, num_classes)  # CLIPモデルの出力次元に合わせる
    
    def forward(self, x, subject_idxs):
        x = self.pretrained_encoder(x)
        x = x.mean(dim=1)  # Global Average Pooling
        x = self.fc(x)
        return x
    

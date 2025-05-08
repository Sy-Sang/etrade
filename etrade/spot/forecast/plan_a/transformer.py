#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""transformer预测器"""

__author__ = "Sy,Sang"
__version__ = ""
__license__ = "GPLv3"
__maintainer__ = "Sy, Sang"
__email__ = "martin9le@163.com"
__status__ = "Development"
__credits__ = []
__date__ = ""
__copyright__ = ""

# 系统模块
import copy
import pickle
import json
from typing import Union, Self
from collections import namedtuple

# 项目模块

# 外部模块
import numpy
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 代码块

class MarketSampleDataset(Dataset):
    def __init__(self, data_array):
        # data_array: numpy array, (n_samples, 64)
        self.data = torch.tensor(data_array, dtype=torch.float32)
        self.data_len, self.data_width = data_array.shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # x = self.data[idx, 3:63]  # 取PPF部分(60维)
        # crps = self.data[idx, 0:3]  # 取CRPS部分(3维)
        x_full = self.data[idx, :self.data_width - 1]  # 合起来63维
        y = self.data[idx, self.data_width - 1]  # zero_quantile
        return x_full, y


class MarketSampleTransformer(nn.Module):
    def __init__(self, features_num=63, emb_dim=32, nhead=4, nlayers=2):
        super().__init__()
        self.embedding = nn.Linear(features_num, emb_dim)  # 60维输入, embed到更高维
        # self.norm = nn.LayerNorm(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Linear(emb_dim, 1)  # 输出 zero_quantile

    def forward(self, x):
        # x: (batch_size, 60)
        x = self.embedding(x)  # (batch_size, emb_dim)
        # x = self.norm(x)
        x = x.unsqueeze(1)  # (batch_size, seq_len=1, emb_dim)
        x = self.transformer_encoder(x)  # (batch_size, seq_len=1, emb_dim)
        x = x.mean(dim=1)  # mean pooling (seq_len维度)
        out = self.head(x).squeeze(-1)  # (batch_size, 1)
        return out


class TabTransformer(nn.Module):
    def __init__(self, num_numeric, emb_dim=32, num_heads=4, num_layers=2, dropout=0.1):
        super(TabTransformer, self).__init__()

        # 单独投影每个 numeric 特征
        self.numeric_proj = nn.ModuleList([
            nn.Linear(1, emb_dim) for _ in range(num_numeric)
        ])

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出 MLP
        self.output = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # 回归预测
        )

    def forward(self, numeric_data):
        # numeric_data: (batch_size, num_numeric)
        tokens = []
        for i, proj in enumerate(self.numeric_proj):
            feature_i = numeric_data[:, i].unsqueeze(1)  # (batch_size, 1)
            token_i = proj(feature_i)  # (batch_size, emb_dim)
            tokens.append(token_i)

        tokens = torch.stack(tokens, dim=1)  # (batch_size, num_numeric, emb_dim)
        tokens = tokens.permute(1, 0, 2)  # (seq_len, batch_size, emb_dim)

        encoded = self.transformer(tokens)  # (seq_len, batch_size, emb_dim)
        pooled = encoded.mean(dim=0)  # (batch_size, emb_dim)

        out = self.output(pooled).squeeze(-1)  # (batch_size)
        return out


if __name__ == "__main__":
    pass

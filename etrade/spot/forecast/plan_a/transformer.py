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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx, 3:63]  # 取PPF部分(60维)
        crps = self.data[idx, 0:3]  # 取CRPS部分(3维)
        x_full = torch.cat([crps, x], dim=0)  # 合起来63维
        y = self.data[idx, 63]  # zero_quantile
        return x_full, y

class MarketSampleTransformer(nn.Module):
    def __init__(self, emb_dim=32, nhead=4, nlayers=2):
        super().__init__()
        self.embedding = nn.Linear(60, emb_dim)  # 60维输入, embed到更高维
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Linear(emb_dim, 1)  # 输出 zero_quantile

    def forward(self, x):
        # x: (batch_size, 60)
        x = self.embedding(x)  # (batch_size, emb_dim)
        x = x.unsqueeze(1)  # (batch_size, seq_len=1, emb_dim)
        x = self.transformer_encoder(x)  # (batch_size, seq_len=1, emb_dim)
        x = x.mean(dim=1)  # mean pooling (seq_len维度)
        out = self.head(x)  # (batch_size, 1)
        return out


if __name__ == "__main__":
    pass

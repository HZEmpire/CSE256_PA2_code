# Author: Haozhou Xu
# PID: A69032157
# Date: 2024.10.25

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):
    def __init__(self, embed_size, head_num):
        super(Attention, self).__init__()
        assert embed_size % head_num == 0
        self.embed_size = embed_size
        self.head_num = head_num
        self.head_dim = embed_size // head_num

        self.attention = nn.Linear(embed_size, embed_size, bias=False)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        N, seq_len, num_features = x.shape
        q = self.attention(x).reshape(N, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        k = self.attention(x).reshape(N, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        v = self.attention(x).reshape(N, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)

        # Calculate
        score = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        weight = F.softmax(score, dim=-1)
        # Perhaps we can add dropout here
        # weight = self.dropout(weight)
        out = torch.matmul(weight, v).permute(0, 2, 1, 3).reshape(N, seq_len, num_features)
        out = self.fc(out)

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(FNN, self).__init__()
        assert num_layers >= 1
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Sequential(
                nn.ReLU(nn.Linear(hidden_size, hidden_size)),
                nn.Dropout(dropout)
            ) for i in range(num_layers - 1)])
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, head_num, hidden_size, num_layers=1, dropout=0):
        super(TransformerEncoder, self).__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList(
            [nn.Sequential(
                Attention(embed_size, head_num),
                FNN(embed_size, hidden_size, embed_size, num_layers=1, dropout=dropout)
            ) for i in range(num_layers)])
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
    def encode(self, x, mask=None):
        return self.forward(x, mask)
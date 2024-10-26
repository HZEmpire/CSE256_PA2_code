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

        # Separate Q, K, V layers for each attention head
        self.q = nn.Linear(embed_size, embed_size, bias=False)
        self.k = nn.Linear(embed_size, embed_size, bias=False)
        self.v = nn.Linear(embed_size, embed_size, bias=False)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        N, seq_len, num_features = x.shape

        # Split the embedding into heads
        q = self.q(x).reshape(N, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(N, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(N, seq_len, self.head_num, self.head_dim).permute(0, 2, 1, 3)

        # Calculate
        score = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            score = score.masked_fill(mask == 0, float('-inf'))
        weight = F.softmax(score, dim=-1)
        # Perhaps we can add dropout here
        # weight = self.dropout(weight)
        out = torch.matmul(weight, v).permute(0, 2, 1, 3).reshape(N, seq_len, self.embed_size)
        out = self.fc(out)
        return out


class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(FNN, self).__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_layers - 1)])
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc2(x)
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
            x = layer[0](x, mask) + x
            x = layer[1](x) + x
        return self.norm(x)
    
    def encode(self, x, mask=None):
        x = self.forward(x, mask)
        x = x.mean(dim=1)
        return x

class FNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, head_num=1, num_layers=1, dropout=0):
        super(FNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer_encoder = TransformerEncoder(embed_size, head_num, hidden_size, num_layers, dropout)
        self.fnn = FNN(embed_size, hidden_size, output_size, num_layers, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.transformer_encoder.encode(x, mask)
        x = self.fnn(x)
        x = self.softmax(x)
        return x
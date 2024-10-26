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

class TransformerOneLayer(nn.Module):
    def __init__(self, embed_size, head_num, hidden_size, dropout=0):
        super(TransformerOneLayer, self).__init__()
        self.attention = Attention(embed_size, head_num)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.fnn = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, embed_size)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_size)
    
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_out = self.attention(x, mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)  # Add & Norm
        # Feedforward
        ff_out = self.fnn(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)  # Add & Norm
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, head_num, hidden_size, num_layers=1, dropout=0):
        super(TransformerEncoder, self).__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList(
            [TransformerOneLayer(embed_size, head_num, hidden_size, dropout)
            for i in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
    def encode(self, x, mask=None):
        x = self.forward(x, mask)
        x = x.mean(dim=1)
        return x

class FNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, max_seq_len, head_num=1, num_layers=1, dropout=0):
        super(FNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_size)
        self.transformer_encoder = TransformerEncoder(embed_size, head_num, hidden_size, num_layers, dropout)
        self.fnn = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer_encoder.encode(x, mask)
        x = self.fnn(x)
        x = self.softmax(x)
        return x
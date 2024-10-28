# Author: Haozhou Xu
# PID: A69032157
# Date: 2024.10.25

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """ Multi-head self-attention module """
    def __init__(self, n_embd, n_head):
        super(Attention, self).__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Separate Q, K, V layers for each attention head
        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)
        self.fc = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask=None):
        N, seq_len, num_features = x.size()

        # Split the embedding into heads
        q = self.q(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / (self.head_dim ** 0.5)  # (N, n_head, seq_len, seq_len)
        if mask is not None:
            # 调整掩码形状以匹配 scores
            if mask.dim() == 2:
                # 掩码形状为 (batch_size, seq_len)，扩展为 (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # 掩码形状为 (batch_size, 1, seq_len)，扩展为 (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # 掩码形状为 (batch_size, 1, seq_len, seq_len) 或 (1, 1, seq_len, seq_len)
                pass
            else:
                raise ValueError("Invalid mask shape")

            # 将掩码转换为与 scores 相同的设备
            mask = mask.to(scores.device)
            # 掩码为 True 的位置表示需要屏蔽
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(N, seq_len, self.n_embd)
        out = self.fc(out)

        return out, weights  # Return output and attention probabilities


class TransformerEncoderLayer(nn.Module):
    """ A single Transformer encoder layer """
    def __init__(self, n_embd, n_head):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = Attention(n_embd, n_head)
        self.norm1 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        out, weights = self.self_attn(x, mask)
        x = x + out  # Residual connection
        x = self.norm1(x)  # Layer normalization

        ffn_out= self.ffn(x)
        x = x + ffn_out  # Residual connection
        x = self.norm2(x)  # Layer normalization

        return x, weights  # Return output and attention probabilities


class TransformerEncoder(nn.Module):
    """ Transformer Encoder consisting of multiple encoder layers """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_seq_len):
        super(TransformerEncoder, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)  # Token embeddings
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)  # Positional embeddings
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(n_embd, n_head) for i in range(n_layer)
        ])
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)  # Add token and positional embeddings

        weights_list = []  # To store attention probabilities from each layer
        for layer in self.layers:
            x, attn_probs = layer(x, mask)  # Process through each layer
            weights_list.append(attn_probs)

        x = self.norm(x)  # Final layer normalization
        return x, weights_list  # Return output embeddings and attention probabilities


class FNNClassifier(nn.Module):
    """ Feedforward Neural Network Classifier """
    def __init__(self, n_input, n_hidden, n_output):
        super(FNNClassifier, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Output logits
    
class TransformerDecoderLayer(nn.Module):
    """A single Transformer decoder layer with masked self-attention."""
    def __init__(self, n_embd, n_head):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = Attention(n_embd, n_head)
        self.norm1 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 100),  # Feedforward hidden size is 100
            nn.ReLU(),
            nn.Linear(100, n_embd)
        )
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        # Masked self-attention
        out, weights = self.self_attn(x, mask=mask)
        x = x + out  # Residual connection
        x = self.norm1(x)  # Layer normalization

        # Feedforward network
        ffn_out = self.ffn(x)
        x = x + ffn_out  # Residual connection
        x = self.norm2(x)  # Layer normalization

        return x, weights  # Return output and attention probabilities

class TransformerDecoder(nn.Module):
    """Transformer Decoder consisting of multiple decoder layers."""
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_seq_len):
        super(TransformerDecoder, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)  # Token embeddings
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)  # Positional embeddings
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(n_embd, n_head) for i in range(n_layer)
        ])
        self.norm = nn.LayerNorm(n_embd)
        self.fc_out = nn.Linear(n_embd, vocab_size)  # Output layer to predict next token

    def forward(self, x):
        N, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(N, seq_len)
        x = self.token_emb(x) + self.pos_emb(positions)  # Add token and positional embeddings

        # Mask future tokens
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)

        weights_list = []  # To store attention probabilities from each layer
        for layer in self.layers:
            x, attn_weights = layer(x, mask=attn_mask)
            weights_list.append(attn_weights)

        x = self.norm(x)  # Final layer normalization
        x = self.fc_out(x)  # Output logits for each token position

        return x, weights_list  # Return logits and attention weights



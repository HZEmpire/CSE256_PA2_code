# Author: Haozhou Xu
# PID: A69032157
# Date: 2024.10.25

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    """Standard Multi-head self-attention module."""
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
            # Adjust mask shape to match scores
            if mask.dim() == 2:
                # Mask shape: (batch_size, seq_len), expand to (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Mask shape: (batch_size, 1, seq_len), expand to (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # Mask shape: (batch_size, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
                pass
            else:
                raise ValueError("Invalid mask shape")

            # Convert mask to the same device as scores
            mask = mask.to(scores.device)
            # Mask positions with True; these positions will be set to -inf
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(N, seq_len, self.n_embd)
        out = self.fc(out)

        return out, weights  # Return output and attention probabilities

# New ALiBi Attention Class
class AliBiAttention(nn.Module):
    """Multi-head self-attention with ALiBi positional bias."""
    def __init__(self, n_embd, n_head):
        super(AliBiAttention, self).__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Q, K, V layers
        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)
        self.fc = nn.Linear(n_embd, n_embd)

        # Precompute ALiBi slopes using geometric sequence
        self.register_buffer('alibi_slopes', self.get_alibi_slopes(n_head))

    def get_alibi_slopes(self, n_heads):
        # Generate geometric sequence of slopes
        slopes = [2 ** (-i) for i in range(1, n_heads + 1)]
        return torch.tensor(slopes).unsqueeze(-1).unsqueeze(-1)  # Shape: (n_heads, 1, 1)

    def get_alibi_bias(self, seq_len, device):
        # Calculate ALiBi bias for a given sequence length
        context_position = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        memory_position = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(1)
        relative_position = memory_position - context_position  # Shape: (seq_len, seq_len)
        alibi = self.alibi_slopes * relative_position.float()  # Shape: (n_heads, seq_len, seq_len)
        alibi = alibi.unsqueeze(0)  # Shape: (1, n_heads, seq_len, seq_len)
        return -alibi

    def forward(self, x, mask=None):
        N, seq_len, num_features = x.size()

        # Split embeddings into heads
        q = self.q(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)  # (N, n_head, seq_len, head_dim)
        k = self.k(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)  # (N, n_head, seq_len, seq_len)

        # Add ALiBi bias
        alibi = self.get_alibi_bias(seq_len, scores.device)  # Shape: (1, n_heads, seq_len, seq_len)
        scores = scores + alibi

        if mask is not None:
            # Adjust mask shape to match scores
            if mask.dim() == 2:
                # Mask shape: (batch_size, seq_len), expand to (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Mask shape: (batch_size, 1, seq_len), expand to (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # Mask shape: (batch_size, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
                pass
            else:
                raise ValueError("Invalid mask shape")

            # Convert mask to the same device as scores
            mask = mask.to(scores.device)
            # Mask positions with True; these positions will be set to -inf
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # (N, n_head, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(N, seq_len, self.n_embd)
        out = self.fc(out)

        return out, weights

# New Disentangled Attention Class
class DisentangledAttention(nn.Module):
    """Disentangled Attention as in DeBERTa."""
    def __init__(self, n_embd, n_head, max_seq_len=512):
        super(DisentangledAttention, self).__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.max_seq_len = max_seq_len

        # Content and position embeddings
        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)

        self.q_pos = nn.Linear(n_embd, n_embd)
        self.k_pos = nn.Linear(n_embd, n_embd)

        self.pos_emb = nn.Embedding(2 * max_seq_len - 1, n_embd)

        self.fc = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask=None):
        N, seq_len, num_features = x.size()

        # Content projections
        q = self.q(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0,2,1,3)  # (N, n_head, seq_len, head_dim)
        k = self.k(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0,2,1,3)
        v = self.v(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0,2,1,3)

        # Position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        rel_position_ids = position_ids[None, :] - position_ids[:, None] + self.max_seq_len - 1  # Shift to positive
        pos_emb = self.pos_emb(rel_position_ids).to(x.device)  # Shape: (seq_len, seq_len, n_embd)

        # Position projections
        q_pos = self.q_pos(pos_emb).reshape(seq_len, seq_len, self.n_head, self.head_dim).permute(2, 0, 1, 3)  # (n_head, seq_len, seq_len, head_dim)
        k_pos = self.k_pos(pos_emb).reshape(seq_len, seq_len, self.n_head, self.head_dim).permute(2, 0, 1, 3)

        # Compute attention scores
        # Content-to-content
        ac = torch.einsum('bhid,bhjd->bhij', q, k)  # (N, n_head, seq_len, seq_len)

        # Content-to-position
        bd = torch.einsum('bhid,hijd->bhij', q, k_pos)  # (N, n_head, seq_len, seq_len)

        # Position-to-content
        bc = torch.einsum('bhjd,hijd->bhij', k, q_pos)  # (N, n_head, seq_len, seq_len)

        # Combine attention scores
        scores = (ac + bd + bc) / (self.head_dim ** 0.5)

        if mask is not None:
            # Adjust mask shape
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                pass
            else:
                raise ValueError("Invalid mask shape")

            mask = mask.to(scores.device)
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # (N, n_head, seq_len, head_dim)
        out = out.transpose(1,2).contiguous().view(N, seq_len, -1)
        out = self.fc(out)

        return out, weights
    
# New Local Window Attention Class
class LocalAttention(nn.Module):
    """Multi-head self-attention with local window attention."""
    def __init__(self, n_embd, n_head, window_size):
        super(LocalAttention, self).__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.window_size = window_size

        # Q, K, V layers
        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)
        self.fc = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask=None):
        N, seq_len, num_features = x.size()

        # Split embeddings into heads
        q = self.q(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)  # (N, n_head, seq_len, head_dim)
        k = self.k(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(self.head_dim)  # (N, n_head, seq_len, seq_len)

        # Create local mask where positions outside the window are masked
        # Mask shape: (1, 1, seq_len, seq_len)
        positions = torch.arange(seq_len, device=x.device, dtype=torch.int32) ######## Must be int32 otherwise mps will fail
        distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))  # (seq_len, seq_len)
        local_mask = distance_matrix > self.window_size  # (seq_len, seq_len)
        local_mask = local_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
        scores = scores.masked_fill(local_mask, float('-inf'))

        if mask is not None:
            # Adjust mask shape to match scores
            if mask.dim() == 2:
                # Mask shape: (batch_size, seq_len), expand to (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Mask shape: (batch_size, 1, seq_len), expand to (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # Mask shape: (batch_size, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
                pass
            else:
                raise ValueError("Invalid mask shape")

            mask = mask.to(scores.device)
            scores = scores.masked_fill(mask, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # (N, n_head, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(N, seq_len, self.n_embd)
        out = self.fc(out)

        return out, weights  # Return output and attention scores (for consistency)


# New Blockwise Attention Class
class BlockwiseAttention(nn.Module):
    """Multi-head self-attention with blockwise attention."""
    def __init__(self, n_embd, n_head, block_size):
        super(BlockwiseAttention, self).__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size

        # Q, K, V layers
        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)
        self.fc = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask=None):
        N, seq_len, _ = x.size()

        # Calculate for each block，use torch.div to avoid floordiv warning
        block_indices = torch.div(torch.arange(seq_len, device=x.device), self.block_size, rounding_mode='floor')  # [seq_len]

        # Create block mask to mask attention between different blocks
        block_mask = (block_indices.unsqueeze(0) != block_indices.unsqueeze(1))  # [seq_len, seq_len]

        # Expand block mask to match scores shape
        block_mask = block_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
        print(block_mask[0][0])
        # Split embeddings into heads
        q = self.q(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)  # [N, n_head, seq_len, head_dim]
        k = self.k(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(N, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [N, n_head, seq_len, seq_len]
        scores = scores.masked_fill(block_mask, float('-inf'))

        if mask is not None:
            # Adjust mask shape to match scores
            if mask.dim() == 2:
                # Mask shape: (batch_size, seq_len), expand to (batch_size, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Mask shape: (batch_size, 1, seq_len), expand to (batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                # Mask shape: (batch_size, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
                pass
            else:
                raise ValueError("Invalid mask shape")

            mask = mask.to(scores.device)
            scores = scores.masked_fill(mask, float('-inf'))

        # Compute attention weights
        weights = F.softmax(scores, dim=-1)  # [N, n_head, seq_len, seq_len]
        out = torch.matmul(weights, v)  # [N, n_head, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(N, seq_len, self.n_embd)  # [N, seq_len, n_embd]
        out = self.fc(out)  # [N, seq_len, n_embd]

        return out, weights  # Return output and attention scores


# Modify TransformerEncoderLayer and TransformerDecoderLayer to include the new attention methods
class TransformerEncoderLayer(nn.Module):
    """A single Transformer encoder layer with selectable attention method."""
    def __init__(self, n_embd, n_head, method='standard', max_seq_len=512, window_size=4, block_size=8):
        super(TransformerEncoderLayer, self).__init__()
        if method == 'standard':
            self.self_attn = Attention(n_embd, n_head)
        elif method == 'alibi':
            self.self_attn = AliBiAttention(n_embd, n_head)
        elif method == 'disentangled':
            self.self_attn = DisentangledAttention(n_embd, n_head, max_seq_len)
        elif method == 'local':
            self.self_attn = LocalAttention(n_embd, n_head, window_size)
        elif method == 'block':
            self.self_attn = BlockwiseAttention(n_embd, n_head, block_size)
        else:
            raise ValueError("Invalid method. Choose 'standard', 'alibi', 'disentangled', 'local', or 'block'.")

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

        ffn_out = self.ffn(x)
        x = x + ffn_out  # Residual connection
        x = self.norm2(x)  # Layer normalization

        return x, weights  # Return output and attention probabilities

class TransformerEncoder(nn.Module):
    """Transformer Encoder consisting of multiple encoder layers."""
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_seq_len, method='standard', window_size=4, block_size=8):
        super(TransformerEncoder, self).__init__()
        self.method = method
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, n_embd)  # Token embeddings

        if method in ['standard', 'disentangled', 'local', 'block']:
            self.pos_emb = nn.Embedding(max_seq_len, n_embd)  # Positional embeddings
        # No positional embeddings for AliBi in the embedding layer

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(n_embd, n_head, method, max_seq_len, window_size, block_size) for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        N, seq_len = x.size()
        x = self.token_emb(x)
        if self.method in ['standard', 'disentangled', 'local', 'block']:
            positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
            x = x + self.pos_emb(positions)  # Add positional embeddings

        weights_list = []
        for layer in self.layers:
            x, attn_probs = layer(x, mask)
            weights_list.append(attn_probs)

        x = self.norm(x)
        return x, weights_list  # Return output embeddings and attention probabilities

class TransformerDecoderLayer(nn.Module):
    """A single Transformer decoder layer with selectable attention method."""
    def __init__(self, n_embd, n_head, method='standard', max_seq_len=512, window_size=4, block_size=8):
        super(TransformerDecoderLayer, self).__init__()
        if method == 'standard':
            self.self_attn = Attention(n_embd, n_head)
        elif method == 'alibi':
            self.self_attn = AliBiAttention(n_embd, n_head)
        elif method == 'disentangled':
            self.self_attn = DisentangledAttention(n_embd, n_head, max_seq_len)
        elif method == 'local':
            self.self_attn = LocalAttention(n_embd, n_head, window_size)
        elif method == 'block':
            self.self_attn = BlockwiseAttention(n_embd, n_head, block_size)
        else:
            raise ValueError("Invalid method. Choose 'standard', 'alibi', 'disentangled', 'local', or 'block'.")

        self.norm1 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 100),  # Feedforward hidden size is 100
            nn.ReLU(),
            nn.Linear(100, n_embd)
        )
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        out, weights = self.self_attn(x, mask=mask)
        x = x + out  # Residual connection
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + ffn_out  # Residual connection
        x = self.norm2(x)

        return x, weights  # Return output and attention probabilities

class TransformerDecoder(nn.Module):
    """Transformer Decoder consisting of multiple decoder layers."""
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_seq_len, method='local', window_size=4, block_size=8):
        super(TransformerDecoder, self).__init__()
        self.method = method
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, n_embd)  # Token embeddings

        if method in ['standard', 'disentangled', 'local', 'block']:
            self.pos_emb = nn.Embedding(max_seq_len, n_embd)  # Positional embeddings
        # No positional embeddings for AliBi in the embedding layer

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(n_embd, n_head, method, max_seq_len, window_size, block_size) for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(n_embd)
        self.fc_out = nn.Linear(n_embd, vocab_size)  # Output layer to predict next token

    def forward(self, x, mask=None):
        N, seq_len = x.size()
        x = self.token_emb(x)
        if self.method in ['standard', 'disentangled', 'local', 'block']:
            positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(N, seq_len)
            x = x + self.pos_emb(positions)  # Add positional embeddings

        # Mask future tokens
        if mask is None:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)
            mask = causal_mask
        else:
            # Combine provided mask with causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            mask = mask | causal_mask  # Combine masks with logical OR

        weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask=mask)
            weights_list.append(attn_weights)

        x = self.norm(x)
        x = self.fc_out(x)
        return x, weights_list  # Return logits and attention weights
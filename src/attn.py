import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Implementing the multi-head attention module for the ViT.
    Takes sequence of embeddings as inputs (positional included) and computes the query, key, value for each embedding.
    These are then used to calculate the attention weights for each token.
    The attention weights are used to calculate new embeddings using a weighted sum of value vectors.
    """
    def __init__(self, dim, num_heads, attn_dropout=0.0, proj_dropout=0.0, qkv_bias=True):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # Embedding dim per head
        self.scale = self.head_dim ** -0.5
        
        # to create QKV from patch + positional embedding.
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # B - batch_size, N - number of tokens (patches) == 16*16
        B, N, C = x.shape
        qkv = self.qkv(x)   # First projection.
        
        # Reshaping QKV --> [batch_size, tokens (N), QKV, number of heads, head_embedding_size]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Splitting the tensor into 3 equal parts along the first dimension.
        # (This corresponds to the q, k, and v tensors).
        q, k, v = qkv.unbind(0)

        # Transposing the last 2 dimensions of k.
        # q, k shape --> B, num_heads, N, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (q.k_t)/sqrt(head_dim)
        attn = attn.softmax(dim=-1) # softmax
        attn = self.attn_drop(attn) # dropout
        
        # Shape of attn --> B, num_heads, N, N
        # Shape of v --> B, num_heads, N, head_dim
        # Shape of attn @ v --> B, num_heads, N, head_dim
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # B, N, num_heads, head_dim
        x = self.proj(x)    # Projecting to embedding dim.
        x = self.proj_drop(x)   # Dropout
        return x
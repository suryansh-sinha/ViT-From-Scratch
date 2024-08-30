import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attn import Attention
from src.mlp import MLP

class Blocks(nn.Module):
    """
    Creating a reusable transformer block.
    """
    
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias=True, proj_dropout=0.0, attn_dropout=0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.attn = Attention(dim, num_heads, attn_dropout, proj_dropout, qkv_bias)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x
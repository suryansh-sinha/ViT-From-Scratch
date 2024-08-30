import torch
import torch.nn as nn
from src.embedding import Embeddings
from src.block import Blocks

class ViT(nn.Module):
    def __init__(self, img_size=384, patch_size=16,
                 in_channels=3, num_classes=100, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4,
                 qkv_bias=True, proj_dropout=0.0, attn_dropout=0.0):
        super().__init__()
        
        # Creating positional embeddings
        self.patch_embed = Embeddings(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.npatches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+num_patches, embed_dim))
        self.pos_drop = nn.Dropout(proj_dropout)
        
        self.blocks = nn.Sequential(*[
            Blocks(embed_dim, num_heads, mlp_ratio, qkv_bias, proj_dropout, attn_dropout) \
                for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)   # Classifier head
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), 1)        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        return self.head(x[:, 0])
        
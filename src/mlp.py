import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron
    """
    def __init__(self, dim, hidden, p=0.0):
        super().__init__()
        
        self.l1 = nn.Linear(dim, hidden)
        self.l2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.l2(self.dropout(self.act(self.l1(x)))))
        return x
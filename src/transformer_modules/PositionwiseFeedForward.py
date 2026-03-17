from torch import nn
import torch

# --------------------------------------------------------------------
# 5) Classe du réseau à propagation avant (FFN)
# --------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """
    Réseau feed-forward positionnel (FFN) avec activation ReLU.
    Composé de deux transformations linéaires avec un dropout.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.dropout(self.activation(self.W1(x))))
        
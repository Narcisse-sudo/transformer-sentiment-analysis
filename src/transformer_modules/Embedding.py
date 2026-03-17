import math
import torch
from torch import nn

# --------------------------------------------------------------------
# 9) Classe de la couche d'embedding
# --------------------------------------------------------------------
class Embeddings(nn.Module):
    """
    Couche d'embedding des tokens.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model)
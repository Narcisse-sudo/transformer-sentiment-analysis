import torch
import torch.nn as nn

# --------------------------------------------------------------------
# 2) Classe pour la normalisation d'une couche
# --------------------------------------------------------------------
class LayerNorm(nn.Module):    
    """
    Normalisation de couche appliquée indépendamment
    à chaque position de la séquence.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
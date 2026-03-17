from torch import nn
from . import LayerNorm

# --------------------------------------------------------------------
# 3) Classe de la connection résiduelle
# --------------------------------------------------------------------
class SubLayerConnection(nn.Module):
    """
    Bloc résiduel avec normalisation préalable (Pre-LN).
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_fn):
        return x + self.dropout(sublayer_fn(self.norm(x)))
    
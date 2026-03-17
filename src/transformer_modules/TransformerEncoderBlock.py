from torch import nn
from . import SubLayerConnection
import torch

# --------------------------------------------------------------------
# 6) Classe d'un block d'encodeur
# --------------------------------------------------------------------
class TransformerEncoderBlock(nn.Module):
    """
    Bloc encodeur Transformer composé d'une self-attention multi-têtes et d'un réseau feed-forward
    positionnel, avec connexions résiduelles et normalisation préalable (Pre-LayerNorm).
    """
    def __init__(self, d_model: int, attention: nn.Module, feed_forward: nn.Module, dropout: float):
        super().__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.d_model = d_model
        self.residual_blocks = nn.ModuleList([
            SubLayerConnection(d_model, dropout),
            SubLayerConnection(d_model, dropout)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention
        x = self.residual_blocks[0](x, lambda x: self.attention(x, x, x, mask))
        # Feed-forward
        x = self.residual_blocks[1](x, self.feed_forward)
        return x
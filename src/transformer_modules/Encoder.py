from torch import nn
from . import build_encoder_stack
from . import LayerNorm
import torch

# --------------------------------------------------------------------
# 7) Classe  pour N blocks d'encodeurs
# --------------------------------------------------------------------
class Encoder(nn.Module):
    """
    Encodeur Transformer composé d'un empilement de blocs encodeurs avec normalisation finale.
    """
    def __init__(self, encoder_block: nn.Module, depth: int):
        super().__init__()
        self.blocks = build_encoder_stack(encoder_block, depth)
        self.final_norm = LayerNorm(encoder_block.d_model)

    def forward( self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return self.final_norm(x)
    
from torch import nn
from .TransformerEncoderBlock import TransformerEncoderBlock
from .MultiHeadedAttention import MultiHeadedAttention
from .PositionwiseFeedForward import PositionwiseFeedForward
from .PositionalEncoding import PositionalEncoding
from .Embedding import Embeddings
from .Encoder import Encoder

# --------------------------------------------------------------------
# 11) Classe finale transformer
# --------------------------------------------------------------------
class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, N, h, d_ff, dropout):
        super().__init__()

        self.embed = Embeddings(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)

        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        block = TransformerEncoderBlock(d_model, attn, ff, dropout)
        self.encoder = Encoder(block, N)

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pos(x)
        return self.encoder(x, mask)
from torch import nn
from . import build_encoder_stack, attention

# --------------------------------------------------------------------
# 8) Classe pour plusieurs têtes d'attentions
# --------------------------------------------------------------------
class MultiHeadedAttention(nn.Module):
    """
    Implémente la self-attention multi-têtes du Transformer. L'entrée est projetée
    en plusieurs têtes d'attention, l'attention est calculée en parallèle sur chaque 
    tête, puis les résultats sont concaténés et reprojetés dans l'espace du modèle.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.h = h                      # nombre de têtes
        self.d_k = d_model // h         # dimension par tête
        self.linears = build_encoder_stack(nn.Linear(d_model, d_model), 4)
        self.attn = None                # poids d'attention 
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # même masque appliqué à toutes les têtes
        if mask is not None:
            # Normalize mask to shape (batch, h, seq_len, seq_len)
            if mask.dim() == 4:
                # expected (batch, 1, seq_len, seq_len)
                mask = mask.expand(-1, self.h, -1, -1)
            elif mask.dim() == 3:
                # (batch, seq_len, seq_len) -> add head dim
                mask = mask.unsqueeze(1).expand(-1, self.h, -1, -1)
            elif mask.dim() == 2:
                # (batch, seq_len) -> create padding mask
                mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.h, query.size(1), -1)

        batch_size = query.size(0)

        # projections linéaires + découpage en h têtes
        query, key, value = [linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears[:3], (query, key, value))]

        # attention appliquée en parallèle sur chaque tête
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # concaténation des têtes + projection finale
        x = ( x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k))

        return self.linears[3](x) 
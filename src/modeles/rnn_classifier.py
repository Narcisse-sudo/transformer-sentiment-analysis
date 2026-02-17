# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@time: 2026/02/02 23:00
"""

# %%
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

class RNNClassifier(nn.Module):
    """
    RNN classifier (GRU ou LSTM) pour la classification binaire de textes.

    Hypothèses
    - Entrée x : Tensor (batch, seq_len) contenant des ids de tokens
    - Padding : token <PAD> = pad_idx (0 ici)
    - Labels : float 0/1 
    - Sortie : 1 logit par séquence (avant sigmoid)

    Pourquoi utiliser pack_padded_sequence ?
    - Pour éviter que le RNN "apprenne" sur les tokens de padding
    - Pour accélérer et stabiliser l’entraînement (le RNN ignore les PAD)
    """

    def __init__(self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pad_idx: int = 0,
        rnn_type: str = "gru",  # "gru" ou "lstm"
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()

        # 1) Embedding : transforme les ids de tokens en vecteurs denses
        # padding_idx permet à PyTorch de garder l'embedding PAD "neutre"
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # 2) Choix du type de RNN : GRU ou LSTM
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM

        # RNN : lit la séquence token par token
        # bidirectional=True: lit de gauche à droite ET droite à gauche
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # entrée en (batch, seq, features)
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # dropout entre les couches
        )

        # 3) Tête de classification
        # Sortie de la dernière couche : hidden_dim * (2 si bidirectionnel)
        out_dim = hidden_dim * (2 if bidirectional else 1)

        # MLP simple 
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1)  # 1 logit (binaire)
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        x : (batch, seq_len)
        retourne logits : (batch,)  (un score par phrase)
        """

        # --- 1) Calcul des longueurs réelles (sans padding) ---
        # On compte le nombre de tokens != PAD sur chaque phrase
        lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1)

        # --- 2) Embedding ---
        # emb : (batch, seq_len, embed_dim)
        emb = self.embedding(x)

        # --- 3) Packing (ignorer PAD) ---
        # pack_padded_sequence attend les lengths sur CPU
        # enforce_sorted=False => pas besoin de trier le batch par longueur
        packed_emb = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # --- 4) Passage dans le RNN ---
        # h est l'état caché final
        _, h = self.rnn(packed_emb)

        # --- 5) Extraire le "résumé" de la phrase ---
        # Pour GRU : h est Tensor (num_layers * num_directions, batch, hidden_dim)
        # Pour LSTM : h = (h_n, c_n) donc on prend h_n
        if self.rnn_type == "lstm":
            h = h[0]  # h_n

        # Dernière couche:
        # - si bidirectionnel : on concatène forward et backward
        # - sinon : on prend juste la dernière direction
        if self.bidirectional:
            # h[-2] = dernier état forward, h[-1] = dernier état backward
            h_last = torch.cat([h[-2], h[-1]], dim=1)  # (batch, 2*hidden_dim)
        else:
            h_last = h[-1]  # (batch, hidden_dim)

        # --- 6) Classification ---
        logits = self.classifier(h_last).squeeze(-1)  # (batch,)
        return logits

# %%

#%%
#!/usr/bin/env python3
# -*- codeing: utf-8 -*-
"""
@time: 2026/02/02 22:57
"""
# %%
# Création d'un réseau de neurones récurrent 

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNNClassifier(nn.Module):
    """
    TextCNN pour classification binaire.
    - x: (B, L) ids de tokens
    - padding_idx: 0
    - output: logits (B,)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes=(3, 4, 5),
        dropout: float = 0.3,
        pad_idx: int = 0
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # Plusieurs convs avec différentes tailles de filtres (n-grams)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        # Tête de classification
        out_dim = num_filters * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, 1)  # 1 logit
        )

    def forward(self, x, mask=None):
        """
        x: (B, L)
        """
        # Embedding : (B, L, E)
        emb = self.embedding(x)

        # Conv1d attend (B, E, L)
        emb = emb.transpose(1, 2)

        # Convolutions + ReLU + Global Max Pooling
        pooled_outputs = []
        for conv in self.convs:
            # feature_map: (B, num_filters, L-k+1)
            feature_map = F.relu(conv(emb))
            # global max pooling sur la dimension temporelle
            pooled = F.max_pool1d(feature_map, kernel_size=feature_map.size(2)).squeeze(2)
            # pooled: (B, num_filters)
            pooled_outputs.append(pooled)

        # Concat : (B, num_filters * len(kernel_sizes))
        features = torch.cat(pooled_outputs, dim=1)

        logits = self.classifier(features).squeeze(-1)  # (B,)
        return logits


# %%

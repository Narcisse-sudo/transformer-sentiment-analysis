#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import torch
from torch import nn
import copy
import math
import numpy as np


# --------------------------------------------------------------------
# 1) Fonction pour la création de plusieurs blocs d'encodeurs ou couches
# --------------------------------------------------------------------
def build_encoder_stack(block: nn.Module, depth: int) -> nn.ModuleList:
    """
    Construit une pile de blocs encodeurs Transformer.
    Chaque bloc possède ses propres paramètres.
    """
    return nn.ModuleList(
        copy.deepcopy(block) for _ in range(depth)
    )


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
    
# --------------------------------------------------------------------
# 4) Fonction d'attention
# --------------------------------------------------------------------
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute Scaled Dot-Product Attention.
    Args:
        query: Tensor (..., seq_len_q, d_k)
        key:   Tensor (..., seq_len_k, d_k)
        value: Tensor (..., seq_len_k, d_v)
        mask:  Tensor optionnel pour masquer certaines positions
        dropout: Module Dropout appliqué aux poids d'attention
    Returns:
        output: Résultat de l'attention
        p_attn: Poids d'attention
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    output = torch.matmul(p_attn, value)

    return output, p_attn


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
    

# --------------------------------------------------------------------
# 10) Classe pour l'encodage positionnel des enbeddings    
# --------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

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

# --------------------------------------------------------------------
# 12) Classe de classificateur basé sur le Transformer
# --------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """
    Classificateur basé sur un Transformer encoder.
    
    Args :
    - transformer : instance de TransformerEncoderModel (l'encodeur)
    - d_model : dimension des embeddings / sorties du Transformer
    - num_classes : nombre de classes pour la classification
    - pool_strategy : stratégie de pooling sur la séquence ("mean", "max", "first", "attention")
    - dropout : taux de dropout pour régularisation
    """
    
    def __init__(self, transformer, d_model, num_classes=2, pool_strategy="mean", dropout=0.1):
        super().__init__()
        self.transformer = transformer
        self.pool_strategy = pool_strategy
        
        #  Tête de classification
        # Transforme le vecteur de dimension d_model en logits pour num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),           
            nn.Linear(d_model, d_model // 2),  
            nn.ReLU(),                     
            nn.Dropout(dropout),           
            nn.Linear(d_model // 2, num_classes) 
        )
        
    def forward(self, x, mask):
        """
        args :
            - x : batch de séquences de tokens (batch, seq_len)
            - mask : mask de padding 
        
        returns :
            - logits : (batch, num_classes)
        """
        x = self.transformer(x, mask)
        
        # Pooling pour obtenir un vecteur par séquence 
        if self.pool_strategy == "mean":
            pooled = x.mean(dim=1)         
        elif self.pool_strategy == "max":
            pooled, _ = x.max(dim=1)      
        elif self.pool_strategy == "first":
            pooled = x[:, 0, :]            
        else:
            raise ValueError(f"Unknown pool_strategy: {self.pool_strategy}")
        
        #Passage dans la tête de classification
        logits = self.classifier(pooled)    # (batch, num_classes)
        return logits

    # --- Option : attention pooling ---
    def _init_attention_pool(self, d_model):
        """
        Initialise une couche linéaire pour attention pooling.
        Chaque token reçoit un poids appris pour le pooling.
        """
        self.attention_pool = nn.Linear(d_model, 1)

# %%

#!/urs/bin/env python3
# -*- coding: utf-8 -*-
""" 
@time: 2026/02/02 00:21
"""

from torch import nn
# --------------------------------------------------------------------
#  Classe de classificateur basé sur le Transformer
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

"""
@author : Seorin Kim(@Seorin-Kim)
@when : 2022-09-03
@github : https://github.com/Seorin-Kim
"""

import torch.nn as nn
from models.layer.LayerNorm import LayerNorm

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        drop = sublayer(self.norm(x))
        return x + self.dropout(drop[0])
"""
@author : Seorin Kim(@Seorin-Kim)
@when : 2022-09-03
@github : https://github.com/Seorin-Kim
"""

import math
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
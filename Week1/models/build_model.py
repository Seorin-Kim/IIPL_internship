"""
@author : Seorin Kim(@Seorin-Kim)
@when : 2022-09-03
@github : https://github.com/Seorin-Kim
"""

import torch
import torch.nn as nn
from torchtext import data

from models.model.TransformerModel import TransformerModel

def build_model(text, 
                model_dim=512,
                n_heads=8,
                hidden_dim=256,
                n_layers=3,
                device=torch.device('cpu'),
                dropout_ratio=0.1):
    INPUT_DIM = len(text.vocab)
    TEXT_PAD_IDX = text.vocab.stoi[text.pad_token]

    model = TransformerModel(INPUT_DIM, model_dim, n_heads, hidden_dim, n_layers, TEXT_PAD_IDX, device, dropout_ratio)

    return model
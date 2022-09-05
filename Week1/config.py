"""
@author : Seorin Kim(@Seorin-Kim)
@when : 2022-09-03
@github : https://github.com/Seorin-Kim
"""

import torch

SEED = 5

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

BATCH_SIZE = 64
lr = 0.001
EPOCHS = 3

MODEL_DIM = 512
HIDDEN_DIM = 256
N_LAYERS = 3
N_HEADS = 8
DROPOUT_RATIO = 0.1
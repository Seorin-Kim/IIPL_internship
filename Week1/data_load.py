"""
@author : Seorin Kim(@Seorin-Kim)
@when : 2022-09-03
@github : https://github.com/Seorin-Kim
"""

import random
import torch
from torchtext import data, datasets

from config import *


def data_load():
    random.seed(SEED)
    torch.manual_seed(SEED)

    TEXT = data.Field(sequential=True, batch_first=True, lower=True, fix_length=512)
    LABEL = data.Field(sequential=False, batch_first=True)

    trainset, testset = datasets.IMDB.splits(TEXT, LABEL)

    TEXT.build_vocab(trainset, min_freq=5)
    LABEL.build_vocab(trainset)

    trainset, valset = trainset.split(split_ratio=0.8)

    train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset,valset,testset), batch_size=BATCH_SIZE, shuffle=True, repeat=False)

    return TEXT, train_iter, val_iter, test_iter
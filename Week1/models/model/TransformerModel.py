"""
@author : Seorin Kim(@Seorin-Kim)
@when : 2022-09-03
@github : https://github.com/Seorin-Kim
"""

import torch.nn as nn
import copy

from models.layer.MultiHeadedAttention import MultiHeadedAttention
from models.layer.PositionwiseFeedForward import PositionwiseFeedForward
from models.embedding.PositionalEncoding import PositionalEncoding
from models.embedding.Embeddings import Embeddings
from models.model.EncoderLayer import EncoderLayer
from models.model.Encoder import Encoder

class TransformerModel(nn.Module):
        
    def __init__(self, n_token, n_dim_model, n_head, n_hidden, n_blocks, src_pad_idx, device, dropout=0.5):
        super(TransformerModel, self).__init__()
        
        # #Multi Headed Attention Layer
        self_attention = MultiHeadedAttention(n_head, n_dim_model, device)
        # #Feedforward Layer
        feed_forward = PositionwiseFeedForward(n_dim_model, n_hidden, dropout)
        # #Positional Encoding
        positional_encoding = PositionalEncoding(n_dim_model, dropout)
        
        encoder_layer = EncoderLayer(n_dim_model, copy.deepcopy(self_attention), copy.deepcopy(feed_forward), dropout)
        self.encoder = Encoder(encoder_layer, n_blocks)

        embedding = Embeddings(n_dim_model, n_token)
        self.src_embed = nn.Sequential(embedding, copy.deepcopy(positional_encoding))
        
        # Fully-Connected Layer
        self.fc = nn.Linear(n_dim_model, 2)

        self.src_pad_idx = src_pad_idx


    # 소스 문장의 <pad> 토큰에 대하여 마스크(mask) 값을 0으로 설정
    def make_src_mask(self, src):

        # src: [batch_size, hid_dim]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask: [batch_size, 1, 1, hid_dim]

        return src_mask

    

    def forward(self, x):
        
        mask = self.make_src_mask(x)

        # # x dimension[k, batch_size = 64]
        embedded_sents = self.src_embed(x)
        encoded_sents = self.encoder(embedded_sents, mask)
        
       
        final_feature_map = encoded_sents[:,-1,:] 
        final_out = self.fc(final_feature_map) 
        
        return final_out
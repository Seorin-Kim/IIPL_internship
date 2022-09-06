"""
@author : Seorin Kim(@Seorin-Kim)
@when : 2022-09-03
@github : https://github.com/Seorin-Kim
"""

import torch
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, hid_dim, device, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert hid_dim % n_heads == 0
        # We assume d_v always equals head_dim
        self.head_dim = hid_dim // n_heads
        self.n_heads = n_heads
        self.hid_dim = hid_dim

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask=None):
        
        # query, key, value = [batch_size, token_length, hid_dim]

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q, K, V = [batch_size, token_length, hid_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0,2,1,3)

        # Q, K, V = [batch_size, n_heads, token_length, head_dim]

        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask==0, -1e10)
        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention),V)
        x = x.permute(0,2,1,3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)

        return x, attention
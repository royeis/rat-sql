import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Linear
import math


class RelationAwareMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.):
        super(RelationAwareMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.dropout = nn.Dropout(dropout)
        self.W_Q = Linear(embed_dim, embed_dim, bias=False)
        self.W_K = Linear(embed_dim, embed_dim, bias=False)
        self.W_V = Linear(embed_dim, embed_dim, bias=False)
        self.relation_bias = nn.Embedding(33, embed_dim)

    def forward(self, query, key, value, relations):
        batch_size = query.size(0)
        seq_len = query.size(1)
        assert seq_len == relations.size(1), "there should be a relation between each pair of items in the input"

        # prepare the correct relation representation for each pair of items in the input
        r = self.relation_bias(relations)

        # apply matrix multiplications
        q_tmp = self.W_Q(query)
        k_tmp = self.W_K(key)
        v_tmp = self.W_V(value)

        # add relation bias
        k_tmp = k_tmp.unsqueeze(2).repeat(1, 1, seq_len, 1)
        v_tmp = v_tmp.unsqueeze(2).repeat(1, 1, seq_len, 1)
        k_tmp = torch.add(k_tmp, 1, r)
        v_tmp = torch.add(v_tmp, 1, r)

        # split all tensors to num_heads
        q_tmp = q_tmp.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_tmp = k_tmp.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim)
        v_tmp = v_tmp.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim)

        # prepare to multiply matrices of dim seq_len * head_dim
        q_tmp = q_tmp.transpose(1, 2)
        k_tmp = k_tmp.transpose(2, 3)
        v_tmp = v_tmp.transpose(2, 3)

        # compute e_ij for each pair
        scores = torch.stack([torch.matmul(q_tmp[:, :, i, :].unsqueeze(2), k_tmp[:, i, :, :, :].transpose(-2, -1)).squeeze(2) for i in range(seq_len)], dim=2) / math.sqrt(self.head_dim)

        # compute alpha_ij for each pair
        alpha = F.softmax(scores, -1)

        if self.dropout is not None:
            alpha = self.dropout(alpha)

        # compute z_i for each token
        z = torch.stack([torch.matmul(alpha[:, :, i, :].unsqueeze(2), v_tmp[:, i, :, :, :]).squeeze(2) for i in range(seq_len)], dim=1)

        # concat outputs of each head and return the new representation of the sequence
        return z.view(batch_size, seq_len, -1)


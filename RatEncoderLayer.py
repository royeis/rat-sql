import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from RelationAwareMultiheadAttention import RelationAwareMultiheadAttention


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class RatEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(RatEncoderLayer, self).__init__()
        self.relation_attn = RelationAwareMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, xi, xj, relation_bias, src_mask=None, src_key_padding_mask=None):
        zi = self.relation_attn(xi, xj, xj, relation_bias, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        yi_tag = xi + self.dropout1(zi)
        yi_tag = self.norm1(yi_tag)
        if hasattr(self, "activation"):
            tmp = self.linear2(self.dropout(self.activation(self.linear1(yi_tag))))
        else:  # for backward compatibility
            tmp = self.linear2(self.dropout(F.relu(self.linear1(yi_tag))))
        yi = yi_tag + self.dropout2(tmp)
        yi = self.norm2(yi)
        return yi

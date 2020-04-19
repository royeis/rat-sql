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

    def forward(self, sequence, relations):
        z = self.relation_attn(sequence, sequence, sequence, relations)
        y_tag = sequence + self.dropout1(z)
        y_tag = self.norm1(y_tag)
        if hasattr(self, "activation"):
            tmp = self.linear2(self.dropout(self.activation(self.linear1(y_tag))))
        else:  # for backward compatibility
            tmp = self.linear2(self.dropout(F.relu(self.linear1(y_tag))))
        y = y_tag + self.dropout2(tmp)
        y = self.norm2(y)
        return y

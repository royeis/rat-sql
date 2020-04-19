import torch.nn as nn
from torch.nn.modules import ModuleList
import copy


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class RatEncoder(nn.Module):

    def __init__(self, rat_encoder_layer, num_layers, norm=None):
        super(RatEncoder, self).__init__()
        self.layers = _get_clones(rat_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, sequence, relations):
        output = sequence

        for i in range(self.num_layers):
            output = self.layers[i](output, relations)

        if self.norm:
            output = self.norm(output)

        return output

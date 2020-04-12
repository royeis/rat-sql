import torch
import torch.nn as nn
from RatEncoderLayer import RatEncoderLayer
import torch.nn.functional as F
from transformers import BertModel
from torch.nn.modules.transformer import TransformerEncoder


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        # We use Bert pretrained embeddings as initial encodings for the input.
        self.initial_encoder = BertModel.from_pretrained('bert-base-uncased')

        # The linear layer transforms the initial encoding to match the dimensions specified in the article.
        self.linear = nn.Linear(768, 256)

        # A single layer of the RAT encoder.
        self.rat_layer = RatEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024)

        # A transformer encoder module consists of 8 RAT layers.
        self.rat_encoder = TransformerEncoder(self.rat_layer, 8)

    def forward(self, ):

        self.initial_encoder(input_question)

if __name__ == '__main__':
    Encoder()

    print('hello')
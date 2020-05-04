import difflib
import os
from functools import partial
from typing import Dict, List, Tuple, Any, Mapping, Sequence

import sqlparse
import torch
from torch.nn.modules import Linear
from allennlp.common.util import pad_sequence_to_length
from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule, ProductionRuleArray
from allennlp.models import Model
from allennlp.modules import Embedding, Attention, FeedForward, \
    TimeDistributed
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util, Activation
from allennlp.state_machines import BeamSearch
from allennlp.state_machines.states import GrammarStatelet

from RatEncoder import RatEncoder
from RatEncoderLayer import RatEncoderLayer
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.training.metrics import Average
from overrides import overrides


class SpiderParser(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 decoder_beam_search: BeamSearch,
                 input_attention: Attention,
                 past_attention: Attention,
                 max_decoding_steps: int,
                 action_embedding_dim: int,
                 decoder_self_attend: bool = True,
                 parse_sql_on_decoding: bool = True,
                 add_action_bias: bool = True,
                 dataset_path: str = 'spider',
                 training_beam_size: int = None,
                 decoder_num_layers: int = 1,
                 dropout: float = 0.0,
                 rule_namespace: str = 'rule_labels',
                 scoring_dev_params: dict = None,
                 debug_parsing: bool = False) -> None:

        super().__init__(vocab)
        self.vocab = vocab

        self._max_decoding_steps = max_decoding_steps
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

        self._rule_namespace = rule_namespace

        self._add_action_bias = add_action_bias
        self._scoring_dev_params = scoring_dev_params or {}
        self.parse_sql_on_decoding = parse_sql_on_decoding
        self._self_attend = decoder_self_attend
        self._action_padding_index = -1

        self._exact_match = Average()
        self._sql_evaluator_match = Average()
        self._action_similarity = Average()
        self._acc_single = Average()
        self._acc_multi = Average()
        self._beam_hit = Average()

        num_actions = vocab.get_vocab_size(self._rule_namespace)
        if self._add_action_bias:
            input_action_dim = action_embedding_dim + 1
        else:
            input_action_dim = action_embedding_dim
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=input_action_dim)
        self._output_action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)

        encoder_output_dim = 256

        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        self._first_attended_utterance = torch.nn.Parameter(torch.FloatTensor(encoder_output_dim))
        self._first_attended_output = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)
        torch.nn.init.normal_(self._first_attended_utterance)
        torch.nn.init.normal_(self._first_attended_output)

        # The linear layer transforms the initial encoding to match the dimensions specified in the article.
        self.input_linear = Linear(768, 256)

        # A single layer of the RAT encoder.
        self.rat_layer = RatEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024)

        # A transformer encoder module consists of 8 RAT layers.
        self.encoder = RatEncoder(self.rat_layer, 8)

        self._decoder_num_layers = decoder_num_layers

        self._beam_search = decoder_beam_search
        self._decoder_trainer = MaximumMarginalLikelihood(training_beam_size)

    def forward(self, sequence):
        pass

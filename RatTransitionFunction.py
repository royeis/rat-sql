from typing import List, Set

from allennlp.state_machines import TransitionFunction
from allennlp.state_machines.transition_functions.transition_function import StateType
import torch.nn as nn

class RatTransitionFunction(TransitionFunction):
    def __init__(self,
                 encoder_output_dim: int,
                 action_embedding_dim: int,
                 num_actions: int,
                 node_embedding_dim: int,
                 num_nodes: int
                 ):
        self.action_embeddings = nn.Embedding(num_actions, action_embedding_dim)
        self.node_embeddings = nn.Embedding(num_nodes, node_embedding_dim)


    def take_step(self, state: StateType, max_actions: int = None, allowed_actions: List[Set] = None) -> List[
        StateType]:
        pass
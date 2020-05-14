import torch.nn as nn


class RatDecoder(nn.Module):
    def __init__(self,
                 encoder_output_sequence,
                 lstm_hidden_size,
                 lstm_dropuout,
                 dim_action_embeddings,
                 num_action_types,
                 dim_node_embeddings,
                 num_node_types,
                 column_alignment_matrix,
                 table_alignment_matrix):

        super(RatDecoder).__init__()
        encoder_output_sequence_size = encoder_output_sequence.size(2)
        self.encoder_output_sequence = encoder_output_sequence
        self.action_embeddings = nn.Embedding(num_action_types, dim_action_embeddings)
        self.node_embeddings = nn.Embedding(num_node_types, dim_node_embeddings)
        self.lstm = nn.LSTM(2 * dim_action_embeddings + encoder_output_sequence_size + dim_action_embeddings,
                            lstm_hidden_size,
                            dropout=lstm_dropuout)
        self.l_col = column_alignment_matrix
        self.l_table = table_alignment_matrix

    def forward(self,
                prev_cell_state,
                prev_hidden_state,
                prev_action_idx,
                parent_hidden_state,
                parent_action_idx,
                frontier_node_idx):
        pass
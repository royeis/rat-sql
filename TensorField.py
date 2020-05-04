from typing import Dict

from allennlp.data.fields import SequenceField
from allennlp.data import DataArray
import torch.nn.functional as F
import torch


class SequenceTensorField(SequenceField[torch.Tensor]):
    def __init__(self, sequence_tensor: torch.Tensor) -> None:
        self.sequence_tensor = sequence_tensor

    def sequence_length(self) -> int:
        return self.sequence_tensor.size(1)

    def get_padding_lengths(self) -> Dict[str, int]:
        return {'sequence_length': self.sequence_length()}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        # We assume tensors of shape (batch size, sequence length, embedding dimension),
        # therefore we pad the sequence length to the right.
        desired_len = padding_lengths['sequence_length']
        current_len = self.sequence_tensor.size(1)
        return F.pad(self.sequence_tensor, [0, 0, 0, desired_len - current_len])

    def empty_field(self) -> SequenceField:
        pass


class RelationsTensorField(SequenceField[torch.Tensor]):
    def __init__(self, relations_tensor: torch.Tensor) -> None:
        self.relations_tensor = relations_tensor

    def sequence_length(self) -> int:
        return self.relations_tensor.size(1)

    def empty_field(self) -> 'SequenceField':
        pass

    def get_padding_lengths(self) -> Dict[str, int]:
        return {'sequence_length': self.sequence_length()}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> DataArray:
        # We assume tensors of shape (batch size, sequence length, sequence length),
        # therefore we pad the last two last dimensions to the right.
        # Padding here is done inserting a dummy relation type
        desired_len = padding_lengths['sequence_length']
        current_len = self.relations_tensor.size(1)
        return F.pad(self.relations_tensor, [0, 0, 0, desired_len - current_len, 0, desired_len - current_len], value=33)

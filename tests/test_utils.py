import torch
import pytest
from minigpt.utils import collate_func


@pytest.fixture
def sample_batch():
    return [
        [1, 2, 3],
        [4, 5],
        [6]
    ]

@pytest.fixture
def default_padding_values():
    return (50256, -100)


def test_collate_func_output_dimension(sample_batch):
    inputs, targets = collate_func(sample_batch)
    num_rows = len(sample_batch)
    max_length = max(len(item) for item in sample_batch)

    assert inputs.shape == (num_rows, max_length)
    assert targets.shape == (num_rows, max_length)

def test_collate_func_padding_values(sample_batch, default_padding_values):
    inputs, targets = collate_func(sample_batch)
    padding_token_id_value, ignore_index_value = default_padding_values

    assert padding_token_id_value in inputs
    assert padding_token_id_value in targets
    assert ignore_index_value in targets
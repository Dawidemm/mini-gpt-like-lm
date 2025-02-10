import pytest
import torch
import tiktoken
from minigpt.utils import PretrainDataset

@pytest.fixture
def sample_text():
    return "Test sentence for the PretrainDataset class."

@pytest.fixture
def dataset(sample_text):
    return PretrainDataset(txt=sample_text, max_lenght=3, stride=1)

def test_length_of_dataset(dataset):
    expected_len = len(dataset.input_ids)
    assert len(dataset) == expected_len

def test_getitem_type(dataset):
    input_chunk, target_chunk = dataset[0]
    assert isinstance(input_chunk, torch.Tensor)
    assert isinstance(target_chunk, torch.Tensor)

def test_getitem_max_lenght(dataset):
    input_chunk, target_chunk = dataset[0]
    assert input_chunk.size(0) == 3
    assert target_chunk.size(0) == 3

def test_tokenization(dataset):
    input_chunk, target_chunk = dataset[0]
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode("Test sentence for the PretrainDataset class.", allowed_special={"<|endoftext|>"})
    assert input_chunk.tolist() == token_ids[:3]
    assert target_chunk.tolist() == token_ids[1:4]
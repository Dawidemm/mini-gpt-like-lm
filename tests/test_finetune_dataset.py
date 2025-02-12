import pytest
from unittest.mock import MagicMock
from minigpt.utils import FineTuneDatamodule
import json
import torch


@pytest.fixture
def sample_data():
    return [
        {
            "instruction": "Evaluate the following phrase by transforming it into the spelling given.",
            "input": "freind --> friend",
            "output": "The spelling of the given phrase \"freind\" is incorrect, the correct spelling is \"friend\"."
        },
        {
            "instruction": "Edit the following sentence for grammar.",
            "input": "He go to the park every day.",
            "output": "He goes to the park every day."
        },
        {
            "instruction": "Convert 45 kilometers to meters.",
            "input": "",
            "output": "45 kilometers is 45000 meters."
        },
        {
            "instruction": "Rewrite this sentence to start with 'Although': Despite the rain, they went for a walk.",
            "input": "",
            "output": "Although it was raining, they went for a walk."
        },
        {
            "instruction": "What are the first 10 square numbers?",
            "input": "",
            "output": "1, 4, 9, 16, 25, 36, 49, 64, 81, 100."
        },
        {
            "instruction": "Suggest a more formal synonym for \"happy.\"",
            "input": "",
            "output": "A more formal synonym for \"happy\" is \"content.\""
        },
        {
            "instruction": "Translate the following sentence into French.",
            "input": "Where is the nearest restaurant?",
            "output": "O\u00f9 est le restaurant le plus proche?"
        },
        {
            "instruction": "Look up the melting point of iron.",
            "input": "",
            "output": "The melting point of iron is approximately 1538 degrees Celsius or 2800 degrees Fahrenheit."
        },
        {
            "instruction": "Rewrite the following sentence so that it is in active voice.",
            "input": "The cake was baked by Sarah.",
            "output": "Sarah baked the cake."
        },
        {
            "instruction": "Convert the active sentence to passive: 'The manager approved the project.'",
            "input": "",
            "output": "The project was approved by the manager."
        }
    ]

@pytest.fixture
def sample_dataset_file(sample_data, tmp_path):
    dataset_path = tmp_path / "sample_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(sample_data, f)
    return dataset_path


def test_finetune_datamodule_initialization(sample_dataset_file):
    data_module = FineTuneDatamodule(dataset_path=str(sample_dataset_file))
    assert len(data_module.data) == 10


def test_finetune_datamodule_setup(sample_dataset_file):
    data_module = FineTuneDatamodule(dataset_path=str(sample_dataset_file))
    data_module.setup("fit")
    
    assert isinstance(data_module.train_dataloader(), torch.utils.data.DataLoader)
    assert isinstance(data_module.val_dataloader(), torch.utils.data.DataLoader)



def test_test_dataloader(sample_dataset_file):
    data_module = FineTuneDatamodule(dataset_path=str(sample_dataset_file), batch_size=1)
    data_module.setup("test")
    test_loader = data_module.tests_dataloader()

    batch = next(iter(test_loader))

    assert isinstance(batch, torch.Tensor)
    assert batch.size(0) == data_module.batch_size

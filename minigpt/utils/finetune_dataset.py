import json
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from minigpt.utils.utils import format_input_text

from typing import Dict


class FineTuneDataset(Dataset):
    def __init__(
            self,
            data: Dict
    ):
        self.data = data
        self.encoded_texts = []
        
        tokenizer = tiktoken.get_encoding("gpt2")

        for entry in data:
            formatted_text = format_input_text(entry)
            self.encoded_texts.append(
                tokenizer.encode(formatted_text)
            )
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.encoded_texts[index]
    

class FineTuneDatamodule(LightningDataModule):
    def __init__(
            self,
            dataset_path: str,
            batch_size: int = 8,
            shuffle: bool = True
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle

        with open(dataset_path, "r") as dataset:
            self.data = json.load(dataset)
        
        self.data_length = len(self.data)

    def setup(self, stage: str):
        if stage == "fit":
            train_data = self.data[:int(self.data_length * 0.8)]
            val_data = self.data[int(self.data_length * 0.8):int(self.data_length * 0.9)]
            self.train_dataset = FineTuneDataset(train_data)
            self.val_dataset = FineTuneDataset(val_data)

        if stage == "test":
            test_data = self.data[int(self.data_length * 0.9):]
            self.test_dataset = FineTuneDataset(test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def tests_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
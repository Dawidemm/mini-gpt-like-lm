import torch
import tiktoken
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(
            self,
            txt: str,
            max_lenght: int,
            stride: int = 1
    ):
        self.input_ids = []
        self.target_ids = []

        tokenizer = tiktoken.get_encoding("gpt2")
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_lenght, stride):
            input_chunk = token_ids[i:i+max_lenght]
            target_chunk = token_ids[i+1:i+max_lenght+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
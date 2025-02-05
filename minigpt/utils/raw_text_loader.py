import os
from typing import List
from enum import Enum


class SpecialToken(Enum):
    SPECIALTOKEN = "<|endoftext|>"


class RawTextLoader():
    def __init__(
            self,
            dataset_path: str,
            special_token: str = SpecialToken.SPECIALTOKEN.value
    ):
        self.dataset_path = dataset_path
        self.file_list = self._get_file_list()
        self.special_token = special_token

    def _get_file_list(self) -> List:

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"The folder '{self.dataset_path}' does not exist.")
        
        files = [f for f in sorted(os.listdir(self.dataset_path)) if os.path.isfile(os.path.join(self.dataset_path, f))]
        return files
    
    def load_text(self) -> str:
        
        combined_text = []
        for filename in self.file_list:
            file_path = os.path.join(self.dataset_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                combined_text.append(text)

        return f" {self.special_token} ".join(combined_text)
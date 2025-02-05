import torch
import lightning as L
from torch.utils.data import DataLoader

from minigpt.utils import RawTextLoader, MiniGPTDataset
from minigpt.models import MiniGPT, MiniGPTSettings

torch.manual_seed(42)


DEFAULT_DATASET_PATH = "dataset"
SETTINGS = MiniGPTSettings()

def pretrain_pipeline():
    
    try:
        raw_txt_loader = RawTextLoader(
        dataset_path=DEFAULT_DATASET_PATH
        )
        raw_txt = raw_txt_loader.load_text()

        pretrain_dataset = MiniGPTDataset(
            txt=raw_txt,
            max_lenght=SETTINGS.context_length
        )

        pretrain_dataloader = DataLoader(
            dataset=pretrain_dataset, 
            batch_size=8,
            shuffle=False
        )

        minigpt = MiniGPT()

        trainer = L.Trainer(
            max_epochs=1,
            accelerator="auto",
            logger=True
        )

        trainer.fit(
            model=minigpt, 
            train_dataloaders=pretrain_dataloader
        )
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    pretrain_pipeline()
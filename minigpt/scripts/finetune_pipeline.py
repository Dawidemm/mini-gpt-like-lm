import torch
import lightning as L

from minigpt.utils import FineTuneDatamodule
from minigpt.models import MiniGPT, MiniGPTSettings

torch.manual_seed(42)


DEFAULT_DATASET_PATH = "dataset/finetune/instruction-data.json"
SETTINGS = MiniGPTSettings()

def pretrain_pipeline():
    
    try:
        datamodule = FineTuneDatamodule(
            dataset_path=DEFAULT_DATASET_PATH,
            batch_size=1
        )

        minigpt = MiniGPT.load_from_checkpoint(
            checkpoint_path="lightning_logs/version_2/checkpoints/epoch=4-step=3060.ckpt",
            map_location="cpu"
        )

        trainer = L.Trainer(
            max_epochs=5,
            accelerator="auto",
            logger=True
        )

        trainer.fit(
            model=minigpt, 
            datamodule=datamodule
        )
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    pretrain_pipeline()


import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

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

        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True
        )

        minigpt = MiniGPT.load_from_checkpoint(
            checkpoint_path="lightning_logs/version_2/checkpoints/epoch=4-step=3060.ckpt",
            map_location="cpu"
        )

        trainer = L.Trainer(
            max_epochs=5,
            accelerator="auto",
            logger=True,
            callbacks=[checkpoint_callback]
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


import torch
import mlflow
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from minigpt.utils import FineTuneDatamodule
from minigpt.models import MiniGPT, MiniGPTSettings

torch.manual_seed(42)


DEFAULT_DATASET_PATH = "dataset/finetune/instruction-data.json"
SETTINGS = MiniGPTSettings()

def pretrain_pipeline():
    
    try:
        mlflow_logger = MLFlowLogger(
            experiment_name="MiniGPT-Finetuning",
            tracking_uri="mlruns/"
        )

        mlflow.log_params({
            "batch_size": 1,
            "num_layers": SETTINGS.num_layers,
            "embedding_dim": SETTINGS.embeddings_dim
        })

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
            max_epochs=3,
            accelerator="auto",
            logger=mlflow_logger,
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


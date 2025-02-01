import torch
import torch.nn as nn
import lightning as L
from transformer import TransformerBlock
from layer_norm import LayerNormalization

from settings import MiniGPTSettings

torch.manual_seed(124)

class MiniGPT(L.LightningModule):
    def __init__(
            self,
            settings = MiniGPTSettings()
    ):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            num_embeddings=settings.vocabulary_size,
            embedding_dim=settings.embeddings_dim
        )
        self.positional_embeddings = nn.Embedding(
            num_embeddings=settings.vocabulary_size,
            embedding_dim=settings.embeddings_dim
        )
        self.dropout = nn.Dropout(p=settings.dropout_rate)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(settings.num_layers)]
        )
        self.layer_norm = LayerNormalization(
            embeddings_dim=settings.embeddings_dim
        )

        self.out_head = nn.Linear(
            settings.embeddings_dim,
            settings.vocabulary_size,
            bias=False
        )

    def forward(self, x: torch.Tensor):
        batch_size, sequence_lenght = x.shape
        token_embeddings = self.token_embeddings(x)
        positional_embeddings = self.positional_embeddings(
            torch.arange(sequence_lenght, device=x.device)
        )

        x = token_embeddings + positional_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        predictions = self.out_head(x)

        return predictions
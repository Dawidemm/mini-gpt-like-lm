import torch
import torch.nn as nn
from multi_head import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNormalization

from settings import MiniGPTSettings


class TransformerBlock(nn.Module):
    def __init__(
            self,
            settings = MiniGPTSettings()
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            input_dim=settings.embeddings_dim, 
            output_dim=settings.embeddings_dim,
            context_length=settings.context_length,
            num_heads=settings.num_heads,
            droput=settings.dropout_rate
        )
        self.feedforward = FeedForward(
            embeddings_dim=settings.embeddings_dim
        )
        self.layer_norm1 = LayerNormalization(
            embeddings_dim=settings.embeddings_dim
        )
        self.layer_norm2 = LayerNormalization(
            embeddings_dim=settings.embeddings_dim
        )
        self.dropout = nn.Dropout(
            p=settings.dropout_rate
        )

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
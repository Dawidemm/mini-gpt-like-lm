import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
            self,
            embeddings_dim: int
    ):
        super().__init__()

        layers = [
            nn.Linear(embeddings_dim, 4 * embeddings_dim),
            nn.GELU(),
            nn.Linear(4 * embeddings_dim, embeddings_dim)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
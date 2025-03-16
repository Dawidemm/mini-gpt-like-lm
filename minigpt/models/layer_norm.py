import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    def __init__(
            self,
            embeddings_dim: int
    ):
        super().__init__()

        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embeddings_dim))
        self.shift = nn.Parameter(torch.zeros(embeddings_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift
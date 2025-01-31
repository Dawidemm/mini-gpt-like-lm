from dataclasses import dataclass


@dataclass
class MiniGPTSettings:
    vocabulary_size: int = 50257
    context_length: int = 512
    embeddings_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout_rate: float = 0.1
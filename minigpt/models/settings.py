from dataclasses import dataclass


@dataclass
class MiniGPTSettings:
    vocabulary_size: int = 50257
    context_length: int = 256
    embeddings_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout_rate: float = 0.1
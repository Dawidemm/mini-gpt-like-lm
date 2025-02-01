import torch
import torch.nn as nn
import lightning as L
import tiktoken
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

        self.tokenizer = tiktoken.get_encoding("gpt2")

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
        logits = self.out_head(x)

        return logits
    
    def generate_text(
            self, 
            input_text: str,
            context_size: int,
            num_tokens_to_generate: int
    ) -> str:

        encoded_input_text = self.tokenizer.encode(input_text)
        encoded_input_tensor = torch.tensor(encoded_input_text).unsqueeze(0)

        encoded_output_text = encoded_input_tensor

        for _ in range(num_tokens_to_generate):

            encoded_input_tensor = encoded_input_tensor[:, -context_size:]

            with torch.no_grad():
                logits = self.forward(encoded_input_tensor)

            logits = logits[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            generated_token = torch.argmax(probabilities, dim=-1, keepdim=True)
            encoded_output_text = torch.cat((encoded_output_text, generated_token), dim=1)

        decoded_text = self.tokenizer.decode(encoded_output_text.squeeze(0).tolist())

        return decoded_text
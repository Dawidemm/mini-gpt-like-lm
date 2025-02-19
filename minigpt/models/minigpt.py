import torch
import torch.nn as nn
import lightning as L
import tiktoken
from minigpt.models.transformer import TransformerBlock
from minigpt.models.layer_norm import LayerNormalization

from minigpt.models.settings import MiniGPTSettings

torch.manual_seed(124)

class MiniGPT(L.LightningModule):
    def __init__(
            self,
            learning_rate: float = 0.001,
            settings = MiniGPTSettings()
    ):
        super().__init__()

        self.learning_rate = learning_rate

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
    
    def _shared_step(self, batch):
        inputs, targets = batch
        predicted_tokens = self(inputs)
        loss = torch.nn.functional.cross_entropy(predicted_tokens.flatten(0, 1), targets.flatten())

        return loss
    
    def training_step(self, batch):
        train_loss = self._shared_step(batch)
        self.log("train_loss", train_loss, prog_bar=True, on_epoch=True, on_step=True)

        return train_loss
    
    def validation_step(self, batch):
        val_loss = self._shared_step(batch)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, on_step=False)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        return optimizer
    
    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 100,
            context_size: int = 256, 
            temperature: float = 0.0,
            top_k: int = 3,
            eos_id=50256
        ):

        token_ids = prompt
        token_ids = torch.tensor(self.tokenizer.encode(token_ids), dtype=torch.long).unsqueeze(0)

        for _ in range(max_new_tokens):
            token_ids_cond = token_ids[:, -context_size:]
            with torch.no_grad():
                logits = self(token_ids_cond)
            logits = logits[:, -1, :]

            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float("-inf")), logits)

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                token_ids_next = torch.multinomial(probs, num_samples=1)
            else:
                token_ids_next = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_id is not None and torch.any(token_ids_next == eos_id):
                break

            token_ids = torch.cat((token_ids, token_ids_next), dim=1)

        generated_text = self.tokenizer.decode(token_ids.squeeze().tolist())
        generated_text = generated_text[len(prompt):].strip()

        return generated_text
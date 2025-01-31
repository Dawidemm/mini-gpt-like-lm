import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            context_length: int,
            num_heads: int,
            droput: float
    ):
        super().__init__()

        self.ouput_dim = output_dim
        self.num_heads = num_heads

        assert (output_dim % num_heads == 0), \
        "output_dim must be divisable by num_heads"

        self.head_dim = output_dim // num_heads

        self.W_query = nn.Linear(input_dim, output_dim, bias=False)
        self.W_key = nn.Linear(input_dim, output_dim, bias=False)
        self.W_value = nn.Linear(input_dim, output_dim, bias=False)

        self.heads_combination = nn.Linear(output_dim, output_dim)

        self.dropout = nn.Dropout(droput)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    def forward(self, x: torch.Tensor):

        batch_size, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries  = queries .transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores =  queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(batch_size, num_tokens, self.ouput_dim)
        context_vector = self.heads_combination(context_vector)

        return context_vector
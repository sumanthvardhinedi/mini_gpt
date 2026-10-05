"""
Phase 7: MiniGPT Decoder-Only Language Model in PyTorch.

Educational Notes:
------------------
1. End-to-End Decoder-Only GPT Architecture:
   MiniGPT stacks all components built in Phases 3–6 into a complete autoregressive
   language model:
       Token IDs (batch_size, seq_len)
         ↓
       InputEmbedding (TokenEmbedding + PositionalEmbedding) -> (batch_size, seq_len, embedding_dim)
         ↓
       N × TransformerBlock (Pre-Norm Causal Multi-Head Attention + Pre-Norm GELU FFN)
         ↓
       Final LayerNorm -> (batch_size, seq_len, embedding_dim)
         ↓
       Language Model Head (Linear: embedding_dim -> vocab_size)
         ↓
       Logits (batch_size, seq_len, vocab_size)

2. What are Logits?
   At every sequence position `t`, the language model head outputs a vector of length
   `vocab_size` containing raw, unnormalized scores ("logits") for what the next token
   (`t + 1`) should be. Applying `softmax` to those logits converts them into a probability
   distribution over the entire vocabulary.

3. Weight Tying (`tie_weights=True`):
   The input `TokenEmbedding` table has shape `(vocab_size, embedding_dim)` (mapping token IDs
   to vectors), and the output `lm_head` weight matrix also has shape `(vocab_size, embedding_dim)`
   (projecting hidden vectors back to vocabulary scores). Sharing the exact same weight tensor
   between both layers reduces parameter count significantly and encourages tokens with similar
   input embeddings to receive similar output logit predictions.
"""

from typing import Optional

import torch
import torch.nn as nn

from .embeddings import InputEmbedding
from .transformer import TransformerBlock


class MiniGPT(nn.Module):
    """
    Configurable Decoder-Only GPT Language Model.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        max_sequence_length: int = 128,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        tie_weights: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be positive, got {max_sequence_length}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})."
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        resolved_hidden_dim = 4 * embedding_dim if hidden_dim is None else hidden_dim
        if resolved_hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {resolved_hidden_dim}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.num_layers: int = num_layers
        self.hidden_dim: int = resolved_hidden_dim
        self.dropout_rate: float = dropout
        self.tie_weights: bool = tie_weights

        # 1. Token + Positional Embeddings (Phase 3)
        self.embeddings: InputEmbedding = InputEmbedding(
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
        )
        self.dropout: nn.Dropout = nn.Dropout(p=dropout)

        # 2. Stack of Causal Pre-Norm Transformer Blocks (Phase 6)
        self.blocks: nn.ModuleList = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=self.hidden_dim,
                    dropout=dropout,
                    causal=True,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. Final LayerNorm after the last Transformer block
        self.final_norm: nn.LayerNorm = nn.LayerNorm(embedding_dim)

        # 4. Linear Language Model Head projecting to vocabulary logits
        self.lm_head: nn.Linear = nn.Linear(embedding_dim, vocab_size, bias=bias)

        # Optional Weight Tying between TokenEmbedding and LM Head
        if self.tie_weights:
            self.lm_head.weight = self.embeddings.token_embedding.embedding.weight

    def num_parameters(self, trainable_only: bool = True) -> int:
        """
        Return the total number of unique parameters in the model.
        When `tie_weights=True`, the shared embedding/lm_head weight tensor is counted once.
        """
        seen_ids = set()
        total = 0
        for param in self.parameters():
            if trainable_only and not getattr(param, "requires_grad", True):
                continue
            if id(param) not in seen_ids:
                seen_ids.add(id(param))
                total += param.numel()
        return total

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute next-token prediction logits for `input_ids`.

        Args:
            input_ids: Integer tensor of shape `(batch_size, sequence_length)`
                       (or 1D `(sequence_length,)`).

        Returns:
            Logits tensor of shape `(batch_size, sequence_length, vocab_size)`
            (or `(sequence_length, vocab_size)` if 1D input was provided).
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")

        is_1d = input_ids.dim() == 1
        if is_1d:
            input_ids = input_ids.unsqueeze(0)

        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected input_ids of shape (batch_size, sequence_length), got {input_ids.dim()}D."
            )

        # 1. Token + Positional Embeddings
        x = self.embeddings(input_ids)
        x = self.dropout(x)

        # 2. Pass sequentially through all TransformerBlocks
        for block in self.blocks:
            x = block(x)

        # 3. Final LayerNorm
        x = self.final_norm(x)

        # 4. Project to vocabulary logits
        logits = self.lm_head(x)

        if is_1d:
            return logits[0]
        return logits

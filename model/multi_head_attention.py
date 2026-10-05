"""
Phase 5: Multi-Head Self-Attention for MiniGPT.

Educational Notes:
------------------
1. Why Multi-Head Attention instead of Single-Head Attention?
   A single attention head computes only one set of attention weights per token,
   which means each token can focus on only one weighted combination of positions
   at a time.
   Multi-Head Attention splits the `embedding_dim` into `num_heads` smaller subspaces,
   each of dimension `head_dim = embedding_dim // num_heads`. This allows the model to
   attend to multiple types of relationships in parallel—for example:
   - Head 1 might track subject-verb agreement,
   - Head 2 might attend to adjacent adjectives,
   - Head 3 might track punctuation or long-range pronouns.

2. Tensor Shape Progression:
   Given input `X` of shape `(batch_size, sequence_length, embedding_dim)`:
   - Linear projections (Q, K, V): `(batch_size, sequence_length, embedding_dim)`
   - Split heads (view + transpose): `(batch_size, num_heads, sequence_length, head_dim)`
   - Scaled Dot-Product scores:      `(batch_size, num_heads, sequence_length, sequence_length)`
   - Softmax attention weights:      `(batch_size, num_heads, sequence_length, sequence_length)`
   - Per-head context (`weights @ V`): `(batch_size, num_heads, sequence_length, head_dim)`
   - Concatenate heads (transpose + view): `(batch_size, sequence_length, embedding_dim)`
   - Output projection (`out_proj`): `(batch_size, sequence_length, embedding_dim)`
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import scaled_dot_product_attention, validate_3d_embeddings


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Self-Attention with optional causal masking.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        causal: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embedding_dim // num_heads
        self.causal: bool = causal

        # Learnable linear projections for Query, Key, Value, and combined Output
        self.q_proj: nn.Linear = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.k_proj: nn.Linear = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.v_proj: nn.Linear = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.out_proj: nn.Linear = nn.Linear(embedding_dim, embedding_dim, bias=bias)

        # Stored after each forward pass for inspection / debugging:
        # shape (batch_size, num_heads, sequence_length, sequence_length)
        self.attention_weights: Optional[torch.Tensor] = None

    def project_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input `x` into combined Query, Key, and Value tensors of shape `(B, T, D)`."""
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension `embedding_dim` into `(num_heads, head_dim)` and
        transpose to shape `(batch_size, num_heads, sequence_length, head_dim)`.
        """
        batch_size, seq_len, _ = tensor.size()
        reshaped = tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return reshaped.transpose(1, 2)

    def combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Transpose `(batch_size, num_heads, sequence_length, head_dim)` back and
        concatenate heads into `(batch_size, sequence_length, embedding_dim)`.
        """
        batch_size, _, seq_len, _ = tensor.size()
        transposed = tensor.transpose(1, 2).contiguous()
        return transposed.view(batch_size, seq_len, self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head scaled dot-product self-attention on input `x` of shape `(B, T, D)`.
        """
        validate_3d_embeddings(x, self.embedding_dim)

        # 1. Project Q, K, V and split into multiple heads: (B, num_heads, T, head_dim)
        Q, K, V = self.project_qkv(x)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 2. Shared scaled dot-product attention per head
        context, weights = scaled_dot_product_attention(
            Q, K, V, head_dim=self.head_dim, causal=self.causal
        )
        self.attention_weights = weights

        # 3. Concatenate heads and apply final linear projection
        combined = self.combine_heads(context)
        return self.out_proj(combined)

"""
Phase 4: Single-Head Scaled Dot-Product Self-Attention for MiniGPT.

Educational Notes:
------------------
1. Why Self-Attention?
   In Phase 3, `InputEmbedding` gave each token a vector encoding its identity and its
   position, but each token vector is still isolated from the tokens around it.
   Self-Attention allows every token in a sequence to "look at" and gather context from
   other tokens in the sequence.

2. Query (Q), Key (K), and Value (V):
   Given input embeddings `X` of shape `(batch_size, seq_len, embedding_dim)`:
   - Query `Q = X @ W_q`: What this token is searching for.
   - Key   `K = X @ W_k`: What label/information each token advertises.
   - Value `V = X @ W_v`: The actual content each token passes along if attended to.

3. Scaled Dot-Product Attention Formula:
       scores  = (Q @ K^T) / sqrt(d_k)
       weights = softmax(scores, dim=-1)
       context = weights @ V
       output  = context @ W_o
   Dividing by `sqrt(d_k)` prevents dot products from growing large in magnitude as
   `d_k` increases, keeping softmax in a region with healthy gradients.

4. Causal Masking (Decoder-Only Autoregressive Property):
   When predicting the next token in GPT, token at position `t` must only attend to
   past and present positions `<= t` and never peek at future tokens `> t`.
   We enforce this with a lower-triangular mask, setting upper-triangular entries
   (`col > row`) of `scores` to `-inf` prior to softmax so their attention weights
   become `exp(-inf) = 0.0`.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def validate_3d_embeddings(x: torch.Tensor, expected_embedding_dim: int) -> None:
    """Validate that `x` is a 3D tensor `(batch_size, sequence_length, embedding_dim)`."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input x must be a torch.Tensor.")
    if x.dim() != 3:
        raise ValueError(
            f"Expected 3D input (batch_size, sequence_length, embedding_dim), got {x.dim()}D."
        )
    if x.size(-1) != expected_embedding_dim:
        raise ValueError(
            f"Expected embedding_dim {expected_embedding_dim}, got {x.size(-1)}."
        )


def apply_causal_mask_to_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Apply a lower-triangular causal mask to attention `scores` of shape `(..., seq_len, seq_len)`
    so future token positions (`col > row`) are filled with `-inf` before softmax.
    """
    seq_len = scores.size(-1)
    causal_mask = torch.tril(
        torch.ones((seq_len, seq_len), device=scores.device, dtype=torch.bool)
    )
    return scores.masked_fill(~causal_mask, float("-inf"))


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    head_dim: int,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention `(context, weights)` for 3D single-head or
    4D multi-head `(Q, K, V)` tensors.
    """
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
    if causal:
        scores = apply_causal_mask_to_scores(scores)
    weights = torch.softmax(scores, dim=-1)
    context = weights @ V
    return context, weights


class SelfAttention(nn.Module):
    """
    Single-head scaled dot-product self-attention with optional causal masking.
    """

    def __init__(
        self,
        embedding_dim: int,
        head_dim: Optional[int] = None,
        causal: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        resolved_head_dim = embedding_dim if head_dim is None else head_dim
        if resolved_head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {resolved_head_dim}.")

        self.embedding_dim: int = embedding_dim
        self.head_dim: int = resolved_head_dim
        self.causal: bool = causal

        # 1-3. Explicit learnable linear projections for Query, Key, and Value
        self.q_proj: nn.Linear = nn.Linear(embedding_dim, self.head_dim, bias=bias)
        self.k_proj: nn.Linear = nn.Linear(embedding_dim, self.head_dim, bias=bias)
        self.v_proj: nn.Linear = nn.Linear(embedding_dim, self.head_dim, bias=bias)

        # 8. Output linear projection back to embedding_dim
        self.out_proj: nn.Linear = nn.Linear(self.head_dim, embedding_dim, bias=bias)

        # Stored after each forward pass for inspection / debugging
        self.attention_weights: Optional[torch.Tensor] = None

    def project_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input tensor `x` into Query (Q), Key (K), and Value (V) tensors."""
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def compute_scaled_scores(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute `QK^T / sqrt(head_dim)`."""
        return (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

    def apply_causal_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply lower-triangular causal mask filling future positions with `-inf`."""
        return apply_causal_mask_to_scores(scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled dot-product self-attention on input `x` of shape `(B, T, D)`.
        """
        validate_3d_embeddings(x, self.embedding_dim)

        Q, K, V = self.project_qkv(x)
        context, weights = scaled_dot_product_attention(
            Q, K, V, head_dim=self.head_dim, causal=self.causal
        )
        self.attention_weights = weights
        return self.out_proj(context)

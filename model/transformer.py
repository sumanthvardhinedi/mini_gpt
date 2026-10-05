"""
Phase 6: Decoder-Style Pre-Norm Transformer Block for MiniGPT.

Educational Notes:
------------------
1. Why a Transformer Block?
   Multi-Head Self-Attention (Phase 5) allows tokens to exchange contextual information
   across sequence positions, while the Position-Wise Feed-Forward Network processes
   each token's representation individually through a non-linear MLP (`Linear -> GELU -> Linear`).
   Combining both inside a repeatable block gives the Transformer both *communication*
   (attention) and *computation* (feed-forward).

2. Why Pre-Norm (`LayerNorm` before sub-layers)?
   Modern GPT architectures use Pre-Norm:
       x = x + Dropout(MultiHeadAttention(LayerNorm(x)))
       x = x + Dropout(FeedForward(LayerNorm(x)))
   Normalizing activations before each sub-layer keeps gradients clean along the
   residual stream ("gradient highway"), stabilizing training in deep networks.

3. Why Residual Connections (`x + sublayer(x)`)?
   During backpropagation, the derivative of `x + f(x)` with respect to `x` is `1 + f'(x)`.
   The `+ 1` term ensures gradients can flow directly backward through all layers without
   vanishing, even at the start of training.

4. Why GELU in the Feed-Forward Network?
   The Feed-Forward Network expands `embedding_dim` to `hidden_dim` (typically `4 * embedding_dim`),
   applies a non-linear Gaussian Error Linear Unit (`nn.GELU`), and projects back to
   `embedding_dim`. Without a non-linear activation, stacking linear layers would collapse
   into a single linear transformation.
"""

from typing import Optional

import torch
import torch.nn as nn

from .attention import validate_3d_embeddings
from .multi_head_attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP):
        Linear(embedding_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> embedding_dim)
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        resolved_hidden_dim = 4 * embedding_dim if hidden_dim is None else hidden_dim
        if resolved_hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {resolved_hidden_dim}.")

        self.embedding_dim: int = embedding_dim
        self.hidden_dim: int = resolved_hidden_dim

        self.fc1: nn.Linear = nn.Linear(embedding_dim, self.hidden_dim, bias=bias)
        self.activation: nn.GELU = nn.GELU()
        self.fc2: nn.Linear = nn.Linear(self.hidden_dim, embedding_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise expansion, GELU non-linearity, and projection back."""
        return self.fc2(self.activation(self.fc1(x)))


class TransformerBlock(nn.Module):
    """
    Pre-Norm Decoder-Style Transformer Block:
        x = x + dropout1(attention(norm1(x)))
        x = x + dropout2(feed_forward(norm2(x)))
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
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

        resolved_hidden_dim = 4 * embedding_dim if hidden_dim is None else hidden_dim
        if resolved_hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {resolved_hidden_dim}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.hidden_dim: int = resolved_hidden_dim
        self.dropout_rate: float = dropout
        self.causal: bool = causal

        # Sub-layer 1: Pre-LayerNorm + Multi-Head Self-Attention + Dropout
        self.norm1: nn.LayerNorm = nn.LayerNorm(embedding_dim)
        self.attention: MultiHeadAttention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            causal=causal,
            bias=bias,
        )
        self.dropout1: nn.Dropout = nn.Dropout(p=dropout)

        # Sub-layer 2: Pre-LayerNorm + Position-Wise Feed-Forward + Dropout
        self.norm2: nn.LayerNorm = nn.LayerNorm(embedding_dim)
        self.feed_forward: FeedForward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=self.hidden_dim,
        )
        self.dropout2: nn.Dropout = nn.Dropout(p=dropout)

    @property
    def attention_weights(self) -> Optional[torch.Tensor]:
        """Expose attention weights from the internal MultiHeadAttention module."""
        return self.attention.attention_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input tensor `x` of shape `(batch_size, sequence_length, embedding_dim)`
        through the pre-norm Transformer block.
        """
        validate_3d_embeddings(x, self.embedding_dim)

        # 1. Pre-Norm Multi-Head Self-Attention with Residual Connection
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout1(attn_out)

        # 2. Pre-Norm Position-Wise Feed-Forward Network with Residual Connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_out)

        return x

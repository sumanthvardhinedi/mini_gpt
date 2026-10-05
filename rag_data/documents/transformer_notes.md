# Decoder Transformer Block Architecture

Phase 6 of MiniGPT (`model/transformer.py`) combines multi-head causal self-attention, layer normalization, residual connections, and a position-wise feed-forward network into a reusable `TransformerBlock`.

## Pre-LayerNorm Residual Structure

Modern GPT-style decoder blocks apply Pre-LayerNorm (`torch.nn.LayerNorm`) before each sub-layer rather than after, improving gradient stability across layers:
1. Attention sub-layer with residual skip connection:
   `x = x + Dropout(MultiHeadAttention(LayerNorm(x)))`
2. Feed-forward sub-layer with residual skip connection:
   `x = x + Dropout(FeedForward(LayerNorm(x)))`

## Position-Wise Feed-Forward Network

The `FeedForward` module applies two linear transformations separated by a `GELU` (Gaussian Error Linear Unit) non-linear activation:
`Linear(embedding_dim, hidden_dim) -> GELU() -> Linear(hidden_dim, embedding_dim)`.
In standard Transformer configurations, `hidden_dim` is typically `4 * embedding_dim`. Causal masking is enforced inside the attention sub-layer so the entire `TransformerBlock` preserves autoregressive ordering.

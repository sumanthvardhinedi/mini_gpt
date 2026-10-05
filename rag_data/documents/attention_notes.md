# Scaled Dot-Product Self-Attention and Causal Masking

Self-attention allows every token in a sequence to gather contextual information from earlier tokens in the sequence. Phase 4 (`model/attention.py`) and Phase 5 (`model/multi_head_attention.py`) implement causal self-attention from scratch in PyTorch.

## Query, Key, and Value Projections

Given input embeddings $X$ of shape `(batch_size, seq_len, embedding_dim)`, three learnable linear layers project $X$ into Query ($Q = XW_q$), Key ($K = XW_k$), and Value ($V = XW_v$) representations. Attention scores are computed via scaled dot-product:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
Scaling by $\sqrt{d_k}$ (the square root of the head dimension) prevents large dot products from pushing the softmax function into regions with vanishingly small gradients.

## What is Causal Masking?

Causal masking prevents a decoder-only language model from looking ahead at future tokens during training and generation. An upper-triangular mask above the main diagonal sets all future attention scores where key index $j >$ query index $i$ to $-\infty$ (`float("-inf")`) before applying softmax. After softmax normalization, attention weights for future positions become strictly `0.0`, guaranteeing that token predictions depend only on past and present tokens.

## Multi-Head Self-Attention

`MultiHeadAttention` splits `embedding_dim` into `num_heads` parallel attention heads, where each head has dimension `head_dim = embedding_dim // num_heads`. It requires `embedding_dim % num_heads == 0`.

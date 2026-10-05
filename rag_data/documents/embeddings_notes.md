# Token Embeddings and Positional Embeddings

Integer token IDs produced by a tokenizer are arbitrary lookup indices with no built-in geometric or semantic meaning. Phase 3 of MiniGPT introduces the embedding layer (`model/embeddings.py`) to map discrete token IDs into continuous vector representations.

## TokenEmbedding

`TokenEmbedding` wraps `torch.nn.Embedding(vocab_size, embedding_dim)` to map each token ID in a batch `(batch_size, seq_len)` to a trainable dense vector of shape `(batch_size, seq_len, embedding_dim)`. It validates that all input token IDs fall within the valid range `[0, vocab_size - 1]`.

## PositionalEmbedding and InputEmbedding

Because self-attention processes all sequence positions in parallel without recurrence, a Transformer has no inherent notion of token order unless positional information is injected. `PositionalEmbedding` uses a learned lookup table `torch.nn.Embedding(max_sequence_length, embedding_dim)` over sequence indices `0, 1, ..., seq_len - 1`. `InputEmbedding` adds the token embeddings and positional embeddings element-wise:
`input_embeddings = token_embeddings + positional_embeddings`.
If an input sequence exceeds `max_sequence_length`, `PositionalEmbedding` raises a `ValueError`.

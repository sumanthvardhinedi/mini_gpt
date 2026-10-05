"""
Phase 3: Token Embeddings and Positional Embeddings for MiniGPT.

Educational Notes:
------------------
1. Why do we need Token Embeddings?
   In Phase 1 and Phase 2, tokenizers converted text into discrete integer IDs
   (e.g., [3, 4, 0, 3, 2, 1]). However, integer IDs are arbitrary indices—ID 4 is not
   mathematically "between" ID 3 and ID 5 in meaning.
   `TokenEmbedding` is a learnable lookup table (`nn.Embedding(vocab_size, embedding_dim)`)
   that maps each discrete token ID to a continuous, dense vector of floating-point
   numbers of size `embedding_dim`. During training, backpropagation adjusts these
   vectors so tokens with similar meanings or grammatical roles move closer in vector space.

2. Why do we need Positional Embeddings?
   Unlike Recurrent Neural Networks (RNNs) that process tokens one by one sequentially,
   a Transformer processes all tokens in a sequence in parallel. Without position
   information, the sentence "The dog bit the man" and "The man bit the dog" would look
   like the exact same bag of token vectors!
   `PositionalEmbedding` (`nn.Embedding(max_sequence_length, embedding_dim)`) learns a
   unique vector for each sequence position (0, 1, 2, ..., max_sequence_length - 1).

3. How InputEmbedding combines them:
   For each token at sequence index `t`:
       input_representation[b, t] = token_embedding(token_ids[b, t]) + positional_embedding(t)
   Both tensors have shape `(batch_size, sequence_length, embedding_dim)`, so they are
   added element-wise to produce a unified representation encoding BOTH *what* each
   token is and *where* it appears in the sequence.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Maps discrete integer token IDs to dense continuous vectors of size `embedding_dim`.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, embedding_dim)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the underlying learnable embedding weight matrix `(vocab_size, embedding_dim)`."""
        return self.embedding.weight

    def _validate_token_ids(self, token_ids: torch.Tensor) -> None:
        """Validate tensor type, dimensions, and token ID range `[0, vocab_size - 1]`."""
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a torch.Tensor.")
        if token_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError(f"token_ids must be an integer tensor, got {token_ids.dtype}.")
        if token_ids.dim() not in (1, 2):
            raise ValueError(
                f"token_ids must be 1D (seq_len,) or 2D (batch_size, seq_len), got {token_ids.dim()}D."
            )
        if token_ids.numel() == 0:
            raise ValueError("token_ids cannot be empty.")
        if bool((token_ids < 0).any().item()) or bool((token_ids >= self.vocab_size).any().item()):
            min_id = int(token_ids.min().item())
            max_id = int(token_ids.max().item())
            raise ValueError(
                f"Token IDs out of range: expected IDs in [0, {self.vocab_size - 1}], "
                f"but found min={min_id}, max={max_id}."
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up dense embedding vectors for `token_ids`.

        Args:
            token_ids: Integer tensor of shape `(seq_len,)` or `(batch_size, seq_len)`.

        Returns:
            Dense tensor of shape `(seq_len, embedding_dim)` or `(batch_size, seq_len, embedding_dim)`.
        """
        self._validate_token_ids(token_ids)
        return self.embedding(token_ids.long())


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding table that produces a dense vector of size `embedding_dim`
    for each sequence position `0, 1, ..., seq_len - 1`.
    """

    def __init__(self, max_sequence_length: int, embedding_dim: int) -> None:
        super().__init__()
        if max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be positive, got {max_sequence_length}.")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        self.max_sequence_length: int = max_sequence_length
        self.embedding_dim: int = embedding_dim
        self.embedding: nn.Embedding = nn.Embedding(max_sequence_length, embedding_dim)

    @property
    def weight(self) -> torch.Tensor:
        """Expose the underlying learnable positional weight matrix `(max_sequence_length, embedding_dim)`."""
        return self.embedding.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute positional embeddings for a given input tensor.

        `inputs` can be:
        - 1D token IDs of shape `(seq_len,)` -> returns `(seq_len, embedding_dim)`
        - 2D batched token IDs of shape `(batch_size, seq_len)` -> returns `(batch_size, seq_len, embedding_dim)`
        - 3D token embeddings of shape `(batch_size, seq_len, embedding_dim)` -> returns `(batch_size, seq_len, embedding_dim)`
        """
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor.")
        if inputs.numel() == 0:
            raise ValueError("inputs cannot be empty.")

        if inputs.dim() == 1:
            seq_len = inputs.size(0)
            if seq_len > self.max_sequence_length:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_sequence_length {self.max_sequence_length}."
                )
            positions = torch.arange(seq_len, device=inputs.device, dtype=torch.long)
            return self.embedding(positions)

        if inputs.dim() in (2, 3):
            batch_size, seq_len = inputs.size(0), inputs.size(1)
            if seq_len > self.max_sequence_length:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_sequence_length {self.max_sequence_length}."
                )
            positions = torch.arange(seq_len, device=inputs.device, dtype=torch.long)
            pos_embeddings = self.embedding(positions)  # (seq_len, embedding_dim)
            return pos_embeddings.unsqueeze(0).expand(batch_size, seq_len, self.embedding_dim)

        raise ValueError(f"Unsupported input tensor dimension: {inputs.dim()}D.")


class InputEmbedding(nn.Module):
    """
    Combines TokenEmbedding and PositionalEmbedding:
        input_representation = token_embedding(token_ids) + positional_embedding(token_ids)
    """

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.embedding_dim: int = embedding_dim

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )
        self.positional_embedding = PositionalEmbedding(
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs into combined (token + positional) dense embeddings.

        Args:
            token_ids: Integer tensor of shape `(seq_len,)` or `(batch_size, seq_len)`.

        Returns:
            Combined tensor of shape `(seq_len, embedding_dim)` or
            `(batch_size, seq_len, embedding_dim)`.
        """
        seq_len = token_ids.size(-1) if isinstance(token_ids, torch.Tensor) and token_ids.dim() >= 1 else 0
        if seq_len > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_sequence_length {self.max_sequence_length}."
            )

        token_vecs = self.token_embedding(token_ids)
        pos_vecs = self.positional_embedding(token_ids)
        return token_vecs + pos_vecs

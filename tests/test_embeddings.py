"""
Pytest test suite for MiniGPT Phase 3: Token and Positional Embeddings.
Covers all 10 required test cases:
1. Token embedding output shape
2. Batch input
3. Positional embedding shape
4. Combined embedding shape
5. Same token produces the same embedding before training
6. Different token IDs can have different embeddings
7. Sequence positions receive different positional embeddings
8. Invalid token IDs
9. Sequence length validation
10. Gradient flow through embeddings
"""

import torch
from model.embeddings import InputEmbedding, PositionalEmbedding, TokenEmbedding


def test_token_embedding_output_shape() -> None:
    """1. Verify 1D token ID input produces (seq_len, embedding_dim) output and exposes weights."""
    embedding = TokenEmbedding(vocab_size=100, embedding_dim=64)
    token_ids = torch.tensor([1, 5, 20])

    vectors = embedding(token_ids)
    assert vectors.shape == torch.Size([3, 64])
    assert embedding.weight.shape == torch.Size([100, 64])


def test_token_embedding_batch_input() -> None:
    """2. Verify 2D batched token IDs produce (batch_size, seq_len, embedding_dim) output."""
    embedding = TokenEmbedding(vocab_size=100, embedding_dim=64)
    token_ids = torch.tensor([
        [1, 5, 20],
        [3, 7, 9],
    ])

    vectors = embedding(token_ids)
    assert vectors.shape == torch.Size([2, 3, 64])


def test_positional_embedding_shape() -> None:
    """3. Verify PositionalEmbedding output shapes for 1D and 2D inputs."""
    position_embedding = PositionalEmbedding(max_sequence_length=128, embedding_dim=64)

    single_seq = torch.tensor([1, 5, 20, 12])
    pos_single = position_embedding(single_seq)
    assert pos_single.shape == torch.Size([4, 64])

    batch_seq = torch.tensor([
        [1, 5, 20],
        [3, 7, 9],
    ])
    pos_batch = position_embedding(batch_seq)
    assert pos_batch.shape == torch.Size([2, 3, 64])
    assert position_embedding.weight.shape == torch.Size([128, 64])


def test_combined_embedding_shape() -> None:
    """4. Verify InputEmbedding combines token + positional embeddings with expected shape."""
    embeddings = InputEmbedding(
        vocab_size=100,
        max_sequence_length=128,
        embedding_dim=64,
    )
    token_ids = torch.tensor([
        [1, 5, 20],
        [3, 7, 9],
    ])

    output = embeddings(token_ids)
    assert output.shape == torch.Size([2, 3, 64])

    # Verify mathematical sum: output == token_embedding + positional_embedding
    expected = embeddings.token_embedding(token_ids) + embeddings.positional_embedding(token_ids)
    assert torch.allclose(output, expected)


def test_same_token_produces_same_embedding() -> None:
    """5. Verify identical token IDs produce the exact same token embedding vector."""
    torch.manual_seed(42)
    embedding = TokenEmbedding(vocab_size=50, embedding_dim=32)

    token_ids = torch.tensor([7, 12, 7, 7])
    vectors = embedding(token_ids)

    assert torch.equal(vectors[0], vectors[2])
    assert torch.equal(vectors[0], vectors[3])


def test_different_tokens_have_different_embeddings() -> None:
    """6. Verify distinct token IDs map to distinct embedding vectors."""
    torch.manual_seed(42)
    embedding = TokenEmbedding(vocab_size=50, embedding_dim=32)

    token_ids = torch.tensor([1, 2, 3])
    vectors = embedding(token_ids)

    assert not torch.equal(vectors[0], vectors[1])
    assert not torch.equal(vectors[1], vectors[2])
    assert not torch.equal(vectors[0], vectors[2])


def test_sequence_positions_receive_different_positional_embeddings() -> None:
    """7. Verify different sequence positions receive distinct positional vectors."""
    torch.manual_seed(42)
    embeddings = InputEmbedding(
        vocab_size=50,
        max_sequence_length=16,
        embedding_dim=32,
    )

    # Repeat the exact same token ID (5) across 4 positions
    repeated_tokens = torch.tensor([[5, 5, 5, 5]])
    pos_vecs = embeddings.positional_embedding(repeated_tokens)
    combined = embeddings(repeated_tokens)

    # Positional vectors at index 0, 1, 2, 3 must differ
    assert not torch.equal(pos_vecs[0, 0], pos_vecs[0, 1])
    assert not torch.equal(pos_vecs[0, 1], pos_vecs[0, 2])

    # Because positional vectors differ, combined representations of the same word at
    # different positions must also differ
    assert not torch.equal(combined[0, 0], combined[0, 1])
    assert not torch.equal(combined[0, 1], combined[0, 2])


def test_invalid_token_ids() -> None:
    """8. Verify negative or out-of-range token IDs raise ValueError."""
    embedding = TokenEmbedding(vocab_size=10, embedding_dim=16)

    # Negative token ID
    try:
        embedding(torch.tensor([0, -1, 3]))
        assert False, "Expected ValueError for negative token ID"
    except ValueError:
        pass

    # Token ID >= vocab_size (10 is out of range for vocab_size=10)
    try:
        embedding(torch.tensor([[1, 5, 10]]))
        assert False, "Expected ValueError for token ID >= vocab_size"
    except ValueError:
        pass


def test_sequence_length_validation() -> None:
    """9. Verify sequences longer than max_sequence_length raise ValueError."""
    pos_embedding = PositionalEmbedding(max_sequence_length=4, embedding_dim=16)
    input_embedding = InputEmbedding(vocab_size=20, max_sequence_length=4, embedding_dim=16)

    too_long_tokens = torch.tensor([[1, 2, 3, 4, 5]])  # length 5 > max_sequence_length 4

    try:
        pos_embedding(too_long_tokens)
        assert False, "Expected ValueError when sequence length exceeds max_sequence_length"
    except ValueError:
        pass

    try:
        input_embedding(too_long_tokens)
        assert False, "Expected ValueError when sequence length exceeds max_sequence_length"
    except ValueError:
        pass


def test_gradient_flow_through_embeddings() -> None:
    """10. Verify gradients flow backward into both token and positional embedding weights."""
    torch.manual_seed(42)
    embeddings = InputEmbedding(
        vocab_size=30,
        max_sequence_length=16,
        embedding_dim=16,
    )

    token_ids = torch.tensor([[2, 5, 8], [1, 5, 9]])
    output = embeddings(token_ids)

    # Compute a scalar loss and backpropagate
    loss = output.sum()
    loss.backward()

    token_grad = embeddings.token_embedding.weight.grad
    pos_grad = embeddings.positional_embedding.weight.grad

    assert token_grad is not None
    assert pos_grad is not None

    # Active token IDs (1, 2, 5, 8, 9) must have non-zero gradients
    for active_id in [1, 2, 5, 8, 9]:
        assert torch.any(token_grad[active_id] != 0)

    # Unused token ID (0) must have zero gradient
    assert torch.all(token_grad[0] == 0)

    # Active positions (0, 1, 2) must have non-zero gradients; position 3 must have zero gradient
    for pos_idx in [0, 1, 2]:
        assert torch.any(pos_grad[pos_idx] != 0)
    assert torch.all(pos_grad[3] == 0)

"""
Pytest test suite for MiniGPT Phase 7: MiniGPT Decoder-Only Language Model.
Covers all 15 required test cases:
1. Model construction
2. Forward-pass output shape
3. Batch input
4. Multiple Transformer layers
5. Correct vocabulary output dimension
6. Causal behavior
7. Different sequence lengths
8. Maximum sequence length validation
9. Invalid embedding_dim / num_heads
10. Gradient flow
11. Parameter counting
12. Weight tying
13. Deterministic output with dropout = 0
14. Model can move between CPU/device
15. Model can process a single token
"""

import torch
import torch.nn as nn
from model.embeddings import InputEmbedding
from model.gpt import MiniGPT
from model.transformer import TransformerBlock


def test_model_construction() -> None:
    """1. Verify MiniGPT initializes InputEmbedding, TransformerBlocks, LayerNorm, and LM head."""
    model = MiniGPT(
        vocab_size=100,
        max_sequence_length=32,
        embedding_dim=32,
        num_heads=4,
        num_layers=3,
        hidden_dim=128,
        dropout=0.0,
    )

    assert isinstance(model.embeddings, InputEmbedding)
    assert isinstance(model.blocks, nn.ModuleList)
    assert len(model.blocks) == 3
    assert all(isinstance(b, TransformerBlock) for b in model.blocks)
    assert isinstance(model.final_norm, nn.LayerNorm)
    assert isinstance(model.lm_head, nn.Linear)


def test_forward_pass_output_shape() -> None:
    """2. Verify forward pass maps (batch_size, seq_len) -> (batch_size, seq_len, vocab_size)."""
    torch.manual_seed(42)
    model = MiniGPT(
        vocab_size=120,
        max_sequence_length=32,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        hidden_dim=64,
        dropout=0.0,
    )
    input_ids = torch.tensor([[1, 5, 10, 20], [2, 6, 11, 21]])
    logits = model(input_ids)

    assert logits.shape == torch.Size([2, 4, 120])


def test_batch_input() -> None:
    """3. Verify batched input execution matches single-sample forward execution."""
    torch.manual_seed(42)
    model = MiniGPT(
        vocab_size=50,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
        dropout=0.0,
    )
    batch_ids = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ])

    batch_logits = model(batch_ids)
    single_logits = model(batch_ids[0:1])

    assert batch_logits.shape == torch.Size([3, 4, 50])
    assert torch.allclose(batch_logits[0:1], single_logits, atol=1e-5)


def test_multiple_transformer_layers() -> None:
    """4. Verify configuring different num_layers stacks the exact number of TransformerBlocks."""
    for n_layers in (1, 2, 4):
        model = MiniGPT(
            vocab_size=40,
            max_sequence_length=16,
            embedding_dim=16,
            num_heads=4,
            num_layers=n_layers,
            hidden_dim=32,
        )
        assert len(model.blocks) == n_layers
        out = model(torch.tensor([[1, 2, 3]]))
        assert out.shape == torch.Size([1, 3, 40])


def test_correct_vocabulary_output_dimension() -> None:
    """5. Verify output logit dimension equals vocab_size across configurations."""
    for vocab_size in (25, 80, 250):
        model = MiniGPT(
            vocab_size=vocab_size,
            max_sequence_length=16,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            hidden_dim=32,
        )
        logits = model(torch.tensor([[0, 1, 2]]))
        assert logits.size(-1) == vocab_size


def test_causal_behavior() -> None:
    """6. Verify changing a future token ID does not alter logits at earlier sequence positions."""
    torch.manual_seed(42)
    model = MiniGPT(
        vocab_size=50,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
        dropout=0.0,
    )

    ids_a = torch.tensor([[2, 4, 6, 8, 10]])
    ids_b = torch.tensor([[2, 4, 6, 8, 45]])  # Only final token (index 4) changed

    logits_a = model(ids_a)
    logits_b = model(ids_b)

    # Logits at positions 0, 1, 2, 3 must be identical
    assert torch.allclose(logits_a[:, :4, :], logits_b[:, :4, :], atol=1e-5)


def test_different_sequence_lengths() -> None:
    """7. Verify MiniGPT processes arbitrary sequence lengths up to max_sequence_length."""
    model = MiniGPT(
        vocab_size=40,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
    )

    for seq_len in (2, 5, 10, 16):
        input_ids = torch.arange(seq_len).unsqueeze(0)
        logits = model(input_ids)
        assert logits.shape == torch.Size([1, seq_len, 40])


def test_maximum_sequence_length_validation() -> None:
    """8. Verify input sequences longer than max_sequence_length raise ValueError."""
    model = MiniGPT(
        vocab_size=40,
        max_sequence_length=6,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
    )
    too_long = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])  # length 7 > 6

    try:
        model(too_long)
        assert False, "Expected ValueError when sequence length > max_sequence_length"
    except ValueError:
        pass


def test_invalid_embedding_dim_and_num_heads() -> None:
    """9. Verify invalid embedding_dim / num_heads and invalid hyperparameters raise ValueError."""
    try:
        MiniGPT(vocab_size=100, embedding_dim=64, num_heads=5)
        assert False, "Expected ValueError when embedding_dim % num_heads != 0"
    except ValueError:
        pass

    try:
        MiniGPT(vocab_size=0, embedding_dim=64, num_heads=4)
        assert False, "Expected ValueError for vocab_size <= 0"
    except ValueError:
        pass

    try:
        MiniGPT(vocab_size=100, embedding_dim=64, num_heads=4, num_layers=0)
        assert False, "Expected ValueError for num_layers <= 0"
    except ValueError:
        pass


def test_gradient_flow() -> None:
    """10. Verify gradients flow from output logits back through all layers and embeddings."""
    torch.manual_seed(42)
    model = MiniGPT(
        vocab_size=30,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
        dropout=0.0,
    )
    input_ids = torch.tensor([[1, 4, 7], [2, 5, 8]])

    logits = model(input_ids)
    loss = (logits * logits).sum()
    loss.backward()

    assert model.lm_head.weight.grad is not None
    assert model.final_norm.weight.grad is not None
    for block in model.blocks:
        assert block.attention.q_proj.weight.grad is not None
        assert block.feed_forward.fc1.weight.grad is not None
    assert model.embeddings.token_embedding.weight.grad is not None
    assert model.embeddings.positional_embedding.weight.grad is not None


def test_parameter_counting() -> None:
    """11. Verify num_parameters() accurately counts trainable parameters."""
    small_model = MiniGPT(
        vocab_size=50,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=32,
    )
    large_model = MiniGPT(
        vocab_size=50,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=3,
        hidden_dim=32,
    )

    small_params = small_model.num_parameters()
    large_params = large_model.num_parameters()

    assert small_params > 0
    assert large_params > small_params


def test_weight_tying() -> None:
    """12. Verify tie_weights=True shares the exact weight parameter between embeddings and lm_head."""
    untied = MiniGPT(
        vocab_size=50,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
        tie_weights=False,
    )
    tied = MiniGPT(
        vocab_size=50,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
        tie_weights=True,
    )

    assert tied.lm_head.weight is tied.embeddings.token_embedding.weight
    assert untied.lm_head.weight is not untied.embeddings.token_embedding.weight

    # Tied model saves exactly vocab_size * embedding_dim (50 * 16 = 800) parameters
    assert untied.num_parameters() - tied.num_parameters() == 50 * 16


def test_deterministic_output_with_dropout_zero() -> None:
    """13. Verify deterministic output logits when dropout = 0.0."""
    torch.manual_seed(42)
    model = MiniGPT(
        vocab_size=40,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
        dropout=0.0,
    )
    input_ids = torch.tensor([[3, 6, 9, 12]])

    out1 = model(input_ids)
    out2 = model(input_ids)
    assert torch.equal(out1, out2)


def test_model_can_move_between_cpu_device() -> None:
    """14. Verify model.to(device) works cleanly and executes forward pass on target device."""
    model = MiniGPT(
        vocab_size=30,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=32,
    )
    device = torch.device("cpu")
    moved_model = model.to(device)
    assert moved_model is model

    input_ids = torch.tensor([[1, 2, 3]], device="cpu")
    logits = moved_model(input_ids)
    assert logits.shape == torch.Size([1, 3, 30])


def test_model_can_process_single_token() -> None:
    """15. Verify MiniGPT can process a single-token sequence (batch_size=1, seq_len=1)."""
    model = MiniGPT(
        vocab_size=60,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        hidden_dim=32,
    )
    single_token = torch.tensor([[7]])
    logits = model(single_token)

    assert logits.shape == torch.Size([1, 1, 60])

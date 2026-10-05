"""
Pytest test suite for MiniGPT Phase 5: Multi-Head Self-Attention.
Covers all 12 required test cases:
1. Output shape
2. Correct number of heads
3. Correct head dimension
4. Q/K/V shapes
5. Attention weight shape
6. Attention weights sum to 1
7. Causal mask blocks future tokens for every head
8. Non-causal attention
9. Batch inputs
10. Invalid embedding/head dimensions
11. Gradient flow
12. Deterministic output
"""

import torch
from model.multi_head_attention import MultiHeadAttention


def test_output_shape() -> None:
    """1. Verify MultiHeadAttention preserves (batch_size, sequence_length, embedding_dim)."""
    torch.manual_seed(42)
    mha = MultiHeadAttention(embedding_dim=64, num_heads=4, causal=True)
    x = torch.randn(2, 6, 64)

    output = mha(x)
    assert output.shape == torch.Size([2, 6, 64])


def test_correct_number_of_heads() -> None:
    """2. Verify num_heads attribute and split-head dimension match configuration."""
    mha = MultiHeadAttention(embedding_dim=64, num_heads=8, causal=True)
    assert mha.num_heads == 8

    x = torch.randn(2, 5, 64)
    Q, _, _ = mha.project_qkv(x)
    q_heads = mha.split_heads(Q)
    assert q_heads.size(1) == 8


def test_correct_head_dimension() -> None:
    """3. Verify head_dim = embedding_dim // num_heads."""
    mha = MultiHeadAttention(embedding_dim=64, num_heads=4, causal=True)
    assert mha.head_dim == 16

    mha_two = MultiHeadAttention(embedding_dim=128, num_heads=8, causal=True)
    assert mha_two.head_dim == 16


def test_qkv_shapes() -> None:
    """4. Verify Q, K, V shapes before and after splitting into heads."""
    torch.manual_seed(42)
    mha = MultiHeadAttention(embedding_dim=64, num_heads=4, causal=True)
    x = torch.randn(2, 5, 64)

    Q, K, V = mha.project_qkv(x)
    assert Q.shape == torch.Size([2, 5, 64])
    assert K.shape == torch.Size([2, 5, 64])
    assert V.shape == torch.Size([2, 5, 64])

    q_split = mha.split_heads(Q)
    k_split = mha.split_heads(K)
    v_split = mha.split_heads(V)
    assert q_split.shape == torch.Size([2, 4, 5, 16])
    assert k_split.shape == torch.Size([2, 4, 5, 16])
    assert v_split.shape == torch.Size([2, 4, 5, 16])

    recombined = mha.combine_heads(q_split)
    assert recombined.shape == torch.Size([2, 5, 64])
    assert torch.allclose(Q, recombined, atol=1e-6)


def test_attention_weight_shape() -> None:
    """5. Verify attention_weights has shape (batch_size, num_heads, seq_len, seq_len)."""
    torch.manual_seed(42)
    mha = MultiHeadAttention(embedding_dim=64, num_heads=4, causal=True)
    x = torch.randn(3, 7, 64)

    _ = mha(x)
    weights = mha.attention_weights
    assert weights is not None
    assert weights.shape == torch.Size([3, 4, 7, 7])


def test_attention_weights_sum_to_one() -> None:
    """6. Verify attention weights sum to 1.0 along the key dimension for every head."""
    torch.manual_seed(42)
    x = torch.randn(2, 5, 32)

    for causal_flag in (True, False):
        mha = MultiHeadAttention(embedding_dim=32, num_heads=4, causal=causal_flag)
        _ = mha(x)
        weights = mha.attention_weights
        assert weights is not None
        row_sums = weights.sum(dim=-1)
        expected_ones = torch.ones(2, 4, 5)
        assert torch.allclose(row_sums, expected_ones, atol=1e-5)


def test_causal_mask_blocks_future_tokens_for_every_head() -> None:
    """7. Verify every attention head assigns 0.0 probability to future tokens (col > row)."""
    torch.manual_seed(42)
    mha = MultiHeadAttention(embedding_dim=32, num_heads=4, causal=True)
    x = torch.randn(2, 4, 32)

    out_orig = mha(x)
    weights = mha.attention_weights
    assert weights is not None

    for b in range(2):
        for h in range(4):
            for i in range(4):
                for j in range(i + 1, 4):
                    assert abs(float(weights[b, h, i, j].item())) < 1e-7

    # Perturbing the final token (pos 3) must not change output at positions 0, 1, 2
    x_perturbed = x.clone()
    x_perturbed[:, 3, :] = x_perturbed[:, 3, :] + 50.0
    out_perturbed = mha(x_perturbed)
    assert torch.allclose(out_orig[:, :3, :], out_perturbed[:, :3, :], atol=1e-5)


def test_non_causal_attention() -> None:
    """8. Verify non-causal MultiHeadAttention allows every head to attend to future tokens."""
    torch.manual_seed(42)
    mha = MultiHeadAttention(embedding_dim=32, num_heads=4, causal=False)
    x = torch.randn(1, 4, 32)

    _ = mha(x)
    weights = mha.attention_weights
    assert weights is not None

    for h in range(4):
        for i in range(4):
            for j in range(4):
                assert float(weights[0, h, i, j].item()) > 0.0


def test_batch_inputs() -> None:
    """9. Verify batched inputs produce consistent per-sample outputs."""
    torch.manual_seed(42)
    mha = MultiHeadAttention(embedding_dim=32, num_heads=4, causal=True)
    batch_x = torch.randn(3, 5, 32)

    batch_out = mha(batch_x)
    assert batch_out.shape == torch.Size([3, 5, 32])

    single_out = mha(batch_x[0:1])
    assert torch.allclose(batch_out[0:1], single_out, atol=1e-5)


def test_invalid_embedding_and_head_dimensions() -> None:
    """10. Verify invalid embedding_dim / num_heads combinations raise ValueError."""
    # embedding_dim not divisible by num_heads
    try:
        MultiHeadAttention(embedding_dim=64, num_heads=5)
        assert False, "Expected ValueError when embedding_dim % num_heads != 0"
    except ValueError:
        pass

    # Non-positive embedding_dim
    try:
        MultiHeadAttention(embedding_dim=0, num_heads=4)
        assert False, "Expected ValueError for embedding_dim <= 0"
    except ValueError:
        pass

    # Non-positive num_heads
    try:
        MultiHeadAttention(embedding_dim=64, num_heads=0)
        assert False, "Expected ValueError for num_heads <= 0"
    except ValueError:
        pass


def test_gradient_flow() -> None:
    """11. Verify gradients flow through Q, K, V, output projections, and input tensor."""
    torch.manual_seed(42)
    mha = MultiHeadAttention(embedding_dim=32, num_heads=4, causal=True)
    x = torch.randn(2, 4, 32, requires_grad=True)

    output = mha(x)
    loss = (output * output).sum()
    loss.backward()

    assert mha.q_proj.weight.grad is not None
    assert mha.k_proj.weight.grad is not None
    assert mha.v_proj.weight.grad is not None
    assert mha.out_proj.weight.grad is not None
    assert x.grad is not None

    assert torch.any(mha.q_proj.weight.grad != 0)
    assert torch.any(mha.k_proj.weight.grad != 0)
    assert torch.any(mha.v_proj.weight.grad != 0)
    assert torch.any(mha.out_proj.weight.grad != 0)
    assert torch.any(x.grad != 0)


def test_deterministic_output() -> None:
    """12. Verify identical inputs and weights yield identical outputs and attention weights."""
    torch.manual_seed(42)
    mha = MultiHeadAttention(embedding_dim=64, num_heads=4, causal=True)
    x = torch.randn(2, 5, 64)

    out1 = mha(x)
    weights1 = mha.attention_weights

    out2 = mha(x)
    weights2 = mha.attention_weights

    assert weights1 is not None and weights2 is not None
    assert torch.equal(out1, out2)
    assert torch.equal(weights1, weights2)

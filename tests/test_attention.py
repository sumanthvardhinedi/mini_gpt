"""
Pytest test suite for MiniGPT Phase 4: Self-Attention.
Covers all 10 required test cases:
1. Output shape
2. Q/K/V projection shapes
3. Attention weight shape
4. Attention weights sum to 1 across the last dimension
5. Causal mask prevents attention to future tokens
6. Non-causal attention can attend to all tokens
7. Scaling is applied correctly
8. Batch inputs work
9. Gradients flow through Q/K/V
10. Deterministic output when using the same input and weights
"""

import math
import torch
from model.attention import SelfAttention


def test_output_shape() -> None:
    """1. Verify SelfAttention preserves (batch_size, sequence_length, embedding_dim) shape."""
    torch.manual_seed(42)
    attention = SelfAttention(embedding_dim=64, causal=True)
    x = torch.randn(2, 5, 64)

    output = attention(x)
    assert output.shape == torch.Size([2, 5, 64])


def test_qkv_projection_shapes() -> None:
    """2. Verify Query, Key, and Value projections produce expected shapes."""
    torch.manual_seed(42)
    x = torch.randn(3, 7, 64)

    # Default head_dim == embedding_dim (64)
    attn_default = SelfAttention(embedding_dim=64)
    Q, K, V = attn_default.project_qkv(x)
    assert Q.shape == torch.Size([3, 7, 64])
    assert K.shape == torch.Size([3, 7, 64])
    assert V.shape == torch.Size([3, 7, 64])

    # Custom head_dim (32)
    attn_custom = SelfAttention(embedding_dim=64, head_dim=32)
    Q2, K2, V2 = attn_custom.project_qkv(x)
    assert Q2.shape == torch.Size([3, 7, 32])
    assert K2.shape == torch.Size([3, 7, 32])
    assert V2.shape == torch.Size([3, 7, 32])
    assert attn_custom(x).shape == torch.Size([3, 7, 64])


def test_attention_weight_shape() -> None:
    """3. Verify attention.attention_weights has shape (batch_size, seq_len, seq_len)."""
    torch.manual_seed(42)
    attention = SelfAttention(embedding_dim=32, causal=True)
    x = torch.randn(2, 6, 32)

    _ = attention(x)
    weights = attention.attention_weights
    assert weights is not None
    assert weights.shape == torch.Size([2, 6, 6])


def test_attention_weights_sum_to_one() -> None:
    """4. Verify attention weights sum to 1.0 across the last dimension for causal and non-causal."""
    torch.manual_seed(42)
    x = torch.randn(2, 5, 32)

    for causal_flag in (True, False):
        attention = SelfAttention(embedding_dim=32, causal=causal_flag)
        _ = attention(x)
        weights = attention.attention_weights
        assert weights is not None
        row_sums = weights.sum(dim=-1)
        ones = torch.ones(2, 5)
        assert torch.allclose(row_sums, ones, atol=1e-5)


def test_causal_mask_prevents_future_attention() -> None:
    """5. Verify causal mask sets all upper-triangular (future) attention weights to 0."""
    torch.manual_seed(42)
    attention = SelfAttention(embedding_dim=16, causal=True)
    x = torch.randn(2, 4, 16)

    out_original = attention(x)
    weights = attention.attention_weights
    assert weights is not None

    # Every position i must have 0.0 weight for future positions j > i
    seq_len = 4
    for b in range(2):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert abs(float(weights[b, i, j].item())) < 1e-7

    # Modifying the last token (index 3) must NOT alter outputs for earlier tokens (0, 1, 2)
    x_modified = x.clone()
    x_modified[:, 3, :] = x_modified[:, 3, :] + 100.0
    out_modified = attention(x_modified)
    assert torch.allclose(out_original[:, :3, :], out_modified[:, :3, :], atol=1e-5)


def test_non_causal_attention_attends_to_all_tokens() -> None:
    """6. Verify non-causal attention assigns strictly positive weight to future tokens."""
    torch.manual_seed(42)
    attention = SelfAttention(embedding_dim=16, causal=False)
    x = torch.randn(1, 4, 16)

    _ = attention(x)
    weights = attention.attention_weights
    assert weights is not None

    # All entries, including upper-triangular future tokens (j > i), must be > 0
    for i in range(4):
        for j in range(4):
            assert float(weights[0, i, j].item()) > 0.0


def test_scaling_is_applied_correctly() -> None:
    """7. Verify scores are divided by sqrt(head_dim)."""
    torch.manual_seed(42)
    head_dim = 16
    attention = SelfAttention(embedding_dim=16, head_dim=head_dim, causal=False)
    x = torch.randn(1, 3, 16)

    Q, K, _ = attention.project_qkv(x)
    raw_dot = Q @ K.transpose(-2, -1)
    expected_scaled = raw_dot / math.sqrt(head_dim)
    actual_scaled = attention.compute_scaled_scores(Q, K)

    assert torch.allclose(actual_scaled, expected_scaled, atol=1e-6)


def test_batch_inputs_work() -> None:
    """8. Verify batched execution matches individual sample execution."""
    torch.manual_seed(42)
    attention = SelfAttention(embedding_dim=16, causal=True)
    batch_x = torch.randn(3, 4, 16)

    batch_out = attention(batch_x)
    assert batch_out.shape == torch.Size([3, 4, 16])

    # First item in batch must match running sample 0 alone with batch_size=1
    single_out = attention(batch_x[0:1])
    assert torch.allclose(batch_out[0:1], single_out, atol=1e-5)


def test_gradients_flow_through_qkv() -> None:
    """9. Verify gradients flow backward into Q, K, V, output projection weights, and input x."""
    torch.manual_seed(42)
    attention = SelfAttention(embedding_dim=16, causal=True)
    x = torch.randn(2, 4, 16, requires_grad=True)

    output = attention(x)
    loss = (output * output).sum()
    loss.backward()

    assert attention.q_proj.weight.grad is not None
    assert attention.k_proj.weight.grad is not None
    assert attention.v_proj.weight.grad is not None
    assert attention.out_proj.weight.grad is not None
    assert x.grad is not None

    assert torch.any(attention.q_proj.weight.grad != 0)
    assert torch.any(attention.k_proj.weight.grad != 0)
    assert torch.any(attention.v_proj.weight.grad != 0)
    assert torch.any(attention.out_proj.weight.grad != 0)
    assert torch.any(x.grad != 0)


def test_deterministic_output() -> None:
    """10. Verify identical input and weights produce deterministic outputs and weights."""
    torch.manual_seed(42)
    attention = SelfAttention(embedding_dim=32, causal=True)
    x = torch.randn(2, 5, 32)

    out1 = attention(x)
    weights1 = attention.attention_weights

    out2 = attention(x)
    weights2 = attention.attention_weights

    assert weights1 is not None and weights2 is not None
    assert torch.equal(out1, out2)
    assert torch.equal(weights1, weights2)

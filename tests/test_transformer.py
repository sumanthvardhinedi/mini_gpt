"""
Pytest test suite for MiniGPT Phase 6: Transformer Block.
Covers all 12 required test cases:
1. Output shape
2. Batch inputs
3. Feed-forward hidden dimension
4. LayerNorm presence
5. Residual connections
6. GELU activation
7. Dropout configuration
8. Gradient flow through the complete block
9. Causal attention
10. Different sequence lengths
11. Deterministic output when dropout = 0
12. Invalid configuration handling
"""

import torch
import torch.nn as nn
from model.transformer import FeedForward, TransformerBlock


def test_output_shape() -> None:
    """1. Verify TransformerBlock preserves (batch_size, sequence_length, embedding_dim)."""
    torch.manual_seed(42)
    block = TransformerBlock(
        embedding_dim=64,
        num_heads=4,
        hidden_dim=256,
        dropout=0.0,
        causal=True,
    )
    x = torch.randn(2, 6, 64)

    output = block(x)
    assert output.shape == torch.Size([2, 6, 64])


def test_batch_inputs() -> None:
    """2. Verify batched forward pass matches individual sequence execution."""
    torch.manual_seed(42)
    block = TransformerBlock(embedding_dim=32, num_heads=4, dropout=0.0, causal=True)
    batch_x = torch.randn(3, 5, 32)

    batch_out = block(batch_x)
    assert batch_out.shape == torch.Size([3, 5, 32])

    single_out = block(batch_x[0:1])
    assert torch.allclose(batch_out[0:1], single_out, atol=1e-5)


def test_feed_forward_hidden_dimension() -> None:
    """3. Verify default hidden_dim = 4 * embedding_dim and custom hidden_dim."""
    # Default 4x expansion
    block_default = TransformerBlock(embedding_dim=64, num_heads=4)
    assert block_default.hidden_dim == 256
    assert block_default.feed_forward.hidden_dim == 256
    assert block_default.feed_forward.fc1.weight.shape == torch.Size([256, 64])
    assert block_default.feed_forward.fc2.weight.shape == torch.Size([64, 256])

    # Custom hidden_dim
    block_custom = TransformerBlock(embedding_dim=64, num_heads=4, hidden_dim=128)
    assert block_custom.hidden_dim == 128
    assert block_custom.feed_forward.fc1.weight.shape == torch.Size([128, 64])
    assert block_custom.feed_forward.fc2.weight.shape == torch.Size([64, 128])


def test_layernorm_presence() -> None:
    """4. Verify norm1 and norm2 are nn.LayerNorm instances and normalize features."""
    torch.manual_seed(42)
    block = TransformerBlock(embedding_dim=32, num_heads=4)

    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)

    x = torch.randn(2, 4, 32) * 5.0 + 10.0
    normed = block.norm1(x)
    assert normed.shape == torch.Size([2, 4, 32])


def test_residual_connections() -> None:
    """5. Verify residual pathways pass input x unchanged when sub-layer outputs are zero."""
    torch.manual_seed(42)
    block = TransformerBlock(embedding_dim=16, num_heads=4, dropout=0.0)

    # Zero out the output projections of both attention and feed-forward sub-layers
    nn.init.zeros_(block.attention.out_proj.weight)
    nn.init.zeros_(block.feed_forward.fc2.weight)
    if block.feed_forward.fc2.bias is not None:
        nn.init.zeros_(block.feed_forward.fc2.bias)

    x = torch.randn(2, 4, 16)
    output = block(x)

    # With both sub-layers contributing 0, output must equal input x via residual connections
    assert torch.allclose(output, x, atol=1e-6)


def test_gelu_activation() -> None:
    """6. Verify FeedForward uses nn.GELU and exhibits expected non-linear behavior."""
    ff = FeedForward(embedding_dim=16, hidden_dim=64)
    assert isinstance(ff.activation, nn.GELU)

    test_vals = torch.tensor([-2.0, 0.0, 2.0])
    activated = ff.activation(test_vals)

    # GELU(0) == 0, GELU(2.0) > 1.9, GELU(-2.0) is small negative (~ -0.0455)
    assert abs(float(activated[1].item())) < 1e-6
    assert float(activated[2].item()) > 1.9
    assert -0.1 < float(activated[0].item()) < 0.0


def test_dropout_configuration() -> None:
    """7. Verify dropout modules are configured with the requested probability."""
    block = TransformerBlock(embedding_dim=32, num_heads=4, dropout=0.25)
    assert isinstance(block.dropout1, nn.Dropout)
    assert isinstance(block.dropout2, nn.Dropout)
    assert block.dropout1.p == 0.25
    assert block.dropout2.p == 0.25


def test_gradient_flow_through_complete_block() -> None:
    """8. Verify gradients flow backward through attention, LayerNorms, FFN, and input x."""
    torch.manual_seed(42)
    block = TransformerBlock(embedding_dim=16, num_heads=4, hidden_dim=32, dropout=0.0)
    x = torch.randn(2, 4, 16, requires_grad=True)

    output = block(x)
    loss = (output * output).sum()
    loss.backward()

    assert block.norm1.weight.grad is not None
    assert block.norm2.weight.grad is not None
    assert block.attention.q_proj.weight.grad is not None
    assert block.attention.k_proj.weight.grad is not None
    assert block.attention.v_proj.weight.grad is not None
    assert block.attention.out_proj.weight.grad is not None
    assert block.feed_forward.fc1.weight.grad is not None
    assert block.feed_forward.fc2.weight.grad is not None
    assert x.grad is not None

    assert torch.any(block.attention.q_proj.weight.grad != 0)
    assert torch.any(block.feed_forward.fc1.weight.grad != 0)
    assert torch.any(block.feed_forward.fc2.weight.grad != 0)
    assert torch.any(x.grad != 0)


def test_causal_attention() -> None:
    """9. Verify causal=True prevents future tokens from influencing past outputs."""
    torch.manual_seed(42)
    block = TransformerBlock(embedding_dim=16, num_heads=4, dropout=0.0, causal=True)
    x = torch.randn(1, 5, 16)

    out1 = block(x)

    # Modify the last token at index 4
    x_mod = x.clone()
    x_mod[:, 4, :] = x_mod[:, 4, :] + 100.0
    out2 = block(x_mod)

    # Outputs at positions 0, 1, 2, 3 must be completely unaffected
    assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5)


def test_different_sequence_lengths() -> None:
    """10. Verify TransformerBlock processes variable sequence lengths dynamically."""
    torch.manual_seed(42)
    block = TransformerBlock(embedding_dim=32, num_heads=4, dropout=0.0)

    for seq_len in (1, 4, 9, 16):
        x = torch.randn(2, seq_len, 32)
        out = block(x)
        assert out.shape == torch.Size([2, seq_len, 32])


def test_deterministic_output_when_dropout_zero() -> None:
    """11. Verify deterministic output across repeated calls when dropout = 0.0."""
    torch.manual_seed(42)
    block = TransformerBlock(embedding_dim=32, num_heads=4, dropout=0.0)
    x = torch.randn(2, 5, 32)

    out1 = block(x)
    out2 = block(x)
    assert torch.equal(out1, out2)


def test_invalid_configuration_handling() -> None:
    """12. Verify invalid parameters raise ValueError."""
    # embedding_dim not divisible by num_heads
    try:
        TransformerBlock(embedding_dim=64, num_heads=6)
        assert False, "Expected ValueError when embedding_dim % num_heads != 0"
    except ValueError:
        pass

    # Invalid hidden_dim
    try:
        TransformerBlock(embedding_dim=64, num_heads=4, hidden_dim=0)
        assert False, "Expected ValueError for hidden_dim <= 0"
    except ValueError:
        pass

    # Invalid dropout (< 0 or >= 1)
    for bad_dropout in (-0.1, 1.0, 1.5):
        try:
            TransformerBlock(embedding_dim=64, num_heads=4, dropout=bad_dropout)
            assert False, f"Expected ValueError for dropout={bad_dropout}"
        except ValueError:
            pass

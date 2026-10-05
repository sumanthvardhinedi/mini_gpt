"""
Pytest test suite for MiniGPT Phase 8: Loss, Training Loop, Evaluation, and Checkpointing.
"""

from pathlib import Path
import tempfile
import torch
from model.gpt import MiniGPT
from tokenizer.basic_tokenizer import BasicTokenizer
from training.dataset import LanguageModelDataset, create_dataloaders, create_datasets
from training.train import (
    TrainingConfig,
    compute_loss,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_model,
    train_step,
)


def test_compute_loss_shape_and_value() -> None:
    """Verify compute_loss flattens (B, T, V) logits and (B, T) targets and returns scalar loss."""
    torch.manual_seed(42)
    logits = torch.randn(2, 4, 10, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    loss = compute_loss(logits, targets)
    assert loss.dim() == 0
    assert float(loss.item()) > 0.0

    loss.backward()
    assert logits.grad is not None


def test_single_train_step_updates_parameters() -> None:
    """Verify a single train_step updates model parameters and reduces loss on that batch."""
    torch.manual_seed(42)
    model = MiniGPT(
        vocab_size=20,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=32,
        dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

    input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    target_ids = torch.tensor([[2, 3, 4, 5], [3, 4, 5, 6]])

    initial_weight = model.lm_head.weight.clone()
    loss_before = float(compute_loss(model(input_ids), target_ids).item())

    step_loss = train_step(model, optimizer, input_ids, target_ids, device="cpu")
    assert abs(step_loss - loss_before) < 1e-5

    # Weights must have changed after optimizer.step()
    assert not torch.equal(model.lm_head.weight, initial_weight)

    # Re-evaluating on the same batch after the step should show lower loss
    loss_after = float(compute_loss(model(input_ids), target_ids).item())
    assert loss_after < loss_before


def test_training_loop_reduces_loss_and_tracks_history() -> None:
    """Verify train_model runs over multiple epochs, tracks train/val loss, and reduces loss."""
    torch.manual_seed(42)
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample.txt"
    text = sample_path.read_text(encoding="utf-8")

    tokenizer = BasicTokenizer(text)
    token_ids = tokenizer.encode(text)

    train_ds, val_ds = create_datasets(token_ids, sequence_length=8, train_ratio=0.85)
    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds, batch_size=8, shuffle_train=False
    )

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=32,
        dropout=0.0,
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=4,
        learning_rate=0.03,
        device="cpu",
    )

    assert len(history["train_loss"]) == 4
    assert len(history["val_loss"]) == 4
    # Final training loss must be lower than initial epoch training loss
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_evaluate_switches_mode_and_restores_training_state() -> None:
    """Verify evaluate() uses eval mode and restores model.training afterward."""
    torch.manual_seed(42)
    dataset = LanguageModelDataset(list(range(20)), sequence_length=4)
    loader, _ = create_dataloaders(dataset, batch_size=4, shuffle_train=False)

    model = MiniGPT(
        vocab_size=25,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=32,
    )
    model.train()
    assert model.training is True

    val_loss = evaluate(model, loader, device="cpu")
    assert isinstance(val_loss, float)
    assert val_loss > 0.0
    # Model must be restored to training mode
    assert model.training is True


def test_save_and_load_checkpoint() -> None:
    """Verify save_checkpoint and load_checkpoint persist model, optimizer, epoch, and loss."""
    torch.manual_seed(42)
    model_a = MiniGPT(
        vocab_size=25,
        max_sequence_length=16,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=32,
    )
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=0.01)

    inp = torch.tensor([[1, 2, 3, 4]])
    tgt = torch.tensor([[2, 3, 4, 5]])
    loss_val = train_step(model_a, opt_a, inp, tgt)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_file = Path(tmp_dir) / "minigpt_epoch1.pt"
        save_checkpoint(model_a, opt_a, epoch=1, loss=loss_val, path=ckpt_file)
        assert ckpt_file.exists()

        # Create a fresh model with different weights and load checkpoint into it
        torch.manual_seed(999)
        model_b = MiniGPT(
            vocab_size=25,
            max_sequence_length=16,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            hidden_dim=32,
        )
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=0.01)

        loaded = load_checkpoint(model_b, opt_b, path=ckpt_file, device="cpu")
        assert loaded["epoch"] == 1
        assert abs(loaded["loss"] - loss_val) < 1e-6

        out_a = model_a(inp)
        out_b = model_b(inp)
        assert torch.allclose(out_a, out_b, atol=1e-6)


def test_training_config_defaults() -> None:
    """Verify TrainingConfig provides small CPU-ready defaults."""
    cfg = TrainingConfig(vocab_size=64)
    assert cfg.vocab_size == 64
    assert cfg.max_sequence_length == 64
    assert cfg.embedding_dim == 64
    assert cfg.num_heads == 4
    assert cfg.device == "cpu"

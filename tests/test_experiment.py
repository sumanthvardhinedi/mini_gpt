"""
Pytest test suite for MiniGPT Phase 10: Production-Quality Training Experiment.
Covers all 10 required test cases:
1. Configuration loading
2. Reproducible seeding
3. Metrics calculation
4. Perplexity calculation
5. Training metrics persistence
6. Best-checkpoint selection
7. Checkpoint metadata
8. Tokenizer state persistence
9. Experiment configuration & end-to-end execution
10. CLI configuration loading
"""

import json
import math
from pathlib import Path
import random
import tempfile
import torch
from model.gpt import MiniGPT
from scripts.train import parse_args
from tokenizer.bpe_tokenizer import BPETokenizer
from training.experiment import (
    ExperimentConfig,
    load_experiment_checkpoint,
    load_experiment_config,
    restore_tokenizer_state,
    run_experiment,
    save_experiment_checkpoint,
    serialize_tokenizer_state,
    set_seed,
)
from training.metrics import MetricsTracker, compute_perplexity


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_configuration_loading() -> None:
    """1. Verify configs/small_gpt.json loads cleanly with all required hyperparameters."""
    config_path = PROJECT_ROOT / "configs" / "small_gpt.json"
    assert config_path.exists()

    cfg = load_experiment_config(config_path)
    assert cfg.vocab_size > 0
    assert cfg.sequence_length > 0
    assert cfg.embedding_dim > 0
    assert cfg.num_heads > 0
    assert cfg.num_layers > 0
    assert cfg.hidden_dim > 0
    assert 0.0 <= cfg.dropout < 1.0
    assert cfg.batch_size > 0
    assert cfg.learning_rate > 0.0
    assert cfg.epochs > 0
    assert cfg.weight_decay >= 0.0
    assert 0.0 < cfg.train_ratio < 1.0
    assert isinstance(cfg.seed, int)
    assert len(cfg.checkpoint_path) > 0


def test_reproducible_seeding() -> None:
    """2. Verify set_seed produces identical Python random and PyTorch tensors across runs."""
    set_seed(1234)
    py_a = [random.random() for _ in range(5)]
    tensor_a = torch.randn(2, 4)

    set_seed(1234)
    py_b = [random.random() for _ in range(5)]
    tensor_b = torch.randn(2, 4)

    assert py_a == py_b
    assert torch.allclose(tensor_a, tensor_b, atol=1e-7)


def test_metrics_calculation() -> None:
    """3. Verify MetricsTracker records epoch losses, perplexities, and best epoch."""
    tracker = MetricsTracker()
    entry_1, is_best_1 = tracker.record_epoch(epoch=1, train_loss=4.0, val_loss=4.2)
    entry_2, is_best_2 = tracker.record_epoch(epoch=2, train_loss=3.5, val_loss=3.8)
    entry_3, is_best_3 = tracker.record_epoch(epoch=3, train_loss=3.2, val_loss=3.9)

    assert is_best_1 is True
    assert is_best_2 is True
    assert is_best_3 is False
    assert tracker.best_epoch == 2
    assert abs(tracker.best_val_loss - 3.8) < 1e-6
    assert abs(entry_1.train_perplexity - math.exp(4.0)) < 1e-5
    assert abs(entry_2.val_perplexity - math.exp(3.8)) < 1e-5
    assert len(tracker.epochs) == 3


def test_perplexity_calculation() -> None:
    """4. Verify compute_perplexity computes exp(loss) and handles overflow and invalid inputs safely."""
    assert abs(compute_perplexity(0.0) - 1.0) < 1e-9
    assert abs(compute_perplexity(2.0) - math.exp(2.0)) < 1e-6

    # Large loss should be clamped safely without raising OverflowError
    huge_ppl = compute_perplexity(5000.0)
    assert math.isfinite(huge_ppl)
    assert abs(huge_ppl - math.exp(20.0)) < 1e-3

    # Negative or NaN loss must raise ValueError
    for bad_loss in [-1.0, float("nan")]:
        try:
            compute_perplexity(bad_loss)
            assert False, f"Expected ValueError for bad_loss={bad_loss}"
        except ValueError:
            pass


def test_training_metrics_persistence() -> None:
    """5. Verify MetricsTracker saves and loads the exact JSON format in training_metrics.json."""
    tracker = MetricsTracker()
    tracker.record_epoch(epoch=1, train_loss=5.2, val_loss=5.4)
    tracker.record_epoch(epoch=2, train_loss=4.6, val_loss=4.9)

    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics_file = Path(tmp_dir) / "training_metrics.json"
        tracker.save(metrics_file)
        assert metrics_file.exists()

        raw = json.loads(metrics_file.read_text(encoding="utf-8"))
        assert "epochs" in raw
        assert len(raw["epochs"]) == 2
        first = raw["epochs"][0]
        assert set(first.keys()) == {
            "epoch",
            "train_loss",
            "val_loss",
            "train_perplexity",
            "val_perplexity",
        }
        assert first["epoch"] == 1

        reloaded = MetricsTracker.load(metrics_file)
        assert len(reloaded.epochs) == 2
        assert reloaded.best_epoch == 2


def test_best_checkpoint_selection() -> None:
    """6. Verify best checkpoint is updated only when validation loss improves."""
    tracker = MetricsTracker()
    with tempfile.TemporaryDirectory() as tmp_dir:
        best_path = Path(tmp_dir) / "best.pt"
        cfg = ExperimentConfig(vocab_size=30, embedding_dim=8, num_heads=2, hidden_dim=16)
        tok = BPETokenizer(vocab_size=25, training_text="hello world hello gpt")
        model = MiniGPT(
            vocab_size=25,
            max_sequence_length=16,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            hidden_dim=16,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)

        simulated_epochs = [
            (1, 4.5, 4.4),
            (2, 4.0, 3.9),  # Best val_loss = 3.9 at epoch 2
            (3, 3.7, 4.1),  # Worse val_loss = 4.1 at epoch 3
        ]
        for ep, tr_l, va_l in simulated_epochs:
            _, is_best = tracker.record_epoch(ep, tr_l, va_l)
            if is_best:
                save_experiment_checkpoint(best_path, model, opt, ep, tr_l, va_l, cfg, tok)

        ckpt, _ = load_experiment_checkpoint(best_path)
        assert ckpt["epoch"] == 2
        assert abs(ckpt["val_loss"] - 3.9) < 1e-6


def test_checkpoint_metadata() -> None:
    """7. Verify saved checkpoint contains all required metadata fields."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "best_model.pt"
        cfg = ExperimentConfig(vocab_size=32, embedding_dim=8, num_heads=2, hidden_dim=16)
        tok = BPETokenizer(vocab_size=28, training_text="language models predict next tokens")
        model = MiniGPT(
            vocab_size=28,
            max_sequence_length=16,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            hidden_dim=16,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

        save_experiment_checkpoint(
            path=ckpt_path,
            model=model,
            optimizer=opt,
            epoch=3,
            train_loss=3.25,
            val_loss=3.40,
            config=cfg,
            tokenizer=tok,
        )

        loaded, _ = load_experiment_checkpoint(ckpt_path)
        required_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "train_loss",
            "val_loss",
            "config",
            "tokenizer_vocab",
            "tokenizer_merges",
        }
        assert required_keys.issubset(set(loaded.keys()))
        assert loaded["epoch"] == 3
        assert abs(loaded["train_loss"] - 3.25) < 1e-6
        assert abs(loaded["val_loss"] - 3.40) < 1e-6
        assert loaded["config"]["vocab_size"] == 32


def test_tokenizer_state_persistence() -> None:
    """8. Verify BPE vocabulary and merge rules survive serialization and encode/decode identically."""
    sample_text = "Subword tokenization with Byte Pair Encoding merges frequent pairs."
    original_tok = BPETokenizer(vocab_size=45, training_text=sample_text)

    state = serialize_tokenizer_state(original_tok)
    restored_tok = restore_tokenizer_state(state)

    assert restored_tok.str_to_int == original_tok.str_to_int
    assert restored_tok.merges == original_tok.merges

    test_phrase = "Byte Pair Encoding tokenization"
    encoded_orig = original_tok.encode(test_phrase)
    encoded_restored = restored_tok.encode(test_phrase)
    assert encoded_orig == encoded_restored
    assert restored_tok.decode(encoded_restored) == test_phrase


def test_experiment_configuration_and_runner() -> None:
    """9. Verify ExperimentConfig validation and end-to-end run_experiment reduces loss."""
    try:
        ExperimentConfig(vocab_size=32, embedding_dim=10, num_heads=4)
        assert False, "Expected ValueError when embedding_dim is not divisible by num_heads"
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cfg = ExperimentConfig(
            vocab_size=64,
            sequence_length=8,
            max_sequence_length=16,
            embedding_dim=16,
            num_heads=4,
            num_layers=1,
            hidden_dim=32,
            dropout=0.0,
            batch_size=12,
            learning_rate=0.04,
            epochs=3,
            weight_decay=0.01,
            train_ratio=0.85,
            seed=42,
            max_tokens=160,
            data_path=str(PROJECT_ROOT / "data" / "training.txt"),
            checkpoint_path=str(tmp_path / "best.pt"),
            best_checkpoint_path=str(tmp_path / "best.pt"),
            final_checkpoint_path=str(tmp_path / "final.pt"),
            metrics_path=str(tmp_path / "training_metrics.json"),
        )

        summary = run_experiment(cfg, base_dir=PROJECT_ROOT)
        assert summary["final_train_loss"] < summary["initial_train_loss"]
        assert Path(summary["best_checkpoint_path"]).exists()
        assert Path(summary["final_checkpoint_path"]).exists()
        assert Path(summary["metrics_path"]).exists()


def test_cli_configuration_loading() -> None:
    """10. Verify scripts/train.py CLI parses default and custom --config paths."""
    default_args = parse_args([])
    assert default_args.config == "configs/small_gpt.json"

    custom_args = parse_args(["--config", "configs/small_gpt.json"])
    assert custom_args.config == "configs/small_gpt.json"
    loaded_cfg = load_experiment_config(PROJECT_ROOT / custom_args.config)
    assert isinstance(loaded_cfg, ExperimentConfig)

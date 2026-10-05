"""
Pytest test suite for MiniGPT Phase 11: Evaluation, Visualization, and Generation Demo.
Covers all 10 required test cases:
1. Checkpoint loading
2. Model reconstruction
3. Validation evaluation
4. Perplexity calculation
5. Parameter counting
6. Deterministic greedy generation
7. Seeded sampling reproducibility
8. Evaluation output generation
9. Training curve generation
10. CLI argument handling
"""

from pathlib import Path
import tempfile
import torch
from evaluation.evaluate import (
    DEFAULT_PROMPTS,
    evaluate_checkpoint,
    generate_from_checkpoint,
    load_model_from_checkpoint,
    run_generation_suite,
)
from evaluation.visualize import plot_training_curves
from model.gpt import MiniGPT
from scripts.evaluate import parse_args
from training.metrics import compute_perplexity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
METRICS_JSON = PROJECT_ROOT / "checkpoints" / "training_metrics.json"


def test_checkpoint_loading() -> None:
    """1. Verify best_model.pt loads model, tokenizer, config, and metadata cleanly."""
    assert BEST_CHECKPOINT.exists(), "Expected checkpoints/best_model.pt to exist"

    model, tokenizer, config, ckpt = load_model_from_checkpoint(
        checkpoint_path=BEST_CHECKPOINT,
        base_dir=PROJECT_ROOT,
    )
    assert isinstance(model, MiniGPT)
    assert model.training is False
    assert tokenizer.current_vocab_size == config.vocab_size
    assert "epoch" in ckpt
    assert "val_loss" in ckpt


def test_model_reconstruction() -> None:
    """2. Verify reconstructed model architecture and weights match saved checkpoint state."""
    model, _, config, ckpt = load_model_from_checkpoint(
        checkpoint_path=BEST_CHECKPOINT,
        base_dir=PROJECT_ROOT,
    )

    assert model.vocab_size == config.vocab_size
    assert model.max_sequence_length == config.max_sequence_length
    assert model.embedding_dim == config.embedding_dim
    assert model.num_heads == config.num_heads
    assert model.num_layers == config.num_layers
    assert model.hidden_dim == config.hidden_dim

    saved_state = ckpt["model_state_dict"]
    current_state = model.state_dict()
    assert set(saved_state.keys()) == set(current_state.keys())
    for key in saved_state:
        assert torch.allclose(saved_state[key], current_state[key], atol=1e-6)


def test_validation_evaluation() -> None:
    """3. Verify evaluate_checkpoint computes validation loss without retraining."""
    summary = evaluate_checkpoint(
        checkpoint_path=BEST_CHECKPOINT,
        base_dir=PROJECT_ROOT,
    )
    assert summary["val_loss"] > 0.0
    assert abs(summary["val_loss"] - summary["checkpoint_val_loss"]) < 1e-5
    assert summary["model"].training is False


def test_perplexity_calculation() -> None:
    """4. Verify validation perplexity matches exp(val_loss)."""
    summary = evaluate_checkpoint(
        checkpoint_path=BEST_CHECKPOINT,
        base_dir=PROJECT_ROOT,
    )
    expected_ppl = compute_perplexity(summary["val_loss"])
    assert abs(summary["val_perplexity"] - expected_ppl) < 1e-6
    assert summary["val_perplexity"] > 1.0


def test_parameter_counting() -> None:
    """5. Verify parameter count of the loaded checkpoint model is positive and matches model.num_parameters()."""
    summary = evaluate_checkpoint(
        checkpoint_path=BEST_CHECKPOINT,
        base_dir=PROJECT_ROOT,
    )
    assert summary["num_parameters"] > 0
    assert summary["num_parameters"] == summary["model"].num_parameters()


def test_deterministic_greedy_generation() -> None:
    """6. Verify greedy generation from the loaded checkpoint is 100% deterministic."""
    model, tokenizer, _, _ = load_model_from_checkpoint(
        checkpoint_path=BEST_CHECKPOINT,
        base_dir=PROJECT_ROOT,
    )

    out_1 = generate_from_checkpoint(
        model, tokenizer, prompt="Transformer", max_new_tokens=8, greedy=True, seed=1
    )
    out_2 = generate_from_checkpoint(
        model, tokenizer, prompt="Transformer", max_new_tokens=8, greedy=True, seed=999
    )
    assert out_1 == out_2
    assert out_1.startswith("Transformer")


def test_seeded_sampling_reproducibility() -> None:
    """7. Verify temperature + top_k sampling produces identical output when given the same seed."""
    model, tokenizer, _, _ = load_model_from_checkpoint(
        checkpoint_path=BEST_CHECKPOINT,
        base_dir=PROJECT_ROOT,
    )

    sample_a = generate_from_checkpoint(
        model,
        tokenizer,
        prompt="MiniGPT",
        max_new_tokens=10,
        temperature=0.8,
        top_k=5,
        greedy=False,
        seed=123,
    )
    sample_b = generate_from_checkpoint(
        model,
        tokenizer,
        prompt="MiniGPT",
        max_new_tokens=10,
        temperature=0.8,
        top_k=5,
        greedy=False,
        seed=123,
    )
    assert sample_a == sample_b
    assert sample_a.startswith("MiniGPT")


def test_evaluation_output_generation() -> None:
    """8. Verify run_generation_suite writes outputs/generations.txt with all 5 prompts and 3 strategies."""
    model, tokenizer, _, _ = load_model_from_checkpoint(
        checkpoint_path=BEST_CHECKPOINT,
        base_dir=PROJECT_ROOT,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        gen_file = Path(tmp_dir) / "generations.txt"
        records = run_generation_suite(
            model=model,
            tokenizer=tokenizer,
            prompts=DEFAULT_PROMPTS,
            max_new_tokens=6,
            temperature=0.8,
            top_k=5,
            seed=42,
            output_path=gen_file,
        )

        # 5 prompts x 3 decoding configurations = 15 generation records
        assert len(records) == 15
        assert gen_file.exists()
        content = gen_file.read_text(encoding="utf-8")
        for prompt in DEFAULT_PROMPTS:
            assert prompt in content
        assert "Greedy decoding" in content
        assert "Temperature = 0.8, top_k = 5" in content
        assert "Temperature = 1.0, top_k = 10" in content


def test_training_curve_generation() -> None:
    """9. Verify plot_training_curves creates a valid PNG file at outputs/training_curves.png."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        png_path = Path(tmp_dir) / "training_curves.png"
        out = plot_training_curves(METRICS_JSON, png_path)
        assert out.exists()

        raw_bytes = out.read_bytes()
        # Verify PNG signature header (\x89PNG\r\n\x1a\n) and non-trivial file size
        assert raw_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(raw_bytes) > 500


def test_cli_argument_handling() -> None:
    """10. Verify scripts/evaluate.py CLI supports default and custom flags."""
    default_args = parse_args([])
    assert default_args.checkpoint == "checkpoints/best_model.pt"
    assert default_args.max_new_tokens == 16
    assert abs(default_args.temperature - 0.8) < 1e-6
    assert default_args.top_k == 5
    assert default_args.seed == 42

    custom_args = parse_args(
        [
            "--checkpoint",
            "checkpoints/best_model.pt",
            "--max-new-tokens",
            "10",
            "--temperature",
            "0.9",
            "--top-k",
            "8",
            "--seed",
            "7",
        ]
    )
    assert custom_args.checkpoint == "checkpoints/best_model.pt"
    assert custom_args.max_new_tokens == 10
    assert abs(custom_args.temperature - 0.9) < 1e-6
    assert custom_args.top_k == 8
    assert custom_args.seed == 7

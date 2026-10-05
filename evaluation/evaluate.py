"""
Phase 11: Checkpoint Loading, Validation Evaluation, and Reproducible Generation Demo.

Responsibilities:
1. Load `checkpoints/best_model.pt` without retraining
2. Reconstruct the exact `MiniGPT` architecture, `ExperimentConfig`, and `BPETokenizer`
3. Switch `model` to `.eval()` mode
4. Compute validation loss, validation perplexity, and trainable parameter count
5. Generate text across standard benchmark prompts using:
   - Greedy decoding
   - Temperature = 0.8, top_k = 5
   - Temperature = 1.0, top_k = 10
   with deterministic seeding (`set_seed`)
6. Save formatted generation results to `outputs/generations.txt` and training curves
   to `outputs/training_curves.png`
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from evaluation.visualize import plot_training_curves
from generation.generate import generate
from model.gpt import MiniGPT
from tokenizer.bpe_tokenizer import BPETokenizer
from training.dataset import create_dataloaders, create_datasets
from training.experiment import (
    ExperimentConfig,
    build_model_from_config,
    encode_corpus_with_limit,
    load_experiment_checkpoint,
    resolve_project_path,
    set_seed,
)
from training.metrics import compute_perplexity
from training.train import evaluate


DEFAULT_PROMPTS: List[str] = [
    "Tokenization",
    "Self attention",
    "Transformer",
    "MiniGPT",
    "Machine learning",
]


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path] = "checkpoints/best_model.pt",
    device: Union[str, torch.device] = "cpu",
    base_dir: Optional[Union[str, Path]] = None,
) -> Tuple[MiniGPT, BPETokenizer, ExperimentConfig, Dict[str, Any]]:
    """
    Load `checkpoints/best_model.pt`, restore the `MiniGPT` model weights, restore
    the `BPETokenizer` vocabulary and merge rules, restore `ExperimentConfig`, and
    switch the model to `.eval()` mode.

    Does NOT retrain the model.
    """
    project_root = (
        Path(base_dir).resolve()
        if base_dir is not None
        else Path(__file__).resolve().parent.parent
    )
    resolved_ckpt = resolve_project_path(project_root, checkpoint_path)
    if not resolved_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resolved_ckpt}")

    checkpoint, tokenizer = load_experiment_checkpoint(resolved_ckpt, device=device)
    config = ExperimentConfig.from_dict(checkpoint.get("config", {}))

    model = build_model_from_config(config, tokenizer, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, config, checkpoint


def evaluate_checkpoint(
    checkpoint_path: Union[str, Path] = "checkpoints/best_model.pt",
    data_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a saved MiniGPT checkpoint on the validation split without retraining.

    Returns:
        Dictionary containing `val_loss`, `val_perplexity`, `num_parameters`, `epoch`,
        `train_loss`, `model`, `tokenizer`, and `config`.
    """
    project_root = (
        Path(base_dir).resolve()
        if base_dir is not None
        else Path(__file__).resolve().parent.parent
    )
    model, tokenizer, config, ckpt = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        base_dir=project_root,
    )

    corpus_rel = data_path if data_path is not None else config.data_path
    corpus_file = resolve_project_path(project_root, corpus_rel)
    if not corpus_file.exists():
        raise FileNotFoundError(f"Evaluation corpus not found: {corpus_file}")

    text = corpus_file.read_text(encoding="utf-8")
    token_ids = encode_corpus_with_limit(text, tokenizer, config.max_tokens)

    _, val_dataset = create_datasets(
        token_ids=token_ids,
        sequence_length=config.sequence_length,
        train_ratio=config.train_ratio,
    )
    _, val_loader = create_dataloaders(
        train_dataset=val_dataset,
        val_dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle_train=False,
    )
    assert val_loader is not None

    model.eval()
    val_loss = evaluate(model, val_loader, device=device)
    val_perplexity = compute_perplexity(val_loss)
    num_params = model.num_parameters()

    return {
        "epoch": int(ckpt.get("epoch", 0)),
        "checkpoint_train_loss": float(ckpt.get("train_loss", 0.0)),
        "checkpoint_val_loss": float(ckpt.get("val_loss", val_loss)),
        "val_loss": float(val_loss),
        "val_perplexity": float(val_perplexity),
        "num_parameters": int(num_params),
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
    }


def generate_from_checkpoint(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 16,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    greedy: bool = False,
    seed: Optional[int] = 42,
) -> str:
    """
    Generate text from a loaded `MiniGPT` model and `BPETokenizer` with optional
    deterministic random seed.
    """
    if seed is not None:
        set_seed(seed)

    model.eval()
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        greedy=greedy,
        return_token_ids=False,
    )
    assert isinstance(output, str)
    return output


def run_generation_suite(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 16,
    temperature: float = 0.8,
    top_k: int = 5,
    seed: int = 42,
    output_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Run the 3 required decoding strategies for every prompt in `prompts`:
    1. Greedy decoding (`greedy=True`)
    2. Temperature = `temperature` (default 0.8), `top_k = top_k` (default 5)
    3. Temperature = 1.0, `top_k = 10`

    Optionally writes a human-readable report to `output_path` (`outputs/generations.txt`).
    """
    prompt_list = prompts if prompts is not None else list(DEFAULT_PROMPTS)
    vocab_limit = getattr(model, "vocab_size", tokenizer.current_vocab_size)
    k_small = min(max(1, top_k), vocab_limit)
    k_large = min(10, vocab_limit)

    strategies = [
        {
            "name": "Greedy decoding",
            "greedy": True,
            "temperature": 1.0,
            "top_k": None,
        },
        {
            "name": f"Temperature = {temperature}, top_k = {k_small}",
            "greedy": False,
            "temperature": float(temperature),
            "top_k": k_small,
        },
        {
            "name": f"Temperature = 1.0, top_k = {k_large}",
            "greedy": False,
            "temperature": 1.0,
            "top_k": k_large,
        },
    ]

    results: List[Dict[str, Any]] = []
    lines: List[str] = [
        "======================================================================",
        "MiniGPT Phase 11 — Autoregressive Text Generation Evaluation Report",
        f"Seed: {seed} | Max New Tokens: {max_new_tokens}",
        "======================================================================",
        "",
    ]

    for prompt in prompt_list:
        lines.append(f"Prompt: {prompt!r}")
        lines.append("-" * 70)

        for strat in strategies:
            generated_text = generate_from_checkpoint(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=strat["temperature"],
                top_k=strat["top_k"],
                greedy=strat["greedy"],
                seed=seed,
            )
            record = {
                "prompt": prompt,
                "strategy": strat["name"],
                "greedy": strat["greedy"],
                "temperature": strat["temperature"],
                "top_k": strat["top_k"],
                "max_new_tokens": max_new_tokens,
                "seed": seed,
                "generated_text": generated_text,
            }
            results.append(record)
            lines.append(f"  [{strat['name']}]")
            lines.append(f"  -> {generated_text}")

        lines.append("")

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("\n".join(lines), encoding="utf-8")

    return results


def run_evaluation_pipeline(
    checkpoint_path: Union[str, Path] = "checkpoints/best_model.pt",
    metrics_path: Union[str, Path] = "checkpoints/training_metrics.json",
    curves_output_path: Union[str, Path] = "outputs/training_curves.png",
    generations_output_path: Union[str, Path] = "outputs/generations.txt",
    max_new_tokens: int = 16,
    temperature: float = 0.8,
    top_k: int = 5,
    seed: int = 42,
    prompts: Optional[List[str]] = None,
    base_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Execute the complete Phase 11 evaluation, visualization, and generation pipeline:
    1. Load best checkpoint and compute validation loss, validation perplexity, and parameter count
    2. Plot training & validation loss/perplexity curves to `outputs/training_curves.png`
    3. Generate text for all benchmark prompts across all 3 decoding strategies and save
       to `outputs/generations.txt`
    """
    project_root = (
        Path(base_dir).resolve()
        if base_dir is not None
        else Path(__file__).resolve().parent.parent
    )

    eval_summary = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        base_dir=project_root,
    )

    resolved_metrics = resolve_project_path(project_root, metrics_path)
    resolved_curves = resolve_project_path(project_root, curves_output_path)
    resolved_generations = resolve_project_path(project_root, generations_output_path)

    if resolved_metrics.exists():
        plot_training_curves(resolved_metrics, resolved_curves)
    else:
        fallback_metrics = {
            "epochs": [
                {
                    "epoch": max(1, eval_summary["epoch"]),
                    "train_loss": eval_summary["checkpoint_train_loss"] or eval_summary["val_loss"],
                    "val_loss": eval_summary["val_loss"],
                    "train_perplexity": compute_perplexity(
                        eval_summary["checkpoint_train_loss"] or eval_summary["val_loss"]
                    ),
                    "val_perplexity": eval_summary["val_perplexity"],
                }
            ]
        }
        plot_training_curves(fallback_metrics, resolved_curves)

    generations = run_generation_suite(
        model=eval_summary["model"],
        tokenizer=eval_summary["tokenizer"],
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        output_path=resolved_generations,
    )

    return {
        "epoch": eval_summary["epoch"],
        "val_loss": eval_summary["val_loss"],
        "val_perplexity": eval_summary["val_perplexity"],
        "num_parameters": eval_summary["num_parameters"],
        "curves_path": str(resolved_curves),
        "generations_path": str(resolved_generations),
        "generations": generations,
    }

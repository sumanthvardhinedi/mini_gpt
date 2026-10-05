"""
Pytest test suite for MiniGPT Phase 8: LanguageModelDataset, Splits, and DataLoaders.
"""

from pathlib import Path
import torch
from tokenizer.basic_tokenizer import BasicTokenizer
from training.dataset import LanguageModelDataset, create_dataloaders, create_datasets


def test_sliding_window_examples() -> None:
    """Verify [10, 20, 30, 40, 50] with sequence_length=3 produces exact shifted windows."""
    token_ids = [10, 20, 30, 40, 50]
    dataset = LanguageModelDataset(token_ids, sequence_length=3)

    assert len(dataset) == 2

    input_0, target_0 = dataset[0]
    assert torch.equal(input_0, torch.tensor([10, 20, 30]))
    assert torch.equal(target_0, torch.tensor([20, 30, 40]))

    input_1, target_1 = dataset[1]
    assert torch.equal(input_1, torch.tensor([20, 30, 40]))
    assert torch.equal(target_1, torch.tensor([30, 40, 50]))


def test_dataset_item_shapes() -> None:
    """Verify each dataset item returns (input_ids, target_ids) of shape (sequence_length,)."""
    token_ids = list(range(50))
    dataset = LanguageModelDataset(token_ids, sequence_length=12)

    assert len(dataset) == 38
    inp, tgt = dataset[5]
    assert inp.shape == torch.Size([12])
    assert tgt.shape == torch.Size([12])
    assert torch.equal(inp[1:], tgt[:-1])


def test_deterministic_train_val_split() -> None:
    """Verify create_datasets keeps validation data strictly after training data without mixing."""
    token_ids = list(range(100))
    train_ds, val_ds = create_datasets(token_ids, sequence_length=5, train_ratio=0.8)

    # First 80 tokens [0..79] in train_ds; last 20 tokens [80..99] in val_ds
    assert train_ds.token_ids.numel() == 80
    assert val_ds.token_ids.numel() == 20
    assert int(train_ds.token_ids.max().item()) < int(val_ds.token_ids.min().item())


def test_dataloader_batching_and_shuffle_flags() -> None:
    """Verify DataLoaders yield (batch_size, sequence_length) batches and respect shuffle settings."""
    token_ids = list(range(60))
    train_ds, val_ds = create_datasets(token_ids, sequence_length=6, train_ratio=0.8)

    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds, batch_size=4, shuffle_train=True
    )
    assert val_loader is not None
    assert train_loader.shuffle is True
    assert val_loader.shuffle is False

    first_inputs, first_targets = next(iter(val_loader))
    assert first_inputs.shape == torch.Size([4, 6])
    assert first_targets.shape == torch.Size([4, 6])


def test_sample_txt_corpus_integration() -> None:
    """Verify data/sample.txt exists, is non-empty, and can be tokenized into datasets."""
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample.txt"
    assert sample_path.exists()

    text = sample_path.read_text(encoding="utf-8")
    assert len(text) > 200

    tokenizer = BasicTokenizer(text)
    token_ids = tokenizer.encode(text)
    train_ds, val_ds = create_datasets(token_ids, sequence_length=8, train_ratio=0.8)

    assert len(train_ds) > 10
    assert len(val_ds) > 2


def test_invalid_dataset_parameters() -> None:
    """Verify invalid sequence lengths or too-short token lists raise ValueError."""
    try:
        LanguageModelDataset([1, 2, 3], sequence_length=3)
        assert False, "Expected ValueError when len(token_ids) <= sequence_length"
    except ValueError:
        pass

    try:
        LanguageModelDataset([1, 2, 3, 4], sequence_length=0)
        assert False, "Expected ValueError for sequence_length <= 0"
    except ValueError:
        pass

    try:
        create_datasets(list(range(50)), sequence_length=4, train_ratio=1.2)
        assert False, "Expected ValueError for invalid train_ratio"
    except ValueError:
        pass

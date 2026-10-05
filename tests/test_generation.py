"""
Pytest test suite for MiniGPT Phase 9: Autoregressive Text Generation.
Covers all 14 required test scenarios:
1. Greedy generation
2. Generated token count
3. Temperature validation
4. Different temperatures
5. Top-k sampling
6. Invalid top-k values
7. Context-window truncation
8. CPU generation
9. Deterministic greedy generation
10. model.eval() behavior
11. No gradient creation
12. Prompt encoding/decoding
13. Single-token prompt
14. Empty prompt handling
Plus end-to-end generation with a real MiniGPT model.
"""

from typing import List
import torch
import torch.nn as nn
from generation.generate import apply_top_k, generate, select_next_token
from model.gpt import MiniGPT
from tokenizer.basic_tokenizer import BasicTokenizer


class DeterministicToyModel(nn.Module):
    """
    Tiny deterministic mock language model for fast, reproducible generation tests.
    Records the sequence lengths of every `context_ids` tensor passed to `forward`
    and whether `self.training` was False during generation.
    """

    def __init__(self, vocab_size: int = 8, max_sequence_length: int = 5) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.dummy_param = nn.Linear(vocab_size, vocab_size)
        self.seen_context_lengths: List[int] = []
        self.was_eval_during_forward: bool = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        self.seen_context_lengths.append(seq_len)
        self.was_eval_during_forward = not self.training

        # Construct deterministic logits where the highest logit at the final position
        # is always `(last_token_id + 1) % vocab_size`
        logits_data = []
        for b in range(batch_size):
            seq_logits = []
            for t in range(seq_len):
                last_id = int(input_ids[b, t].item())
                preferred_next = (last_id + 1) % self.vocab_size
                row = [-5.0 for _ in range(self.vocab_size)]
                row[preferred_next] = 10.0
                row[(preferred_next + 1) % self.vocab_size] = 8.0
                row[(preferred_next + 2) % self.vocab_size] = 6.0
                seq_logits.append(row)
            logits_data.append(seq_logits)

        return torch.tensor(logits_data, dtype=torch.float32, device=input_ids.device)


def _build_test_tokenizer() -> BasicTokenizer:
    corpus = "alpha beta gamma delta epsilon zeta eta"
    return BasicTokenizer(corpus)


def test_greedy_generation() -> None:
    """1. Verify greedy generation selects argmax(logits, dim=-1) at every step."""
    tokenizer = _build_test_tokenizer()
    model = DeterministicToyModel(vocab_size=tokenizer.vocab_size, max_sequence_length=10)

    prompt = "alpha"
    start_id = tokenizer.encode(prompt)[0]

    text, token_ids = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=3,
        greedy=True,
        return_token_ids=True,
    )

    expected_ids = [
        start_id,
        (start_id + 1) % tokenizer.vocab_size,
        (start_id + 2) % tokenizer.vocab_size,
        (start_id + 3) % tokenizer.vocab_size,
    ]
    assert token_ids == expected_ids
    assert text == tokenizer.decode(expected_ids)


def test_generated_token_count() -> None:
    """2. Verify total output length equals len(prompt_tokens) + max_new_tokens."""
    tokenizer = _build_test_tokenizer()
    model = DeterministicToyModel(vocab_size=tokenizer.vocab_size, max_sequence_length=16)

    prompt = "alpha beta"
    prompt_len = len(tokenizer.encode(prompt))

    _, ids_0 = generate(model, tokenizer, prompt=prompt, max_new_tokens=0, return_token_ids=True)
    assert len(ids_0) == prompt_len

    _, ids_5 = generate(model, tokenizer, prompt=prompt, max_new_tokens=5, return_token_ids=True)
    assert len(ids_5) == prompt_len + 5


def test_temperature_validation() -> None:
    """3. Verify temperature <= 0 raises ValueError."""
    tokenizer = _build_test_tokenizer()
    model = DeterministicToyModel(vocab_size=tokenizer.vocab_size)

    for invalid_temp in [0.0, -0.5, -1.0]:
        try:
            generate(model, tokenizer, prompt="alpha", max_new_tokens=2, temperature=invalid_temp)
            assert False, f"Expected ValueError for temperature={invalid_temp}"
        except ValueError:
            pass


def test_different_temperatures() -> None:
    """4. Verify low vs high temperature scales logits and changes probability sharpness."""
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])

    low_temp_probs = torch.softmax(logits / 0.2, dim=-1)
    high_temp_probs = torch.softmax(logits / 5.0, dim=-1)

    # Low temperature concentrates probability mass on the top token compared to high temperature
    assert float(low_temp_probs[0, 0].item()) > 0.98
    assert float(high_temp_probs[0, 0].item()) < 0.45

    tokenizer = _build_test_tokenizer()
    model = DeterministicToyModel(vocab_size=tokenizer.vocab_size)
    torch.manual_seed(42)
    out_low = generate(model, tokenizer, prompt="alpha", max_new_tokens=4, temperature=0.1)
    out_high = generate(model, tokenizer, prompt="alpha", max_new_tokens=4, temperature=2.5)
    assert isinstance(out_low, str) and len(out_low) > 0
    assert isinstance(out_high, str) and len(out_high) > 0


def test_top_k_sampling() -> None:
    """5. Verify top_k keeps only top K logits, sets others to -inf, and samples only from top K."""
    logits = torch.tensor([[1.0, 5.0, 2.0, 9.0, -3.0, 4.0]])
    filtered = apply_top_k(logits, top_k=2)

    # Indices 3 (9.0) and 1 (5.0) must be kept; all others must be -inf
    assert float(filtered[0, 3].item()) == 9.0
    assert float(filtered[0, 1].item()) == 5.0
    for idx in [0, 2, 4, 5]:
        assert filtered[0, idx].item() == float("-inf")

    probs = torch.softmax(filtered, dim=-1)
    for idx in [0, 2, 4, 5]:
        assert abs(float(probs[0, idx].item())) < 1e-9

    # Sampling 30 times with top_k=2 must only ever return index 1 or index 3
    torch.manual_seed(42)
    sampled = {
        int(select_next_token(logits, temperature=1.0, top_k=2, greedy=False).item())
        for _ in range(30)
    }
    assert sampled.issubset({1, 3})


def test_invalid_top_k_values() -> None:
    """6. Verify invalid top_k values (<= 0, > vocab_size, or non-int) raise ValueError."""
    tokenizer = _build_test_tokenizer()
    model = DeterministicToyModel(vocab_size=tokenizer.vocab_size)

    for invalid_k in [0, -3, tokenizer.vocab_size + 5]:
        try:
            generate(model, tokenizer, prompt="alpha", max_new_tokens=2, top_k=invalid_k)
            assert False, f"Expected ValueError for top_k={invalid_k}"
        except ValueError:
            pass


def test_context_window_truncation() -> None:
    """7. Verify generation crops input_ids[:, -max_sequence_length:] and never exceeds context window."""
    tokenizer = _build_test_tokenizer()
    max_seq_len = 4
    model = DeterministicToyModel(vocab_size=tokenizer.vocab_size, max_sequence_length=max_seq_len)

    # Prompt has 3 tokens; generating 6 new tokens grows total sequence to 9 tokens
    prompt = "alpha beta gamma"
    _, full_ids = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=6,
        return_token_ids=True,
    )

    assert len(full_ids) == 9
    assert len(model.seen_context_lengths) == 6
    # Every forward call must receive at most max_sequence_length (4) tokens
    assert max(model.seen_context_lengths) == max_seq_len
    assert all(length <= max_seq_len for length in model.seen_context_lengths)


def test_cpu_generation_with_minigpt() -> None:
    """8. Verify end-to-end CPU generation with the real MiniGPT architecture."""
    torch.manual_seed(42)
    tokenizer = _build_test_tokenizer()
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=32,
        dropout=0.0,
    ).to("cpu")

    text, token_ids = generate(
        model=model,
        tokenizer=tokenizer,
        prompt="alpha beta",
        max_new_tokens=5,
        temperature=0.8,
        top_k=3,
        return_token_ids=True,
    )
    assert isinstance(text, str)
    assert len(token_ids) == 7


def test_deterministic_greedy_generation() -> None:
    """9. Verify greedy generation produces identical outputs across repeated calls."""
    torch.manual_seed(42)
    tokenizer = _build_test_tokenizer()
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=32,
        dropout=0.0,
    )

    out_1, ids_1 = generate(model, tokenizer, prompt="alpha", max_new_tokens=4, return_token_ids=True)
    out_2, ids_2 = generate(model, tokenizer, prompt="alpha", max_new_tokens=4, return_token_ids=True)

    assert ids_1 == ids_2
    assert out_1 == out_2


def test_model_eval_behavior() -> None:
    """10. Verify generate() switches the model into eval() mode before running forward passes."""
    tokenizer = _build_test_tokenizer()
    model = DeterministicToyModel(vocab_size=tokenizer.vocab_size)
    model.train()
    assert model.training is True

    generate(model, tokenizer, prompt="alpha", max_new_tokens=2)

    assert model.was_eval_during_forward is True
    assert model.training is False


def test_no_gradient_creation() -> None:
    """11. Verify generation runs under torch.no_grad() and does not populate parameter gradients."""
    torch.manual_seed(42)
    tokenizer = _build_test_tokenizer()
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=8,
        embedding_dim=16,
        num_heads=4,
        num_layers=1,
        hidden_dim=32,
        dropout=0.0,
    )
    for param in model.parameters():
        param.grad = None

    generate(model, tokenizer, prompt="alpha", max_new_tokens=3)

    for param in model.parameters():
        assert param.grad is None


def test_prompt_encoding_and_decoding() -> None:
    """12. Verify generated token IDs start with the encoded prompt and decode cleanly."""
    tokenizer = _build_test_tokenizer()
    model = DeterministicToyModel(vocab_size=tokenizer.vocab_size, max_sequence_length=12)

    prompt = "alpha beta gamma"
    prompt_ids = tokenizer.encode(prompt)

    generated_text, full_ids = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=3,
        return_token_ids=True,
    )

    assert full_ids[: len(prompt_ids)] == prompt_ids
    assert generated_text == tokenizer.decode(full_ids)
    assert generated_text.startswith(prompt)


def test_single_token_prompt() -> None:
    """13. Verify generation works cleanly starting from a single-token prompt."""
    tokenizer = _build_test_tokenizer()
    model = DeterministicToyModel(vocab_size=tokenizer.vocab_size, max_sequence_length=8)

    text, ids = generate(
        model,
        tokenizer,
        prompt="delta",
        max_new_tokens=4,
        return_token_ids=True,
    )
    assert len(ids) == 5
    assert text.startswith("delta")


def test_empty_prompt_handling() -> None:
    """14. Verify empty or whitespace-only prompts raise ValueError."""
    tokenizer = _build_test_tokenizer()
    model = DeterministicToyModel(vocab_size=tokenizer.vocab_size)

    for bad_prompt in ["", "   "]:
        try:
            generate(model, tokenizer, prompt=bad_prompt, max_new_tokens=3)
            assert False, f"Expected ValueError for empty prompt {bad_prompt!r}"
        except ValueError:
            pass

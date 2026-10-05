"""
Phase 9: Autoregressive Text Generation for MiniGPT.

Educational Notes:
------------------
1. Autoregressive Decoding Loop:
   A decoder-only GPT language model generates text one token at a time:
   ```text
   prompt -> tokenizer.encode() -> input_ids -> MiniGPT -> final-step logits
          -> next_token -> append to input_ids -> repeat -> tokenizer.decode()
   ```
   Because each newly generated token is appended to `input_ids` and fed back into
   the model to predict the following token, the process is called *autoregressive*.

2. Greedy Decoding vs. Temperature & Top-K Sampling:
   - **Greedy decoding** (`next_token = torch.argmax(logits, dim=-1)`) always picks the
     single highest-scoring token. It is completely deterministic, which is great for
     testing, but can become repetitive on longer sequences.
   - **Temperature scaling** (`logits = logits / temperature`, with `temperature > 0`)
     controls the sharpness of the probability distribution before `softmax`:
     * `0 < temperature < 1.0`: sharpens the distribution (makes high-probability tokens
       even more likely).
     * `temperature > 1.0`: flattens the distribution (increases diversity/randomness).
   - **Top-K filtering** (`top_k=K`) keeps only the `K` highest logits and sets all
     other logits to `-inf` so their post-softmax probability is strictly `0.0`. This
     prevents sampling from the long tail of implausible tokens.

3. Context Window Cropping:
   Because learned positional embeddings only exist up to `max_sequence_length`, when
   `input_ids` grows longer than `max_sequence_length`, we slice the most recent tokens
   (`input_ids[:, -max_sequence_length:]`) before calling `model(...)`.
"""

from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn


def _get_model_device(model: nn.Module) -> Union[str, torch.device]:
    """Infer the device of `model` from its parameters or `.device` attribute."""
    if hasattr(model, "parameters"):
        params = list(model.parameters())
        if len(params) > 0 and hasattr(params[0], "device"):
            return params[0].device
    if hasattr(model, "device"):
        return getattr(model, "device")
    return torch.device("cpu")


def _get_model_max_sequence_length(model: nn.Module) -> Optional[int]:
    """Retrieve `max_sequence_length` from `model` if configured."""
    max_seq_len = getattr(model, "max_sequence_length", None)
    if isinstance(max_seq_len, int) and max_seq_len > 0:
        return max_seq_len
    embeddings = getattr(model, "embeddings", None)
    if embeddings is not None:
        emb_max_len = getattr(embeddings, "max_sequence_length", None)
        if isinstance(emb_max_len, int) and emb_max_len > 0:
            return emb_max_len
    return None


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Keep only the top `top_k` highest logits along the last dimension and set all
    other positions to `-inf`.

    Args:
        logits: Tensor of shape `(..., vocab_size)`.
        top_k: Number of highest-logit entries to keep (`1 <= top_k <= vocab_size`).

    Returns:
        Filtered logits tensor of the same shape as `logits`.
    """
    vocab_size = logits.size(-1)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"top_k must be an integer, got {top_k!r}.")
    if top_k <= 0 or top_k > vocab_size:
        raise ValueError(
            f"top_k must be between 1 and vocab_size ({vocab_size}), got {top_k}."
        )

    top_values, top_indices = torch.topk(logits, top_k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, top_indices, top_values)
    return filtered


def select_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    greedy: bool = False,
) -> torch.Tensor:
    """
    Select or sample the next token ID from final-position `logits` of shape `(batch_size, vocab_size)`.

    Args:
        logits: Unnormalized next-token scores of shape `(batch_size, vocab_size)`.
        temperature: Positive temperature scaling factor (`temperature > 0`).
        top_k: Optional number of highest logits to retain before softmax sampling.
        greedy: If True, deterministically select `torch.argmax(logits, dim=-1)`.

    Returns:
        Tensor of shape `(batch_size, 1)` containing the selected next token IDs.
    """
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature <= 0.0:
        raise ValueError(f"temperature must be strictly positive (> 0), got {temperature}.")

    if top_k is not None:
        vocab_size = logits.size(-1)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0 or top_k > vocab_size:
            raise ValueError(
                f"top_k must be an integer between 1 and vocab_size ({vocab_size}), got {top_k}."
            )

    if greedy:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        return next_token

    scaled_logits = logits / float(temperature)

    if top_k is not None:
        scaled_logits = apply_top_k(scaled_logits, top_k=top_k)

    probs = torch.softmax(scaled_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    greedy: Optional[bool] = None,
    do_sample: Optional[bool] = None,
    return_token_ids: bool = False,
) -> Union[str, Tuple[str, List[int]]]:
    """
    Autoregressively generate text from `prompt` using `model` and `tokenizer`.

    Pipeline:
        prompt -> tokenizer.encode() -> input_ids -> model(context_ids)
        -> logits[:, -1, :] -> select/sample next_token -> append -> repeat
        -> tokenizer.decode()

    Decoding strategy:
        - By default, if `top_k is None`, `temperature == 1.0`, and neither `greedy`
          nor `do_sample` is specified, greedy decoding (`torch.argmax`) is used.
        - If `top_k` is provided, `temperature != 1.0`, `do_sample=True`, or
          `greedy=False`, multinomial sampling with temperature and optional top-k
          filtering is used.
        - Passing `greedy=True` always forces greedy decoding.

    Args:
        model: MiniGPT (or compatible decoder language model) returning logits `(B, T, V)`.
        tokenizer: Tokenizer implementing `.encode(str) -> List[int]` and `.decode(List[int]) -> str`.
        prompt: Non-empty initial text prompt string.
        max_new_tokens: Number of new tokens to generate (`>= 0`).
        temperature: Positive temperature divisor applied to logits (`temperature > 0`).
        top_k: Optional integer (`1 <= top_k <= vocab_size`) restricting sampling to top K logits.
        greedy: Optional explicit flag to force greedy decoding (`True`) or sampling (`False`).
        do_sample: Optional explicit flag to enable sampling (`True`) or greedy (`False`).
        return_token_ids: If True, return `(generated_text, token_ids_list)`.

    Returns:
        `generated_text` string, or `(generated_text, full_token_ids)` if `return_token_ids=True`.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be a string, got {type(prompt).__name__}.")
    if len(prompt.strip()) == 0:
        raise ValueError("prompt cannot be empty or whitespace-only.")

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be a non-negative integer, got {max_new_tokens}.")

    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature <= 0.0:
        raise ValueError(f"temperature must be strictly positive (> 0), got {temperature}.")

    model_vocab_size = getattr(model, "vocab_size", None)
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")
        if isinstance(model_vocab_size, int) and top_k > model_vocab_size:
            raise ValueError(
                f"top_k ({top_k}) cannot exceed model vocab_size ({model_vocab_size})."
            )

    prompt_token_ids: List[int] = list(tokenizer.encode(prompt))
    if len(prompt_token_ids) == 0:
        raise ValueError("prompt encoded to zero tokens; cannot run autoregressive generation.")

    # Determine whether to use deterministic greedy decoding or temperature/top-k sampling
    if greedy is not None:
        use_greedy = bool(greedy)
    elif do_sample is not None:
        use_greedy = not bool(do_sample)
    else:
        use_greedy = (top_k is None) and (float(temperature) == 1.0)

    device = _get_model_device(model)
    max_seq_len = _get_model_max_sequence_length(model)

    model.eval()

    with torch.no_grad():
        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            # Crop context window to the model's maximum sequence length if exceeded
            if max_seq_len is not None and input_ids.size(1) > max_seq_len:
                context_ids = input_ids[:, -max_seq_len:]
            else:
                context_ids = input_ids

            logits = model(context_ids)
            next_token_logits = logits[:, -1, :]

            next_token = select_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                greedy=use_greedy,
            )

            input_ids = torch.cat([input_ids, next_token], dim=1)

        output_token_ids: List[int] = [int(tid) for tid in input_ids[0].tolist()]

    generated_text: str = tokenizer.decode(output_token_ids)

    if return_token_ids:
        return generated_text, output_token_ids
    return generated_text

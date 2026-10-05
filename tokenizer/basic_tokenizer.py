"""
Phase 1: Basic Tokenizer for MiniGPT.

Educational Notes:
------------------
1. What is a token?
   A token is a discrete unit of text produced by splitting raw strings into smaller
   pieces—such as words, numbers, or punctuation marks (e.g., "Hello", "world", "!").
   In later phases (like Byte Pair Encoding / BPE), tokens can also be subwords.

2. Why are token IDs needed?
   Neural networks (like PyTorch models) cannot directly process raw text strings or
   characters—they only operate on numbers (tensors of integers and floats).
   Assigning each unique token a deterministic integer ID (0, 1, 2, ...) translates
   human-readable text into a numerical sequence that a model can index and process.

3. Difference between tokenization and embeddings:
   - Tokenization (this phase) is a deterministic lookup step: it splits text into
     tokens and maps each token to an arbitrary integer ID (e.g., "Hello" -> 2).
   - Embeddings (coming in Phase 2) map each integer token ID to a learnable, dense
     vector of continuous floating-point numbers (e.g., ID 2 -> [0.14, -0.82, 0.55, ...]).

4. Why the tokenizer itself does NOT create semantic vectors:
   Token IDs are just arbitrary labels (like jersey numbers or dictionary row indices).
   Token ID 4 is not "closer in meaning" to Token ID 5 than to Token ID 50, and the
   tokenizer performs no mathematical training or similarity learning. Semantic meaning
   and relationships between words are learned later by the Embedding layer and
   Transformer attention layers during model training.
"""

import re
from typing import Dict, List


UNK_TOKEN = "<UNK>"

# Regular expression pattern that splits text into words (alphanumeric/underscore)
# and individual punctuation characters, discarding whitespace.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


class BasicTokenizer:
    """
    A beginner-friendly regular-expression tokenizer that splits text into words
    and punctuation, builds a deterministic vocabulary, and encodes/decodes text.
    """

    def __init__(self, training_text: str) -> None:
        """
        Initialize the tokenizer and build the vocabulary from training_text.

        Args:
            training_text: Raw text corpus used to discover known tokens.
        """
        self.unk_token: str = UNK_TOKEN
        self.str_to_int: Dict[str, int] = {}
        self.int_to_str: Dict[int, str] = {}

        self._build_vocab(training_text)

    def tokenize(self, text: str) -> List[str]:
        """
        Split raw text into a list of word and punctuation tokens.

        Example:
            "Hello world! Hello GPT." -> ["Hello", "world", "!", "Hello", "GPT", "."]
        """
        if not text:
            return []
        return TOKEN_PATTERN.findall(text)

    def _build_vocab(self, training_text: str) -> None:
        """
        Build deterministic str_to_int and int_to_str mappings from training text.
        Sorted unique tokens ensure deterministic IDs across runs, with <UNK> included
        so unknown words encountered at inference time never crash the tokenizer.
        """
        tokens = self.tokenize(training_text)
        unique_tokens = sorted(set(tokens))

        # Ensure <UNK> is always present in the vocabulary
        if self.unk_token in unique_tokens:
            unique_tokens.remove(self.unk_token)
        vocab_list = unique_tokens + [self.unk_token]

        self.str_to_int = {token: idx for idx, token in enumerate(vocab_list)}
        self.int_to_str = {idx: token for idx, token in enumerate(vocab_list)}

    @property
    def vocab_size(self) -> int:
        """Return the total number of tokens in the vocabulary (including <UNK>)."""
        return len(self.str_to_int)

    def encode(self, text: str) -> List[int]:
        """
        Convert a string of text into a list of integer token IDs.
        Any token not seen in the training text is mapped to the <UNK> token ID.
        """
        tokens = self.tokenize(text)
        unk_id = self.str_to_int[self.unk_token]
        return [self.str_to_int.get(token, unk_id) for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        Reconstruct human-readable text from a list of integer token IDs.
        Punctuation tokens are attached cleanly without preceding spaces.
        """
        if not token_ids:
            return ""

        tokens = [self.int_to_str.get(token_id, self.unk_token) for token_id in token_ids]

        # Join tokens with spaces, then remove spaces before common punctuation marks
        raw_joined = " ".join(tokens)
        cleaned_text = re.sub(r"\s+([.,!?;:)\]}])", r"\1", raw_joined)
        cleaned_text = re.sub(r"([(\[{])\s+", r"\1", cleaned_text)
        return cleaned_text

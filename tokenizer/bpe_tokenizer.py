"""
Phase 2: Byte Pair Encoding (BPE) Tokenizer for MiniGPT.

Educational Notes:
------------------
1. Why move from a word-level BasicTokenizer (Phase 1) to BPE (Phase 2)?
   A word-level tokenizer treats every unique word as a separate token. If it encounters
   a new word at inference time (e.g., "tokenization" when it only saw "tokenize"), it
   must fall back to `<UNK>`, losing information.
   Byte Pair Encoding (BPE) solves this by starting with individual characters and
   iteratively merging the most frequent adjacent pairs into subword and word tokens.

2. How BPE Training Works:
   - Start with the training text split into a sequence of individual characters.
   - Initialize the vocabulary with `<UNK>` plus all unique characters in the text.
   - Loop until the vocabulary reaches `vocab_size` (or no pairs remain):
     a) Count the frequency of every adjacent pair of tokens `(tokens[i], tokens[i+1])`.
     b) Find the most frequent pair `(left, right)`.
     c) Merge every occurrence of `(left, right)` into a single new token `left + right`.
     d) Record this merge rule and add the new token to the vocabulary.

3. How BPE Encoding & Decoding Work:
   - To encode new text, split it into characters (replacing any completely unseen
     character with `<UNK>`), then replay the learned merge rules in the exact order
     they were discovered during training.
   - Finally, map each subword/character token to its integer ID in `str_to_int`.
   - To decode, look up each token string in `int_to_str` and concatenate them (`"".join`).

4. Reminder: Still Discrete IDs (Not Embeddings!)
   Just like Phase 1, BPE outputs discrete integer IDs. It compresses common words into
   single IDs and breaks rare words into subword IDs, but it still does NOT create
   continuous semantic vectors—that happens in the Embedding layer (next phase).
"""

from typing import Dict, List, Optional, Tuple


UNK_TOKEN = "<UNK>"
Pair = Tuple[str, str]


class BPETokenizer:
    """
    A clean, from-scratch Byte Pair Encoding (BPE) tokenizer built for educational clarity.
    """

    def __init__(self, vocab_size: int = 100, training_text: Optional[str] = None) -> None:
        """
        Initialize the BPE tokenizer with a target vocabulary size.

        Args:
            vocab_size: Maximum number of tokens in the vocabulary (including <UNK>).
            training_text: Optional corpus to train on immediately upon initialization.
        """
        if vocab_size < 1:
            raise ValueError("vocab_size must be at least 1.")

        self.vocab_size: int = vocab_size
        self.unk_token: str = UNK_TOKEN
        self.str_to_int: Dict[str, int] = {self.unk_token: 0}
        self.int_to_str: Dict[int, str] = {0: self.unk_token}

        # Ordered list of learned merge rules: [((left, right), merged_token), ...]
        self.merges: List[Tuple[Pair, str]] = []

        if training_text is not None:
            self.train(training_text)

    @property
    def current_vocab_size(self) -> int:
        """Return the current number of tokens in the built vocabulary."""
        return len(self.str_to_int)

    def _init_base_vocab(self, text: str) -> None:
        """
        Reset and build the initial character-level vocabulary from the training text.
        `<UNK>` is always assigned ID 0 for determinism, followed by sorted unique characters.
        """
        unique_chars = sorted(set(text))
        if self.unk_token in unique_chars:
            unique_chars.remove(self.unk_token)

        vocab_list = [self.unk_token] + unique_chars
        # Respect vocab_size if vocab_size is smaller than base characters + <UNK>
        vocab_list = vocab_list[: self.vocab_size]

        self.str_to_int = {token: idx for idx, token in enumerate(vocab_list)}
        self.int_to_str = {idx: token for idx, token in enumerate(vocab_list)}
        self.merges = []

    def get_pair_frequencies(self, tokens: List[str]) -> Dict[Pair, int]:
        """
        Count frequencies of all adjacent token pairs `(tokens[i], tokens[i + 1])`.
        Pairs involving `<UNK>` are ignored so unknown placeholders never merge.
        """
        counts: Dict[Pair, int] = {}
        for i in range(len(tokens) - 1):
            left, right = tokens[i], tokens[i + 1]
            if left == self.unk_token or right == self.unk_token:
                continue
            pair = (left, right)
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def get_most_frequent_pair(self, pair_counts: Dict[Pair, int]) -> Pair:
        """
        Find the pair with the highest frequency.
        Uses deterministic tie-breaking: highest frequency first, then lexicographically
        smallest `(left, right)` tuple so results are 100% reproducible across runs.
        """
        if not pair_counts:
            raise ValueError("Cannot find most frequent pair from empty pair_counts.")
        return min(pair_counts.keys(), key=lambda pair: (-pair_counts[pair], pair[0], pair[1]))

    def merge_pair(
        self,
        tokens: List[str],
        pair: Pair,
        merged_token: Optional[str] = None,
    ) -> List[str]:
        """
        Replace all non-overlapping occurrences of `pair = (left, right)` in `tokens`
        with `merged_token` (defaults to `left + right`), scanning left-to-right.
        """
        if merged_token is None:
            merged_token = pair[0] + pair[1]

        merged: List[str] = []
        i = 0
        n = len(tokens)
        target_left, target_right = pair

        while i < n:
            if i < n - 1 and tokens[i] == target_left and tokens[i + 1] == target_right:
                merged.append(merged_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def train(self, text: str) -> None:
        """
        Train the BPE tokenizer on `text`:
        1. Initialize the vocabulary with `<UNK>` and unique characters.
        2. Iteratively find and merge the most frequent adjacent token pair
           until `vocab_size` is reached or no more pairs can be merged.
        """
        self._init_base_vocab(text)
        if not text:
            return

        # Convert training text into initial character-level tokens
        tokens: List[str] = [
            ch if ch in self.str_to_int else self.unk_token for ch in text
        ]

        while len(self.str_to_int) < self.vocab_size and len(tokens) >= 2:
            pair_counts = self.get_pair_frequencies(tokens)
            if not pair_counts:
                break

            best_pair = self.get_most_frequent_pair(pair_counts)
            merged_token = best_pair[0] + best_pair[1]

            tokens = self.merge_pair(tokens, best_pair, merged_token)
            self.merges.append((best_pair, merged_token))

            if merged_token not in self.str_to_int:
                new_id = len(self.str_to_int)
                self.str_to_int[merged_token] = new_id
                self.int_to_str[new_id] = merged_token

    def tokenize(self, text: str) -> List[str]:
        """
        Split `text` into BPE subword/character tokens by starting with characters
        (mapping unseen characters to `<UNK>`) and applying learned merges in order.
        """
        if not text:
            return []

        tokens: List[str] = [
            ch if ch in self.str_to_int else self.unk_token for ch in text
        ]

        for pair, merged_token in self.merges:
            if len(tokens) < 2:
                break
            tokens = self.merge_pair(tokens, pair, merged_token)

        return tokens

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert a list of token strings into integer token IDs."""
        unk_id = self.str_to_int[self.unk_token]
        return [self.str_to_int.get(token, unk_id) for token in tokens]

    def ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert a list of integer token IDs back into token strings."""
        return [self.int_to_str.get(token_id, self.unk_token) for token_id in token_ids]

    def encode(self, text: str) -> List[int]:
        """
        Encode a text string into a list of integer token IDs using learned BPE merges.
        """
        tokens = self.tokenize(text)
        return self.tokens_to_ids(tokens)

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of integer token IDs back into a reconstructed string.
        """
        if not token_ids:
            return ""
        tokens = self.ids_to_tokens(token_ids)
        return "".join(tokens)

    def to_state_dict(self) -> Dict[str, object]:
        """Serialize vocabulary and learned merge rules into JSON-compatible primitives."""
        merges_serialized = [[[pair[0], pair[1]], merged] for pair, merged in self.merges]
        return {
            "vocab_size": self.vocab_size,
            "unk_token": self.unk_token,
            "str_to_int": dict(self.str_to_int),
            "int_to_str": {int(k): v for k, v in self.int_to_str.items()},
            "merges": merges_serialized,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "BPETokenizer":
        """Reconstruct a `BPETokenizer` instance from a serialized state dictionary."""
        vocab_map = state["str_to_int"]
        assert isinstance(vocab_map, dict)
        vocab_size = int(state.get("vocab_size", len(vocab_map)))  # type: ignore[arg-type]
        tokenizer = cls(vocab_size=max(1, vocab_size))
        tokenizer.unk_token = str(state.get("unk_token", UNK_TOKEN))
        tokenizer.str_to_int = {str(k): int(v) for k, v in vocab_map.items()}

        raw_inv = state.get("int_to_str")
        if isinstance(raw_inv, dict):
            tokenizer.int_to_str = {int(k): str(v) for k, v in raw_inv.items()}
        else:
            tokenizer.int_to_str = {int(v): str(k) for k, v in tokenizer.str_to_int.items()}

        raw_merges = state.get("merges", [])
        assert isinstance(raw_merges, list)
        tokenizer.merges = [
            ((str(pair[0]), str(pair[1])), str(merged))
            for pair, merged in raw_merges
        ]
        return tokenizer

"""
Pytest test suite for MiniGPT Phase 1: BasicTokenizer.
Tests vocabulary creation, encoding, decoding, punctuation handling, and <UNK> behavior.
"""

from tokenizer.basic_tokenizer import BasicTokenizer, UNK_TOKEN


def test_vocabulary_creation() -> None:
    """Verify deterministic vocabulary creation and mappings."""
    text = "Hello world! Hello GPT."
    tokenizer = BasicTokenizer(text)

    # Unique tokens from text: "!", ".", "GPT", "Hello", "world" plus "<UNK>"
    expected_tokens = {"Hello", "world", "!", "GPT", ".", UNK_TOKEN}
    assert set(tokenizer.str_to_int.keys()) == expected_tokens
    assert len(tokenizer.str_to_int) == 6
    assert len(tokenizer.int_to_str) == 6

    # Check bijection between str_to_int and int_to_str
    for token, token_id in tokenizer.str_to_int.items():
        assert tokenizer.int_to_str[token_id] == token


def test_tokenize_with_punctuation() -> None:
    """Verify regex splitting separates words and punctuation cleanly."""
    text = "Hello world! Hello GPT."
    tokenizer = BasicTokenizer(text)

    tokens = tokenizer.tokenize(text)
    assert tokens == ["Hello", "world", "!", "Hello", "GPT", "."]


def test_encode_deterministic() -> None:
    """Verify encoding produces deterministic integer IDs and repeated words share IDs."""
    text = "Hello world! Hello GPT."
    tokenizer = BasicTokenizer(text)

    encoded = tokenizer.encode(text)
    assert len(encoded) == 6
    # "Hello" appears at index 0 and index 3, so their token IDs must match
    assert encoded[0] == encoded[3]
    # Sorted unique tokens: "!" (0), "." (1), "GPT" (2), "Hello" (3), "world" (4), "<UNK>" (5)
    assert encoded == [3, 4, 0, 3, 2, 1]


def test_decode_reconstructs_text() -> None:
    """Verify decoding reconstructs readable text with proper punctuation spacing."""
    text = "Hello world! Hello GPT."
    tokenizer = BasicTokenizer(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == "Hello world! Hello GPT."


def test_unknown_token_behavior() -> None:
    """Verify unknown words map to <UNK> instead of raising a KeyError."""
    training_text = "Hello world! Hello GPT."
    tokenizer = BasicTokenizer(training_text)

    test_sentence = "Hello universe! Welcome, GPT."
    encoded = tokenizer.encode(test_sentence)

    unk_id = tokenizer.str_to_int[UNK_TOKEN]
    # "universe", "Welcome", and "," were not in training_text
    assert encoded.count(unk_id) == 3

    decoded = tokenizer.decode(encoded)
    assert decoded == "Hello <UNK>! <UNK> <UNK> GPT."


def test_sentence_with_mixed_punctuation() -> None:
    """Test encoding and decoding on a sentence with multiple punctuation marks."""
    corpus = "Transformers, attention, and tokens: how do LLMs work? Fast!"
    tokenizer = BasicTokenizer(corpus)

    sample = "Transformers, tokens: how do LLMs work?"
    token_ids = tokenizer.encode(sample)
    reconstructed = tokenizer.decode(token_ids)

    assert reconstructed == sample

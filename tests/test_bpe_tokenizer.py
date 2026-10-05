"""
Pytest test suite for MiniGPT Phase 2: BPETokenizer.
Covers all 10 required BPE test scenarios:
1. Training the tokenizer
2. Vocabulary creation
3. Pair-frequency calculation
4. Merge operation
5. Encoding
6. Decoding
7. Unknown/unseen text
8. Deterministic results
9. Increasing vocabulary size
10. Round-trip: text -> tokens -> IDs -> tokens -> text
"""

from tokenizer.bpe_tokenizer import BPETokenizer, UNK_TOKEN


def test_training_tokenizer() -> None:
    """1. Verify training populates merges and expands vocabulary beyond base characters."""
    text = "aaabdaaabac"
    tokenizer = BPETokenizer(vocab_size=8)
    tokenizer.train(text)

    # Base vocab: <UNK>, 'a', 'b', 'c', 'd' (5 tokens). Target vocab_size=8 -> 3 merges learned.
    assert len(tokenizer.merges) == 3
    assert tokenizer.current_vocab_size == 8
    assert tokenizer.merges[0] == (("a", "a"), "aa")


def test_vocabulary_creation() -> None:
    """2. Verify base characters, <UNK>, and merged subwords are stored in bijective maps."""
    text = "Hello world! Hello GPT."
    tokenizer = BPETokenizer(vocab_size=20)
    tokenizer.train(text)

    assert UNK_TOKEN in tokenizer.str_to_int
    assert tokenizer.str_to_int[UNK_TOKEN] == 0
    assert tokenizer.int_to_str[0] == UNK_TOKEN

    # Every character from training text must be in str_to_int
    for ch in set(text):
        assert ch in tokenizer.str_to_int

    # Bijection check between str_to_int and int_to_str
    assert len(tokenizer.str_to_int) == len(tokenizer.int_to_str)
    for token_str, token_id in tokenizer.str_to_int.items():
        assert tokenizer.int_to_str[token_id] == token_str


def test_pair_frequency_calculation() -> None:
    """3. Verify adjacent pair frequency calculation and <UNK> exclusion."""
    tokenizer = BPETokenizer(vocab_size=50)
    tokens = ["a", "a", "a", "b", "a", "b"]
    freqs = tokenizer.get_pair_frequencies(tokens)

    # Adjacent pairs: (a,a) at 0-1 and 1-2 -> 2; (a,b) at 2-3 and 4-5 -> 2; (b,a) at 3-4 -> 1
    assert freqs == {
        ("a", "a"): 2,
        ("a", "b"): 2,
        ("b", "a"): 1,
    }
    # Deterministic tie-breaking selects ("a", "a") before ("a", "b")
    assert tokenizer.get_most_frequent_pair(freqs) == ("a", "a")


def test_merge_operation() -> None:
    """4. Verify non-overlapping left-to-right merge operation."""
    tokenizer = BPETokenizer(vocab_size=50)
    tokens = ["a", "a", "a", "b", "a", "a"]

    merged = tokenizer.merge_pair(tokens, ("a", "a"))
    # First two "a"s merge into "aa", third "a" remains "a", last two "a"s merge into "aa"
    assert merged == ["aa", "a", "b", "aa"]


def test_encoding() -> None:
    """5. Verify encoding applies learned merges and produces compressed integer IDs."""
    text = "Hello world! Hello world!"
    tokenizer = BPETokenizer(vocab_size=25)
    tokenizer.train(text)

    ids = tokenizer.encode("Hello world")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    # Learned merges must compress "Hello world" to fewer tokens than its 11 raw characters
    assert len(ids) < len("Hello world")


def test_decoding() -> None:
    """6. Verify decoding reconstructs the original text string from token IDs."""
    text = "Hello world! Hello GPT."
    tokenizer = BPETokenizer(vocab_size=30)
    tokenizer.train(text)

    ids = tokenizer.encode("Hello world")
    decoded = tokenizer.decode(ids)
    assert decoded == "Hello world"


def test_unknown_and_unseen_text() -> None:
    """7. Verify subword composition on unseen words and <UNK> fallback on unseen characters."""
    training_text = "low lower lowest"
    tokenizer = BPETokenizer(vocab_size=15)
    tokenizer.train(training_text)

    # Case A: Unseen combination of known characters ("low " + "est" -> "lowest low")
    unseen_word_ids = tokenizer.encode("lowest low")
    unk_id = tokenizer.str_to_int[UNK_TOKEN]
    assert unk_id not in unseen_word_ids
    assert tokenizer.decode(unseen_word_ids) == "lowest low"

    # Case B: Completely unseen characters ('Z', '!') map safely to <UNK>
    unseen_char_ids = tokenizer.encode("lowZ!")
    assert unseen_char_ids.count(unk_id) == 2
    assert tokenizer.decode(unseen_char_ids) == "low<UNK><UNK>"


def test_deterministic_results() -> None:
    """8. Verify identical training inputs produce identical vocabularies, merges, and IDs."""
    corpus = "Transformers learn subword tokenization via Byte Pair Encoding."
    tok1 = BPETokenizer(vocab_size=45)
    tok2 = BPETokenizer(vocab_size=45)

    tok1.train(corpus)
    tok2.train(corpus)

    assert tok1.str_to_int == tok2.str_to_int
    assert tok1.merges == tok2.merges
    assert tok1.encode(corpus) == tok2.encode(corpus)


def test_increasing_vocabulary_size() -> None:
    """9. Verify larger vocab_size learns more merges and compresses text into fewer token IDs."""
    corpus = "hello world! hello world! hello gpt! hello world!"

    small_tok = BPETokenizer(vocab_size=15)
    small_tok.train(corpus)

    large_tok = BPETokenizer(vocab_size=30)
    large_tok.train(corpus)

    assert large_tok.current_vocab_size > small_tok.current_vocab_size
    assert len(large_tok.merges) > len(small_tok.merges)

    small_encoded = small_tok.encode(corpus)
    large_encoded = large_tok.encode(corpus)
    assert len(large_encoded) < len(small_encoded)


def test_round_trip_pipeline() -> None:
    """10. Verify full round-trip: text -> tokens -> IDs -> tokens -> text."""
    corpus = "Hello world! Hello GPT. Subword BPE handles punctuation & spaces."
    tokenizer = BPETokenizer(vocab_size=60)
    tokenizer.train(corpus)

    sample_text = "Hello GPT! Subword BPE."

    # Step 1: text -> tokens
    tokens = tokenizer.tokenize(sample_text)
    assert isinstance(tokens, list) and len(tokens) > 0

    # Step 2: tokens -> IDs
    ids = tokenizer.tokens_to_ids(tokens)
    assert ids == tokenizer.encode(sample_text)

    # Step 3: IDs -> tokens
    reconstructed_tokens = tokenizer.ids_to_tokens(ids)
    assert reconstructed_tokens == tokens

    # Step 4: tokens -> text
    reconstructed_text = tokenizer.decode(ids)
    assert reconstructed_text == sample_text

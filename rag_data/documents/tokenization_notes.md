# Tokenization and Byte Pair Encoding (BPE)

Tokenization is the first step in every language model pipeline. Because neural networks operate on numerical tensors rather than raw strings, a tokenizer converts text into a sequence of discrete integer token IDs.

## Word-Level Basic Tokenizer

In Phase 1 of MiniGPT, the `BasicTokenizer` splits text into words and punctuation marks using the regular expression `\w+|[^\w\s]`. It builds a deterministic sorted vocabulary and includes a special `<UNK>` token (assigned to unseen words during encoding). While simple to understand, word-level tokenization suffers from out-of-vocabulary words whenever new word forms appear at test time.

## Subword Byte Pair Encoding (BPE)

In Phase 2 of MiniGPT, the `BPETokenizer` implements subword tokenization from scratch without external libraries. BPE starts with individual characters as the initial vocabulary alongside `<UNK>`. During training, it repeatedly counts adjacent token pairs, selects the most frequent pair using deterministic lexicographical tie-breaking, merges all non-overlapping occurrences of that pair into a new subword token, and records the learned merge rule. Common words like "the" or "attention" merge into single tokens, while rare words decompose into known subword pieces.

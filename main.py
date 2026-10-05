"""
MiniGPT — Phase 1: Basic Tokenizer Entry Point.

Run from the mini-gpt directory:
    python main.py
"""

from tokenizer.basic_tokenizer import BasicTokenizer


def main() -> None:
    training_text = "Hello world! Hello GPT."
    tokenizer = BasicTokenizer(training_text)

    print("=== MiniGPT Phase 1: Basic Tokenizer ===")
    print(f"Training Text : {training_text!r}")
    print(f"Tokens        : {tokenizer.tokenize(training_text)}")
    print(f"str_to_int    : {tokenizer.str_to_int}")
    print(f"int_to_str    : {tokenizer.int_to_str}")

    encoded = tokenizer.encode(training_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nEncoded IDs   : {encoded}")
    print(f"Decoded Text  : {decoded!r}")

    # Demonstrate <UNK> behavior on unseen words
    unseen_text = "Hello Python! Hello Transformer."
    unseen_encoded = tokenizer.encode(unseen_text)
    unseen_decoded = tokenizer.decode(unseen_encoded)
    print("\n--- Unknown Token (<UNK>) Demo ---")
    print(f"Input Text    : {unseen_text!r}")
    print(f"Encoded IDs   : {unseen_encoded}")
    print(f"Decoded Text  : {unseen_decoded!r}")


if __name__ == "__main__":
    main()

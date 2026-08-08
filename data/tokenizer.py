"""
Tokenizer module.

Handles text encoding and decoding.
"""

import tiktoken


class GPTTokenizer:

    def __init__(self, encoding_name="gpt2"):

        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def encode(self, text):

        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids):

        return self.tokenizer.decode(token_ids)

    @property
    def vocabulary_size(self):

        return self.tokenizer.n_vocab
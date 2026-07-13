"""Custom 4k BPE tokenizer for TinyStories.

The small vocab (default 4096) is the experiment's single most important design choice
(spec §5): it keeps the embedding/LM-head table small so the ternary block savings show
up in the byte total. Do NOT swap in a large pretrained vocab.

Special tokens are pinned to stable ids: ``<|bos|>=0``, ``<|eos|>=1``, ``<|pad|>=2``.
"""

from __future__ import annotations

from typing import Iterable

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

BOS = "<|bos|>"
EOS = "<|eos|>"
PAD = "<|pad|>"
SPECIAL_TOKENS = [BOS, EOS, PAD]  # ids 0, 1, 2


def _new_tokenizer() -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None))
    # ByteLevel pre-tokenizer + decoder gives lossless roundtrip over arbitrary bytes.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(
    texts: Iterable[str], save_path: str, vocab_size: int = 4096
) -> None:
    """Train a ByteLevel BPE tokenizer on `texts` and save it to `save_path`.

    Special tokens are registered first so they keep ids 0/1/2.
    """
    tok = _new_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(save_path)


def load_tokenizer(path: str) -> Tokenizer:
    """Load a saved tokenizer from `path`."""
    return Tokenizer.from_file(path)

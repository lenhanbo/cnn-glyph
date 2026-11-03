"""The immutable VATr/IAM vocabulary and its numeric ID contract."""

from __future__ import annotations

import json
from pathlib import Path

IAM_ALPHABET = "Only thewigsofrcvdampbkuq.A-210xT5'MDL,RYHJ\"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%"
SPECIAL_ALPHABET = "ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω"
PADDING_ID = 0
VOCAB_SIZE_WITH_PADDING = 130


def validate_vocabulary() -> None:
    assert len(IAM_ALPHABET) == 80
    assert len(SPECIAL_ALPHABET) == 49
    combined = IAM_ALPHABET + SPECIAL_ALPHABET
    if len(set(combined)) != len(combined):
        raise ValueError("VATr vocabulary contains duplicate characters")


def char_to_id() -> dict[str, int]:
    validate_vocabulary()
    return {char: index for index, char in enumerate(IAM_ALPHABET + SPECIAL_ALPHABET, 1)}


def write_vocab_json(path: str | Path) -> None:
    mapping = char_to_id()
    payload = {
        "padding_id": PADDING_ID,
        "iam_alphabet": IAM_ALPHABET,
        "special_alphabet": SPECIAL_ALPHABET,
        "characters": [{"id": idx, "character": char} for char, idx in mapping.items()],
        "size_with_padding": VOCAB_SIZE_WITH_PADDING,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


validate_vocabulary()

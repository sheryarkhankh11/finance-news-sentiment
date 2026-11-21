"""Tokenisation and vocabulary for the from-scratch deep model."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*|[%$€£]")
_URL_RE = re.compile(r"https?://\S+")


def tokenize(text: str) -> list[str]:
    text = _URL_RE.sub(" ", str(text).lower())
    toks = _TOKEN_RE.findall(text)
    return ["<num>" if t[0].isdigit() else t for t in toks]


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts, min_freq: int = 1, max_size: int = 20000) -> "Vocab":
        counts: Counter = Counter()
        for t in texts:
            counts.update(tokenize(t))
        words = [w for w, c in counts.most_common() if c >= min_freq][: max_size - 2]
        return cls([PAD, UNK] + words)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.stoi.get(t, UNK_ID) for t in tokenize(text)][:max_len]
        if not ids:  # empty / all-symbol text must still have one real token
            ids = [UNK_ID]
        return ids + [PAD_ID] * (max_len - len(ids))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

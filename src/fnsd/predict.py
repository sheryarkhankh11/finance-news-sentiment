"""Uniform `predict_proba(texts) -> (n, 3)` interface over all models.

Column order is always [negative, neutral, positive].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TfidfModel:
    """Classical baseline: TF-IDF (1-2 grams) + logistic regression. No torch needed."""

    name = "tfidf"

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def fit(self, texts, labels, ngram_max: int = 2, C: float = 2.0, seed: int = 42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.pipeline = make_pipeline(
            TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True, lowercase=True),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed),
        )
        self.pipeline.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts) -> np.ndarray:
        raw = self.pipeline.predict_proba(list(texts))
        classes = list(self.pipeline.classes_)
        out = np.zeros((raw.shape[0], 3))
        for j, c in enumerate(classes):
            out[:, int(c)] = raw[:, j]
        return out

    def save(self, out_dir: str | Path) -> None:
        import joblib
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, Path(out_dir) / "tfidf.joblib")

    @classmethod
    def load(cls, out_dir: str | Path) -> "TfidfModel":
        import joblib
        return cls(joblib.load(Path(out_dir) / "tfidf.joblib"))


class BiLSTMModel:
    name = "bilstm"

    def __init__(self, model, vocab, max_len: int):
        self.model, self.vocab, self.max_len = model, vocab, max_len

    @classmethod
    def load(cls, out_dir: str | Path) -> "BiLSTMModel":
        import torch
        from .models.bilstm import AttnBiLSTM
        from .text import Vocab

        out_dir = Path(out_dir)
        ckpt = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True)
        model = AttnBiLSTM(**ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return cls(model, Vocab.load(out_dir / "vocab.json"), ckpt["max_len"])

    def predict_proba(self, texts, batch_size: int = 256) -> np.ndarray:
        import torch

        texts = list(texts)
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                ids = [self.vocab.encode(t, self.max_len) for t in texts[i:i + batch_size]]
                logits = self.model(torch.tensor(ids, dtype=torch.long))
                probs.append(torch.softmax(logits, dim=1).numpy())
        return np.concatenate(probs) if probs else np.zeros((0, 3))


class FinBERTModel:
    """Pre-trained FinBERT used for inference only (optional; needs `transformers`+`torch`).

    Weights (~440 MB) download once from the Hugging Face Hub and are cached by HF.
    """

    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 16, max_len: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.batch_size, self.max_len = batch_size, max_len
        # Re-order the model's own label ids into our [neg, neu, pos] columns.
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self._order = [next(i for i, l in id2label.items() if l == lab)
                       for lab in ("negative", "neutral", "positive")]
        self._torch = torch

    def predict_proba(self, texts) -> np.ndarray:
        torch = self._torch
        texts = list(texts)
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
                p = torch.softmax(self.model(**enc).logits, dim=1).numpy()
                out.append(p[:, self._order])
        return np.concatenate(out) if out else np.zeros((0, 3))


def model_dir(cfg: dict, model: str, trained_on: str) -> Path:
    return Path(cfg["paths"]["artifacts_dir"]) / f"{model}_{trained_on}"


def load_model(cfg: dict, model: str, trained_on: str = "sample"):
    if model == "finbert":
        fb = cfg["finbert"]
        return FinBERTModel(fb["model_name"], fb["batch_size"], fb["max_len"])
    d = model_dir(cfg, model, trained_on)
    if not d.exists():
        raise FileNotFoundError(f"{d} not found - run: fnsd train --model {model} --data {trained_on}")
    return TfidfModel.load(d) if model == "tfidf" else BiLSTMModel.load(d)

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED = ["headline", "label"]


def load_labeled(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df.dropna(subset=REQUIRED).copy()
    df["headline"] = df["headline"].astype(str)
    df["label"] = df["label"].astype(int)
    if not set(df["label"]).issubset({0, 1, 2}):
        raise ValueError("labels must be 0=negative, 1=neutral, 2=positive")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    """Stratified train/val/test split (deterministic for a given seed)."""
    train_val, test = train_test_split(df, test_size=test_frac, stratify=df["label"], random_state=seed)
    rel_val = val_frac / (1.0 - test_frac)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_hf_financial_news(name: str, max_rows: int, seed: int, out_csv: str | Path) -> pd.DataFrame:
    """Download a small public financial-news sentiment set via Hugging Face `datasets`,
    subsample it, map labels to our convention and cache it as CSV.

    Default dataset has labels 0=Bearish, 1=Bullish, 2=Neutral -> we use 0/2/1.
    Check the dataset card for licence/terms before redistributing anything.
    """
    out_csv = Path(out_csv)
    if out_csv.exists():
        return load_labeled(out_csv)
    from datasets import load_dataset  # lazy: optional dependency

    df = load_dataset(name, split="train").to_pandas()
    df = df.rename(columns={"text": "headline"})
    df["label"] = df["label"].map({0: 0, 1: 2, 2: 1})
    df = df.dropna(subset=["headline", "label"])
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    df = df.assign(date="", ticker="NA")[["date", "ticker", "headline", "label"]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return load_labeled(out_csv)

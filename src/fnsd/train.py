from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .predict import TfidfModel
from .text import Vocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def train_tfidf(train_df: pd.DataFrame, cfg: dict, out_dir: str | Path) -> TfidfModel:
    set_seed(cfg["seed"])
    m = TfidfModel().fit(train_df["headline"], train_df["label"],
                         ngram_max=cfg["tfidf"]["ngram_max"], C=cfg["tfidf"]["C"], seed=cfg["seed"])
    m.save(out_dir)
    return m


def train_bilstm(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: dict, out_dir: str | Path):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from .models.bilstm import AttnBiLSTM
    from .predict import BiLSTMModel

    seed, tcfg, bcfg = cfg["seed"], cfg["text"], cfg["bilstm"]
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocab.build(train_df["headline"], tcfg["min_freq"], tcfg["max_vocab"])

    def encode(df):
        x = torch.tensor([vocab.encode(t, tcfg["max_len"]) for t in df["headline"]], dtype=torch.long)
        y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
        return x, y

    xtr, ytr = encode(train_df)
    xva, yva = encode(val_df)

    arch = dict(vocab_size=len(vocab), emb_dim=bcfg["emb_dim"], hidden=bcfg["hidden"],
                num_classes=3, dropout=bcfg["dropout"])
    model = AttnBiLSTM(**arch)

    counts = np.bincount(ytr.numpy(), minlength=3)
    weights = counts.sum() / (3.0 * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float))
    opt = torch.optim.Adam(model.parameters(), lr=bcfg["lr"])
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=bcfg["batch_size"], shuffle=True, generator=gen)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, bcfg["epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val_pred = model(xva).argmax(1).numpy()
        f1 = float(f1_score(yva.numpy(), val_pred, average="macro", labels=[0, 1, 2], zero_division=0))
        history.append({"epoch": epoch, "train_loss": total / len(xtr), "val_macro_f1": f1})
        print(f"epoch {epoch:02d}  loss {total / len(xtr):.4f}  val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= bcfg["patience"]:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    model.eval()
    torch.save({"state_dict": model.state_dict(), "arch": arch, "max_len": tcfg["max_len"]}, out_dir / "model.pt")
    vocab.save(out_dir / "vocab.json")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return BiLSTMModel(model, vocab, tcfg["max_len"])

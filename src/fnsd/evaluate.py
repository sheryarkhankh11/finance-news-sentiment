from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import LABELS


def evaluate_model(model, df) -> dict:
    proba = model.predict_proba(df["headline"].tolist())
    pred = proba.argmax(1)
    y = df["label"].to_numpy()
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "per_class": classification_report(y, pred, labels=[0, 1, 2], target_names=LABELS,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def save_metrics(metrics: dict, out_dir: str | Path, stem: str = "metrics") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS, rotation=30)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title(f"macro-F1 {metrics['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_confusion.png", dpi=130)
    plt.close(fig)

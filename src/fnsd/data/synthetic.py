"""Generate a small SYNTHETIC labelled headline set (fictional companies).

Purpose: lets the whole pipeline (train -> evaluate -> desk report -> CI) run offline in
seconds with zero licensing or privacy concerns. It is NOT a benchmark: templates are
simple, so scores on it are optimistic. Use the real-data option (`--data hf`) for
meaningful numbers.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

import pandas as pd

COMPANIES = [
    ("ACME", "Acme Corp"), ("NVLX", "Novalux"), ("BRTN", "Brighton Energy"),
    ("QNTM", "Quantum Freight"), ("HLXB", "Helix Biotech"), ("ORBT", "Orbit Retail"),
]

POS = [
    "{co} beats quarterly earnings estimates as revenue climbs {n}%",
    "{co} raises full-year guidance after strong demand",
    "Analysts upgrade {co} to buy on improving margins",
    "{co} shares jump after record profit",
    "{co} wins major contract, boosting order backlog",
    "{co} announces share buyback and higher dividend",
    "{co} posts surprise profit as sales rise {n}%",
    "{co} reports revenue growth of {n}% and lifts outlook, with only minor cost pressure",
]
NEG = [
    "{co} misses profit forecasts as costs surge {n}%",
    "{co} cuts annual outlook amid weak demand",
    "Analysts downgrade {co} to sell over margin concerns",
    "{co} shares tumble after regulator opens probe",
    "{co} announces layoffs as revenue falls {n}%",
    "{co} recalls products and warns of lower earnings",
    "{co} swings to quarterly loss as sales drop {n}%",
    "{co} reports revenue growth of {n}% but slashes outlook on surging costs",
]
NEU = [
    "{co} to present at investor conference next week",
    "{co} schedules quarterly results announcement",
    "{co} appoints new chief financial officer",
    "{co} completes routine debt refinancing",
    "{co} announces date of annual shareholder meeting",
    "{co} files quarterly report with regulator",
    "{co} opens new regional office",
    "{co} comments on market rumours, no change to outlook",
]
PREFIX = ["", "", "", "Breaking: ", "Update: ", "Exclusive: "]
SUFFIX = ["", "", "", ", sources say", " in early trading", ", company statement shows"]


def make_sample(n: int = 900, seed: int = 42, label_noise: float = 0.05,
                start: date = date(2025, 1, 2), days: int = 60) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        label = rng.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
        tpl = rng.choice([NEG, NEU, POS][label])
        tick, name = rng.choice(COMPANIES)
        text = rng.choice(PREFIX) + tpl.format(co=name, n=rng.randint(2, 40)) + rng.choice(SUFFIX)
        if rng.random() < label_noise:  # simulate annotator disagreement
            label = rng.choice([l for l in (0, 1, 2) if l != label])
        d = start + timedelta(days=rng.randrange(days))
        rows.append({"date": d.isoformat(), "ticker": tick, "headline": text, "label": label})
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data/sample/headlines_sample.csv"
    make_sample().to_csv(out, index=False)
    print("wrote", out)

"""The 'desk': turn scored headlines into per-ticker daily sentiment, alerts and a report."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import LABELS


def score_frame(df: pd.DataFrame, model) -> pd.DataFrame:
    """Add p_neg/p_neu/p_pos, a score in [-1, 1] (p_pos - p_neg) and the predicted label."""
    out = df.copy()
    proba = model.predict_proba(out["headline"].tolist())
    out[["p_neg", "p_neu", "p_pos"]] = proba
    out["score"] = out["p_pos"] - out["p_neg"]
    out["pred"] = [LABELS[i] for i in proba.argmax(1)]
    return out


def daily_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    s = scored.copy()
    s["date"] = pd.to_datetime(s["date"])
    g = s.groupby(["ticker", "date"])
    daily = g.agg(mean_score=("score", "mean"), n=("score", "size"),
                  pct_pos=("pred", lambda x: float((x == "positive").mean())),
                  pct_neg=("pred", lambda x: float((x == "negative").mean()))).reset_index()
    daily = daily.sort_values(["ticker", "date"])
    daily["roll3"] = daily.groupby("ticker")["mean_score"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    return daily.reset_index(drop=True)


def build_alerts(daily: pd.DataFrame, threshold: float, min_headlines: int, lookback_days: int) -> pd.DataFrame:
    """Flag tickers whose latest day deviates from their trailing mean by >= threshold."""
    rows = []
    for tick, g in daily.groupby("ticker"):
        g = g.sort_values("date")
        latest = g.iloc[-1]
        prior = g[(g["date"] < latest["date"]) &
                  (g["date"] >= latest["date"] - pd.Timedelta(days=lookback_days))]
        if latest["n"] < min_headlines or prior.empty:
            continue
        delta = float(latest["mean_score"] - prior["mean_score"].mean())
        if abs(delta) >= threshold:
            rows.append({"ticker": tick, "date": latest["date"].date().isoformat(),
                         "latest_score": float(latest["mean_score"]), "delta_vs_trailing": delta,
                         "direction": "improving" if delta > 0 else "deteriorating", "n": int(latest["n"])})
    cols = ["ticker", "date", "latest_score", "delta_vs_trailing", "direction", "n"]
    return pd.DataFrame(rows, columns=cols).sort_values("delta_vs_trailing", key=abs, ascending=False,
                                                         ignore_index=True)


def plot_trend(daily: pd.DataFrame, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for tick, g in daily.groupby("ticker"):
        ax.plot(g["date"], g["roll3"], marker="o", ms=3, label=tick)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("sentiment (3-day rolling mean of p_pos - p_neg)")
    ax.set_title("Finance-News Sentiment Desk")
    ax.legend(ncol=3, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_markdown(scored: pd.DataFrame, daily: pd.DataFrame, alerts: pd.DataFrame,
                    model_name: str, top_k: int) -> str:
    latest = daily.sort_values("date").groupby("ticker").tail(1).sort_values("mean_score", ascending=False)
    lines = [f"# Finance-News Sentiment Desk report", "",
             f"Model: `{model_name}` | headlines scored: {len(scored)} | tickers: {scored['ticker'].nunique()}", "",
             "> Research/education tool. Sentiment scores are not investment advice.", "",
             "## Latest sentiment by ticker", "",
             "| Ticker | Date | Mean score | Headlines | % positive | % negative |", "|---|---|---|---|---|---|"]
    for _, r in latest.iterrows():
        lines.append(f"| {r['ticker']} | {r['date'].date()} | {r['mean_score']:+.2f} | {int(r['n'])} | "
                     f"{r['pct_pos']:.0%} | {r['pct_neg']:.0%} |")
    lines += ["", "## Alerts", ""]
    if alerts.empty:
        lines.append("No ticker moved beyond the alert threshold.")
    else:
        for _, a in alerts.iterrows():
            lines.append(f"- **{a['ticker']}** {a['direction']} ({a['delta_vs_trailing']:+.2f} vs trailing mean, "
                         f"{a['n']} headlines on {a['date']})")
    lines += ["", "## Most extreme headlines", ""]
    for tick, g in scored.groupby("ticker"):
        g = g.sort_values("score")
        lines.append(f"### {tick}")
        for _, r in g.tail(top_k)[::-1].iterrows():
            lines.append(f"- (+{r['score']:.2f}) {r['headline']}")
        for _, r in g.head(top_k).iterrows():
            lines.append(f"- ({r['score']:+.2f}) {r['headline']}")
        lines.append("")
    lines += ["![trend](sentiment_trend.png)", ""]
    return "\n".join(lines)


def run_desk(df: pd.DataFrame, model, model_name: str, cfg: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = cfg["desk"]
    scored = score_frame(df, model)
    daily = daily_sentiment(scored)
    alerts = build_alerts(daily, d["alert_threshold"], d["min_headlines"], d["lookback_days"])
    scored.to_csv(out_dir / "scored_headlines.csv", index=False)
    daily.to_csv(out_dir / "daily_sentiment.csv", index=False)
    plot_trend(daily, out_dir / "sentiment_trend.png")
    (out_dir / "desk_report.md").write_text(
        render_markdown(scored, daily, alerts, model_name, d["top_k_headlines"]), encoding="utf-8")
    return {"scored": scored, "daily": daily, "alerts": alerts}

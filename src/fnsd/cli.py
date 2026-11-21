from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from .config import load_config
from .data.loaders import load_hf_financial_news, load_labeled, split


def _dataset(cfg: dict, name: str) -> pd.DataFrame:
    if name == "sample":
        return load_labeled(cfg["paths"]["sample_csv"])
    d = cfg["data"]
    return load_hf_financial_news(d["hf_dataset"], d["hf_max_rows"], cfg["seed"], cfg["paths"]["external_csv"])


def _splits(cfg: dict, name: str):
    d = cfg["data"]
    return split(_dataset(cfg, name), d["val_frac"], d["test_frac"], cfg["seed"])


def cmd_make_sample(args, cfg):
    from .data.synthetic import make_sample
    out = Path(cfg["paths"]["sample_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    make_sample(seed=cfg["seed"]).to_csv(out, index=False)
    print("wrote", out)


def cmd_train(args, cfg):
    from .evaluate import evaluate_model, save_metrics
    from .predict import model_dir
    from .train import train_bilstm, train_tfidf

    train_df, val_df, test_df = _splits(cfg, args.data)
    out = model_dir(cfg, args.model, args.data)
    print(f"train {len(train_df)} / val {len(val_df)} / test {len(test_df)}  -> {out}")
    model = train_tfidf(train_df, cfg, out) if args.model == "tfidf" else train_bilstm(train_df, val_df, cfg, out)
    metrics = evaluate_model(model, test_df)
    save_metrics(metrics, out)
    print(f"TEST accuracy {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}")


def cmd_evaluate(args, cfg):
    from .evaluate import evaluate_model, save_metrics
    from .predict import load_model, model_dir

    _, _, test_df = _splits(cfg, args.data)
    model = load_model(cfg, args.model, args.trained_on)
    metrics = evaluate_model(model, test_df)
    out = model_dir(cfg, args.model, args.trained_on)
    save_metrics(metrics, out if args.model != "finbert" else Path(cfg["paths"]["artifacts_dir"]) / "finbert",
                 stem=f"metrics_on_{args.data}")
    print(f"{args.model} on {args.data} test: accuracy {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}")


def cmd_fetch(args, cfg):
    from .data.sources import fetch_many
    tickers = {t: cfg["tickers"].get(t, t) for t in args.tickers.split(",")} if args.tickers else cfg["tickers"]
    df = fetch_many(args.source, tickers, args.days, cfg)
    out = Path(cfg["paths"]["live_dir"]) / f"news_{args.source}_{date.today().isoformat()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} headlines -> {out}")


def cmd_desk(args, cfg):
    from .desk import run_desk
    from .predict import load_model

    path = args.input or cfg["paths"]["sample_csv"]
    df = pd.read_csv(path).dropna(subset=["headline"])
    for col, default in (("ticker", "NA"), ("date", date.today().isoformat())):
        if col not in df.columns:
            df[col] = default
    model = load_model(cfg, args.model, args.trained_on)
    res = run_desk(df, model, args.model, cfg, cfg["paths"]["reports_dir"])
    print(f"scored {len(res['scored'])} headlines; {len(res['alerts'])} alert(s); "
          f"report -> {cfg['paths']['reports_dir']}/desk_report.md")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fnsd", description="Finance-News Sentiment Desk")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("make-sample", help="(re)generate the synthetic sample CSV").set_defaults(fn=cmd_make_sample)

    t = sub.add_parser("train", help="train a model and evaluate on the held-out test split")
    t.add_argument("--model", choices=["tfidf", "bilstm"], default="bilstm")
    t.add_argument("--data", choices=["sample", "hf"], default="sample")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a model (incl. zero-shot finbert) on the test split")
    e.add_argument("--model", choices=["tfidf", "bilstm", "finbert"], default="bilstm")
    e.add_argument("--data", choices=["sample", "hf"], default="sample")
    e.add_argument("--trained-on", choices=["sample", "hf"], default="sample")
    e.set_defaults(fn=cmd_evaluate)

    f = sub.add_parser("fetch", help="download recent headlines from a free API (cached, rate-limited)")
    f.add_argument("--source", choices=["finnhub", "gdelt"], default="gdelt")
    f.add_argument("--tickers", default=None, help="comma list, e.g. AAPL,MSFT (default: config)")
    f.add_argument("--days", type=int, default=7)
    f.set_defaults(fn=cmd_fetch)

    d = sub.add_parser("desk", help="score headlines and write the sentiment report")
    d.add_argument("--input", default=None, help="CSV with headline[,ticker,date] (default: sample)")
    d.add_argument("--model", choices=["tfidf", "bilstm", "finbert"], default="bilstm")
    d.add_argument("--trained-on", choices=["sample", "hf"], default="sample")
    d.set_defaults(fn=cmd_desk)

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    args.fn(args, cfg)


if __name__ == "__main__":
    sys.exit(main())

# Finance-News Sentiment Desk

A small, reproducible deep-learning project: classify financial news headlines as
**negative / neutral / positive**, aggregate the scores per ticker and day, and produce a
"desk" report with trends and alerts.

```
headlines ──► model (BiLSTM+attention | FinBERT | TF-IDF baseline) ──► p(neg,neu,pos)
          └──► score = p_pos − p_neg ──► per-ticker daily mean ──► alerts + markdown report + chart
```

## Quick start (offline, ~1 minute, CPU only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # DL model
pip install -r requirements.txt
pip install -e .                      # or: export PYTHONPATH=src

fnsd train --model tfidf              # classical baseline
fnsd train --model bilstm             # deep model (writes artifacts/bilstm_sample/)
fnsd desk  --model bilstm             # writes reports/desk_report.md + sentiment_trend.png
pytest -q
```

Run all commands from the repository root (paths in `configs/default.yaml` are relative).

## Data

| Source | What | Size | Notes |
|---|---|---|---|
| `data/sample/headlines_sample.csv` | 900 **synthetic** headlines, *fictional* companies | ~100 KB | Bundled so everything runs offline and in CI. Template-generated, so scores on it are **optimistic** – it is for pipeline development, not a benchmark. |
| Hugging Face `zeroshot/twitter-financial-news-sentiment` | real labelled finance-news snippets | subsampled to ≤5000 rows (≈1 MB) | `--data hf` (needs `pip install datasets`). Check the dataset card for licence/terms. Cached to `data/external/` (git-ignored). |
| Finnhub company news API | live headlines per ticker | ≤200/ticker | free key, 60 calls/min |
| GDELT DOC 2.0 API | live headlines by company name | ≤100–250/query | no key; ≤1 request / 5 s |

Everything is far below the 500 MB limit; larger sources are subsampled (`data.hf_max_rows`).

### Using the real dataset (recommended for your write-up)

```bash
pip install datasets transformers
fnsd train --model tfidf  --data hf
fnsd train --model bilstm --data hf
fnsd evaluate --model finbert --data hf      # zero-shot pre-trained FinBERT for comparison
```

## Live news (optional)

```bash
export FINNHUB_API_KEY=...            # or copy .env.example -> .env locally (git-ignored)
fnsd fetch --source finnhub --tickers AAPL,MSFT --days 3
fnsd fetch --source gdelt   --days 3  # no key needed
fnsd desk --input data/live/news_gdelt_<date>.csv --model bilstm
```

API hygiene built in:
* **Cache** – every response is stored under `.cache/api/` (TTL `api.cache_ttl_hours`); re-runs make no network calls.
* **Rate limiting** – `RateLimiter` enforces a minimum gap per source; `429` responses are retried with `Retry-After`/exponential back-off.
* **No secrets in the repo** – keys are read from environment variables only, are sent in HTTP headers, and are excluded from cache keys and cached files (covered by a unit test). In CI the key comes from `secrets.FINNHUB_API_KEY` (see `.github/workflows/daily-desk.yml`).

## Models

* **BiLSTM + attention** (`src/fnsd/models/bilstm.py`) – embeddings trained from scratch, packed bidirectional LSTM, additive attention over real tokens (padding masked), class-weighted cross-entropy, Adam, gradient clipping, early stopping on validation macro-F1. Attention weights can be returned for explanations.
* **FinBERT** (`ProsusAI/finbert`) – pre-trained transformer, inference only (optional).
* **TF-IDF + logistic regression** – strong, cheap baseline every DL result should be compared against.

Metrics: accuracy, **macro-F1** (classes are imbalanced), per-class P/R/F1 and a confusion matrix, written to `artifacts/<model>_<data>/metrics.json` and `metrics_confusion.png`. Split is stratified 70/15/15 with a fixed seed.

## Desk output

`reports/` contains `desk_report.md`, `daily_sentiment.csv`, `scored_headlines.csv`, `sentiment_trend.png`. An **alert** fires when a ticker's latest daily mean score differs from its trailing mean (default 7 days) by ≥ `desk.alert_threshold` with at least `desk.min_headlines` headlines.

## Privacy and ethics

* Only public news headlines and synthetic text are used. **No personal or patient data** – and none should ever be added to this repo (the MedMNIST note in the brief does not apply here; this project uses no images).
* Scores are a research/teaching signal, **not investment advice**. Headline sentiment is noisy, the sample set is synthetic, and models can be wrong on sarcasm, negation and mixed-news headlines.

## Repo layout

```
configs/default.yaml        all hyper-parameters, paths, API limits, tickers
data/sample/                bundled synthetic CSV (+ generator in src/fnsd/data/synthetic.py)
src/fnsd/
  cache.py                  disk cache + rate limiter
  text.py                   tokenizer + vocabulary
  data/                     loaders, synthetic generator, Finnhub/GDELT clients
  models/bilstm.py          PyTorch model
  predict.py                common predict_proba interface (tfidf / bilstm / finbert)
  train.py evaluate.py desk.py cli.py
tests/                      unit tests (network is mocked; torch tests skip if torch missing)
.github/workflows/          ci.yml (tests + smoke run), daily-desk.yml (uses CI secret)
```

## Ideas for extensions

Ablations (BiLSTM vs. TF-IDF vs. FinBERT, with/without attention), error analysis on
mixed-sentiment headlines, calibration plots, fine-tuning FinBERT on the HF subset,
correlating daily scores with next-day returns (use a free price source and mind look-ahead bias),
a Streamlit front-end.

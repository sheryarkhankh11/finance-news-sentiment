"""Free news APIs (Finnhub, GDELT) with local caching, rate limiting and 429 back-off.

* API keys come ONLY from environment variables (e.g. a CI secret) - never from files in
  the repo, and never inside cache keys or cached payloads.
* Responses are cached on disk (TTL from config), so re-runs make zero network calls.
"""
from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from ..cache import DiskCache, RateLimiter, make_key

FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
COLUMNS = ["date", "ticker", "headline", "source", "url"]


def cached_get_json(url, params, headers, cache: DiskCache, limiter: RateLimiter,
                    session=None, timeout: int = 20, retries: int = 3, sleep=time.sleep):
    key = make_key(url, params)  # headers (secrets) deliberately excluded
    hit = cache.get(key)
    if hit is not None:
        return hit
    session = session or requests.Session()
    for attempt in range(retries + 1):
        limiter.wait()
        resp = session.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 429:
            try:
                delay = float(resp.headers.get("Retry-After", ""))
            except ValueError:
                delay = 2.0 ** (attempt + 1)
            sleep(min(delay, 60.0))
            continue
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(f"Non-JSON response from {url}: {resp.text[:120]!r}") from exc
        cache.set(key, data)
        return data
    raise RuntimeError(f"Rate-limited by {url} after {retries + 1} attempts; try again later")


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.drop_duplicates(subset=["ticker", "headline"]).reset_index(drop=True)


def fetch_finnhub(ticker: str, start: date, end: date, cache: DiskCache, limiter: RateLimiter,
                  api_key: str | None = None, session=None, max_rows: int = 200) -> pd.DataFrame:
    api_key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY is not set (use an env var / CI secret; never commit it).")
    params = {"symbol": ticker, "from": start.isoformat(), "to": end.isoformat()}
    data = cached_get_json(FINNHUB_URL, params, {"X-Finnhub-Token": api_key}, cache, limiter, session)
    rows = []
    for item in data[:max_rows]:
        headline = (item.get("headline") or "").strip()
        if not headline:
            continue
        d = datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc).date().isoformat()
        rows.append({"date": d, "ticker": ticker, "headline": headline,
                     "source": item.get("source", ""), "url": item.get("url", "")})
    return _frame(rows)


def fetch_gdelt(ticker: str, name: str, days: int, cache: DiskCache, limiter: RateLimiter,
                session=None, max_rows: int = 100) -> pd.DataFrame:
    params = {"query": f'"{name}" sourcelang:english', "mode": "artlist", "format": "json",
              "maxrecords": str(min(max_rows, 250)), "timespan": f"{days}d", "sort": "datedesc"}
    data = cached_get_json(GDELT_URL, params, None, cache, limiter, session)
    rows = []
    for art in data.get("articles", []):
        title = (art.get("title") or "").strip()
        if not title:
            continue
        seen = art.get("seendate", "")  # e.g. 20250101T120000Z
        try:
            d = datetime.strptime(seen, "%Y%m%dT%H%M%SZ").date().isoformat()
        except ValueError:
            d = date.today().isoformat()
        rows.append({"date": d, "ticker": ticker, "headline": title,
                     "source": art.get("domain", ""), "url": art.get("url", "")})
    return _frame(rows)


def fetch_many(source: str, tickers: dict[str, str], days: int, cfg: dict, session=None) -> pd.DataFrame:
    """Fetch headlines for several tickers; returns one combined DataFrame."""
    api = cfg["api"]
    cache = DiskCache(cfg["paths"]["cache_dir"], ttl_seconds=api["cache_ttl_hours"] * 3600)
    limiter = RateLimiter(api[source]["min_interval_s"])
    end = date.today()
    start = end - timedelta(days=days)
    frames = []
    for tick, name in tickers.items():
        if source == "finnhub":
            frames.append(fetch_finnhub(tick, start, end, cache, limiter, session=session,
                                        max_rows=api["max_per_ticker"]))
        elif source == "gdelt":
            frames.append(fetch_gdelt(tick, name, days, cache, limiter, session=session,
                                      max_rows=api["max_per_ticker"]))
        else:
            raise ValueError(f"unknown source {source!r}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)

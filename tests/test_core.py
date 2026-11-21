import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from fnsd.cache import DiskCache, RateLimiter, make_key
from fnsd.data.loaders import split
from fnsd.data.synthetic import make_sample
from fnsd.desk import build_alerts, daily_sentiment, score_frame
from fnsd.text import PAD_ID, UNK_ID, Vocab, tokenize


class TextTests(unittest.TestCase):
    def test_tokenize_numbers_and_urls(self):
        self.assertEqual(tokenize("Profit up 12.5% see https://x.co/a"), ["profit", "up", "<num>", "%", "see"])

    def test_vocab_roundtrip_and_padding(self):
        v = Vocab.build(["profit jumps", "profit falls"], min_freq=1)
        ids = v.encode("profit jumps zzz", 6)
        self.assertEqual(len(ids), 6)
        self.assertEqual(ids[2], UNK_ID)
        self.assertEqual(ids[3:], [PAD_ID] * 3)
        with tempfile.TemporaryDirectory() as d:
            v.save(Path(d) / "v.json")
            self.assertEqual(Vocab.load(Path(d) / "v.json").itos, v.itos)

    def test_empty_text_has_one_real_token(self):
        v = Vocab.build(["a b"], 1)
        self.assertEqual(v.encode("!!!", 4)[0], UNK_ID)


class CacheTests(unittest.TestCase):
    def test_set_get_and_ttl(self):
        with tempfile.TemporaryDirectory() as d:
            c = DiskCache(d, ttl_seconds=0.05)
            k = make_key("http://x", {"a": 1})
            self.assertIsNone(c.get(k))
            c.set(k, [])
            self.assertEqual(c.get(k), [])          # empty list is a valid hit
            time.sleep(0.08)
            self.assertIsNone(c.get(k))             # expired

    def test_rate_limiter_waits(self):
        t = [0.0]
        slept = []
        rl = RateLimiter(5.0, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
        rl.wait()
        t[0] += 1.0
        rl.wait()
        self.assertEqual(slept, [4.0])


class FakeResp:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code, self._p, self.headers, self.text = status, payload, headers or {}, ""

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


class SourceTests(unittest.TestCase):
    def test_finnhub_cache_secret_and_retry(self):
        from datetime import date
        from fnsd.data.sources import fetch_finnhub

        payload = [{"headline": "Acme beats estimates", "datetime": 1735776000, "source": "x", "url": "u"},
                   {"headline": "Acme beats estimates", "datetime": 1735776000, "source": "x", "url": "u"},
                   {"headline": "", "datetime": 1}]
        sess = FakeSession([FakeResp(429, headers={"Retry-After": "0"}), FakeResp(200, payload)])
        secret = "SUPER-SECRET-KEY"
        with tempfile.TemporaryDirectory() as d:
            cache, rl = DiskCache(d, 3600), RateLimiter(0)
            df = fetch_finnhub("ACME", date(2025, 1, 1), date(2025, 1, 2), cache, rl, api_key=secret, session=sess)
            self.assertEqual(len(df), 1)             # dedup + empty dropped
            self.assertEqual(sess.calls, 2)          # one 429 then success
            fetch_finnhub("ACME", date(2025, 1, 1), date(2025, 1, 2), cache, rl, api_key=secret, session=sess)
            self.assertEqual(sess.calls, 2)          # second call served from cache
            for f in Path(d).rglob("*.json"):        # key must never reach disk
                self.assertNotIn(secret, f.read_text())

    def test_missing_key_raises(self):
        import os
        from datetime import date
        from fnsd.data.sources import fetch_finnhub

        os.environ.pop("FINNHUB_API_KEY", None)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                fetch_finnhub("A", date.today(), date.today(), DiskCache(d), RateLimiter(0))

    def test_gdelt_parse(self):
        from fnsd.data.sources import fetch_gdelt

        payload = {"articles": [{"title": "Apple shares rise", "seendate": "20250102T101500Z", "domain": "d", "url": "u"}]}
        with tempfile.TemporaryDirectory() as d:
            df = fetch_gdelt("AAPL", "Apple", 7, DiskCache(d), RateLimiter(0), session=FakeSession([FakeResp(200, payload)]))
        self.assertEqual(df.loc[0, "date"], "2025-01-02")


class DataAndBaselineTests(unittest.TestCase):
    def test_sample_reproducible_and_split_disjoint(self):
        a, b = make_sample(200, seed=1), make_sample(200, seed=1)
        pd.testing.assert_frame_equal(a, b)
        tr, va, te = split(make_sample(400, seed=2), 0.15, 0.15, 0)
        self.assertEqual(len(tr) + len(va) + len(te), 400)
        self.assertEqual(set(tr["label"]), {0, 1, 2})

    def test_tfidf_learns_and_probs_sum_to_one(self):
        from fnsd.evaluate import evaluate_model
        from fnsd.predict import TfidfModel

        tr, _, te = split(make_sample(600, seed=3), 0.15, 0.15, 0)
        m = TfidfModel().fit(tr["headline"], tr["label"])
        p = m.predict_proba(te["headline"])
        self.assertTrue(abs(p.sum(1) - 1).max() < 1e-6)
        self.assertGreater(evaluate_model(m, te)["macro_f1"], 0.8)


class DeskTests(unittest.TestCase):
    class Const:
        """Fake model: 'good' headlines positive, 'bad' negative."""
        def predict_proba(self, texts):
            import numpy as np
            return np.array([[0.05, 0.05, 0.9] if "good" in t else [0.9, 0.05, 0.05] for t in texts])

    def test_daily_and_alerts(self):
        rows = [("2025-01-0%d" % d, "AAA", "good news") for d in range(1, 6)]
        rows += [("2025-01-06", "AAA", "bad news"), ("2025-01-06", "AAA", "bad again")]
        rows += [("2025-01-0%d" % d, "BBB", "good news") for d in range(1, 7)]
        df = pd.DataFrame(rows, columns=["date", "ticker", "headline"])
        scored = score_frame(df, self.Const())
        daily = daily_sentiment(scored)
        alerts = build_alerts(daily, threshold=0.25, min_headlines=2, lookback_days=7)
        self.assertEqual(list(alerts["ticker"]), ["AAA"])
        self.assertEqual(alerts.loc[0, "direction"], "deteriorating")


try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class BiLSTMTests(unittest.TestCase):
    def test_forward_shapes_and_attention_mask(self):
        import torch
        from fnsd.models.bilstm import AttnBiLSTM

        m = AttnBiLSTM(vocab_size=50, emb_dim=8, hidden=8).eval()
        x = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 9]])
        logits, w = m(x, return_attention=True)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertTrue(torch.allclose(w.sum(1), torch.ones(2), atol=1e-5))
        self.assertEqual(float(w[0, 3:].detach().sum()), 0.0)  # no attention on padding

    def test_one_epoch_training_and_reload(self):
        import copy
        from fnsd.predict import BiLSTMModel
        from fnsd.train import train_bilstm
        from fnsd.config import load_config

        cfg = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
        cfg["bilstm"].update(epochs=2, emb_dim=16, hidden=16)
        tr, va, te = split(make_sample(300, seed=4), 0.15, 0.15, 0)
        with tempfile.TemporaryDirectory() as d:
            model = train_bilstm(tr, va, cfg, d)
            p1 = model.predict_proba(te["headline"][:5])
            p2 = BiLSTMModel.load(d).predict_proba(te["headline"][:5])
            self.assertTrue(abs(p1 - p2).max() < 1e-5)
            self.assertTrue(abs(p1.sum(1) - 1).max() < 1e-5)


if __name__ == "__main__":
    unittest.main()

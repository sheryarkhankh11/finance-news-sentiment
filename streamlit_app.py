"""Streamlit app for Finance-News Sentiment Desk.

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Cloud: push this repo to GitHub, then connect the repo at
https://share.streamlit.io. The entry point is `streamlit_app.py`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fnsd.cli import main as cli_main
from fnsd.config import load_config
from fnsd.desk import run_desk
from fnsd.predict import load_model, model_dir
from fnsd.train import train_bilstm, train_tfidf
from fnsd.data.loaders import load_labeled, split

st.set_page_config(page_title="Finance-News Sentiment Desk", page_icon="\U0001F4CA", layout="wide")

CFG = load_config()
LABELS = ["negative", "neutral", "positive"]
LABEL_EMOJI = {"negative": "\U0001F534", "neutral": "\U0001F7E0", "positive": "\U0001F7E2"}


def _ensure_model(model: str, data: str = "sample"):
    out = model_dir(CFG, model, data)
    if not out.exists():
        with st.spinner(f"Training {model} on {data} data (one-time)..."):
            train_df, val_df, test_df = split(load_labeled(CFG["paths"]["sample_csv"]),
                                              CFG["data"]["val_frac"],
                                              CFG["data"]["test_frac"], CFG["seed"])
            if model == "tfidf":
                train_tfidf(train_df, CFG, out)
            else:
                train_bilstm(train_df, val_df, CFG, out)
            from fnsd.evaluate import evaluate_model, save_metrics
            m = load_model(CFG, model, data)
            save_metrics(evaluate_model(m, test_df), out)
    return load_model(CFG, model, data)


@st.cache_resource(show_spinner="Loading model...")
def _cached_model(model: str, data: str = "sample"):
    return _ensure_model(model, data)


def _predict_df(df: pd.DataFrame, model) -> pd.DataFrame:
    from fnsd.desk import score_frame
    return score_frame(df.dropna(subset=["headline"]), model)


st.title("\U0001F4CA Finance-News Sentiment Desk")
st.caption("Classify financial news headlines as negative / neutral / positive and build a desk report.")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Model", ["tfidf", "bilstm", "finbert"], index=0)
    data_choice = st.selectbox("Data", ["sample", "hf"], index=0)
    if st.button("Train / (re)load model"):
        try:
            m = _ensure_model(model_choice, data_choice)
            st.success(f"{model_choice} ready on {data_choice} data")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
    st.divider()
    st.markdown("**Deploy**: connect this GitHub repo at https://share.streamlit.io")

tab_single, tab_batch, tab_desk = st.tabs(
    ["\u2709 Single headline", "\U0001F4CB Batch CSV", "\U0001F4CA Desk report"]
)

with tab_single:
    st.subheader("Score one headline")
    text = st.text_area("Headline", "Acme Corp beats quarterly earnings estimates as revenue climbs 12%")
    if st.button("Predict", key="predict_single"):
        try:
            model = _cached_model(model_choice, data_choice)
            proba = model.predict_proba([text])[0]
            pred_idx = int(proba.argmax())
            pred = LABELS[pred_idx]
            c1, c2, c3 = st.columns(3)
            for col, (lab, p) in zip((c1, c2, c3), zip(LABELS, proba)):
                with col:
                    st.metric(f"{LABEL_EMOJI[lab]} {lab.title()}", f"{p:.1%}")
            st.success(f"Prediction: {LABEL_EMOJI[pred]} **{pred.title()}** (score {proba[2]-proba[0]:+.2f})")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab_batch:
    st.subheader("Batch score from CSV")
    up = st.file_uploader("Upload CSV with a `headline` column (optional: `ticker`, `date`)", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            if "headline" not in df.columns:
                st.error("CSV must contain a `headline` column.")
            else:
                model = _cached_model(model_choice, data_choice)
                res = _predict_df(df, model)
                st.dataframe(res[["headline", "pred", "score", "p_neg", "p_neu", "p_pos"]].head(50), use_container_width=True)
                csv = res.to_csv(index=False).encode("utf-8")
                st.download_button("Download scored CSV", csv, "scored_headlines.csv", "text/csv")
        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")

with tab_desk:
    st.subheader("Desk report on sample data")
    if st.button("Generate desk report", key="gen_desk"):
        try:
            model = _cached_model(model_choice, data_choice)
            df = load_labeled(CFG["paths"]["sample_csv"])
            res = run_desk(df, model, model_choice, CFG, Path("reports"))
            st.success(f"Scored {len(res['scored'])} headlines, {len(res['alerts'])} alert(s)")
            st.subheader("Alerts")
            if res["alerts"].empty:
                st.info("No alerts.")
            else:
                st.dataframe(res["alerts"], use_container_width=True)
            st.subheader("Daily sentiment (last 10 rows)")
            st.dataframe(res["daily"].tail(10), use_container_width=True)
            report_path = Path("reports") / "desk_report.md"
            if report_path.exists():
                st.markdown(report_path.read_text(encoding="utf-8"))
            chart_path = Path("reports") / "sentiment_trend.png"
            if chart_path.exists():
                st.image(str(chart_path), caption="Sentiment trend")
        except Exception as exc:
            st.error(f"Desk report failed: {exc}")

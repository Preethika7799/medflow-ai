from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RESULTS = Path("evaluation_results")


def load_latest() -> dict | None:
    if not RESULTS.exists():
        return None
    files = sorted(RESULTS.glob("eval_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


st.title("Evaluation metrics")
latest = load_latest()
if latest:
    agg = latest.get("aggregate_metrics", {})
    if agg:
        df = pd.DataFrame({"metric": list(agg.keys()), "score": list(agg.values())})
        st.plotly_chart(px.bar(df, x="metric", y="score", title="Latest aggregate scores"), use_container_width=True)
    st.dataframe(pd.DataFrame(latest.get("per_question_results", [])))
else:
    st.info("Run `make evaluate` to produce evaluation JSON in evaluation_results/.")

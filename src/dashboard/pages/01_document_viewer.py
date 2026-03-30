from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE = os.environ.get("MEDFLOW_API_BASE", "http://127.0.0.1:8000")


@st.cache_data(ttl=30)
def fetch_documents() -> list[dict]:
    """List documents from API."""
    try:
        r = httpx.get(f"{API_BASE}/api/v1/documents", timeout=30.0)
        r.raise_for_status()
        return list(r.json())
    except Exception as e:
        st.error(f"Could not reach API: {e}")
        return []


st.title("Document viewer")
rows = fetch_documents()
if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    if "doc_type" in df.columns:
        fig = px.pie(df, names="doc_type", title="Class distribution")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No documents yet — run ingestion or `make seed`.")

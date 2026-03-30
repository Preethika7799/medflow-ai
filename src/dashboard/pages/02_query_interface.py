from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE = os.environ.get("MEDFLOW_API_BASE", "http://127.0.0.1:8000")

st.title("Ask your documents")
query = st.text_area("Question", height=100)
strategy = st.selectbox("Strategy", ["auto", "dense", "sparse", "hybrid"])
if st.button("Run query"):
    try:
        payload = {"query": query, "strategy": strategy}
        r = httpx.post(f"{API_BASE}/api/v1/query", json=payload, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        st.success("Response")
        st.markdown(data.get("answer", ""))
        st.json(
            {
                "citations": data.get("citations"),
                "strategy_used": data.get("strategy_used"),
                "metrics": data.get("metrics"),
            },
        )
    except Exception as e:
        st.error(str(e))

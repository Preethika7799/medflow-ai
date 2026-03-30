from __future__ import annotations

import os

import httpx
import streamlit as st
from medflow.providers.metrics import get_provider_metrics

st.title("System health")
st.subheader("In-process LLM metrics")
st.json(get_provider_metrics().snapshot())

API_BASE = os.environ.get("MEDFLOW_API_BASE", "http://127.0.0.1:8000")
if st.button("Ping API /health"):
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=10.0)
        st.json(r.json())
    except Exception as e:
        st.error(str(e))

st.caption("Langfuse: set MEDFLOW / Langfuse keys and open Langfuse UI from docker-compose.")

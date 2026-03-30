from __future__ import annotations

import streamlit as st


def section(title: str) -> None:
    """Render a styled section header."""
    st.markdown(f"### {title}")

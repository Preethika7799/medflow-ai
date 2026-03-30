from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


def bar_chart(labels: list[str], values: list[float], title: str) -> go.Figure:
    """Build a simple bar chart."""
    return go.Figure(data=[go.Bar(x=labels, y=values)], layout=go.Layout(title=title))

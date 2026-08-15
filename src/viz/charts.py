"""Wrappers Plotly padronizados com o tema do app.

Todas as páginas devem usar estas funções (ou `apply_theme`) para que os
gráficos tenham tipografia, cores e hover consistentes.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.styling import GOLD, LIGHT, NAVY, PALETTE, TEAL, channel_color


def apply_theme(fig: go.Figure, height: int = 420, legend_bottom: bool = True) -> go.Figure:
    """Aplica layout padrão do app a qualquer figura Plotly."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family="Segoe UI, Helvetica, sans-serif", size=13, color="#2D3748"),
        title=dict(font=dict(size=16, color=NAVY)),
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    if legend_bottom:
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None)
        )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#EDF2F7", zerolinecolor="#E2E8F0")
    return fig


def line_chart(
    df: pd.DataFrame, x: str, y: str | Sequence[str], title: str = "", **kwargs
) -> go.Figure:
    fig = px.line(df, x=x, y=y, title=title, color_discrete_sequence=PALETTE, **kwargs)
    fig.update_traces(line=dict(width=2.2))
    return apply_theme(fig)


def bar_chart(
    df: pd.DataFrame, x: str, y: str, title: str = "", color: str | None = None, **kwargs
) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, title=title, color_discrete_sequence=PALETTE, **kwargs)
    return apply_theme(fig)


def grouped_bar(
    df: pd.DataFrame, x: str, y: str, color: str, title: str = "", barmode: str = "group"
) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, title=title,
                 color_discrete_sequence=PALETTE)
    return apply_theme(fig, height=460)


def stacked_area(df: pd.DataFrame, x: str, columns: Iterable[str], title: str = "") -> go.Figure:
    """Área empilhada (usada na decomposição de contribuições do MMM)."""
    fig = go.Figure()
    for i, col in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[col],
                name=col,
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5, color=channel_color(col, i)),
                fillcolor=channel_color(col, i),
                hovertemplate=f"<b>{col}</b>: %{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_layout(title=title)
    return apply_theme(fig, height=460)


def waterfall_compare(labels: Sequence[str], current: Sequence[float], optimal: Sequence[float],
                      title: str = "") -> go.Figure:
    """Barras lado a lado: alocação atual vs. ótima."""
    fig = go.Figure()
    fig.add_bar(x=list(labels), y=list(current), name="Alocação atual", marker_color=NAVY)
    fig.add_bar(x=list(labels), y=list(optimal), name="Alocação ótima", marker_color=GOLD)
    fig.update_layout(barmode="group", title=title)
    return apply_theme(fig, height=440)


def heatmap(df: pd.DataFrame, title: str = "", colorscale: str = "Blues") -> go.Figure:
    fig = px.imshow(df, text_auto=".2f", aspect="auto", title=title, color_continuous_scale=colorscale)
    return apply_theme(fig, height=480, legend_bottom=False)


def sankey(labels: Sequence[str], source: Sequence[int], target: Sequence[int],
           value: Sequence[float], title: str = "") -> go.Figure:
    colors = [channel_color(lbl, i) for i, lbl in enumerate(labels)]
    fig = go.Figure(
        go.Sankey(
            node=dict(label=list(labels), pad=18, thickness=16,
                      line=dict(color="#FFFFFF", width=0.5), color=colors),
            link=dict(source=list(source), target=list(target), value=list(value),
                      color="rgba(28,114,147,0.28)"),
        )
    )
    fig.update_layout(title=title)
    return apply_theme(fig, height=520, legend_bottom=False)


def funnel(labels: Sequence[str], values: Sequence[float], title: str = "") -> go.Figure:
    fig = go.Figure(
        go.Funnel(y=list(labels), x=list(values), marker=dict(color=PALETTE[: len(labels)]),
                  textinfo="value+percent initial")
    )
    fig.update_layout(title=title)
    return apply_theme(fig, height=460, legend_bottom=False)


def gauge(value: float, title: str = "", suffix: str = "%") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": TEAL},
                "steps": [
                    {"range": [0, 50], "color": LIGHT},
                    {"range": [50, 95], "color": "#DCE6F1"},
                    {"range": [95, 100], "color": "#FBE7C6"},
                ],
                "threshold": {"line": {"color": GOLD, "width": 3}, "value": 95},
            },
        )
    )
    return apply_theme(fig, height=280, legend_bottom=False)

"""Wrappers Plotly padronizados com o tema dark do app.

Regras de layout adotadas para que nenhum texto cubra outro:
  * o título fica alinhado à esquerda, no topo, com margem própria;
  * a legenda vai ABAIXO do gráfico (nunca sobre o título ou sobre as barras);
  * todos os eixos usam `automargin`, então rótulos longos empurram a margem
    em vez de serem cortados ou sobrepostos;
  * rótulos de dado usam `uniformtext`, escondendo o texto que não couber.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.styling import (
    BORDER,
    GOLD,
    GRID,
    NAVY,
    PALETTE,
    SURFACE,
    TEAL,
    TEXT,
    TEXT_MUTED,
    channel_color,
)

FONT = "Segoe UI, Inter, Helvetica, sans-serif"

# Escalas contínuas legíveis sobre fundo escuro
SEQUENTIAL = ["#1A2145", "#2A4A75", "#2F7C97", "#38B2C4", "#8FD9E3"]
DIVERGING = ["#E5636A", "#5A4A6B", "#2E3766", "#2F7C97", "#38B2C4"]


def apply_theme(fig: go.Figure, height: int = 420, legend_bottom: bool = True) -> go.Figure:
    """Aplica o layout dark padrão do app a qualquer figura Plotly."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        # margem inferior generosa: é onde a legenda passa a viver
        margin=dict(l=10, r=20, t=64, b=90 if legend_bottom else 50),
        font=dict(family=FONT, size=13, color=TEXT),
        title=dict(
            font=dict(size=15.5, color=TEXT, family=FONT),
            x=0, xanchor="left", y=0.97, yanchor="top", pad=dict(b=14),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT, family=FONT, size=12)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        uniformtext=dict(minsize=9, mode="hide"),
        coloraxis_colorbar=dict(outlinewidth=0, tickfont=dict(color=TEXT_MUTED, size=11),
                                title=dict(font=dict(color=TEXT_MUTED, size=11))),
    )
    if legend_bottom:
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                title=None, font=dict(size=11.5, color=TEXT_MUTED),
                bgcolor="rgba(0,0,0,0)", itemsizing="constant",
            )
        )
    else:
        fig.update_layout(showlegend=fig.layout.showlegend)

    fig.update_xaxes(
        showgrid=False, linecolor=BORDER, zerolinecolor=BORDER, automargin=True,
        tickfont=dict(color=TEXT_MUTED, size=11.5),
        title=dict(font=dict(color=TEXT_MUTED, size=12)),
    )
    fig.update_yaxes(
        gridcolor=GRID, linecolor=BORDER, zerolinecolor=BORDER, automargin=True,
        tickfont=dict(color=TEXT_MUTED, size=11.5),
        title=dict(font=dict(color=TEXT_MUTED, size=12)),
    )
    # Anotações de vline/hline não devem colidir com o título
    for annotation in fig.layout.annotations or ():
        if annotation.font and annotation.font.color is None:
            annotation.font.color = TEXT_MUTED
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
    return apply_theme(fig, legend_bottom=color is not None)


def grouped_bar(
    df: pd.DataFrame, x: str, y: str, color: str, title: str = "", barmode: str = "group"
) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, title=title,
                 color_discrete_sequence=PALETTE)
    fig.update_layout(bargap=0.24, bargroupgap=0.06)
    return apply_theme(fig, height=470)


def stacked_area(df: pd.DataFrame, x: str, columns: Iterable[str], title: str = "") -> go.Figure:
    """Área empilhada (usada na decomposição de contribuições do MMM)."""
    fig = go.Figure()
    for i, col in enumerate(columns):
        color = channel_color(col, i)
        fig.add_trace(
            go.Scatter(
                x=df[x], y=df[col], name=col, mode="lines", stackgroup="one",
                line=dict(width=0.6, color=color), fillcolor=color,
                hovertemplate=f"<b>{col}</b>: %{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_layout(title=title)
    return apply_theme(fig, height=470)


def waterfall_compare(labels: Sequence[str], current: Sequence[float], optimal: Sequence[float],
                      title: str = "") -> go.Figure:
    """Barras lado a lado: alocação atual vs. ótima."""
    fig = go.Figure()
    fig.add_bar(x=list(labels), y=list(current), name="Alocação atual", marker_color=NAVY)
    fig.add_bar(x=list(labels), y=list(optimal), name="Alocação ótima", marker_color=GOLD)
    fig.update_layout(barmode="group", title=title, bargap=0.28)
    return apply_theme(fig, height=450)


def heatmap(df: pd.DataFrame, title: str = "", colorscale: str | list | None = None) -> go.Figure:
    fig = px.imshow(df, text_auto=".2f", aspect="auto", title=title,
                    color_continuous_scale=colorscale or SEQUENTIAL)
    fig.update_traces(textfont=dict(size=11, color=TEXT))
    return apply_theme(fig, height=480, legend_bottom=False)


def sankey(labels: Sequence[str], source: Sequence[int], target: Sequence[int],
           value: Sequence[float], title: str = "") -> go.Figure:
    colors = [channel_color(lbl, i) for i, lbl in enumerate(labels)]
    fig = go.Figure(
        go.Sankey(
            node=dict(label=list(labels), pad=22, thickness=16,
                      line=dict(color=BORDER, width=0.5), color=colors),
            link=dict(source=list(source), target=list(target), value=list(value),
                      color="rgba(56,178,196,0.22)"),
            textfont=dict(color=TEXT, size=12.5, family=FONT),
        )
    )
    fig.update_layout(title=title)
    return apply_theme(fig, height=540, legend_bottom=False)


def funnel(labels: Sequence[str], values: Sequence[float], title: str = "") -> go.Figure:
    fig = go.Figure(
        go.Funnel(y=list(labels), x=list(values), marker=dict(color=PALETTE[: len(labels)]),
                  textinfo="value+percent initial",
                  textfont=dict(color="#0F1428", size=12, family=FONT))
    )
    fig.update_layout(title=title)
    return apply_theme(fig, height=470, legend_bottom=False)


def gauge(value: float, title: str = "", suffix: str = "%") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"color": TEXT}},
            title={"text": title, "font": {"size": 13, "color": TEXT_MUTED}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": BORDER,
                         "tickfont": {"color": TEXT_MUTED, "size": 10}},
                "bar": {"color": TEAL},
                "bgcolor": SURFACE,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#161C3A"},
                    {"range": [50, 95], "color": "#1E2650"},
                    {"range": [95, 100], "color": "#3A3050"},
                ],
                "threshold": {"line": {"color": GOLD, "width": 3}, "value": 95},
            },
        )
    )
    return apply_theme(fig, height=290, legend_bottom=False)

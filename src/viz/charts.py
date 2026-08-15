"""Wrappers Plotly padronizados, sensíveis ao tema claro/escuro.

Tudo que depende do tema é resolvido em tempo de render por `tokens()` — nunca
no import — para que a troca de tema no menu do Streamlit se reflita nos
gráficos no rerun seguinte.

Regras de layout adotadas para que nenhum texto cubra outro:
  * o título fica alinhado à esquerda, no topo, com margem própria;
  * a legenda vai ABAIXO do gráfico (nunca sobre o título ou sobre as barras);
  * todos os eixos usam `automargin`, então rótulos longos empurram a margem
    em vez de serem cortados ou sobrepostos;
  * rótulos de dado usam `uniformtext`, escondendo o que não couber.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.utils.styling import GOLD, NAVY, PALETTE, TEAL, channel_color, tokens

FONT = "Segoe UI, Inter, Helvetica, sans-serif"


def sequential() -> list[str]:
    """Escala contínua sequencial legível no tema ativo."""
    return list(tokens().sequential)


def diverging() -> list[str]:
    """Escala contínua divergente legível no tema ativo."""
    return list(tokens().diverging)


def apply_theme(fig: go.Figure, height: int = 420, legend_bottom: bool = True) -> go.Figure:
    """Aplica o layout padrão do app, no tema ativo, a qualquer figura Plotly."""
    t = tokens()
    fig.update_layout(
        template=t.plotly_template,
        height=height,
        # margem inferior generosa: é onde a legenda passa a viver
        margin=dict(l=10, r=20, t=64, b=90 if legend_bottom else 50),
        font=dict(family=FONT, size=13, color=t.text),
        title=dict(
            font=dict(size=15.5, color=t.text, family=FONT),
            x=0, xanchor="left", y=0.97, yanchor="top", pad=dict(b=14),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t.surface, bordercolor=t.border,
                        font=dict(color=t.text, family=FONT, size=12)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        uniformtext=dict(minsize=9, mode="hide"),
        coloraxis_colorbar=dict(outlinewidth=0, tickfont=dict(color=t.text_muted, size=11),
                                title=dict(font=dict(color=t.text_muted, size=11))),
    )
    if legend_bottom:
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                title=None, font=dict(size=11.5, color=t.text_muted),
                bgcolor="rgba(0,0,0,0)", itemsizing="constant",
            )
        )

    fig.update_xaxes(
        showgrid=False, linecolor=t.border, zerolinecolor=t.border, automargin=True,
        tickfont=dict(color=t.text_muted, size=11.5),
        title=dict(font=dict(color=t.text_muted, size=12)),
    )
    fig.update_yaxes(
        gridcolor=t.grid, linecolor=t.border, zerolinecolor=t.border, automargin=True,
        tickfont=dict(color=t.text_muted, size=11.5),
        title=dict(font=dict(color=t.text_muted, size=12)),
    )
    # Anotações de vline/hline seguem o texto secundário do tema
    for annotation in fig.layout.annotations or ():
        if annotation.font and annotation.font.color is None:
            annotation.font.color = t.text_muted
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
                    color_continuous_scale=colorscale or sequential())
    fig.update_traces(textfont=dict(size=11))
    return apply_theme(fig, height=480, legend_bottom=False)


def sankey(labels: Sequence[str], source: Sequence[int], target: Sequence[int],
           value: Sequence[float], title: str = "") -> go.Figure:
    t = tokens()
    colors = [channel_color(lbl, i) for i, lbl in enumerate(labels)]
    fig = go.Figure(
        go.Sankey(
            node=dict(label=list(labels), pad=22, thickness=16,
                      line=dict(color=t.border, width=0.5), color=colors),
            link=dict(source=list(source), target=list(target), value=list(value),
                      color="rgba(42,156,181,0.24)"),
            textfont=dict(color=t.text, size=12.5, family=FONT),
        )
    )
    fig.update_layout(title=title)
    return apply_theme(fig, height=540, legend_bottom=False)


def funnel(labels: Sequence[str], values: Sequence[float], title: str = "") -> go.Figure:
    fig = go.Figure(
        go.Funnel(y=list(labels), x=list(values), marker=dict(color=PALETTE[: len(labels)]),
                  textinfo="value+percent initial",
                  textfont=dict(color="#FFFFFF", size=12, family=FONT))
    )
    fig.update_layout(title=title)
    return apply_theme(fig, height=470, legend_bottom=False)


def gauge(value: float, title: str = "", suffix: str = "%") -> go.Figure:
    t = tokens()
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"color": t.text}},
            title={"text": title, "font": {"size": 13, "color": t.text_muted}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": t.border,
                         "tickfont": {"color": t.text_muted, "size": 10}},
                "bar": {"color": TEAL},
                "bgcolor": t.surface,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": t.surface},
                    {"range": [50, 95], "color": t.surface_alt},
                    {"range": [95, 100], "color": t.grid},
                ],
                "threshold": {"line": {"color": GOLD, "width": 3}, "value": 95},
            },
        )
    )
    return apply_theme(fig, height=290, legend_bottom=False)

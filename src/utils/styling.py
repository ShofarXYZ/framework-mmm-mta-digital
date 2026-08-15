"""Paleta, CSS custom e componentes visuais compartilhados pelo app.

A paleta reproduz o tema do framework "MMM x MTA — Digital Analytics":
navy (estrutura), teal/azul (camadas analíticas) e dourado (accent/ação).
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------
# Paleta
# --------------------------------------------------------------------------
NAVY = "#21295C"
TEAL = "#1C7293"
BLUE = "#065A82"
GOLD = "#E8A33D"
LIGHT = "#F2F5FA"
GREY = "#9AA5B1"

PALETTE = [NAVY, TEAL, GOLD, BLUE, "#5C6BC0", "#4DB6AC", "#C97B2C", "#7E57C2", GREY]

# Cores fixas por canal, para que o mesmo canal tenha a mesma cor em todas as páginas.
CHANNEL_COLORS = {
    # MMM
    "tv_spend": NAVY,
    "newspaper_spend": GREY,
    "instagram_spend": "#C13584",
    "google_ads_spend": TEAL,
    "youtube_spend": "#D62828",
    "influencer_spend": GOLD,
    "ott_spend": "#7E57C2",
    "competitor_spend": "#B0BEC5",
    "Base": "#CFD8DC",
    # MTA
    "Social Media": "#C13584",
    "Email": TEAL,
    "PPC": GOLD,
    "SEO": BLUE,
    "Referral": "#7E57C2",
}

# Camadas do framework -> (rótulo, cor)
LAYERS = {
    "MMM": ("Camada Estratégica — MMM (top-down)", NAVY),
    "MTA": ("Camada Tática — MTA / Atribuição (bottom-up)", TEAL),
    "AB": ("Camada de Validação Causal — Testes A/B", GOLD),
    "GOV": ("Camada de Governança — Learning Repository", BLUE),
}


def channel_color(name: str, fallback_index: int = 0) -> str:
    """Cor determinística por canal (com fallback pela paleta)."""
    if name in CHANNEL_COLORS:
        return CHANNEL_COLORS[name]
    return PALETTE[fallback_index % len(PALETTE)]


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------
_CSS = f"""
<style>
:root {{
    --navy: {NAVY};
    --teal: {TEAL};
    --blue: {BLUE};
    --gold: {GOLD};
}}
h1, h2, h3 {{ color: {NAVY}; font-weight: 700; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{ background-color: {LIGHT}; }}
div[data-testid="stMetric"] {{
    background: {LIGHT};
    border-left: 4px solid {GOLD};
    border-radius: 8px;
    padding: 12px 16px;
}}
div[data-testid="stMetricValue"] {{ color: {NAVY}; font-size: 1.5rem; }}
.mmm-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #FFFFFF;
}}
.mmm-context {{
    color: #4A5568;
    font-size: 0.95rem;
    margin: 6px 0 4px 0;
}}
.mmm-hr {{ border: none; border-top: 1px solid #E2E8F0; margin: 8px 0 18px 0; }}
.mmm-card {{
    background: {LIGHT};
    border-radius: 10px;
    padding: 14px 16px;
    border-top: 3px solid {TEAL};
    height: 100%;
}}
.mmm-card h4 {{ margin: 0 0 6px 0; color: {NAVY}; font-size: 0.98rem; }}
.mmm-card p {{ margin: 0; color: #4A5568; font-size: 0.85rem; }}
</style>
"""


def inject_css() -> None:
    """Injeta o CSS global do app (idempotente por rerun)."""
    st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Componentes
# --------------------------------------------------------------------------
def page_header(title: str, context: str, layer: str = "MMM") -> None:
    """Cabeçalho padrão: título, badge de camada e 1 frase de contexto."""
    label, color = LAYERS.get(layer, LAYERS["MMM"])
    st.markdown(
        f'<span class="mmm-badge" style="background:{color}">{label}</span>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(f'<p class="mmm-context">{context}</p>', unsafe_allow_html=True)
    st.markdown('<hr class="mmm-hr">', unsafe_allow_html=True)


def nav_card(title: str, description: str) -> None:
    """Card usado na Home para explicar o 'quando usar' de cada página."""
    st.markdown(
        f'<div class="mmm-card"><h4>{title}</h4><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def fmt_money(value: float, prefix: str = "R$ ") -> str:
    """Formata número grande de forma compacta e legível."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= div:
            return f"{prefix}{value / div:,.1f}{unit}".replace(",", ".")
    return f"{prefix}{value:,.0f}".replace(",", ".")

"""Paleta, CSS custom e componentes visuais compartilhados pelo app.

Tema **dark** derivado das cores do framework "MMM x MTA — Digital Analytics":
navy como base do fundo, teal/azul para as camadas analíticas e dourado como
accent. Todas as cores de texto foram escolhidas para manter contraste legível
sobre o fundo escuro (mínimo ~7:1 no texto corrido).
"""

from __future__ import annotations

import streamlit as st

# --------------------------------------------------------------------------
# Paleta — marca
# --------------------------------------------------------------------------
NAVY = "#4C5BC4"  # navy clareado, para ter contraste sobre o fundo escuro
TEAL = "#38B2C4"
BLUE = "#4A90D9"
GOLD = "#E8A33D"
GREY = "#8B97AD"

# --------------------------------------------------------------------------
# Paleta — superfícies e texto (dark)
# --------------------------------------------------------------------------
BG = "#0F1428"          # fundo da página
SURFACE = "#1A2145"     # cards, sidebar, métricas
SURFACE_ALT = "#232B52"  # hover / superfície elevada
BORDER = "#2E3766"      # divisórias
TEXT = "#E6EAF3"        # texto principal
TEXT_MUTED = "#A9B4CC"  # texto secundário / captions
GRID = "#242C55"        # grade dos gráficos

POSITIVE = "#3DBE7A"
NEGATIVE = "#E5636A"
WARNING = "#E8A33D"

PALETTE = [TEAL, GOLD, BLUE, "#7E8FE8", "#4DD4C0", "#D98BC7", "#C97B2C", "#9B7EE0", GREY]

# Cores fixas por canal, para que o mesmo canal tenha a mesma cor em todas as páginas.
CHANNEL_COLORS = {
    # MMM
    "tv_spend": "#7E8FE8",
    "newspaper_spend": GREY,
    "instagram_spend": "#E06BB0",
    "google_ads_spend": TEAL,
    "youtube_spend": "#E5636A",
    "influencer_spend": GOLD,
    "ott_spend": "#9B7EE0",
    "competitor_spend": "#61708F",
    "Base": "#3B456F",
    # MTA
    "Social Media": "#E06BB0",
    "Email": TEAL,
    "PPC": GOLD,
    "SEO": BLUE,
    "Referral": "#9B7EE0",
}

# Camadas do framework -> (rótulo, cor)
LAYERS = {
    "MMM": ("Camada Estratégica — MMM (top-down)", "#7E8FE8"),
    "MTA": ("Camada Tática — MTA / Atribuição (bottom-up)", TEAL),
    "AB": ("Camada de Validação Causal — Testes A/B", GOLD),
    "GOV": ("Camada de Governança — Learning Repository", BLUE),
}


def channel_color(name: str, fallback_index: int = 0) -> str:
    """Cor determinística por canal (com fallback pela paleta)."""
    if name in CHANNEL_COLORS:
        return CHANNEL_COLORS[name]
    return PALETTE[fallback_index % len(PALETTE)]


def tint(hex_color: str, alpha: float = 0.16) -> str:
    """Versão translúcida de uma cor hex, para fundos de destaque."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


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
    --surface: {SURFACE};
    --border: {BORDER};
    --text: {TEXT};
    --text-muted: {TEXT_MUTED};
}}

/* ---- Tipografia --------------------------------------------------------- */
html, body, [class*="st-"] {{
    font-family: "Segoe UI", "Inter", Helvetica, Arial, sans-serif;
}}
.stApp {{ background-color: {BG}; }}
h1 {{
    color: {TEXT};
    font-weight: 700;
    letter-spacing: -0.02em;
    font-size: 2.05rem;
    line-height: 1.25;
    margin: 0.15rem 0 0.35rem 0;
}}
h2 {{
    color: {TEXT};
    font-weight: 650;
    font-size: 1.4rem;
    line-height: 1.35;
    margin: 1.6rem 0 0.6rem 0;
}}
h3 {{
    color: {TEXT};
    font-weight: 620;
    font-size: 1.13rem;
    line-height: 1.4;
    margin: 1.2rem 0 0.5rem 0;
}}
p, li, .stMarkdown {{ color: {TEXT}; line-height: 1.62; }}
a {{ color: {TEAL}; }}
code {{
    background: {SURFACE};
    color: {GOLD};
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.86em;
}}
[data-testid="stCaptionContainer"], .stCaption, small {{
    color: {TEXT_MUTED} !important;
    line-height: 1.55;
}}

/* ---- Layout: respiro para nada encostar em nada ------------------------- */
.block-container {{ padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1500px; }}
hr {{ border-color: {BORDER}; }}
[data-testid="stVerticalBlock"] {{ gap: 0.85rem; }}
[data-testid="stHorizontalBlock"] {{ gap: 1.1rem; align-items: stretch; }}

/* ---- Sidebar ------------------------------------------------------------ */
section[data-testid="stSidebar"] {{
    background-color: {SURFACE};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT}; }}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ font-size: 1.02rem; margin-top: 1.1rem; }}

/* ---- Métricas: alturas iguais, sem corte de texto ----------------------- */
div[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid {GOLD};
    border-radius: 10px;
    padding: 14px 16px;
    height: 100%;
    overflow-wrap: anywhere;
}}
div[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-size: 0.78rem;
    line-height: 1.35;
    white-space: normal;      /* rótulo longo quebra em vez de sobrepor */
}}
div[data-testid="stMetricLabel"] p {{ color: {TEXT_MUTED} !important; font-size: 0.78rem; }}
div[data-testid="stMetricValue"] {{
    color: {TEXT};
    font-size: 1.42rem;
    line-height: 1.3;
}}
div[data-testid="stMetricDelta"] {{ font-size: 0.78rem; line-height: 1.3; }}

/* ---- Badge de camada ---------------------------------------------------- */
.mmm-badge {{
    display: inline-block;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {BG};
    line-height: 1.5;
    white-space: normal;
}}
.mmm-context {{
    color: {TEXT_MUTED};
    font-size: 0.98rem;
    line-height: 1.6;
    margin: 10px 0 2px 0;
    max-width: 78ch;          /* linha curta o bastante para leitura confortável */
}}
.mmm-hr {{ border: none; border-top: 1px solid {BORDER}; margin: 14px 0 22px 0; }}

/* ---- Cards -------------------------------------------------------------- */
.mmm-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-top: 3px solid {TEAL};
    border-radius: 10px;
    padding: 16px 18px;
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow-wrap: anywhere;   /* nada transborda para o card vizinho */
}}
.mmm-card h4 {{
    margin: 0 0 8px 0;
    color: {TEXT};
    font-size: 0.97rem;
    font-weight: 640;
    line-height: 1.4;
}}
.mmm-card p {{
    margin: 0;
    color: {TEXT_MUTED};
    font-size: 0.85rem;
    line-height: 1.58;
}}
.mmm-card b {{ color: {TEXT}; }}

/* ---- Destaque grande (lift, score) -------------------------------------- */
.mmm-highlight {{
    border-radius: 12px;
    padding: 20px 24px;
    line-height: 1.45;
}}
.mmm-highlight .value {{
    font-size: 2.5rem;
    font-weight: 800;
    display: block;
    line-height: 1.15;
    margin-bottom: 4px;
}}
.mmm-highlight .label {{ color: {TEXT_MUTED}; font-size: 0.95rem; }}

/* ---- Tabs: sem rótulos colados ----------------------------------------- */
button[data-baseweb="tab"] {{
    padding: 10px 4px;
    font-size: 0.92rem;
}}
div[data-baseweb="tab-list"] {{
    gap: 22px;
    border-bottom: 1px solid {BORDER};
    flex-wrap: wrap;           /* muitas abas quebram em linha em vez de espremer */
}}
div[data-baseweb="tab-highlight"] {{ background-color: {GOLD}; }}

/* ---- Inputs, tabelas e avisos ------------------------------------------ */
div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
div[data-testid="stExpander"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background: {SURFACE};
}}
div[data-testid="stExpander"] summary p {{ font-weight: 600; }}
div[data-testid="stAlert"] {{ border-radius: 8px; line-height: 1.6; }}
.stSlider label, .stSelectbox label, .stNumberInput label,
.stRadio label, .stMultiSelect label, .stTextInput label, .stTextArea label {{
    color: {TEXT} !important;
    font-size: 0.87rem;
    line-height: 1.45;
}}
.stButton button, .stDownloadButton button {{ border-radius: 8px; font-weight: 600; }}
</style>
"""


def inject_css() -> None:
    """Injeta o CSS global do app (idempotente por rerun)."""
    st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Componentes
# --------------------------------------------------------------------------
def page_header(title: str, context: str, layer: str = "MMM") -> None:
    """Cabeçalho padrão: badge de camada, título e 1 frase de contexto."""
    label, color = LAYERS.get(layer, LAYERS["MMM"])
    st.markdown(
        f'<span class="mmm-badge" style="background:{color}">{label}</span>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(f'<p class="mmm-context">{context}</p>', unsafe_allow_html=True)
    st.markdown('<hr class="mmm-hr">', unsafe_allow_html=True)


def nav_card(title: str, description: str, accent: str = TEAL) -> None:
    """Card usado na Home para explicar o 'quando usar' de cada página."""
    st.markdown(
        f'<div class="mmm-card" style="border-top-color:{accent}">'
        f"<h4>{title}</h4><p>{description}</p></div>",
        unsafe_allow_html=True,
    )


def highlight(value: str, label: str, color: str = GOLD) -> None:
    """Bloco de destaque grande (lift esperado, score de propensão)."""
    st.markdown(
        f'<div class="mmm-highlight" style="background:{tint(color)};border-left:5px solid {color}">'
        f'<span class="value" style="color:{color}">{value}</span>'
        f'<span class="label">{label}</span></div>',
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

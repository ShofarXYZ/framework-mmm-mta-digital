"""Paleta, CSS custom e componentes visuais compartilhados pelo app.

O app suporta **tema claro e escuro**. A troca é feita pelo controle nativo do
Streamlit (☰ → Settings → Appearance), que já reconfigura os widgets, tabelas e
menus; este módulo lê o tema ativo via `st.context.theme.type` e devolve os
tokens de superfície/texto correspondentes, para que o CSS custom e os gráficos
Plotly acompanhem a mesma escolha.

Divisão importante:
  * **Cores de marca** (navy, teal, azul, dourado) são fixas nos dois temas —
    foram escolhidas em tons médios, legíveis tanto sobre branco quanto sobre o
    fundo escuro. É o que mantém a identidade visual estável.
  * **Tokens de superfície e texto** mudam com o tema — via `tokens()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

# --------------------------------------------------------------------------
# Cores de marca — idênticas nos dois temas
# --------------------------------------------------------------------------
NAVY = "#4A5AC0"
TEAL = "#2A9CB5"
BLUE = "#3D7FC1"
GOLD = "#D9922F"
GREY = "#7C89A3"

POSITIVE = "#2FA36B"
NEGATIVE = "#D9534F"
WARNING = GOLD

PALETTE = [TEAL, GOLD, NAVY, BLUE, "#7E8FE8", "#3FBFA8", "#C77BB8", "#9B7EE0", GREY]

CHANNEL_COLORS = {
    # MMM
    "tv_spend": "#7E8FE8",
    "newspaper_spend": GREY,
    "instagram_spend": "#C7639F",
    "google_ads_spend": TEAL,
    "youtube_spend": NEGATIVE,
    "influencer_spend": GOLD,
    "ott_spend": "#9B7EE0",
    "competitor_spend": "#5F6E8C",
    "Base": "#8A95AD",
    # MTA
    "Social Media": "#C7639F",
    "Email": TEAL,
    "PPC": GOLD,
    "SEO": BLUE,
    "Referral": "#9B7EE0",
}

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


def tint(hex_color: str, alpha: float = 0.16) -> str:
    """Versão translúcida de uma cor hex, para fundos de destaque."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------
# Tokens dependentes do tema
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Tokens:
    """Superfícies e texto do tema ativo."""

    name: str
    bg: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    grid: str
    on_accent: str  # cor de texto sobre um fundo de accent sólido (badge)
    plotly_template: str
    sequential: tuple[str, ...]
    diverging: tuple[str, ...]


DARK = Tokens(
    name="dark",
    bg="#0F1428",
    surface="#1A2145",
    surface_alt="#232B52",
    border="#2E3766",
    text="#E6EAF3",
    text_muted="#A9B4CC",
    grid="#242C55",
    on_accent="#0F1428",
    plotly_template="plotly_dark",
    sequential=("#1A2145", "#27527A", "#2F7C97", "#38B2C4", "#8FD9E3"),
    diverging=("#D9534F", "#6B4A5E", "#2E3766", "#2F7C97", "#38B2C4"),
)

LIGHT = Tokens(
    name="light",
    bg="#FFFFFF",
    surface="#F1F4FA",
    surface_alt="#E6ECF7",
    border="#D8DFEC",
    text="#1B2440",
    text_muted="#5A6786",
    grid="#E7ECF5",
    on_accent="#FFFFFF",
    plotly_template="plotly_white",
    sequential=("#EAF1F6", "#B9D9E4", "#7BBFD1", "#3D9BB5", "#1C6B85"),
    diverging=("#C0392B", "#E3A9A3", "#EDF1F7", "#7BBFD1", "#1C6B85"),
)


def theme_type() -> str:
    """'light' ou 'dark' conforme a escolha do usuário no menu do Streamlit."""
    try:
        active = st.context.theme.type  # type: ignore[union-attr]
        if active in ("light", "dark"):
            return active
    except Exception:
        pass
    return "dark"


def tokens() -> Tokens:
    """Tokens do tema ativo. Chame em tempo de render, nunca no import."""
    return LIGHT if theme_type() == "light" else DARK


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------
def _css(t: Tokens) -> str:
    return f"""
<style>
:root {{
    --navy: {NAVY};
    --teal: {TEAL};
    --gold: {GOLD};
    --surface: {t.surface};
    --border: {t.border};
    --text: {t.text};
    --text-muted: {t.text_muted};
}}

/* ---- Tipografia --------------------------------------------------------- */
html, body, [class*="st-"] {{
    font-family: "Segoe UI", "Inter", Helvetica, Arial, sans-serif;
}}
h1 {{
    color: {t.text};
    font-weight: 700;
    letter-spacing: -0.02em;
    font-size: 2.05rem;
    line-height: 1.25;
    margin: 0.15rem 0 0.35rem 0;
}}
h2 {{
    color: {t.text};
    font-weight: 650;
    font-size: 1.4rem;
    line-height: 1.35;
    margin: 1.6rem 0 0.6rem 0;
}}
h3 {{
    color: {t.text};
    font-weight: 620;
    font-size: 1.13rem;
    line-height: 1.4;
    margin: 1.2rem 0 0.5rem 0;
}}
p, li, .stMarkdown {{ line-height: 1.62; }}
code {{ color: {GOLD}; padding: 1px 6px; border-radius: 4px; font-size: 0.86em; }}
[data-testid="stCaptionContainer"], .stCaption, small {{
    color: {t.text_muted} !important;
    line-height: 1.55;
}}

/* ---- Layout: respiro para nada encostar em nada ------------------------- */
.block-container {{ padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1500px; }}
[data-testid="stVerticalBlock"] {{ gap: 0.85rem; }}
[data-testid="stHorizontalBlock"] {{ gap: 1.1rem; align-items: stretch; }}

/* ---- Sidebar ------------------------------------------------------------ */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ font-size: 1.02rem; margin-top: 1.1rem; }}

/* ---- Métricas: alturas iguais, sem corte de texto ----------------------- */
div[data-testid="stMetric"] {{
    background: {t.surface};
    border: 1px solid {t.border};
    border-left: 3px solid {GOLD};
    border-radius: 10px;
    padding: 14px 16px;
    height: 100%;
    overflow-wrap: anywhere;
}}
div[data-testid="stMetricLabel"] {{
    color: {t.text_muted} !important;
    font-size: 0.78rem;
    line-height: 1.35;
    white-space: normal;      /* rótulo longo quebra em vez de sobrepor */
}}
div[data-testid="stMetricLabel"] p {{ color: {t.text_muted} !important; font-size: 0.78rem; }}
div[data-testid="stMetricValue"] {{ color: {t.text}; font-size: 1.42rem; line-height: 1.3; }}
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
    color: #FFFFFF;
    line-height: 1.5;
    white-space: normal;
}}
.mmm-context {{
    color: {t.text_muted};
    font-size: 0.98rem;
    line-height: 1.6;
    margin: 10px 0 2px 0;
    max-width: 78ch;          /* linha curta o bastante para leitura confortável */
}}
.mmm-hr {{ border: none; border-top: 1px solid {t.border}; margin: 14px 0 22px 0; }}

/* ---- Cards -------------------------------------------------------------- */
.mmm-card {{
    background: {t.surface};
    border: 1px solid {t.border};
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
    color: {t.text};
    font-size: 0.97rem;
    font-weight: 640;
    line-height: 1.4;
}}
.mmm-card p {{ margin: 0; color: {t.text_muted}; font-size: 0.85rem; line-height: 1.58; }}
.mmm-card b {{ color: {t.text}; }}

/* ---- Destaque grande (lift, veredito, score) ---------------------------- */
.mmm-highlight {{ border-radius: 12px; padding: 20px 24px; line-height: 1.45; }}
.mmm-highlight .value {{
    font-size: 2.5rem;
    font-weight: 800;
    display: block;
    line-height: 1.15;
    margin-bottom: 4px;
}}
.mmm-highlight .label {{ color: {t.text_muted}; font-size: 0.95rem; }}

/* ---- Tabs: sem rótulos colados ----------------------------------------- */
button[data-baseweb="tab"] {{ padding: 10px 4px; font-size: 0.92rem; }}
div[data-baseweb="tab-list"] {{
    gap: 22px;
    border-bottom: 1px solid {t.border};
    flex-wrap: wrap;           /* muitas abas quebram em linha em vez de espremer */
}}
div[data-baseweb="tab-highlight"] {{ background-color: {GOLD}; }}

/* ---- Inputs, tabelas e avisos ------------------------------------------ */
div[data-testid="stExpander"] {{ border-radius: 8px; }}
div[data-testid="stExpander"] summary p {{ font-weight: 600; }}
div[data-testid="stAlert"] {{ border-radius: 8px; line-height: 1.6; }}
.stSlider label, .stSelectbox label, .stNumberInput label,
.stRadio label, .stMultiSelect label, .stTextInput label, .stTextArea label {{
    font-size: 0.87rem;
    line-height: 1.45;
}}
.stButton button, .stDownloadButton button {{ font-weight: 600; }}
</style>
"""


def inject_css() -> None:
    """Injeta o CSS do tema ativo (reexecutado a cada rerun, então acompanha a troca)."""
    st.markdown(_css(tokens()), unsafe_allow_html=True)


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
    """Bloco de destaque grande (lift esperado, veredito A/B, score de propensão)."""
    st.markdown(
        f'<div class="mmm-highlight" style="background:{tint(color)};border-left:5px solid {color}">'
        f'<span class="value" style="color:{color}">{value}</span>'
        f'<span class="label">{label}</span></div>',
        unsafe_allow_html=True,
    )


def recommendation_panel(headline: str, detail: str, invest: str, watch: str,
                         invest_note: str = "", watch_note: str = "") -> None:
    """Painel 'onde investir / onde ter atenção', usado nas páginas de MMM e MTA."""
    t = tokens()
    st.markdown(
        f"""
<div class="mmm-highlight" style="background:{tint(GOLD, 0.10)};border-left:5px solid {GOLD};
     margin-bottom:14px">
  <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;
       color:{GOLD};margin-bottom:8px">Recomendação do modelo</div>
  <div style="font-size:1.12rem;color:{t.text};line-height:1.55">{headline}</div>
  <div style="color:{t.text_muted};font-size:0.92rem;margin-top:8px;max-width:88ch">{detail}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"<div class='mmm-card' style='border-top-color:{POSITIVE}'>"
            f"<h4 style='color:{POSITIVE}'>➕ Investir mais em {invest}</h4>"
            f"<p>{invest_note}</p></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"<div class='mmm-card' style='border-top-color:{NEGATIVE}'>"
            f"<h4 style='color:{NEGATIVE}'>⚠️ Atenção com {watch}</h4>"
            f"<p>{watch_note}</p></div>",
            unsafe_allow_html=True,
        )


def theme_hint() -> None:
    """Indica o tema ativo e onde trocá-lo (o controle é nativo do Streamlit)."""
    active = theme_type()
    icon = "🌙" if active == "dark" else "☀️"
    name = "Escuro" if active == "dark" else "Claro"
    st.caption(
        f"{icon} Tema **{name}** · troque em ☰ → *Settings* → *Appearance* "
        "(Light / Dark / System). O app se adapta aos dois."
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

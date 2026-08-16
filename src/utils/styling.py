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
/* ATENÇÃO ao seletor: os ícones do Streamlit são uma FONTE DE LIGADURAS
   (Material Symbols) e vivem em spans que também carregam classes começando com
   "st-". Um seletor amplo como [class*="st-"] troca a fonte deles, a ligadura
   deixa de resolver e o nome do ícone vaza como texto na tela
   ("keyboard_double_arrow_left", "_arrow_right"). Por isso a fonte do app é
   aplicada a contêineres de conteúdo, e a exceção dos ícones vem logo abaixo. */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
.stMarkdown, .stButton, .stDownloadButton, .stSelectbox, .stMultiSelect,
.stTextInput, .stTextArea, .stNumberInput, .stSlider, .stRadio, .stCheckbox,
.stMetric, .stDataFrame, .stTabs, .stExpander, .stAlert, .stCaption {{
    font-family: "Segoe UI", "Inter", Helvetica, Arial, sans-serif;
}}

/* Exceção obrigatória: preserva a fonte de ícones do Streamlit. */
[data-testid="stIconMaterial"],
span.material-symbols-rounded,
span.material-symbols-outlined,
span.material-symbols-sharp,
[class*="material-symbols"],
[class*="MaterialIcon"],
.material-icons, .material-icons-outlined, .material-icons-round {{
    font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                 "Material Symbols Sharp", "Material Icons" !important;
    font-weight: normal !important;
    font-style: normal !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap;
    word-wrap: normal;
    direction: ltr;
    -webkit-font-feature-settings: "liga";
    font-feature-settings: "liga";
    -webkit-font-smoothing: antialiased;
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
[data-testid="stVerticalBlock"] {{ gap: 1rem; }}
/* Linhas de colunas (cards, KPIs) precisam de margem PRÓPRIA: como os cards têm
   height:100% para ficarem alinhados entre si, eles encostam na linha seguinte
   se o espaçamento depender só do gap do bloco pai. */
[data-testid="stHorizontalBlock"] {{
    gap: 1.1rem;
    align-items: stretch;
    margin-bottom: 0.9rem;
}}

/* ---- Sidebar ------------------------------------------------------------ */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ font-size: 1.02rem; margin-top: 1.1rem; }}

/* ---- Métricas: linha de KPIs sempre alinhada ---------------------------- */
/* O desalinhamento clássico vem de três fontes: rótulos que quebram em 1 ou 2
   linhas, cards que têm delta e cards que não têm, e colunas que não esticam.
   A solução: coluna vira flex, o card ocupa a altura toda, o rótulo reserva
   duas linhas (então todos os VALORES nascem na mesma altura) e o delta é
   empurrado para a base. */
/* Para um card esticar até a altura do vizinho mais alto, TODOS os wrappers que
   o Streamlit cria entre a coluna e o conteúdo precisam esticar junto. Parar no
   primeiro filho não basta: sobra um wrapper intermediário com altura de
   conteúdo, e os cards da mesma linha terminam em alturas diferentes. */
div[data-testid="stColumn"], div[data-testid="column"] {{
    display: flex;
    flex-direction: column;
}}
div[data-testid="stColumn"] > div,
div[data-testid="column"] > div,
div[data-testid="stColumn"] > div > div,
div[data-testid="stColumn"] [data-testid="stVerticalBlock"],
div[data-testid="stColumn"] [data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stColumn"] [data-testid="stElementContainer"],
div[data-testid="stColumn"] [data-testid="stMarkdown"],
div[data-testid="stColumn"] [data-testid="stMarkdownContainer"] {{
    height: 100%;
}}
/* ...mas só o conteúdo em card deve esticar: gráficos e tabelas mantêm a
   própria altura, senão ficam esticados e distorcidos. */
div[data-testid="stColumn"] [data-testid="stPlotlyChart"],
div[data-testid="stColumn"] [data-testid="stDataFrame"],
div[data-testid="stColumn"] [data-testid="stExpander"] {{ height: auto; }}
div[data-testid="stMetric"] {{
    margin-bottom: 2px;
    background: {t.surface};
    border: 1px solid {t.border};
    border-left: 3px solid {GOLD};
    border-radius: 10px;
    padding: 14px 16px 12px 16px;
    height: 100%;
    min-height: 112px;
    display: flex;
    flex-direction: column;
    overflow-wrap: anywhere;
}}
div[data-testid="stMetricLabel"] {{
    color: {t.text_muted} !important;
    font-size: 0.78rem;
    line-height: 1.32;
    white-space: normal;      /* rótulo longo quebra em vez de sobrepor */
    min-height: 2.05em;       /* reserva 2 linhas: alinha os valores entre cards */
    display: flex;
    align-items: flex-start;
}}
div[data-testid="stMetricLabel"] p {{ color: {t.text_muted} !important; font-size: 0.78rem; }}
div[data-testid="stMetricValue"] {{
    color: {t.text};
    font-size: 1.38rem;
    line-height: 1.28;
    margin-top: 6px;
}}
div[data-testid="stMetricDelta"] {{
    font-size: 0.76rem;
    line-height: 1.25;
    margin-top: auto;         /* encosta na base: cards com e sem delta ficam iguais */
    padding-top: 8px;
}}
div[data-testid="stMetricDelta"] svg {{ transform: scale(0.85); }}

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
    margin-bottom: 2px;
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
    st.markdown(f'<p class="mmm-context">{md_inline(context)}</p>', unsafe_allow_html=True)
    st.markdown('<hr class="mmm-hr">', unsafe_allow_html=True)


def nav_card(title: str, description: str, accent: str = TEAL) -> None:
    """Card usado na Home para explicar o 'quando usar' de cada página."""
    title, description = md_inline(title), md_inline(description)
    st.markdown(
        f'<div class="mmm-card" style="border-top-color:{accent}">'
        f"<h4>{title}</h4><p>{description}</p></div>",
        unsafe_allow_html=True,
    )


def highlight(value: str, label: str, color: str = GOLD) -> None:
    """Bloco de destaque grande (lift esperado, veredito A/B, score de propensão)."""
    value, label = md_inline(value), md_inline(label)
    st.markdown(
        f'<div class="mmm-highlight" style="background:{tint(color)};border-left:5px solid {color}">'
        f'<span class="value" style="color:{color}">{value}</span>'
        f'<span class="label">{label}</span></div>',
        unsafe_allow_html=True,
    )


def md_inline(text: str) -> str:
    """Converte markdown inline (**negrito**, *itálico*, `código`) para HTML.

    Os componentes deste módulo montam HTML cru, onde o markdown do Streamlit não
    é interpretado — sem esta conversão, `**investir em TV**` aparece na tela com
    os asteriscos à mostra.
    """
    import re

    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)(?<!\s)\*(?![\*\w])", r"<i>\1</i>", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def recommendation_panel(headline: str, detail: str, invest: str, watch: str,
                         invest_note: str = "", watch_note: str = "") -> None:
    """Painel 'onde investir / onde ter atenção', usado nas páginas de MMM e MTA."""
    t = tokens()
    headline, detail = md_inline(headline), md_inline(detail)
    invest_note, watch_note = md_inline(invest_note), md_inline(watch_note)
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
    spacer(18)


def stage_header(number: int, name: str, question: str, intro: str, color: str = TEAL) -> None:
    """Cabeçalho das etapas das páginas guiadas (Descritivo → Prescritivo)."""
    t = tokens()
    question, intro = md_inline(question), md_inline(intro)
    st.markdown(
        f"""
<div style="display:flex;gap:16px;align-items:flex-start;margin:4px 0 18px 0">
  <div style="flex:0 0 46px;height:46px;border-radius:50%;background:{color};color:#FFFFFF;
       display:flex;align-items:center;justify-content:center;font-size:1.25rem;font-weight:800">
    {number}
  </div>
  <div style="flex:1">
    <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;
         color:{color}">{name}</div>
    <div style="font-size:1.32rem;font-weight:680;color:{t.text};line-height:1.35;margin-top:2px">
      {question}</div>
    <div style="color:{t.text_muted};font-size:0.95rem;line-height:1.6;margin-top:8px;max-width:88ch">
      {intro}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def plain_box(title: str, text: str, icon: str = "💡", color: str = GOLD) -> None:
    """Caixa de tradução: o mesmo achado dito em linguagem do dia a dia."""
    t = tokens()
    title, text = md_inline(title), md_inline(text)
    st.markdown(
        f"""
<div style="background:{tint(color, 0.10)};border-left:4px solid {color};border-radius:8px;
     padding:14px 18px;margin:6px 0 14px 0">
  <div style="font-weight:680;color:{t.text};font-size:0.95rem">{icon} {title}</div>
  <div style="color:{t.text_muted};font-size:0.9rem;line-height:1.6;margin-top:5px">{text}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def spacer(height: int = 16) -> None:
    """Espaço vertical explícito entre blocos que precisam respirar."""
    st.markdown(f"<div style='height:{height}px'></div>", unsafe_allow_html=True)


def theme_hint() -> None:
    """Indica o tema ativo e onde trocá-lo (o controle é nativo do Streamlit)."""
    active = theme_type()
    icon = "🌙" if active == "dark" else "☀️"
    name = "Escuro" if active == "dark" else "Claro"
    st.caption(
        f"{icon} Tema **{name}** · troque em ☰ → *Settings* → *Appearance* "
        "(Light / Dark / System). O app se adapta aos dois."
    )


def brl(value: float, decimals: int = 2, prefix: str = "R$ ") -> str:
    """Moeda no padrão brasileiro: R$ 13.450,00.

    O Streamlit não tem preset de Real e o `format` printf das tabelas não faz
    separador de milhar, então a formatação é feita aqui e o valor entra na
    tabela já como texto.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if value != value:  # NaN
        return "—"
    # Formata no padrão en-US e troca os separadores: 1,234.56 -> 1.234,56
    formatted = f"{abs(value):,.{decimals}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    sign = "-" if value < 0 else ""
    return f"{sign}{prefix}{formatted}"


def brl_compact(value: float, prefix: str = "R$ ") -> str:
    """Versão curta para espaços apertados: R$ 13,5 mi."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if value != value:
        return "—"
    for unit, div in ((" bi", 1e9), (" mi", 1e6), (" mil", 1e3)):
        if abs(value) >= div:
            number = f"{abs(value) / div:,.1f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
            sign = "-" if value < 0 else ""
            return f"{sign}{prefix}{number}{unit}"
    return brl(value, 2, prefix)


def brl_series(values) -> list[str]:
    """Aplica `brl` a uma coluna inteira, para exibição em tabela."""
    return [brl(v) for v in values]


def money_columns(df, columns: list[str]):
    """Devolve uma cópia do dataframe com as colunas monetárias já em texto R$.

    Use junto de `st.column_config.TextColumn` — a formatação brasileira exige
    texto, então a ordenação por clique naquela coluna deixa de ser numérica;
    por isso as tabelas do app já vêm ordenadas pelo critério relevante.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = brl_series(out[col])
    return out


def fmt_money(value: float, prefix: str = "R$ ") -> str:
    """Compatibilidade: agora entrega o padrão brasileiro completo."""
    return brl(value, 2, prefix)

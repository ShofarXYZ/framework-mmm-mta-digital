"""MMM x MTA — Framework de Digital Analytics.

Entrypoint do app Streamlit multi-página (API st.Page + st.navigation).
Rode com:  streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="MMM x MTA — Digital Analytics",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.data_loader import segment_options  # noqa: E402
from src.utils import repository  # noqa: E402
from src.utils.styling import NAVY, inject_css, theme_hint  # noqa: E402

inject_css()

# ---------------------------------------------------------------------------
# Navegação
# ---------------------------------------------------------------------------
PAGES = {
    "Visão geral": [
        st.Page("pages/0_home.py", title="Home", icon="🏠", default=True),
    ],
    "Estratégico — MMM": [
        st.Page("pages/1_mmm_explorer.py", title="MMM Explorer", icon="📊"),
        st.Page("pages/2_mmm_modelagem.py", title="MMM Modelagem", icon="🧪"),
        st.Page("pages/3_mmm_otimizador.py", title="Otimizador de Budget", icon="💰"),
    ],
    "Tático — MTA": [
        st.Page("pages/4_mta_atribuicao.py", title="Modelos de Atribuição", icon="🔀"),
        st.Page("pages/5_mta_markov_shapley.py", title="Markov e Shapley", icon="🕸️"),
        st.Page("pages/10_mta_propensao.py", title="Propensão à Conversão", icon="🎯"),
    ],
    "Validação — Testes A/B": [
        st.Page("pages/6_ab_calculadora.py", title="Calculadora A/B", icon="🧬"),
        st.Page("pages/7_ab_simulador.py", title="Simulador e Resultados", icon="📈"),
        st.Page("pages/8_ab_geo_holdout.py", title="Geo-Holdout x MMM", icon="🌍"),
    ],
    "Governança": [
        st.Page("pages/9_governanca.py", title="Learning Repository", icon="🔁"),
    ],
}

navigation = st.navigation(PAGES)

# ---------------------------------------------------------------------------
# Sidebar global
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<h3 style='color:{NAVY};margin-bottom:0'>📐 MMM × MTA</h3>"
        "<p style='font-size:0.8rem;color:#64748B;margin-top:2px'>Framework de Digital Analytics</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    try:
        options = segment_options()
    except Exception:
        options = ["Todos"]

    st.session_state.setdefault("segmento", "Todos")
    st.selectbox(
        "Projeto / Segmento",
        options,
        key="segmento",
        help="Filtro global de segmento de cliente (aplicado nas páginas que usam o dataset de campanhas).",
    )

    st.metric("Learning Repository", f"{repository.count()} registros")
    st.divider()

    theme_hint()
    st.caption(
        "Camadas: **Estratégico** (MMM, top-down) · **Tático** (MTA, bottom-up) · "
        "**Validação** (Testes A/B, causal) · **Governança** (Learning Repository)."
    )

navigation.run()

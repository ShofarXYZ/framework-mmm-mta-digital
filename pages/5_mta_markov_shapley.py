"""Página 5 — MTA: Markov Chain (Removal Effect) e Shapley Value."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_digital
from src.mta.heuristics import attribute_all, to_share
from src.mta.journey_sim import build_journeys
from src.mta.markov import build_transition_counts, markov_attribution, sankey_data, to_probabilities
from src.mta.shapley import shapley_attribution
from src.utils import repository
from src.utils.styling import GOLD, NAVY, TEAL, page_header
from src.viz.charts import apply_theme, grouped_bar, sankey

page_header(
    "MTA — Markov Chain e Shapley Value",
    "Crédito algorítmico sobre a mesma jornada sintética: em vez de uma regra fixa, o dado decide "
    "quanto cada canal vale. É o 'DDA caseiro' do app.",
    layer="MTA",
)

journeys = st.session_state.get("mta_journeys")
if journeys is None:
    try:
        journeys = build_journeys(load_digital())
        st.session_state["mta_journeys"] = journeys
    except Exception as exc:
        st.error(f"Não foi possível preparar as jornadas: {exc}")
        st.stop()

channels = sorted({c for path in journeys["path"] for c in path})

st.caption(
    f"Mesma base da Página 4: {len(journeys):,} jornadas simuladas, {len(channels)} canais, "
    f"{int(journeys['converted'].sum()):,} conversões.".replace(",", ".")
)

tab_markov, tab_shapley, tab_all = st.tabs(
    ["🕸️ Markov + Removal Effect", "🤝 Shapley Value", "🏁 Todos os modelos"]
)

# ---------------------------------------------------------------------------
# Markov
# ---------------------------------------------------------------------------
with tab_markov:
    st.markdown(
        "**Como funciona:** a jornada vira uma cadeia de Markov com os estados `(start)`, um por canal, "
        "`(conversion)` e `(null)`. A probabilidade de conversão é a chance de absorção em `(conversion)` "
        "partindo de `(start)`. O **Removal Effect** de um canal é o quanto essa probabilidade cai quando "
        "removemos o canal da cadeia — uma aproximação causal do valor daquele canal."
    )

    with st.spinner("Calculando cadeia de Markov e removal effects..."):
        try:
            markov = markov_attribution(journeys, channels)
        except Exception as exc:
            st.error(f"Falha no cálculo de Markov: {exc}")
            st.stop()

    p_base = markov.attrs.get("p_base", np.nan)
    c1, c2 = st.columns([1, 3])
    c1.metric("P(conversão) da cadeia", f"{p_base * 100:.2f}%")
    c2.caption(
        "Compare com a taxa de conversão bruta da base: a cadeia reproduz a probabilidade de absorção "
        "considerando todos os caminhos possíveis entre canais."
    )

    display = markov.reset_index()
    fig = px.bar(display.sort_values("removal_effect"), x="removal_effect", y="canal",
                 orientation="h", title="Removal Effect por canal",
                 color="removal_effect", color_continuous_scale="Teal")
    st.plotly_chart(apply_theme(fig, height=380, legend_bottom=False), width="stretch")

    st.dataframe(
        display.round(4), width="stretch", hide_index=True,
        column_config={
            "p_sem_canal": st.column_config.NumberColumn("P(conv) sem o canal", format="%.4f"),
            "removal_effect": st.column_config.NumberColumn("Removal Effect", format="%.4f"),
            "removal_effect_norm": st.column_config.ProgressColumn(
                "Crédito normalizado", min_value=0.0, max_value=1.0, format="%.3f"),
            "conversoes_creditadas": st.column_config.NumberColumn("Conversões creditadas", format="%.0f"),
        },
    )

    st.subheader("Fluxo de transição entre canais")
    min_flow = st.slider("Fluxo mínimo para exibir a ligação", 10, 500, 50, 10)
    try:
        labels, source, target, value = sankey_data(journeys, min_flow)
        if source:
            st.plotly_chart(sankey(labels, source, target, value,
                                   "Do (start) à conversão: por onde a jornada passa"),
                            width="stretch")
        else:
            st.info("Nenhum fluxo acima do limite escolhido — reduza o filtro.")
    except Exception as exc:
        st.warning(f"Não foi possível montar o Sankey: {exc}")

    with st.expander("Ver a matriz de transição (probabilidades)"):
        probs = to_probabilities(build_transition_counts(journeys))
        st.dataframe(probs.round(3), width="stretch")

# ---------------------------------------------------------------------------
# Shapley
# ---------------------------------------------------------------------------
with tab_shapley:
    st.markdown(
        "**Como funciona:** cada canal é um jogador; a coalizão `S` vale `v(S)` = conversões geradas por "
        "jornadas cujo conjunto de canais cabe dentro de `S`. O Shapley Value de um canal é a média da sua "
        "contribuição marginal em todas as ordens de entrada. Com 5 canais são apenas 32 coalizões, "
        "então o cálculo **exato** é viável."
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        method = st.radio("Método", ["Exato", "Monte Carlo"], horizontal=True)
        n_perm = st.slider("Permutações (Monte Carlo)", 500, 20000, 3000, 500,
                           disabled=(method == "Exato"))

    with st.spinner("Calculando Shapley Values..."):
        try:
            shap_table = shapley_attribution(journeys, channels, method, n_perm)
        except Exception as exc:
            st.error(f"Falha no cálculo de Shapley: {exc}")
            st.stop()

    display = shap_table.reset_index()
    fig = px.bar(display.sort_values("share_%"), x="share_%", y="canal", orientation="h",
                 title=f"Crédito por Shapley Value ({method})", color_discrete_sequence=[GOLD])
    st.plotly_chart(apply_theme(fig, height=380, legend_bottom=False), width="stretch")
    st.dataframe(display.round(3), width="stretch", hide_index=True)

    st.caption(
        "Diferença conceitual: **Markov é sensível à ordem** dos touchpoints; **Shapley ignora a ordem** e "
        "olha só a presença do canal na jornada. Quando os dois discordam bastante, o canal tem papel "
        "posicional forte (ex.: sempre aparece no fim) — informação útil por si só."
    )

# ---------------------------------------------------------------------------
# Comparação geral
# ---------------------------------------------------------------------------
with tab_all:
    st.subheader("Todos os modelos de atribuição lado a lado")

    heuristics = to_share(attribute_all(journeys))
    markov_share = (markov_attribution(journeys, channels)["removal_effect_norm"] * 100).rename("Markov")
    shapley_share = shapley_attribution(journeys, channels)["share_%"].rename("Shapley")

    full = heuristics.join(markov_share, how="outer").join(shapley_share, how="outer").fillna(0.0)
    full.index.name = "canal"
    st.session_state["mta_full_attribution"] = full

    st.dataframe(
        full.round(1), width="stretch",
        column_config={c: st.column_config.NumberColumn(c, format="%.1f%%") for c in full.columns},
    )

    long = full.reset_index().melt(id_vars="canal", var_name="modelo", value_name="crédito %")
    st.plotly_chart(
        grouped_bar(long, x="canal", y="crédito %", color="modelo",
                    title="Heurísticos × algorítmicos — % de crédito por canal"),
        width="stretch",
    )

    spread = (full.max(axis=1) - full.min(axis=1)).sort_values(ascending=False)
    st.info(
        f"**Maior divergência entre modelos:** `{spread.index[0]}`, com {spread.iloc[0]:.1f} p.p. "
        f"de diferença entre o modelo mais e o menos generoso. Canais assim são os melhores candidatos "
        "a um teste de incrementalidade — nenhum modelo correlacional vai resolver a dúvida sozinho.",
        icon="🎯",
    )

    algo_mean = full[["Markov", "Shapley"]].mean(axis=1)
    delta_lastclick = (full["Last-Click"] - algo_mean).sort_values()
    fig = px.bar(
        x=delta_lastclick.values, y=delta_lastclick.index, orientation="h",
        title="Quanto o last-click distorce em relação aos modelos algorítmicos (p.p.)",
        color=delta_lastclick.values, color_continuous_scale=["#D62828", "#EDF2F7", TEAL],
        labels={"x": "Δ p.p. (last-click − média Markov/Shapley)", "y": "canal"},
    )
    st.plotly_chart(apply_theme(fig, height=360, legend_bottom=False), width="stretch")

    repository.save_widget(
        key="mta_algo",
        origem="MTA",
        canal_driver=str(spread.index[0]),
        hipotese_default=f"O crédito de {spread.index[0]} depende fortemente do modelo escolhido — "
                         "precisa de validação causal.",
        resultado_default="Oportunidade",
        insight_default=(
            f"{spread.index[0]} varia {spread.iloc[0]:.1f} p.p. entre os modelos de atribuição. "
            f"Markov e Shapley creditam em média {algo_mean.get(spread.index[0], float('nan')):.1f}%, "
            f"contra {full.loc[spread.index[0], 'Last-Click']:.1f}% no last-click."
        ),
        proximo_passo_default="Desenhar teste A/B ou geo-holdout para medir a incrementalidade real do canal.",
        etapa_default="Hypothesis",
    )

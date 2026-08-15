"""Página 4 — MTA: modelos de atribuição heurísticos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import campaign_channel_summary, load_digital
from src.mta.heuristics import HEURISTIC_MODELS, attribute_all, cpa_table, to_share
from src.mta.journey_sim import adspend_by_channel, build_journeys, journey_stats, top_paths
from src.utils import repository
from src.utils.styling import GOLD, NAVY, PALETTE, TEAL, page_header
from src.viz.charts import apply_theme, grouped_bar

page_header(
    "MTA — Modelos de Atribuição",
    "Camada tática: repartir o crédito da conversão entre os touchpoints digitais da jornada. "
    "É onde fica evidente o quanto o last-click distorce a leitura de performance.",
    layer="MTA",
)

st.info(
    "**Este dataset não é clickstream.** `digital_marketing_campaign_dataset.csv` traz uma linha por "
    "cliente/campanha, com UM canal e um desfecho binário — sem sequência nem timestamp. "
    "Para exercitar os modelos de atribuição, o app **simula** uma jornada multi-touch por cliente a "
    "partir dos sinais de engajamento reais (WebsiteVisits, EmailOpens/Clicks, SocialShares, "
    "PreviousPurchases, LoyaltyPoints). A lógica é determinística e está aberta no expander abaixo e "
    "em `src/mta/journey_sim.py`. Nada no alvo (`Conversion`) é inventado.",
    icon="🧪",
)

try:
    digital = load_digital()
    journeys = build_journeys(digital)
except Exception as exc:
    st.error(f"Não foi possível preparar as jornadas: {exc}")
    st.stop()

st.session_state["mta_journeys"] = journeys

# ---------------------------------------------------------------------------
# Como a jornada foi simulada
# ---------------------------------------------------------------------------
with st.expander("🔍 Como a jornada foi simulada?"):
    st.markdown(
        """
Para cada `CustomerID` o app monta de 1 a 4 touchpoints, cada um com um **stage rank**
(quanto menor, mais distante da conversão). A jornada é ordenada por esse rank e o
canal real da campanha fica **sempre por último**, como o touchpoint mais próximo da conversão:

| # | Regra | Condição no dado real | Touchpoint adicionado |
|---|---|---|---|
| 1 | Relacionamento prévio | `PreviousPurchases > 0` **ou** `LoyaltyPoints` acima da mediana | `Email` (se abriu e-mail) senão `SEO` — ou `Referral` se o principal já for SEO |
| 2 | Descoberta social | `SocialShares > 0` e canal principal ≠ Social Media | `Social Media` |
| 3 | Nutrição por e-mail | `EmailOpens > 0` ou `EmailClicks > 0` e canal principal ≠ Email | `Email` |
| 4 | Pesquisa ativa | `WebsiteVisits` acima da mediana e canal principal ≠ SEO | `SEO` |
| 5 | Campanha | sempre | `CampaignChannel` da linha (último) |

Touchpoints repetidos em sequência são colapsados e a jornada é truncada nos 4 mais
próximos da conversão. **Nenhuma aleatoriedade** entra na construção — dois runs produzem
exatamente a mesma jornada.

> **Limitação assumida:** isto é uma aproximação pedagógica, não tracking real. Com uma tabela
> `fct_mta_touchpoint` de verdade, basta trocar `build_journeys()` e todo o resto do app segue igual.
        """
    )

stats = journey_stats(journeys)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Clientes", f"{stats['clientes']:,}".replace(",", "."))
c2.metric("Conversões", f"{stats['conversoes']:,}".replace(",", "."))
c3.metric("Taxa de conversão", f"{stats['taxa_conversao'] * 100:.1f}%")
c4.metric("Touchpoints por jornada", f"{stats['touchpoints_medio']:.2f}")
c5.metric("Jornadas multi-touch", f"{stats['jornadas_multitouch_%']:.0f}%")

st.divider()

# ---------------------------------------------------------------------------
# Atribuição
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parâmetros de atribuição")
    half_life = st.slider("Meia-vida do Time-Decay (touchpoints)", 0.5, 4.0, 1.0, 0.5,
                          help="Menor = mais crédito concentrado perto da conversão.")

table = attribute_all(journeys, half_life)
shares = to_share(table)
adspend = adspend_by_channel(journeys)

tab_compare, tab_table, tab_paths, tab_cross = st.tabs(
    ["📊 Comparação de modelos", "📋 Tabela e CPA", "🛤️ Caminhos", "🔎 Validação cruzada"]
)

with tab_compare:
    st.subheader("% de crédito por canal em cada modelo")
    long = shares.reset_index().melt(id_vars="canal", var_name="modelo", value_name="crédito %")
    fig = grouped_bar(long, x="canal", y="crédito %", color="modelo",
                      title="Mesmos dados, cinco leituras diferentes do mesmo resultado")
    st.plotly_chart(fig, width="stretch")

    # Quanto o last-click difere dos demais
    diff = (shares["Last-Click"] - shares[["First-Click", "Linear", "Position-Based (U)"]].mean(axis=1))
    worst_over = diff.idxmax()
    worst_under = diff.idxmin()
    st.warning(
        f"**A distorção do last-click:** `{worst_over}` recebe "
        f"{diff.max():.1f} p.p. **a mais** no last-click do que na média dos outros modelos, "
        f"enquanto `{worst_under}` perde {abs(diff.min()):.1f} p.p. "
        "Canais de topo de funil são sistematicamente subvalorizados — e acabam tendo o budget cortado "
        "justamente por aparecerem mal no relatório.",
        icon="⚠️",
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(shares.round(1), text_auto=True, aspect="auto",
                        color_continuous_scale="Blues", title="Heatmap: canal × modelo (% de crédito)")
        st.plotly_chart(apply_theme(fig, height=380, legend_bottom=False), width="stretch")
    with c2:
        fig = px.line_polar(
            long, r="crédito %", theta="canal", color="modelo", line_close=True,
            color_discrete_sequence=PALETTE, title="Perfil de crédito por modelo",
        )
        st.plotly_chart(apply_theme(fig, height=380), width="stretch")

with tab_table:
    st.subheader("Crédito absoluto, investimento e CPA implícito")
    summary = table.copy()
    summary.insert(0, "AdSpend", adspend.reindex(summary.index).fillna(0.0))
    st.dataframe(summary.round(1), width="stretch")
    st.caption(
        "O AdSpend do dataset é por cliente/campanha; distribuímos igualmente entre os touchpoints "
        "da jornada daquele cliente para chegar ao investimento por canal."
    )

    st.subheader("CPA implícito por modelo")
    cpa = cpa_table(table, adspend)
    cpa_only = cpa[[c for c in cpa.columns if c.startswith("CPA")]]
    st.dataframe(cpa.round(2), width="stretch")

    long_cpa = cpa_only.reset_index().melt(id_vars="canal", var_name="modelo", value_name="CPA")
    long_cpa["modelo"] = long_cpa["modelo"].str.replace("CPA ", "", regex=False)
    fig = grouped_bar(long_cpa, x="canal", y="CPA", color="modelo",
                      title="O CPA de um canal muda conforme o modelo de atribuição escolhido")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Mesma verba, mesmas conversões: o CPA varia só por causa da regra de crédito. "
        "Por isso o modelo de atribuição precisa ser uma **decisão de governança**, não um default de ferramenta."
    )

with tab_paths:
    st.subheader("Caminhos mais frequentes")
    paths = top_paths(journeys, 15)
    fig = px.bar(paths.sort_values("clientes"), x="clientes", y="path_str", orientation="h",
                 color="taxa_conversao", color_continuous_scale="Teal",
                 title="Top caminhos e sua taxa de conversão", labels={"path_str": "jornada"})
    st.plotly_chart(apply_theme(fig, height=520, legend_bottom=False), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        by_len = journeys.groupby("n_touchpoints").agg(
            clientes=("converted", "size"), taxa=("converted", "mean")).reset_index()
        fig = px.bar(by_len, x="n_touchpoints", y="taxa", title="Taxa de conversão × nº de touchpoints",
                     color_discrete_sequence=[TEAL])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")
    with c2:
        by_type = journeys.groupby("CampaignType").agg(
            clientes=("converted", "size"), taxa=("converted", "mean")).reset_index()
        fig = px.bar(by_type.sort_values("taxa"), x="CampaignType", y="taxa",
                     title="Taxa de conversão × tipo de campanha", color_discrete_sequence=[NAVY])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")

with tab_cross:
    st.subheader("Crédito atribuído × performance real de campanhas")
    st.caption(
        "Validação cruzada com `marketing_campaign_dataset.csv` (200k campanhas). São bases diferentes "
        "e nomes de canal só parcialmente comparáveis — a leitura aqui é de **direção**, não de nível."
    )
    segment = st.session_state.get("segmento", "Todos")
    try:
        camp = campaign_channel_summary(segment)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(camp.sort_values("roi"), x="roi", y="canal", orientation="h",
                         title=f"ROI médio real por canal (segmento: {segment})",
                         color_discrete_sequence=[GOLD])
            st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")
        with c2:
            model_pick = st.selectbox("Modelo de atribuição para comparar", HEURISTIC_MODELS, index=2)
            comp = shares[model_pick].reset_index()
            comp.columns = ["canal", "crédito %"]
            fig = px.bar(comp.sort_values("crédito %"), x="crédito %", y="canal", orientation="h",
                         title=f"Crédito atribuído — {model_pick}", color_discrete_sequence=[TEAL])
            st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")
        st.dataframe(camp.round(3), width="stretch", hide_index=True)
    except Exception as exc:
        st.warning(f"Não foi possível carregar o dataset de campanhas: {exc}")

# ---------------------------------------------------------------------------
# Learning Repository
# ---------------------------------------------------------------------------
repository.save_widget(
    key="mta_heur",
    origem="MTA",
    canal_driver=str(worst_under),
    hipotese_default=f"{worst_under} está sendo subvalorizado pelo last-click e merece reavaliação de budget.",
    resultado_default="Insight",
    insight_default=(
        f"No last-click, {worst_over} recebe {diff.max():.1f} p.p. a mais que na média dos demais modelos "
        f"e {worst_under} perde {abs(diff.min()):.1f} p.p. O CPA de cada canal muda conforme a regra de crédito."
    ),
    proximo_passo_default="Confirmar com Markov/Shapley (Página 5) e desenhar um teste de incrementalidade no canal subvalorizado.",
    etapa_default="Hypothesis",
)

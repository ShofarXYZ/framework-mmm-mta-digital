"""Página 9 — Loop de Governança e Learning Repository."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import repository
from src.utils.styling import GOLD, NAVY, PALETTE, TEAL, page_header
from src.viz.charts import apply_theme, funnel

page_header(
    "Loop de Governança e Learning Repository",
    "O que transforma análises soltas em capacidade instalada: um registro vivo do que já foi "
    "testado, do que funcionou e do que a próxima decisão não precisa redescobrir.",
    layer="GOV",
)

df = repository.get_df()

# ---------------------------------------------------------------------------
# Estado vazio
# ---------------------------------------------------------------------------
if df.empty:
    st.info(
        "O repositório desta sessão ainda está vazio. Em cada página de MMM, MTA e Teste A/B há um "
        "bloco **📌 Salvar este resultado no Learning Repository** — os registros aparecem aqui.",
        icon="📭",
    )
    with st.expander("➕ Adicionar um registro manualmente"):
        with st.form("manual_entry"):
            c1, c2 = st.columns(2)
            with c1:
                origem = st.selectbox("Origem", repository.ORIGINS)
                canal = st.text_input("Canal / Driver")
                etapa = st.selectbox("Etapa", repository.STAGES, index=6)
            with c2:
                resultado = st.selectbox("Resultado", repository.RESULTS)
                lift = st.number_input("Lift (%)", value=0.0, step=0.5)
                p_value = st.number_input("P-value", value=0.05, step=0.01, format="%.4f")
            hipotese = st.text_area("Hipótese", height=68)
            insight = st.text_area("Insight", height=68)
            proximo = st.text_input("Próximo passo")
            if st.form_submit_button("Salvar", type="primary"):
                repository.add_entry(origem, canal, hipotese, resultado, insight, proximo,
                                     etapa, lift, p_value)
                st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", len(df))
c2.metric("Winners", int((df["resultado"] == "Winner").sum()))
c3.metric("Origens distintas", df["origem"].nunique())
c4.metric("Canais/drivers cobertos", df["canal_driver"].nunique())

st.divider()

tab_table, tab_funnel, tab_analysis = st.tabs(
    ["📋 Repositório", "🔻 Funil do framework", "📊 Leitura agregada"]
)

# ---------------------------------------------------------------------------
# Tabela + filtros
# ---------------------------------------------------------------------------
with tab_table:
    c1, c2, c3 = st.columns(3)
    with c1:
        f_origem = st.multiselect("Origem", sorted(df["origem"].unique()),
                                  default=sorted(df["origem"].unique()))
    with c2:
        f_canal = st.multiselect("Canal / Driver", sorted(df["canal_driver"].unique()),
                                 default=sorted(df["canal_driver"].unique()))
    with c3:
        f_result = st.multiselect("Resultado", sorted(df["resultado"].unique()),
                                  default=sorted(df["resultado"].unique()))

    filtered = df[
        df["origem"].isin(f_origem)
        & df["canal_driver"].isin(f_canal)
        & df["resultado"].isin(f_result)
    ]

    st.dataframe(
        filtered, width="stretch", hide_index=True,
        column_config={
            "experiment_id": st.column_config.TextColumn("ID", width="small"),
            "lift_pct": st.column_config.NumberColumn("Lift %", format="%.2f%%"),
            "p_value": st.column_config.NumberColumn("P-value", format="%.4f"),
            "hipotese": st.column_config.TextColumn("Hipótese", width="medium"),
            "insight": st.column_config.TextColumn("Insight", width="large"),
            "proximo_passo": st.column_config.TextColumn("Próximo passo", width="medium"),
        },
    )

    c1, c2 = st.columns([1, 4])
    with c1:
        st.download_button(
            "⬇️ Exportar CSV (formato Roadmap)",
            data=filtered.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="learning_repository.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )
    with c2:
        st.caption(
            "Colunas exatamente na ordem da planilha original de Roadmap de Testes A/B: "
            + " · ".join(repository.COLUMNS)
        )

    with st.expander("🗑️ Limpar repositório da sessão"):
        st.caption("Os registros vivem apenas em `st.session_state` — exporte o CSV antes de limpar.")
        if st.button("Confirmar limpeza"):
            repository.clear()
            st.rerun()

# ---------------------------------------------------------------------------
# Funil
# ---------------------------------------------------------------------------
with tab_funnel:
    st.subheader("Onde está cada item no framework")
    counts = df["etapa"].value_counts().reindex(repository.STAGES).fillna(0)
    st.plotly_chart(
        funnel(repository.STAGES, counts.to_numpy(),
               "Opportunity → Hypothesis → Priorization → Design → Tracking → Results → Learning"),
        width="stretch",
    )

    empty_stages = [s for s in repository.STAGES if counts.get(s, 0) == 0]
    if empty_stages:
        st.info(
            "Etapas ainda sem nenhum item: **" + ", ".join(empty_stages) + "**. "
            "Um funil saudável tem volume no topo (oportunidades) e conversão até o aprendizado — "
            "muita coisa parada em *Design* é sinal de gargalo de execução.",
            icon="🔎",
        )

    by_stage_origin = df.groupby(["etapa", "origem"]).size().reset_index(name="itens")
    fig = px.bar(by_stage_origin, x="etapa", y="itens", color="origem", barmode="stack",
                 category_orders={"etapa": repository.STAGES},
                 color_discrete_sequence=PALETTE, title="Itens por etapa e origem")
    st.plotly_chart(apply_theme(fig), width="stretch")

# ---------------------------------------------------------------------------
# Leitura agregada
# ---------------------------------------------------------------------------
with tab_analysis:
    c1, c2 = st.columns(2)
    with c1:
        by_result = df["resultado"].value_counts().reset_index()
        by_result.columns = ["resultado", "itens"]
        fig = px.pie(by_result, values="itens", names="resultado", hole=0.5,
                     title="Distribuição de resultados",
                     color_discrete_sequence=[TEAL, GOLD, NAVY, "#D62828", "#7E57C2"])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")
    with c2:
        by_origin = df["origem"].value_counts().reset_index()
        by_origin.columns = ["origem", "itens"]
        fig = px.bar(by_origin, x="origem", y="itens", title="Registros por camada do framework",
                     color_discrete_sequence=[NAVY])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")

    with_lift = df.dropna(subset=["lift_pct"])
    if len(with_lift):
        fig = px.bar(with_lift.sort_values("lift_pct"), x="lift_pct", y="experiment_id",
                     orientation="h", color="origem", color_discrete_sequence=PALETTE,
                     hover_data=["canal_driver", "resultado"],
                     title="Lift registrado por experimento")
        fig.add_vline(x=0, line_color="#CBD5E0")
        st.plotly_chart(apply_theme(fig, height=420), width="stretch")

    st.subheader("Próximos passos em aberto")
    pending = df[df["etapa"] != "Learning"][
        ["experiment_id", "origem", "canal_driver", "proximo_passo", "etapa"]
    ]
    if len(pending):
        st.dataframe(pending, width="stretch", hide_index=True)
    else:
        st.success("Nenhum item pendente — todos os aprendizados foram fechados.")

    st.caption(
        "O ciclo se fecha aqui: cada 'próximo passo' vira uma hipótese na próxima rodada de MMM, "
        "MTA ou Teste A/B. É essa memória que impede a organização de testar duas vezes a mesma coisa."
    )

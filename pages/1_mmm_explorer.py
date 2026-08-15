"""Página 1 — MMM Explorer: exploração e Data Quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    CONTROL_COLUMNS,
    DIGITAL_CHANNELS,
    IMPUTATION_METHODS,
    MEDIA_CHANNELS,
    OFFLINE_CHANNELS,
    clean_mmm,
    detect_outliers,
    label,
    load_mmm_raw,
    missing_report,
)
from src.utils.styling import GOLD, NAVY, TEAL, page_header
from src.viz.charts import apply_theme, bar_chart, heatmap, sequential

page_header(
    "MMM Explorer — Exploração e Data Quality",
    "Primeira etapa do fluxo estratégico: antes de modelar, entender a série e tratar "
    "explicitamente o que está faltando. Nenhuma imputação acontece sem decisão do analista.",
    layer="MMM",
)

try:
    raw = load_mmm_raw()
except Exception as exc:
    st.error(f"Não foi possível ler `data/mmm_dataset.csv`: {exc}")
    st.stop()

st.info(
    "Este dataset mistura **canais offline** (TV, jornal) e **digitais** (Instagram, Google Ads, "
    "YouTube, Influencer, OTT) de propósito: é justamente essa visão completa que o MTA não consegue dar.",
    icon="🏛️",
)

tab_dq, tab_outliers, tab_uni, tab_bi, tab_ts = st.tabs(
    ["🧹 Data Quality", "🎯 Outliers", "📊 Univariada", "🔗 Bivariada", "📈 Série temporal"]
)

# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------
with tab_dq:
    st.subheader("Missing values — ANTES de qualquer tratamento")
    report = missing_report(raw)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(report, width="stretch", hide_index=True)
    with col2:
        fig = bar_chart(
            report[report["faltantes"] > 0], x="% faltante", y="coluna",
            title="% de valores ausentes por coluna", orientation="h",
        )
        fig.update_traces(marker_color=GOLD)
        st.plotly_chart(fig, width="stretch")

    with st.expander("Mapa de ausência ao longo do tempo"):
        mask = raw[MEDIA_CHANNELS + ["sales"]].isna().astype(int).T
        mask.columns = raw["date"].dt.strftime("%Y-%m-%d")
        fig = px.imshow(mask, aspect="auto", color_continuous_scale=sequential(),
                        title="1 = valor ausente")
        st.plotly_chart(apply_theme(fig, height=380, legend_bottom=False), width="stretch")

    st.subheader("Método de tratamento por canal")
    st.caption(
        "Escolha, canal a canal, como preencher os buracos. A regra do framework: **decida e documente** — "
        "'replace com zero' assume que não houve investimento; 'interpolação' assume continuidade da campanha."
    )

    methods: dict[str, str] = {}
    cols = st.columns(3)
    for i, channel in enumerate(MEDIA_CHANNELS):
        n_missing = int(raw[channel].isna().sum())
        with cols[i % 3]:
            methods[channel] = st.selectbox(
                f"{label(channel)} ({n_missing} faltantes)",
                IMPUTATION_METHODS,
                index=4,  # Interpolação linear como padrão
                key=f"method_{channel}",
            )

    sales_method = st.selectbox(
        "Tratamento da variável dependente `sales` (3 faltantes)",
        IMPUTATION_METHODS, index=4, key="method_sales",
    )

    clean = clean_mmm(raw, methods, sales_method)
    st.session_state["mmm_clean"] = clean
    st.session_state["mmm_methods"] = methods

    st.success(
        f"Dataset tratado em memória e disponível para a página de Modelagem "
        f"({len(clean)} semanas, {clean[MEDIA_CHANNELS].isna().sum().sum()} células ainda ausentes). "
        "O CSV original **não foi alterado**."
    )

    st.markdown("**Antes × depois** — estatísticas descritivas")
    channel_view = st.selectbox("Canal para comparar", MEDIA_CHANNELS + ["sales"],
                                format_func=label, key="ba_channel")
    before = raw[channel_view].describe().rename("antes")
    after = clean[channel_view].describe().rename("depois")
    comparison = pd.concat([before, after], axis=1)
    comparison["Δ"] = comparison["depois"] - comparison["antes"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(comparison.round(1), width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=raw["date"], y=raw[channel_view], name="Antes (com buracos)",
                        mode="lines+markers", line=dict(color=NAVY, width=1.6), marker=dict(size=4))
        fig.add_scatter(x=clean["date"], y=clean[channel_view], name="Depois (tratado)",
                        mode="lines", line=dict(color=GOLD, width=1.6, dash="dot"))
        fig.update_layout(title=f"{label(channel_view)}: antes × depois do tratamento")
        st.plotly_chart(apply_theme(fig), width="stretch")

# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------
with tab_outliers:
    clean = st.session_state.get("mmm_clean", clean_mmm(raw, {c: "Interpolação linear" for c in MEDIA_CHANNELS}))
    st.subheader("Detecção de outliers com leitura de contexto")
    c1, c2, c3 = st.columns(3)
    with c1:
        out_col = st.selectbox("Variável", ["sales"] + MEDIA_CHANNELS, format_func=label, key="out_col")
    with c2:
        method = st.radio("Método", ["z-score", "IQR"], horizontal=True, key="out_method")
    with c3:
        threshold = st.slider("Limiar (z-score)", 1.5, 4.0, 2.5, 0.1, key="out_thr",
                              disabled=(method == "IQR"))

    mask = detect_outliers(clean, out_col, method, threshold)
    context_cols = ["date", out_col, "holiday", "sales_promotion", "sales"]
    context_cols = list(dict.fromkeys(context_cols))  # out_col pode ser o próprio 'sales'
    outliers = clean.loc[mask, context_cols]

    fig = go.Figure()
    fig.add_scatter(x=clean["date"], y=clean[out_col], mode="lines", name=label(out_col),
                    line=dict(color=TEAL, width=1.8))
    fig.add_scatter(x=clean.loc[mask, "date"], y=clean.loc[mask, out_col], mode="markers",
                    name="Outlier", marker=dict(color=GOLD, size=11, symbol="diamond"))
    fig.update_layout(title=f"Outliers em {label(out_col)} ({int(mask.sum())} pontos)")
    st.plotly_chart(apply_theme(fig), width="stretch")

    if mask.any():
        explained = outliers[(outliers["holiday"] == 1) | (outliers["sales_promotion"] != "Normal")]
        share = 100 * len(explained) / len(outliers)
        st.info(
            f"**{len(explained)} de {len(outliers)} outliers ({share:.0f}%)** coincidem com feriado ou "
            "promoção ativa — ou seja, não são erro de dado, são efeito de negócio. "
            "Esses pontos devem ser **modelados** (via dummies), não removidos.",
            icon="🎯",
        )
        st.dataframe(outliers.sort_values("date"), width="stretch", hide_index=True)
    else:
        st.success("Nenhum outlier detectado com este critério.")

# ---------------------------------------------------------------------------
# Univariada
# ---------------------------------------------------------------------------
with tab_uni:
    clean = st.session_state.get("mmm_clean", clean)
    st.subheader("Distribuição de cada variável")
    uni_col = st.selectbox("Variável", ["sales"] + MEDIA_CHANNELS + CONTROL_COLUMNS,
                           format_func=label, key="uni_col")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(clean, x=uni_col, nbins=30, title=f"Histograma — {label(uni_col)}",
                           color_discrete_sequence=[TEAL])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")
    with c2:
        fig = px.box(clean, y=uni_col, points="outliers", title=f"Boxplot — {label(uni_col)}",
                     color_discrete_sequence=[NAVY])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")

    st.markdown("**Investimento total por canal no período**")
    totals = clean[MEDIA_CHANNELS].sum().sort_values(ascending=False).reset_index()
    totals.columns = ["canal", "investimento"]
    totals["tipo"] = np.where(totals["canal"].isin(OFFLINE_CHANNELS), "Offline", "Digital")
    totals["canal"] = totals["canal"].map(label)
    fig = px.bar(totals, x="canal", y="investimento", color="tipo",
                 color_discrete_map={"Offline": NAVY, "Digital": TEAL},
                 title="Mix de investimento: offline × digital")
    st.plotly_chart(apply_theme(fig), width="stretch")

# ---------------------------------------------------------------------------
# Bivariada
# ---------------------------------------------------------------------------
with tab_bi:
    clean = st.session_state.get("mmm_clean", clean)
    st.subheader("Relação entre investimento e vendas")
    bi_col = st.selectbox("Canal", MEDIA_CHANNELS + ["competitor_spend"], format_func=label, key="bi_col")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.scatter(
            clean, x=bi_col, y="sales", trendline="ols", color="sales_promotion",
            title=f"{label(bi_col)} × Vendas", hover_data=["date", "holiday"],
        )
        st.plotly_chart(apply_theme(fig), width="stretch")
    with c2:
        corr = clean[[bi_col, "sales"]].corr().iloc[0, 1]
        st.metric(f"Correlação {label(bi_col)} × sales", f"{corr:.3f}")
        st.caption(
            "Correlação **não é** contribuição incremental: canais são investidos juntos e "
            "sofrem sazonalidade comum. É exatamente por isso que existe o MMM — "
            "com adstock, saturação e controles."
        )

    st.markdown("**Matriz de correlação** — antecipa problemas de multicolinearidade (VIF alto na Página 2)")
    corr_matrix = clean[MEDIA_CHANNELS + ["competitor_spend", "sales"]].corr().round(2)
    corr_matrix.index = [label(c) for c in corr_matrix.index]
    corr_matrix.columns = [label(c) for c in corr_matrix.columns]
    st.plotly_chart(heatmap(corr_matrix, "Correlação entre canais e vendas", "RdBu"),
                    width="stretch")

# ---------------------------------------------------------------------------
# Série temporal
# ---------------------------------------------------------------------------
with tab_ts:
    clean = st.session_state.get("mmm_clean", clean)
    st.subheader("Vendas ao longo do tempo, com feriados e promoções")

    fig = go.Figure()
    fig.add_scatter(x=clean["date"], y=clean["sales"], name="Vendas", mode="lines",
                    line=dict(color=NAVY, width=2.2))

    holidays = clean[clean["holiday"] == 1]
    if len(holidays):
        fig.add_scatter(x=holidays["date"], y=holidays["sales"], mode="markers", name="Feriado",
                        marker=dict(color=GOLD, size=10, symbol="star"))

    y_min = float(clean["sales"].min())
    promos = clean[clean["sales_promotion"] != "Normal"]
    if len(promos):
        fig.add_scatter(
            x=promos["date"], y=[y_min * 0.97] * len(promos), mode="markers",
            name="Promoção ativa", marker=dict(color=TEAL, size=8, symbol="line-ns-open"),
            text=promos["sales_promotion"], hovertemplate="%{text}<extra></extra>",
        )
    fig.update_layout(title="Vendas semanais · rug plot de promoções · estrelas = feriado")
    st.plotly_chart(apply_theme(fig, height=460), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        by_promo = clean.groupby("sales_promotion")["sales"].agg(["mean", "count"]).reset_index()
        by_promo.columns = ["promoção", "venda média", "semanas"]
        fig = px.bar(by_promo.sort_values("venda média", ascending=False), x="promoção",
                     y="venda média", title="Venda média por tipo de promoção",
                     color_discrete_sequence=[TEAL])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")
    with c2:
        by_holiday = clean.groupby("holiday")["sales"].mean().reset_index()
        by_holiday["holiday"] = by_holiday["holiday"].map({0: "Sem feriado", 1: "Feriado"})
        fig = px.bar(by_holiday, x="holiday", y="sales", title="Venda média: feriado × dia normal",
                     color_discrete_sequence=[NAVY])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")

    st.markdown(
        "➡️ **Próximo passo:** com o dado tratado, vá para **MMM Modelagem** para decompor "
        "essas vendas em Base + contribuição de cada canal."
    )

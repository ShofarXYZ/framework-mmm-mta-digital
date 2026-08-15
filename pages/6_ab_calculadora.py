"""Página 6 — Teste A/B: calculadoras de duração e tamanho de amostra."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.abtest.power import (
    CONFIDENCE_LEVELS,
    POWER_LEVELS,
    duration_estimation,
    mde_for_sample,
    midrange_impact,
    sample_size_per_variation,
    sample_size_v2,
)
from src.data_loader import load_digital
from src.utils import repository
from src.utils.styling import GOLD, NAVY, TEAL, page_header
from src.viz.charts import apply_theme

page_header(
    "Teste A/B — Calculadora de Duração e Amostra",
    "Antes de subir o teste: quantos visitantes e quantos dias são necessários para detectar o efeito "
    "esperado. Testar sem poder estatístico é queimar tráfego para produzir um 'não sei'.",
    layer="AB",
)

# ---------------------------------------------------------------------------
# Pré-população com dado real
# ---------------------------------------------------------------------------
try:
    digital = load_digital()
    channels = ["Todos"] + sorted(digital["CampaignChannel"].dropna().unique().tolist())
except Exception as exc:
    st.error(f"Não foi possível ler o dataset digital: {exc}")
    st.stop()

c1, c2 = st.columns([1, 3])
with c1:
    channel = st.selectbox("Canal de referência", channels, index=0)
with c2:
    st.caption(
        "Os campos abaixo já vêm **pré-populados com dado real** deste canal "
        "(`digital_marketing_campaign_dataset.csv`) — a calculadora não nasce em branco como na planilha."
    )

subset = digital if channel == "Todos" else digital[digital["CampaignChannel"] == channel]
real_rate = float(subset["Conversion"].mean())
real_visits = float(subset["WebsiteVisits"].mean())
n_customers = int(len(subset))

k1, k2, k3 = st.columns(3)
k1.metric("Taxa de conversão real", f"{real_rate * 100:.2f}%", f"canal: {channel}")
k2.metric("Clientes na base", f"{n_customers:,}".replace(",", "."))
k3.metric("Visitas médias por cliente", f"{real_visits:.1f}")

st.divider()

with st.sidebar:
    st.header("⚙️ Parâmetros estatísticos")
    confidence_label = st.selectbox("Nível de confiança", list(CONFIDENCE_LEVELS), index=2)
    power_label = st.selectbox("Poder estatístico", list(POWER_LEVELS), index=0)
confidence = CONFIDENCE_LEVELS[confidence_label]
power = POWER_LEVELS[power_label]

tab_a, tab_b, tab_c = st.tabs(
    ["🅰️ Duration Estimation", "🅱️ Mid-range Impact", "🅲 Sample Size / Duration V2"]
)

# ---------------------------------------------------------------------------
# (a) Duration Estimation
# ---------------------------------------------------------------------------
with tab_a:
    st.subheader("Quantos dias o teste precisa rodar?")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_variations = st.number_input("Nº de variações (com o controle)", 2, 8, 2, key="a_var")
        users_day = st.number_input("Usuários por dia", 100, 5_000_000, 5000, 100, key="a_users")
    with c2:
        traffic_pct = st.slider("% do tráfego alocado ao teste", 5, 100, 100, 5, key="a_traffic")
        baseline = st.number_input("Taxa de conversão atual (%)", 0.01, 99.0,
                                   float(round(real_rate * 100, 2)), 0.01, key="a_base")
    with c3:
        uplift = st.number_input("Uplift esperado (%)", 0.1, 200.0, 10.0, 0.5, key="a_uplift")

    res = duration_estimation(int(n_variations), users_day, traffic_pct, baseline / 100,
                              uplift, confidence, power)

    m1, m2, m3 = st.columns(3)
    m1.metric("Amostra por variação", f"{res['amostra_por_variacao']:,.0f}".replace(",", "."))
    m2.metric("Amostra total", f"{res['amostra_total']:,.0f}".replace(",", "."))
    m3.metric("Duração estimada", f"{res['dias']:,.0f} dias".replace(",", "."),
              f"{res['semanas']:.1f} semanas")

    if res["dias"] > 56:
        st.warning(
            f"⏳ {res['dias']:.0f} dias é longo demais para um único teste — o risco de contaminação "
            "por sazonalidade e mudanças de produto cresce. Considere aumentar o tráfego alocado, "
            "aceitar um uplift mínimo detectável maior, ou usar teste sequencial (Página 7).",
        )
    elif res["dias"] < 7:
        st.info("Menos de uma semana: rode pelo menos 7 dias completos para cobrir o ciclo semanal.")

    st.markdown("**Sensibilidade:** como a duração muda conforme o uplift esperado")
    grid = np.arange(2, 31, 1.0)
    days = [duration_estimation(int(n_variations), users_day, traffic_pct, baseline / 100, u,
                                confidence, power)["dias"] for u in grid]
    fig = go.Figure(go.Scatter(x=grid, y=days, mode="lines", line=dict(color=TEAL, width=3)))
    fig.add_vline(x=uplift, line_dash="dot", line_color=GOLD, annotation_text="seu uplift")
    fig.add_hline(y=28, line_dash="dot", line_color=NAVY, annotation_text="4 semanas")
    fig.update_layout(title="Uplift esperado × dias necessários", xaxis_title="uplift (%)",
                      yaxis_title="dias", yaxis_type="log")
    st.plotly_chart(apply_theme(fig), width="stretch")
    st.caption("Escala logarítmica: detectar efeitos pequenos custa **muito** mais tráfego — a relação é quadrática.")

# ---------------------------------------------------------------------------
# (b) Mid-range Impact
# ---------------------------------------------------------------------------
with tab_b:
    st.subheader("Quantas conversões incrementais o teste deve gerar?")
    c1, c2 = st.columns(2)
    with c1:
        total_sessions = st.number_input("Total de sessões no período", 1000, 50_000_000,
                                         int(max(n_customers * 10, 10000)), 1000, key="b_sess")
        allocation = st.slider("% alocado ao experimento", 5, 100, 50, 5, key="b_alloc")
    with c2:
        total_conv = st.number_input(
            "Conversões totais no período", 1, 10_000_000,
            int(max(round(total_sessions * real_rate), 1)), 10, key="b_conv")
        uplift_b = st.number_input("Uplift esperado (%)", 0.1, 200.0, 10.0, 0.5, key="b_uplift")

    res_b = midrange_impact(total_sessions, allocation, total_conv, uplift_b)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sessões no teste", f"{res_b['sessoes_no_teste']:,.0f}".replace(",", "."))
    m2.metric("Taxa atual", f"{res_b['taxa_atual'] * 100:.2f}%")
    m3.metric("Taxa projetada", f"{res_b['taxa_projetada'] * 100:.2f}%",
              f"{uplift_b:+.1f}% relativo")
    m4.metric("Conversões incrementais", f"{res_b['conversoes_incrementais']:,.0f}".replace(",", "."))

    fig = go.Figure()
    fig.add_bar(x=["Cenário base", "Cenário com uplift"],
                y=[res_b["conversoes_base"], res_b["conversoes_projetadas"]],
                marker_color=[NAVY, GOLD],
                text=[f"{res_b['conversoes_base']:,.0f}", f"{res_b['conversoes_projetadas']:,.0f}"],
                textposition="outside")
    fig.update_layout(title="Conversões esperadas no período do experimento")
    st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")
    st.caption(
        "Este número é o que sustenta o business case do teste: se o incremento não paga o custo "
        "de desenvolvimento e o risco de perder tráfego, o teste não deveria entrar no roadmap."
    )

# ---------------------------------------------------------------------------
# (c) Sample Size V2
# ---------------------------------------------------------------------------
with tab_c:
    st.subheader("Tamanho de amostra e duração — versão completa")
    c1, c2, c3 = st.columns(3)
    with c1:
        base_c = st.number_input("Conversão atual (%)", 0.01, 99.0,
                                 float(round(real_rate * 100, 2)), 0.01, key="c_base")
        uplift_c = st.number_input("Uplift esperado (%)", 0.1, 200.0, 10.0, 0.5, key="c_uplift")
    with c2:
        var_c = st.number_input("Nº de variações", 2, 8, 2, key="c_var")
        daily_c = st.number_input("Visitantes médios por dia", 100, 5_000_000, 5000, 100, key="c_daily")
    with c3:
        st.caption(f"Confiança: **{confidence_label}** · Poder: **{power_label}** "
                   "(ajuste na barra lateral)")

    res_c = sample_size_v2(base_c / 100, uplift_c, int(var_c), daily_c, confidence, power)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Visitantes por variação", f"{res_c['amostra_por_variacao']:,.0f}".replace(",", "."))
    m2.metric("Visitantes no total", f"{res_c['amostra_total']:,.0f}".replace(",", "."))
    m3.metric("Duração", f"{res_c['dias']:,.0f} dias".replace(",", "."), f"{res_c['semanas']:.1f} semanas")
    m4.metric("MDE absoluto", f"{res_c['mde_absoluto'] * 100:.3f} p.p.")

    st.markdown("**E se eu só tiver X dias?** Efeito mínimo detectável conforme a duração disponível")
    days_grid = np.array([7, 14, 21, 28, 42, 56, 90])
    rows = []
    for d in days_grid:
        n_per_var = daily_c * d / max(int(var_c), 2)
        rows.append({"dias": int(d), "amostra por variação": n_per_var,
                     "MDE (%)": mde_for_sample(base_c / 100, n_per_var, confidence, power)})
    mde_df = pd.DataFrame(rows)
    fig = go.Figure(go.Scatter(x=mde_df["dias"], y=mde_df["MDE (%)"], mode="lines+markers",
                               line=dict(color=GOLD, width=3)))
    fig.add_hline(y=uplift_c, line_dash="dot", line_color=TEAL,
                  annotation_text="uplift que você espera")
    fig.update_layout(title="Duração disponível × menor efeito que o teste consegue detectar",
                      xaxis_title="dias de teste", yaxis_title="MDE (%)")
    st.plotly_chart(apply_theme(fig), width="stretch")
    st.dataframe(mde_df.round(2), width="stretch", hide_index=True)
    st.caption(
        "Leitura: se o MDE da duração escolhida for **maior** que o uplift que você espera, o teste "
        "vai terminar 'inconclusivo' mesmo que a variação seja de fato melhor."
    )

# ---------------------------------------------------------------------------
# Matemática
# ---------------------------------------------------------------------------
with st.expander("🧮 Ver a matemática por trás"):
    st.markdown(
        r"""
Tamanho de amostra para **teste de duas proporções** (bicaudal):

$$
n_{\text{por variação}} = \frac{\left(z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} +
z_{\beta}\sqrt{p_1(1-p_1) + p_2(1-p_2)}\right)^2}{(p_2 - p_1)^2}
$$

onde:
- $p_1$ = taxa de conversão atual (controle)
- $p_2 = p_1 \cdot (1 + \text{uplift})$ = taxa esperada na variação
- $\bar{p} = (p_1 + p_2)/2$ = taxa combinada
- $z_{\alpha/2}$ = valor crítico do nível de confiança (1,96 para 95%)
- $z_{\beta}$ = valor crítico do poder (0,84 para 80%)

**Duração** = $n_{\text{total}} / (\text{visitantes por dia} \times \% \text{alocado})$,
com $n_{\text{total}} = n_{\text{por variação}} \times \text{nº de variações}$.

O denominador $(p_2-p_1)^2$ explica a regra prática: **detectar metade do efeito custa quatro
vezes mais amostra.**

Implementação em `src/abtest/power.py`.
        """
    )

repository.save_widget(
    key="ab_calc",
    origem="Teste A/B",
    canal_driver=channel,
    hipotese_default=f"Um uplift de {uplift:.0f}% em {channel} é detectável dentro do tráfego disponível.",
    resultado_default="Insight",
    insight_default=(
        f"Com conversão base de {real_rate * 100:.2f}% e confiança {confidence_label}/poder {power_label}, "
        f"o teste exige {res['amostra_por_variacao']:,.0f} visitantes por variação "
        f"({res['dias']:.0f} dias)."
    ).replace(",", "."),
    proximo_passo_default="Priorizar no roadmap e desenhar o experimento (tracking + critério de parada).",
    etapa_default="Design",
)

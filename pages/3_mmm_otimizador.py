"""Página 3 — MMM Otimizador de Budget."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import label
from src.mmm.optimizer import current_allocation, optimize_budget, what_if
from src.utils import repository
from src.utils.styling import GOLD, NAVY, fmt_money, page_header
from src.viz.charts import apply_theme, waterfall_compare

page_header(
    "Otimizador de Budget",
    "A saída prática do MMM: dado o mesmo orçamento, como redistribuir entre canais para "
    "vender mais — respeitando saturação e restrições de negócio.",
    layer="MMM",
)

result = st.session_state.get("mmm_result")
if result is None:
    st.warning(
        "Nenhum modelo ajustado nesta sessão. Vá até **🧪 MMM Modelagem**, configure e clique em "
        "*Rodar modelo* — o resultado fica disponível aqui automaticamente.",
        icon="👈",
    )
    st.stop()

base = current_allocation(result)
total_current = float(base.sum())

st.caption(
    f"Modelo em uso: **{result.config.form} / {result.config.regularizer}** · "
    f"MAPE de holdout {result.metrics['mape_holdout']:.1f}% · "
    f"investimento atual no período: {fmt_money(total_current)}"
)

tab_opt, tab_whatif = st.tabs(["🎯 Otimização", "🎚️ Cenário 'E se'"])

# ---------------------------------------------------------------------------
# Otimização
# ---------------------------------------------------------------------------
with tab_opt:
    c1, c2 = st.columns([2, 3])
    with c1:
        budget = st.number_input(
            "Orçamento total disponível", min_value=float(total_current * 0.3),
            max_value=float(total_current * 3.0), value=float(total_current), step=float(total_current * 0.05),
            help="Mesma unidade e mesmo horizonte do dataset (todo o período histórico).",
        )
        budget_delta = 100 * (budget - total_current) / total_current
        st.caption(f"{budget_delta:+.1f}% em relação ao investimento atual.")

    with c2:
        st.markdown("**Restrições por canal** (% do investimento atual do canal)")
        st.caption("Ex.: mínimo 50% em TV para não perder presença de marca; máximo 150% em Influencer.")

    bounds: dict[str, tuple[float, float]] = {}
    cols = st.columns(3)
    for i, channel in enumerate(result.config.media_columns):
        current = float(base[channel])
        with cols[i % 3]:
            lo, hi = st.slider(
                label(channel), 0, 300, (50, 150), 10, key=f"bound_{channel}",
                help=f"Atual: {fmt_money(current)}",
            )
            bounds[channel] = (current * lo / 100, current * hi / 100)

    if st.button("🎯 Otimizar alocação", type="primary"):
        with st.spinner("Rodando SLSQP sobre as curvas de resposta..."):
            st.session_state["mmm_opt"] = optimize_budget(result, budget, bounds)

    opt = st.session_state.get("mmm_opt")
    if opt is None:
        st.info("Defina o orçamento e as restrições e clique em **🎯 Otimizar alocação**.")
    elif not opt["ok"]:
        st.error(f"Otimização inviável: {opt['message']}")
    else:
        if not opt.get("converged", True):
            st.warning(f"O otimizador não convergiu plenamente: {opt['message']}")

        lift = opt["lift_pct"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Vendas — alocação atual", fmt_money(opt["baseline_sales"], ""))
        c2.metric("Vendas — alocação ótima", fmt_money(opt["optimal_sales"], ""))
        c3.metric("Lift esperado", f"{lift:+.2f}%")

        st.markdown(
            f"<div style='background:{GOLD}22;border-left:5px solid {GOLD};padding:16px 20px;"
            f"border-radius:8px'><span style='font-size:2.4rem;font-weight:800;color:{NAVY}'>"
            f"{lift:+.2f}%</span><br><span style='color:#4A5568'>de lift esperado em vendas "
            "apenas realocando o mesmo orçamento entre canais.</span></div>",
            unsafe_allow_html=True,
        )
        st.write("")

        table = opt["table"].copy()
        labels = [label(c) for c in table["canal"]]
        st.plotly_chart(
            waterfall_compare(labels, table["alocacao_atual"], table["alocacao_otima"],
                              "Alocação atual × alocação ótima"),
            width="stretch",
        )

        display = table.copy()
        display["canal"] = labels
        st.dataframe(
            display.round(1), width="stretch", hide_index=True,
            column_config={
                "alocacao_atual": st.column_config.NumberColumn("Atual", format="%.0f"),
                "alocacao_otima": st.column_config.NumberColumn("Ótima", format="%.0f"),
                "delta": st.column_config.NumberColumn("Δ", format="%.0f"),
                "delta_%": st.column_config.NumberColumn("Δ %", format="%.1f%%"),
                "share_atual_%": st.column_config.NumberColumn("Share atual", format="%.1f%%"),
                "share_otimo_%": st.column_config.NumberColumn("Share ótimo", format="%.1f%%"),
            },
        )

        movers = display.reindex(display["delta"].abs().sort_values(ascending=False).index)
        top_up = movers[movers["delta"] > 0].head(1)
        top_down = movers[movers["delta"] < 0].head(1)
        recommendation = []
        if len(top_up):
            recommendation.append(
                f"**aumentar {top_up.iloc[0]['canal']}** em {top_up.iloc[0]['delta_%']:.0f}%")
        if len(top_down):
            recommendation.append(
                f"**reduzir {top_down.iloc[0]['canal']}** em {abs(top_down.iloc[0]['delta_%']):.0f}%")
        if recommendation:
            st.info("Principal movimento sugerido: " + " e ".join(recommendation) +
                    ". Valide com um experimento antes de executar.", icon="💡")

        repository.save_widget(
            key="mmm_opt",
            origem="MMM",
            canal_driver=top_up.iloc[0]["canal"] if len(top_up) else "Mix de mídia",
            hipotese_default="Realocar o orçamento conforme as curvas de resposta aumenta as vendas "
                             "sem investimento adicional.",
            resultado_default="Oportunidade",
            insight_default=f"A otimização SLSQP indica lift de {lift:+.2f}% mantendo o mesmo orçamento "
                            f"de {fmt_money(budget)}. " + ("Movimento: " + " e ".join(recommendation) if recommendation else ""),
            proximo_passo_default="Rodar um geo-holdout no canal com maior aumento sugerido para confirmar a incrementalidade.",
            lift_pct=lift,
            etapa_default="Priorization",
        )

# ---------------------------------------------------------------------------
# Cenário "E se"
# ---------------------------------------------------------------------------
with tab_whatif:
    st.subheader("Simulação manual — resposta instantânea, sem re-otimizar")
    st.caption("Mova os sliders para simular +/- % de investimento por canal.")

    deltas: dict[str, float] = {}
    cols = st.columns(3)
    for i, channel in enumerate(result.config.media_columns):
        with cols[i % 3]:
            deltas[channel] = st.slider(f"{label(channel)} (%)", -100, 200, 0, 5,
                                        key=f"whatif_{channel}")

    scenario = what_if(result, deltas)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vendas base", fmt_money(scenario["baseline_sales"], ""))
    c2.metric("Vendas no cenário", fmt_money(scenario["scenario_sales"], ""),
              f"{scenario['lift_pct']:+.2f}%")
    c3.metric("Investimento base", fmt_money(scenario["baseline_spend"]))
    spend_delta = 100 * (scenario["scenario_spend"] - scenario["baseline_spend"]) / scenario["baseline_spend"]
    c4.metric("Investimento no cenário", fmt_money(scenario["scenario_spend"]), f"{spend_delta:+.1f}%")

    incremental_spend = scenario["scenario_spend"] - scenario["baseline_spend"]
    if abs(incremental_spend) > 1:
        marginal_roi = scenario["delta_sales"] / incremental_spend
        st.metric("ROI marginal do movimento", f"{marginal_roi:.2f} venda por R$",
                  help="Vendas incrementais divididas pelo investimento incremental do cenário.")

    spend_table = scenario["spend_by_channel"].copy()
    spend_table["canal"] = spend_table["canal"].map(label)
    st.plotly_chart(
        waterfall_compare(spend_table["canal"], spend_table["atual"], spend_table["cenario"],
                          "Investimento: atual × cenário simulado"),
        width="stretch",
    )

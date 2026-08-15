"""Página 8 — Geo-Holdout simulado, ligado ao MMM (fecha o ciclo MMM → experimento)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.abtest.geo_holdout import did_table, run_geo_holdout
from src.data_loader import label
from src.utils import repository
from src.utils.styling import GOLD, NAVY, TEAL, fmt_money, page_header
from src.viz.charts import apply_theme

page_header(
    "Geo-Holdout × MMM",
    "O elo que fecha o ciclo: o MMM diz quanto um canal vale, o experimento de holdout tenta provar. "
    "Quando os dois discordam, quem manda é o experimento — e o modelo é recalibrado.",
    layer="AB",
)

result = st.session_state.get("mmm_result")
if result is None:
    st.warning(
        "Este teste usa o modelo ajustado na página **🧪 MMM Modelagem** como contrafactual. "
        "Rode o modelo lá primeiro.",
        icon="👈",
    )
    st.stop()

st.info(
    "**Limitação assumida:** `mmm_dataset.csv` não tem recorte geográfico, então um geo-experiment real "
    "(mercados teste × controle) não é possível. Simulamos um **holdout temporal**: você escolhe um canal, "
    "uma janela de semanas e um % de corte. O contrafactual vem do próprio MMM. É uma aproximação "
    "pedagógica da mecânica do teste, não um experimento real.",
    icon="🌍",
)

data = result.data
n_weeks_total = len(data)

c1, c2, c3 = st.columns(3)
with c1:
    channel = st.selectbox("Canal em holdout", result.config.media_columns, format_func=label,
                           index=min(3, len(result.config.media_columns) - 1))
with c2:
    window_len = st.slider("Duração do holdout (semanas)", 2, 16, 8)
with c3:
    reduction = st.slider("Redução de investimento no período (%)", 10, 100, 100, 5,
                          help="100% = holdout puro (canal desligado na janela).")

start = st.slider(
    "Semana de início da janela", 0, max(n_weeks_total - window_len - 1, 0),
    max(n_weeks_total - window_len - 12, 0),
    format="%d",
)
st.caption(
    f"Janela: **{data['date'].iloc[start]:%d/%m/%Y}** a "
    f"**{data['date'].iloc[min(start + window_len - 1, n_weeks_total - 1)]:%d/%m/%Y}**"
)

try:
    res = run_geo_holdout(result, channel, start, window_len, reduction)
except Exception as exc:
    st.error(f"Falha ao rodar o holdout: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Vendas reais na janela", fmt_money(res["real_post"], ""))
m2.metric("Contrafactual MMM (base)", fmt_money(res["base_post"], ""))
m3.metric("Cenário com holdout", fmt_money(res["holdout_post"], ""),
          f"{res['lift_previsto_mmm_%']:+.2f}%")
m4.metric("Investimento cortado", fmt_money(res["investimento_cortado"]))

alert_render = {"error": st.error, "warning": st.warning, "success": st.success, "info": st.info}
alert_render[res["nivel"]](res["alerta"])

tab_series, tab_did, tab_compare = st.tabs(
    ["📈 Séries", "🧮 Diferença-em-diferenças", "⚖️ Medido × previsto"]
)

with tab_series:
    series = res["series"]
    fig = go.Figure()
    fig.add_scatter(x=series["date"], y=series["real"], name="Real observado",
                    line=dict(color=NAVY, width=2.4))
    fig.add_scatter(x=series["date"], y=series["contrafactual_base"], name="Contrafactual MMM (sem corte)",
                    line=dict(color=TEAL, width=2, dash="dash"))
    fig.add_scatter(x=series["date"], y=series["cenario_holdout"], name="Cenário com holdout",
                    line=dict(color=GOLD, width=2, dash="dot"))

    window_dates = series.loc[series["janela"], "date"]
    if len(window_dates):
        fig.add_vrect(x0=window_dates.iloc[0], x1=window_dates.iloc[-1], fillcolor=GOLD,
                      opacity=0.12, line_width=0, annotation_text="janela de holdout",
                      annotation_position="top left")
    pre_dates = series.loc[series["pre"], "date"]
    if len(pre_dates):
        fig.add_vrect(x0=pre_dates.iloc[0], x1=pre_dates.iloc[-1], fillcolor=TEAL,
                      opacity=0.07, line_width=0, annotation_text="período pré",
                      annotation_position="top left")

    fig.update_layout(title=f"Holdout de {label(channel)} — real × contrafactual × cenário")
    st.plotly_chart(apply_theme(fig, height=470), width="stretch")

    st.metric("Vendas perdidas previstas pelo MMM no corte",
              fmt_money(res["vendas_perdidas_previstas"], ""),
              help="Diferença entre o contrafactual base e o cenário com o investimento cortado.")

with tab_did:
    st.subheader("Diferença-em-diferenças")
    st.markdown(
        "A DiD compara a variação do **real observado** com a variação do **contrafactual do MMM** "
        "entre o período pré e a janela. Se o modelo estivesse perfeitamente calibrado, as duas "
        "variações seriam iguais e a DiD seria zero."
    )
    table = did_table(res)
    st.dataframe(table.round(0), width="stretch", hide_index=True)

    if np.isfinite(res["did_absoluto"]):
        c1, c2 = st.columns(2)
        c1.metric("DiD absoluto", fmt_money(res["did_absoluto"], ""))
        c2.metric("DiD relativo", f"{res['did_%']:+.2f}%")

        fig = go.Figure(go.Waterfall(
            x=["Δ Real (pós−pré)", "Δ Contrafactual (pós−pré)", "DiD"],
            measure=["relative", "relative", "total"],
            y=[res["real_post"] - res["real_pre"],
               -(res["base_post"] - res["base_pre"]),
               0],
            increasing=dict(marker=dict(color=TEAL)),
            decreasing=dict(marker=dict(color="#D62828")),
            totals=dict(marker=dict(color=GOLD)),
        ))
        fig.update_layout(title="Composição da diferença-em-diferenças")
        st.plotly_chart(apply_theme(fig, height=380, legend_bottom=False), width="stretch")
    else:
        st.warning(
            "Não há período pré suficiente antes da janela escolhida para calcular a DiD. "
            "Mova a janela para mais adiante na série."
        )

with tab_compare:
    st.subheader("O lift medido bate com o que o MMM previa?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Lift previsto pelo MMM", f"{res['lift_previsto_mmm_%']:+.2f}%",
              help="Impacto que o modelo atribui ao corte de investimento na janela.")
    c2.metric("Lift medido no holdout", f"{res['lift_medido_%']:+.2f}%",
              help="Quanto o real observado ficou acima/abaixo do contrafactual do modelo.")
    c3.metric("Divergência", f"{res['divergencia_pp']:.1f} p.p.")

    fig = go.Figure()
    fig.add_bar(x=["Previsto pelo MMM", "Medido no holdout"],
                y=[res["lift_previsto_mmm_%"], res["lift_medido_%"]],
                marker_color=[TEAL, GOLD],
                text=[f"{res['lift_previsto_mmm_%']:+.2f}%", f"{res['lift_medido_%']:+.2f}%"],
                textposition="outside")
    fig.add_hline(y=0, line_color="#CBD5E0")
    fig.update_layout(title="Modelo × experimento", yaxis_title="lift (%)")
    st.plotly_chart(apply_theme(fig, height=380, legend_bottom=False), width="stretch")

    st.markdown(
        """
**Como usar esta leitura no loop de governança:**

| Divergência | Interpretação | Ação |
|---|---|---|
| até ~7 p.p. | o MMM está calibrado para o canal | manter o modelo e seguir com a alocação sugerida |
| 7 a 15 p.p. | sinal amarelo | rodar um teste confirmatório antes de mover budget |
| acima de 15 p.p. | o MMM está super ou subestimando o canal | **recalibrar**: revisar adstock, saturação, controles e priors |

O resultado do experimento sempre vence o modelo — é ele que vira prior do próximo MMM.
        """
    )

repository.save_widget(
    key="geo_holdout",
    origem="Geo-Holdout",
    canal_driver=label(channel),
    hipotese_default=f"Cortar {reduction:.0f}% do investimento em {label(channel)} por {window_len} "
                     "semanas reduz as vendas na magnitude prevista pelo MMM.",
    resultado_default="Insight" if res["nivel"] == "success" else "Oportunidade",
    insight_default=(
        f"Lift previsto pelo MMM: {res['lift_previsto_mmm_%']:+.2f}%. "
        f"Lift medido no holdout: {res['lift_medido_%']:+.2f}%. "
        f"Divergência de {res['divergencia_pp']:.1f} p.p. DiD: {res['did_%']:+.2f}%."
    ),
    proximo_passo_default=(
        "Manter o modelo e executar a realocação sugerida pelo otimizador."
        if res["nivel"] == "success"
        else f"Recalibrar o MMM para {label(channel)} (adstock, saturação e controles) e repetir o holdout."
    ),
    lift_pct=res["lift_medido_%"],
    etapa_default="Results",
)

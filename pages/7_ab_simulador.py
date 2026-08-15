"""Página 7 — Teste A/B: simulador e leitura de resultados (significância)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.abtest.bayesian import bayesian_test, decision_label, posterior_curves
from src.abtest.frequentist import observed_power, two_proportion_ztest
from src.abtest.sequential import sprt_test, sprt_trajectory
from src.data_loader import load_campaigns, load_digital
from src.utils import repository
from src.utils.styling import GOLD, NAVY, TEAL, page_header
from src.viz.charts import apply_theme

page_header(
    "Teste A/B — Simulador e Resultados",
    "A leitura do experimento em três lentes: frequentista (p-value), bayesiana (probabilidade de "
    "vencer) e sequencial (já dá para parar?). O veredito final é Winner / Neutral / Loser.",
    layer="AB",
)

VERDICT_STYLE = {
    "Winner": ("#1B7F3B", "#E7F6EC", "🏆"),
    "Loser": ("#B3261E", "#FDECEA", "🔻"),
    "Neutral": ("#8A6D1F", "#FEF6E7", "➖"),
}


def verdict_banner(verdict: str, lift: float, p_value: float) -> None:
    color, background, icon = VERDICT_STYLE[verdict]
    st.markdown(
        f"<div style='background:{background};border-left:6px solid {color};padding:18px 22px;"
        f"border-radius:10px'><span style='font-size:2.1rem;font-weight:800;color:{color}'>"
        f"{icon} {verdict.upper()}</span><br>"
        f"<span style='color:#4A5568;font-size:1rem'>Lift de <b>{lift:+.2f}%</b> · "
        f"p-value <b>{p_value:.4f}</b></span></div>",
        unsafe_allow_html=True,
    )


def significance_table(res: dict) -> pd.DataFrame:
    rows = []
    for level, label_ in ((0.90, "90%"), (0.95, "95%"), (0.99, "99%")):
        info = res["intervalos"][level]
        rows.append(
            {
                "Significant At": label_,
                "Resultado": "YES" if info["significativo"] else "NO",
                "IC Control": f"{info['control'][0] * 100:.2f}% – {info['control'][1] * 100:.2f}%",
                "IC Variation": f"{info['variation'][0] * 100:.2f}% – {info['variation'][1] * 100:.2f}%",
            }
        )
    return pd.DataFrame(rows)


def render_results(conv_a: int, n_a: int, conv_b: int, n_b: int, name_a: str, name_b: str,
                   key: str) -> None:
    """Bloco completo de análise: frequentista + bayesiano + sequencial."""
    try:
        freq = two_proportion_ztest(conv_a, n_a, conv_b, n_b)
    except Exception as exc:
        st.error(f"Não foi possível rodar o teste: {exc}")
        return

    verdict_banner(freq["veredito"], freq["lift_pct"], freq["p_value"])
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Control — {name_a}", f"{freq['control']['taxa'] * 100:.2f}%",
              f"{conv_a:,}/{n_a:,}".replace(",", "."))
    c2.metric(f"Variation — {name_b}", f"{freq['variation']['taxa'] * 100:.2f}%",
              f"{conv_b:,}/{n_b:,}".replace(",", "."))
    c3.metric("Z-score", f"{freq['z_score']:.3f}")
    c4.metric("P-value", f"{freq['p_value']:.4f}")

    tab_f, tab_b, tab_s = st.tabs(["📐 Frequentista", "🎲 Bayesiano", "⏱️ Sequencial (SPRT)"])

    # --- Frequentista ---
    with tab_f:
        st.dataframe(significance_table(freq), width="stretch", hide_index=True)

        fig = go.Figure()
        for level, color in ((0.99, "#CBD5E0"), (0.95, TEAL), (0.90, NAVY)):
            info = freq["intervalos"][level]
            for i, (variant, name) in enumerate(((info["control"], name_a), (info["variation"], name_b))):
                fig.add_trace(go.Scatter(
                    x=[variant[0] * 100, variant[1] * 100], y=[name, name],
                    mode="lines", line=dict(color=color, width=10 - level * 6),
                    name=f"IC {int(level * 100)}%", showlegend=(i == 0),
                    hovertemplate=f"IC {int(level*100)}%: %{{x:.2f}}%<extra></extra>",
                ))
        fig.add_trace(go.Scatter(
            x=[freq["control"]["taxa"] * 100, freq["variation"]["taxa"] * 100], y=[name_a, name_b],
            mode="markers", marker=dict(color=GOLD, size=14, symbol="diamond"), name="taxa observada"))
        fig.update_layout(title="Intervalos de confiança 90 / 95 / 99%",
                          xaxis_title="taxa de conversão (%)")
        st.plotly_chart(apply_theme(fig, height=340), width="stretch")

        c1, c2, c3 = st.columns(3)
        c1.metric("Lift absoluto", f"{freq['lift_absoluto'] * 100:+.3f} p.p.")
        c2.metric("Qui-quadrado (p)", f"{freq['chi2_p']:.4f}")
        c3.metric("Poder observado", f"{observed_power(conv_a, n_a, conv_b, n_b) * 100:.1f}%")
        st.caption(f"Engine estatística: {freq['engine']}. "
                   "Veredito: significativo a 95% **e** lift positivo = Winner; significativo e negativo = "
                   "Loser; não significativo = Neutral.")

    # --- Bayesiano ---
    with tab_b:
        bayes = bayesian_test(conv_a, n_a, conv_b, n_b)
        c1, c2, c3 = st.columns(3)
        c1.metric("P(Variation > Control)", f"{bayes['prob_b_maior_a'] * 100:.2f}%")
        c2.metric("Uplift esperado", f"{bayes['uplift_esperado_%']:+.2f}%",
                  f"HDI 95%: {bayes['uplift_hdi_95'][0]:+.1f}% a {bayes['uplift_hdi_95'][1]:+.1f}%")
        c3.metric("Veredito bayesiano", decision_label(bayes["prob_b_maior_a"]))

        x, pdf_a, pdf_b = posterior_curves(bayes)
        fig = go.Figure()
        fig.add_scatter(x=x * 100, y=pdf_a, name=f"Control — {name_a}", fill="tozeroy",
                        line=dict(color=NAVY, width=2), fillcolor="rgba(33,41,92,0.25)")
        fig.add_scatter(x=x * 100, y=pdf_b, name=f"Variation — {name_b}", fill="tozeroy",
                        line=dict(color=GOLD, width=2), fillcolor="rgba(232,163,61,0.30)")
        fig.update_layout(title="Distribuições posteriores (Beta-Binomial, prior Beta(1,1))",
                          xaxis_title="taxa de conversão (%)", yaxis_title="densidade")
        st.plotly_chart(apply_theme(fig, height=400), width="stretch")

        st.caption(
            f"Perda esperada ao escolher a variação: **{bayes['perda_esperada_escolher_B'] * 100:.4f} p.p.** · "
            f"ao manter o controle: **{bayes['perda_esperada_escolher_A'] * 100:.4f} p.p.** "
            "A leitura bayesiana responde 'qual a chance de eu estar certo?', "
            "que é a pergunta que o negócio de fato faz."
        )

    # --- Sequencial ---
    with tab_s:
        mde = st.slider("Efeito mínimo relevante para o SPRT (%)", 1.0, 50.0, 10.0, 1.0,
                        key=f"sprt_mde_{key}")
        try:
            sprt = sprt_test(conv_a, n_a, conv_b, n_b, mde)
        except Exception as exc:
            st.error(f"Falha no SPRT: {exc}")
            return

        status_render = {"H1": st.success, "H0": st.error, "continuar": st.info}
        status_render[sprt["status"]](sprt["decisao"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Log-likelihood ratio", f"{sprt['llr']:.2f}")
        c2.metric("Limite superior (aceita H1)", f"{sprt['limite_superior']:.2f}")
        c3.metric("Limite inferior (aceita H0)", f"{sprt['limite_inferior']:.2f}")

        checkpoints, llr = sprt_trajectory(conv_a, n_a, conv_b, n_b, mde)
        fig = go.Figure()
        fig.add_scatter(x=checkpoints, y=llr, mode="lines", name="LLR acumulado",
                        line=dict(color=TEAL, width=3))
        fig.add_hline(y=sprt["limite_superior"], line_dash="dash", line_color="#1B7F3B",
                      annotation_text="aceita H1 (há efeito)")
        fig.add_hline(y=sprt["limite_inferior"], line_dash="dash", line_color="#B3261E",
                      annotation_text="aceita H0 (sem efeito)")
        fig.update_layout(title="Trajetória do teste sequencial",
                          xaxis_title="visitantes acumulados na variação", yaxis_title="LLR")
        st.plotly_chart(apply_theme(fig, height=400), width="stretch")
        st.caption(
            "A ordem dos eventos é reconstruída com semente fixa a partir dos totais observados "
            "(o dataset não tem log evento a evento), então a trajetória é **ilustrativa** — "
            "a decisão final usa o LLR do total acumulado."
        )

    repository.save_widget(
        key=f"abtest_{key}",
        origem="Teste A/B",
        canal_driver=f"{name_a} vs {name_b}",
        hipotese_default=f"{name_b} converte melhor que {name_a}.",
        resultado_default=freq["veredito"],
        insight_default=(
            f"{name_b}: {freq['variation']['taxa'] * 100:.2f}% vs {name_a}: "
            f"{freq['control']['taxa'] * 100:.2f}% — lift de {freq['lift_pct']:+.2f}% "
            f"(p={freq['p_value']:.4f}). P(B>A) bayesiana: {bayesian_test(conv_a, n_a, conv_b, n_b)['prob_b_maior_a'] * 100:.1f}%."
        ),
        proximo_passo_default=(
            "Escalar a variação vencedora e registrar o aprendizado."
            if freq["veredito"] == "Winner"
            else "Rodar com mais amostra ou reformular a hipótese."
        ),
        lift_pct=freq["lift_pct"],
        p_value=freq["p_value"],
    )


mode = st.radio("Modo de análise", ["📊 Dados reais", "✍️ Simulação manual"], horizontal=True)

# ---------------------------------------------------------------------------
# Modo 1 — dados reais
# ---------------------------------------------------------------------------
if mode == "📊 Dados reais":
    source = st.selectbox(
        "Fonte de dados",
        ["digital_marketing_campaign_dataset.csv (por cliente)",
         "marketing_campaign_dataset.csv (200k campanhas)"],
    )

    try:
        if source.startswith("digital"):
            df = load_digital()
            dimension = st.selectbox("Dimensão de comparação", ["CampaignChannel", "CampaignType", "Gender"])
            options = sorted(df[dimension].dropna().unique().tolist())
            c1, c2 = st.columns(2)
            group_a = c1.selectbox("Control (A)", options, index=0)
            group_b = c2.selectbox("Variation (B)", options, index=min(1, len(options) - 1))

            a = df[df[dimension] == group_a]
            b = df[df[dimension] == group_b]
            n_a, conv_a = int(len(a)), int(a["Conversion"].sum())
            n_b, conv_b = int(len(b)), int(b["Conversion"].sum())
        else:
            df = load_campaigns()
            segment = st.session_state.get("segmento", "Todos")
            if segment != "Todos":
                df = df[df["Customer_Segment"] == segment]
                st.caption(f"Filtrado pelo segmento global: **{segment}**")
            dimension = st.selectbox("Dimensão de comparação",
                                     ["Channel_Used", "Campaign_Type", "Target_Audience", "Location"])
            options = sorted(df[dimension].dropna().unique().tolist())
            c1, c2 = st.columns(2)
            group_a = c1.selectbox("Control (A)", options, index=0)
            group_b = c2.selectbox("Variation (B)", options, index=min(1, len(options) - 1))

            a = df[df[dimension] == group_a]
            b = df[df[dimension] == group_b]
            # Aqui a "amostra" são cliques e as conversões vêm de Clicks × Conversion_Rate.
            n_a, conv_a = int(a["Clicks"].sum()), int(a["Conversions"].sum())
            n_b, conv_b = int(b["Clicks"].sum()), int(b["Conversions"].sum())
            st.caption(
                "Neste dataset a unidade amostral é o **clique** e as conversões são derivadas de "
                "`Clicks × Conversion_Rate` — não é um experimento randomizado, é uma comparação "
                "observacional. Trate o resultado como indício, não como prova causal."
            )

        if group_a == group_b:
            st.warning("Escolha duas categorias diferentes para comparar.")
        elif n_a == 0 or n_b == 0:
            st.warning("Uma das categorias não tem volume suficiente.")
        else:
            st.divider()
            render_results(conv_a, n_a, conv_b, n_b, str(group_a), str(group_b), key="real")
    except Exception as exc:
        st.error(f"Falha ao preparar a comparação: {exc}")

# ---------------------------------------------------------------------------
# Modo 2 — simulação manual
# ---------------------------------------------------------------------------
else:
    st.caption("Entrada manual, como na calculadora de significância clássica.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Control (A)**")
        n_a = st.number_input("Visitantes A", 1, 100_000_000, 10000, 100, key="m_na")
        conv_a = st.number_input("Conversões A", 0, 100_000_000, 500, 10, key="m_ca")
    with c2:
        st.markdown("**Variation (B)**")
        n_b = st.number_input("Visitantes B", 1, 100_000_000, 10000, 100, key="m_nb")
        conv_b = st.number_input("Conversões B", 0, 100_000_000, 570, 10, key="m_cb")

    if conv_a > n_a or conv_b > n_b:
        st.error("O número de conversões não pode ser maior que o de visitantes.")
    else:
        st.divider()
        render_results(int(conv_a), int(n_a), int(conv_b), int(n_b), "Control", "Variation",
                       key="manual")

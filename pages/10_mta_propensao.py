"""Página 10 — MTA Preditivo: propensão à conversão (ML supervisionado)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_digital
from src.mta.journey_sim import build_journeys
from src.mta.markov import markov_attribution
from src.mta.propensity_model import (
    ALGORITHMS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SHAP_AVAILABLE,
    XGBOOST_AVAILABLE,
    confusion,
    curves,
    explain_model,
    opportunity_gaps,
    predicted_propensity_by_channel,
    score_profile,
    train_model,
)
from src.utils import repository
from src.utils.styling import BORDER, GOLD, NAVY, NEGATIVE, POSITIVE, TEAL, highlight, page_header
from src.viz.charts import DIVERGING, SEQUENTIAL, apply_theme

page_header(
    "MTA Preditivo — Propensão à Conversão",
    "A camada tática olhando para frente: em vez de repartir o crédito do que já aconteceu, prever "
    "quem tem chance de converter. É o modelo por trás de lead scoring e otimização de audiência.",
    layer="MTA",
)

st.info(
    "Enquanto as Páginas 4 e 5 **explicam o crédito histórico** por canal, esta prevê a "
    "**probabilidade individual de conversão** antes de acontecer. As duas leituras se completam: "
    "atribuição informa onde investir, propensão informa **em quem** investir.",
    icon="🎯",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Modelo de propensão")
    algorithm = st.selectbox("Algoritmo", ALGORITHMS, index=0)

    params: dict = {}
    if algorithm == "Regressão Logística":
        params["C"] = st.slider("C (inverso da regularização)", 0.01, 10.0, 1.0, 0.01)
    elif algorithm == "Random Forest":
        params["n_estimators"] = st.slider("Nº de árvores", 100, 800, 300, 50)
        params["max_depth"] = st.slider("Profundidade máxima", 2, 20, 10, 1)
        params["min_samples_leaf"] = st.slider("Mín. amostras por folha", 1, 20, 2, 1)
    else:
        params["n_estimators"] = st.slider("Nº de estimadores", 100, 800, 300, 50)
        params["max_depth"] = st.slider("Profundidade máxima", 2, 10, 4, 1)
        params["learning_rate"] = st.slider("Learning rate", 0.01, 0.3, 0.08, 0.01)
        if not XGBOOST_AVAILABLE:
            st.caption("ℹ️ `xgboost` não instalado — usando `GradientBoostingClassifier` do sklearn.")

    threshold = st.slider("Limiar de decisão", 0.05, 0.95, 0.50, 0.05,
                          help="Probabilidade a partir da qual o cliente é classificado como 'vai converter'.")
    if not SHAP_AVAILABLE:
        st.caption("ℹ️ `shap` não instalado — a explicabilidade usa a importância nativa do modelo.")

try:
    result = train_model(algorithm, tuple(sorted(params.items())), threshold)
except Exception as exc:
    st.error(f"Falha ao treinar o modelo: {exc}")
    st.stop()

if result.used_fallback:
    st.caption("Gradient Boosting via `sklearn` (fallback), com pesos de classe aplicados no fit.")

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
m = result.metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%")
c2.metric("Precision", f"{m['precision'] * 100:.1f}%")
c3.metric("Recall", f"{m['recall'] * 100:.1f}%")
c4.metric("F1", f"{m['f1']:.3f}")
c5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
c6.metric("PR-AUC", f"{m['pr_auc']:.3f}")

st.caption(
    f"⚠️ As classes são desbalanceadas: **{m['baseline_positivos'] * 100:.1f}%** da base converte. "
    "Com esse desequilíbrio a **accuracy engana** (um modelo que chuta sempre a classe majoritária já "
    f"acerta {max(m['baseline_positivos'], 1 - m['baseline_positivos']) * 100:.1f}%) e a **PR-AUC é a métrica mais informativa**, "
    f"porque compara com o baseline de {m['baseline_positivos']:.3f} da classe positiva. "
    "Usamos `class_weight=\"balanced\"` no treino."
)

tab_perf, tab_imp, tab_score, tab_gap = st.tabs(
    ["📈 Performance", "🔍 Importância e SHAP", "🧮 Score este perfil", "⚖️ Propensão × Atribuição"]
)

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
with tab_perf:
    c1, c2 = st.columns([2, 3])
    with c1:
        cm = confusion(result)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=SEQUENTIAL,
                        title="Matriz de confusão (conjunto de teste)")
        st.plotly_chart(apply_theme(fig, height=380, legend_bottom=False), width="stretch")
        st.caption(f"Limiar de decisão: {result.threshold:.2f}. "
                   "Baixe o limiar para capturar mais convertedores (↑ recall, ↓ precisão).")
    with c2:
        pts = curves(result)
        fpr, tpr = pts["roc"]
        recall, precision = pts["pr"]

        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={m['roc_auc']:.3f})",
                        line=dict(color=TEAL, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório",
                        line=dict(color=BORDER, dash="dash"))
        fig.add_scatter(x=recall, y=precision, mode="lines",
                        name=f"Precision-Recall (AUC={m['pr_auc']:.3f})",
                        line=dict(color=GOLD, width=3))
        fig.add_hline(y=m["baseline_positivos"], line_dash="dot", line_color=NAVY,
                      annotation_text="baseline PR")
        fig.update_layout(title="Curvas ROC e Precision-Recall sobrepostas",
                          xaxis_title="FPR (ROC) / Recall (PR)",
                          yaxis_title="TPR (ROC) / Precision (PR)")
        st.plotly_chart(apply_theme(fig, height=440), width="stretch")

    st.subheader("Distribuição das probabilidades previstas")
    dist = pd.DataFrame({"probabilidade": result.y_proba,
                         "real": np.where(result.y_test == 1, "Converteu", "Não converteu")})
    fig = px.histogram(dist, x="probabilidade", color="real", nbins=40, barmode="overlay",
                       opacity=0.75, color_discrete_map={"Converteu": TEAL, "Não converteu": NEGATIVE},
                       title="Separação entre as classes")
    fig.add_vline(x=result.threshold, line_dash="dash", line_color=NAVY,
                  annotation_text="limiar")
    st.plotly_chart(apply_theme(fig), width="stretch")
    st.caption("Quanto menos as duas distribuições se sobrepõem, melhor o modelo separa as classes.")

# ---------------------------------------------------------------------------
# Importância / SHAP
# ---------------------------------------------------------------------------
with tab_imp:
    explanation = explain_model(result)
    if explanation["message"]:
        st.warning(explanation["message"], icon="ℹ️")

    summary = explanation["summary"].head(20)
    fig = px.bar(summary.sort_values("importancia"), x="importancia", y="feature",
                 orientation="h", color_discrete_sequence=[TEAL],
                 title=("SHAP — impacto médio absoluto por feature"
                        if explanation["method"] == "shap"
                        else "Importância nativa do modelo"))
    st.plotly_chart(apply_theme(fig, height=560, legend_bottom=False), width="stretch")

    if explanation["method"] == "shap":
        st.subheader("Direção do efeito")
        directional = explanation["summary"].head(15).copy()
        fig = px.bar(directional.sort_values("efeito_medio"), x="efeito_medio", y="feature",
                     orientation="h", color="efeito_medio",
                     color_continuous_scale=DIVERGING,
                     title="Features que empurram a probabilidade para cima (→) e para baixo (←)")
        st.plotly_chart(apply_theme(fig, height=480, legend_bottom=False), width="stretch")

    if algorithm == "Regressão Logística" and "odds_ratio" in result.importance.columns:
        st.subheader("Odds ratio (interpretação direta)")
        odds = result.importance.dropna(subset=["odds_ratio"]).head(15).copy()
        odds["efeito"] = np.where(odds["odds_ratio"] > 1,
                                  "aumenta a chance de conversão", "reduz a chance de conversão")
        st.dataframe(odds.round(4), width="stretch", hide_index=True)
        st.caption(
            "Odds ratio > 1: cada desvio-padrão a mais na feature multiplica as chances de conversão "
            "por esse fator, mantendo o resto constante."
        )

# ---------------------------------------------------------------------------
# Simulador
# ---------------------------------------------------------------------------
with tab_score:
    st.subheader("Score este perfil")
    st.caption("Informe um perfil e o modelo treinado devolve a probabilidade de conversão em tempo real.")

    digital = load_digital()
    defaults = digital[NUMERIC_FEATURES].median()

    with st.form("score_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", sorted(digital["Gender"].dropna().unique()))
            channel = st.selectbox("CampaignChannel", sorted(digital["CampaignChannel"].dropna().unique()))
            campaign_type = st.selectbox("CampaignType", sorted(digital["CampaignType"].dropna().unique()))
            age = st.slider("Age", 18, 80, int(defaults["Age"]))
            income = st.number_input("Income", 0, 500_000, int(defaults["Income"]), 1000)
        with c2:
            ad_spend = st.number_input("AdSpend", 0.0, 50_000.0, float(defaults["AdSpend"]), 100.0)
            ctr = st.slider("ClickThroughRate", 0.0, 1.0, float(defaults["ClickThroughRate"]), 0.01)
            visits = st.slider("WebsiteVisits", 0, 60, int(defaults["WebsiteVisits"]))
            pages = st.slider("PagesPerVisit", 0.0, 12.0, float(defaults["PagesPerVisit"]), 0.1)
            time_on_site = st.slider("TimeOnSite", 0.0, 20.0, float(defaults["TimeOnSite"]), 0.1)
        with c3:
            shares = st.slider("SocialShares", 0, 100, int(defaults["SocialShares"]))
            opens = st.slider("EmailOpens", 0, 30, int(defaults["EmailOpens"]))
            clicks = st.slider("EmailClicks", 0, 20, int(defaults["EmailClicks"]))
            previous = st.slider("PreviousPurchases", 0, 20, int(defaults["PreviousPurchases"]))
            loyalty = st.slider("LoyaltyPoints", 0, 5000, int(defaults["LoyaltyPoints"]), 50)

        submitted = st.form_submit_button("🎯 Calcular propensão", type="primary")

    if submitted:
        profile = {
            "Gender": gender, "CampaignChannel": channel, "CampaignType": campaign_type,
            "Age": age, "Income": income, "AdSpend": ad_spend, "ClickThroughRate": ctr,
            "WebsiteVisits": visits, "PagesPerVisit": pages, "TimeOnSite": time_on_site,
            "SocialShares": shares, "EmailOpens": opens, "EmailClicks": clicks,
            "PreviousPurchases": previous, "LoyaltyPoints": loyalty,
        }
        try:
            proba, drivers = score_profile(result, profile)
        except Exception as exc:
            st.error(f"Não foi possível pontuar o perfil: {exc}")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                band = "Alta" if proba >= 0.8 else ("Média" if proba >= 0.5 else "Baixa")
                color = POSITIVE if proba >= 0.8 else (GOLD if proba >= 0.5 else NEGATIVE)
                highlight(f"{proba * 100:.1f}%",
                          f"propensão à conversão — faixa <b>{band}</b>", color=color)
            with c2:
                st.markdown("**Os 3 fatores que mais pesaram nesta decisão**")
                drivers_display = drivers.copy()
                drivers_display["direção"] = np.where(
                    drivers_display["contribuicao"] >= 0, "↑ aumenta", "↓ reduz")
                st.dataframe(drivers_display.round(4), width="stretch", hide_index=True)
                st.caption(
                    "Fatores via SHAP quando disponível; caso contrário, coeficiente × valor "
                    "padronizado (Regressão Logística) ou importância nativa (árvores)."
                )

# ---------------------------------------------------------------------------
# Propensão x Atribuição
# ---------------------------------------------------------------------------
with tab_gap:
    st.subheader("Propensão prevista × crédito atribuído")
    st.caption(
        "Duas leituras independentes do mesmo canal: o modelo preditivo diz **quem tende a converter**; "
        "Markov/Shapley dizem **quem levou o crédito**. Divergência grande é pista de investigação."
    )

    propensity = predicted_propensity_by_channel(result)

    attribution = st.session_state.get("mta_full_attribution")
    if attribution is not None and "Markov" in attribution.columns:
        share = attribution[["Markov", "Shapley"]].mean(axis=1)
        source_note = "Markov + Shapley (calculados na Página 5)"
    else:
        try:
            journeys = st.session_state.get("mta_journeys") or build_journeys(load_digital())
            share = markov_attribution(journeys)["removal_effect_norm"] * 100
            source_note = "Markov (calculado nesta página)"
        except Exception:
            share = pd.Series(dtype=float)
            source_note = "indisponível"

    gaps = opportunity_gaps(propensity, share)
    st.caption(f"Fonte do crédito atribuído: {source_note}.")

    long = gaps.melt(id_vars="canal", value_vars=["propensao_media", "credito_atribuido_%"],
                     var_name="métrica", value_name="valor")
    long["valor"] = np.where(long["métrica"] == "propensao_media",
                             long["valor"] * 100, long["valor"])
    long["métrica"] = long["métrica"].map({"propensao_media": "Propensão média prevista (%)",
                                           "credito_atribuido_%": "Crédito atribuído (%)"})
    fig = px.bar(long, x="canal", y="valor", color="métrica", barmode="group",
                 color_discrete_sequence=[TEAL, GOLD],
                 title="Canal: propensão prevista × crédito de atribuição")
    st.plotly_chart(apply_theme(fig, height=440), width="stretch")

    st.dataframe(
        gaps.round(3), width="stretch", hide_index=True,
        column_config={
            "propensao_media": st.column_config.NumberColumn("Propensão média", format="%.3f"),
            "conversao_real": st.column_config.NumberColumn("Conversão real", format="%.3f"),
            "credito_atribuido_%": st.column_config.NumberColumn("Crédito atribuído", format="%.1f%%"),
            "gap_rank": st.column_config.NumberColumn("Gap de ranking", format="%.0f"),
        },
    )

    opportunities = gaps[gaps["sinal"].str.startswith("🔎")]
    if len(opportunities):
        top = opportunities.iloc[0]
        st.warning(
            f"**Oportunidade de investigação:** `{top['canal']}` tem alta propensão prevista "
            f"({top['propensao_media'] * 100:.1f}%) mas recebe apenas {top['credito_atribuido_%']:.1f}% "
            "do crédito de atribuição. Ou o canal está subvalorizado pelo modelo de crédito, ou a "
            "audiência que ele atinge já converteria de qualquer forma. "
            "Um teste de incrementalidade resolve a dúvida.",
            icon="🔎",
        )
        default_channel, default_result = str(top["canal"]), "Oportunidade"
        default_insight = (
            f"{top['canal']} aparece com propensão média prevista de {top['propensao_media'] * 100:.1f}% "
            f"(modelo {algorithm}, PR-AUC {m['pr_auc']:.3f}) mas recebe {top['credito_atribuido_%']:.1f}% "
            "do crédito em Markov/Shapley — divergência de ranking a investigar."
        )
        default_next = f"Desenhar teste de incrementalidade em {top['canal']} para separar propensão de causalidade."
        default_stage = "Opportunity"
    else:
        st.success("Nenhuma divergência relevante: propensão prevista e crédito atribuído estão alinhados.")
        default_channel = str(gaps.iloc[0]["canal"])
        default_result, default_stage = "Insight", "Results"
        default_insight = (
            f"Modelo {algorithm} com PR-AUC {m['pr_auc']:.3f} e ROC-AUC {m['roc_auc']:.3f}. "
            "Ranking de propensão por canal alinhado com o crédito de Markov/Shapley."
        )
        default_next = "Usar os scores de propensão para priorizar audiências nas campanhas ativas."

    repository.save_widget(
        key="propensity",
        origem="MTA Preditivo",
        canal_driver=default_channel,
        hipotese_default=f"O modelo de propensão consegue priorizar audiências melhor que o baseline em {default_channel}.",
        resultado_default=default_result,
        insight_default=default_insight,
        proximo_passo_default=default_next,
        etapa_default=default_stage,
    )

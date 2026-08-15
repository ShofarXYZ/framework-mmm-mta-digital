"""Página 2 — MMM Modelagem: o motor estatístico."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    CONTROL_COLUMNS,
    DIGITAL_CHANNELS,
    MEDIA_CHANNELS,
    clean_mmm,
    label,
    load_mmm_raw,
    promo_dummy_columns,
)
from src.mmm.model import (
    PYMC_AVAILABLE,
    REGULARIZERS,
    MMMConfig,
    contribution_summary,
    due_to_analysis,
    fit_mmm,
    optimize_adstock,
    response_curve,
)
from src.mmm.transforms import FUNCTIONAL_FORMS, default_hill_params, hill_saturation
from src.utils import repository
from src.utils.styling import BORDER, GOLD, NAVY, NEGATIVE, POSITIVE, TEAL, fmt_money, page_header
from src.viz.charts import DIVERGING, apply_theme, stacked_area

page_header(
    "MMM Modelagem — o motor estatístico",
    "Adstock, saturação e regressão regularizada para decompor as vendas em Base + contribuição "
    "de cada canal. É aqui que o 'quanto cada mídia gerou' deixa de ser opinião.",
    layer="MMM",
)

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
if "mmm_clean" in st.session_state:
    df = st.session_state["mmm_clean"]
    st.caption("Usando o dataset tratado na página **MMM Explorer**.")
else:
    try:
        df = clean_mmm(load_mmm_raw(), {c: "Interpolação linear" for c in MEDIA_CHANNELS})
        st.warning(
            "Você ainda não passou pelo **MMM Explorer**. Aplicamos interpolação linear como padrão — "
            "recomendado voltar e escolher o tratamento canal a canal.",
            icon="⚠️",
        )
    except Exception as exc:
        st.error(f"Não foi possível preparar o dataset: {exc}")
        st.stop()

# ---------------------------------------------------------------------------
# Sidebar de configuração
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuração do modelo")

    form = st.selectbox("Forma funcional", FUNCTIONAL_FORMS, index=0,
                        help="Linear: efeito aditivo. Log-Linear: retornos decrescentes no y. "
                             "Log-Log: coeficientes lidos como elasticidades.")
    regularizer = st.selectbox("Regularização", REGULARIZERS, index=0)
    alpha = st.slider("Alpha (força da regularização)", 0.01, 20.0, 1.0, 0.01)
    l1_ratio = st.slider("L1 ratio (ElasticNet)", 0.0, 1.0, 0.5, 0.05,
                         disabled=(regularizer != "ElasticNet"))

    st.subheader("Adstock (carryover)")
    auto_adstock = st.checkbox("Otimizar decay automaticamente (grid search)", value=False,
                               help="Busca coordenada minimizando o MAPE de holdout. Mais lento.")
    decays: dict[str, float] = {}
    for channel in MEDIA_CHANNELS:
        default = 0.4 if channel in ("tv_spend", "newspaper_spend") else 0.2
        decays[channel] = st.slider(f"{label(channel)}", 0.0, 0.9, default, 0.05,
                                    key=f"decay_{channel}", disabled=auto_adstock)

    st.subheader("Saturação (Hill)")
    saturation_on = st.toggle("Aplicar curva de saturação", value=True)
    hill_defaults = default_hill_params(df, MEDIA_CHANNELS)
    half_sat_mult = st.slider("Half-saturation (× mediana do canal)", 0.2, 3.0, 1.0, 0.1,
                              disabled=not saturation_on)
    slope = st.slider("Slope (s) — >1 gera curva em S", 0.3, 3.0, 1.0, 0.1,
                      disabled=not saturation_on)
    hill_params = {c: (v[0] * half_sat_mult, slope) for c, v in hill_defaults.items()}

    st.subheader("Validação")
    holdout_weeks = st.slider("Semanas no holdout temporal", 8, 12, 10)

    bayesian = st.toggle("Modo Bayesiano (PyMC)", value=False, disabled=not PYMC_AVAILABLE)
    if not PYMC_AVAILABLE:
        st.caption("ℹ️ `pymc` não instalado — o app roda no modo frequentista. "
                   "Instale com `pip install pymc arviz` para habilitar.")

    run = st.button("▶️ Rodar modelo", type="primary", width="stretch")

control_columns = CONTROL_COLUMNS + promo_dummy_columns(df)

if saturation_on:
    with st.sidebar.expander("Preview da curva de saturação"):
        x = np.linspace(0, 3, 100)
        y = hill_saturation(x, 1.0, slope)
        fig = go.Figure(go.Scatter(x=x, y=y, line=dict(color=GOLD, width=3)))
        fig.add_vline(x=1.0, line_dash="dot", line_color=NAVY,
                      annotation_text="half-saturation")
        fig.update_layout(title="Hill normalizada", xaxis_title="investimento / k",
                          yaxis_title="efeito")
        st.plotly_chart(apply_theme(fig, height=240, legend_bottom=False), width="stretch")

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
if run:
    with st.spinner("Ajustando o MMM..."):
        try:
            cfg = MMMConfig(
                form=form, regularizer=regularizer, alpha=alpha, l1_ratio=l1_ratio,
                saturation_on=saturation_on, holdout_weeks=holdout_weeks, decays=decays,
                hill_params=hill_params, media_columns=MEDIA_CHANNELS,
                control_columns=control_columns, bayesian=bayesian,
            )
            if auto_adstock:
                cfg.decays = optimize_adstock(df, cfg)
                st.toast("Adstock otimizado por grid search.")
            st.session_state["mmm_result"] = fit_mmm(df, cfg)
            st.success("Modelo ajustado. Resultado disponível também no Otimizador e no Geo-Holdout.")
        except Exception as exc:
            st.error(f"Falha ao ajustar o modelo: {exc}")

result = st.session_state.get("mmm_result")
if result is None:
    st.info("Configure o modelo na barra lateral e clique em **▶️ Rodar modelo**.", icon="👈")
    st.stop()

if result.config.bayesian and result.posterior is None and PYMC_AVAILABLE:
    st.warning("A amostragem Bayesiana falhou — os resultados abaixo são do modo frequentista.")

# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
m = result.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("R² (treino)", f"{m['r2_treino']:.3f}")
c2.metric("R² ajustado", f"{m['r2_ajustado_treino']:.3f}")
c3.metric("MAPE (holdout)", f"{m['mape_holdout']:.1f}%", f"{int(m['holdout_weeks'])} semanas")
c4.metric("MAE (holdout)", fmt_money(m["mae_holdout"], ""))

tab_fit, tab_coef, tab_decomp, tab_curves, tab_dueto = st.tabs(
    ["📉 Ajuste", "🔢 Coeficientes e VIF", "🧱 Decomposição", "📐 Curvas de resposta", "🔄 Due-to"]
)

# --- Ajuste ---------------------------------------------------------------
with tab_fit:
    split = len(result.data) - int(m["holdout_weeks"])
    fig = go.Figure()
    fig.add_scatter(x=result.data["date"], y=result.data["sales"], name="Real",
                    line=dict(color=NAVY, width=2.2))
    fig.add_scatter(x=result.data["date"], y=result.fitted, name="Previsto",
                    line=dict(color=GOLD, width=2, dash="dash"))
    fig.add_vrect(x0=result.data["date"].iloc[split], x1=result.data["date"].iloc[-1],
                  fillcolor=TEAL, opacity=0.08, line_width=0,
                  annotation_text="holdout temporal", annotation_position="top left")
    fig.update_layout(title="Real × Previsto")
    st.plotly_chart(apply_theme(fig, height=440), width="stretch")

    residuals = result.data["sales"].to_numpy() - result.fitted.to_numpy()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=result.fitted, y=residuals, title="Resíduos × Previsto",
                         labels={"x": "previsto", "y": "resíduo"},
                         color_discrete_sequence=[TEAL], trendline="lowess")
        fig.add_hline(y=0, line_dash="dot", line_color=NAVY)
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")
    with c2:
        fig = px.histogram(x=residuals, nbins=25, title="Distribuição dos resíduos",
                           color_discrete_sequence=[NAVY])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")

# --- Coeficientes ---------------------------------------------------------
with tab_coef:
    st.subheader("Coeficientes")
    coefs = result.coefficients.to_frame("coeficiente").reset_index(names="variavel")
    coefs["sinal"] = np.where(coefs["coeficiente"] >= 0, "positivo ↑", "negativo ↓")
    coefs["magnitude"] = coefs["coeficiente"].abs()

    def interpret(row) -> str:
        v, name = row["coeficiente"], row["variavel"]
        if name in MEDIA_CHANNELS:
            if v > 0:
                return "Investimento associado a AUMENTO de vendas — coerente com o esperado."
            return "Sinal negativo: revise adstock/saturação, colinearidade ou período de investimento."
        if name == "competitor_spend":
            return "Pressão da concorrência. Sinal negativo é o esperado."
        if name == "holiday":
            return "Efeito de feriado sobre a demanda."
        return "Variável de controle (promoção/sazonalidade)."

    coefs["interpretação"] = coefs.apply(interpret, axis=1)
    coefs["variavel"] = coefs["variavel"].map(lambda c: label(c) if c in MEDIA_CHANNELS else c)

    fig = px.bar(coefs.sort_values("coeficiente"), x="coeficiente", y="variavel",
                 orientation="h", title="Coeficientes padronizados",
                 color="coeficiente", color_continuous_scale=DIVERGING)
    st.plotly_chart(apply_theme(fig, height=460, legend_bottom=False), width="stretch")
    st.dataframe(coefs.round(4), width="stretch", hide_index=True)

    if result.posterior is not None:
        st.subheader("Incerteza dos coeficientes (HDI 94% — modo Bayesiano)")
        try:
            import arviz as az

            summary = az.summary(result.posterior, var_names=["beta"], hdi_prob=0.94)
            summary.index = result.feature_names[: len(summary)]
            st.dataframe(summary, width="stretch")
        except Exception as exc:
            st.warning(f"Não foi possível montar o sumário posterior: {exc}")

    st.subheader("VIF — multicolinearidade")
    vif = result.vif.copy()
    st.caption("VIF > 5 sugere colinearidade relevante; > 10 é crítico. "
               "Canais investidos sempre juntos inflam o VIF — é o caso clássico do MMM.")
    st.dataframe(
        vif.style.map(
            lambda v: "background-color:rgba(229,99,106,0.22)" if isinstance(v, (int, float)) and v > 10
            else ("background-color:rgba(232,163,61,0.20)" if isinstance(v, (int, float)) and v > 5 else ""),
            subset=["VIF"],
        ).format({"VIF": "{:.2f}"}),
        width="stretch", hide_index=True,
    )

# --- Decomposição ---------------------------------------------------------
with tab_decomp:
    st.subheader("Contribuição de cada driver ao longo do tempo")
    contrib = result.contributions
    if contrib.attrs.get("base_negativa"):
        st.warning(
            "A **Base** ficou negativa em parte do período — sinal clássico de má especificação: "
            "com todos os canais sempre ativos, o modelo extrapola mal o cenário 'mídia zero'. "
            "Zeramos a Base nessas semanas e rateamos o excedente entre os canais. "
            "Para corrigir de verdade, tente a forma **Log-Log**, aumente o alpha ou inclua mais controles.",
            icon="⚠️",
        )
    columns = ["Base"] + list(result.config.media_columns)
    st.plotly_chart(stacked_area(contrib, "date", columns, "Decomposição das vendas previstas"),
                    width="stretch")

    summary = contribution_summary(result)
    c1, c2 = st.columns([3, 2])
    with c1:
        display = summary.copy()
        display["canal"] = display["canal"].map(lambda c: label(c) if c != "Base" else "Base")
        st.dataframe(
            display.round(2), width="stretch", hide_index=True,
            column_config={
                "contribuicao": st.column_config.NumberColumn("Contribuição", format="%.0f"),
                "% do previsto": st.column_config.ProgressColumn(
                    "% do total", min_value=0, max_value=100, format="%.1f%%"),
                "investimento": st.column_config.NumberColumn("Investimento", format="%.0f"),
                "ROI (sales/R$)": st.column_config.NumberColumn("ROI (venda por R$)", format="%.2f"),
            },
        )
    with c2:
        pie = summary[summary["contribuicao"] > 0]
        fig = px.pie(pie, values="contribuicao", names="canal", hole=0.5,
                     title="Share de contribuição")
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")

    st.info(
        "**MMM não substitui MTA.** Os canais digitais abaixo aparecem aqui com o peso "
        "estratégico de longo prazo — mas a otimização tática (criativo, público, lance) "
        f"acontece nas páginas de MTA: {', '.join(label(c) for c in DIGITAL_CHANNELS)}.",
        icon="🔬",
    )

# --- Curvas de resposta ---------------------------------------------------
with tab_curves:
    st.subheader("Curvas de resposta — onde cada canal satura")
    selected = st.multiselect("Canais", MEDIA_CHANNELS, default=MEDIA_CHANNELS[:4],
                              format_func=label)
    max_mult = st.slider("Simular até quantas vezes o investimento atual", 1.5, 4.0, 3.0, 0.5)

    if selected:
        with st.spinner("Calculando curvas..."):
            fig = go.Figure()
            marginal_rows = []
            for i, channel in enumerate(selected):
                curve = response_curve(result, channel, max_mult, points=20)
                fig.add_scatter(x=curve["investimento"], y=curve["contribuicao_incremental"],
                                name=label(channel), mode="lines", line=dict(width=2.6))
                current = curve[curve["multiplicador"].between(0.9, 1.1)]
                marginal_rows.append({
                    "canal": label(channel),
                    "retorno marginal no nível atual": float(current["retorno_marginal"].mean())
                    if len(current) else np.nan,
                    "retorno marginal em 2x": float(
                        curve.loc[(curve["multiplicador"] - 2).abs().idxmin(), "retorno_marginal"]),
                })
            fig.update_layout(title="Investimento total × contribuição incremental",
                              xaxis_title="investimento acumulado no período",
                              yaxis_title="vendas incrementais")
            st.plotly_chart(apply_theme(fig, height=460), width="stretch")

            marginal = pd.DataFrame(marginal_rows)
            st.dataframe(marginal.round(3), width="stretch", hide_index=True)
            st.caption(
                "Quando o retorno marginal em 2x é bem menor que o atual, o canal já está **saturando**: "
                "cada real adicional rende menos. Esse é exatamente o insumo do Otimizador de Budget."
            )

# --- Due-to --------------------------------------------------------------
with tab_dueto:
    st.subheader("Due-to analysis — o que mudou entre períodos")
    dueto = due_to_analysis(result)
    if dueto.empty:
        st.warning("Série curta demais para comparar dois períodos.")
    else:
        dueto_display = dueto.copy()
        dueto_display["driver"] = dueto_display["driver"].map(
            lambda c: label(c) if c != "Base" else "Base")
        fig = go.Figure(go.Waterfall(
            x=dueto_display["driver"], y=dueto_display["delta"], orientation="v",
            connector=dict(line=dict(color=BORDER)),
            increasing=dict(marker=dict(color=POSITIVE)),
            decreasing=dict(marker=dict(color=NEGATIVE)),
        ))
        fig.update_layout(title="Δ de contribuição por driver (período atual × anterior)")
        st.plotly_chart(apply_theme(fig, height=420, legend_bottom=False), width="stretch")
        st.dataframe(dueto_display.round(1), width="stretch", hide_index=True)
        st.caption(
            "Leitura de due-to: a variação de vendas entre os dois períodos é explicada pela soma "
            "das variações de cada driver — inclusive da Base (demanda orgânica)."
        )

# ---------------------------------------------------------------------------
# Learning Repository
# ---------------------------------------------------------------------------
summary = contribution_summary(result)
top = summary[summary["canal"] != "Base"].iloc[0] if len(summary) > 1 else None
if top is not None:
    repository.save_widget(
        key="mmm_model",
        origem="MMM",
        canal_driver=label(top["canal"]),
        hipotese_default=f"{label(top['canal'])} é o canal de maior contribuição incremental no período.",
        resultado_default="Insight",
        insight_default=(
            f"{label(top['canal'])} responde por {top['% do previsto']:.1f}% das vendas previstas, "
            f"com ROI de {top['ROI (sales/R$)']:.2f} venda por R$ investido. "
            f"Modelo {result.config.form} / {result.config.regularizer}, "
            f"MAPE de holdout {m['mape_holdout']:.1f}%."
        ),
        proximo_passo_default="Validar a incrementalidade com um geo-holdout antes de realocar budget.",
        etapa_default="Opportunity",
    )

"""Página 11 — MMM: jornada guiada (Descritivo → Diagnóstico → Preditivo → Prescritivo).

Escrita para público leigo: cada etapa explica o que está sendo visto, por que
aquilo importa e o que fazer com a informação. O jargão aparece sempre traduzido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import DIGITAL_CHANNELS, MEDIA_CHANNELS, OFFLINE_CHANNELS, label
from src.insights import marginal_returns, mmm_recommendation
from src.mmm.model import contribution_summary
from src.mta.attribution import (
    ALGORITHMIC_MODELS,
    DDA,
    MODEL_HELP,
    MODEL_KIND,
    attribution_shares,
    dda_signal,
    model_cpa,
    model_credit,
    model_selector,
    model_spread,
)
from src.scenarios import (
    ALLOCATION_STRATEGIES,
    BUDGET_MAX,
    BUDGET_MIN,
    MMM_HORIZONS,
    get_mmm_result,
    mmm_scenario,
)
from src.utils import repository
from src.utils.styling import (
    GOLD,
    NAVY,
    NEGATIVE,
    POSITIVE,
    TEAL,
    fmt_money,
    highlight,
    money_columns,
    page_header,
    plain_box,
    recommendation_panel,
    spacer,
    stage_header,
    tokens,
)
from src.viz.charts import apply_theme, stacked_area

page_header(
    "MMM na prática — a jornada completa em 4 passos",
    "A visão de quem planeja o ano: quanto cada mídia devolve em vendas e onde colocar a verba "
    "do próximo trimestre, semestre ou ano.",
    layer="MMM",
)

# ---------------------------------------------------------------------------
# Abertura — o mapa da jornada
# ---------------------------------------------------------------------------
st.markdown(
    "Toda análise séria percorre a mesma escada. Cada degrau responde a uma pergunta diferente, "
    "e só faz sentido subir na ordem — **prescrever sem diagnosticar é chute com planilha**."
)

ladder = st.columns(4)
LADDER = [
    ("1️⃣ Descritivo", "O que está acontecendo?", "Os fatos. Quanto foi investido, quanto foi vendido.", TEAL),
    ("2️⃣ Diagnóstico", "Por que aconteceu?", "As causas. Qual mídia puxou a venda e qual só acompanhou.", NAVY),
    ("3️⃣ Preditivo", "O que vai acontecer?", "O futuro simulado. Se eu investir R$ X, vendo quanto?", GOLD),
    ("4️⃣ Prescritivo", "O que devo fazer?", "A decisão. Onde colocar e de onde tirar a verba.", POSITIVE),
]
for col, (step, question, desc, color) in zip(ladder, LADDER):
    with col:
        st.markdown(
            f"<div class='mmm-card' style='border-top-color:{color}'>"
            f"<h4>{step}</h4><p><b>{question}</b><br><br>{desc}</p></div>",
            unsafe_allow_html=True,
        )
st.write("")

st.info(
    "**Onde estamos no framework:** o MMM é a camada **estratégica**. Ele enxerga o ano inteiro e "
    "todas as mídias juntas — inclusive TV e jornal, que não têm clique para rastrear. "
    "Por isso as decisões aqui são de calendário: planejamento anual, revisão de semestre, "
    "trimestre, bimestre. Para decisões de hoje e desta semana, use a página **MTA na prática**.",
    icon="🏛️",
)

try:
    result, is_default = get_mmm_result()
except Exception as exc:
    st.error(f"Não foi possível preparar o modelo: {exc}")
    st.stop()

if is_default:
    st.caption(
        "ℹ️ Usando um modelo com configuração padrão — você não precisa ajustar nada para navegar "
        "por esta página. Quem quiser controlar adstock, saturação e regularização pode fazer isso "
        "em **🧪 MMM Modelagem**, e esta página passa a usar aquele modelo."
    )

data = result.data
contrib = result.contributions
summary = contribution_summary(result)
total_sales_hist = float(data["sales"].sum())
total_spend_hist = float(data[MEDIA_CHANNELS].sum().sum())
weeks = len(data)

# ---------------------------------------------------------------------------
# Régua de atribuição — a leitura tática da fatia digital
# ---------------------------------------------------------------------------
st.markdown("#### ⚖️ Escolha a régua de atribuição do digital")
c1, c2 = st.columns([2, 3])
with c1:
    attr_model = model_selector(c1, key="mmm_journey_model")
with c2:
    st.markdown(
        f"<div class='mmm-card' style='border-top-color:{GOLD}'>"
        f"<h4>{attr_model} · {MODEL_KIND[attr_model]}</h4><p>{MODEL_HELP[attr_model]}</p></div>",
        unsafe_allow_html=True,
    )
    if attr_model == DDA:
        try:
            auc = dda_signal()
            if auc == auc:
                if auc < 0.6:
                    st.caption(
                        f"🔎 Poder de separação do modelo: **AUC {auc:.2f}**. Como 0,50 é o "
                        "puro acaso, a jornada aqui explica pouco da conversão — e o Data-Driven "
                        "fica mais próximo do Linear. Isso é honestidade do algoritmo, não defeito: "
                        "sem sinal no dado, nenhum modelo consegue eleger um canal decisivo. "
                        "Com clickstream real, essa separação costuma ser bem maior."
                    )
                else:
                    st.caption(
                        f"🔎 Poder de separação do modelo: **AUC {auc:.2f}** — a composição da "
                        "jornada explica de forma relevante quem converte."
                    )
        except Exception:
            pass

try:
    attr_shares = attribution_shares()
    attr_share = attr_shares[attr_model]
    attr_cpa = model_cpa(attr_model)
    attr_ok = True
except Exception as exc:  # a página de MMM não pode quebrar por causa do MTA
    attr_shares = attr_share = attr_cpa = None
    attr_ok = False
    st.caption(f"Leitura tática indisponível: {exc}")

st.info(
    "**Divisão de trabalho entre as duas camadas.** O MMM decide o tamanho do bolo — quanto vai "
    "para TV, jornal e para o digital como um todo. A régua de atribuição não muda esse total: "
    "ela orienta como a fatia digital se reparte entre os canais da jornada. As duas coisas "
    "aparecem juntas nas quatro etapas abaixo, e é exatamente assim que o framework funciona: "
    "**estratégico define o quanto, tático define o onde.**",
    icon="⚖️",
)
spacer(8)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1️⃣ Descritivo — o que aconteceu",
        "2️⃣ Diagnóstico — por que aconteceu",
        "3️⃣ Preditivo — o que vai acontecer",
        "4️⃣ Prescritivo — o que fazer",
    ]
)

# ===========================================================================
# 1. DESCRITIVO
# ===========================================================================
with tab1:
    stage_header(
        1, "Descritivo", "O que está acontecendo?",
        "Antes de explicar qualquer coisa, é preciso concordar sobre os fatos. Este passo não "
        "interpreta nada: mostra o que foi gasto, o que foi vendido e quando. É o extrato da conta.",
        TEAL,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Período analisado", f"{weeks} semanas",
              f"{data['date'].min():%m/%Y} a {data['date'].max():%m/%Y}")
    c2.metric("Total investido em mídia", fmt_money(total_spend_hist))
    c3.metric("Total vendido", fmt_money(total_sales_hist, ""))
    c4.metric("Vendas por real investido", f"{total_sales_hist / total_spend_hist:.2f}",
              help="Divisão simples: total vendido ÷ total investido. Ainda NÃO é o retorno da mídia.")

    plain_box(
        "Cuidado com esse último número",
        f"“{total_sales_hist / total_spend_hist:.2f} de venda por real” é só uma divisão. "
        "Boa parte dessas vendas aconteceria mesmo sem propaganda nenhuma — gente que já é cliente, "
        "quem passou na loja, quem buscou a marca pelo nome. Separar o que a mídia realmente "
        "trouxe do que teria acontecido de qualquer jeito é exatamente o trabalho do próximo passo.",
        "⚠️",
    )

    st.subheader("As vendas ao longo do tempo")
    fig = go.Figure()
    fig.add_scatter(x=data["date"], y=data["sales"], name="Vendas na semana", mode="lines",
                    line=dict(color=NAVY, width=2.2))
    fig.add_scatter(x=data["date"], y=data["sales"].rolling(8, min_periods=1).mean(),
                    name="Tendência (média de 8 semanas)", mode="lines",
                    line=dict(color=GOLD, width=2.4, dash="dash"))
    fig.update_layout(title="Vendas semanais e a linha de tendência")
    st.plotly_chart(apply_theme(fig, height=400), width="stretch")

    first_half = float(data["sales"].iloc[: weeks // 2].mean())
    second_half = float(data["sales"].iloc[weeks // 2 :].mean())
    trend = 100 * (second_half - first_half) / first_half if first_half else 0
    plain_box(
        "Como ler este gráfico",
        f"A linha fina é o que aconteceu semana a semana — ela sobe e desce muito, e isso é normal. "
        f"A linha tracejada é a tendência, que ignora o ruído. Comparando a primeira metade do "
        f"período com a segunda, as vendas médias "
        f"{'subiram' if trend >= 0 else 'caíram'} **{abs(trend):.1f}%**. "
        "Esse é o fato. O porquê vem no passo 2.",
        "📖",
    )

    st.subheader("Para onde foi o dinheiro")
    spend = data[MEDIA_CHANNELS].sum().sort_values(ascending=False).reset_index()
    spend.columns = ["canal", "investimento"]
    spend["Tipo de mídia"] = np.where(
        spend["canal"].isin(OFFLINE_CHANNELS), "Offline (TV, jornal)", "Digital (internet)"
    )
    spend["canal"] = spend["canal"].map(label)
    spend["share"] = 100 * spend["investimento"] / spend["investimento"].sum()

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.bar(spend, x="canal", y="investimento", color="Tipo de mídia",
                     color_discrete_map={"Offline (TV, jornal)": NAVY, "Digital (internet)": TEAL},
                     title="Investimento por mídia no período")
        st.plotly_chart(apply_theme(fig), width="stretch")
    with c2:
        st.dataframe(
            money_columns(spend[["canal", "investimento", "share"]], ["investimento"]), width="stretch", hide_index=True,
            column_config={
                "canal": "Mídia",
                "investimento": st.column_config.TextColumn("Investido"),
                "share": st.column_config.ProgressColumn("Fatia da verba", min_value=0,
                                                         max_value=100, format="%.1f%%"),
            },
        )

    if attr_ok:
        st.subheader(f"Dentro do digital: como a régua {attr_model} reparte o crédito")
        tactical = pd.DataFrame({
            "canal": attr_share.index,
            "credito_%": attr_share.to_numpy(),
        }).sort_values("credito_%", ascending=False)
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.bar(tactical.sort_values("credito_%"), x="credito_%", y="canal",
                         orientation="h", color_discrete_sequence=[TEAL],
                         title=f"Crédito das conversões digitais segundo {attr_model}")
            st.plotly_chart(apply_theme(fig, height=330, legend_bottom=False), width="stretch")
        with c2:
            st.dataframe(
                tactical, width="stretch", hide_index=True,
                column_config={
                    "canal": "Canal da jornada",
                    "credito_%": st.column_config.ProgressColumn("Crédito", min_value=0,
                                                                 max_value=100, format="%.1f%%"),
                },
            )
        plain_box(
            "Duas listas de canais, dois mundos",
            "Repare que os nomes aqui (Social Media, Email, PPC, SEO, Referral) são diferentes dos "
            "de cima (TV, Instagram, Google Ads...). Não é inconsistência: o MMM enxerga **compra de "
            "mídia** — onde a verba foi gasta — e o MTA enxerga **pontos de contato da jornada** — "
            "por onde a pessoa passou. O MMM decide quanto o digital merece no total; esta régua "
            "orienta como distribuir esse valor por dentro.",
            "🧭",
        )
        spacer(6)

    digital_share = 100 * float(data[DIGITAL_CHANNELS].sum().sum()) / total_spend_hist
    plain_box(
        "O que isso já revela",
        f"**{digital_share:.0f}%** da verba foi para mídia digital e **{100 - digital_share:.0f}%** "
        "para offline. Repare que TV e jornal aparecem aqui lado a lado com Instagram e Google Ads. "
        "Nenhuma ferramenta de clique consegue fazer isso — o Google Analytics não sabe que sua TV "
        "existe. É por isso que o MMM é a única camada capaz de comparar a verba inteira.",
        "🏛️",
    )

# ===========================================================================
# 2. DIAGNÓSTICO
# ===========================================================================
with tab2:
    stage_header(
        2, "Diagnóstico", "Por que isso aconteceu?",
        "Agora saímos dos fatos e entramos nas causas. A pergunta muda de “quanto vendemos?” para "
        "“quanto das vendas cada mídia realmente causou?”. O modelo separa o bolo em fatias: "
        "a parte que teria acontecido de qualquer jeito e a parte que cada mídia trouxe.",
        NAVY,
    )

    base_share = float(summary.loc[summary["canal"] == "Base", "% do previsto"].iloc[0]) \
        if "Base" in summary["canal"].values else 0.0
    media_share = 100 - base_share

    c1, c2 = st.columns(2)
    c1.metric("Vendas que viriam de qualquer forma (Base)", f"{base_share:.0f}%",
              help="Força da marca, clientes recorrentes, sazonalidade. Não depende da campanha.")
    c2.metric("Vendas causadas pela mídia", f"{media_share:.0f}%",
              help="A parte que o investimento em propaganda trouxe a mais.")

    plain_box(
        "A pergunta que separa amador de profissional",
        "Não é “quanto vendi?”, é “**quanto eu não teria vendido se não tivesse anunciado?**”. "
        f"Segundo o modelo, {base_share:.0f}% das vendas viriam mesmo sem mídia nenhuma. "
        f"Os outros {media_share:.0f}% são o que a propaganda efetivamente construiu — "
        "e é só sobre essa fatia que faz sentido discutir retorno.",
        "🎯",
    )

    st.subheader("O bolo das vendas, fatiado")
    st.plotly_chart(
        stacked_area(contrib, "date", ["Base"] + list(result.config.media_columns),
                     "Cada faixa colorida é a contribuição de uma mídia"),
        width="stretch",
    )
    plain_box(
        "Como ler",
        "A altura total é a venda prevista. A faixa de baixo é a Base. Cada faixa acima é o quanto "
        "aquela mídia acrescentou naquela semana. Quando uma faixa engorda, aquela mídia estava "
        "trabalhando; quando some, ela parou de investir ou parou de funcionar.",
        "📖",
    )

    st.subheader("Quem realmente puxou a venda")
    media_summary = summary[summary["canal"] != "Base"].copy()
    media_summary["canal_label"] = media_summary["canal"].map(label)
    fig = px.bar(
        media_summary.sort_values("ROI (sales/R$)"), x="ROI (sales/R$)", y="canal_label",
        orientation="h", color="ROI (sales/R$)", color_continuous_scale=[NEGATIVE, GOLD, POSITIVE],
        title="Retorno de cada mídia: vendas geradas por real investido",
    )
    st.plotly_chart(apply_theme(fig, height=420, legend_bottom=False), width="stretch")

    best_roi = media_summary.sort_values("ROI (sales/R$)", ascending=False).iloc[0]
    worst_roi = media_summary.sort_values("ROI (sales/R$)").iloc[0]
    st.dataframe(
        money_columns(media_summary[["canal_label", "investimento", "contribuicao", "% do previsto", "ROI (sales/R$)"]], ["investimento", "contribuicao"]),
        width="stretch", hide_index=True,
        column_config={
            "canal_label": "Mídia",
            "investimento": st.column_config.TextColumn("Investido"),
            "contribuicao": st.column_config.TextColumn("Vendas que gerou"),
            "% do previsto": st.column_config.ProgressColumn("Fatia das vendas", min_value=0,
                                                            max_value=100, format="%.1f%%"),
            "ROI (sales/R$)": st.column_config.NumberColumn("Retorno por R$", format="%.2f"),
        },
    )

    plain_box(
        "Traduzindo a tabela",
        f"Cada real investido em **{label(best_roi['canal'])}** devolveu "
        f"**{best_roi['ROI (sales/R$)']:.2f}** em vendas — o melhor da casa. "
        f"Já **{label(worst_roi['canal'])}** devolveu apenas "
        f"**{worst_roi['ROI (sales/R$)']:.2f}**. "
        "Atenção a uma armadilha: mídia com retorno alto **não** significa “joga tudo lá”. "
        "Toda mídia satura — chega um ponto em que o público já viu o anúncio vezes demais e o "
        "real seguinte rende menos. Quanto cabe em cada uma é o que o próximo passo calcula.",
        "🔍",
    )

    if attr_ok:
        st.subheader("Cruzando as duas camadas")
        digital_contrib = float(
            sum(contrib[c].sum() for c in DIGITAL_CHANNELS if c in contrib.columns)
        )
        total_pred = float(contrib["previsto"].sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Digital na contribuição total (MMM)",
                  f"{100 * digital_contrib / total_pred:.1f}%",
                  help="Fatia das vendas que o MMM atribui aos canais digitais.")
        c2.metric("Canal digital nº 1 pela régua", str(attr_share.idxmax()),
                  f"{attr_share.max():.1f}% do crédito")
        spread_tbl = model_spread()
        c3.metric("Canal mais dependente da régua", str(spread_tbl.iloc[0]["canal"]),
                  f"varia {spread_tbl.iloc[0]['variacao_pp']:.1f} p.p.")

        plain_box(
            "Por que as duas leituras precisam conversar",
            f"O MMM diz que o digital responde por **{100 * digital_contrib / total_pred:.1f}%** das "
            f"vendas previstas. A régua **{attr_model}** diz que, dentro do digital, quem mais "
            f"contribui é **{attr_share.idxmax()}**. Se você só olhasse o MMM, saberia o tamanho do "
            "digital mas não onde mexer; se só olhasse a atribuição, otimizaria o digital sem saber "
            "se ele merece mais ou menos verba que a TV. As duas juntas fecham a decisão.",
            "🔗",
        )
        spacer(6)

    st.subheader("Fatores fora do seu controle")
    c1, c2 = st.columns(2)
    with c1:
        by_holiday = data.groupby("holiday")["sales"].mean().reset_index()
        by_holiday["holiday"] = by_holiday["holiday"].map({0: "Semana normal", 1: "Semana com feriado"})
        fig = px.bar(by_holiday, x="holiday", y="sales", title="Efeito dos feriados nas vendas",
                     color_discrete_sequence=[NAVY])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")
    with c2:
        by_promo = data.groupby("sales_promotion")["sales"].mean().reset_index()
        fig = px.bar(by_promo.sort_values("sales", ascending=False), x="sales_promotion", y="sales",
                     title="Efeito das promoções nas vendas", color_discrete_sequence=[TEAL])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")

    plain_box(
        "Por que isso está aqui",
        "Se uma semana vendeu muito porque teve feriado e promoção, seria injusto dar o crédito para "
        "a campanha de TV que rodou no mesmo período. O modelo desconta esses efeitos antes de "
        "calcular o retorno das mídias — é isso que impede a análise de confundir **coincidência** "
        "com **causa**.",
        "⚖️",
    )

# ===========================================================================
# 3. PREDITIVO
# ===========================================================================
with tab3:
    stage_header(
        3, "Preditivo", "O que vai acontecer se eu investir?",
        "Aqui você brinca com o futuro. Escolha um valor de verba, o período de planejamento e "
        "em quais mídias colocar — o modelo responde quanto isso deve gerar em vendas. "
        "Ele já sabe que mídia satura, então não vai prometer que o dobro do dinheiro traz o dobro "
        "do resultado.",
        GOLD,
    )

    st.markdown("#### 🎛️ Monte o seu cenário")
    c1, c2 = st.columns([3, 2])
    with c1:
        budget = st.slider(
            "Quanto você tem para investir? (R$)",
            min_value=BUDGET_MIN, max_value=BUDGET_MAX, value=250_000, step=1_000,
            format="R$ %d",
            help="Verba ADICIONAL, além do que já é investido hoje.",
        )
        budget = float(st.number_input(
            "Ou digite o valor exato (R$)", min_value=float(BUDGET_MIN),
            max_value=float(BUDGET_MAX), value=float(budget), step=1_000.0,
        ))
    with c2:
        horizon_name = st.selectbox(
            "Para qual período?", list(MMM_HORIZONS), index=1,
            help="O MMM é uma ferramenta de calendário: pensa em bimestre, trimestre, semestre e ano.",
        )
        horizon_weeks = MMM_HORIZONS[horizon_name]
        strategy = st.selectbox("Como dividir a verba entre as mídias?", ALLOCATION_STRATEGIES)

    selected = st.multiselect(
        "Em quais mídias você quer investir?",
        MEDIA_CHANNELS, default=MEDIA_CHANNELS[:4], format_func=label,
    )

    with st.expander("⚙️ Ajuste avançado (opcional)"):
        max_share = st.slider(
            "Teto de concentração: no máximo quanto da verba pode ir para uma única mídia?",
            20, 100, 50, 5, format="%d%%",
            help="Regra de prudência, não do modelo. Com verbas pequenas diante do investimento "
                 "histórico, a saturação mal aparece e a conta puramente ótima manda 100% numa mídia "
                 "só — o que deixa o resultado refém de um leilão e de um formato.",
        ) / 100

    st.caption(
        f"💡 Você está simulando **{fmt_money(budget)}** distribuídos ao longo de "
        f"**{horizon_weeks} semanas** ({horizon_name.split(' (')[0].lower()}), o que dá "
        f"**{fmt_money(budget / horizon_weeks)} por semana** de investimento adicional."
    )

    if not selected:
        st.warning("Escolha ao menos uma mídia para simular.")
    elif st.button("🔮 Simular este cenário", type="primary"):
        with st.spinner("Rodando o cenário no modelo..."):
            st.session_state["mmm_journey_scenario"] = mmm_scenario(
                result, budget, selected, horizon_weeks, strategy, max_share
            )

    scenario = st.session_state.get("mmm_journey_scenario")
    if scenario and scenario["ok"]:
        st.divider()
        st.markdown("#### 📊 O que o modelo prevê")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Você investe", fmt_money(scenario["budget"]))
        c2.metric("Vendas adicionais previstas", fmt_money(scenario["vendas_incrementais"], ""))
        c3.metric("Retorno por real", f"{scenario['retorno_por_real']:.2f}")
        c4.metric("Resultado líquido", fmt_money(scenario["lucro_estimado"], ""),
                  help="Vendas adicionais menos o que você investiu.")

        payback = scenario["retorno_por_real"]
        if payback >= 1:
            highlight(
                f"{payback:.2f}x",
                f"Para cada R$ 1,00 investido, o modelo prevê <b>{fmt_money(payback, 'R$ ')}</b> "
                "em vendas adicionais neste período.",
                color=POSITIVE,
            )
        else:
            highlight(
                f"{payback:.2f}x",
                "Cada R$ 1,00 investido volta como menos de R$ 1,00 em vendas neste cenário — "
                "reveja as mídias escolhidas ou reduza o valor.",
                color=NEGATIVE,
            )
        st.write("")

        table = scenario["table"]
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = px.bar(
                table.sort_values("investimento"), x="investimento", y="canal_label",
                orientation="h", color="retorno_por_real",
                color_continuous_scale=[NEGATIVE, GOLD, POSITIVE],
                title="Quanto vai para cada mídia (cor = retorno por real)",
            )
            st.plotly_chart(apply_theme(fig, height=400, legend_bottom=False), width="stretch")
        with c2:
            st.dataframe(
                money_columns(table[["canal_label", "investimento", "share_%", "vendas_incrementais", "retorno_por_real"]],
                              ["investimento", "vendas_incrementais"]),
                width="stretch", hide_index=True,
                column_config={
                    "canal_label": "Mídia",
                    "investimento": st.column_config.TextColumn("Recebe"),
                    "share_%": st.column_config.ProgressColumn("Fatia", min_value=0, max_value=100,
                                                               format="%.0f%%"),
                    "vendas_incrementais": st.column_config.TextColumn("Gera em vendas"),
                    "retorno_por_real": st.column_config.NumberColumn("Por R$", format="%.2f"),
                },
            )

        plain_box(
            "Por que a divisão ficou assim",
            {
                "Deixar o modelo decidir (recomendado)":
                    "O modelo dividiu a verba em fatias e, a cada fatia, perguntou: "
                    "“qual mídia rende mais com este próximo pedaço?”. Conforme uma mídia vai "
                    "enchendo e saturando, a fatia seguinte migra para outra. É por isso que a "
                    "divisão não é igual nem proporcional — ela segue o retorno na margem. "
                    f"O teto de concentração de {scenario.get('max_share', 0.5) * 100:.0f}% "
                    "impede que tudo caia numa mídia só.",
                "Seguir o investimento atual":
                    "A verba foi dividida na mesma proporção do que já é investido hoje. "
                    "É o cenário “mais do mesmo”: seguro, mas não corrige nenhum desequilíbrio.",
                "Dividir igualmente":
                    "Cada mídia recebeu a mesma quantia. É o cenário mais simples de explicar, "
                    "mas ignora que as mídias têm retornos muito diferentes.",
            }[scenario["strategy"]],
            "🧠",
        )

        if attr_ok:
            digital_budget = float(
                sum(v for k, v in scenario["allocation"].items() if k in DIGITAL_CHANNELS)
            )
            if digital_budget > 0:
                st.markdown("#### 🔬 Detalhamento tático da fatia digital")
                st.caption(
                    f"Do total simulado, **{fmt_money(digital_budget)}** foram para mídias digitais. "
                    f"A régua **{attr_model}** sugere como distribuir esse valor entre os canais da "
                    "jornada — e o custo por conversão de cada um estima o retorno em conversões."
                )
                split = pd.DataFrame({
                    "canal": attr_share.index,
                    "credito_%": attr_share.to_numpy(),
                })
                split["investimento"] = digital_budget * split["credito_%"] / 100
                cpa_map = attr_cpa.set_index("canal")["cpa"]
                split["cpa"] = cpa_map.reindex(split["canal"]).to_numpy()
                split["conversoes_estimadas"] = split.apply(
                    lambda r: r["investimento"] / r["cpa"] if r["cpa"] and r["cpa"] == r["cpa"] and r["cpa"] > 0 else float("nan"),
                    axis=1,
                )
                split = split.sort_values("investimento", ascending=False)

                c1, c2 = st.columns([3, 2])
                with c1:
                    fig = px.bar(split.sort_values("investimento"), x="investimento", y="canal",
                                 orientation="h", color_discrete_sequence=[GOLD],
                                 title=f"Fatia digital repartida pela régua {attr_model}")
                    st.plotly_chart(apply_theme(fig, height=340, legend_bottom=False),
                                    width="stretch")
                with c2:
                    st.dataframe(
                        money_columns(split, ["investimento", "cpa"]),
                        width="stretch", hide_index=True,
                        column_config={
                            "canal": "Canal",
                            "credito_%": st.column_config.NumberColumn("Crédito", format="%.1f%%"),
                            "investimento": st.column_config.TextColumn("Recebe"),
                            "cpa": st.column_config.TextColumn("Custo/conversão"),
                            "conversoes_estimadas": st.column_config.NumberColumn("Conversões", format="%.0f"),
                        },
                    )

                total_conv = float(split["conversoes_estimadas"].sum(skipna=True))
                sem_credito = split[split["credito_%"] <= 0.01]["canal"].tolist()
                plain_box(
                    "Como ler este detalhamento",
                    f"Os mesmos {fmt_money(digital_budget)} rendem cerca de "
                    f"**{total_conv:,.0f} conversões** sob a régua **{attr_model}**. "
                    "Troque a régua no topo da página e este quadro inteiro muda — mesma verba, "
                    "mesma operação, leitura diferente. É por isso que a escolha da régua precisa "
                    "ser combinada com a diretoria ANTES de a campanha rodar, não depois."
                    .replace(",", ".")
                    + (f"<br><br>⚠️ Com esta régua, {' e '.join(sem_credito)} não recebe crédito "
                       "nenhum — e por isso ficaria sem verba nesta divisão. É um bom exemplo de "
                       "como a régua errada apaga um canal do mapa." if sem_credito else ""),
                    "⚖️",
                )
                spacer(8)

        st.markdown("#### 📈 E se a verba fosse outra?")
        with st.spinner("Testando outros valores de verba..."):
            grid = [budget * m for m in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)]
            rows = []
            for value in grid:
                if value > BUDGET_MAX * 3:
                    continue
                sim = mmm_scenario(result, value, selected, horizon_weeks, "Seguir o investimento atual")
                if sim["ok"]:
                    rows.append({
                        "verba": value,
                        "vendas_adicionais": sim["vendas_incrementais"],
                        "retorno_por_real": sim["retorno_por_real"],
                    })
        curve = pd.DataFrame(rows)
        if len(curve):
            fig = go.Figure()
            fig.add_scatter(x=curve["verba"], y=curve["vendas_adicionais"], mode="lines+markers",
                            name="Vendas adicionais", line=dict(color=TEAL, width=3))
            fig.add_scatter(x=curve["verba"], y=curve["verba"], mode="lines",
                            name="Linha do empate (R$ 1 investido = R$ 1 vendido)",
                            line=dict(color=tokens().border, dash="dash"))
            fig.add_vline(x=budget, line_dash="dot", line_color=GOLD,
                          annotation_text="seu cenário")
            fig.update_layout(title="Quanto mais eu invisto, quanto mais eu vendo?",
                              xaxis_title="verba investida (R$)", yaxis_title="vendas adicionais")
            st.plotly_chart(apply_theme(fig, height=420), width="stretch")
            plain_box(
                "A lição mais importante desta página",
                "Repare que a curva **cresce cada vez menos**. Isso é a saturação: as primeiras "
                "pessoas impactadas são as mais fáceis de convencer; depois, você passa a pagar "
                "caro para falar de novo com quem já viu o anúncio. Enquanto a curva azul estiver "
                "acima da linha tracejada, investir ainda compensa. Quando ela cruza para baixo, "
                "você está pagando mais do que recebe.",
                "📉",
            )

        repository.save_widget(
            key="mmm_journey",
            origem="MMM",
            canal_driver=str(scenario["melhor_canal"]),
            hipotese_default=f"Investir {fmt_money(budget)} em {horizon_name.split(' (')[0].lower()} "
                             f"gera vendas incrementais com retorno de {payback:.2f}x.",
            resultado_default="Oportunidade",
            insight_default=(
                f"Cenário de {fmt_money(budget)} em {len(selected)} mídias, horizonte de "
                f"{horizon_weeks} semanas, estratégia '{scenario['strategy']}': "
                f"{fmt_money(scenario['vendas_incrementais'], '')} em vendas adicionais "
                f"({payback:.2f} por real)."
            ),
            proximo_passo_default="Validar com um geo-holdout antes de comprometer a verba.",
            etapa_default="Priorization",
        )
    elif scenario and not scenario["ok"]:
        st.warning(scenario["message"])
    else:
        st.info("Monte o cenário acima e clique em **🔮 Simular este cenário**.", icon="👆")

# ===========================================================================
# 4. PRESCRITIVO
# ===========================================================================
with tab4:
    stage_header(
        4, "Prescritivo", "O que eu devo fazer?",
        "O último degrau. Aqui não há mais gráfico para interpretar: há uma decisão a tomar. "
        "O modelo aponta onde colocar o próximo real, de onde tirar, e — igualmente importante — "
        "o que fazer para ter certeza de que ele está certo.",
        POSITIVE,
    )

    with st.spinner("Calculando a recomendação..."):
        rec = mmm_recommendation(result)

    if not rec["ok"]:
        st.warning(f"Recomendação indisponível: {rec.get('message')}")
    else:
        recommendation_panel(
            rec["headline"], rec["detail"] + ".", rec["invest"], rec["watch"],
            invest_note=(
                f"Cada real a mais aqui devolve <b>{rec['invest_marginal']:.2f}</b> em vendas — "
                f"é a melhor porta de entrada da verba. Sugestão: <b>+{fmt_money(rec['amount'])}</b>."
            ),
            watch_note=(
                f"Cada real a mais aqui devolve apenas <b>{rec['watch_marginal']:.2f}</b>. "
                f"Sugestão: <b>economizar {fmt_money(rec['amount'])}</b> e levar para a mídia ao lado."
            ),
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Verba a remanejar", fmt_money(rec["amount"]))
        c2.metric("Vendas adicionais esperadas", fmt_money(rec["delta_sales"], ""),
                  f"{rec['lift_pct']:+.2f}%")
        c3.metric("Custo do movimento", "R$ 0",
                  help="Não é dinheiro novo: é o mesmo orçamento, apenas em outro lugar.")

        plain_box(
            "O que exatamente fazer na segunda-feira",
            f"1. Reduzir <b>{fmt_money(rec['amount'])}</b> do investimento em "
            f"<b>{rec['watch']}</b> no próximo ciclo de planejamento.<br>"
            f"2. Levar esse mesmo valor para <b>{rec['invest']}</b>.<br>"
            f"3. Não mexer no resto ainda — mudança grande de uma vez impede saber o que causou o quê.<br>"
            f"4. Medir o resultado com um teste real (próximo bloco) antes de repetir a dose.",
            "✅",
        )

        st.subheader("O ranking completo, em ordem de prioridade")
        ranking = rec["table"][["canal_label", "investimento", "roi_medio", "retorno_marginal", "saturacao"]]
        st.dataframe(
            money_columns(ranking, ["investimento"]), width="stretch", hide_index=True,
            column_config={
                "canal_label": "Mídia",
                "investimento": st.column_config.TextColumn("Investido hoje"),
                "roi_medio": st.column_config.NumberColumn("Retorno médio", format="%.2f"),
                "retorno_marginal": st.column_config.NumberColumn("Retorno do PRÓXIMO real", format="%.2f"),
                "saturacao": st.column_config.ProgressColumn("Quanto já saturou", min_value=0.0,
                                                             max_value=1.0, format="%.0f%%"),
            },
        )
        plain_box(
            "As duas colunas de retorno são diferentes — e a segunda é a que importa",
            "**Retorno médio** olha para trás: é tudo que a mídia já gerou dividido por tudo que já "
            "foi gasto. **Retorno do próximo real** olha para frente: é o que você ganha ao "
            "adicionar mais um real agora. Uma mídia pode ter média excelente e estar completamente "
            "saturada — ou seja, ótima no passado, péssima como destino do próximo investimento. "
            "Decisão se toma pela segunda coluna.",
            "🎓",
        )
        if rec["saturated"]:
            st.warning(
                "**Mídias já saturando** (perdem mais de 60% do retorno se você dobrar a verba): "
                + ", ".join(rec["saturated"])
                + ". Colocar mais dinheiro nelas é jogar contra a matemática.",
                icon="📉",
            )

    if attr_ok:
        st.subheader("E dentro do digital, por onde começar")
        best = attr_cpa.dropna(subset=["cpa"]).head(1)
        worst = attr_cpa.dropna(subset=["cpa"]).tail(1)
        if len(best) and len(worst):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"<div class='mmm-card' style='border-top-color:{POSITIVE}'>"
                    f"<h4 style='color:{POSITIVE}'>➕ Melhor custo por conversão: {best.iloc[0]['canal']}</h4>"
                    f"<p>Traz conversão a <b>{fmt_money(best.iloc[0]['cpa'])}</b> segundo a régua "
                    f"{attr_model}, com {best.iloc[0]['credito_%']:.1f}% do crédito. "
                    "É por onde a fatia digital deve começar.</p></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='mmm-card' style='border-top-color:{NEGATIVE}'>"
                    f"<h4 style='color:{NEGATIVE}'>⚠️ Custo mais alto: {worst.iloc[0]['canal']}</h4>"
                    f"<p>Cada conversão sai por <b>{fmt_money(worst.iloc[0]['cpa'])}</b>. "
                    "Antes de cortar, confira se ele não é o canal que apresenta a marca — "
                    "custo alto no topo do funil costuma ser preço de descoberta, não desperdício."
                    "</p></div>",
                    unsafe_allow_html=True,
                )
            spacer(14)
            plain_box(
                "A ordem correta da decisão",
                f"1. O <b>MMM</b> define quanto do orçamento vai para o digital como um todo — "
                f"essa é a decisão de {fmt_money(rec['amount']) if rec.get('ok') else 'realocação'} "
                "que aparece acima.<br>"
                f"2. A <b>régua {attr_model}</b> orienta como repartir esse valor entre os canais "
                "da jornada.<br>"
                "3. O <b>teste de incrementalidade</b> confirma se a leitura estava certa.<br><br>"
                "Inverter essa ordem é o erro mais comum do mercado: otimizar o digital no detalhe "
                "enquanto o tamanho do bolo digital está errado.",
                "🧭",
            )
            spacer(6)

    st.subheader("Antes de executar: prove que o modelo está certo")
    plain_box(
        "O passo que quase todo mundo pula",
        "Tudo nesta página vem de um modelo estatístico — ele encontra <b>padrões</b>, e padrão não "
        "é prova. A única forma de ter certeza é fazer um teste real: cortar de propósito o "
        "investimento de uma mídia por algumas semanas e ver se a venda cai como o modelo previu. "
        "Se cair, o modelo está calibrado e você pode confiar. Se não cair, o modelo estava "
        "superestimando aquela mídia — e você acabou de economizar muito dinheiro.<br><br>"
        "Isso está pronto na página <b>🌍 Geo-Holdout × MMM</b>.",
        "🔬",
        NAVY,
    )

    st.markdown(
        "**Resumo da jornada:** você viu o que aconteceu (1), entendeu por que aconteceu (2), "
        "simulou o que vai acontecer (3) e recebeu o que fazer (4). O ciclo fecha registrando a "
        "decisão no **🔁 Learning Repository** — assim, no próximo planejamento, ninguém começa do zero."
    )

"""Página 12 — MTA: jornada guiada (Descritivo → Diagnóstico → Preditivo → Prescritivo).

Mesma escada didática da página de MMM, mas no horizonte tático: mês, semana, dia.
Escrita para público leigo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_digital
from src.insights import mta_recommendation
from src.mta.heuristics import attribute_all, to_share
from src.mta.journey_sim import adspend_by_channel, build_journeys, journey_stats, top_paths
from src.mta.markov import markov_attribution
from src.scenarios import (
    BUDGET_MAX,
    BUDGET_MIN,
    DEFAULT_HISTORY_DAYS,
    MTA_ALLOCATION_MODES,
    MTA_HORIZONS,
    mta_scenario,
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
    page_header,
    plain_box,
    recommendation_panel,
    stage_header,
    tokens,
)
from src.viz.charts import apply_theme, grouped_bar, sankey

page_header(
    "MTA na prática — a jornada completa em 4 passos",
    "A visão de quem opera a mídia digital no dia a dia: qual canal merece o crédito da conversão "
    "e onde colocar a verba de hoje, desta semana ou deste mês.",
    layer="MTA",
)

st.markdown(
    "A mesma escada da página de MMM, agora no **curto prazo**. Enquanto o MMM decide o ano, "
    "o MTA decide **hoje** — qual campanha pausar, qual público reforçar, onde colocar os "
    "próximos mil reais."
)

ladder = st.columns(4)
LADDER = [
    ("1️⃣ Descritivo", "O que está acontecendo?", "Os fatos: quem converteu e por quais canais passou.", TEAL),
    ("2️⃣ Diagnóstico", "Por que aconteceu?", "As causas: qual canal realmente empurrou a conversão.", NAVY),
    ("3️⃣ Preditivo", "O que vai acontecer?", "O futuro: com R$ X neste mês, quantas conversões espero?", GOLD),
    ("4️⃣ Prescritivo", "O que devo fazer?", "A decisão: onde colocar a verba desta semana.", POSITIVE),
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
    "**Onde estamos no framework:** o MTA é a camada **tática**. Ele só enxerga o digital "
    "rastreável — nada de TV ou jornal — mas enxerga com detalhe de pessoa e de clique. "
    "Por isso o horizonte aqui é curto: **mês, semana, dia**. Para decidir a verba do ano inteiro, "
    "incluindo mídia offline, use a página **MMM na prática**.",
    icon="🔬",
)

try:
    digital = load_digital()
    journeys = st.session_state.get("mta_journeys")
    if journeys is None:
        journeys = build_journeys(digital)
        st.session_state["mta_journeys"] = journeys
except Exception as exc:
    st.error(f"Não foi possível preparar os dados: {exc}")
    st.stop()

stats = journey_stats(journeys)
adspend = adspend_by_channel(journeys)
heuristics = to_share(attribute_all(journeys))
channels = list(heuristics.index)

with st.expander("⚠️ Antes de começar: de onde vêm estes dados (importante)"):
    st.markdown(
        """
Estes dados **não são um rastreamento real de cliques**. A base traz uma linha por cliente, com
um canal e o resultado final (converteu ou não) — sem a ordem em que a pessoa passou pelos canais.

Modelos de atribuição precisam dessa ordem para existir. Então o app **reconstrói uma jornada
provável** para cada cliente, usando os sinais de engajamento que a base realmente tem: se a pessoa
abriu e-mail, o e-mail entra na jornada; se compartilhou nas redes, o social entra; se já era
cliente, entra um contato anterior de relacionamento.

A reconstrução é **determinística** (dois cálculos dão exatamente o mesmo resultado) e está aberta
em `src/mta/journey_sim.py` e na página **🔀 Modelos de Atribuição**. O resultado final — quem
converteu — é dado real, nada foi inventado ali.

**Na prática:** os números desta página servem para você entender o raciocínio e comparar canais
entre si. Numa operação real, essa jornada viria do rastreamento verdadeiro e as contas seriam as mesmas.
        """
    )

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
        "Os fatos da operação digital: quantas pessoas foram impactadas, quantas converteram e "
        "por quantos canais elas passaram antes de decidir. Sem interpretação ainda.",
        TEAL,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pessoas impactadas", f"{stats['clientes']:,}".replace(",", "."))
    c2.metric("Converteram", f"{stats['conversoes']:,}".replace(",", "."),
              f"{stats['taxa_conversao'] * 100:.1f}% do total")
    c3.metric("Canais por pessoa", f"{stats['touchpoints_medio']:.1f}",
              help="Quantos contatos diferentes uma pessoa teve com a marca antes de converter.")
    c4.metric("Investimento digital", fmt_money(float(adspend.sum())))

    plain_box(
        "O número que muda tudo",
        f"Em média, cada pessoa passou por **{stats['touchpoints_medio']:.1f} canais diferentes** "
        f"antes de converter, e **{stats['jornadas_multitouch_%']:.0f}%** das jornadas tiveram mais "
        "de um contato. Isso significa que quase ninguém vê um anúncio e compra na hora. "
        "A pessoa descobre a marca num lugar, pesquisa em outro, recebe um e-mail, e só então "
        "converte. **Guardar essa frase é entender o resto da página.**",
        "🔑",
    )

    st.subheader("Por onde as pessoas passam")
    paths = top_paths(journeys, 10)
    fig = px.bar(
        paths.sort_values("clientes"), x="clientes", y="path_str", orientation="h",
        color="taxa_conversao", color_continuous_scale=[NEGATIVE, GOLD, POSITIVE],
        title="Os caminhos mais comuns (cor = taxa de conversão do caminho)",
        labels={"path_str": "caminho", "clientes": "pessoas"},
    )
    st.plotly_chart(apply_theme(fig, height=460, legend_bottom=False), width="stretch")
    plain_box(
        "Como ler",
        "Cada barra é uma sequência de canais, na ordem em que a pessoa passou. A seta → significa "
        "“depois foi para”. Barras mais longas são caminhos mais comuns; a cor mostra quais desses "
        "caminhos convertem melhor. Note que os caminhos campeões quase nunca têm um canal só.",
        "📖",
    )

    c1, c2 = st.columns(2)
    with c1:
        volume = journeys["CampaignChannel"].value_counts().reset_index()
        volume.columns = ["canal", "pessoas"]
        fig = px.bar(volume, x="canal", y="pessoas", title="Pessoas alcançadas por canal",
                     color_discrete_sequence=[TEAL])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")
    with c2:
        spend_df = adspend.reset_index()
        spend_df.columns = ["canal", "investimento"]
        fig = px.bar(spend_df, x="canal", y="investimento", title="Investimento por canal",
                     color_discrete_sequence=[GOLD])
        st.plotly_chart(apply_theme(fig, legend_bottom=False), width="stretch")

# ===========================================================================
# 2. DIAGNÓSTICO
# ===========================================================================
with tab2:
    stage_header(
        2, "Diagnóstico", "Por que isso aconteceu?",
        "Se a pessoa passou por 4 canais antes de comprar, qual deles merece o crédito da venda? "
        "Essa é a pergunta de um bilhão de reais da mídia digital — e a resposta muda "
        "completamente conforme a régua que você usa.",
        NAVY,
    )

    plain_box(
        "A armadilha do último clique",
        "A maioria das ferramentas dá 100% do crédito para o **último** canal antes da conversão. "
        "É como dar todo o mérito do gol para quem fez o passe final, ignorando quem roubou a bola "
        "e atravessou o campo. O resultado prático é grave: os canais que apresentam a marca para "
        "quem ainda não conhece aparecem mal no relatório e acabam tendo a verba cortada — "
        "matando justamente o começo da jornada.",
        "⚽",
    )

    st.subheader("O mesmo resultado, cinco réguas diferentes")
    long = heuristics.reset_index().melt(id_vars="canal", var_name="régua", value_name="crédito %")
    st.plotly_chart(
        grouped_bar(long, x="canal", y="crédito %", color="régua",
                    title="Quanto cada canal recebe de crédito conforme a régua escolhida"),
        width="stretch",
    )

    with st.expander("O que cada régua faz, em uma linha"):
        st.markdown(
            """
| Régua | Como reparte o crédito | Quando faz sentido |
|---|---|---|
| **First-Click** | 100% para quem apresentou a marca | Descobrir o que gera descoberta |
| **Last-Click** | 100% para o último contato | Padrão das ferramentas — e o mais distorcido |
| **Linear** | Divide igualmente entre todos | Quando todos os contatos parecem igualmente importantes |
| **Time-Decay** | Quem estava mais perto da conversão leva mais | Ciclos de compra curtos, promoções |
| **Position-Based** | 40% para o primeiro, 40% para o último, 20% para o meio | Equilibrar descoberta e fechamento |
            """
        )

    st.subheader("A régua que aprende sozinha")
    with st.spinner("Calculando o crédito algorítmico..."):
        try:
            markov = markov_attribution(journeys, channels)
            markov_share = (markov["removal_effect_norm"] * 100).rename("Markov")
        except Exception as exc:
            markov_share = pd.Series(dtype=float)
            st.warning(f"Não foi possível calcular o modelo algorítmico: {exc}")

    if len(markov_share):
        full = heuristics.join(markov_share, how="left").fillna(0.0)
        plain_box(
            "Como o modelo algorítmico decide",
            "Em vez de usar uma regra fixa, ele faz uma pergunta muito mais inteligente: "
            "**“se este canal desaparecesse, quantas conversões deixariam de acontecer?”** "
            "O canal que, ao sumir, derruba mais conversões é o mais valioso — não importa se ele "
            "aparecia no começo ou no fim da jornada. É assim que funciona o modelo data-driven "
            "do Google Analytics, e é o mais próximo de uma medida de causa que a atribuição alcança.",
            "🧠",
        )

        comparison = pd.DataFrame({
            "canal": full.index,
            "Pelo último clique (%)": full["Last-Click"].to_numpy(),
            "Pelo modelo algorítmico (%)": full["Markov"].to_numpy(),
        })
        comparison["Diferença (p.p.)"] = (
            comparison["Pelo modelo algorítmico (%)"] - comparison["Pelo último clique (%)"]
        )
        comparison = comparison.sort_values("Diferença (p.p.)", ascending=False)

        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            fig.add_bar(x=comparison["canal"], y=comparison["Pelo último clique (%)"],
                        name="Último clique", marker_color=NAVY)
            fig.add_bar(x=comparison["canal"], y=comparison["Pelo modelo algorítmico (%)"],
                        name="Modelo algorítmico", marker_color=GOLD)
            fig.update_layout(barmode="group", title="O quanto o último clique erra",
                              bargap=0.28, yaxis_title="crédito (%)")
            st.plotly_chart(apply_theme(fig, height=420), width="stretch")
        with c2:
            st.dataframe(
                comparison, width="stretch", hide_index=True,
                column_config={
                    "canal": "Canal",
                    "Pelo último clique (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Pelo modelo algorítmico (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Diferença (p.p.)": st.column_config.NumberColumn(format="%+.1f"),
                },
            )

        under = comparison.iloc[0]
        over = comparison.iloc[-1]
        plain_box(
            "O que essa diferença significa em dinheiro",
            f"**{under['canal']}** vale {under['Diferença (p.p.)']:.1f} pontos percentuais a mais do "
            f"que o relatório padrão mostra — ele está sendo <b>subestimado</b> e é candidato a corte "
            f"indevido de verba. Já **{over['canal']}** aparece "
            f"{abs(over['Diferença (p.p.)']):.1f} pontos <b>acima</b> do que a jornada justifica: "
            "provavelmente ele está apenas colhendo conversões que outros canais construíram.",
            "💰",
        )

    st.subheader("O fluxo da jornada")
    try:
        from src.mta.markov import sankey_data

        labels, source, target, value = sankey_data(journeys, 60)
        if source:
            st.plotly_chart(
                sankey(labels, source, target, value, "Por onde as pessoas entram, passam e saem"),
                width="stretch",
            )
            plain_box(
                "Como ler",
                "Cada faixa é um fluxo de pessoas. A largura mostra o volume. `(start)` é o começo "
                "da jornada, `(conversion)` é quem comprou e `(null)` é quem foi embora sem comprar. "
                "Seguir as faixas mais grossas mostra o caminho típico do seu cliente.",
                "📖",
            )
    except Exception as exc:
        st.caption(f"Fluxo indisponível: {exc}")

# ===========================================================================
# 3. PREDITIVO
# ===========================================================================
with tab3:
    stage_header(
        3, "Preditivo", "O que vai acontecer se eu investir?",
        "Escolha quanto tem para gastar, em quanto tempo e em quais canais — a simulação estima "
        "quantas conversões isso deve trazer, usando o custo por conversão real de cada canal.",
        GOLD,
    )

    st.markdown("#### 🎛️ Monte o seu cenário")
    c1, c2 = st.columns([3, 2])
    with c1:
        budget = st.slider(
            "Quanto você tem para investir? (R$)",
            min_value=BUDGET_MIN, max_value=BUDGET_MAX, value=50_000, step=1_000, format="R$ %d",
        )
        budget = float(st.number_input(
            "Ou digite o valor exato (R$)", min_value=float(BUDGET_MIN),
            max_value=float(BUDGET_MAX), value=float(budget), step=1_000.0,
        ))
    with c2:
        horizon_name = st.selectbox(
            "Para qual período?", list(MTA_HORIZONS), index=2,
            help="O MTA é ferramenta de operação: o gestor mexe no lance hoje, na campanha esta "
                 "semana, no plano deste mês.",
        )
        horizon_days = MTA_HORIZONS[horizon_name]
        mode = st.selectbox("Como dividir a verba?", MTA_ALLOCATION_MODES)

    selected = st.multiselect(
        "Em quais canais você quer investir?", channels, default=channels,
    )

    with st.expander("⚙️ Ajuste avançado (opcional)"):
        history_days = st.slider(
            "A quantos dias de operação o histórico equivale?", 7, 90, DEFAULT_HISTORY_DAYS, 1,
            help="A base de MTA não tem datas. Assumimos que ela representa um mês de operação — "
                 "é isso que permite falar em 'por dia' e 'por semana'. Ajuste se souber o período real.",
        )

    st.caption(
        f"💡 Você está simulando **{fmt_money(budget)}** para **{horizon_name.split(' (')[0].lower()}**"
        + (f", o que dá **{fmt_money(budget / horizon_days)} por dia**." if horizon_days > 1 else ".")
    )

    if not selected:
        st.warning("Escolha ao menos um canal para simular.")
    else:
        credit = markov_share if len(markov_share) else heuristics["Linear"]
        # o crédito precisa estar em CONVERSÕES, não em %
        conversions = credit / credit.sum() * float(journeys["converted"].sum())
        scenario = mta_scenario(conversions, adspend, budget, selected, horizon_days,
                                history_days, mode)

        if not scenario["ok"]:
            st.warning(scenario["message"])
        else:
            st.divider()
            st.markdown("#### 📊 O que a simulação prevê")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Você investe a mais", fmt_money(scenario["budget"]))
            c2.metric("Conversões a mais", f"{scenario['conversoes_estimadas']:,.0f}".replace(",", "."),
                      f"{scenario['conversoes_por_dia']:.1f} por dia")
            c3.metric("Custo por conversão", fmt_money(scenario["cpa_medio"]))
            c4.metric("Operação atual no período",
                      f"{scenario['conversoes_hoje_no_horizonte']:,.0f} conv.".replace(",", "."),
                      f"com {fmt_money(scenario['investimento_hoje_no_horizonte'])}")

            plain_box(
                "Leia com atenção: esta verba é ADICIONAL",
                f"A simulação assume que <b>{fmt_money(budget)}</b> entram <b>além</b> do que já é "
                f"investido hoje. No mesmo período, a operação atual já gasta "
                f"<b>{fmt_money(scenario['investimento_hoje_no_horizonte'])}</b> e entrega "
                f"<b>{scenario['conversoes_hoje_no_horizonte']:,.0f}</b> conversões. "
                f"Sua verba acrescenta <b>{scenario['acrescimo_vs_hoje_%']:.1f}%</b> sobre isso. "
                "Comparar o pedaço novo com a operação inteira é o que evita a decepção clássica de "
                "achar que uma verba pequena vai mudar o patamar do negócio."
                .replace(",", "."),
                "🧾",
            )

            highlight(
                f"{scenario['conversoes_estimadas']:,.0f}".replace(",", ".") + " conversões",
                f"é o que <b>{fmt_money(budget)}</b> deve trazer em "
                f"<b>{horizon_name.split(' (')[0].lower()}</b>, a um custo médio de "
                f"<b>{fmt_money(scenario['cpa_medio'])}</b> cada.",
                color=TEAL,
            )
            st.write("")

            table = scenario["table"]
            c1, c2 = st.columns([3, 2])
            with c1:
                fig = go.Figure()
                fig.add_bar(x=table["canal"], y=table["investimento"], name="Investimento",
                            marker_color=GOLD, yaxis="y")
                fig.add_scatter(x=table["canal"], y=table["conversoes_estimadas"],
                                name="Conversões estimadas", mode="lines+markers",
                                line=dict(color=TEAL, width=3), yaxis="y2")
                fig.update_layout(
                    title="Onde vai a verba e o que ela traz de volta",
                    yaxis=dict(title="investimento (R$)"),
                    yaxis2=dict(title="conversões", overlaying="y", side="right", showgrid=False),
                )
                st.plotly_chart(apply_theme(fig, height=420), width="stretch")
            with c2:
                st.dataframe(
                    table[["canal", "investimento", "cpa_hist", "cpa_estimado", "conversoes_estimadas"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "canal": "Canal",
                        "investimento": st.column_config.NumberColumn("Recebe", format="%.0f"),
                        "cpa_hist": st.column_config.NumberColumn("Custo/conversão hoje", format="%.0f"),
                        "cpa_estimado": st.column_config.NumberColumn("Custo/conversão no cenário", format="%.0f"),
                        "conversoes_estimadas": st.column_config.NumberColumn("Conversões", format="%.0f"),
                    },
                )

            plain_box(
                "Por que o custo por conversão sobe quando eu invisto mais",
                "Repare que a coluna “custo no cenário” pode ficar maior que a de hoje. Não é erro: "
                "os primeiros clientes de um canal são os mais baratos de conquistar — são os que "
                "já estavam quase decididos. Conforme você investe mais no mesmo canal, passa a "
                "pagar para falar com gente cada vez menos interessada, e cada conversão sai mais "
                "cara. A simulação aplica esse desconto de eficiência automaticamente.",
                "📉",
            )

            if scenario["canais_esticados"]:
                st.warning(
                    "**Você está esticando demais estes canais:** "
                    + ", ".join(scenario["canais_esticados"])
                    + ". O investimento simulado é mais que o dobro do patamar histórico deles. "
                    "Na prática, o público disponível pode simplesmente não existir nesse volume — "
                    "considere distribuir melhor ou aumentar o prazo.",
                    icon="⚠️",
                )

            plain_box(
                "Os limites honestos desta simulação",
                "Ela usa o custo por conversão histórico de cada canal e aplica um desconto de "
                "eficiência conforme a verba cresce. O que ela <b>não</b> faz: prever mudanças de "
                "concorrência, sazonalidade ou criativo novo. E, principalmente, ela não mede "
                "causa — se um canal parece barato porque só fala com quem já ia comprar, a conta "
                "vai continuar parecendo boa. Para separar isso, use um teste A/B de verdade.",
                "🎓",
                NAVY,
            )

            repository.save_widget(
                key="mta_journey",
                origem="MTA",
                canal_driver=str(table.iloc[0]["canal"]),
                hipotese_default=f"Investir {fmt_money(budget)} em "
                                 f"{horizon_name.split(' (')[0].lower()} traz "
                                 f"{scenario['conversoes_estimadas']:.0f} conversões.",
                resultado_default="Oportunidade",
                insight_default=(
                    f"Cenário de {fmt_money(budget)} em {len(selected)} canais "
                    f"({horizon_name.split(' (')[0].lower()}): "
                    f"{scenario['conversoes_estimadas']:.0f} conversões estimadas a "
                    f"{fmt_money(scenario['cpa_medio'])} cada."
                ),
                proximo_passo_default="Rodar um teste A/B para confirmar a eficiência antes de escalar.",
                etapa_default="Priorization",
            )

# ===========================================================================
# 4. PRESCRITIVO
# ===========================================================================
with tab4:
    stage_header(
        4, "Prescritivo", "O que eu devo fazer?",
        "A decisão da semana: qual canal reforçar, qual segurar, e como não repetir o erro de "
        "cortar justamente o canal que sustenta o começo da jornada.",
        POSITIVE,
    )

    reference = heuristics.copy()
    if len(markov_share):
        reference = reference.join(markov_share, how="left").fillna(0.0)

    rec = mta_recommendation(reference, adspend)
    if not rec["ok"]:
        st.warning(f"Recomendação indisponível: {rec.get('message')}")
    else:
        recommendation_panel(
            rec["headline"], rec["detail"], rec["invest"], rec["watch"],
            invest_note=(
                f"Está <b>{rec['gap_invest_pp']:.1f} pontos subestimado</b> pelo relatório padrão. "
                + (f"Seguindo o crédito justo, mereceria <b>{fmt_money(abs(rec['amount']))}</b> a mais."
                   if rec["amount"] == rec["amount"] else "")
            ),
            watch_note=(
                f"Está <b>{rec['gap_watch_pp']:.1f} pontos superestimado</b>. "
                + (f"Dá para <b>economizar {fmt_money(rec['save'])}</b> aqui sem perder resultado."
                   if rec["save"] == rec["save"] else "")
            ),
        )

        plain_box(
            "O que exatamente fazer nesta semana",
            f"1. Aumentar gradualmente a verba de <b>{rec['invest']}</b> — comece com 15% a 20%, "
            "não o valor cheio de uma vez.<br>"
            f"2. Segurar a mão em <b>{rec['watch']}</b>: ele provavelmente está recebendo crédito "
            "por conversões que outro canal construiu.<br>"
            "3. Mexer em <b>um canal por vez</b>. Se mudar tudo junto, você nunca vai saber o que "
            "funcionou.<br>"
            "4. Esperar pelo menos uma semana completa antes de avaliar — dias da semana têm "
            "comportamentos muito diferentes.",
            "✅",
        )

        st.subheader("Eficiência de cada canal hoje")
        eff = pd.DataFrame({
            "canal": reference.index,
            "investimento": adspend.reindex(reference.index).fillna(0.0).to_numpy(),
        })
        credit_col = "Markov" if "Markov" in reference.columns else "Linear"
        eff["credito_%"] = reference[credit_col].to_numpy()
        total_conv = float(journeys["converted"].sum())
        eff["conversoes"] = eff["credito_%"] / 100 * total_conv
        eff["custo_por_conversao"] = np.where(
            eff["conversoes"] > 0, eff["investimento"] / eff["conversoes"], np.nan
        )
        eff = eff.sort_values("custo_por_conversao")

        st.dataframe(
            eff, width="stretch", hide_index=True,
            column_config={
                "canal": "Canal",
                "investimento": st.column_config.NumberColumn("Investido", format="%.0f"),
                "credito_%": st.column_config.ProgressColumn("Crédito justo", min_value=0,
                                                             max_value=100, format="%.1f%%"),
                "conversoes": st.column_config.NumberColumn("Conversões", format="%.0f"),
                "custo_por_conversao": st.column_config.NumberColumn("Custo por conversão", format="%.0f"),
            },
        )
        plain_box(
            "A regra de bolso",
            f"O canal mais eficiente hoje é **{eff.iloc[0]['canal']}**, a "
            f"{fmt_money(eff.iloc[0]['custo_por_conversao'])} por conversão. O mais caro é "
            f"**{eff.iloc[-1]['canal']}**, a {fmt_money(eff.iloc[-1]['custo_por_conversao'])}. "
            "Mas cuidado: canal caro não é necessariamente canal ruim. Se ele é o que apresenta a "
            "marca para quem nunca ouviu falar dela, o custo alto é o preço da descoberta — "
            "e cortá-lo seca o funil inteiro alguns meses depois.",
            "📏",
        )

    st.subheader("O elo com o resto do framework")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"<div class='mmm-card' style='border-top-color:{NAVY}'>"
            "<h4>⬆️ Suba para o estratégico</h4>"
            "<p>O MTA não vê TV, jornal nem o efeito de marca de longo prazo. Antes de decidir a "
            "verba do trimestre, confira a página <b>MMM na prática</b> — lá a comparação inclui "
            "toda a mídia, não só a digital.</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='mmm-card' style='border-top-color:{GOLD}'>"
            "<h4>🧪 Prove antes de escalar</h4>"
            "<p>Tudo aqui é correlação: o modelo vê quem passou por onde, não o que teria "
            "acontecido sem aquele canal. Um teste A/B na página "
            "<b>Simulador e Resultados</b> é o que transforma suspeita em certeza.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "**Resumo da jornada:** você viu o que aconteceu (1), entendeu qual canal merece o crédito "
        "(2), simulou o retorno da sua verba (3) e recebeu o plano da semana (4). "
        "Registre a decisão no **🔁 Learning Repository** para que o próximo ciclo comece adiante."
    )

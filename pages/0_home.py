"""Home — resumo executivo do framework e loop de governança."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import reference
from src.data_loader import kpi_snapshot
from src.utils.styling import BLUE, GOLD, NAVY, TEAL, fmt_money, nav_card, page_header, tokens
from src.viz.charts import apply_theme

page_header(
    "MMM × MTA — Framework de Digital Analytics",
    "Uma ferramenta única para as três camadas de medição: o estratégico que dimensiona, "
    "o tático que otimiza e o experimento que prova causalidade.",
    layer="GOV",
)

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
kpis = kpi_snapshot()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Investimento em mídia (MMM)", fmt_money(kpis["mmm_investimento"]),
          help="Soma de todos os canais on/offline no mmm_dataset.csv")
c2.metric("Vendas no período (MMM)", fmt_money(kpis["mmm_sales"]),
          f"{kpis['mmm_semanas']} semanas")
c3.metric("Taxa de conversão (MTA)", f"{kpis['mta_conv_rate'] * 100:.1f}%",
          f"{kpis['mta_clientes']:,} clientes".replace(",", "."))
c4.metric("ROI médio (campanhas)", f"{kpis['ab_roi']:.2f}x",
          f"{kpis['ab_campanhas']:,} campanhas".replace(",", "."))

st.caption(
    f"⚠️ Data Quality: {kpis['mmm_missing_pct']:.0f}% das células de investimento do dataset de MMM "
    "estão ausentes — o tratamento é explícito na página *MMM Explorer*."
)

st.divider()

# ---------------------------------------------------------------------------
# As três camadas
# ---------------------------------------------------------------------------
st.subheader("Por que isso importa agora")
st.markdown(
    "A atribuição tradicional (last-click / MTA puro) está estruturalmente quebrada no ambiente de "
    "privacidade atual — iOS 14.5+, cookieless, LGPD/GDPR e walled gardens. O resultado prático é "
    "**overspend em resposta direta** e **underinvestment em topo de funil**."
)
stat_cols = st.columns(len(reference.EXECUTIVE_STATS))
for col, (number, headline, detail) in zip(stat_cols, reference.EXECUTIVE_STATS):
    with col:
        st.markdown(
            f"<div class='mmm-card' style='border-top-color:{NAVY}'>"
            f"<span style='font-size:2rem;font-weight:800;color:{GOLD}'>{number}</span>"
            f"<p style='margin-top:6px'><b>{headline}</b><br>{detail}</p></div>",
            unsafe_allow_html=True,
        )
st.caption("Fonte: sumário executivo do framework de referência.")

st.divider()

st.subheader("As três camadas de medição")
col_a, col_b, col_c = st.columns(3)
with col_a:
    nav_card(
        "🏛️ MMM — Estratégico (top-down)",
        "Modela o negócio inteiro a partir de dados agregados: mídia on e offline, sazonalidade, "
        "promoção e concorrência. Responde 'quanto cada canal contribuiu' e 'como realocar o budget'. "
        "Imune a cookies e privacidade, mas lento e de baixa granularidade.",
    )
with col_b:
    nav_card(
        "🔬 MTA — Tático (bottom-up)",
        "Reparte o crédito da conversão entre os touchpoints da jornada do usuário. "
        "Responde 'qual canal empurrou esta conversão' com granularidade de campanha e criativo. "
        "Rápido e acionável, mas só enxerga o digital rastreável.",
    )
with col_c:
    nav_card(
        "🧪 Testes A/B — Validação causal",
        "O árbitro. MMM e MTA são correlacionais; o experimento controlado é a única evidência "
        "causal. Alimentado pelas hipóteses das outras duas camadas e devolve a verdade que "
        "recalibra os modelos.",
    )

st.divider()

# ---------------------------------------------------------------------------
# Loop de governança
# ---------------------------------------------------------------------------
st.subheader("O loop de governança")
st.markdown(
    "Nenhuma camada substitui a outra. O valor está no ciclo: o MMM aponta **onde** há oportunidade, "
    "o MTA diz **como** executar, o Teste A/B **prova** e o Learning Repository **retroalimenta** os modelos."
)


def loop_diagram() -> go.Figure:
    """Diagrama do loop MMM ⇄ MTA ⇄ A/B ⇄ Learning Repository (Plotly shapes)."""
    nodes = [
        ("MMM\nEstratégico", 0.5, 0.88, NAVY),
        ("MTA\nTático", 0.90, 0.5, TEAL),
        ("Testes A/B\nValidação causal", 0.5, 0.12, GOLD),
        ("Learning\nRepository", 0.10, 0.5, BLUE),
    ]
    fig = go.Figure()
    for label, x, y, color in nodes:
        fig.add_shape(
            type="circle", x0=x - 0.15, x1=x + 0.15, y0=y - 0.11, y1=y + 0.11,
            fillcolor=color, line=dict(color=color), opacity=0.95, layer="below",
        )
        fig.add_annotation(x=x, y=y, text=f"<b>{label}</b>", showarrow=False,
                           font=dict(color="#FFFFFF", size=13))

    arrows = [(0.5, 0.75, 0.85, 0.60), (0.88, 0.38, 0.60, 0.16),
              (0.38, 0.10, 0.14, 0.38), (0.13, 0.62, 0.40, 0.84)]
    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.2, arrowcolor="#94A3B8",
        )

    labels = [
        (0.76, 0.76, "aponta onde investir"),
        (0.76, 0.24, "gera hipóteses"),
        (0.22, 0.24, "registra o aprendizado"),
        (0.22, 0.76, "recalibra o modelo"),
    ]
    for x, y, text in labels:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                           font=dict(size=11, color=tokens().text_muted))

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(showlegend=False)
    return apply_theme(fig, height=520, legend_bottom=False)


st.plotly_chart(loop_diagram(), width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Matriz de decisão / navegação
# ---------------------------------------------------------------------------
st.subheader("Quando usar cada página")

GUIDE = [
    ("📊 MMM Explorer", "Antes de qualquer modelo: entender a série, tratar missings e outliers e checar se o dado sustenta a análise."),
    ("🧪 MMM Modelagem", "Quando a pergunta é 'quanto cada canal contribuiu para as vendas?' — inclusive TV e jornal, que o MTA não enxerga."),
    ("💰 Otimizador de Budget", "No planejamento anual/trimestral: como redistribuir o mesmo orçamento para vender mais."),
    ("🔀 Modelos de Atribuição", "Quando a pergunta é 'qual canal digital merece o crédito desta conversão?' — e para expor a distorção do last-click."),
    ("🕸️ Markov e Shapley", "Quando a heurística não basta: crédito algorítmico com removal effect e teoria dos jogos."),
    ("🎯 Propensão à Conversão", "Para prever quem vai converter ANTES de acontecer — lead scoring e otimização de audiência."),
    ("🧬 Calculadora A/B", "No desenho do teste: quantos visitantes e quantos dias são necessários para detectar o efeito esperado."),
    ("📈 Simulador e Resultados", "Na leitura do teste: significância frequentista, bayesiana e sequencial, com veredito Winner/Neutral/Loser."),
    ("🌍 Geo-Holdout x MMM", "Para provar causalidade de um canal e checar se o MMM está calibrado — o elo MMM → experimento."),
    ("🔁 Learning Repository", "Sempre ao final: registrar o aprendizado para que a próxima decisão comece de onde esta parou."),
]

for i in range(0, len(GUIDE), 2):
    cols = st.columns(2)
    for col, (title, desc) in zip(cols, GUIDE[i : i + 2]):
        with col:
            nav_card(title, desc)
    st.write("")

st.divider()

# ---------------------------------------------------------------------------
# Matriz de decisão do framework (slide 07 do PPT de referência)
# ---------------------------------------------------------------------------
st.subheader("Matriz de decisão — qual técnica usar em cada cenário")
st.caption(
    "Reproduzida do slide *07 · Matriz de Decisão* do framework de referência, com a coluna extra "
    "apontando onde cada cenário é exercitado neste app."
)
st.dataframe(
    reference.DECISION_MATRIX, width="stretch", hide_index=True,
    column_config={
        "Cenário de negócio": st.column_config.TextColumn(width="medium"),
        "Por quê": st.column_config.TextColumn(width="medium"),
        "Página do app": st.column_config.TextColumn(width="small"),
    },
)

st.divider()

# ---------------------------------------------------------------------------
# As 7 etapas do Roadmap de Testes A/B
# ---------------------------------------------------------------------------
st.subheader("As 7 etapas do Roadmap de Testes A/B")
st.caption(
    "O funil de experimentação do framework. Cada registro salvo no Learning Repository carrega "
    "a etapa em que está — é o que o gráfico de funil da página de Governança mostra."
)
# 4 + 3 em vez de 7 colunas: com 7 os cards ficam estreitos demais para ler.
for start, size in ((0, 4), (4, 3)):
    row = reference.ROADMAP_STAGES[start : start + size]
    cols = st.columns(4)
    for col, (stage, what, source) in zip(cols, row):
        with col:
            st.markdown(
                f"<div class='mmm-card' style='border-top-color:{GOLD}'>"
                f"<h4>{start + row.index((stage, what, source)) + 1}. {stage}</h4>"
                f"<p><b>{what}</b><br><br>{source}</p></div>",
                unsafe_allow_html=True,
            )
    st.write("")

st.divider()

# ---------------------------------------------------------------------------
# Documento de referência
# ---------------------------------------------------------------------------
slides = reference.load_slides()
if not slides.empty:
    with st.expander(f"📄 Documento de referência do framework ({len(slides)} slides)"):
        st.caption(
            "Lido diretamente de `reference/Framework_MMM_x_MTA_Digital_Analytics.pptx` com "
            "`python-pptx`. É a fonte da linguagem, da ordem das camadas e das calculadoras deste app."
        )
        query = st.text_input("Buscar no documento", placeholder="ex.: geo-experiment, adstock, SPRT")
        found = reference.search_slides(query)
        if query and found.empty:
            st.info("Nenhum slide contém esse termo.")
        else:
            if query:
                st.caption(f"{len(found)} slide(s) encontrado(s).")
            options = {f"{r.slide:02d} · {r.titulo}": r.slide for r in found.itertuples()}
            if options:
                choice = st.selectbox("Slide", list(options), key="ref_slide")
                row = slides[slides["slide"] == options[choice]].iloc[0]
                if row["secao"]:
                    st.caption(row["secao"])
                st.text(row["texto"])
else:
    st.caption(
        "ℹ️ Nenhum `.pptx` encontrado em `reference/` — coloque o documento do framework lá para "
        "navegá-lo aqui dentro do app."
    )

st.divider()
st.caption(
    "Fontes: `mmm_dataset.csv` (semanal, mídia on+offline), "
    "`digital_marketing_campaign_dataset.csv` (cliente/campanha) e "
    "`marketing_campaign_dataset.csv` (200k campanhas). "
    "As jornadas multi-touch (páginas de MTA) e o geo-holdout (página 8) são **simulações documentadas** — "
    "veja o README para as limitações assumidas."
)

"""Ponto único de acesso a TODOS os modelos de atribuição do app.

Reúne os cinco heurísticos (first-click → last-click) e os três algorítmicos
(Data-Driven no estilo GA4, Markov e Shapley) numa interface só, para que as
páginas guiadas possam usar a mesma escolha do usuário nas quatro etapas —
descritivo, diagnóstico, preditivo e prescritivo.

O padrão do app é o **Data-Driven**, pelo mesmo motivo que o GA4 o adotou como
padrão em 2023: é o único que aprende os pesos com o dado e usa também as
jornadas que não converteram como grupo de comparação.

A régua escolhida muda tudo o que vem depois: o crédito por canal, o custo por
conversão implícito, a divisão da verba na simulação e a recomendação final.
É exatamente essa a lição do framework — **modelo de atribuição é decisão de
governança, não um default de ferramenta.**
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import load_digital
from src.mta.heuristics import HEURISTIC_MODELS, attribute_all, to_share
from src.mta.journey_sim import adspend_by_channel, build_journeys
from src.mta.dda import dda_attribution
from src.mta.markov import markov_attribution
from src.mta.shapley import shapley_attribution

MARKOV = "Markov (algorítmico)"
SHAPLEY = "Shapley (algorítmico)"
DDA = "Data-Driven (estilo GA4)"

# Ordem didática: do começo da jornada ao fim, depois os algorítmicos.
ATTRIBUTION_MODELS = [
    "First-Click",
    "Linear",
    "Time-Decay",
    "Position-Based (U)",
    "Last-Click",
    DDA,
    MARKOV,
    SHAPLEY,
]

ALGORITHMIC_MODELS = [DDA, MARKOV, SHAPLEY]

# Descrição em linguagem de leigo, exibida junto do seletor.
MODEL_HELP = {
    "First-Click": (
        "100% do crédito para o **primeiro** canal da jornada. Responde “quem apresentou a marca”. "
        "Ignora todo o esforço de convencimento que veio depois."
    ),
    "Linear": (
        "Divide o crédito **igualmente** entre todos os canais da jornada. Justo e simples, "
        "mas trata como iguais contatos que tiveram pesos bem diferentes."
    ),
    "Time-Decay": (
        "Quanto **mais perto da conversão**, mais crédito. Bom para ciclo de compra curto e "
        "promoção — ainda subestima o topo de funil."
    ),
    "Position-Based (U)": (
        "**40% para o primeiro, 40% para o último, 20% para o meio.** Tenta equilibrar descoberta "
        "e fechamento. Os pesos são arbitrários, não saem do seu dado."
    ),
    "Last-Click": (
        "100% para o **último** canal antes da conversão. É o padrão da maioria das ferramentas "
        "e o mais distorcido: superestima quem só colhe o que os outros plantaram."
    ),
    DDA: (
        "O padrão do GA4 desde 2023. É o único da lista que olha também as jornadas que "
        "**não** converteram — sem esse grupo de comparação não dá para saber se um canal "
        "aparece nas conversões porque ele convence ou porque ele aparece em tudo. "
        "Aprende os pesos com o próprio dado e leva em conta a posição do contato."
    ),
    MARKOV: (
        "Pergunta “**se este canal sumisse, quantas conversões eu perderia?**”. Aprende com o seu "
        "dado, considera a ordem dos contatos e é o mais próximo de medir causa."
    ),
    SHAPLEY: (
        "Vindo da teoria dos jogos: mede a contribuição média de cada canal em **todas as "
        "combinações possíveis**. Ignora a ordem e olha a presença do canal na jornada."
    ),
}

MODEL_KIND = {m: ("Algorítmico (aprende com o dado)" if m in ALGORITHMIC_MODELS
                  else "Heurístico (regra fixa)") for m in ATTRIBUTION_MODELS}

SESSION_KEY = "attribution_model"


# ---------------------------------------------------------------------------
# Cálculo
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Calculando os modelos de atribuição...")
def attribution_shares(half_life: float = 1.0) -> pd.DataFrame:
    """% de crédito por canal em TODOS os modelos. Índice = canal, colunas = modelos."""
    journeys = build_journeys(load_digital())
    table = to_share(attribute_all(journeys, half_life))

    channels = sorted({c for path in journeys["path"] for c in path})
    try:
        markov = markov_attribution(journeys, channels)["removal_effect_norm"] * 100
        table[MARKOV] = markov
    except Exception:
        table[MARKOV] = float("nan")
    try:
        table[SHAPLEY] = shapley_attribution(journeys, channels)["share_%"]
    except Exception:
        table[SHAPLEY] = float("nan")
    try:
        dda = dda_attribution(journeys, channels)
        table[DDA] = dda["share_%"]
        auc = float(dda.attrs.get("auc", float("nan")))
    except Exception:
        table[DDA] = float("nan")
        auc = float("nan")

    table = table.reindex(columns=[m for m in ATTRIBUTION_MODELS if m in table.columns])
    table.index.name = "canal"
    table = table.fillna(0.0)
    table.attrs["dda_auc"] = auc
    return table


@st.cache_data(show_spinner=False)
def dda_signal(half_life: float = 1.0) -> float:
    """AUC do modelo do DDA: o quanto a jornada separa quem converte de quem não converte.

    Perto de 0,5 a jornada quase não explica a conversão — o DDA então se aproxima
    do Linear. Isso é honestidade do modelo, não defeito: sem sinal no dado, não há
    como um algoritmo descobrir qual canal decide.
    """
    return float(attribution_shares(half_life).attrs.get("dda_auc", float("nan")))


@st.cache_data(show_spinner=False)
def total_conversions() -> float:
    return float(build_journeys(load_digital())["converted"].sum())


@st.cache_data(show_spinner=False)
def channel_adspend() -> pd.Series:
    return adspend_by_channel(build_journeys(load_digital()))


def model_credit(model: str, half_life: float = 1.0) -> pd.Series:
    """Conversões creditadas por canal segundo o modelo escolhido."""
    shares = attribution_shares(half_life)
    if model not in shares.columns:
        model = "Linear"
    return shares[model] / 100 * total_conversions()


def model_cpa(model: str, half_life: float = 1.0) -> pd.DataFrame:
    """Custo por conversão implícito de cada canal, sob a régua escolhida."""
    credit = model_credit(model, half_life)
    spend = channel_adspend().reindex(credit.index).fillna(0.0)
    out = pd.DataFrame({"canal": credit.index, "investimento": spend.to_numpy(),
                        "conversoes": credit.to_numpy()})
    out["cpa"] = out.apply(
        lambda r: r["investimento"] / r["conversoes"] if r["conversoes"] > 0 else float("nan"), axis=1
    )
    out["credito_%"] = attribution_shares(half_life)[model].reindex(out["canal"]).to_numpy()
    return out.sort_values("cpa").reset_index(drop=True)


def model_spread(half_life: float = 1.0) -> pd.DataFrame:
    """Quanto cada canal varia entre a régua mais e a menos generosa.

    Canal com muita variação é o que mais depende de uma decisão de governança —
    e o melhor candidato a um teste de incrementalidade.
    """
    shares = attribution_shares(half_life)
    out = pd.DataFrame({
        "canal": shares.index,
        "minimo_%": shares.min(axis=1).to_numpy(),
        "maximo_%": shares.max(axis=1).to_numpy(),
        "modelo_mais_generoso": shares.idxmax(axis=1).to_numpy(),
        "modelo_menos_generoso": shares.idxmin(axis=1).to_numpy(),
    })
    out["variacao_pp"] = out["maximo_%"] - out["minimo_%"]
    return out.sort_values("variacao_pp", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Seletor compartilhado entre as páginas
# ---------------------------------------------------------------------------
def model_selector(location=st, key: str = "attr_model", default: str = DDA) -> str:
    """Seletor de régua de atribuição, com a escolha compartilhada entre páginas."""
    stored = st.session_state.get(SESSION_KEY, default)
    if stored not in ATTRIBUTION_MODELS:
        stored = default
    choice = location.selectbox(
        "Régua de atribuição (vale para as 4 etapas)",
        ATTRIBUTION_MODELS,
        index=ATTRIBUTION_MODELS.index(stored),
        key=key,
        help="Como o crédito de uma conversão é repartido entre os canais pelos quais a pessoa "
             "passou. A escolha muda o crédito, o custo por conversão, a simulação e a recomendação.",
    )
    st.session_state[SESSION_KEY] = choice
    return choice

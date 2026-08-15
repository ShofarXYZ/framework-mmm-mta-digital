"""Simulação transparente de jornadas multi-touch.

POR QUE SIMULAR
---------------
`digital_marketing_campaign_dataset.csv` é um dataset **cliente/campanha**: cada
linha é um cliente com UM canal de campanha (`CampaignChannel`) e um desfecho
binário (`Conversion`). Não existe clickstream, nem timestamp, nem sequência de
touchpoints. Modelos de atribuição (first/last-click, Markov, Shapley) exigem
uma SEQUÊNCIA. Portanto o app constrói uma jornada sintética, determinística e
auditável a partir dos sinais de engajamento que o dataset realmente tem.

Isto NÃO é tracking real. É uma aproximação pedagógica para exercitar o
framework — quando houver uma tabela `fct_mta_touchpoint` de verdade, basta
substituir esta função e todo o resto do app continua funcionando.

REGRAS DE CONSTRUÇÃO (determinísticas, sem aleatoriedade)
--------------------------------------------------------
Para cada CustomerID monta-se de 1 a 4 touchpoints. Cada touchpoint recebe um
`stage_rank` (quanto menor, mais distante da conversão) e a jornada é ordenada
por esse rank, com o canal real da campanha SEMPRE em último (o mais próximo da
conversão):

1. TOPO / retenção prévia (rank 0)
   Se `PreviousPurchases > 0` ou `LoyaltyPoints` acima da mediana da base, o
   cliente já tinha relacionamento: adiciona-se um touchpoint anterior de canal
   diferente do principal — `Email` se houve abertura de e-mail, senão `SEO`
   (e `Referral` caso o principal já seja `SEO`).

2. AWARENESS social (rank 1)
   Se `SocialShares > 0` e o canal principal não é `Social Media`, adiciona-se
   `Social Media` como touchpoint de descoberta.

3. MEIO / nutrição por e-mail (rank 2)
   Se `EmailOpens > 0` ou `EmailClicks > 0` e o canal principal não é `Email`,
   adiciona-se `Email` como touchpoint intermediário.

4. PESQUISA ativa (rank 3)
   Se `WebsiteVisits` está acima da mediana da base e o canal principal não é
   `SEO`, adiciona-se `SEO` (intenção declarada de busca).

5. PRINCIPAL (rank 4)
   O `CampaignChannel` real da linha, sempre o último touchpoint.

Ajustes finais: touchpoints repetidos em sequência são colapsados e a jornada é
truncada nos 4 touchpoints mais próximos da conversão. O desfecho é a coluna
`Conversion` (0/1) — nada é inventado no alvo.

O `CampaignType` (Awareness/Consideration/Conversion/Retention) desempata a
posição do touchpoint principal em jornadas de tamanho 1 e é preservado na saída
para leitura na UI.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

MTA_CHANNELS = ["Social Media", "Email", "PPC", "SEO", "Referral"]
MAX_TOUCHPOINTS = 4

# Ordem canônica das etapas de campanha (usada apenas para exibição/ordenação).
CAMPAIGN_STAGE_ORDER = {"Awareness": 0, "Consideration": 1, "Retention": 2, "Conversion": 3}


def _build_path(row: pd.Series, loyalty_median: float, visits_median: float) -> list[str]:
    """Aplica as regras 1..5 da docstring do módulo a uma linha de cliente."""
    main = str(row["CampaignChannel"])
    touchpoints: list[tuple[int, str]] = []

    # 1. Retenção prévia
    if (row.get("PreviousPurchases", 0) or 0) > 0 or (row.get("LoyaltyPoints", 0) or 0) > loyalty_median:
        if (row.get("EmailOpens", 0) or 0) > 0 and main != "Email":
            prior = "Email"
        elif main != "SEO":
            prior = "SEO"
        else:
            prior = "Referral"
        touchpoints.append((0, prior))

    # 2. Awareness social
    if (row.get("SocialShares", 0) or 0) > 0 and main != "Social Media":
        touchpoints.append((1, "Social Media"))

    # 3. Nutrição por e-mail
    if ((row.get("EmailOpens", 0) or 0) > 0 or (row.get("EmailClicks", 0) or 0) > 0) and main != "Email":
        touchpoints.append((2, "Email"))

    # 4. Pesquisa ativa
    if (row.get("WebsiteVisits", 0) or 0) > visits_median and main != "SEO":
        touchpoints.append((3, "SEO"))

    # 5. Canal principal — sempre o último
    touchpoints.append((4, main))

    touchpoints.sort(key=lambda t: t[0])
    path = [channel for _, channel in touchpoints]

    # Colapsa repetições consecutivas e trunca nos N mais próximos da conversão.
    collapsed: list[str] = []
    for channel in path:
        if not collapsed or collapsed[-1] != channel:
            collapsed.append(channel)
    return collapsed[-MAX_TOUCHPOINTS:]


@st.cache_data(show_spinner="Simulando jornadas multi-touch...")
def build_journeys(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói a jornada sintética por cliente.

    Returns:
        DataFrame com uma linha por CustomerID:
        CustomerID | path (tupla de canais) | n_touchpoints | converted |
        AdSpend | CampaignChannel | CampaignType | path_str
    """
    loyalty_median = float(pd.to_numeric(df["LoyaltyPoints"], errors="coerce").median())
    visits_median = float(pd.to_numeric(df["WebsiteVisits"], errors="coerce").median())

    paths = [_build_path(row, loyalty_median, visits_median) for _, row in df.iterrows()]

    out = pd.DataFrame(
        {
            "CustomerID": df["CustomerID"].to_numpy(),
            "path": [tuple(p) for p in paths],
            "n_touchpoints": [len(p) for p in paths],
            "converted": df["Conversion"].astype(int).to_numpy(),
            "AdSpend": pd.to_numeric(df["AdSpend"], errors="coerce").fillna(0.0).to_numpy(),
            "CampaignChannel": df["CampaignChannel"].to_numpy(),
            "CampaignType": df["CampaignType"].to_numpy(),
        }
    )
    out["path_str"] = out["path"].map(lambda p: " → ".join(p))
    return out


def journey_stats(journeys: pd.DataFrame) -> dict:
    """Estatísticas descritivas para exibir no expander 'Como a jornada foi simulada?'."""
    return {
        "clientes": int(len(journeys)),
        "conversoes": int(journeys["converted"].sum()),
        "taxa_conversao": float(journeys["converted"].mean()),
        "touchpoints_medio": float(journeys["n_touchpoints"].mean()),
        "jornadas_multitouch_%": float(100 * (journeys["n_touchpoints"] > 1).mean()),
        "caminhos_unicos": int(journeys["path_str"].nunique()),
    }


def top_paths(journeys: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Caminhos mais frequentes, com taxa de conversão de cada um."""
    grp = (
        journeys.groupby("path_str")
        .agg(clientes=("converted", "size"), conversoes=("converted", "sum"))
        .reset_index()
    )
    grp["taxa_conversao"] = grp["conversoes"] / grp["clientes"]
    return grp.sort_values("clientes", ascending=False).head(n).reset_index(drop=True)


def adspend_by_channel(journeys: pd.DataFrame) -> pd.Series:
    """AdSpend total alocado por canal.

    O AdSpend do dataset é por cliente/campanha, então distribuímos o valor
    igualmente entre os touchpoints da jornada daquele cliente.
    """
    totals: dict[str, float] = {}
    for path, spend in zip(journeys["path"], journeys["AdSpend"]):
        if not path:
            continue
        share = float(spend) / len(path)
        for channel in path:
            totals[channel] = totals.get(channel, 0.0) + share
    return pd.Series(totals, name="AdSpend").sort_values(ascending=False)

"""Modelos de atribuição heurísticos (regra fixa de repartição de crédito).

Todos operam sobre a mesma jornada sintética (`journey_sim.build_journeys`) e
distribuem 1 crédito por conversão. Jornadas sem conversão não geram crédito.

Modelos implementados:
  * first-click     — 100% para o primeiro touchpoint
  * last-click      — 100% para o último touchpoint (o padrão do mercado, o que
                      mais distorce em jornadas longas)
  * linear          — crédito igual para todos os touchpoints
  * time-decay      — peso exponencial crescente rumo à conversão (meia-vida
                      configurável em número de touchpoints)
  * position-based  — U-shaped 40/20/40: 40% primeiro, 40% último, 20% no meio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HEURISTIC_MODELS = ["First-Click", "Last-Click", "Linear", "Time-Decay", "Position-Based (U)"]


def _weights(path: tuple[str, ...], model: str, half_life: float = 1.0) -> np.ndarray:
    """Vetor de pesos (soma 1) para um caminho, segundo o modelo escolhido."""
    n = len(path)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    if model == "First-Click":
        w = np.zeros(n)
        w[0] = 1.0
    elif model == "Last-Click":
        w = np.zeros(n)
        w[-1] = 1.0
    elif model == "Linear":
        w = np.full(n, 1.0 / n)
    elif model == "Time-Decay":
        # distância até a conversão em "passos"; peso = 2^(-dist/half_life)
        distance = np.arange(n - 1, -1, -1, dtype=float)
        w = np.power(2.0, -distance / max(half_life, 1e-6))
        w = w / w.sum()
    elif model == "Position-Based (U)":
        if n == 2:
            w = np.array([0.5, 0.5])
        else:
            w = np.full(n, 0.20 / (n - 2))
            w[0] = 0.40
            w[-1] = 0.40
    else:
        w = np.full(n, 1.0 / n)
    return w


def attribute(journeys: pd.DataFrame, model: str, half_life: float = 1.0) -> pd.Series:
    """Crédito de conversões por canal para um modelo heurístico."""
    credit: dict[str, float] = {}
    converted = journeys[journeys["converted"] == 1]
    for path in converted["path"]:
        w = _weights(tuple(path), model, half_life)
        for channel, weight in zip(path, w):
            credit[channel] = credit.get(channel, 0.0) + float(weight)
    return pd.Series(credit, dtype=float).sort_values(ascending=False)


def attribute_all(journeys: pd.DataFrame, half_life: float = 1.0) -> pd.DataFrame:
    """Tabela canal x modelo com o crédito absoluto de conversões."""
    frames = {m: attribute(journeys, m, half_life) for m in HEURISTIC_MODELS}
    table = pd.DataFrame(frames).fillna(0.0)
    table.index.name = "canal"
    return table.sort_values(HEURISTIC_MODELS[1], ascending=False)


def to_share(table: pd.DataFrame) -> pd.DataFrame:
    """Converte crédito absoluto em % do total, por modelo (coluna)."""
    totals = table.sum(axis=0).replace(0, np.nan)
    return (100 * table / totals).fillna(0.0)


def cpa_table(table: pd.DataFrame, adspend: pd.Series) -> pd.DataFrame:
    """CPA implícito por modelo: AdSpend do canal / conversões creditadas."""
    out = pd.DataFrame(index=table.index)
    out["AdSpend"] = adspend.reindex(table.index).fillna(0.0)
    for model in table.columns:
        credited = table[model].replace(0, np.nan)
        out[f"CPA {model}"] = out["AdSpend"] / credited
    return out

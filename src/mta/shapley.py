"""Shapley Value entre canais (teoria dos jogos cooperativos).

Função característica: v(S) = número de conversões geradas por jornadas cujo
CONJUNTO de canais está contido na coalizão S. É a definição clássica usada em
atribuição (Dalessandro et al.), e ignora a ordem dos touchpoints — por isso é
complementar ao Markov, que é sensível à ordem.

Com 5 canais existem apenas 2^5 = 32 coalizões, então o cálculo EXATO
(média das contribuições marginais sobre as 120 permutações, via fórmula
fatorial) é viável. A versão Monte Carlo por amostragem de permutações fica
disponível para quando o número de canais crescer.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd


def coalition_values(journeys: pd.DataFrame, channels: list[str]) -> dict[frozenset, float]:
    """v(S) para toda coalizão S: conversões de jornadas com canais ⊆ S."""
    converted = journeys[journeys["converted"] == 1]
    journey_sets = [frozenset(p) for p in converted["path"]]

    counts: dict[frozenset, float] = {}
    for s in journey_sets:
        counts[s] = counts.get(s, 0.0) + 1.0

    values: dict[frozenset, float] = {}
    for size in range(len(channels) + 1):
        for combo in combinations(channels, size):
            S = frozenset(combo)
            values[S] = float(sum(v for k, v in counts.items() if k and k.issubset(S)))
    return values


def shapley_exact(values: dict[frozenset, float], channels: list[str]) -> pd.Series:
    """Shapley exato: φ_i = Σ_S |S|!(n-|S|-1)!/n! · [v(S∪{i}) − v(S)]."""
    n = len(channels)
    phi = {c: 0.0 for c in channels}
    others = {c: [x for x in channels if x != c] for c in channels}

    for channel in channels:
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for combo in combinations(others[channel], size):
                S = frozenset(combo)
                marginal = values.get(S | {channel}, 0.0) - values.get(S, 0.0)
                phi[channel] += weight * marginal
    return pd.Series(phi, dtype=float).sort_values(ascending=False)


def shapley_monte_carlo(
    values: dict[frozenset, float], channels: list[str], n_permutations: int = 2000, seed: int = 42
) -> pd.Series:
    """Aproximação por amostragem de permutações (útil se o nº de canais crescer)."""
    rng = np.random.default_rng(seed)
    phi = {c: 0.0 for c in channels}
    order = np.array(channels, dtype=object)

    for _ in range(n_permutations):
        perm = rng.permutation(order)
        current: set = set()
        prev = values.get(frozenset(), 0.0)
        for channel in perm:
            current.add(channel)
            v = values.get(frozenset(current), 0.0)
            phi[channel] += v - prev
            prev = v
    return pd.Series({c: v / n_permutations for c, v in phi.items()}, dtype=float).sort_values(
        ascending=False
    )


def shapley_attribution(
    journeys: pd.DataFrame,
    channels: list[str] | None = None,
    method: str = "Exato",
    n_permutations: int = 2000,
) -> pd.DataFrame:
    """Crédito de conversões por canal via Shapley."""
    if channels is None:
        channels = sorted({c for path in journeys["path"] for c in path})

    values = coalition_values(journeys, channels)
    phi = (
        shapley_exact(values, channels)
        if method == "Exato"
        else shapley_monte_carlo(values, channels, n_permutations)
    )

    total_conversions = float(journeys["converted"].sum())
    out = phi.to_frame("shapley_value")
    total = out["shapley_value"].clip(lower=0).sum()
    out["share_%"] = 100 * out["shapley_value"].clip(lower=0) / total if total > 0 else np.nan
    out["conversoes_creditadas"] = out["share_%"] / 100 * total_conversions
    out.index.name = "canal"
    return out

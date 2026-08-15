"""Markov Chain de ordem 1 + Removal Effect (crédito algorítmico, estilo DDA).

Implementação manual com numpy/pandas — sem dependência externa. Se a lib
`pychattr` estiver instalada, `markov_attribution` pode ser trocada por ela sem
mudar a interface (mantemos o fallback manual como caminho padrão porque é
determinístico e auditável).

Estados: (start) + um estado por canal + (conversion) + (null).
Probabilidade de conversão = probabilidade de absorção em (conversion) partindo
de (start), obtida resolvendo o sistema linear (I - Q)^-1 * R.

Removal Effect de um canal C = 1 - P(conversão | C removido) / P(conversão).

ATENÇÃO ao que significa "remover": não basta apagar o canal do caminho — se a
jornada continuasse e convertesse do mesmo jeito, a probabilidade não mudaria e
todo removal effect daria zero. A formulação correta é **cirurgia na matriz de
transição**: toda transição que ia PARA o canal removido passa a ir para
`(null)`, ou seja, as jornadas que dependiam daquele canal morrem. O crédito de
cada canal é o seu removal effect normalizado pela soma de todos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

START = "(start)"
CONVERSION = "(conversion)"
NULL = "(null)"


def build_transition_counts(journeys: pd.DataFrame) -> pd.DataFrame:
    """Matriz de CONTAGENS de transição entre estados (linhas=origem, colunas=destino)."""
    counts: dict[tuple[str, str], float] = {}

    for path, converted in zip(journeys["path"], journeys["converted"]):
        states = [START] + list(path) + [CONVERSION if converted == 1 else NULL]
        for a, b in zip(states[:-1], states[1:]):
            counts[(a, b)] = counts.get((a, b), 0.0) + 1.0

    origins = sorted({a for a, _ in counts})
    targets = sorted({b for _, b in counts})
    labels = sorted(set(origins) | set(targets))
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)
    for (a, b), v in counts.items():
        matrix.loc[a, b] = v
    return matrix


def to_probabilities(counts: pd.DataFrame) -> pd.DataFrame:
    """Normaliza cada linha para somar 1 (estados absorventes viram auto-loop)."""
    probs = counts.copy().astype(float)
    for state in probs.index:
        row_sum = probs.loc[state].sum()
        if row_sum > 0:
            probs.loc[state] = probs.loc[state] / row_sum
        elif state in (CONVERSION, NULL):
            probs.loc[state, state] = 1.0
    return probs


def conversion_probability(probs: pd.DataFrame) -> float:
    """P(absorção em (conversion) | início em (start)) via (I - Q)^-1 R."""
    if START not in probs.index or CONVERSION not in probs.columns:
        return 0.0
    absorbing = [s for s in (CONVERSION, NULL) if s in probs.index]
    transient = [s for s in probs.index if s not in absorbing]
    if not transient:
        return 0.0

    Q = probs.loc[transient, transient].to_numpy(dtype=float)
    R = probs.loc[transient, absorbing].to_numpy(dtype=float)
    try:
        N = np.linalg.inv(np.eye(len(transient)) - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(np.eye(len(transient)) - Q)
    B = N @ R
    start_idx = transient.index(START)
    conv_idx = absorbing.index(CONVERSION)
    return float(np.clip(B[start_idx, conv_idx], 0.0, 1.0))


def remove_channel(probs: pd.DataFrame, channel: str) -> pd.DataFrame:
    """Remove um canal da cadeia redirecionando para `(null)` tudo que ia até ele."""
    if channel not in probs.index:
        return probs
    out = probs.copy()
    if NULL not in out.columns:
        out[NULL] = 0.0
        out.loc[NULL] = 0.0
        out.loc[NULL, NULL] = 1.0
    out[NULL] = out[NULL] + out[channel]
    return out.drop(index=channel, columns=channel)


def markov_attribution(journeys: pd.DataFrame, channels: list[str] | None = None) -> pd.DataFrame:
    """Removal Effect por canal e crédito de conversões redistribuído.

    Returns:
        DataFrame indexado por canal com:
        p_sem_canal | removal_effect | removal_effect_norm | conversoes_creditadas
    """
    if channels is None:
        channels = sorted({c for path in journeys["path"] for c in path})

    base_probs = to_probabilities(build_transition_counts(journeys))
    p_base = conversion_probability(base_probs)
    total_conversions = float(journeys["converted"].sum())

    rows = []
    for channel in channels:
        p_without = conversion_probability(remove_channel(base_probs, channel))
        removal = 1.0 - (p_without / p_base) if p_base > 0 else 0.0
        rows.append(
            {"canal": channel, "p_sem_canal": p_without, "removal_effect": max(removal, 0.0)}
        )

    out = pd.DataFrame(rows).set_index("canal")
    total_removal = out["removal_effect"].sum()
    out["removal_effect_norm"] = (
        out["removal_effect"] / total_removal if total_removal > 0 else np.nan
    )
    out["conversoes_creditadas"] = out["removal_effect_norm"] * total_conversions
    out.attrs["p_base"] = p_base
    out.attrs["total_conversions"] = total_conversions
    return out.sort_values("removal_effect", ascending=False)


def sankey_data(journeys: pd.DataFrame, min_flow: int = 30):
    """Prepara nós/links do fluxo de transição para o gráfico Sankey."""
    counts = build_transition_counts(journeys)
    labels = list(counts.index)
    index = {label: i for i, label in enumerate(labels)}
    source, target, value = [], [], []
    for a in counts.index:
        for b in counts.columns:
            v = float(counts.loc[a, b])
            if v >= min_flow and a != b:
                source.append(index[a])
                target.append(index[b])
                value.append(v)
    return labels, source, target, value

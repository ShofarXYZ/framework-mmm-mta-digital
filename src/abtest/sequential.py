"""Teste sequencial simplificado (SPRT de Wald).

Permite decidir ANTES do fim do período planejado, controlando erros α e β.

Log-likelihood ratio acumulado para Bernoulli:
    LLR = Σ log( P(x|p1) / P(x|p0) )
Fronteiras: A = log((1-β)/α)  -> aceita H1 (há efeito)
            B = log(β/(1-α))  -> aceita H0 (não há efeito)
"""

from __future__ import annotations

import math

import numpy as np


def sprt_boundaries(alpha: float = 0.05, beta: float = 0.20) -> tuple[float, float]:
    upper = math.log((1 - beta) / alpha)
    lower = math.log(beta / (1 - alpha))
    return upper, lower


def sprt_test(
    conv_a: int,
    n_a: int,
    conv_b: int,
    n_b: int,
    mde_pct: float = 10.0,
    alpha: float = 0.05,
    beta: float = 0.20,
) -> dict:
    """SPRT sobre os dados acumulados da variação, com H0: p=p_control.

    H0: taxa da variação = taxa do controle
    H1: taxa da variação = taxa do controle * (1 + MDE)
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Número de visitantes precisa ser maior que zero nas duas variantes.")

    p0 = conv_a / n_a
    p1 = min(max(p0 * (1 + mde_pct / 100.0), 1e-9), 0.999999)
    p0 = min(max(p0, 1e-9), 0.999999)

    successes, failures = int(conv_b), int(n_b - conv_b)
    llr = successes * math.log(p1 / p0) + failures * math.log((1 - p1) / (1 - p0))
    upper, lower = sprt_boundaries(alpha, beta)

    if llr >= upper:
        decision = "Parar — evidência suficiente A FAVOR da variação (H1)"
        status = "H1"
    elif llr <= lower:
        decision = "Parar — evidência suficiente CONTRA a variação (H0)"
        status = "H0"
    else:
        decision = "Continuar coletando — ainda dentro da zona de indecisão"
        status = "continuar"

    progress = 0.0
    if llr >= 0 and upper > 0:
        progress = min(100.0, 100 * llr / upper)
    elif llr < 0 and lower < 0:
        progress = min(100.0, 100 * llr / lower)

    return {
        "llr": float(llr),
        "limite_superior": float(upper),
        "limite_inferior": float(lower),
        "p0": p0,
        "p1": p1,
        "decisao": decision,
        "status": status,
        "progresso_%": float(progress),
    }


def sprt_trajectory(
    conv_a: int, n_a: int, conv_b: int, n_b: int, mde_pct: float = 10.0,
    alpha: float = 0.05, beta: float = 0.20, steps: int = 60, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Trajetória do LLR conforme a amostra da variação cresce.

    Como só temos os totais (não o log evento a evento), reconstruímos uma
    sequência de Bernoulli com a mesma proporção observada e a embaralhamos com
    semente fixa — serve para ilustrar QUANDO a fronteira teria sido cruzada.
    """
    p0 = min(max(conv_a / max(n_a, 1), 1e-9), 0.999999)
    p1 = min(max(p0 * (1 + mde_pct / 100.0), 1e-9), 0.999999)

    rng = np.random.default_rng(seed)
    outcomes = np.zeros(int(n_b), dtype=int)
    outcomes[: int(conv_b)] = 1
    rng.shuffle(outcomes)

    checkpoints = np.linspace(1, len(outcomes), min(steps, len(outcomes))).astype(int)
    cumulative = np.cumsum(outcomes)
    llr = []
    for c in checkpoints:
        s = int(cumulative[c - 1])
        f = int(c - s)
        llr.append(s * math.log(p1 / p0) + f * math.log((1 - p1) / (1 - p0)))
    return checkpoints, np.array(llr)

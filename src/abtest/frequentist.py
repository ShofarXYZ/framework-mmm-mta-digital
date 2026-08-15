"""Teste frequentista de duas proporções (Control vs Variation).

Usa `statsmodels` quando disponível (proportions_ztest / proportion_confint) e
cai para um cálculo manual com `scipy.stats.norm` caso não esteja instalado.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2_contingency, norm

try:
    from statsmodels.stats.proportion import proportion_confint, proportions_ztest

    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False


def _manual_ztest(c_a: int, n_a: int, c_b: int, n_b: int) -> tuple[float, float]:
    p_a, p_b = c_a / n_a, c_b / n_b
    p_pool = (c_a + c_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    return z, float(2 * (1 - norm.cdf(abs(z))))


def _manual_confint(c: int, n: int, alpha: float) -> tuple[float, float]:
    p = c / n
    z = norm.ppf(1 - alpha / 2)
    se = math.sqrt(max(p * (1 - p), 0) / n)
    return float(p - z * se), float(p + z * se)


def two_proportion_ztest(
    conv_a: int, n_a: int, conv_b: int, n_b: int, confidence_levels=(0.90, 0.95, 0.99)
) -> dict:
    """Z-test de duas proporções + ICs + classificação Winner/Neutral/Loser.

    A = Control, B = Variation.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Número de visitantes precisa ser maior que zero nas duas variantes.")
    conv_a, conv_b = int(min(conv_a, n_a)), int(min(conv_b, n_b))

    p_a, p_b = conv_a / n_a, conv_b / n_b

    if STATSMODELS_AVAILABLE:
        z, p_value = proportions_ztest([conv_b, conv_a], [n_b, n_a])
        z, p_value = float(z), float(p_value)
    else:
        z, p_value = _manual_ztest(conv_a, n_a, conv_b, n_b)

    intervals = {}
    for level in confidence_levels:
        alpha = 1 - level
        if STATSMODELS_AVAILABLE:
            lo_a, hi_a = proportion_confint(conv_a, n_a, alpha=alpha, method="wilson")
            lo_b, hi_b = proportion_confint(conv_b, n_b, alpha=alpha, method="wilson")
        else:
            lo_a, hi_a = _manual_confint(conv_a, n_a, alpha)
            lo_b, hi_b = _manual_confint(conv_b, n_b, alpha)
        intervals[level] = {
            "control": (float(lo_a), float(hi_a)),
            "variation": (float(lo_b), float(hi_b)),
            "significativo": bool(p_value < alpha),
        }

    lift = ((p_b - p_a) / p_a * 100) if p_a > 0 else float("nan")
    significant_95 = p_value < 0.05
    if significant_95 and lift > 0:
        verdict = "Winner"
    elif significant_95 and lift < 0:
        verdict = "Loser"
    else:
        verdict = "Neutral"

    # Qui-quadrado como checagem complementar
    try:
        chi2, chi_p, _, _ = chi2_contingency(
            [[conv_a, n_a - conv_a], [conv_b, n_b - conv_b]], correction=True
        )
    except Exception:
        chi2, chi_p = float("nan"), float("nan")

    return {
        "control": {"visitantes": n_a, "conversoes": conv_a, "taxa": p_a},
        "variation": {"visitantes": n_b, "conversoes": conv_b, "taxa": p_b},
        "z_score": z,
        "p_value": p_value,
        "lift_pct": lift,
        "lift_absoluto": p_b - p_a,
        "intervalos": intervals,
        "significativo_90": p_value < 0.10,
        "significativo_95": p_value < 0.05,
        "significativo_99": p_value < 0.01,
        "veredito": verdict,
        "chi2": float(chi2),
        "chi2_p": float(chi_p),
        "engine": "statsmodels" if STATSMODELS_AVAILABLE else "scipy (fallback manual)",
    }


def observed_power(conv_a: int, n_a: int, conv_b: int, n_b: int, confidence: float = 0.95) -> float:
    """Poder observado (post-hoc) — informativo, com as ressalvas usuais."""
    p_a, p_b = conv_a / max(n_a, 1), conv_b / max(n_b, 1)
    se = math.sqrt(max(p_a * (1 - p_a) / max(n_a, 1) + p_b * (1 - p_b) / max(n_b, 1), 1e-12))
    z_crit = norm.ppf(1 - (1 - confidence) / 2)
    effect = abs(p_b - p_a) / se
    return float(np.clip(norm.cdf(effect - z_crit) + norm.cdf(-effect - z_crit), 0, 1))

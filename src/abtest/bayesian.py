"""Teste A/B Bayesiano — modelo Beta-Binomial.

Prior não-informativo Beta(1, 1). Posterior de cada variante:
    Beta(1 + conversões, 1 + não-conversões)

P(B > A) e a perda esperada (expected loss) saem por amostragem com numpy.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist

PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0


def posteriors(conv_a: int, n_a: int, conv_b: int, n_b: int):
    """Parâmetros (alpha, beta) das posteriores de Control e Variation."""
    a_alpha = PRIOR_ALPHA + conv_a
    a_beta = PRIOR_BETA + max(n_a - conv_a, 0)
    b_alpha = PRIOR_ALPHA + conv_b
    b_beta = PRIOR_BETA + max(n_b - conv_b, 0)
    return (a_alpha, a_beta), (b_alpha, b_beta)


def bayesian_test(
    conv_a: int, n_a: int, conv_b: int, n_b: int, n_samples: int = 100_000, seed: int = 42
) -> dict:
    """P(B > A), uplift esperado e perda esperada de escolher cada variante."""
    (a_alpha, a_beta), (b_alpha, b_beta) = posteriors(conv_a, n_a, conv_b, n_b)
    rng = np.random.default_rng(seed)
    samples_a = rng.beta(a_alpha, a_beta, n_samples)
    samples_b = rng.beta(b_alpha, b_beta, n_samples)

    prob_b_better = float(np.mean(samples_b > samples_a))
    diff = samples_b - samples_a
    uplift = np.divide(diff, samples_a, out=np.zeros_like(diff), where=samples_a > 0)

    return {
        "posterior_control": (a_alpha, a_beta),
        "posterior_variation": (b_alpha, b_beta),
        "prob_b_maior_a": prob_b_better,
        "prob_a_maior_b": 1 - prob_b_better,
        "uplift_esperado_%": float(np.mean(uplift) * 100),
        "uplift_hdi_95": (float(np.percentile(uplift, 2.5) * 100), float(np.percentile(uplift, 97.5) * 100)),
        "perda_esperada_escolher_B": float(np.mean(np.maximum(samples_a - samples_b, 0))),
        "perda_esperada_escolher_A": float(np.mean(np.maximum(samples_b - samples_a, 0))),
        "media_control": float(np.mean(samples_a)),
        "media_variation": float(np.mean(samples_b)),
    }


def posterior_curves(result: dict, points: int = 500):
    """Pontos (x, densidade) das duas posteriores, para o gráfico sobreposto."""
    a_alpha, a_beta = result["posterior_control"]
    b_alpha, b_beta = result["posterior_variation"]
    lo = min(beta_dist.ppf(0.0005, a_alpha, a_beta), beta_dist.ppf(0.0005, b_alpha, b_beta))
    hi = max(beta_dist.ppf(0.9995, a_alpha, a_beta), beta_dist.ppf(0.9995, b_alpha, b_beta))
    x = np.linspace(max(lo, 0), min(hi, 1), points)
    return x, beta_dist.pdf(x, a_alpha, a_beta), beta_dist.pdf(x, b_alpha, b_beta)


def decision_label(prob_b_better: float, threshold: float = 0.95) -> str:
    """Traduz a probabilidade posterior no vocabulário Winner/Neutral/Loser."""
    if prob_b_better >= threshold:
        return "Winner"
    if prob_b_better <= 1 - threshold:
        return "Loser"
    return "Neutral"

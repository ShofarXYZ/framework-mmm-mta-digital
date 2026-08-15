"""Calculadoras de duração e tamanho de amostra (aba "Calculator | AB Test").

Fórmula clássica para teste de duas proporções:

    n_por_variacao = (z_{α/2}·√(2·p̄·(1−p̄)) + z_β·√(p1(1−p1) + p2(1−p2)))² / (p2 − p1)²

com p̄ = (p1 + p2)/2. Devolve o N POR VARIAÇÃO; o total multiplica pelo número
de variações (controle incluído).
"""

from __future__ import annotations

import math

from scipy.stats import norm

CONFIDENCE_LEVELS = {"80%": 0.80, "90%": 0.90, "95%": 0.95, "99%": 0.99}
POWER_LEVELS = {"80%": 0.80, "90%": 0.90}


def z_alpha(confidence: float, two_sided: bool = True) -> float:
    alpha = 1 - confidence
    return float(norm.ppf(1 - alpha / 2)) if two_sided else float(norm.ppf(1 - alpha))


def z_beta(power: float) -> float:
    return float(norm.ppf(power))


def sample_size_per_variation(
    baseline_rate: float,
    uplift_pct: float,
    confidence: float = 0.95,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """Visitantes necessários POR variação para detectar o uplift informado."""
    p1 = float(baseline_rate)
    p2 = p1 * (1 + uplift_pct / 100.0)
    p2 = min(max(p2, 1e-9), 0.999999)
    if p1 <= 0 or p1 >= 1 or abs(p2 - p1) < 1e-12:
        return float("inf")

    p_bar = (p1 + p2) / 2
    za, zb = z_alpha(confidence, two_sided), z_beta(power)
    numerator = (
        za * math.sqrt(2 * p_bar * (1 - p_bar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return numerator / ((p2 - p1) ** 2)


def duration_estimation(
    n_variations: int,
    users_per_day: float,
    traffic_allocation_pct: float,
    baseline_rate: float,
    uplift_pct: float,
    confidence: float = 0.95,
    power: float = 0.80,
) -> dict:
    """(a) Duration Estimation — quantos dias o teste precisa rodar."""
    per_variation = sample_size_per_variation(baseline_rate, uplift_pct, confidence, power)
    total_needed = per_variation * max(int(n_variations), 2)
    daily_in_test = float(users_per_day) * (float(traffic_allocation_pct) / 100.0)
    days = total_needed / daily_in_test if daily_in_test > 0 else float("inf")
    return {
        "amostra_por_variacao": per_variation,
        "amostra_total": total_needed,
        "usuarios_dia_no_teste": daily_in_test,
        "dias": days,
        "semanas": days / 7 if math.isfinite(days) else float("inf"),
    }


def midrange_impact(
    total_sessions: float,
    allocation_pct: float,
    total_conversions: float,
    uplift_pct: float,
) -> dict:
    """(b) Mid-range Impact Estimation — conversões incrementais esperadas."""
    sessions_in_test = float(total_sessions) * (float(allocation_pct) / 100.0)
    baseline_rate = (total_conversions / total_sessions) if total_sessions else 0.0
    baseline_conversions = sessions_in_test * baseline_rate
    projected_rate = baseline_rate * (1 + uplift_pct / 100.0)
    projected_conversions = sessions_in_test * projected_rate
    return {
        "sessoes_no_teste": sessions_in_test,
        "taxa_atual": baseline_rate,
        "taxa_projetada": projected_rate,
        "conversoes_base": baseline_conversions,
        "conversoes_projetadas": projected_conversions,
        "conversoes_incrementais": projected_conversions - baseline_conversions,
    }


def sample_size_v2(
    baseline_rate: float,
    uplift_pct: float,
    n_variations: int,
    daily_visitors: float,
    confidence: float = 0.95,
    power: float = 0.80,
) -> dict:
    """(c) Sample Size / Duration V2 — visitantes necessários e dias."""
    per_variation = sample_size_per_variation(baseline_rate, uplift_pct, confidence, power)
    total = per_variation * max(int(n_variations), 2)
    days = total / daily_visitors if daily_visitors > 0 else float("inf")
    return {
        "amostra_por_variacao": per_variation,
        "amostra_total": total,
        "dias": days,
        "semanas": days / 7 if math.isfinite(days) else float("inf"),
        "mde_absoluto": baseline_rate * uplift_pct / 100.0,
    }


def mde_for_sample(
    baseline_rate: float, n_per_variation: float, confidence: float = 0.95, power: float = 0.80
) -> float:
    """Efeito mínimo detectável (%) para um tamanho de amostra dado — busca binária."""
    lo, hi = 0.01, 500.0
    for _ in range(80):
        mid = (lo + hi) / 2
        needed = sample_size_per_variation(baseline_rate, mid, confidence, power)
        if needed > n_per_variation:
            lo = mid
        else:
            hi = mid
    return hi

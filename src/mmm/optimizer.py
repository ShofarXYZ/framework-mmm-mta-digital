"""Otimização de budget sobre as curvas de resposta ajustadas pelo MMM.

Maximiza o `sales` previsto sujeito a:
  * soma dos investimentos = orçamento total disponível
  * bounds mínimo/máximo por canal (ex.: não zerar TV, teto em Influencer)

Implementado com scipy.optimize.minimize (SLSQP) sobre o vetor de investimento
por canal. Como adstock e saturação são não-lineares, cada avaliação da função
objetivo refaz o pipeline de transformação do MMM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.mmm.model import MMMResult, total_sales


def current_allocation(result: MMMResult) -> pd.Series:
    """Investimento total observado por canal no período do dataset."""
    return result.data[result.config.media_columns].sum().astype(float)


def _sales_for_spend(result: MMMResult, spend: np.ndarray, base: pd.Series) -> float:
    """Converte um vetor de investimento absoluto em multiplicadores e prevê o sales."""
    scalers = {}
    for i, col in enumerate(base.index):
        b = float(base.iloc[i])
        scalers[col] = float(spend[i] / b) if b > 0 else 0.0
    return total_sales(result, scalers)


def optimize_budget(
    result: MMMResult,
    total_budget: float,
    bounds: dict[str, tuple[float, float]],
    max_iter: int = 120,
) -> dict:
    """Roda o SLSQP e devolve alocação atual vs. ótima + lift esperado.

    Args:
        result: modelo ajustado na Página 2.
        total_budget: orçamento total a distribuir (mesma unidade do dataset).
        bounds: {canal: (min, max)} em valor absoluto de investimento.
    """
    base = current_allocation(result)
    channels = list(base.index)

    lo = np.array([max(0.0, bounds.get(c, (0.0, np.inf))[0]) for c in channels], dtype=float)
    hi = np.array([bounds.get(c, (0.0, np.inf))[1] for c in channels], dtype=float)
    hi = np.maximum(hi, lo)

    # Viabilidade: o orçamento precisa caber entre a soma dos mínimos e a dos máximos.
    feasible = True
    message = ""
    if total_budget < lo.sum():
        feasible = False
        message = "Orçamento total é menor que a soma dos mínimos por canal."
    elif total_budget > hi.sum():
        feasible = False
        message = "Orçamento total é maior que a soma dos máximos por canal."

    baseline_sales = _sales_for_spend(result, base.to_numpy(), base)

    if not feasible:
        return {
            "ok": False,
            "message": message,
            "table": pd.DataFrame(
                {"canal": channels, "alocacao_atual": base.to_numpy(), "alocacao_otima": np.nan}
            ),
            "baseline_sales": baseline_sales,
            "optimal_sales": np.nan,
            "lift_pct": np.nan,
        }

    # Ponto de partida: alocação atual reescalada para o orçamento e clipada nos bounds.
    x0 = base.to_numpy(dtype=float)
    x0 = x0 * (total_budget / x0.sum()) if x0.sum() > 0 else np.full(len(channels), total_budget / len(channels))
    x0 = np.clip(x0, lo, hi)
    if x0.sum() > 0:
        x0 = x0 * (total_budget / x0.sum())
        x0 = np.clip(x0, lo, hi)

    scale = max(baseline_sales, 1.0)

    # A otimização roda em FRAÇÕES do orçamento (variáveis ~O(1)). Em valores
    # absolutos (~1e7) os gradientes por diferenças finitas ficam da ordem de
    # 1e-9 e o SLSQP para no ponto inicial achando que já convergiu.
    def negative_sales(frac: np.ndarray) -> float:
        spend = np.clip(frac, 0, None) * total_budget
        return -_sales_for_spend(result, spend, base) / scale

    constraints = [{"type": "eq", "fun": lambda f: float(np.sum(f) - 1.0)}]
    frac_lo, frac_hi = lo / total_budget, hi / total_budget

    try:
        res = minimize(
            negative_sales,
            x0 / total_budget,
            method="SLSQP",
            bounds=list(zip(frac_lo, frac_hi)),
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": 1e-10, "eps": 1e-4},
        )
        optimal = np.clip(res.x * total_budget, lo, hi)
        converged = bool(res.success)
        message = str(res.message)
    except Exception as exc:  # nunca derrubar a página por falha numérica
        optimal = x0
        converged = False
        message = f"Falha na otimização ({exc}). Exibindo a alocação inicial."

    # Reprojeta na restrição de orçamento (SLSQP pode violar levemente a igualdade).
    if optimal.sum() > 0:
        optimal = np.clip(optimal * (total_budget / optimal.sum()), lo, hi)

    optimal_sales = _sales_for_spend(result, optimal, base)
    lift = 100 * (optimal_sales - baseline_sales) / baseline_sales if baseline_sales else np.nan

    table = pd.DataFrame(
        {
            "canal": channels,
            "alocacao_atual": base.to_numpy(),
            "alocacao_otima": optimal,
        }
    )
    table["delta"] = table["alocacao_otima"] - table["alocacao_atual"]
    table["delta_%"] = np.where(
        table["alocacao_atual"] > 0, 100 * table["delta"] / table["alocacao_atual"], np.nan
    )
    table["share_atual_%"] = 100 * table["alocacao_atual"] / table["alocacao_atual"].sum()
    table["share_otimo_%"] = 100 * table["alocacao_otima"] / max(table["alocacao_otima"].sum(), 1e-9)

    return {
        "ok": True,
        "converged": converged,
        "message": message,
        "table": table,
        "baseline_sales": baseline_sales,
        "optimal_sales": optimal_sales,
        "lift_pct": lift,
    }


def what_if(result: MMMResult, deltas_pct: dict[str, float]) -> dict:
    """Cenário 'E se': aplica +/- % por canal sem re-otimizar (resposta instantânea)."""
    scalers = {col: 1.0 + pct / 100.0 for col, pct in deltas_pct.items()}
    baseline = total_sales(result)
    scenario = total_sales(result, scalers)
    base_spend = current_allocation(result)
    new_spend = pd.Series({col: base_spend[col] * scalers.get(col, 1.0) for col in base_spend.index})
    return {
        "baseline_sales": baseline,
        "scenario_sales": scenario,
        "delta_sales": scenario - baseline,
        "lift_pct": 100 * (scenario - baseline) / baseline if baseline else np.nan,
        "baseline_spend": float(base_spend.sum()),
        "scenario_spend": float(new_spend.sum()),
        "spend_by_channel": pd.DataFrame(
            {"canal": base_spend.index, "atual": base_spend.to_numpy(), "cenario": new_spend.to_numpy()}
        ),
    }

"""Geração de recomendações acionáveis a partir dos modelos.

Traduz a saída estatística do MMM e do MTA na pergunta que o analista realmente
faz: **onde investir mais, onde ter cautela, quanto mover e quanto economizar.**

As regras são explícitas e auditáveis — nada de "caixa-preta":

MMM (`mmm_recommendation`)
  * Ordena os canais pelo **retorno marginal no nível atual de investimento**,
    lido da curva de resposta (não pelo ROI médio, que ignora saturação).
  * Canal a INVESTIR = maior retorno marginal, ou seja, o próximo real investido
    ali rende mais do que em qualquer outro canal.
  * Canal de ATENÇÃO = menor retorno marginal (ou coeficiente negativo/saturado):
    é de onde o dinheiro deve sair primeiro.
  * O movimento sugerido é conservador: realoca uma fração do investimento do
    canal de atenção, e o impacto é medido rodando o cenário no próprio modelo —
    não é regra de três.

MTA (`mta_recommendation`)
  * Compara o crédito do **last-click** com o dos modelos que enxergam a jornada
    inteira (linear/posicional ou Markov/Shapley quando disponíveis).
  * Canal a INVESTIR = o mais SUBVALORIZADO pelo last-click (perde crédito no
    relatório padrão e por isso costuma sofrer corte indevido de budget).
  * Canal de ATENÇÃO = o mais SUPERVALORIZADO, que tende a receber budget além
    do que a jornada justifica.
  * O CPA implícito por modelo entra como evidência de eficiência.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import label
from src.mmm.model import MMMResult, contribution_summary, response_curve, total_sales

# Fração do investimento do canal de atenção que a recomendação propõe realocar.
# Deliberadamente conservadora: mover pouco, medir, e só então mover mais.
REALLOCATION_FRACTION = 0.20


# ---------------------------------------------------------------------------
# MMM
# ---------------------------------------------------------------------------
def marginal_returns(result: MMMResult, points: int = 9) -> pd.DataFrame:
    """Retorno marginal de cada canal no nível ATUAL de investimento.

    Lê a inclinação da curva de resposta em torno do multiplicador 1.0 — é a
    resposta para "o próximo real investido aqui gera quanto de venda?".
    """
    rows = []
    for channel in result.config.media_columns:
        try:
            curve = response_curve(result, channel, max_multiplier=2.0, points=points)
        except Exception:
            continue
        spend = float(result.data[channel].sum())
        near = curve[(curve["multiplicador"] >= 0.75) & (curve["multiplicador"] <= 1.25)]
        marginal = float(near["retorno_marginal"].mean()) if len(near) else np.nan

        # saturação: quanto o retorno marginal cai ao dobrar o investimento
        at_2x = curve.loc[(curve["multiplicador"] - 2).abs().idxmin(), "retorno_marginal"]
        saturation = 1 - (float(at_2x) / marginal) if marginal and np.isfinite(marginal) and marginal > 0 else np.nan

        contribution = float(result.contributions[channel].sum())
        rows.append(
            {
                "canal": channel,
                "canal_label": label(channel),
                "investimento": spend,
                "contribuicao": contribution,
                "roi_medio": contribution / spend if spend > 0 else np.nan,
                "retorno_marginal": marginal,
                "saturacao": saturation,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("retorno_marginal", ascending=False).reset_index(drop=True)


def mmm_recommendation(result: MMMResult, fraction: float = REALLOCATION_FRACTION) -> dict:
    """Recomendação de alocação a partir do MMM ajustado."""
    table = marginal_returns(result)
    valid = table.dropna(subset=["retorno_marginal"])
    if len(valid) < 2:
        return {"ok": False, "message": "Não há canais suficientes com curva de resposta válida."}

    best = valid.iloc[0]
    worst = valid.iloc[-1]

    if best["canal"] == worst["canal"]:
        return {"ok": False, "message": "Apenas um canal com resposta mensurável."}

    # Quanto tirar do canal de atenção e colocar no canal de oportunidade
    amount = float(worst["investimento"]) * fraction
    scalers = {
        worst["canal"]: 1 - fraction,
        best["canal"]: 1 + (amount / float(best["investimento"]) if best["investimento"] > 0 else 0.0),
    }

    baseline = total_sales(result)
    scenario = total_sales(result, scalers)
    delta = scenario - baseline
    lift = 100 * delta / baseline if baseline else np.nan

    plan = pd.DataFrame(
        [
            {
                "movimento": "➕ Investir mais",
                "canal": best["canal_label"],
                "valor": amount,
                "investimento_atual": float(best["investimento"]),
                "investimento_sugerido": float(best["investimento"]) + amount,
                "retorno_marginal": float(best["retorno_marginal"]),
                "por_que": "Maior retorno marginal — ainda não saturou, o próximo real rende mais aqui.",
            },
            {
                "movimento": "➖ Economizar",
                "canal": worst["canal_label"],
                "valor": -amount,
                "investimento_atual": float(worst["investimento"]),
                "investimento_sugerido": float(worst["investimento"]) - amount,
                "retorno_marginal": float(worst["retorno_marginal"]),
                "por_que": "Menor retorno marginal — é de onde o dinheiro deve sair primeiro.",
            },
        ]
    )

    saturated = valid[valid["saturacao"] > 0.6]["canal_label"].tolist()

    return {
        "ok": True,
        "invest": best["canal_label"],
        "invest_marginal": float(best["retorno_marginal"]),
        "watch": worst["canal_label"],
        "watch_marginal": float(worst["retorno_marginal"]),
        "amount": amount,
        "fraction": fraction,
        "delta_sales": delta,
        "lift_pct": lift,
        "plan": plan,
        "table": table,
        "saturated": saturated,
        "headline": (
            f"Com base nestes dados, o melhor cenário é **investir em {best['canal_label']}** "
            f"e **atenção com {worst['canal_label']}**."
        ),
        "detail": (
            f"Cada real adicional em {best['canal_label']} gera hoje "
            f"{best['retorno_marginal']:.2f} de venda, contra {worst['retorno_marginal']:.2f} em "
            f"{worst['canal_label']} — uma diferença de "
            f"{best['retorno_marginal'] / worst['retorno_marginal']:.1f}x"
            if worst["retorno_marginal"] > 0
            else f"Cada real adicional em {best['canal_label']} gera {best['retorno_marginal']:.2f} de venda, "
            f"enquanto {worst['canal_label']} já não responde ao investimento"
        ),
    }


def optimizer_recommendation(opt: dict) -> dict:
    """Traduz a saída do otimizador em 'onde colocar' e 'onde economizar'."""
    if not opt.get("ok"):
        return {"ok": False, "message": opt.get("message", "Otimização indisponível.")}

    table = opt["table"].copy()
    table["canal_label"] = table["canal"].map(label)
    gains = table[table["delta"] > 0].sort_values("delta", ascending=False)
    cuts = table[table["delta"] < 0].sort_values("delta")

    return {
        "ok": True,
        "invest": gains.iloc[0]["canal_label"] if len(gains) else None,
        "invest_amount": float(gains.iloc[0]["delta"]) if len(gains) else 0.0,
        "watch": cuts.iloc[0]["canal_label"] if len(cuts) else None,
        "watch_amount": float(abs(cuts.iloc[0]["delta"])) if len(cuts) else 0.0,
        "gains": gains[["canal_label", "alocacao_atual", "alocacao_otima", "delta", "delta_%"]],
        "cuts": cuts[["canal_label", "alocacao_atual", "alocacao_otima", "delta", "delta_%"]],
        "total_realocado": float(gains["delta"].sum()) if len(gains) else 0.0,
        "lift_pct": opt["lift_pct"],
    }


# ---------------------------------------------------------------------------
# MTA
# ---------------------------------------------------------------------------
def mta_recommendation(
    shares: pd.DataFrame, adspend: pd.Series | None = None, reference_models: list[str] | None = None
) -> dict:
    """Recomendação de alocação a partir da comparação entre modelos de atribuição.

    Args:
        shares: tabela canal × modelo com o % de crédito (saída de `to_share`).
        adspend: investimento por canal, para calcular o CPA implícito.
        reference_models: modelos considerados a "leitura justa" da jornada.
            Usa Markov e Shapley quando existirem; senão, Linear e Position-Based.
    """
    if shares.empty or "Last-Click" not in shares.columns:
        return {"ok": False, "message": "Tabela de atribuição incompleta."}

    if reference_models is None:
        algorithmic = [m for m in ("Markov", "Shapley") if m in shares.columns]
        reference_models = algorithmic or [
            m for m in ("Linear", "Position-Based (U)") if m in shares.columns
        ]
    if not reference_models:
        return {"ok": False, "message": "Nenhum modelo de referência disponível para comparar."}

    fair = shares[reference_models].mean(axis=1)
    gap = (fair - shares["Last-Click"]).sort_values(ascending=False)

    invest, watch = gap.index[0], gap.index[-1]

    table = pd.DataFrame(
        {
            "canal": shares.index,
            "credito_last_click_%": shares["Last-Click"].to_numpy(),
            "credito_justo_%": fair.to_numpy(),
            "gap_pp": (fair - shares["Last-Click"]).to_numpy(),
        }
    )
    if adspend is not None and len(adspend):
        table["investimento"] = adspend.reindex(shares.index).fillna(0.0).to_numpy()
        total_spend = float(table["investimento"].sum())
        # Quanto o budget mudaria se seguisse o crédito justo em vez do last-click
        table["budget_pelo_last_click"] = total_spend * table["credito_last_click_%"] / 100
        table["budget_pelo_credito_justo"] = total_spend * table["credito_justo_%"] / 100
        table["realocar"] = table["budget_pelo_credito_justo"] - table["budget_pelo_last_click"]
    table = table.sort_values("gap_pp", ascending=False).reset_index(drop=True)

    amount = float(table.loc[table["canal"] == invest, "realocar"].iloc[0]) if "realocar" in table else np.nan
    save = float(abs(table.loc[table["canal"] == watch, "realocar"].iloc[0])) if "realocar" in table else np.nan

    return {
        "ok": True,
        "invest": invest,
        "watch": watch,
        "gap_invest_pp": float(gap.iloc[0]),
        "gap_watch_pp": float(abs(gap.iloc[-1])),
        "amount": amount,
        "save": save,
        "table": table,
        "reference_models": reference_models,
        "headline": (
            f"Com base nestes dados, o melhor cenário é **investir em {invest}** "
            f"e **atenção com {watch}**."
        ),
        "detail": (
            f"{invest} recebe {gap.iloc[0]:.1f} p.p. a MENOS no last-click do que na leitura de "
            f"{' e '.join(reference_models)} — está subvalorizado no relatório padrão e é candidato "
            f"a corte indevido de budget. {watch} está no sentido oposto, com "
            f"{abs(gap.iloc[-1]):.1f} p.p. a mais."
        ),
    }

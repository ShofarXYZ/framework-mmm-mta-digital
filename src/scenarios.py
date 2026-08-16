"""Simulações de cenário para as páginas guiadas (público leigo).

Duas perguntas, dois horizontes:

* **MMM — estratégico.** "Se eu tiver R$ X para o próximo trimestre/semestre/ano,
  em quais mídias coloco?" A resposta sai do modelo de MMM ajustado: aplicamos o
  investimento adicional sobre o padrão das últimas N semanas e rodamos a
  predição de novo. Adstock e saturação entram na conta, então o modelo sabe que
  dobrar o investimento **não** dobra o retorno.

* **MTA — tático.** "Se eu tiver R$ X para este mês/semana/dia, em quais canais
  digitais coloco?" Aqui não existe curva de resposta (o dataset de MTA é por
  cliente, não série temporal), então trabalhamos com o **CPA implícito** de cada
  canal, obtido do modelo de atribuição, com um desconto de eficiência quando o
  investimento foge muito do patamar histórico.

Toda simplificação está documentada e é exibida na tela — nenhuma conta acontece
escondida do usuário.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
    CONTROL_COLUMNS,
    MEDIA_CHANNELS,
    clean_mmm,
    label,
    load_mmm_raw,
    promo_dummy_columns,
)
from src.mmm.model import MMMConfig, MMMResult, fit_mmm, predict_df
from src.mmm.transforms import default_hill_params

# ---------------------------------------------------------------------------
# Horizontes
# ---------------------------------------------------------------------------
# MMM é decisão de calendário: planejamento anual, revisão semestral/trimestral.
MMM_HORIZONS = {
    "Bimestre (8 semanas)": 8,
    "Trimestre (13 semanas)": 13,
    "Semestre (26 semanas)": 26,
    "Ano (52 semanas)": 52,
}

# MTA é decisão de operação: o gestor de mídia mexe no lance hoje, na campanha
# esta semana, no plano do mês.
MTA_HORIZONS = {
    "Diário (1 dia)": 1,
    "Semanal (7 dias)": 7,
    "Mensal (30 dias)": 30,
}

BUDGET_MIN = 1_000
BUDGET_MAX = 1_000_000


# ---------------------------------------------------------------------------
# Modelo padrão (para quem chega sem passar pela página de modelagem)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparando o modelo de MMM...")
def default_mmm_result() -> MMMResult:
    """Ajusta um MMM com configuração padrão sensata.

    Existe para que o público leigo não precise configurar adstock, saturação e
    regularização antes de ver qualquer coisa. Quem quiser controlar isso usa a
    página **MMM Modelagem** — e o resultado de lá tem prioridade.
    """
    df = clean_mmm(load_mmm_raw(), {c: "Interpolação linear" for c in MEDIA_CHANNELS})
    cfg = MMMConfig(
        form="Linear",
        regularizer="Ridge",
        alpha=1.0,
        saturation_on=True,
        holdout_weeks=10,
        decays={c: (0.4 if c in ("tv_spend", "newspaper_spend") else 0.2) for c in MEDIA_CHANNELS},
        hill_params=default_hill_params(df, MEDIA_CHANNELS),
        media_columns=MEDIA_CHANNELS,
        control_columns=CONTROL_COLUMNS + promo_dummy_columns(df),
    )
    return fit_mmm(df, cfg)


def get_mmm_result() -> tuple[MMMResult, bool]:
    """Modelo da sessão se existir; senão, o padrão. Retorna (modelo, é_padrão)."""
    existing = st.session_state.get("mmm_result")
    if existing is not None:
        return existing, False
    return default_mmm_result(), True


# ---------------------------------------------------------------------------
# Cenário MMM
# ---------------------------------------------------------------------------
def _incremental_sales(result: MMMResult, extra: dict[str, float], weeks: int) -> float:
    """Vendas incrementais ao somar `extra` de investimento nas últimas `weeks` semanas."""
    df = result.data.reset_index(drop=True)
    n = len(df)
    window = np.arange(max(n - weeks, 0), n)

    scenario = df.copy()
    for channel, amount in extra.items():
        if channel in scenario.columns and amount:
            scenario.loc[window, channel] = (
                scenario.loc[window, channel].astype(float) + amount / len(window)
            )

    base_pred = predict_df(result, df)
    new_pred = predict_df(result, scenario)
    return float(np.sum(new_pred[window]) - np.sum(base_pred[window]))


def _greedy_allocation(
    result: MMMResult, budget: float, channels: list[str], weeks: int,
    steps: int = 12, max_share: float = 0.5,
) -> dict[str, float]:
    """Distribui o orçamento em fatias, sempre no canal que rende mais na margem.

    É a versão "gulosa" da otimização: como cada canal satura, o melhor destino
    da próxima fatia muda conforme o dinheiro entra.

    `max_share` é um **teto de concentração**: nenhum canal recebe mais que essa
    fração do orçamento. Não é matemática do modelo, é gestão de risco — com
    verbas pequenas diante do investimento histórico, a saturação mal aparece e a
    solução puramente ótima manda 100% num canal só. Concentrar tudo em uma mídia
    deixa o resultado refém de um algoritmo, de um leilão e de um formato. O teto
    é exposto na tela para que o usuário decida o quanto quer concentrar.
    """
    allocation = {c: 0.0 for c in channels}
    cap = budget * max_share if len(channels) > 1 else budget
    chunk = budget / steps
    for _ in range(steps):
        best_channel, best_gain = None, -np.inf
        for channel in channels:
            if allocation[channel] + chunk > cap + 1e-9:
                continue  # canal já atingiu o teto de concentração
            trial = dict(allocation)
            trial[channel] += chunk
            gain = _incremental_sales(result, trial, weeks)
            if gain > best_gain:
                best_channel, best_gain = channel, gain
        if best_channel is None:  # todos no teto: reparte o que sobrou
            remaining = budget - sum(allocation.values())
            for channel in channels:
                allocation[channel] += remaining / len(channels)
            break
        allocation[best_channel] += chunk
    return allocation


ALLOCATION_STRATEGIES = [
    "Deixar o modelo decidir (recomendado)",
    "Seguir o investimento atual",
    "Dividir igualmente",
]


def mmm_scenario(
    result: MMMResult,
    budget: float,
    channels: list[str],
    horizon_weeks: int,
    strategy: str = ALLOCATION_STRATEGIES[0],
    max_share: float = 0.5,
) -> dict:
    """Simula um investimento ADICIONAL de `budget` no horizonte escolhido."""
    channels = [c for c in channels if c in result.config.media_columns]
    if not channels or budget <= 0:
        return {"ok": False, "message": "Escolha ao menos uma mídia e um orçamento maior que zero."}

    historical = result.data[channels].sum()

    if strategy == "Dividir igualmente":
        allocation = {c: budget / len(channels) for c in channels}
    elif strategy == "Seguir o investimento atual":
        total = float(historical.sum())
        allocation = (
            {c: budget * float(historical[c]) / total for c in channels}
            if total > 0
            else {c: budget / len(channels) for c in channels}
        )
    else:
        allocation = _greedy_allocation(result, budget, channels, horizon_weeks,
                                        max_share=max_share)

    total_incremental = _incremental_sales(result, allocation, horizon_weeks)

    rows = []
    for channel in channels:
        amount = allocation[channel]
        alone = _incremental_sales(result, {channel: amount}, horizon_weeks) if amount > 0 else 0.0
        rows.append(
            {
                "canal": channel,
                "canal_label": label(channel),
                "investimento": amount,
                "share_%": 100 * amount / budget if budget else 0.0,
                "vendas_incrementais": alone,
                "retorno_por_real": alone / amount if amount > 0 else np.nan,
            }
        )
    table = pd.DataFrame(rows).sort_values("investimento", ascending=False).reset_index(drop=True)

    return {
        "ok": True,
        "budget": budget,
        "horizon_weeks": horizon_weeks,
        "strategy": strategy,
        "max_share": max_share,
        "allocation": allocation,
        "table": table,
        "vendas_incrementais": total_incremental,
        "retorno_por_real": total_incremental / budget if budget else np.nan,
        "lucro_estimado": total_incremental - budget,
        "melhor_canal": table.iloc[0]["canal_label"] if len(table) else None,
        "melhor_retorno": float(table["retorno_por_real"].max()) if len(table) else np.nan,
    }


# ---------------------------------------------------------------------------
# Cenário MTA
# ---------------------------------------------------------------------------
# O histórico do dataset de MTA não tem datas. Assumimos, de forma explícita e
# ajustável na tela, que ele representa um mês de operação — é o que permite
# falar em "por dia", "por semana" e "por mês".
DEFAULT_HISTORY_DAYS = 30

# Desconto de eficiência: quanto mais o investimento foge do patamar histórico do
# canal, mais caro fica cada conversão adicional. Sem isso, a conta viraria regra
# de três e prometeria conversões infinitas.
EFFICIENCY_K = 0.5


def mta_scenario(
    credit: pd.Series,
    adspend: pd.Series,
    budget: float,
    channels: list[str],
    horizon_days: int,
    history_days: int = DEFAULT_HISTORY_DAYS,
    allocation_mode: str = "Pelo melhor CPA (recomendado)",
) -> dict:
    """Simula um investimento no horizonte tático (dia/semana/mês).

    A verba simulada é tratada como investimento **adicional** ao que já roda
    hoje — por isso as conversões estimadas são um acréscimo, não o total da
    operação.

    Args:
        credit: conversões creditadas por canal (saída de qualquer modelo de atribuição).
        adspend: investimento histórico por canal.
        budget: verba ADICIONAL a distribuir no horizonte.
        channels: canais que vão receber investimento.
        horizon_days: 1, 7 ou 30.
        history_days: a quantos dias o histórico do dataset equivale.
    """
    channels = [c for c in channels if c in credit.index]
    if not channels or budget <= 0:
        return {"ok": False, "message": "Escolha ao menos um canal e um orçamento maior que zero."}

    base = pd.DataFrame(
        {
            "canal": channels,
            "conversoes_hist": credit.reindex(channels).fillna(0.0).to_numpy(),
            "investimento_hist": adspend.reindex(channels).fillna(0.0).to_numpy(),
        }
    )
    base["cpa_hist"] = np.where(
        base["conversoes_hist"] > 0, base["investimento_hist"] / base["conversoes_hist"], np.nan
    )
    # Patamar histórico reescalado para o horizonte escolhido
    base["investimento_horizonte"] = base["investimento_hist"] * horizon_days / max(history_days, 1)

    valid = base.dropna(subset=["cpa_hist"])
    if valid.empty:
        return {"ok": False, "message": "Não há CPA calculável para os canais escolhidos."}

    if allocation_mode == "Dividir igualmente":
        weights = pd.Series(1.0, index=base.index)
    elif allocation_mode == "Seguir o investimento atual":
        weights = base["investimento_hist"].clip(lower=0)
    else:  # pelo melhor CPA: quanto menor o custo por conversão, mais verba
        inverse = 1 / base["cpa_hist"].replace(0, np.nan)
        weights = inverse.fillna(0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=base.index)
    base["investimento"] = budget * weights / weights.sum()

    # Eficiência decrescente conforme o investimento supera o patamar histórico
    intensity = np.where(
        base["investimento_horizonte"] > 0,
        base["investimento"] / base["investimento_horizonte"],
        1.0,
    )
    base["intensidade"] = intensity
    base["eficiencia"] = 1 / (1 + EFFICIENCY_K * np.clip(intensity - 1, 0, None))
    base["cpa_estimado"] = base["cpa_hist"] / base["eficiencia"]
    base["conversoes_estimadas"] = np.where(
        base["cpa_estimado"] > 0, base["investimento"] / base["cpa_estimado"], 0.0
    )
    base["canal_label"] = base["canal"]

    total_conversions = float(base["conversoes_estimadas"].sum())
    blended_cpa = budget / total_conversions if total_conversions > 0 else np.nan

    # Baseline do horizonte: o que os canais escolhidos já entregam hoje nesse
    # mesmo intervalo de tempo. É o que dá sentido a "por dia" / "por semana".
    scale = horizon_days / max(history_days, 1)
    base["conversoes_horizonte_hoje"] = base["conversoes_hist"] * scale
    baseline_conversions = float(base["conversoes_horizonte_hoje"].sum())
    baseline_spend = float(base["investimento_horizonte"].sum())

    ranked = base.sort_values("cpa_hist")
    stretched = base[base["intensidade"] > 2]["canal"].tolist()

    return {
        "ok": True,
        "budget": budget,
        "horizon_days": horizon_days,
        "history_days": history_days,
        "table": base.sort_values("investimento", ascending=False).reset_index(drop=True),
        "conversoes_estimadas": total_conversions,
        "conversoes_por_dia": total_conversions / max(horizon_days, 1),
        "conversoes_hoje_no_horizonte": baseline_conversions,
        "investimento_hoje_no_horizonte": baseline_spend,
        # A verba simulada é ADICIONAL ao que já roda hoje, então o acréscimo é
        # medido como um percentual SOBRE a operação atual do mesmo período.
        "acrescimo_vs_hoje_%": (
            100 * total_conversions / baseline_conversions if baseline_conversions > 0 else np.nan
        ),
        "cpa_medio": blended_cpa,
        "melhor_cpa_canal": ranked.iloc[0]["canal"] if len(ranked) else None,
        "melhor_cpa": float(ranked.iloc[0]["cpa_hist"]) if len(ranked) else np.nan,
        "pior_cpa_canal": ranked.iloc[-1]["canal"] if len(ranked) else None,
        "pior_cpa": float(ranked.iloc[-1]["cpa_hist"]) if len(ranked) else np.nan,
        "canais_esticados": stretched,
    }


MTA_ALLOCATION_MODES = [
    "Pelo melhor CPA (recomendado)",
    "Seguir o investimento atual",
    "Dividir igualmente",
]

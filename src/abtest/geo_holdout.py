"""Geo-Holdout simulado + diferença-em-diferenças (DiD), ligado ao MMM.

LIMITAÇÃO ASSUMIDA: `mmm_dataset.csv` não tem recorte geográfico. Um geo-holdout
real dividiria mercados em teste/controle. Aqui simulamos um **holdout temporal**:
o usuário escolhe um canal, uma janela de semanas e um % de corte de investimento.

Três séries entram na conta:
  (a) `sales` REAL observado
  (b) contrafactual do MMM com o investimento CHEIO (cenário base)
  (c) contrafactual do MMM com o investimento REDUZIDO na janela (holdout)

Lift previsto pelo MMM = (b) − (c) na janela: quanto o modelo acha que o corte
custaria. DiD = [(real_pós − real_pré) − (base_pós − base_pré)]: o quanto o dado
observado se afasta do que o modelo previa. A divergência entre os dois é o
gatilho de recalibração do loop de governança.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.mmm.model import MMMResult, predict_df


def run_geo_holdout(
    result: MMMResult, channel: str, start_week: int, n_weeks: int, reduction_pct: float
) -> dict:
    """Executa o holdout simulado e o DiD.

    Args:
        result: modelo ajustado na Página 2.
        channel: coluna de spend a cortar (ex.: 'google_ads_spend').
        start_week: índice (0-based) da primeira semana da janela de holdout.
        n_weeks: duração da janela em semanas.
        reduction_pct: % de corte do investimento no canal durante a janela.
    """
    df = result.data.reset_index(drop=True)
    n = len(df)
    start = int(np.clip(start_week, 0, max(n - 2, 0)))
    end = int(np.clip(start + n_weeks, start + 1, n))
    window = np.arange(start, end)

    # Período pré: mesma duração imediatamente antes da janela (para o DiD).
    pre_len = min(len(window), start)
    pre = np.arange(start - pre_len, start) if pre_len > 0 else np.array([], dtype=int)

    # (c) cenário holdout: corta o investimento SOMENTE dentro da janela.
    df_holdout = df.copy()
    df_holdout.loc[window, channel] = df_holdout.loc[window, channel].astype(float) * (
        1 - reduction_pct / 100.0
    )

    base_pred = predict_df(result, df)         # (b)
    holdout_pred = predict_df(result, df_holdout)  # (c)
    real = df["sales"].astype(float).to_numpy()    # (a)

    real_post = float(real[window].sum())
    base_post = float(base_pred[window].sum())
    holdout_post = float(holdout_pred[window].sum())

    real_pre = float(real[pre].sum()) if len(pre) else np.nan
    base_pre = float(base_pred[pre].sum()) if len(pre) else np.nan

    # Lift que o MMM prevê para o corte (negativo = perda de vendas).
    lift_previsto_mmm = 100 * (holdout_post - base_post) / base_post if base_post else np.nan

    # DiD: diferença observada vs. diferença prevista pelo modelo entre pré e pós.
    if len(pre):
        did_absoluto = (real_post - real_pre) - (base_post - base_pre)
        did_pct = 100 * did_absoluto / base_post if base_post else np.nan
    else:
        did_absoluto = did_pct = np.nan

    # Lift incremental "medido": o quanto o real ficou acima/abaixo do contrafactual.
    lift_medido = 100 * (real_post - base_post) / base_post if base_post else np.nan

    divergencia = abs(lift_medido - lift_previsto_mmm) if np.isfinite(lift_medido) else np.nan
    if not np.isfinite(divergencia):
        alerta, nivel = "Sem dados suficientes para avaliar a divergência.", "info"
    elif divergencia > 15:
        alerta = (
            "🔴 Recalibrar o modelo — a diferença entre o lift medido no holdout e o previsto "
            f"pelo MMM é de {divergencia:.1f} p.p. O MMM pode estar super ou subestimando este canal."
        )
        nivel = "error"
    elif divergencia > 7:
        alerta = (
            f"🟡 Atenção — divergência de {divergencia:.1f} p.p. entre holdout e MMM. "
            "Vale rodar um teste confirmatório antes de mudar a alocação."
        )
        nivel = "warning"
    else:
        alerta = (
            f"🟢 MMM calibrado para este canal — divergência de apenas {divergencia:.1f} p.p. "
            "entre o lift medido e o previsto."
        )
        nivel = "success"

    series = pd.DataFrame(
        {
            "date": df["date"],
            "real": real,
            "contrafactual_base": base_pred,
            "cenario_holdout": holdout_pred,
            "janela": np.isin(np.arange(n), window),
            "pre": np.isin(np.arange(n), pre),
        }
    )

    return {
        "series": series,
        "canal": channel,
        "janela": (start, end),
        "semanas_janela": int(len(window)),
        "reducao_pct": float(reduction_pct),
        "real_post": real_post,
        "base_post": base_post,
        "holdout_post": holdout_post,
        "real_pre": real_pre,
        "base_pre": base_pre,
        "lift_previsto_mmm_%": lift_previsto_mmm,
        "lift_medido_%": lift_medido,
        "did_absoluto": did_absoluto,
        "did_%": did_pct,
        "divergencia_pp": divergencia,
        "alerta": alerta,
        "nivel": nivel,
        "investimento_cortado": float(
            df.loc[window, channel].sum() * reduction_pct / 100.0
        ),
        "vendas_perdidas_previstas": base_post - holdout_post,
    }


def did_table(res: dict) -> pd.DataFrame:
    """Tabela 2x2 clássica da diferença-em-diferenças."""
    return pd.DataFrame(
        {
            "Período": ["Pré-holdout", "Janela de holdout", "Δ (pós − pré)"],
            "Real observado": [
                res["real_pre"],
                res["real_post"],
                res["real_post"] - res["real_pre"],
            ],
            "Contrafactual MMM": [
                res["base_pre"],
                res["base_post"],
                res["base_post"] - res["base_pre"],
            ],
        }
    )

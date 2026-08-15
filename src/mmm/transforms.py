"""Transformações de mídia do MMM: adstock, saturação e formas funcionais.

Referência do briefing (4c):
  * Adstock geométrico:  x_t^ad = x_t + decay * x_{t-1}^ad
  * Saturação Hill:      y = x^s / (x^s + k^s)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FUNCTIONAL_FORMS = ["Linear", "Log-Linear", "Log-Log"]


# ---------------------------------------------------------------------------
# Adstock
# ---------------------------------------------------------------------------
def geometric_adstock(x: np.ndarray | pd.Series, decay: float) -> np.ndarray:
    """Adstock geométrico (carryover infinito com decaimento constante).

    decay=0 -> sem carryover (efeito só na semana do investimento).
    decay=0.7 -> 70% do efeito acumulado transborda para a semana seguinte.
    """
    values = np.asarray(x, dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    decay = float(np.clip(decay, 0.0, 0.95))
    out = np.zeros_like(values)
    carry = 0.0
    for i, v in enumerate(values):
        carry = v + decay * carry
        out[i] = carry
    return out


def apply_adstock(df: pd.DataFrame, decays: dict[str, float]) -> pd.DataFrame:
    """Aplica adstock coluna a coluna, devolvendo um novo dataframe."""
    out = df.copy()
    for col, decay in decays.items():
        if col in out.columns:
            out[col] = geometric_adstock(out[col], decay)
    return out


# ---------------------------------------------------------------------------
# Saturação
# ---------------------------------------------------------------------------
def hill_saturation(x: np.ndarray | pd.Series, half_sat: float, slope: float = 1.0) -> np.ndarray:
    """Curva de Hill: y = x^s / (x^s + k^s), retorno em [0, 1).

    half_sat (k): investimento em que se atinge 50% do efeito máximo.
    slope (s):    inclinação; s>1 gera formato de S (retornos crescentes no início).
    """
    values = np.asarray(x, dtype=float)
    values = np.clip(np.nan_to_num(values, nan=0.0), 0, None)
    k = max(float(half_sat), 1e-9)
    s = max(float(slope), 1e-6)
    num = np.power(values, s)
    return num / (num + np.power(k, s))


def apply_saturation(
    df: pd.DataFrame, params: dict[str, tuple[float, float]], enabled: bool = True
) -> pd.DataFrame:
    """Aplica Hill a cada coluna. `params` = {coluna: (half_sat, slope)}."""
    out = df.copy()
    if not enabled:
        return out
    for col, (half_sat, slope) in params.items():
        if col in out.columns:
            out[col] = hill_saturation(out[col], half_sat, slope)
    return out


def default_hill_params(df: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    """Chute inicial razoável: half-saturation = mediana histórica do canal, slope = 1."""
    params = {}
    for col in columns:
        if col in df.columns:
            median = float(pd.to_numeric(df[col], errors="coerce").median() or 1.0)
            params[col] = (max(median, 1.0), 1.0)
    return params


# ---------------------------------------------------------------------------
# Formas funcionais
# ---------------------------------------------------------------------------
def transform_target(y: np.ndarray | pd.Series, form: str) -> np.ndarray:
    """Aplica a transformação do lado esquerdo da equação (y)."""
    y = np.asarray(y, dtype=float)
    if form in ("Log-Linear", "Log-Log"):
        return np.log(np.clip(y, 1e-9, None))
    return y


def inverse_target(y: np.ndarray, form: str) -> np.ndarray:
    """Volta da escala do modelo para a escala original de `sales`."""
    y = np.asarray(y, dtype=float)
    if form in ("Log-Linear", "Log-Log"):
        return np.exp(np.clip(y, -50, 50))
    return y


def transform_features(X: pd.DataFrame, form: str, media_columns: list[str]) -> pd.DataFrame:
    """Aplica log(1+x) às colunas de mídia quando a forma funcional é Log-Log."""
    out = X.copy()
    if form == "Log-Log":
        for col in media_columns:
            if col in out.columns:
                out[col] = np.log1p(np.clip(out[col].astype(float), 0, None))
    return out


def build_media_matrix(
    df: pd.DataFrame,
    media_columns: list[str],
    decays: dict[str, float],
    hill_params: dict[str, tuple[float, float]],
    saturation_on: bool,
) -> pd.DataFrame:
    """Pipeline completo de mídia: adstock -> saturação. Devolve só as colunas de mídia."""
    media = df[media_columns].copy()
    media = apply_adstock(media, decays)
    if saturation_on:
        media = apply_saturation(media, hill_params, enabled=True)
    return media

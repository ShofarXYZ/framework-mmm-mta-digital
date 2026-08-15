"""Motor estatístico do MMM.

Fluxo: spend bruto -> adstock -> saturação (Hill) -> forma funcional -> regressão
regularizada (Ridge / Lasso / ElasticNet) -> decomposição de contribuições,
curvas de resposta e métricas em holdout temporal.

O modo Bayesiano (PyMC) é opcional e envolvido em try/except: se a lib não
estiver instalada o app avisa e continua no modo frequentista.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.mmm.transforms import (
    build_media_matrix,
    inverse_target,
    transform_features,
    transform_target,
)

REGULARIZERS = ["Ridge", "Lasso", "ElasticNet"]

try:  # pragma: no cover - dependência opcional
    import pymc as pm  # type: ignore
    import arviz as az  # type: ignore

    PYMC_AVAILABLE = True
except Exception:  # ImportError e também falhas de backend
    pm = None
    az = None
    PYMC_AVAILABLE = False


@dataclass
class MMMConfig:
    """Configuração vinda da sidebar da Página 2."""

    form: str = "Linear"
    regularizer: str = "Ridge"
    alpha: float = 1.0
    l1_ratio: float = 0.5
    saturation_on: bool = True
    holdout_weeks: int = 10
    decays: dict[str, float] = field(default_factory=dict)
    hill_params: dict[str, tuple[float, float]] = field(default_factory=dict)
    media_columns: list[str] = field(default_factory=list)
    control_columns: list[str] = field(default_factory=list)
    bayesian: bool = False


@dataclass
class MMMResult:
    """Tudo que as páginas 2, 3 e 8 precisam consumir do modelo ajustado."""

    config: MMMConfig
    data: pd.DataFrame
    feature_names: list[str]
    coefficients: pd.Series
    intercept: float
    scaler: Any
    model: Any
    metrics: dict[str, float]
    fitted: pd.Series          # predição em escala de `sales`, série completa
    contributions: pd.DataFrame  # colunas: date, Base, <canais>
    vif: pd.DataFrame
    posterior: Any = None      # arviz InferenceData quando em modo Bayesiano


# ---------------------------------------------------------------------------
# Construção da matriz de features
# ---------------------------------------------------------------------------
def _design_matrix(df: pd.DataFrame, cfg: MMMConfig, spend_scalers: dict[str, float] | None = None) -> pd.DataFrame:
    """Monta X aplicando (opcionalmente) um multiplicador de investimento por canal.

    `spend_scalers` é o que permite simular cenários (Página 3 e 8) sem refitar:
    escalamos o spend BRUTO e refazemos adstock+saturação, que são não-lineares.
    """
    raw = df.copy()
    if spend_scalers:
        for col, factor in spend_scalers.items():
            if col in raw.columns:
                raw[col] = raw[col].astype(float) * float(factor)

    media = build_media_matrix(
        raw, cfg.media_columns, cfg.decays, cfg.hill_params, cfg.saturation_on
    )
    controls = raw[[c for c in cfg.control_columns if c in raw.columns]].astype(float)
    X = pd.concat([media, controls], axis=1)
    X = transform_features(X, cfg.form, cfg.media_columns)
    return X.fillna(0.0)


def _make_estimator(cfg: MMMConfig):
    if cfg.regularizer == "Lasso":
        return Lasso(alpha=cfg.alpha, max_iter=20000)
    if cfg.regularizer == "ElasticNet":
        return ElasticNet(alpha=cfg.alpha, l1_ratio=cfg.l1_ratio, max_iter=20000)
    return Ridge(alpha=cfg.alpha)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF por variável (multicolinearidade). Fallback manual se statsmodels faltar."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        Xc = X.loc[:, X.std(ddof=0) > 0].copy()
        Xc = Xc.assign(_const=1.0)
        values = []
        for i, col in enumerate(Xc.columns):
            if col == "_const":
                continue
            try:
                values.append({"variavel": col, "VIF": float(variance_inflation_factor(Xc.values, i))})
            except Exception:
                values.append({"variavel": col, "VIF": np.nan})
        return pd.DataFrame(values).sort_values("VIF", ascending=False).reset_index(drop=True)
    except Exception:
        corr = X.corr().abs()
        return (
            pd.DataFrame({"variavel": corr.columns, "VIF": (1 / (1 - corr.pow(2).mean())).values})
            .sort_values("VIF", ascending=False)
            .reset_index(drop=True)
        )


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def fit_mmm(df: pd.DataFrame, cfg: MMMConfig) -> MMMResult:
    """Ajusta o MMM e devolve tudo que as páginas consomem."""
    df = df.sort_values("date").reset_index(drop=True)
    X = _design_matrix(df, cfg)
    y_raw = df["sales"].astype(float).to_numpy()
    y = transform_target(y_raw, cfg.form)

    holdout = int(np.clip(cfg.holdout_weeks, 4, max(4, len(df) // 3)))
    split = len(df) - holdout

    scaler = StandardScaler().fit(X.iloc[:split])
    Xs_train = scaler.transform(X.iloc[:split])
    Xs_all = scaler.transform(X)

    model = _make_estimator(cfg)
    model.fit(Xs_train, y[:split])

    posterior = None
    if cfg.bayesian and PYMC_AVAILABLE:
        posterior, bayes_coefs, bayes_intercept = _fit_bayesian(Xs_train, y[:split], list(X.columns))
        if bayes_coefs is not None:
            model.coef_ = bayes_coefs
            model.intercept_ = bayes_intercept

    pred_model_scale = model.predict(Xs_all)
    fitted = inverse_target(pred_model_scale, cfg.form)

    metrics = {
        "r2_treino": float(r2_score(y[:split], model.predict(Xs_train))),
        "r2_holdout": float(r2_score(y_raw[split:], fitted[split:])) if holdout > 1 else float("nan"),
        "mae_holdout": float(mean_absolute_error(y_raw[split:], fitted[split:])),
        "mape_holdout": _mape(y_raw[split:], fitted[split:]),
        "holdout_weeks": float(holdout),
    }
    n, p = split, X.shape[1]
    metrics["r2_ajustado_treino"] = (
        float(1 - (1 - metrics["r2_treino"]) * (n - 1) / max(n - p - 1, 1)) if n > p + 1 else float("nan")
    )

    coefficients = pd.Series(np.asarray(model.coef_, dtype=float), index=X.columns, name="coeficiente")

    result = MMMResult(
        config=cfg,
        data=df,
        feature_names=list(X.columns),
        coefficients=coefficients,
        intercept=float(model.intercept_),
        scaler=scaler,
        model=model,
        metrics=metrics,
        fitted=pd.Series(fitted, index=df.index, name="previsto"),
        contributions=pd.DataFrame(),
        vif=compute_vif(X),
        posterior=posterior,
    )
    result.contributions = decompose_contributions(result)
    return result


def _fit_bayesian(X: np.ndarray, y: np.ndarray, names: list[str]):
    """Ridge Bayesiano simples via PyMC. Devolve (idata, coefs médios, intercepto)."""
    try:  # pragma: no cover - depende de lib opcional
        with pm.Model():
            beta = pm.Normal("beta", mu=0.0, sigma=1.0, shape=X.shape[1])
            intercept = pm.Normal("intercept", mu=float(np.mean(y)), sigma=float(np.std(y) + 1e-6))
            sigma = pm.HalfNormal("sigma", sigma=float(np.std(y) + 1e-6))
            mu = intercept + pm.math.dot(X, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y)
            idata = pm.sample(600, tune=600, chains=2, progressbar=False, random_seed=42)
        coefs = idata.posterior["beta"].mean(dim=("chain", "draw")).values
        intercept_hat = float(idata.posterior["intercept"].mean().values)
        idata.attrs["coef_names"] = names
        return idata, coefs, intercept_hat
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# Predição em cenários
# ---------------------------------------------------------------------------
def predict_scenario(result: MMMResult, spend_scalers: dict[str, float] | None = None) -> np.ndarray:
    """Predição semanal de `sales` sob um cenário de multiplicadores de investimento."""
    X = _design_matrix(result.data, result.config, spend_scalers)
    X = X.reindex(columns=result.feature_names, fill_value=0.0)
    return inverse_target(result.model.predict(result.scaler.transform(X)), result.config.form)


def predict_df(result: MMMResult, df: pd.DataFrame) -> np.ndarray:
    """Predição semanal para um dataframe arbitrário de spend (usado no geo-holdout)."""
    X = _design_matrix(df, result.config)
    X = X.reindex(columns=result.feature_names, fill_value=0.0)
    return inverse_target(result.model.predict(result.scaler.transform(X)), result.config.form)


def total_sales(result: MMMResult, spend_scalers: dict[str, float] | None = None) -> float:
    return float(np.sum(predict_scenario(result, spend_scalers)))


# ---------------------------------------------------------------------------
# Decomposição de contribuições
# ---------------------------------------------------------------------------
def decompose_contributions(result: MMMResult) -> pd.DataFrame:
    """Contribuição incremental por canal, semana a semana.

    Método (decomposição aditiva no espaço do modelo):
      * 'Base' = predição com TODA a mídia zerada — ou seja, vendas orgânicas
        explicadas pelo intercepto, sazonalidade, promoção e concorrência.
      * contribuição do canal i = coef_i · (z_i − z_i|spend=0), onde z é a feature
        padronizada. Em modelo linear a soma fecha exatamente com a predição.

    Para as formas log (multiplicativas) o total incremental em escala de vendas
    (predição − Base) é rateado proporcionalmente às contribuições do espaço do
    modelo, de modo que a área empilhada continue somando o `sales` previsto.
    """
    cfg = result.config
    base_pred = predict_scenario(result)
    out = pd.DataFrame({"date": result.data["date"].to_numpy()})

    # Features com a mídia no nível real e com toda a mídia zerada.
    X_full = _design_matrix(result.data, cfg).reindex(columns=result.feature_names, fill_value=0.0)
    zero_scalers = {col: 0.0 for col in cfg.media_columns}
    X_zero = _design_matrix(result.data, cfg, zero_scalers).reindex(
        columns=result.feature_names, fill_value=0.0
    )
    Z_full = result.scaler.transform(X_full)
    Z_zero = result.scaler.transform(X_zero)

    zero_pred = inverse_target(result.model.predict(Z_zero), cfg.form)
    coefs = result.coefficients.reindex(result.feature_names).fillna(0.0).to_numpy()
    idx = {name: i for i, name in enumerate(result.feature_names)}

    model_contrib = {}
    for col in cfg.media_columns:
        i = idx[col]
        model_contrib[col] = np.clip(coefs[i] * (Z_full[:, i] - Z_zero[:, i]), 0, None)

    stacked = np.sum(list(model_contrib.values()), axis=0)
    incremental = np.clip(base_pred - zero_pred, 0, None)
    share = np.divide(incremental, stacked, out=np.zeros_like(stacked), where=stacked > 1e-12)

    total_media = np.zeros_like(base_pred)
    for col in cfg.media_columns:
        contrib = model_contrib[col] * share
        out[col] = contrib
        total_media += contrib

    base = base_pred - total_media

    # Base negativa = extrapolação ruim para "mídia zero" (típico de forma Linear
    # com todos os canais sempre ativos). Não escondemos: zeramos a Base, rateamos
    # o excedente entre os canais e sinalizamos para a UI avisar o analista.
    base_negativa = bool((base < 0).any())
    if base_negativa:
        deficit = np.clip(-base, 0, None)
        scale_down = np.divide(
            np.clip(total_media - deficit, 0, None), np.maximum(total_media, 1e-9),
            out=np.ones_like(total_media), where=total_media > 1e-9,
        )
        for col in cfg.media_columns:
            out[col] = out[col].to_numpy() * np.where(base < 0, scale_down, 1.0)
        total_media = out[cfg.media_columns].sum(axis=1).to_numpy()
        base = base_pred - total_media

    out["Base"] = base
    out["previsto"] = base_pred
    out["real"] = result.data["sales"].to_numpy()
    out.attrs["base_negativa"] = base_negativa
    return out


def contribution_summary(result: MMMResult) -> pd.DataFrame:
    """Resumo por canal: contribuição total, % do sales previsto, investimento e ROI."""
    contrib = result.contributions
    total_pred = float(contrib["previsto"].sum())
    rows = []
    for col in result.config.media_columns:
        c = float(contrib[col].sum())
        spend = float(result.data[col].sum())
        rows.append(
            {
                "canal": col,
                "contribuicao": c,
                "% do previsto": 100 * c / total_pred if total_pred else np.nan,
                "investimento": spend,
                "ROI (sales/R$)": c / spend if spend > 0 else np.nan,
            }
        )
    base = float(contrib["Base"].sum())
    rows.append(
        {"canal": "Base", "contribuicao": base,
         "% do previsto": 100 * base / total_pred if total_pred else np.nan,
         "investimento": np.nan, "ROI (sales/R$)": np.nan}
    )
    return pd.DataFrame(rows).sort_values("contribuicao", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Curvas de resposta
# ---------------------------------------------------------------------------
def response_curve(result: MMMResult, channel: str, max_multiplier: float = 3.0, points: int = 25) -> pd.DataFrame:
    """Curva de resposta do canal: investimento total x contribuição incremental."""
    baseline_spend = float(result.data[channel].sum())
    zero_sales = total_sales(result, {channel: 0.0})
    multipliers = np.linspace(0.0, max_multiplier, points)
    rows = []
    for m in multipliers:
        rows.append(
            {
                "multiplicador": float(m),
                "investimento": baseline_spend * m,
                "contribuicao_incremental": total_sales(result, {channel: float(m)}) - zero_sales,
            }
        )
    curve = pd.DataFrame(rows)
    curve["retorno_marginal"] = curve["contribuicao_incremental"].diff() / curve["investimento"].diff()
    return curve


def due_to_analysis(result: MMMResult, periods: int = 2) -> pd.DataFrame:
    """Due-to simplificada: variação da contribuição de cada driver entre dois períodos."""
    contrib = result.contributions.copy()
    contrib = contrib.sort_values("date")
    n = len(contrib)
    half = n // periods
    if half < 2:
        return pd.DataFrame()
    p1 = contrib.iloc[-2 * half : -half]
    p2 = contrib.iloc[-half:]
    drivers = list(result.config.media_columns) + ["Base"]
    rows = []
    for col in drivers:
        a, b = float(p1[col].sum()), float(p2[col].sum())
        rows.append(
            {
                "driver": col,
                "periodo_anterior": a,
                "periodo_atual": b,
                "delta": b - a,
                "delta_%": (100 * (b - a) / a) if a else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Otimização automática de adstock
# ---------------------------------------------------------------------------
def optimize_adstock(df: pd.DataFrame, cfg: MMMConfig, grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)) -> dict[str, float]:
    """Grid search coordenado: para cada canal, escolhe o decay que minimiza o MAPE de holdout."""
    best = dict(cfg.decays)
    for col in cfg.media_columns:
        scores = {}
        for decay in grid:
            trial = dict(best)
            trial[col] = decay
            trial_cfg = MMMConfig(**{**cfg.__dict__, "decays": trial, "bayesian": False})
            try:
                res = fit_mmm(df, trial_cfg)
                scores[decay] = res.metrics["mape_holdout"]
            except Exception:
                scores[decay] = np.inf
        best[col] = min(scores, key=lambda d: scores[d] if np.isfinite(scores[d]) else np.inf)
    return best

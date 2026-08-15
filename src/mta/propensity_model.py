"""MTA Preditivo — modelo de propensão à conversão (classificação supervisionada).

Complementa a atribuição (Páginas 4 e 5): em vez de explicar o crédito histórico
por canal, prevê a PROBABILIDADE de conversão de um cliente ANTES de acontecer —
o mesmo tipo de modelo usado para lead scoring e otimização de audiência.

Pipeline: ColumnTransformer (OneHot nas categóricas + StandardScaler nas
numéricas) -> classificador. Split estratificado e `class_weight="balanced"`
para lidar com o desbalanceamento (~88/12).

NOTA sobre desbalanceamento: usamos reponderação de classes por simplicidade e
reprodutibilidade. Uma alternativa comum seria oversampling sintético (SMOTE, via
`imbalanced-learn`), que exigiria dependência extra e cuidado para só reamostrar
dentro do treino — fica registrado como caminho alternativo, não implementado.

xgboost e shap são OPCIONAIS: ambos entram por try/except e o app degrada
graciosamente para GradientBoostingClassifier e importância nativa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_digital

try:  # pragma: no cover - dependência opcional
    from xgboost import XGBClassifier  # type: ignore

    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

try:  # pragma: no cover - dependência opcional
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False


CATEGORICAL_FEATURES = ["Gender", "CampaignChannel", "CampaignType"]
NUMERIC_FEATURES = [
    "Age",
    "Income",
    "AdSpend",
    "ClickThroughRate",
    "WebsiteVisits",
    "PagesPerVisit",
    "TimeOnSite",
    "SocialShares",
    "EmailOpens",
    "EmailClicks",
    "PreviousPurchases",
    "LoyaltyPoints",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Conversion"

ALGORITHMS = ["Regressão Logística", "Random Forest", "Gradient Boosting"]
RANDOM_STATE = 42


@dataclass
class PropensityResult:
    """Saída completa do treino, consumida pela Página 10."""

    algorithm: str
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    y_proba: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]
    feature_names: list[str]
    importance: pd.DataFrame
    threshold: float
    used_fallback: bool = False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_pipeline(algorithm: str, params: dict[str, Any] | None = None) -> tuple[Pipeline, bool]:
    """Monta o Pipeline (pré-processamento + classificador).

    Returns:
        (pipeline, used_fallback) — used_fallback=True quando o usuário pediu
        Gradient Boosting e o xgboost não está disponível.
    """
    params = params or {}
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    used_fallback = False
    if algorithm == "Regressão Logística":
        clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif algorithm == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth") or None,
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:  # Gradient Boosting
        if XGBOOST_AVAILABLE:
            clf = XGBClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=int(params.get("max_depth") or 4),
                learning_rate=float(params.get("learning_rate", 0.08)),
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:
            # Fallback sklearn: não tem class_weight, então usamos sample_weight no fit.
            clf = GradientBoostingClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                max_depth=int(params.get("max_depth") or 3),
                learning_rate=float(params.get("learning_rate", 0.08)),
                random_state=RANDOM_STATE,
            )
            used_fallback = True

    return Pipeline([("pre", pre), ("clf", clf)]), used_fallback


def _feature_names(pipeline: Pipeline) -> list[str]:
    try:
        return list(pipeline.named_steps["pre"].get_feature_names_out())
    except Exception:
        return CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _importance(pipeline: Pipeline, names: list[str]) -> pd.DataFrame:
    """Importância nativa (RF/GBM) ou odds ratio (Regressão Logística)."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_, dtype=float)
        df = pd.DataFrame({"feature": names[: len(values)], "importancia": values})
        df["odds_ratio"] = np.nan
    else:
        coefs = np.asarray(clf.coef_, dtype=float).ravel()
        df = pd.DataFrame({"feature": names[: len(coefs)], "coeficiente": coefs})
        df["odds_ratio"] = np.exp(df["coeficiente"])
        df["importancia"] = df["coeficiente"].abs()
    return df.sort_values("importancia", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Treino / avaliação
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Treinando modelo de propensão...")
def train_model(
    algorithm: str,
    params_key: tuple,
    threshold: float = 0.5,
    segment_channel: str = "Todos",
) -> PropensityResult:
    """Treina e avalia o modelo. Cacheado: só retreina se algoritmo/params mudarem.

    `params_key` é uma tupla ordenada de (nome, valor) — precisa ser hashable
    para o cache_resource do Streamlit.
    """
    params = dict(params_key)
    df = load_digital()
    if segment_channel and segment_channel != "Todos":
        df = df[df["CampaignChannel"] == segment_channel]

    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline, used_fallback = build_pipeline(algorithm, params)

    if used_fallback:
        # class_weight="balanced" manual para o GradientBoostingClassifier
        pos = max(int((y_train == 1).sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        weights = np.where(y_train == 1, len(y_train) / (2 * pos), len(y_train) / (2 * neg))
        pipeline.fit(X_train, y_train, clf__sample_weight=weights)
    else:
        pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = evaluate_model(y_test, y_proba, y_pred)
    names = _feature_names(pipeline)

    return PropensityResult(
        algorithm=algorithm,
        pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_proba=y_proba,
        y_pred=y_pred,
        metrics=metrics,
        feature_names=names,
        importance=_importance(pipeline, names),
        threshold=threshold,
        used_fallback=used_fallback,
    )


def evaluate_model(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, Precision, Recall, F1, ROC-AUC e PR-AUC."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "baseline_positivos": float(np.mean(y_true)),
    }


def curves(result: PropensityResult) -> dict[str, tuple]:
    """Pontos das curvas ROC e Precision-Recall."""
    fpr, tpr, _ = roc_curve(result.y_test, result.y_proba)
    precision, recall, _ = precision_recall_curve(result.y_test, result.y_proba)
    return {"roc": (fpr, tpr), "pr": (recall, precision)}


def confusion(result: PropensityResult) -> pd.DataFrame:
    cm = confusion_matrix(result.y_test, result.y_pred)
    return pd.DataFrame(
        cm,
        index=["Real: Não converteu", "Real: Converteu"],
        columns=["Previsto: Não converteu", "Previsto: Converteu"],
    )


# ---------------------------------------------------------------------------
# Explicabilidade
# ---------------------------------------------------------------------------
def explain_model(result: PropensityResult, sample_size: int = 400) -> dict:
    """SHAP se disponível; senão, cai para a importância nativa com aviso.

    Returns:
        {"method": "shap"|"native", "summary": DataFrame, "shap_values": ndarray|None,
         "sample": DataFrame|None}
    """
    if not SHAP_AVAILABLE:
        return {
            "method": "native",
            "summary": result.importance,
            "shap_values": None,
            "sample": None,
            "message": "Biblioteca `shap` não instalada — exibindo a importância nativa do modelo.",
        }

    try:  # pragma: no cover - depende de lib opcional
        sample = result.X_test.sample(
            n=min(sample_size, len(result.X_test)), random_state=RANDOM_STATE
        )
        pre = result.pipeline.named_steps["pre"]
        clf = result.pipeline.named_steps["clf"]
        transformed = pre.transform(sample)

        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
        else:
            background = shap.sample(pre.transform(result.X_train), 100, random_state=RANDOM_STATE)
            explainer = shap.LinearExplainer(clf, background)

        values = explainer.shap_values(transformed)
        if isinstance(values, list):  # binário -> lista de 2 arrays em versões antigas
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]

        summary = (
            pd.DataFrame(
                {
                    "feature": result.feature_names[: values.shape[1]],
                    "importancia": np.abs(values).mean(axis=0),
                    "efeito_medio": values.mean(axis=0),
                }
            )
            .sort_values("importancia", ascending=False)
            .reset_index(drop=True)
        )
        return {
            "method": "shap",
            "summary": summary,
            "shap_values": values,
            "sample": sample,
            "message": "",
        }
    except Exception as exc:
        return {
            "method": "native",
            "summary": result.importance,
            "shap_values": None,
            "sample": None,
            "message": f"Falha ao calcular SHAP ({exc}). Exibindo a importância nativa.",
        }


def score_profile(result: PropensityResult, profile: dict) -> tuple[float, pd.DataFrame]:
    """Score de um perfil informado no formulário + os 3 fatores que mais pesaram.

    Os fatores vêm de SHAP quando disponível; senão, do produto
    coeficiente x valor padronizado (LogReg) ou da importância nativa (árvores).
    """
    row = pd.DataFrame([{f: profile.get(f) for f in FEATURES}])
    proba = float(result.pipeline.predict_proba(row)[0, 1])

    drivers = pd.DataFrame(columns=["feature", "contribuicao"])
    try:
        pre = result.pipeline.named_steps["pre"]
        clf = result.pipeline.named_steps["clf"]
        transformed = np.asarray(pre.transform(row))

        if SHAP_AVAILABLE and hasattr(clf, "feature_importances_"):
            values = np.asarray(shap.TreeExplainer(clf).shap_values(transformed))
            if values.ndim == 3:
                values = values[:, :, 1]
            contrib = values[0]
        elif hasattr(clf, "coef_"):
            contrib = np.asarray(clf.coef_).ravel() * transformed[0]
        else:
            contrib = np.asarray(clf.feature_importances_) * transformed[0]

        drivers = pd.DataFrame(
            {"feature": result.feature_names[: len(contrib)], "contribuicao": contrib}
        )
        drivers["abs"] = drivers["contribuicao"].abs()
        drivers = drivers.sort_values("abs", ascending=False).head(3).drop(columns="abs")
    except Exception:
        drivers = result.importance.head(3).rename(columns={"importancia": "contribuicao"})[
            ["feature", "contribuicao"]
        ]

    return proba, drivers.reset_index(drop=True)


def predicted_propensity_by_channel(result: PropensityResult) -> pd.DataFrame:
    """Probabilidade média prevista de conversão por canal (no conjunto de teste)."""
    df = result.X_test.copy()
    df["propensao_prevista"] = result.y_proba
    df["conversao_real"] = result.y_test
    grp = (
        df.groupby("CampaignChannel")
        .agg(
            clientes=("propensao_prevista", "size"),
            propensao_media=("propensao_prevista", "mean"),
            conversao_real=("conversao_real", "mean"),
        )
        .reset_index()
        .rename(columns={"CampaignChannel": "canal"})
    )
    return grp.sort_values("propensao_media", ascending=False).reset_index(drop=True)


def opportunity_gaps(propensity: pd.DataFrame, attribution_share: pd.Series) -> pd.DataFrame:
    """Cruza propensão prevista x crédito do Markov/Shapley e sinaliza divergências.

    Canal com ALTA propensão prevista e BAIXO crédito atribuído = candidato a
    "oportunidade de investigação" para o Learning Repository (Página 9).
    """
    out = propensity.copy()
    out["rank_propensao"] = out["propensao_media"].rank(ascending=False)
    share = attribution_share.reindex(out["canal"]).fillna(0.0)
    out["credito_atribuido_%"] = share.to_numpy()
    out["rank_credito"] = out["credito_atribuido_%"].rank(ascending=False)
    out["gap_rank"] = out["rank_credito"] - out["rank_propensao"]
    out["sinal"] = np.where(
        out["gap_rank"] >= 2,
        "🔎 Oportunidade de investigação",
        np.where(out["gap_rank"] <= -2, "⚠️ Crédito acima da propensão", "✅ Alinhado"),
    )
    return out.sort_values("gap_rank", ascending=False).reset_index(drop=True)

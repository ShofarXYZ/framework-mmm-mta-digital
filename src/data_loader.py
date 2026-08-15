"""Leitura e limpeza cacheada dos 3 datasets do app.

Regra de ouro (seção 4a do briefing): **nunca** sobrescrevemos os CSVs originais.
Toda limpeza acontece em memória e o resultado é guardado em `st.session_state`
pelas páginas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MMM_FILE = DATA_DIR / "mmm_dataset.csv"
DIGITAL_FILE = DATA_DIR / "digital_marketing_campaign_dataset.csv"
CAMPAIGN_FILE = DATA_DIR / "marketing_campaign_dataset.csv"

# ---------------------------------------------------------------------------
# Constantes do MMM
# ---------------------------------------------------------------------------
DIGITAL_CHANNELS = [
    "instagram_spend",
    "google_ads_spend",
    "youtube_spend",
    "influencer_spend",
    "ott_spend",
]
OFFLINE_CHANNELS = ["tv_spend", "newspaper_spend"]
MEDIA_CHANNELS = OFFLINE_CHANNELS + DIGITAL_CHANNELS
CONTROL_COLUMNS = ["competitor_spend", "holiday"]

CHANNEL_LABELS = {
    "tv_spend": "TV",
    "newspaper_spend": "Jornal",
    "instagram_spend": "Instagram",
    "google_ads_spend": "Google Ads",
    "youtube_spend": "YouTube",
    "influencer_spend": "Influencer",
    "ott_spend": "OTT",
    "competitor_spend": "Concorrência",
}

IMPUTATION_METHODS = [
    "Imputação (mediana)",
    "Imputação (média)",
    "Forecast (média móvel 4 semanas)",
    "Replace com zero",
    "Interpolação linear",
]


def label(col: str) -> str:
    """Rótulo amigável para uma coluna de canal."""
    return CHANNEL_LABELS.get(col, col.replace("_spend", "").replace("_", " ").title())


# ---------------------------------------------------------------------------
# MMM
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando mmm_dataset.csv...")
def load_mmm_raw() -> pd.DataFrame:
    """Lê o dataset semanal de MMM, faz parse de data DD-MM-AAAA e ordena.

    Não imputa nada — os missings ficam visíveis para a página de Data Quality.
    """
    df = pd.read_csv(MMM_FILE)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["sales_promotion"] = df["sales_promotion"].fillna("Normal").astype(str).str.strip()
    numeric = MEDIA_CHANNELS + ["competitor_spend", "sales", "holiday"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """% de valores ausentes por coluna — exibido ANTES de qualquer imputação."""
    total = len(df)
    rows = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        rows.append(
            {
                "coluna": col,
                "faltantes": n_missing,
                "% faltante": round(100 * n_missing / total, 1) if total else 0.0,
                "tipo": str(df[col].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values("% faltante", ascending=False).reset_index(drop=True)


def _impute_series(s: pd.Series, method: str) -> pd.Series:
    """Aplica um dos métodos de tratamento de missing do framework a uma série."""
    s = s.copy()
    if method == "Imputação (mediana)":
        return s.fillna(s.median())
    if method == "Imputação (média)":
        return s.fillna(s.mean())
    if method == "Forecast (média móvel 4 semanas)":
        # média móvel das 4 semanas anteriores; sobras nas pontas caem para a mediana
        rolling = s.rolling(window=4, min_periods=1).mean().shift(1)
        return s.fillna(rolling).fillna(s.median())
    if method == "Replace com zero":
        return s.fillna(0.0)
    if method == "Interpolação linear":
        return s.interpolate(method="linear", limit_direction="both").fillna(s.median())
    return s


def clean_mmm(df: pd.DataFrame, methods: dict[str, str], sales_method: str = "Interpolação linear") -> pd.DataFrame:
    """Aplica o método de tratamento escolhido pelo usuário coluna a coluna.

    Args:
        df: dataframe cru de `load_mmm_raw`.
        methods: mapa {coluna_de_spend: método}.
        sales_method: método aplicado à variável dependente `sales`.
    """
    out = df.copy()
    for col, method in methods.items():
        if col in out.columns:
            out[col] = _impute_series(out[col], method)
    if "sales" in out.columns:
        out["sales"] = _impute_series(out["sales"], sales_method)
    if "competitor_spend" in out.columns:
        out["competitor_spend"] = _impute_series(out["competitor_spend"], "Interpolação linear")
    out["holiday"] = out["holiday"].fillna(0).astype(int)

    # Dummies de promoção (seção 4a): is_bogo, is_normal, ...
    promo = out["sales_promotion"].str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True)
    for value in sorted(promo.unique()):
        out[f"is_{value}"] = (promo == value).astype(int)
    return out


def promo_dummy_columns(df: pd.DataFrame) -> list[str]:
    """Colunas dummy de promoção criadas por `clean_mmm` (dropa a primeira p/ evitar dummy trap)."""
    cols = sorted(c for c in df.columns if c.startswith("is_"))
    return cols[1:] if len(cols) > 1 else []


def detect_outliers(df: pd.DataFrame, column: str, method: str = "z-score", threshold: float = 3.0) -> pd.Series:
    """Máscara booleana de outliers por z-score ou IQR."""
    s = pd.to_numeric(df[column], errors="coerce")
    if method == "IQR":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
    std = s.std(ddof=0)
    if not std or np.isnan(std):
        return pd.Series(False, index=df.index)
    return ((s - s.mean()).abs() / std) > threshold


# ---------------------------------------------------------------------------
# MTA — digital_marketing_campaign_dataset.csv
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando digital_marketing_campaign_dataset.csv...")
def load_digital() -> pd.DataFrame:
    """Dataset cliente/campanha usado nas páginas de MTA e A/B."""
    df = pd.read_csv(DIGITAL_FILE)
    df = df.drop(columns=[c for c in ("AdvertisingPlatform", "AdvertisingTool") if c in df.columns])
    numeric = [
        "Age", "Income", "AdSpend", "ClickThroughRate", "ConversionRate", "WebsiteVisits",
        "PagesPerVisit", "TimeOnSite", "SocialShares", "EmailOpens", "EmailClicks",
        "PreviousPurchases", "LoyaltyPoints", "Conversion",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["CustomerID", "CampaignChannel", "Conversion"])
    df["Conversion"] = df["Conversion"].astype(int)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# A/B — marketing_campaign_dataset.csv (200k linhas)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Carregando marketing_campaign_dataset.csv (200k linhas)...")
def load_campaigns() -> pd.DataFrame:
    """Dataset de 200k campanhas. Converte Acquisition_Cost ('$1,234.00') para float."""
    df = pd.read_csv(CAMPAIGN_FILE)
    if "Acquisition_Cost" in df.columns:
        df["Acquisition_Cost"] = (
            df["Acquisition_Cost"].astype(str).str.replace(r"[$,]", "", regex=True)
        )
        df["Acquisition_Cost"] = pd.to_numeric(df["Acquisition_Cost"], errors="coerce")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Duration" in df.columns:
        df["Duration_days"] = pd.to_numeric(
            df["Duration"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
    for col in ("Conversion_Rate", "ROI", "Clicks", "Impressions", "Engagement_Score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Conversões implícitas: o dataset traz taxa, não contagem.
    df["Conversions"] = (df["Clicks"] * df["Conversion_Rate"]).round().astype("Int64")
    return df


@st.cache_data(show_spinner=False)
def campaign_channel_summary(segment: str | None = None) -> pd.DataFrame:
    """Agregado por canal do dataset de 200k campanhas (usado na validação cruzada da MTA)."""
    df = load_campaigns()
    if segment and segment != "Todos" and "Customer_Segment" in df.columns:
        df = df[df["Customer_Segment"] == segment]
    grp = df.groupby("Channel_Used", dropna=True).agg(
        campanhas=("Campaign_ID", "count"),
        clicks=("Clicks", "sum"),
        impressions=("Impressions", "sum"),
        conversions=("Conversions", "sum"),
        conversion_rate=("Conversion_Rate", "mean"),
        roi=("ROI", "mean"),
        custo=("Acquisition_Cost", "sum"),
        engagement=("Engagement_Score", "mean"),
    )
    grp = grp.reset_index().rename(columns={"Channel_Used": "canal"})
    grp["conversions"] = grp["conversions"].astype(float)
    grp["cpa"] = np.where(grp["conversions"] > 0, grp["custo"] / grp["conversions"], np.nan)
    return grp.sort_values("roi", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def kpi_snapshot() -> dict:
    """KPIs rápidos dos 3 datasets, para os cards da Home."""
    out: dict = {}
    try:
        mmm = load_mmm_raw()
        out["mmm_investimento"] = float(mmm[MEDIA_CHANNELS].sum().sum())
        out["mmm_sales"] = float(mmm["sales"].sum())
        out["mmm_semanas"] = int(len(mmm))
        out["mmm_missing_pct"] = float(
            100 * mmm[MEDIA_CHANNELS].isna().sum().sum() / (len(mmm) * len(MEDIA_CHANNELS))
        )
    except Exception:  # dado ausente/corrompido não pode derrubar a Home
        out["mmm_investimento"] = out["mmm_sales"] = np.nan
        out["mmm_semanas"] = 0
        out["mmm_missing_pct"] = np.nan
    try:
        dig = load_digital()
        out["mta_clientes"] = int(len(dig))
        out["mta_conv_rate"] = float(dig["Conversion"].mean())
        out["mta_adspend"] = float(dig["AdSpend"].sum())
    except Exception:
        out["mta_clientes"] = 0
        out["mta_conv_rate"] = out["mta_adspend"] = np.nan
    try:
        camp = load_campaigns()
        out["ab_campanhas"] = int(len(camp))
        out["ab_roi"] = float(camp["ROI"].mean())
        out["ab_conv_rate"] = float(camp["Conversion_Rate"].mean())
    except Exception:
        out["ab_campanhas"] = 0
        out["ab_roi"] = out["ab_conv_rate"] = np.nan
    return out


@st.cache_data(show_spinner=False)
def segment_options() -> list[str]:
    """Opções do seletor global de segmento (sidebar)."""
    try:
        df = load_campaigns()
        return ["Todos"] + sorted(df["Customer_Segment"].dropna().unique().tolist())
    except Exception:
        return ["Todos"]

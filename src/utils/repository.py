"""Learning Repository — memória de aprendizados compartilhada entre as páginas.

Persistente dentro da sessão (`st.session_state`) e exportável em CSV no mesmo
formato de colunas do Roadmap de Testes A/B, para colar de volta na planilha.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

STATE_KEY = "learning_repository"

# Ordem das colunas do export (espelha as abas do framework original).
COLUMNS = [
    "experiment_id",
    "origem",
    "canal_driver",
    "hipotese",
    "etapa",
    "resultado",
    "lift_pct",
    "p_value",
    "insight",
    "proximo_passo",
    "data",
]

ORIGINS = ["MMM", "MTA", "MTA Preditivo", "Geo-Holdout", "Teste A/B"]
RESULTS = ["Winner", "Neutral", "Loser", "Insight", "Oportunidade"]
STAGES = [
    "Opportunity",
    "Hypothesis",
    "Priorization",
    "Design",
    "Tracking",
    "Results",
    "Learning",
]


def _ensure_state() -> None:
    if STATE_KEY not in st.session_state:
        st.session_state[STATE_KEY] = []


def add_entry(
    origem: str,
    canal_driver: str,
    hipotese: str,
    resultado: str,
    insight: str,
    proximo_passo: str,
    etapa: str = "Results",
    lift_pct: float | None = None,
    p_value: float | None = None,
) -> str:
    """Adiciona um registro e devolve o experiment_id gerado."""
    _ensure_state()
    entries = st.session_state[STATE_KEY]
    prefix = {"MMM": "MMM", "MTA": "MTA", "MTA Preditivo": "PRP", "Geo-Holdout": "GEO", "Teste A/B": "ABT"}.get(
        origem, "EXP"
    )
    experiment_id = f"{prefix}-{len(entries) + 1:03d}"
    entries.append(
        {
            "experiment_id": experiment_id,
            "origem": origem,
            "canal_driver": canal_driver,
            "hipotese": hipotese,
            "etapa": etapa,
            "resultado": resultado,
            "lift_pct": round(float(lift_pct), 2) if lift_pct is not None and pd.notna(lift_pct) else None,
            "p_value": round(float(p_value), 5) if p_value is not None and pd.notna(p_value) else None,
            "insight": insight,
            "proximo_passo": proximo_passo,
            "data": date.today().isoformat(),
        }
    )
    return experiment_id


def get_df() -> pd.DataFrame:
    """Repositório atual como DataFrame (colunas sempre na ordem do export)."""
    _ensure_state()
    df = pd.DataFrame(st.session_state[STATE_KEY])
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    return df.reindex(columns=COLUMNS)


def clear() -> None:
    st.session_state[STATE_KEY] = []


def count() -> int:
    _ensure_state()
    return len(st.session_state[STATE_KEY])


def save_widget(
    key: str,
    origem: str,
    canal_driver: str,
    hipotese_default: str,
    resultado_default: str,
    insight_default: str,
    proximo_passo_default: str,
    lift_pct: float | None = None,
    p_value: float | None = None,
    etapa_default: str = "Results",
) -> None:
    """Bloco reutilizável '📌 Salvar este resultado no Learning Repository'."""
    with st.expander("📌 Salvar este resultado no Learning Repository"):
        with st.form(f"repo_form_{key}"):
            col1, col2 = st.columns(2)
            with col1:
                canal = st.text_input("Canal / Driver", value=canal_driver, key=f"{key}_canal")
                etapa = st.selectbox(
                    "Etapa do framework", STAGES, index=STAGES.index(etapa_default), key=f"{key}_etapa"
                )
            with col2:
                resultado = st.selectbox(
                    "Resultado",
                    RESULTS,
                    index=RESULTS.index(resultado_default) if resultado_default in RESULTS else 3,
                    key=f"{key}_res",
                )
                st.caption(f"Origem: **{origem}**")
            hipotese = st.text_area("Hipótese", value=hipotese_default, height=70, key=f"{key}_hip")
            insight = st.text_area("Insight", value=insight_default, height=70, key=f"{key}_ins")
            proximo = st.text_input(
                "Próximo passo sugerido", value=proximo_passo_default, key=f"{key}_next"
            )
            submitted = st.form_submit_button("Salvar no repositório", type="primary")

        if submitted:
            exp_id = add_entry(
                origem=origem,
                canal_driver=canal,
                hipotese=hipotese,
                resultado=resultado,
                insight=insight,
                proximo_passo=proximo,
                etapa=etapa,
                lift_pct=lift_pct,
                p_value=p_value,
            )
            st.success(f"Registrado como **{exp_id}** — veja na página *Loop de Governança*.")

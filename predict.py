"""
predict.py
==========
Scoring batch : renvoie POL_Num + probabilité de résiliation calibrée.

Usage notebook Databricks :
---------------------------
    from src.modeling.predict import score

    df_scores = score(run_id="...", df=df_a_scorer)
    # -> DataFrame [POL_Num, proba_resiliation]
"""

from __future__ import annotations

import pandas as pd

from src.modeling.save_load import load_model


def score(
    run_id: str,
    df: pd.DataFrame,
    id_col: str = "POL_Num",
    proba_col: str = "proba_resiliation",
    artifact_path: str = "model",
) -> pd.DataFrame:
    """Score un DataFrame avec un modèle loggé dans MLflow."""
    model = load_model(run_id, artifact_path)
    proba = model.predict(df)
    return pd.DataFrame({id_col: df[id_col].values, proba_col: proba})


def score_with_loaded_model(
    model,
    df: pd.DataFrame,
    id_col: str = "POL_Num",
    proba_col: str = "proba_resiliation",
) -> pd.DataFrame:
    """Variante quand le modèle est déjà chargé (évite de recharger à chaque appel)."""
    proba = model.predict(df)
    return pd.DataFrame({id_col: df[id_col].values, proba_col: proba})


def score_elasticity(
    model,
    df: pd.DataFrame,
    tarif_feature: str,
    deltas_pct: list[float] = (-10, -5, 0, 5, 10, 15, 20),
    id_col: str = "POL_Num",
) -> pd.DataFrame:
    """Courbe d'élasticité : proba de résiliation en fonction d'un choc tarifaire.

    Simule des variations de la feature de tarif (en %) et renvoie la proba
    pour chaque scénario — directement exploitable pour l'optimisation tarifaire.
    Grâce à la contrainte de monotonie positive, la courbe est garantie croissante.
    """
    out = df[[id_col]].copy()
    for d in deltas_pct:
        df_s = df.copy()
        df_s[tarif_feature] = df_s[tarif_feature] * (1 + d / 100.0)
        out[f"proba_delta_{d:+g}pct"] = model.predict(df_s)
    return out

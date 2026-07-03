"""
save_load.py
============
Sérialisation uniforme des 4 modèles via un wrapper MLflow pyfunc qui
embarque : préprocesseur + modèle + calibrateur.

Avantage : quel que soit le modèle (LightGBM, XGBoost, CatBoost, EBM),
le chargement et le scoring dans un notebook Databricks sont identiques :

    import mlflow
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    proba = model.predict(df_scoring)        # -> proba de résiliation calibrée
"""

from __future__ import annotations

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd


class CalibratedModelWrapper(mlflow.pyfunc.PythonModel):
    """Préprocesseur + modèle + calibrateur dans un seul objet pyfunc."""

    def __init__(self, preprocessor, model, calibrator=None):
        self.preprocessor = preprocessor
        self.model = model
        self.calibrator = calibrator

    def predict(self, context, model_input: pd.DataFrame, params=None) -> np.ndarray:
        X = self.preprocessor.transform(model_input)
        p = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.transform(p)
        return p

    # Accès direct hors pyfunc (chargé via load_wrapper)
    def predict_proba_df(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict(None, df)


def log_model_to_mlflow(
    wrapper: CalibratedModelWrapper,
    artifact_path: str = "model",
    input_example: pd.DataFrame | None = None,
    registered_name: str | None = None,
):
    """Log le wrapper pyfunc dans le run MLflow actif."""
    pip_reqs = [
        "pandas", "numpy", "scikit-learn",
        "lightgbm", "xgboost", "catboost", "interpret",
    ]
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=wrapper,
        input_example=input_example,
        registered_model_name=registered_name,
        pip_requirements=pip_reqs,
        code_paths=["src"],  # embarque le code du repo pour désérialiser
    )


def load_model(run_id: str, artifact_path: str = "model"):
    """Charge le modèle pyfunc depuis un run MLflow (usage notebook)."""
    return mlflow.pyfunc.load_model(f"runs:/{run_id}/{artifact_path}")


def load_wrapper(run_id: str, artifact_path: str = "model") -> CalibratedModelWrapper:
    """Charge le wrapper Python natif (accès au modèle brut, à l'EBM, etc.)."""
    pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{artifact_path}")
    return pyfunc_model.unwrap_python_model()

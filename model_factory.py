"""
model_factory.py
================
Fabrique de modèles gérant les spécificités de chaque librairie :

- Prétraitement des variables catégorielles :
    * LightGBM  -> dtype pandas 'category' (support natif)
    * CatBoost  -> colonnes en str (support natif via cat_features)
    * XGBoost   -> dtype 'category' + enable_categorical=True
    * EBM       -> colonnes en str (support natif interpretml)
- Valeurs manquantes :
    * LightGBM / XGBoost / EBM -> gérées nativement
    * CatBoost -> NaN catégoriels remplacés par "MISSING"
- Contraintes de monotonie (format propre à chaque lib)
- Déséquilibre de classes (scale_pos_weight / class_weights)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUPPORTED_MODELS = ["lightgbm", "xgboost", "catboost", "ebm"]


# ----------------------------------------------------------------------
# Prétraitement par modèle
# ----------------------------------------------------------------------
class Preprocessor:
    """Prétraitement minimal, spécifique à chaque librairie.

    Sérialisable (joblib/pickle) -> embarqué dans le pyfunc MLflow pour
    garantir que le scoring applique exactement les mêmes transformations.
    """

    def __init__(self, model_name: str, id_col: str, target: str):
        assert model_name in SUPPORTED_MODELS, f"Modèle inconnu : {model_name}"
        self.model_name = model_name
        self.id_col = id_col
        self.target = target
        self.feature_names_: list[str] = []
        self.cat_cols_: list[str] = []
        self.num_cols_: list[str] = []
        self.cat_categories_: dict[str, list] = {}

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = df.drop(columns=[c for c in [self.id_col, self.target] if c in df.columns])
        self.feature_names_ = list(X.columns)
        from pandas.api import types as ptypes
        # object / string / category -> catégorielle (compatible pandas 2 et 3)
        self.cat_cols_ = [
            c for c in X.columns
            if not ptypes.is_numeric_dtype(X[c]) and not ptypes.is_datetime64_any_dtype(X[c])
        ]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]
        # Mémorise les modalités du train pour un encodage 'category' stable
        for c in self.cat_cols_:
            self.cat_categories_[c] = (
                X[c].astype("string").fillna("MISSING").unique().tolist()
            )
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.drop(
            columns=[c for c in [self.id_col, self.target] if c in df.columns]
        ).copy()
        X = X[self.feature_names_]  # ordre stable (requis pour la monotonie)

        if self.model_name in ("lightgbm", "xgboost"):
            for c in self.cat_cols_:
                s = X[c].astype("string").fillna("MISSING")
                X[c] = pd.Categorical(s, categories=self.cat_categories_[c])
            for c in self.num_cols_:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        elif self.model_name in ("catboost", "ebm"):
            for c in self.cat_cols_:
                X[c] = X[c].astype("string").fillna("MISSING").astype(str)
            for c in self.num_cols_:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            if self.model_name == "ebm":
                # interpretml préfère les objets pour les catégorielles
                X[self.cat_cols_] = X[self.cat_cols_].astype(object)

        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ----------------------------------------------------------------------
# Contraintes de monotonie
# ----------------------------------------------------------------------
def build_monotone_constraints(
    model_name: str,
    feature_names: list[str],
    monotone_positive: list[str],
    monotone_negative: list[str] | None = None,
):
    """Construit la contrainte de monotonie au format de chaque librairie."""
    monotone_negative = monotone_negative or []
    missing = [
        f for f in monotone_positive + monotone_negative if f not in feature_names
    ]
    if missing:
        raise ValueError(
            f"Features monotones absentes de la base : {missing}. "
            f"Vérifier config/parameters.yaml (features.monotone_positive)."
        )

    vec = [
        1 if f in monotone_positive else (-1 if f in monotone_negative else 0)
        for f in feature_names
    ]

    if model_name == "lightgbm":
        return vec                                       # liste alignée
    if model_name == "xgboost":
        return "(" + ",".join(str(v) for v in vec) + ")" # string "(1,0,...)"
    if model_name == "catboost":
        return {f: v for f, v in zip(feature_names, vec) if v != 0}  # dict
    if model_name == "ebm":
        return vec                                       # liste alignée
    raise ValueError(model_name)


# ----------------------------------------------------------------------
# Instanciation des modèles
# ----------------------------------------------------------------------
def create_model(
    model_name: str,
    params: dict,
    preprocessor: Preprocessor,
    monotone_positive: list[str],
    monotone_negative: list[str] | None = None,
    scale_pos_weight: float = 1.0,
    seed: int = 42,
):
    """Instancie un modèle prêt à être fit, avec monotonie + déséquilibre."""
    feats = preprocessor.feature_names_
    constraints = build_monotone_constraints(
        model_name, feats, monotone_positive, monotone_negative
    )

    if model_name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            objective="binary",
            monotone_constraints=constraints,
            monotone_constraints_method="advanced",  # moins de perte de perf
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **params,
        )

    if model_name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            monotone_constraints=constraints,
            scale_pos_weight=scale_pos_weight,
            enable_categorical=True,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            monotone_constraints=constraints,
            scale_pos_weight=scale_pos_weight,
            cat_features=preprocessor.cat_cols_,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            **params,
        )

    if model_name == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier

        # EBM ne supporte pas scale_pos_weight -> poids par observation
        # passés au fit via sample_weight (géré dans train.py)
        return ExplainableBoostingClassifier(
            monotone_constraints=constraints,
            feature_names=feats,
            random_state=seed,
            n_jobs=-1,
            **params,
        )

    raise ValueError(model_name)


def fit_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    early_stopping_rounds: int = 100,
    scale_pos_weight: float = 1.0,
):
    """Fit avec early stopping sur VALID, selon l'API de chaque librairie."""
    if model_name == "lightgbm":
        import lightgbm as lgb

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
    elif model_name == "xgboost":
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    elif model_name == "catboost":
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=early_stopping_rounds,
        )
    elif model_name == "ebm":
        # Pas d'early stopping externe (géré en interne par max_rounds).
        # Déséquilibre géré via sample_weight.
        sw = np.where(y_train == 1, scale_pos_weight, 1.0)
        model.fit(X_train, y_train, sample_weight=sw)
    return model

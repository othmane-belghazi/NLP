"""
train.py
========
Pipeline d'entraînement multi-modèles avec :

1. Recherche d'hyperparamètres Optuna (maximise l'AUC sur VALID),
   incluant le poids de classe (scale_pos_weight) comme hyperparamètre
   pour gérer le fort déséquilibre résiliés / non-résiliés.
2. Réentraînement final avec les meilleurs hyperparamètres.
3. Calibration isotonique des probabilités sur VALID (indispensable :
   les probas alimentent l'optimisation tarifaire, le ranking seul
   ne suffit pas — la repondération de classe distord les probas,
   la calibration les remet sur la bonne échelle).
4. Tracking MLflow complet (params, métriques train/valid/test,
   figures, modèle pyfunc rechargeable dans un notebook Databricks).

Utilisation dans un notebook Databricks :
-----------------------------------------
    import sys; sys.path.append("/Workspace/Repos/.../project")
    from src.modeling.train import TrainingPipeline

    pipe = TrainingPipeline(config_path="config/parameters.yaml")
    results = pipe.run(df_train, df_valid, df_test)   # pandas DataFrames
"""

from __future__ import annotations

import logging

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import compute_all_metrics
from src.evaluation.report import EvaluationReport
from src.modeling.model_factory import Preprocessor, create_model, fit_model
from src.modeling.save_load import CalibratedModelWrapper, log_model_to_mlflow

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ----------------------------------------------------------------------
# Calibrateur isotonique simple et robuste (fit sur VALID)
# ----------------------------------------------------------------------
class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._cal = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self._cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._cal.fit(p_raw, y)
        elif self.method == "sigmoid":
            from sklearn.linear_model import LogisticRegression
            self._cal = LogisticRegression(C=1e10, solver="lbfgs")
            eps = 1e-7
            logit = np.log(np.clip(p_raw, eps, 1 - eps) / np.clip(1 - p_raw, eps, 1 - eps))
            self._cal.fit(logit.reshape(-1, 1), y)
        else:
            raise ValueError(self.method)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if self._cal is None:
            return p_raw
        if self.method == "isotonic":
            return self._cal.predict(p_raw)
        eps = 1e-7
        logit = np.log(np.clip(p_raw, eps, 1 - eps) / np.clip(1 - p_raw, eps, 1 - eps))
        return self._cal.predict_proba(logit.reshape(-1, 1))[:, 1]


# ----------------------------------------------------------------------
# Échantillonnage Optuna à partir du YAML
# ----------------------------------------------------------------------
def _sample_params(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        if isinstance(spec, list):                     # valeur(s) fixe(s)
            params[name] = spec[0] if len(spec) == 1 else trial.suggest_categorical(name, spec)
        elif isinstance(spec, dict):
            low, high = spec["low"], spec["high"]
            log = bool(spec.get("log", False))
            if isinstance(low, int) and isinstance(high, int) and not log:
                params[name] = trial.suggest_int(name, low, high)
            elif isinstance(low, int) and isinstance(high, int) and log:
                params[name] = trial.suggest_int(name, low, high, log=True)
            else:
                params[name] = trial.suggest_float(name, float(low), float(high), log=log)
        else:
            params[name] = spec
    return params


# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------
class TrainingPipeline:
    def __init__(self, config_path: str = "config/parameters.yaml", config: dict | None = None):
        if config is None:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        self.cfg = config
        self.target = config["data"]["target"]
        self.id_col = config["data"]["id_col"]
        self.seed = config["training"]["seed"]
        self.mono_pos = config["features"]["monotone_positive"]
        self.mono_neg = config["features"].get("monotone_negative", []) or []
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # ------------------------------------------------------------------
    def _prepare(self, model_name, df_train, df_valid, df_test):
        prep = Preprocessor(model_name, self.id_col, self.target).fit(df_train)
        X_tr, X_va, X_te = (prep.transform(d) for d in (df_train, df_valid, df_test))
        y_tr = df_train[self.target].astype(int).values
        y_va = df_valid[self.target].astype(int).values
        y_te = df_test[self.target].astype(int).values
        return prep, (X_tr, y_tr), (X_va, y_va), (X_te, y_te)

    # ------------------------------------------------------------------
    def _tune(self, model_name, prep, train, valid) -> tuple[dict, float, optuna.Study]:
        X_tr, y_tr = train
        X_va, y_va = valid
        space = self.cfg["search_spaces"][model_name]
        opt_cfg = self.cfg["optuna"]
        imb_cfg = self.cfg["training"]["imbalance"]
        neg_pos_ratio = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

        def objective(trial: optuna.Trial) -> float:
            params = _sample_params(trial, space)
            if imb_cfg.get("tune_class_weight", True):
                spw = trial.suggest_float(
                    "scale_pos_weight",
                    1.0,
                    max(1.0 + 1e-6, neg_pos_ratio * imb_cfg.get("max_scale_pos_weight_factor", 1.0)),
                    log=True,
                )
            else:
                spw = 1.0
            model = create_model(
                model_name, params, prep, self.mono_pos, self.mono_neg,
                scale_pos_weight=spw, seed=self.seed,
            )
            fit_model(
                model_name, model, X_tr, y_tr, X_va, y_va,
                early_stopping_rounds=opt_cfg["early_stopping_rounds"],
                scale_pos_weight=spw,
            )
            p_va = model.predict_proba(X_va)[:, 1]
            return roc_auc_score(y_va, p_va)

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner() if opt_cfg.get("pruning", True) else None
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(
            objective,
            n_trials=opt_cfg["n_trials"],
            timeout=opt_cfg.get("timeout_seconds"),
            show_progress_bar=False,
        )
        best = dict(study.best_params)
        spw = best.pop("scale_pos_weight", 1.0)
        # Réinjecte les valeurs fixes du YAML (non échantillonnées)
        for k, v in space.items():
            if isinstance(v, list) and len(v) == 1:
                best.setdefault(k, v[0])
        best["_scale_pos_weight"] = spw
        return best, study.best_value, study

    # ------------------------------------------------------------------
    def train_one_model(self, model_name, df_train, df_valid, df_test) -> dict:
        logger.info("=== %s ===", model_name)
        prep, (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = self._prepare(
            model_name, df_train, df_valid, df_test
        )

        with mlflow.start_run(run_name=f"{model_name}"):
            mlflow.set_tags({"model_name": model_name, "target": self.target})
            mlflow.log_params({
                "monotone_positive": ",".join(self.mono_pos),
                "n_train": len(y_tr), "n_valid": len(y_va), "n_test": len(y_te),
                "taux_resiliation_train": round(float(y_tr.mean()), 5),
            })

            # 1) Tuning ------------------------------------------------
            best_params, best_auc_valid, study = self._tune(
                model_name, prep, (X_tr, y_tr), (X_va, y_va)
            )
            spw = best_params.pop("_scale_pos_weight", 1.0)
            mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})
            mlflow.log_param("hp_scale_pos_weight", round(spw, 4))
            mlflow.log_metric("optuna_best_auc_valid", best_auc_valid)

            # 2) Fit final ---------------------------------------------
            model = create_model(
                model_name, best_params, prep, self.mono_pos, self.mono_neg,
                scale_pos_weight=spw, seed=self.seed,
            )
            fit_model(
                model_name, model, X_tr, y_tr, X_va, y_va,
                early_stopping_rounds=self.cfg["optuna"]["early_stopping_rounds"],
                scale_pos_weight=spw,
            )

            # 3) Calibration sur VALID ---------------------------------
            cal_cfg = self.cfg["training"]["calibration"]
            calibrator = None
            p_va_raw = model.predict_proba(X_va)[:, 1]
            if cal_cfg.get("enabled", True):
                calibrator = ProbabilityCalibrator(cal_cfg.get("method", "isotonic"))
                calibrator.fit(p_va_raw, y_va)

            def proba(X):
                p = model.predict_proba(X)[:, 1]
                return calibrator.transform(p) if calibrator else p

            preds = {
                "train": (y_tr, proba(X_tr)),
                "valid": (y_va, proba(X_va)),
                "test": (y_te, proba(X_te)),
            }

            # 4) Métriques + rapport visuel ----------------------------
            all_metrics = {}
            for split, (y, p) in preds.items():
                m = compute_all_metrics(y, p, n_deciles=self.cfg["evaluation"]["n_deciles"])
                all_metrics[split] = m
                mlflow.log_metrics({f"{split}_{k}": v for k, v in m.items()})

            report = EvaluationReport(model_name=model_name, output_dir=self.cfg["evaluation"]["artifacts_dir"])
            report.generate(
                preds=preds,
                model=model,
                X_valid=X_va,
                monotone_features=self.mono_pos,
                log_to_mlflow=True,
            )

            # 5) Log du modèle pyfunc (préproc + modèle + calibrateur) --
            wrapper = CalibratedModelWrapper(prep, model, calibrator)
            log_model_to_mlflow(
                wrapper,
                artifact_path="model",
                input_example=df_valid.drop(columns=[self.target]).head(5),
                registered_name=(
                    f"{self.cfg['mlflow']['registered_model_prefix']}_{model_name}"
                    if self.cfg["mlflow"].get("register_model") else None
                ),
            )

            run_id = mlflow.active_run().info.run_id

        return {
            "model_name": model_name,
            "run_id": run_id,
            "best_params": best_params,
            "scale_pos_weight": spw,
            "metrics": all_metrics,
        }

    # ------------------------------------------------------------------
    def run(self, df_train, df_valid, df_test, models: list[str] | None = None) -> pd.DataFrame:
        """Entraîne tous les modèles et renvoie un tableau comparatif."""
        models = models or self.cfg["training"]["models"]
        results = [self.train_one_model(m, df_train, df_valid, df_test) for m in models]

        rows = []
        for r in results:
            for split, m in r["metrics"].items():
                rows.append({"model": r["model_name"], "run_id": r["run_id"], "split": split, **m})
        summary = pd.DataFrame(rows)

        # Log du comparatif dans un run dédié
        with mlflow.start_run(run_name="comparaison_modeles"):
            path = "/tmp/comparaison_modeles.csv"
            summary.to_csv(path, index=False)
            mlflow.log_artifact(path)
            best = (
                summary[summary["split"] == "valid"]
                .sort_values("auc", ascending=False)
                .iloc[0]
            )
            mlflow.log_param("best_model_valid_auc", best["model"])
            mlflow.log_metric("best_valid_auc", best["auc"])
        return summary

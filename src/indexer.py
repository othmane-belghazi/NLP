import optuna
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

PRICE_FEATURE = "ratio_prix"   # ta feature prix -> monotonie croissante
CAT_FEATURES  = []             # noms/indices de tes variables catégorielles

# Early stopping sur une tranche INTERNE de X_train ; X_val = objectif seulement.
X_fit, X_es, y_fit, y_es = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)
# Données temporelles ? Remplace par un découpage chronologique :
# X_es = la fin de X_train, X_fit = le début (pas de split aléatoire).

class CatBoostPruningCallback:
    def __init__(self, trial, metric="Logloss"):
        self.trial, self.metric = trial, metric
    def after_iteration(self, info):
        self.trial.report(info.metrics["validation"][self.metric][-1], info.iteration)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True

def objective(trial):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 5000,
        "od_type": "Iter",
        "od_wait": 250,
        "leaf_estimation_method": "Newton",
        "monotone_constraints": {PRICE_FEATURE: 1},
        "random_seed": 42,
        "verbose": False,

        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth":           trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg":     trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10, log=True),
        "border_count":    trial.suggest_int("border_count", 128, 1024),
        "rsm":             trial.suggest_float("rsm", 0.8, 1.0),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
        "bootstrap_type":  trial.suggest_categorical("bootstrap_type",
                               ["Bayesian", "Bernoulli", "MVS"]),
    }
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)

    model = CatBoostClassifier(**params)
    model.fit(
        X_fit, y_fit,
        eval_set=(X_es, y_es),         # pilote l'early stopping
        cat_features=CAT_FEATURES,
        use_best_model=True,
        callbacks=[CatBoostPruningCallback(trial)],
    )
    trial.set_user_attr("best_iteration", model.get_best_iteration())

    # objectif = LogLoss sur X_val, jamais vu par l'early stopping
    p_val = model.predict_proba(X_val)[:, 1]
    return log_loss(y_val, p_val)

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=300),
)
study.optimize(objective, n_trials=100, show_progress_bar=True)

best_params    = study.best_params
best_iteration = study.best_trial.user_attrs["best_iteration"]
print("Best LogLoss :", study.best_value)
print("Best params  :", best_params)
print("Best iter    :", best_iteration)
params = {
    # ---- Fixés ----
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 5000,                 # haut, borné par early stopping
    "od_type": "Iter", "od_wait": 200,
    "leaf_estimation_method": "Newton", # meilleur pour LogLoss/calibration
    "monotone_constraints": {"ratio_prix": 1},
    "random_seed": 42,

    # ---- Cherchés ----
    "learning_rate":  trial.suggest_float("lr", 0.01, 0.1, log=True),
    "depth":          trial.suggest_int("depth", 4, 10),
    "l2_leaf_reg":    trial.suggest_float("l2", 1, 30, log=True),
    "random_strength":trial.suggest_float("rs", 1e-3, 10, log=True),
    "border_count":   trial.suggest_int("borders", 128, 1024),
    "rsm":            trial.suggest_float("rsm", 0.8, 1.0),
    "bootstrap_type": trial.suggest_categorical("bt",
                          ["Bayesian", "Bernoulli", "MVS"]),
    "leaf_estimation_iterations": trial.suggest_int("lei", 1, 10),
}
# conditionnels :
if params["bootstrap_type"] == "Bayesian":
    params["bagging_temperature"] = trial.suggest_float("bag", 0.0, 1.0)
else:
    params["subsample"] = trial.suggest_float("sub", 0.6, 1.0)
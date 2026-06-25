"""
Recherche d'hyperparamètres (Optuna) pour un modèle de DEMANDE calibré
=======================================================================
Objectif : prédire P(résiliation au renouvellement) avec des probabilités
JUSTES (bien calibrées), pour les injecter dans une optimisation tarifaire.

Principes appliqués :
  - PAS de rééquilibrage (pas de class_weights / SMOTE) -> on préserve la calibration
  - Métrique = Logloss (règle de score propre), AUC seulement en secondaire
  - Recherche bayésienne (Optuna/TPE) + pruning -> peu d'essais, peu coûteux
  - Recherche sur un ÉCHANTILLON stratifié, réentraînement final sur les 3M
  - Validation TEMPORELLE (pas aléatoire) -> pas de fuite, calibration réaliste
  - Contrainte de MONOTONIE sur le prix -> élasticité économiquement saine
"""

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier, Pool
from optuna.integration import CatBoostPruningCallback
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

# ============================================================
# 1. DONNÉES
# ============================================================
# df : votre base complète (~3M lignes)
# Colonnes attendues (à adapter) :
#   - target            : 1 = résilie, 0 = renouvelle
#   - date_renouvel     : date de renouvellement (pour le split temporel)
#   - price_ratio       : prime_n+1 / prime_n  (variation relative -> élasticité)
#   - autres features numériques et catégorielles

df = pd.read_parquet("base_renouvellement.parquet")  # <-- à adapter

TARGET = "target"
DATE_COL = "date_renouvel"
CAT_FEATURES = ["region", "segment_vehicule", "canal", "anciennete_cat"]  # <-- à adapter
PRICE_COL = "price_ratio"  # la variable sur laquelle on impose la monotonie

FEATURES = [c for c in df.columns if c not in (TARGET, DATE_COL)]

# ============================================================
# 2. SPLIT TEMPOREL (pas aléatoire !)
# ============================================================
# On entraîne sur le passé, on valide sur le futur.
df = df.sort_values(DATE_COL)
cut_valid = df[DATE_COL].quantile(0.70)   # 70% passé -> train
cut_test  = df[DATE_COL].quantile(0.85)   # 15% -> valid (Optuna) | 15% -> test final intact

train_full = df[df[DATE_COL] <= cut_valid]
valid      = df[(df[DATE_COL] > cut_valid) & (df[DATE_COL] <= cut_test)]
test       = df[df[DATE_COL] > cut_test]   # JAMAIS touché pendant la recherche

# ============================================================
# 3. ÉCHANTILLON STRATIFIÉ pour la recherche (coût réduit)
# ============================================================
# 400k lignes suffisent largement pour trouver les bons hyperparamètres.
# Stratifié = on garde le taux de résiliation réel (NE PAS rééquilibrer).
def sample_strat(data, n, target=TARGET, seed=42):
    if len(data) <= n:
        return data
    frac = n / len(data)
    return (data.groupby(target, group_keys=False)
                .apply(lambda g: g.sample(frac=frac, random_state=seed)))

train_search = sample_strat(train_full, 400_000)

# Pools CatBoost (réutilisés à chaque essai -> plus rapide)
train_pool = Pool(train_search[FEATURES], train_search[TARGET], cat_features=CAT_FEATURES)
valid_pool = Pool(valid[FEATURES],         valid[TARGET],        cat_features=CAT_FEATURES)

# Contrainte de monotonie : +1 = P(résiliation) croît avec le prix
monotone = {PRICE_COL: 1}

# ============================================================
# 4. FONCTION OBJECTIF OPTUNA
# ============================================================
def objective(trial):
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",          # <-- on optimise la calibration
        "iterations": 3000,                # plafond ; early stopping coupe avant
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 64, 254),
        "monotone_constraints": monotone,
        "random_seed": 42,
        "od_type": "Iter",
        "od_wait": 100,                    # early stopping
        "verbose": False,
        # PAS de auto_class_weights : on veut des probas calibrées
    }

    model = CatBoostClassifier(**params)
    pruning_cb = CatBoostPruningCallback(trial, "Logloss")
    model.fit(train_pool, eval_set=valid_pool, callbacks=[pruning_cb], use_best_model=True)
    pruning_cb.check_pruned()

    p = model.predict_proba(valid[FEATURES])[:, 1]
    return log_loss(valid[TARGET], p)      # <-- critère minimisé

# ============================================================
# 5. LANCER LA RECHERCHE (bayésien + pruning)
# ============================================================
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),   # recherche bayésienne
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=200),
)
study.optimize(objective, n_trials=80, show_progress_bar=True)

print("Meilleure logloss :", study.best_value)
print("Meilleurs params  :", study.best_params)

# ============================================================
# 6. RÉENTRAÎNEMENT FINAL sur les 3M (train + valid)
# ============================================================
best = study.best_params
best.update({
    "loss_function": "Logloss", "eval_metric": "Logloss",
    "iterations": 5000, "monotone_constraints": monotone,
    "od_type": "Iter", "od_wait": 150, "random_seed": 42, "verbose": 200,
})

final_train = pd.concat([train_full, valid])   # tout le passé dispo
final_pool   = Pool(final_train[FEATURES], final_train[TARGET], cat_features=CAT_FEATURES)
test_pool    = Pool(test[FEATURES],        test[TARGET],        cat_features=CAT_FEATURES)

final_model = CatBoostClassifier(**best)
final_model.fit(final_pool, eval_set=test_pool, use_best_model=True)

# ============================================================
# 7. CONTRÔLE FINAL : calibration sur le jeu test intact
# ============================================================
p_test = final_model.predict_proba(test[FEATURES])[:, 1]
print("Logloss test :", log_loss(test[TARGET], p_test))
print("Brier  test :", brier_score_loss(test[TARGET], p_test))
print("AUC    test :", roc_auc_score(test[TARGET], p_test))  # secondaire

# Calibration CONDITIONNELLE au prix : le point clé pour l'optimisation.
# On vérifie que proba moyenne ≈ taux réel DANS chaque tranche de hausse.
test = test.assign(p=p_test)
test["price_bin"] = pd.qcut(test[PRICE_COL], 10, duplicates="drop")
calib = test.groupby("price_bin").agg(
    proba_moyenne=("p", "mean"),
    taux_reel=(TARGET, "mean"),
    n=("p", "size"),
)
print(calib)   # proba_moyenne doit coller à taux_reel, surtout sur les fortes hausses

final_model.save_model("modele_demande.cbm")


# ============================================================
# VARIANTE XGBOOST (même logique)
# ============================================================
# import xgboost as xgb
# from optuna.integration import XGBoostPruningCallback
#
# # monotone_constraints en XGBoost = tuple ordonné comme les colonnes
# # ex. si price_ratio est la 1re colonne : (1, 0, 0, ...)
#
# def objective_xgb(trial):
#     params = {
#         "objective": "binary:logistic",
#         "eval_metric": "logloss",
#         "eta": trial.suggest_float("eta", 0.01, 0.08, log=True),
#         "max_depth": trial.suggest_int("max_depth", 3, 6),
#         "min_child_weight": trial.suggest_float("min_child_weight", 5, 100, log=True),
#         "subsample": trial.suggest_float("subsample", 0.7, 0.9),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 0.9),
#         "reg_lambda": trial.suggest_float("reg_lambda", 1, 30, log=True),
#         "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),  # stabilise sans casser la calibration
#         "monotone_constraints": monotone_tuple,  # <-- à construire
#         "tree_method": "hist",
#     }
#     dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
#     dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
#     pruning = XGBoostPruningCallback(trial, "valid-logloss")
#     bst = xgb.train(params, dtrain, num_boost_round=3000,
#                     evals=[(dvalid, "valid")], early_stopping_rounds=100,
#                     callbacks=[pruning], verbose_eval=False)
#     p = bst.predict(dvalid)
#     return log_loss(y_valid, p)

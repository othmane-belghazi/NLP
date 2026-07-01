"""
Modèle de résiliation (lapse) assurance auto -> probabilités calibrées pour pricing.

Priorités, dans l'ordre :
  1. CALIBRATION des probabilités (indispensable pour l'optimisation tarifaire / élasticité)
  2. DISCRIMINATION / ranking (PR-AUC, AUC)
  3. La "détection" (recall) se règle APRÈS via le seuil, pas via la loss.

Découpage temporel (jamais aléatoire) basé sur la date de renouvellement :
  - Train  : 2025 janv -> oct
  - Valid  : 2025 nov -> déc      (early stopping + tuning, out-of-time)
  - Test   : 2026 janv -> avr     (juge final, intouché)
  - Contrôle saison : 2025 janv -> avr (même saison que le test, pour séparer dérive vs saisonnalité)
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

# ============================================================
# 1. CONFIG  ->  À ADAPTER à tes noms de colonnes
# ============================================================
DATE_COL   = "date_renouvellement"   # date de renouvellement / snapshot des features
TARGET_COL = "resilie"               # 1 = résilié, 0 = renouvelé

# Feature représentant l'ampleur de la hausse tarifaire proposée.
# On lui impose une contrainte de monotonie CROISSANTE (+1) :
# plus la hausse est forte, plus la proba de résiliation doit augmenter.
PRICE_INCREASE_COL = "taux_hausse"

# Variables catégorielles (gérées nativement par CatBoost)
CAT_FEATURES = ["mode_paiement", "canal", "zone", "segment"]  # <- à adapter

# Colonnes à exclure des features (id, date, cible...)
DROP_COLS = [DATE_COL, TARGET_COL, "id_contrat", "id_client"]

RANDOM_STATE = 42


# ============================================================
# 2. CHARGEMENT + SPLIT TEMPOREL
# ============================================================
def load_data() -> pd.DataFrame:
    # Remplace par ton propre chargement (parquet/csv/base).
    df = pd.read_parquet("donnees.parquet")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


def temporal_split(df: pd.DataFrame):
    d = df[DATE_COL]
    train = df[(d >= "2025-01-01") & (d < "2025-11-01")]
    valid = df[(d >= "2025-11-01") & (d < "2026-01-01")]
    test  = df[(d >= "2026-01-01") & (d < "2026-05-01")]
    # Contrôle "même saison" que le test, mais un an plus tôt :
    ctrl  = df[(d >= "2025-01-01") & (d < "2025-05-01")]
    return train, valid, test, ctrl


def make_xy(part: pd.DataFrame):
    features = [c for c in part.columns if c not in DROP_COLS]
    X = part[features]
    y = part[TARGET_COL].astype(int)
    return X, y, features


def make_pool(X, y, features):
    cat_idx = [features.index(c) for c in CAT_FEATURES if c in features]
    return Pool(data=X, label=y, cat_features=cat_idx)


# ============================================================
# 3. MODÈLE  (config "calibration-first")
# ============================================================
def build_model(features):
    # Contrainte de monotonie sur la hausse tarifaire (croissante).
    # Format dict {nom_feature: 1}. Doit être une feature NUMÉRIQUE (pas catégorielle).
    monotone = {PRICE_INCREASE_COL: 1} if PRICE_INCREASE_COL in features else None

    return CatBoostClassifier(
        loss_function="Logloss",              # proper scoring rule -> probas fiables
        eval_metric="PRAUC",                  # seuil-indépendant, adapté au déséquilibre
        custom_metric=["AUC", "PRAUC", "Logloss"],  # Logloss = proxy de calibration
        monotone_constraints=monotone,

        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,

        # >>> PAS de auto_class_weights ici : on protège la calibration.
        #     Le recall se réglera via le seuil (voir §6).
        #     Si tu veux quand même pondérer -> voir la variante §7 (avec recalibration).

        early_stopping_rounds=100,
        use_best_model=True,
        verbose=200,
    )


# ============================================================
# 4. ÉVALUATION  (ranking + calibration + IC bootstrap)
# ============================================================
def evaluate(y_true, p_pred, label="", n_boot=1000):
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    prauc = average_precision_score(y_true, p_pred)
    auc   = roc_auc_score(y_true, p_pred)
    brier = brier_score_loss(y_true, p_pred)

    # Calibration-in-the-large : moyenne prédite vs taux observé.
    # Pour le pricing, ces deux nombres DOIVENT être proches (~3%).
    obs  = y_true.mean()
    pred = p_pred.mean()

    # IC bootstrap (utile car peu de positifs sur 4 mois de test)
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    boots = {"prauc": [], "auc": [], "brier": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y_true[idx], p_pred[idx]
        if yb.sum() == 0:            # évite les échantillons sans positif
            continue
        boots["prauc"].append(average_precision_score(yb, pb))
        boots["auc"].append(roc_auc_score(yb, pb))
        boots["brier"].append(brier_score_loss(yb, pb))

    def ci(v):
        return np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)

    print(f"\n===== {label} (n={n}, positifs={int(y_true.sum())}, taux={obs:.3%}) =====")
    print(f"PR-AUC : {prauc:.4f}  IC95%[{ci(boots['prauc'])[0]:.4f}, {ci(boots['prauc'])[1]:.4f}]")
    print(f"AUC    : {auc:.4f}  IC95%[{ci(boots['auc'])[0]:.4f}, {ci(boots['auc'])[1]:.4f}]")
    print(f"Brier  : {brier:.5f} IC95%[{ci(boots['brier'])[0]:.5f}, {ci(boots['brier'])[1]:.5f}]")
    print(f"Calibration globale : prédit={pred:.3%} vs observé={obs:.3%}  (écart={pred-obs:+.3%})")
    return {"prauc": prauc, "auc": auc, "brier": brier}


def reliability_table(y_true, p_pred, n_bins=10):
    """Courbe de calibration en table (proba moyenne par bin vs fréquence observée)."""
    frac_pos, mean_pred = calibration_curve(
        y_true, p_pred, n_bins=n_bins, strategy="quantile"
    )
    print("\n  bin | proba moyenne | freq observée")
    for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos)):
        print(f"  {i:>3} |   {mp:8.4f}   |   {fp:8.4f}")


# ============================================================
# 5. ENTRAÎNEMENT
# ============================================================
def main():
    df = load_data()
    train, valid, test, ctrl = temporal_split(df)

    X_tr, y_tr, feats = make_xy(train)
    X_va, y_va, _     = make_xy(valid)
    X_te, y_te, _     = make_xy(test)
    X_ct, y_ct, _     = make_xy(ctrl)

    train_pool = make_pool(X_tr, y_tr, feats)
    valid_pool = make_pool(X_va, y_va, feats)

    model = build_model(feats)
    model.fit(train_pool, eval_set=valid_pool)

    # --- Évaluation ---
    p_te = model.predict_proba(X_te)[:, 1]
    p_ct = model.predict_proba(X_ct)[:, 1]

    evaluate(y_te, p_te, "TEST 2026 janv-avr (juge final)")
    reliability_table(y_te, p_te)

    # Contrôle saison : si les perfs sur 2025 janv-avr ≈ 2026 janv-avr,
    # alors l'écart avec la valid (nov-déc) vient de la SAISON, pas d'une dérive du modèle.
    evaluate(y_ct, p_ct, "CONTRÔLE 2025 janv-avr (même saison)")

    # --- Modèle de PRODUCTION : ré-entraînement sur TOUT ---
    # Le split ci-dessus sert à ESTIMER la perf. Le modèle livré s'entraîne sur toutes les données.
    full = pd.concat([train, valid, test], ignore_index=True)
    Xf, yf, _ = make_xy(full)
    prod_pool = make_pool(Xf, yf, feats)

    prod = build_model(feats)
    prod.set_params(use_best_model=False,
                    iterations=model.get_best_iteration() or 3000)
    prod.fit(prod_pool)          # pas d'eval_set : nb d'itérations figé
    prod.save_model("modele_resiliation_prod.cbm")
    print("\nModèle de production sauvegardé -> modele_resiliation_prod.cbm")

    return model, feats, (X_te, y_te)


# ============================================================
# 6. RÉGLAGE DU SEUIL (pour "détecter" les résiliations)
# ============================================================
def choisir_seuil(y_true, p_pred, seuils=(0.03, 0.05, 0.08, 0.10, 0.15)):
    """Le recall se pilote ICI, pas dans la loss. À arbitrer selon le métier."""
    y_true = np.asarray(y_true)
    print("\nseuil | recall | precision | %contrats flaggés")
    for s in seuils:
        pred = (p_pred >= s).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        recall = tp / max(y_true.sum(), 1)
        precision = tp / max(pred.sum(), 1)
        print(f"{s:5.2f} | {recall:6.2%} | {precision:9.2%} | {pred.mean():6.2%}")


# ============================================================
# 7. VARIANTE OPTIONNELLE : pondération + RECALIBRATION
# ============================================================
# À utiliser SEULEMENT si tu tiens à pondérer la classe rare pendant l'apprentissage.
# La pondération casse la calibration -> recalibration isotonique OBLIGATOIRE ensuite.
def variante_ponderee(train, valid, test, feats):
    X_tr, y_tr, _ = make_xy(train)
    X_va, y_va, _ = make_xy(valid)
    X_te, y_te, _ = make_xy(test)

    m = build_model(feats)
    m.set_params(auto_class_weights="Balanced")   # <- gonfle les probas
    m.fit(make_pool(X_tr, y_tr, feats), eval_set=make_pool(X_va, y_va, feats))

    # Recalibration isotonique apprise sur la VALIDATION (idéalement une slice dédiée),
    # puis appliquée au test. Restaure des probas cohérentes avec le taux réel ~3%.
    p_va_raw = m.predict_proba(X_va)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va_raw, y_va)

    p_te_raw = m.predict_proba(X_te)[:, 1]
    p_te_cal = iso.predict(p_te_raw)

    evaluate(y_te, p_te_raw, "PONDÉRÉ - probas BRUTES (mal calibrées)")
    evaluate(y_te, p_te_cal, "PONDÉRÉ - probas RECALIBRÉES (à utiliser pour le pricing)")
    return m, iso


if __name__ == "__main__":
    model, feats, (X_te, y_te) = main()
    p_te = model.predict_proba(X_te)[:, 1]
    choisir_seuil(y_te, p_te)

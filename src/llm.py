"""
Visuels de diagnostic pour le modèle de résiliation (usage pricing).
À exécuter APRÈS l'entraînement, avec :
    y_te   : labels du test 2026 (array / Series)
    p_te   : probas prédites sur le test (model.predict_proba(X_te)[:,1])
    model  : le CatBoostClassifier entraîné
    df_te  : le dataframe de test (pour le graphe d'élasticité)

Chaque fonction sauvegarde un PNG et peut aussi s'afficher (plt.show()).
Ordre d'importance pour TON cas :
    1. calibration        -> fiabilité des probas (critique pricing)
    2. elasticite_prix    -> le modèle capte-t-il bien la sensibilité à la hausse ?
    3. gains_lift         -> valeur business du ciblage
    4. distributions      -> pouvoir de séparation
    5. roc_pr / ks        -> discrimination
    6. learning_curve     -> diagnostic sur/sous-apprentissage
    7. importance         -> drivers du modèle
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
)
from sklearn.calibration import calibration_curve

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


# ------------------------------------------------------------
# 1. CALIBRATION (le plus important pour le pricing)
#    Courbe de fiabilité + histogramme des probas prédites.
# ------------------------------------------------------------
def plot_calibration(y, p, n_bins=10, fname="01_calibration.png"):
    y, p = np.asarray(y), np.asarray(p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7),
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax1.plot([0, max(mean_pred) * 1.1] * 1, [0, max(mean_pred) * 1.1],
             "--", color="grey", label="Calibration parfaite")
    ax1.plot(mean_pred, frac_pos, "o-", color="#2b6cb0", label="Modèle")
    ax1.set_ylabel("Fréquence observée")
    ax1.set_title(f"Calibration — prédit moyen={p.mean():.3%} vs observé={y.mean():.3%}")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.hist(p, bins=50, color="#2b6cb0", alpha=0.7)
    ax2.set_xlabel("Probabilité prédite"); ax2.set_ylabel("Nb contrats")
    ax2.set_yscale("log"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname); plt.show()


# ------------------------------------------------------------
# 2. ÉLASTICITÉ PRIX (validation métier de la contrainte monotone)
#    Par décile de taux_hausse : proba prédite moyenne vs résiliation observée.
#    On veut voir les deux croître ensemble -> le modèle capte la pente.
# ------------------------------------------------------------
def plot_elasticite_prix(taux_hausse, y, p, n_bins=10, fname="02_elasticite.png"):
    th, y, p = np.asarray(taux_hausse), np.asarray(y), np.asarray(p)
    q = np.quantile(th, np.linspace(0, 1, n_bins + 1))
    q[-1] += 1e-9
    idx = np.clip(np.digitize(th, q[1:-1]), 0, n_bins - 1)

    centres, obs, pred = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        centres.append(th[m].mean()); obs.append(y[m].mean()); pred.append(p[m].mean())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(centres, obs, "o-", color="#c53030", label="Résiliation observée")
    ax.plot(centres, pred, "s--", color="#2b6cb0", label="Proba prédite moyenne")
    ax.set_xlabel("Taux de hausse tarifaire (moyenne par décile)")
    ax.set_ylabel("Taux de résiliation")
    ax.set_title("Sensibilité à la hausse — le modèle suit-il l'observé ?")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname); plt.show()


# ------------------------------------------------------------
# 3. GAINS CUMULÉS + LIFT PAR DÉCILE (valeur business du ciblage)
# ------------------------------------------------------------
def plot_gains_lift(y, p, fname="03_gains_lift.png"):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted) / y.sum()
    frac_pop = np.arange(1, len(y) + 1) / len(y)

    # Lift par décile
    deciles = np.array_split(y_sorted, 10)
    base = y.mean()
    lift = [d.mean() / base for d in deciles]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(frac_pop, cum_pos, color="#2b6cb0", label="Modèle")
    ax1.plot([0, 1], [0, 1], "--", color="grey", label="Aléatoire")
    ax1.set_xlabel("% du portefeuille ciblé"); ax1.set_ylabel("% des résiliés captés")
    ax1.set_title("Courbe de gains cumulés"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.bar(range(1, 11), lift, color="#2b6cb0", alpha=0.8)
    ax2.axhline(1, color="grey", ls="--")
    ax2.set_xlabel("Décile (1 = probas les + hautes)"); ax2.set_ylabel("Lift")
    ax2.set_title("Lift par décile"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname); plt.show()


# ------------------------------------------------------------
# 4. DISTRIBUTION DES SCORES PAR CLASSE (pouvoir de séparation)
# ------------------------------------------------------------
def plot_distributions(y, p, fname="04_distributions.png"):
    y, p = np.asarray(y), np.asarray(p)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, np.quantile(p, 0.999), 60)
    ax.hist(p[y == 0], bins=bins, alpha=0.6, density=True,
            color="#2b6cb0", label="Renouvelés (0)")
    ax.hist(p[y == 1], bins=bins, alpha=0.6, density=True,
            color="#c53030", label="Résiliés (1)")
    ax.set_xlabel("Probabilité prédite"); ax.set_ylabel("Densité")
    ax.set_title("Séparation des scores par classe (idéal : peu de recouvrement)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname); plt.show()


# ------------------------------------------------------------
# 5. ROC + PRECISION-RECALL
# ------------------------------------------------------------
def plot_roc_pr(y, p, fname="05_roc_pr.png"):
    y, p = np.asarray(y), np.asarray(p)
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    auc_v = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    base = y.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(fpr, tpr, color="#2b6cb0", label=f"AUC = {auc_v:.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="grey")
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(rec, prec, color="#2b6cb0", label=f"PR-AUC = {ap:.3f}")
    ax2.axhline(base, ls="--", color="grey", label=f"Aléatoire = {base:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("Precision-Recall")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname); plt.show()


# ------------------------------------------------------------
# 6. COURBE DE SEUIL (recall / precision / F1 / % flaggé)
# ------------------------------------------------------------
def plot_threshold_sweep(y, p, fname="06_seuils.png"):
    y, p = np.asarray(y), np.asarray(p)
    seuils = np.linspace(0.01, 0.30, 60)
    rec, prec, f1, flag = [], [], [], []
    for s in seuils:
        pred = (p >= s).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        r = tp / max(y.sum(), 1)
        pr = tp / max(pred.sum(), 1)
        rec.append(r); prec.append(pr)
        f1.append(2 * r * pr / max(r + pr, 1e-9)); flag.append(pred.mean())

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(seuils, rec, label="Recall", color="#c53030")
    ax.plot(seuils, prec, label="Precision", color="#2b6cb0")
    ax.plot(seuils, f1, label="F1", color="#2f855a")
    ax.plot(seuils, flag, label="% portefeuille flaggé", color="grey", ls="--")
    ax.set_xlabel("Seuil de décision"); ax.set_ylabel("Valeur")
    ax.set_title("Compromis selon le seuil (le recall se pilote ICI)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname); plt.show()


# ------------------------------------------------------------
# 7. KS (séparation des distributions cumulées)
# ------------------------------------------------------------
def plot_ks(y, p, fname="07_ks.png"):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p)
    y_s, p_s = y[order], p[order]
    cum_pos = np.cumsum(y_s) / y.sum()
    cum_neg = np.cumsum(1 - y_s) / (len(y) - y.sum())
    ks = np.max(np.abs(cum_neg - cum_pos)); ks_i = np.argmax(np.abs(cum_neg - cum_pos))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(p_s, cum_neg, color="#2b6cb0", label="Renouvelés")
    ax.plot(p_s, cum_pos, color="#c53030", label="Résiliés")
    ax.vlines(p_s[ks_i], cum_pos[ks_i], cum_neg[ks_i], color="black",
              label=f"KS = {ks:.3f}")
    ax.set_xlabel("Probabilité prédite"); ax.set_ylabel("Proportion cumulée")
    ax.set_title("Statistique KS"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname); plt.show()


# ------------------------------------------------------------
# 8. LEARNING CURVE (learn vs validation par itération)
#    Montre le plateau / sur-apprentissage observé.
# ------------------------------------------------------------
def plot_learning_curve(model, metric="PRAUC", fname="08_learning_curve.png"):
    res = model.get_evals_result()
    learn = res["learn"][metric]
    valid = res.get("validation", {}).get(metric)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(learn, label=f"learn {metric}", color="#2b6cb0")
    if valid:
        ax.plot(valid, label=f"validation {metric}", color="#c53030")
    ax.set_xlabel("Itération"); ax.set_ylabel(metric)
    ax.set_title("Courbe d'apprentissage (écart = overfit)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fname); plt.show()


# ------------------------------------------------------------
# 9. IMPORTANCE DES VARIABLES
# ------------------------------------------------------------
def plot_importance(model, top=20, fname="09_importance.png"):
    imp = model.get_feature_importance()
    names = model.feature_names_
    order = np.argsort(imp)[-top:]
    fig, ax = plt.subplots(figsize=(7, max(4, top * 0.3)))
    ax.barh(np.array(names)[order], imp[order], color="#2b6cb0")
    ax.set_xlabel("Importance"); ax.set_title(f"Top {top} variables")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(fname); plt.show()


# ------------------------------------------------------------
# Lancement de tous les visuels
# ------------------------------------------------------------
def tous_les_visuels(y_te, p_te, model, taux_hausse=None):
    plot_calibration(y_te, p_te)
    if taux_hausse is not None:
        plot_elasticite_prix(taux_hausse, y_te, p_te)
    plot_gains_lift(y_te, p_te)
    plot_distributions(y_te, p_te)
    plot_roc_pr(y_te, p_te)
    plot_threshold_sweep(y_te, p_te)
    plot_ks(y_te, p_te)
    plot_learning_curve(model)
    plot_importance(model)


# Exemple d'appel (à adapter) :
# tous_les_visuels(y_te, p_te, model, taux_hausse=df_te["taux_hausse"].values)

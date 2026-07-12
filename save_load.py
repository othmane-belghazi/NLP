# -*- coding: utf-8 -*-
"""
Élasticité prédite du portefeuille — EBM résiliation au renouvellement.

Variables tarifaires ajustées de façon COHÉRENTE à chaque choc :
    majoration_new   = majoration + choc
    prime_apres_new  = prime_avant * (1 + majoration_new)
    delta_new        = prime_apres_new - prime_avant
(prime_avant reste fixe, elle sert de base de calcul)
"""

import numpy as np
import pandas as pd

# --- À adapter : noms de colonnes ---
COL_MAJ    = "majoration"            # ex: 0.05 = +5%
COL_APRES  = "prime_apres_renouv"
COL_DELTA  = "delta_cotisation"
COL_AVANT  = "prime_avant_renouv"


def appliquer_choc(X, choc):
    """Renvoie une copie de X avec les 3 variables tarifaires recalculées."""
    Xs = X.copy()
    maj_new = X[COL_MAJ] + choc
    Xs[COL_MAJ]   = maj_new
    Xs[COL_APRES] = X[COL_AVANT] * (1 + maj_new)
    Xs[COL_DELTA] = Xs[COL_APRES] - X[COL_AVANT]
    return Xs


def courbe_retention(ebm, X, chocs=np.arange(-0.05, 0.101, 0.01)):
    """Rétention prédite du portefeuille pour chaque choc de taux."""
    lignes = []
    for c in chocs:
        Xs = appliquer_choc(X, c)
        p_resil = ebm.predict_proba(Xs)[:, 1]
        retention = 1 - p_resil
        lignes.append({
            "choc": c,
            "taux_moyen": Xs[COL_MAJ].mean(),
            "retention_contrats": retention.mean(),
            "retention_primes": np.average(retention, weights=Xs[COL_APRES]),
            "prime_conservee": (retention * Xs[COL_APRES]).sum(),
        })
    return pd.DataFrame(lignes)


def elasticite_arc(courbe, sur="retention_primes"):
    """
    Élasticité d'arc entre chocs adjacents :
        eps = (Δ rétention / rétention) / (Δ prime / prime)
    avec Δprime/prime = Δtaux / (1 + taux moyen).
    eps = -2  =>  +1% de prime  =>  -2% de rétention.
    """
    r = courbe[sur].to_numpy()
    t = courbe["taux_moyen"].to_numpy()
    r_mid, t_mid = (r[1:] + r[:-1]) / 2, (t[1:] + t[:-1]) / 2
    eps = (np.diff(r) / r_mid) / (np.diff(t) / (1 + t_mid))
    return pd.DataFrame({
        "choc_milieu": (courbe["choc"].to_numpy()[1:] + courbe["choc"].to_numpy()[:-1]) / 2,
        "elasticite": eps,
    })


if __name__ == "__main__":
    # ebm = joblib.load("ebm_resiliation.pkl")
    # X = df[ebm.feature_names_in_]

    # courbe = courbe_retention(ebm, X)
    # print(courbe)
    # print(elasticite_arc(courbe))
    # Élasticité globale du portefeuille = moyenne autour du choc 0 :
    # eps_pf = elasticite_arc(courbe).query("abs(choc_milieu) <= 0.01")["elasticite"].mean()
    pass

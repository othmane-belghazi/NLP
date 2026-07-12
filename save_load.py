# -*- coding: utf-8 -*-
"""
ÉLASTICITÉ PRÉDITE DU PORTEFEUILLE — Modèle EBM de résiliation tarifaire
=========================================================================

MÉTHODE (3 étapes) :

  1. CHOC TARIFAIRE : on décale la majoration de chaque contrat d'un
     choc c, et on recalcule les 2 autres variables tarifaires de façon
     cohérente à partir de Prime_N-1 (qui reste fixe) :

         Majoration_N     = Majoration_N + c
         Prime_N          = Prime_N-1 * (1 + Majoration_N)
         Delta_cotisation = Prime_N - Prime_N-1

  2. COURBE DE RÉTENTION : pour chaque choc c d'une grille (-5% à +10%),
     on re-score le portefeuille et on calcule la rétention prédite
     (= 1 - taux de résiliation), pondérée par la prime.

  3. ÉLASTICITÉ D'ARC entre chocs adjacents :

         eps = (Δ rétention / rétention) / (Δ prime / prime)

     L'élasticité du portefeuille = valeur autour du choc 0.
     Lecture : eps = -2  =>  +1% de prime  =>  -2% de rétention.

NB : on utilise des chocs discrets (pas de dérivée infinitésimale) car
les fonctions d'un EBM sont en escalier.
"""

import numpy as np
import pandas as pd

COL_MAJ   = "Majoration_N"
COL_PRIME = "Prime_N"
COL_DELTA = "Delta_cotisation"
COL_AVANT = "Prime_N-1"


# ÉTAPE 1 — appliquer un choc tarifaire cohérent
def appliquer_choc(X, choc):
    Xs = X.copy()
    Xs[COL_MAJ]   = X[COL_MAJ] + choc
    Xs[COL_PRIME] = X[COL_AVANT] * (1 + Xs[COL_MAJ])
    Xs[COL_DELTA] = Xs[COL_PRIME] - X[COL_AVANT]
    return Xs


# ÉTAPE 2 — courbe de rétention du portefeuille
def courbe_retention(ebm, X, chocs=np.arange(-0.05, 0.101, 0.01)):
    lignes = []
    for c in chocs:
        Xs = appliquer_choc(X, c)
        taux_resil = ebm.predict_proba(Xs)[:, 1]
        retention = 1 - taux_resil
        lignes.append({
            "choc": round(c, 3),
            "majoration_moy": Xs[COL_MAJ].mean(),
            "taux_resil_moy": taux_resil.mean(),
            "retention_primes": np.average(retention, weights=Xs[COL_PRIME]),
        })
    return pd.DataFrame(lignes)


# ÉTAPE 3 — élasticité d'arc
def elasticite_arc(courbe):
    r = courbe["retention_primes"].to_numpy()
    t = courbe["majoration_moy"].to_numpy()
    r_mid, t_mid = (r[1:] + r[:-1]) / 2, (t[1:] + t[:-1]) / 2
    eps = (np.diff(r) / r_mid) / (np.diff(t) / (1 + t_mid))
    chocs = courbe["choc"].to_numpy()
    return pd.DataFrame({
        "choc_milieu": (chocs[1:] + chocs[:-1]) / 2,
        "elasticite": eps,
    })


# ---------------------------------------------------------------------
# UTILISATION
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # ebm = joblib.load("ebm_resiliation.pkl")
    # X = df[ebm.feature_names_in_]        # doit contenir les 4 colonnes

    # courbe = courbe_retention(ebm, X)
    # elas = elasticite_arc(courbe)
    # print(courbe)
    # print(elas)

    # Élasticité prédite du portefeuille (autour du tarif réel, choc ~ 0) :
    # eps_portefeuille = elas.loc[elas["choc_milieu"].abs() <= 0.011,
    #                             "elasticite"].mean()
    # print(f"Élasticité du portefeuille : {eps_portefeuille:.2f}")
    pass

# -*- coding: utf-8 -*-
"""
Élasticité prix prédite d'un portefeuille auto à partir d'un modèle EBM
de résiliation tarifaire au renouvellement.

Hypothèses sur les données :
- ebm : ExplainableBoostingClassifier entraîné (package interpret),
        cible = 1 si résiliation dans la fenêtre d'échéance.
- X   : DataFrame des features au format attendu par le modèle.
- COL_EVOL : nom de la variable d'évolution tarifaire (ex: 0.05 = +5%
        de majoration de la prime au renouvellement).
- primes : Series des primes proposées au renouvellement (pour pondérer).

Point méthodologique CLÉ avec un EBM :
Les fonctions de forme d'un EBM sont CONSTANTES PAR MORCEAUX (escaliers).
Une dérivée avec un pas h trop petit donne 0 ou des pics artificiels.
=> On utilise des chocs discrets "métier" (ex: +/- 1 point de taux)
   et une élasticité d'ARC entre scénarios, pas une dérivée infinitésimale.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Prédiction de base
# ---------------------------------------------------------------------
def proba_resiliation(ebm, X):
    """P(résiliation) prédite pour chaque contrat."""
    return ebm.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------
# 2. Simulation d'un choc tarifaire uniforme (approche recommandée)
# ---------------------------------------------------------------------
def simuler_choc(ebm, X, col_evol, choc, mode="additif"):
    """
    Applique un choc de taux au portefeuille et renvoie P(résiliation).

    mode="additif"  : taux_simulé = taux_réel + choc
                      (on décale le tarif proposé de +/- x points)
    mode="remplace" : taux_simulé = choc pour tout le monde
                      (scénario 'et si on appliquait +x% à tous ?')
    """
    Xs = X.copy()
    if mode == "additif":
        Xs[col_evol] = X[col_evol] + choc
    elif mode == "remplace":
        Xs[col_evol] = choc
    else:
        raise ValueError("mode doit être 'additif' ou 'remplace'")
    return proba_resiliation(ebm, Xs)


def courbe_retention(ebm, X, col_evol, primes,
                     chocs=np.arange(-0.05, 0.101, 0.01),
                     mode="additif"):
    """
    Courbe de demande du portefeuille : pour chaque choc de taux,
    rétention attendue (en contrats et en primes) et prime conservée.

    Retourne un DataFrame trié par choc.
    """
    primes = np.asarray(primes, dtype=float)
    taux_reel = X[col_evol].to_numpy(dtype=float)
    lignes = []

    for c in chocs:
        p_resil = simuler_choc(ebm, X, col_evol, c, mode=mode)
        retention_i = 1.0 - p_resil

        # Prime au renouvellement sous le scénario
        if mode == "additif":
            taux_scen = taux_reel + c
        else:
            taux_scen = np.full_like(taux_reel, c)
        # primes = prime proposée au taux réel -> on la ramène au scénario
        prime_scen = primes / (1.0 + taux_reel) * (1.0 + taux_scen)

        lignes.append({
            "choc": c,
            "taux_moyen": taux_scen.mean(),
            "p_resil_moy": p_resil.mean(),
            "retention_contrats": retention_i.mean(),
            "retention_primes": np.average(retention_i, weights=primes),
            "prime_conservee": (retention_i * prime_scen).sum(),
        })

    return pd.DataFrame(lignes).sort_values("choc").reset_index(drop=True)


# ---------------------------------------------------------------------
# 3. Élasticité d'arc du portefeuille (la "bonne façon" avec un EBM)
# ---------------------------------------------------------------------
def elasticite_arc_portefeuille(courbe, sur="retention_primes"):
    """
    Élasticité d'arc entre scénarios adjacents de la courbe de rétention :

        eps = (Δ rétention / rétention_moyenne) / (Δ prime / prime_moyenne)

    Variation relative de prime induite par le choc :
        prime ~ (1 + taux) => Δprime/prime = Δtaux / (1 + taux_moyen)

    Interprétation : eps = -2 signifie qu'une hausse de prime de 1%
    fait baisser la rétention d'environ 2%.
    """
    c = courbe.copy()
    r = c[sur].to_numpy()
    t = c["taux_moyen"].to_numpy()

    r_mid = (r[1:] + r[:-1]) / 2
    t_mid = (t[1:] + t[:-1]) / 2
    d_r = np.diff(r)
    d_prime_rel = np.diff(t) / (1.0 + t_mid)

    eps = (d_r / r_mid) / d_prime_rel
    return pd.DataFrame({
        "choc_milieu": (c["choc"].to_numpy()[1:] + c["choc"].to_numpy()[:-1]) / 2,
        "elasticite": eps,
    })


# ---------------------------------------------------------------------
# 4. Élasticité locale par contrat (différences finies centrées)
# ---------------------------------------------------------------------
def elasticite_locale(ebm, X, col_evol, primes, h=0.01):
    """
    Élasticité de rétention contrat par contrat, avec un pas discret h
    (par défaut 1 point de taux, JAMAIS un h infinitésimal avec un EBM).

        sensibilite_i = [P(t+h) - P(t-h)] / (2h)       (pts de proba / pt de taux)
        eps_i = -sensibilite_i * (1 + t_i) / retention_i

    Le facteur (1+t) convertit une variation de taux en variation
    relative de prime : d(prime)/prime = dt / (1+t).
    """
    t = X[col_evol].to_numpy(dtype=float)
    p_plus = simuler_choc(ebm, X, col_evol, +h, mode="additif")
    p_moins = simuler_choc(ebm, X, col_evol, -h, mode="additif")
    p0 = proba_resiliation(ebm, X)
    retention = 1.0 - p0

    sensibilite = (p_plus - p_moins) / (2.0 * h)
    eps = -sensibilite * (1.0 + t) / np.clip(retention, 1e-6, None)

    out = pd.DataFrame({
        "p_resil": p0,
        "sensibilite_pts": sensibilite,
        "elasticite": eps,
        "prime": np.asarray(primes, dtype=float),
    }, index=X.index)
    return out


def elasticite_portefeuille(elas_locales, ponderation="prime"):
    """Agrégation : élasticité moyenne pondérée par la prime (ou simple)."""
    if ponderation == "prime":
        w = elas_locales["prime"].to_numpy()
        return np.average(elas_locales["elasticite"], weights=w)
    return elas_locales["elasticite"].mean()


# ---------------------------------------------------------------------
# 5. Garde-fou : support observé de la variable tarifaire
# ---------------------------------------------------------------------
def verifier_support(X, col_evol, chocs, quantiles=(0.01, 0.99)):
    """
    L'élasticité n'est fiable que dans la plage d'évolutions tarifaires
    OBSERVÉES à l'entraînement. Au-delà, l'EBM extrapole en plateau
    (dernier bin constant) => élasticité artificiellement nulle.
    """
    q_lo, q_hi = X[col_evol].quantile(list(quantiles))
    t = X[col_evol].to_numpy(dtype=float)
    hors = [c for c in chocs
            if (t + c).min() < q_lo or (t + c).max() > q_hi]
    if hors:
        print(f"⚠️ Chocs sortant du support observé [{q_lo:.3f}, {q_hi:.3f}] : "
              f"{[round(c, 3) for c in hors]} — élasticité non fiable sur ces points.")
    return q_lo, q_hi


# ---------------------------------------------------------------------
# Exemple d'utilisation
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # ebm, X, primes à charger selon votre environnement, ex :
    # import joblib
    # ebm = joblib.load("ebm_resiliation.pkl")
    # df = pd.read_parquet("portefeuille_renouvellement.parquet")
    # X = df[ebm.feature_names_in_]
    # primes = df["prime_renouvellement"]
    COL_EVOL = "evolution_tarif"   # à adapter

    chocs = np.arange(-0.05, 0.101, 0.01)
    # verifier_support(X, COL_EVOL, chocs)

    # 1) Courbe de rétention du portefeuille + élasticité d'arc (recommandé)
    # courbe = courbe_retention(ebm, X, COL_EVOL, primes, chocs)
    # eps_arc = elasticite_arc_portefeuille(courbe)
    # print(courbe) ; print(eps_arc)

    # 2) Élasticité locale + agrégation pondérée par la prime
    # elas = elasticite_locale(ebm, X, COL_EVOL, primes, h=0.01)
    # print("Élasticité portefeuille (pondérée prime) :",
    #       elasticite_portefeuille(elas))

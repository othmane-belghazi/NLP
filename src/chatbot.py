"""
Élasticité PRÉDITE à partir du modèle CatBoost de résiliation.

Principe : le modèle est la fonction de réponse. On simule (ceteris paribus)
plusieurs niveaux de majoration, on met à jour de façon COHÉRENTE les variables
qui en dépendent, on re-prédit, puis :
  - courbe de réponse (résiliation / rétention moyenne du portefeuille)
  - élasticité ponctuelle par différence finie centrée

Colonnes attendues dans df (features du modèle) :
  PRIME_PREV_COL   : PrimeN-1 (connu)
  MAJORATION_COL   : MajorationN   (taux, ex: 0.10 = +10%)
  PRIME_COL        : PrimeN
  DELTA_COL        : Delta_cotis
+ toutes les autres features du modèle.

Relations imposées (majoration = TAUX) :
  PrimeN     = PrimeN-1 * (1 + m)
  Delta_cotis = PrimeN - PrimeN-1 = PrimeN-1 * m
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- noms de colonnes (à adapter) ---
PRIME_PREV_COL = "PrimeN_1"
MAJORATION_COL = "MajorationN"
PRIME_COL      = "PrimeN"
DELTA_COL      = "Delta_cotis"
MAJORATION_EST_TAUX = True   # False si MajorationN est un MONTANT, pas un taux


# ------------------------------------------------------------
# 1. Construire un scénario cohérent pour un niveau de majoration m
# ------------------------------------------------------------
def scenario(df, m):
    """Retourne une copie de df avec Majoration/Prime/Delta recalculés pour m."""
    d = df.copy()
    p0 = d[PRIME_PREV_COL].values
    if MAJORATION_EST_TAUX:
        prime_n = p0 * (1.0 + m)
        delta   = p0 * m
    else:                                   # m est un montant absolu
        prime_n = p0 + m
        delta   = np.full_like(p0, m, dtype=float)
    d[MAJORATION_COL] = m
    d[PRIME_COL]      = prime_n
    d[DELTA_COL]      = delta
    return d


def _proba(model, d):
    return model.predict_proba(d[model.feature_names_])[:, 1]


# ------------------------------------------------------------
# 2. Courbe de réponse du portefeuille (ou d'un segment)
#    -> on MOYENNE les probas par contrat, pas l'inverse.
# ------------------------------------------------------------
def courbe_reponse(model, df, grille=None):
    if grille is None:
        grille = np.round(np.arange(-0.05, 0.301, 0.01), 3)   # -5% à +30%
    lapse = np.array([_proba(model, scenario(df, m)).mean() for m in grille])
    return pd.DataFrame({
        "majoration": grille,
        "resiliation": lapse,
        "retention": 1.0 - lapse,
    })


# ------------------------------------------------------------
# 3. Élasticité PONCTUELLE par différence finie centrée
#    (au niveau contrat, au point m0 propre à chaque contrat)
# ------------------------------------------------------------
def elasticite_ponctuelle(model, df, m0, eps=0.01):
    """
    m0  : niveau de majoration où évaluer l'élasticité (scalaire ou array par contrat).
    eps : demi-pas (0.01 = 1 point de majoration).

    Retourne un DataFrame par contrat avec :
      - dLapse_dm         : dérivée de la proba de résiliation p/r à la majoration
      - semi_elast_pt     : points de résiliation en plus par +1 point de majoration
      - elast_retention   : élasticité économique (rétention p/r à la prime), sans dimension
    """
    m0 = np.asarray(m0, dtype=float)
    if m0.ndim == 0:
        m0 = np.full(len(df), float(m0))

    # différence centrée : on doit gérer un m0 différent par contrat
    def proba_at(m_vec):
        d = df.copy()
        p0 = d[PRIME_PREV_COL].values
        if MAJORATION_EST_TAUX:
            d[PRIME_COL] = p0 * (1.0 + m_vec)
            d[DELTA_COL] = p0 * m_vec
        else:
            d[PRIME_COL] = p0 + m_vec
            d[DELTA_COL] = m_vec
        d[MAJORATION_COL] = m_vec
        return _proba(model, d)

    p_plus  = proba_at(m0 + eps)
    p_minus = proba_at(m0 - eps)
    dlapse_dm = (p_plus - p_minus) / (2 * eps)      # par unité de m (+100%)

    lapse_0   = proba_at(m0)
    retention = 1.0 - lapse_0

    # Élasticité économique de la rétention p/r à la prime :
    #   Prime = P0*(1+m) -> dPrime/Prime = dm/(1+m)
    #   R = 1 - lapse    -> dR = -dlapse
    #   E = (dR/R)/(dPrime/Prime) = -(dLapse/dm)*(1+m)/R
    if MAJORATION_EST_TAUX:
        elast_ret = -dlapse_dm * (1.0 + m0) / np.clip(retention, 1e-6, None)
    else:
        elast_ret = np.full(len(df), np.nan)        # non défini simplement si montant

    return pd.DataFrame({
        "m0": m0,
        "lapse_0": lapse_0,
        "dLapse_dm": dlapse_dm,
        "semi_elast_pt": dlapse_dm * 0.01,          # par +1 point de majoration
        "elast_retention": elast_ret,
    })


# ------------------------------------------------------------
# 4. Tracé
# ------------------------------------------------------------
def plot_elasticite(courbe, fname="elasticite.png"):
    m = courbe["majoration"].values * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    ax1.plot(m, courbe["resiliation"] * 100, "o-", color="#c53030", label="Résiliation")
    ax1.plot(m, courbe["retention"] * 100, "s--", color="#2b6cb0", label="Rétention")
    ax1.set_xlabel("Majoration (%)"); ax1.set_ylabel("Taux (%)")
    ax1.set_title("Courbe de réponse du portefeuille")
    ax1.legend(); ax1.grid(alpha=0.3)

    # pente locale (semi-élasticité) = dérivée de la courbe de résiliation
    slope = np.gradient(courbe["resiliation"].values, courbe["majoration"].values) * 0.01
    ax2.plot(m, slope * 100, "o-", color="#2f855a")
    ax2.set_xlabel("Majoration (%)")
    ax2.set_ylabel("Points de résiliation par +1 pt")
    ax2.set_title("Semi-élasticité locale")
    ax2.grid(alpha=0.3)

    fig.tight_layout(); fig.savefig(fname); plt.show()


# ------------------------------------------------------------
# Exemple d'utilisation
# ------------------------------------------------------------
if __name__ == "__main__":
    # df_te : contrats de test avec PrimeN_1 et toutes les features du modèle
    courbe = courbe_reponse(model, df_te)                    # noqa: F821
    print(courbe)
    plot_elasticite(courbe)

    # Élasticité au niveau de majoration réellement envisagé (ex: +8% partout,
    # ou la colonne de majoration cible propre à chaque contrat) :
    elast = elasticite_ponctuelle(model, df_te, m0=0.08)     # noqa: F821
    print(elast.describe())
    # -> semi_elast_pt : combien de points de résiliation en plus par +1 pt de hausse
    # -> elast_retention : élasticité-prix de la rétention (négative attendue)

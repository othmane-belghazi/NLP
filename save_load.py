import numpy as np
import pandas as pd

def _predict_probability(model, X):
    """
    Retourne la probabilité prédite de résiliation.
    - Si le modèle a predict_proba, on prend la classe 1
    - Sinon on utilise predict
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Cas classique binaire
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    return np.asarray(model.predict(X), dtype=float)


def _weighted_mean(values, weights=None):
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=float)
    return float(np.average(values, weights=weights))


def build_price_scenario(
    df,
    price_increase_pct,
    prime_prev_col="Prime_Nm1",
    prime_col="Prime_N",
    delta_col="Delta_cotisation",
    majoration_col="Majoration_N",
):
    """
    Construit un scénario de prix.

    Hypothèse simple :
    - Prime_N augmente de price_increase_pct
    - Delta_cotisation = Prime_N - Prime_Nm1
    - Majoration_N est ajustée au même taux (à adapter si votre règle métier est différente)
    """
    scen = df.copy()

    scen[prime_col] = scen[prime_col] * (1.0 + price_increase_pct)
    scen[delta_col] = scen[prime_col] - scen[prime_prev_col]

    # Règle simple et cohérente.
    # Si Majoration_N a une autre définition chez vous, remplacez cette ligne.
    scen[majoration_col] = scen[majoration_col] * (1.0 + price_increase_pct)

    return scen


def portfolio_elasticity_ebm(
    model,
    df,
    feature_cols,
    levels=(0.00, 0.02, 0.05, 0.10),
    weight_col=None,
    prime_prev_col="Prime_Nm1",
    prime_col="Prime_N",
    delta_col="Delta_cotisation",
    majoration_col="Majoration_N",
):
    """
    Calcule l'élasticité prédite du portefeuille pour plusieurs niveaux de hausse de prime.

    Retour :
    - un DataFrame avec les résultats par niveau
    - le taux de base du portefeuille
    - la prime moyenne de base du portefeuille
    """
    weights = df[weight_col].to_numpy() if weight_col is not None else None

    # Base
    X_base = df[feature_cols]
    p0 = _predict_probability(model, X_base)
    r0 = _weighted_mean(p0, weights)
    prime0 = _weighted_mean(df[prime_col], weights)

    results = []

    for lvl in levels:
        scen = build_price_scenario(
            df=df,
            price_increase_pct=lvl,
            prime_prev_col=prime_prev_col,
            prime_col=prime_col,
            delta_col=delta_col,
            majoration_col=majoration_col,
        )

        X_scen = scen[feature_cols]
        p1 = _predict_probability(model, X_scen)
        r1 = _weighted_mean(p1, weights)
        prime1 = _weighted_mean(scen[prime_col], weights)

        delta_rate_pct = np.nan if r0 == 0 else (r1 - r0) / r0
        delta_prime_pct = np.nan if prime0 == 0 else (prime1 - prime0) / prime0
        elasticity = np.nan if (not np.isfinite(delta_prime_pct) or delta_prime_pct == 0) else delta_rate_pct / delta_prime_pct

        results.append({
            "niveau_hausse": lvl,
            "prime_base": prime0,
            "prime_scenario": prime1,
            "taux_resiliation_base": r0,
            "taux_resiliation_scenario": r1,
            "variation_rate_pct": delta_rate_pct,
            "variation_prime_pct": delta_prime_pct,
            "elasticite_portefeuille": elasticity,
        })

    return pd.DataFrame(results), r0, prime0
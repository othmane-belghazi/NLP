"""
=====================================================================
 PRET-A-PARTIR  --  Optimisation tarifaire (concours AXA)
=====================================================================
 Objectif du concours :
   Step 1 : etre a moins de 20% du meilleur profit  -> il faut etre RENTABLE
   Step 2 : parmi les qualifies, le PLUS GROS VOLUME de prime gagne
 => Strategie : maximiser le VOLUME sous contrainte "profit >= cible".

 Methode : Lagrangien. Pour un multiplicateur mu fixe, le probleme se
 DECOUPLE par quote -> on optimise chaque client independamment par une
 simple recherche sur grille (optimisation discrete). On balaye mu pour
 tracer la frontiere profit<->volume et choisir le point voulu.

 Modele de conversion (logistique, elasticite lambda par profil) :
       conv(p) = 1 / (1 + exp( lambda * (p / p_ref - 1) ))
   - p_ref = prime de reference (ici la prime "incumbent" du marche)
   - a p = p_ref  -> conv = 0.5
   - lambda eleve  = client tres sensible au prix (elastique)
 >>> REMPLACE cette fonction par la formule exacte fournie par le concours.
=====================================================================
"""

import numpy as np
import pandas as pd

# =====================================================================
# 1) HYPOTHESES P&L  (a remplir avec les chiffres fournis par le concours)
# =====================================================================
COMMISSION  = 0.20   # commission Pret-a-Partir, en % de la prime
EXPENSE_VAR = 0.10   # frais variables, en % de la prime
EXPENSE_FIX = 5.0    # frais fixes par police, en EUR

# Fraction de prime conservee avant sinistres & frais fixes
K = 1.0 - COMMISSION - EXPENSE_VAR          # ici 0.70

# =====================================================================
# 2) DONNEES QUOTE
#    >>> REMPLACE ce bloc synthetique par TES donnees reelles.
#    Colonnes attendues :
#       QuoteID, pure_premium (deja calcule par toi), p_ref (incumbent),
#       lambda (elasticite du profil), + variables de profil pour le
#       tarif agent : age_band, destination, duration_band, coverage
# =====================================================================
def make_synthetic_quotes(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    age_band      = rng.choice(["18-30", "31-50", "51-70", "70+"], n, p=[.3, .35, .25, .10])
    destination   = rng.choice(["Europe", "Monde", "USA-Canada"], n, p=[.55, .30, .15])
    duration_band = rng.choice(["court", "moyen", "long"], n, p=[.5, .35, .15])
    coverage      = rng.choice(["basic", "premium"], n, p=[.6, .4])

    # Pure premium "vrai" qui depend du profil (toi tu l'as deja calcule)
    base = 20.0
    fa = {"18-30": 1.0, "31-50": 0.9, "51-70": 1.4, "70+": 2.2}
    fd = {"Europe": 1.0, "Monde": 1.6, "USA-Canada": 2.3}
    fr = {"court": 1.0, "moyen": 1.5, "long": 2.4}
    fc = {"basic": 1.0, "premium": 1.7}
    pp = base * np.array([fa[a]*fd[d]*fr[r]*fc[c]
                          for a, d, r, c in zip(age_band, destination, duration_band, coverage)])
    pp *= rng.lognormal(0, 0.15, n)          # bruit individuel

    # Prime incumbent (sous-tarifee : c'est tout le probleme du portefeuille)
    p_ref = pp * rng.uniform(1.05, 1.20, n)  # marge brute faible -> combine > 100%

    # Elasticite par profil : les jeunes / Europe / court = plus sensibles
    lam = np.full(n, 6.0)
    lam += np.where(age_band == "18-30", 3.0, 0.0)
    lam += np.where(destination == "USA-Canada", -2.0, 0.0)   # captifs, moins sensibles
    lam += np.where(coverage == "premium", -1.5, 0.0)
    lam = np.clip(lam, 2.0, 12.0)

    return pd.DataFrame({
        "QuoteID": np.arange(1, n + 1),
        "pure_premium": pp.round(2),
        "p_ref": p_ref.round(2),
        "lambda": lam.round(2),
        "age_band": age_band, "destination": destination,
        "duration_band": duration_band, "coverage": coverage,
    })


# =====================================================================
# 3) BRIQUES DE BASE
# =====================================================================
def conversion(price, p_ref, lam):
    """Probabilite de conversion logistique (vectorise)."""
    return 1.0 / (1.0 + np.exp(lam * (price / p_ref - 1.0)))

def breakeven(pure_premium):
    """Prime technique (profit = 0)."""
    return (pure_premium + EXPENSE_FIX) / K

def profit_per_sale(price, pure_premium):
    """Profit si la police est vendue (EUR)."""
    return K * price - EXPENSE_FIX - pure_premium


# =====================================================================
# 4) OPTIMISATION LAGRANGIENNE DISCRETE
#    Pour mu donne, chaque quote maximise sur une grille de prix :
#       conv(p) * [ p + mu * profit(p) ]
#    (= volume + mu*profit ; mu=0 -> volume pur, mu grand -> profit)
# =====================================================================
def optimize_for_mu(df, mu, grid):
    """Retourne le prix optimal par quote pour un multiplicateur mu.
    grid = multiplicateurs appliques a la prime technique (break-even)."""
    be   = breakeven(df["pure_premium"].to_numpy())          # (n,)
    pref = df["p_ref"].to_numpy()
    lam  = df["lambda"].to_numpy()
    pp   = df["pure_premium"].to_numpy()

    # Matrice des prix candidats : (n quotes) x (m points de grille)
    prices = be[:, None] * grid[None, :]                     # (n, m)
    conv   = conversion(prices, pref[:, None], lam[:, None]) # (n, m)
    prof   = K * prices - EXPENSE_FIX - pp[:, None]          # (n, m)

    score  = conv * (prices + mu * prof)                     # objectif lagrangien
    best   = np.argmax(score, axis=1)                        # meilleur point par quote

    idx = np.arange(len(df))
    return pd.DataFrame({
        "price_opt": prices[idx, best],
        "conv":      conv[idx, best],
        "exp_volume": conv[idx, best] * prices[idx, best],
        "exp_profit": conv[idx, best] * prof[idx, best],
    })

def frontier(df, grid, mus):
    """Trace la frontiere profit<->volume en balayant mu."""
    rows = []
    for mu in mus:
        r = optimize_for_mu(df, mu, grid)
        rows.append({"mu": mu,
                     "volume": r["exp_volume"].sum(),
                     "profit": r["exp_profit"].sum(),
                     "polices": r["conv"].sum()})
    return pd.DataFrame(rows)

def pick_mu(front, profit_target):
    """Plus petit mu (=> volume max) qui atteint le profit cible."""
    ok = front[front["profit"] >= profit_target]
    if ok.empty:
        return front.iloc[-1]   # cible inatteignable -> mu le plus profitable
    return ok.iloc[0]           # mu mini admissible = volume maxi


# =====================================================================
# 5) TARIF SIMPLIFIE "UNE PAGE" POUR LES AGENTS
#    On ajuste un modele MULTIPLICATIF sur les prix optimaux :
#       Prime = BASE * f(age) * f(destination) * f(duree) * f(couverture)
#    Ajustement = regression lineaire sur log(prix_opt) avec variables
#    indicatrices (numpy lstsq) -> facteurs = exp(coefficients).
# =====================================================================
RATING_VARS = ["age_band", "destination", "duration_band", "coverage"]

def fit_simple_tariff(df, price_opt):
    y = np.log(price_opt)
    # Design matrix : intercept + indicatrices (1 niveau de reference omis par variable)
    cols, names, levels_kept = [np.ones(len(df))], ["BASE"], {}
    for v in RATING_VARS:
        levels = sorted(df[v].unique())
        levels_kept[v] = levels[1:]              # 1er niveau = reference (facteur 1.0)
        for lev in levels[1:]:
            cols.append((df[v] == lev).to_numpy().astype(float))
            names.append(f"{v}={lev}")
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    base = float(np.exp(coef[0]))
    factors = {v: {lev: 1.0 for lev in sorted(df[v].unique())} for v in RATING_VARS}
    k = 1
    for v in RATING_VARS:
        for lev in levels_kept[v]:
            factors[v][lev] = float(np.exp(coef[k])); k += 1
    return base, factors

def apply_simple_tariff(df, base, factors):
    p = np.full(len(df), base)
    for v in RATING_VARS:
        p *= df[v].map(factors[v]).to_numpy()
    return np.round(p, 2)


# =====================================================================
# 6) REPORTING
# =====================================================================
def evaluate(df, price, label):
    pref = df["p_ref"].to_numpy(); lam = df["lambda"].to_numpy()
    pp   = df["pure_premium"].to_numpy()
    conv = conversion(price, pref, lam)
    vol  = (conv * price).sum()
    prof = (conv * (K * price - EXPENSE_FIX - pp)).sum()
    earned = (conv * price).sum()
    losses = (conv * pp).sum()
    combined = (losses + (conv * (COMMISSION + EXPENSE_VAR) * price).sum()
                + (conv * EXPENSE_FIX).sum()) / earned
    print(f"  [{label}]  volume={vol:,.0f} EUR | profit={prof:,.0f} EUR "
          f"| combine={combined*100:,.1f}% | polices={conv.sum():,.0f}")
    return vol, prof, combined


# =====================================================================
# 7) EXECUTION
# =====================================================================
if __name__ == "__main__":
    df = make_synthetic_quotes()
    grid = np.round(np.arange(0.90, 1.81, 0.05), 4)   # prix de -10% a +80% du break-even
    mus  = np.round(np.linspace(0.0, 5.0, 51), 3)

    print("=== Diagnostic du portefeuille au tarif INCUMBENT ===")
    evaluate(df, df["p_ref"].to_numpy(), "incumbent")

    print("\n=== Frontiere profit <-> volume (extrait) ===")
    front = frontier(df, grid, mus)
    print(front.iloc[::10].to_string(index=False,
          formatters={"volume": "{:,.0f}".format, "profit": "{:,.0f}".format,
                      "polices": "{:,.0f}".format}))

    # Choix : on veut un profit confortable (qualifie Step 1) mais volume max (Step 2).
    PROFIT_TARGET = 0.6 * front["profit"].max()       # ex : 60% du profit maximal possible
    chosen = pick_mu(front, PROFIT_TARGET)
    print(f"\n=== mu choisi = {chosen['mu']:.2f}  (cible profit = {PROFIT_TARGET:,.0f} EUR) ===")

    opt = optimize_for_mu(df, chosen["mu"], grid)
    df["price_opt"] = opt["price_opt"].to_numpy()
    evaluate(df, df["price_opt"].to_numpy(), "optimal (theorique)")

    # Tarif simplifie pour agents
    base, factors = fit_simple_tariff(df, df["price_opt"].to_numpy())
    df["prime_finale"] = apply_simple_tariff(df, base, factors)
    evaluate(df, df["prime_finale"].to_numpy(), "tarif simplifie agents")

    print(f"\n=== TARIF UNE PAGE  (Prime = BASE x facteurs) ===")
    print(f"  BASE = {base:.2f} EUR")
    for v in RATING_VARS:
        facs = "  ".join(f"{lev}:{f:.3f}" for lev, f in factors[v].items())
        print(f"  {v:14s} -> {facs}")

    # Livrable 1 : CSV des primes (genere par le MEME tarif simplifie -> coherence)
    df[["QuoteID", "prime_finale"]].rename(columns={"prime_finale": "Prem"}) \
        .to_csv("/home/claude/submission_quotes.csv", index=False)

    # Livrable 2 : table des facteurs pour Excel
    rows = [{"variable": "BASE", "niveau": "", "facteur": round(base, 2)}]
    for v in RATING_VARS:
        for lev, f in factors[v].items():
            rows.append({"variable": v, "niveau": lev, "facteur": round(f, 4)})
    pd.DataFrame(rows).to_csv("/home/claude/tarif_facteurs.csv", index=False)
    print("\nFichiers ecrits : submission_quotes.csv  +  tarif_facteurs.csv")

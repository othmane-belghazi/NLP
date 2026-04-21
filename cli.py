"""
==========================================================================
ANALYSE COMPARATIVE DE DEUX VERSIONS TARIFAIRES
==========================================================================
Objectif : Mesurer l'impact d'un changement de moteur tarifaire sur :
  - La prime totale (affaire nouvelle)
  - Le prix de chaque garantie
  - Le coefficient tarifaire (prime AN / prime payée actuellement)

Approche : Data Science / Actuariat
Palette  : bleu ciel + blanc (minimaliste)
==========================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# -------------------------------------------------------------------------
# 1. PALETTE & STYLE
# -------------------------------------------------------------------------
SKY_BLUE   = "#87CEEB"   # bleu ciel principal
DEEP_BLUE  = "#4A90B8"   # bleu ciel foncé (accent)
LIGHT_BLUE = "#E8F4F9"   # bleu très clair (fond / zones)
GREY       = "#8C8C8C"   # gris pour axes / textes secondaires
WHITE      = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor"  : WHITE,
    "axes.facecolor"    : WHITE,
    "axes.edgecolor"    : GREY,
    "axes.labelcolor"   : "#333333",
    "axes.titlesize"    : 12,
    "axes.titleweight"  : "bold",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "xtick.color"       : GREY,
    "ytick.color"       : GREY,
    "grid.color"        : "#EEEEEE",
    "grid.linestyle"    : "--",
    "font.family"       : "DejaVu Sans",
})

# -------------------------------------------------------------------------
# 2. CHARGEMENT DES DONNEES
# -------------------------------------------------------------------------
def load_data(path_v1, path_v2, id_col="police_id",
              prime_col="prime_totale", prime_actuelle_col="prime_actuelle"):
    """
    Charge les deux versions tarifaires et les fusionne sur l'identifiant police.
    Chaque fichier doit contenir :
      - un identifiant unique (id_col)
      - la prime totale (prime_col)
      - le prix de chaque garantie (colonnes commençant par 'gar_')
      - optionnel : la prime payée actuellement (prime_actuelle_col)
    """
    v1 = pd.read_csv(path_v1)
    v2 = pd.read_csv(path_v2)

    merged = v1.merge(v2, on=id_col, suffixes=("_v1", "_v2"))
    return merged, v1, v2


# -------------------------------------------------------------------------
# 3. CALCUL DES INDICATEURS
# -------------------------------------------------------------------------
def compute_metrics(df, prime_col="prime_totale", prime_actuelle_col="prime_actuelle"):
    """
    Calcule :
      - Ecart absolu et relatif entre V1 et V2 sur la prime totale
      - Coefficient tarifaire V1 et V2 (prime AN / prime actuelle)
      - Variation du coefficient tarifaire
    """
    df = df.copy()
    df["ecart_abs"]    = df[f"{prime_col}_v2"] - df[f"{prime_col}_v1"]
    df["ecart_rel"]    = df["ecart_abs"] / df[f"{prime_col}_v1"]

    if prime_actuelle_col in df.columns:
        prime_act = df[prime_actuelle_col]
    elif f"{prime_actuelle_col}_v1" in df.columns:
        prime_act = df[f"{prime_actuelle_col}_v1"]
    else:
        prime_act = None

    if prime_act is not None:
        df["coef_v1"]     = df[f"{prime_col}_v1"] / prime_act
        df["coef_v2"]     = df[f"{prime_col}_v2"] / prime_act
        df["delta_coef"]  = df["coef_v2"] - df["coef_v1"]

    return df


def summary_stats(df, prime_col="prime_totale"):
    """Statistiques globales comparatives."""
    stats = pd.DataFrame({
        "V1": [
            df[f"{prime_col}_v1"].mean(),
            df[f"{prime_col}_v1"].median(),
            df[f"{prime_col}_v1"].std(),
            df[f"{prime_col}_v1"].sum(),
        ],
        "V2": [
            df[f"{prime_col}_v2"].mean(),
            df[f"{prime_col}_v2"].median(),
            df[f"{prime_col}_v2"].std(),
            df[f"{prime_col}_v2"].sum(),
        ],
    }, index=["Moyenne", "Médiane", "Ecart-type", "Total portefeuille"])

    stats["Δ absolu"]  = stats["V2"] - stats["V1"]
    stats["Δ relatif"] = (stats["V2"] - stats["V1"]) / stats["V1"]
    return stats


def garantie_impact(df, prefix="gar_"):
    """Impact par garantie : écart moyen absolu et relatif."""
    g_cols_v1 = [c for c in df.columns if c.startswith(prefix) and c.endswith("_v1")]
    rows = []
    for c1 in g_cols_v1:
        name = c1.replace("_v1", "").replace(prefix, "")
        c2 = c1.replace("_v1", "_v2")
        if c2 not in df.columns:
            continue
        mean_v1 = df[c1].mean()
        mean_v2 = df[c2].mean()
        rows.append({
            "garantie"       : name,
            "prime_moy_v1"   : mean_v1,
            "prime_moy_v2"   : mean_v2,
            "ecart_abs_moy"  : mean_v2 - mean_v1,
            "ecart_rel_moy"  : (mean_v2 - mean_v1) / mean_v1 if mean_v1 else np.nan,
        })
    return pd.DataFrame(rows).sort_values("ecart_rel_moy", key=abs, ascending=False)


# -------------------------------------------------------------------------
# 4. VISUALISATIONS
# -------------------------------------------------------------------------
def plot_distribution_primes(df, prime_col="prime_totale", save=None):
    """Distribution comparée des primes totales V1 vs V2."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = 40
    ax.hist(df[f"{prime_col}_v1"], bins=bins, alpha=0.55, color=SKY_BLUE,
            label="Version 1", edgecolor=WHITE)
    ax.hist(df[f"{prime_col}_v2"], bins=bins, alpha=0.55, color=DEEP_BLUE,
            label="Version 2", edgecolor=WHITE)
    ax.axvline(df[f"{prime_col}_v1"].mean(), color=SKY_BLUE, linestyle="--", linewidth=1)
    ax.axvline(df[f"{prime_col}_v2"].mean(), color=DEEP_BLUE, linestyle="--", linewidth=1)
    ax.set_title("Distribution des primes totales  —  V1 vs V2")
    ax.set_xlabel("Prime totale (€)")
    ax.set_ylabel("Nombre de polices")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_ecart_relatif(df, save=None):
    """Distribution des écarts relatifs entre V1 et V2 (en %)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    data = df["ecart_rel"] * 100
    ax.hist(data, bins=40, color=SKY_BLUE, edgecolor=WHITE)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.axvline(data.mean(), color=DEEP_BLUE, linestyle="--",
               label=f"Moyenne : {data.mean():+.2f} %")
    ax.axvline(data.median(), color="#2E5E7E", linestyle=":",
               label=f"Médiane : {data.median():+.2f} %")
    ax.set_title("Distribution de l'écart relatif de prime  (V2 − V1) / V1")
    ax.set_xlabel("Écart relatif (%)")
    ax.set_ylabel("Nombre de polices")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_scatter_v1_v2(df, prime_col="prime_totale", save=None):
    """Nuage de points V1 vs V2 avec droite y = x."""
    fig, ax = plt.subplots(figsize=(7, 7))
    x = df[f"{prime_col}_v1"]
    y = df[f"{prime_col}_v2"]
    ax.scatter(x, y, alpha=0.4, s=18, color=SKY_BLUE, edgecolor=DEEP_BLUE, linewidth=0.3)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, color=GREY, linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Prime V1 (€)")
    ax.set_ylabel("Prime V2 (€)")
    ax.set_title("Prime V2 vs Prime V1 par police")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_garantie_impact(g_df, save=None):
    """Barres horizontales : écart relatif moyen par garantie."""
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(g_df))))
    colors = [DEEP_BLUE if v >= 0 else SKY_BLUE for v in g_df["ecart_rel_moy"]]
    ax.barh(g_df["garantie"], g_df["ecart_rel_moy"] * 100,
            color=colors, edgecolor=WHITE)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.set_xlabel("Écart relatif moyen (%)")
    ax.set_title("Impact du changement tarifaire par garantie")
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=1))
    for i, (val, name) in enumerate(zip(g_df["ecart_rel_moy"] * 100, g_df["garantie"])):
        ax.text(val + (0.3 if val >= 0 else -0.3), i, f"{val:+.1f}%",
                va="center", ha="left" if val >= 0 else "right",
                fontsize=9, color="#333")
    ax.grid(True, axis="x", alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_coef_tarifaire(df, save=None):
    """Comparaison du coefficient tarifaire V1 vs V2."""
    if "coef_v1" not in df.columns:
        print("⚠ prime_actuelle non disponible — coefficient non calculé.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ----- (a) distribution des deux coefficients
    ax = axes[0]
    ax.hist(df["coef_v1"], bins=40, alpha=0.55, color=SKY_BLUE,
            label=f"V1 (moy={df['coef_v1'].mean():.2f})", edgecolor=WHITE)
    ax.hist(df["coef_v2"], bins=40, alpha=0.55, color=DEEP_BLUE,
            label=f"V2 (moy={df['coef_v2'].mean():.2f})", edgecolor=WHITE)
    ax.axvline(1.0, color=GREY, linestyle="--", linewidth=1, label="coef = 1")
    ax.set_title("Distribution du coefficient tarifaire\n(prime AN / prime actuelle)")
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Nombre de polices")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.6)

    # ----- (b) variation du coefficient
    ax = axes[1]
    data = df["delta_coef"]
    ax.hist(data, bins=40, color=SKY_BLUE, edgecolor=WHITE)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.axvline(data.mean(), color=DEEP_BLUE, linestyle="--",
               label=f"Moyenne : {data.mean():+.3f}")
    ax.set_title("Variation du coefficient tarifaire  (V2 − V1)")
    ax.set_xlabel("Δ coefficient")
    ax.set_ylabel("Nombre de polices")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.6)

    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_winners_losers(df, save=None):
    """Répartition gagnants / neutres / perdants sur l'écart relatif."""
    er = df["ecart_rel"] * 100
    bins   = [-np.inf, -5, -1, 1, 5, np.inf]
    labels = ["Baisse >5%", "Baisse 1-5%", "Stable ±1%", "Hausse 1-5%", "Hausse >5%"]
    cats   = pd.cut(er, bins=bins, labels=labels)
    counts = cats.value_counts().reindex(labels)
    pct    = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [SKY_BLUE, "#B8DCEA", "#DDDDDD", "#6FB0D0", DEEP_BLUE]
    bars = ax.bar(labels, pct.values, color=colors, edgecolor=WHITE)
    for b, v, n in zip(bars, pct.values, counts.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.5,
                f"{v:.1f}%\n({int(n)})", ha="center", fontsize=9, color="#333")
    ax.set_ylabel("Part du portefeuille (%)")
    ax.set_title("Répartition des polices par niveau d'impact de la V2")
    ax.grid(True, axis="y", alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# -------------------------------------------------------------------------
# 5. PIPELINE COMPLET
# -------------------------------------------------------------------------
def run_full_analysis(df, out_dir="."):
    """Execute l'ensemble de l'analyse et sauvegarde les figures."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    df = compute_metrics(df)

    print("=" * 70)
    print("STATISTIQUES GLOBALES")
    print("=" * 70)
    print(summary_stats(df).round(2).to_string())
    print()

    g_df = garantie_impact(df)
    print("=" * 70)
    print("IMPACT PAR GARANTIE")
    print("=" * 70)
    print(g_df.round(3).to_string(index=False))
    print()

    if "coef_v1" in df.columns:
        print("=" * 70)
        print("COEFFICIENT TARIFAIRE (prime AN / prime actuelle)")
        print("=" * 70)
        print(f"Coef V1 — moyen : {df['coef_v1'].mean():.3f} | médian : {df['coef_v1'].median():.3f}")
        print(f"Coef V2 — moyen : {df['coef_v2'].mean():.3f} | médian : {df['coef_v2'].median():.3f}")
        print(f"Δ coef moyen    : {df['delta_coef'].mean():+.3f}")
        print(f"% polices avec coef V2 > 1 : {(df['coef_v2'] > 1).mean()*100:.1f}%")
        print()

    plot_distribution_primes(df, save=f"{out_dir}/01_distribution_primes.png")
    plot_ecart_relatif     (df, save=f"{out_dir}/02_ecart_relatif.png")
    plot_scatter_v1_v2     (df, save=f"{out_dir}/03_scatter_v1_v2.png")
    plot_garantie_impact   (g_df, save=f"{out_dir}/04_garantie_impact.png")
    plot_coef_tarifaire    (df, save=f"{out_dir}/05_coef_tarifaire.png")
    plot_winners_losers    (df, save=f"{out_dir}/06_winners_losers.png")

    return df, g_df

"""
============================================================================
ANALYSE COMPARATIVE DE VERSIONS TARIFAIRES — PySpark
Versions : vt24 (actuelle) vs vt26 (nouvelle)
============================================================================

Hypothèses sur le schéma de df_all :
  - premium_total_vt24, premium_total_vt26                    (prime totale AN)
  - gar_<XXX>_vt24, gar_<XXX>_vt26                            (prix par garantie)
  - coef_tarifaire_vt24, coef_tarifaire_vt26                  (prime AN / prime ptf)
  - un identifiant unique (ex: police_id) — adapter POLICY_ID

Approche :
  - Agrégats, percentiles, segmentations : côté Spark (scalable)
  - Visualisations : collect vers pandas (volumes maîtrisés) puis matplotlib
  - Metrics robustes pour distribution asymétrique (CV élevé)
============================================================================
"""

from pyspark.sql import SparkSession, DataFrame, functions as F, Window
from pyspark.sql.types import DoubleType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter

# ---------------------------------------------------------------------------
# 0. STYLE — bleu ciel / blanc
# ---------------------------------------------------------------------------
SKY        = "#87CEEB"
DEEP_BLUE  = "#4A90B8"
DARK_BLUE  = "#2E5E7E"
LIGHT_BLUE = "#E8F4F9"
GREY       = "#8C8C8C"
WHITE      = "#FFFFFF"

plt.rcParams.update({
    "figure.facecolor" : WHITE, "axes.facecolor": WHITE,
    "axes.edgecolor"   : GREY,  "axes.labelcolor": "#333",
    "axes.titlesize"   : 12,    "axes.titleweight": "bold",
    "axes.spines.top"  : False, "axes.spines.right": False,
    "xtick.color"      : GREY,  "ytick.color": GREY,
    "grid.color"       : "#EEE","grid.linestyle": "--",
    "font.family"      : "DejaVu Sans",
})

# ---------------------------------------------------------------------------
# CONFIGURATION — adapte ces variables
# ---------------------------------------------------------------------------
V1 = "vt24"           # version actuelle
V2 = "vt26"           # version nouvelle
POLICY_ID   = "police_id"
PRIME_V1    = f"premium_total_{V1}"
PRIME_V2    = f"premium_total_{V2}"
COEF_V1     = f"coef_tarifaire_{V1}"
COEF_V2     = f"coef_tarifaire_{V2}"
GAR_PREFIX  = "gar_"                     # préfixe commun aux garanties

# ===========================================================================
# 1. PREPARATION DU DATAFRAME
# ===========================================================================
def prepare_df(df_all: DataFrame) -> DataFrame:
    """
    Ajoute toutes les métriques utiles par police :
      - écart absolu et relatif de prime
      - ratio V2/V1 et log-ratio
      - delta coefficient tarifaire
      - décile de prime V1 (pour segmentation)
    """
    df = (df_all
        .filter((F.col(PRIME_V1) > 0) & (F.col(PRIME_V2) > 0))  # sécurité
        .withColumn("ecart_abs",   F.col(PRIME_V2) - F.col(PRIME_V1))
        .withColumn("ecart_rel",   (F.col(PRIME_V2) - F.col(PRIME_V1)) / F.col(PRIME_V1))
        .withColumn("ratio_v2_v1", F.col(PRIME_V2) / F.col(PRIME_V1))
        .withColumn("log_ratio",   F.log(F.col(PRIME_V2) / F.col(PRIME_V1)))
        .withColumn("delta_coef",  F.col(COEF_V2) - F.col(COEF_V1))
    )

    # décile de prime V1 — ntile retourne 1..10
    w = Window.orderBy(F.col(PRIME_V1))
    df = df.withColumn("decile_prime_v1", F.ntile(10).over(w))
    return df


# ===========================================================================
# 2. METRICS ROBUSTES (distribution asymétrique)
# ===========================================================================
def robust_metrics(df: DataFrame) -> pd.DataFrame:
    """
    Metrics robustes : percentiles + pondéré vs équipondéré + géométrique.
    Retourne un petit pandas (agrégat) — safe à collect.
    """
    # percentiles sur prime V1, V2 et écart relatif
    q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    p1  = df.approxQuantile(PRIME_V1, q, 0.001)
    p2  = df.approxQuantile(PRIME_V2, q, 0.001)
    per = df.approxQuantile("ecart_rel", q, 0.001)
    pcf = df.approxQuantile("delta_coef", q, 0.001)

    # agrégats pondérés et équipondérés
    agg = df.agg(
        F.count("*").alias("n"),
        F.sum(PRIME_V1).alias("sum_v1"),
        F.sum(PRIME_V2).alias("sum_v2"),
        F.mean("ecart_rel").alias("ecart_rel_moy"),
        F.mean("log_ratio").alias("log_ratio_moy"),
        F.mean("delta_coef").alias("delta_coef_moy"),
        F.mean(COEF_V1).alias("coef_v1_moy"),
        F.mean(COEF_V2).alias("coef_v2_moy"),
        F.expr("percentile_approx(ecart_rel, 0.5)").alias("ecart_rel_med"),
        F.mean((F.abs("ecart_rel") > 0.10).cast("int")).alias("pct_ecart_sup_10"),
        F.mean((F.abs("ecart_rel") > 0.20).cast("int")).alias("pct_ecart_sup_20"),
        F.mean((F.col("ecart_rel") > 0).cast("int")).alias("pct_hausse"),
    ).collect()[0]

    impact_pondere = agg["sum_v2"] / agg["sum_v1"] - 1          # effet P&L
    impact_equipondere_moy = agg["ecart_rel_moy"]               # client moyen
    impact_equipondere_med = agg["ecart_rel_med"]               # client médian
    geom_mean_ratio = float(np.exp(agg["log_ratio_moy"]))       # tendance centrale propre

    rows = [
        ("Nombre de polices",                   f"{agg['n']:,}"),
        ("Prime totale V1",                     f"{agg['sum_v1']:,.0f} €"),
        ("Prime totale V2",                     f"{agg['sum_v2']:,.0f} €"),
        ("─── Impact global ───",               ""),
        ("Impact pondéré par prime (P&L)",      f"{impact_pondere:+.2%}"),
        ("Impact équipondéré — moyenne",        f"{impact_equipondere_moy:+.2%}"),
        ("Impact équipondéré — médiane",        f"{impact_equipondere_med:+.2%}"),
        ("Ratio moy. géométrique V2/V1",        f"{geom_mean_ratio:.4f}"),
        ("Divergence pondéré vs équipondéré",   f"{(impact_pondere - impact_equipondere_moy)*100:+.2f} pts"),
        ("─── Distribution écart relatif ───",  ""),
        ("P01", f"{per[0]:+.2%}"),
        ("P05", f"{per[1]:+.2%}"),
        ("P10", f"{per[2]:+.2%}"),
        ("P25", f"{per[3]:+.2%}"),
        ("P50 (médiane)", f"{per[4]:+.2%}"),
        ("P75", f"{per[5]:+.2%}"),
        ("P90", f"{per[6]:+.2%}"),
        ("P95", f"{per[7]:+.2%}"),
        ("P99", f"{per[8]:+.2%}"),
        ("IQR (P75 - P25)", f"{(per[5]-per[3])*100:.2f} pts"),
        ("─── Volumes sensibles ───",           ""),
        ("% polices avec |écart| > 10%",        f"{agg['pct_ecart_sup_10']:.1%}"),
        ("% polices avec |écart| > 20%",        f"{agg['pct_ecart_sup_20']:.1%}"),
        ("% polices en hausse",                 f"{agg['pct_hausse']:.1%}"),
        ("─── Coefficient tarifaire ───",       ""),
        ("Coef V1 moyen",                       f"{agg['coef_v1_moy']:.4f}"),
        ("Coef V2 moyen",                       f"{agg['coef_v2_moy']:.4f}"),
        ("Δ coef moyen",                        f"{agg['delta_coef_moy']:+.4f}"),
        ("Δ coef P10 / P50 / P90",              f"{pcf[2]:+.3f} / {pcf[4]:+.3f} / {pcf[6]:+.3f}"),
    ]
    return pd.DataFrame(rows, columns=["Indicateur", "Valeur"])


# ===========================================================================
# 3. IMPACT PAR GARANTIE (pondéré + équipondéré)
# ===========================================================================
def garantie_impact(df: DataFrame) -> pd.DataFrame:
    cols = df.columns
    gar_names = sorted({c.replace(f"_{V1}", "").replace(GAR_PREFIX, "")
                        for c in cols
                        if c.startswith(GAR_PREFIX) and c.endswith(f"_{V1}")})

    aggs = []
    for g in gar_names:
        c1, c2 = f"{GAR_PREFIX}{g}_{V1}", f"{GAR_PREFIX}{g}_{V2}"
        if c2 not in cols:
            continue
        aggs += [
            F.sum(c1).alias(f"sum_v1__{g}"),
            F.sum(c2).alias(f"sum_v2__{g}"),
            F.avg(F.when(F.col(c1) > 0, (F.col(c2)-F.col(c1))/F.col(c1))).alias(f"er_moy__{g}"),
            F.expr(f"percentile_approx((({c2}-{c1})/{c1}), 0.5)").alias(f"er_med__{g}"),
        ]
    if not aggs:
        return pd.DataFrame()

    res = df.agg(*aggs).collect()[0].asDict()
    rows = []
    for g in gar_names:
        s1 = res.get(f"sum_v1__{g}")
        s2 = res.get(f"sum_v2__{g}")
        if not s1 or s1 == 0:
            continue
        rows.append({
            "garantie"         : g,
            "masse_v1"         : s1,
            "masse_v2"         : s2,
            "impact_pondere"   : s2/s1 - 1,
            "impact_equip_moy" : res.get(f"er_moy__{g}"),
            "impact_equip_med" : res.get(f"er_med__{g}"),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("impact_pondere", key=lambda s: s.abs(), ascending=False)


# ===========================================================================
# 4. ECART RELATIF PAR DECILE DE PRIME V1 (segmentation clé)
# ===========================================================================
def ecart_par_decile(df: DataFrame) -> pd.DataFrame:
    agg = (df.groupBy("decile_prime_v1")
             .agg(
                F.count("*").alias("n"),
                F.min(PRIME_V1).alias("prime_min"),
                F.max(PRIME_V1).alias("prime_max"),
                F.expr("percentile_approx(ecart_rel, 0.10)").alias("p10"),
                F.expr("percentile_approx(ecart_rel, 0.25)").alias("p25"),
                F.expr("percentile_approx(ecart_rel, 0.50)").alias("p50"),
                F.expr("percentile_approx(ecart_rel, 0.75)").alias("p75"),
                F.expr("percentile_approx(ecart_rel, 0.90)").alias("p90"),
                (F.sum(PRIME_V2)/F.sum(PRIME_V1) - 1).alias("impact_pondere"),
             )
             .orderBy("decile_prime_v1")
             .toPandas())
    return agg


# ===========================================================================
# 5. ECHANTILLONNAGE POUR VISUELS (scatter / histogrammes)
# ===========================================================================
def sample_for_plot(df: DataFrame, n: int = 50_000) -> pd.DataFrame:
    """
    Echantillon aléatoire limité pour les visuels qui demandent les points
    individuels (scatter, histogramme fin). Les metrics ci-dessus sont
    calculées sur la totalité du dataset, pas sur l'échantillon.
    """
    total = df.count()
    frac = min(1.0, n / max(total, 1))
    cols = [PRIME_V1, PRIME_V2, "ecart_rel", "ratio_v2_v1",
            COEF_V1, COEF_V2, "delta_coef", "decile_prime_v1"]
    return df.select(*cols).sample(False, frac, seed=42).toPandas()


# ===========================================================================
# 6. VISUELS
# ===========================================================================
def plot_log_distribution(sample: pd.DataFrame, save=None):
    """Histogramme des primes en axe log → rend la distribution lisible."""
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(np.log10(max(sample[PRIME_V1].min(), 1)),
                       np.log10(sample[[PRIME_V1, PRIME_V2]].max().max()), 50)
    ax.hist(sample[PRIME_V1], bins=bins, alpha=0.55, color=SKY,       label=f"V1 ({V1})", edgecolor=WHITE)
    ax.hist(sample[PRIME_V2], bins=bins, alpha=0.55, color=DEEP_BLUE, label=f"V2 ({V2})", edgecolor=WHITE)
    ax.set_xscale("log")
    ax.set_title("Distribution des primes — axe log (adapté aux distributions asymétriques)")
    ax.set_xlabel("Prime (€, log)"); ax.set_ylabel("Nombre de polices")
    ax.legend(frameon=False); ax.grid(True, axis="y", alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_ecart_relatif(sample: pd.DataFrame, metrics_row, save=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    data = sample["ecart_rel"].clip(-0.5, 0.5) * 100
    ax.hist(data, bins=60, color=SKY, edgecolor=WHITE)
    ax.axvline(0,              color=GREY, linewidth=1)
    ax.axvline(data.median(),  color=DEEP_BLUE, linestyle="--",
               label=f"Médiane : {data.median():+.2f}%")
    ax.axvline(data.mean(),    color=DARK_BLUE, linestyle=":",
               label=f"Moyenne : {data.mean():+.2f}%")
    ax.set_title("Distribution de l'écart relatif  (V2 − V1) / V1")
    ax.set_xlabel("Écart relatif (%)"); ax.set_ylabel("Nombre de polices")
    ax.legend(frameon=False); ax.grid(True, axis="y", alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_scatter_log(sample: pd.DataFrame, save=None):
    """Scatter V2 vs V1 en log-log — indispensable si asymétrie."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sample[PRIME_V1], sample[PRIME_V2], s=8, alpha=0.25,
               color=SKY, edgecolor=DEEP_BLUE, linewidth=0.2)
    lo = max(sample[[PRIME_V1, PRIME_V2]].min().min(), 1)
    hi = sample[[PRIME_V1, PRIME_V2]].max().max()
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=GREY, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"Prime V1 ({V1}) — log")
    ax.set_ylabel(f"Prime V2 ({V2}) — log")
    ax.set_title("V2 vs V1 par police — échelle log")
    ax.legend(frameon=False); ax.grid(True, which="both", alpha=0.5)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_ratio_vs_prime(sample: pd.DataFrame, save=None):
    """Ratio V2/V1 en fonction de la prime V1 (log X) — hétéroscédasticité visible."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(sample[PRIME_V1], sample["ratio_v2_v1"], s=6, alpha=0.2,
               color=SKY, edgecolor=DEEP_BLUE, linewidth=0.2)
    ax.axhline(1.0, color=GREY, linestyle="--", label="ratio = 1")
    ax.set_xscale("log")
    ax.set_xlabel(f"Prime V1 ({V1}) — log"); ax.set_ylabel("Ratio V2 / V1")
    ax.set_title("Ratio V2/V1 en fonction de la taille de prime")
    ax.set_ylim(sample["ratio_v2_v1"].quantile(0.005),
                sample["ratio_v2_v1"].quantile(0.995))
    ax.legend(frameon=False); ax.grid(True, alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_boxplot_par_decile(dec_df: pd.DataFrame, save=None):
    """Boxplot synthétique par décile de prime V1 — construit depuis les percentiles Spark."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = dec_df["decile_prime_v1"].values
    p10, p25, p50, p75, p90 = (dec_df[c].values*100 for c in ["p10","p25","p50","p75","p90"])

    for xi, a, b in zip(x, p10, p90):     # moustaches
        ax.plot([xi, xi], [a, b], color=DEEP_BLUE, linewidth=1)
    for xi, a, b in zip(x, p25, p75):     # boîte IQR
        ax.add_patch(plt.Rectangle((xi-0.35, a), 0.70, b-a,
                                   color=SKY, alpha=0.75, ec=DEEP_BLUE))
    ax.scatter(x, p50, color=DARK_BLUE, zorder=5, label="Médiane")

    # impact pondéré en overlay
    ax.plot(x, dec_df["impact_pondere"].values*100, color=DARK_BLUE,
            marker="D", linewidth=1, label="Impact pondéré du décile")

    ax.axhline(0, color=GREY, linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("Décile de prime V1 (1 = plus petites primes → 10 = plus grosses)")
    ax.set_ylabel("Écart relatif (%)")
    ax.set_title("Écart relatif par décile de prime — P10 / P25 / médiane / P75 / P90")
    ax.legend(frameon=False); ax.grid(True, axis="y", alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_qqplot(sample: pd.DataFrame, save=None):
    """QQ-plot V1 vs V2 — détecte les shifts de quantiles."""
    fig, ax = plt.subplots(figsize=(7, 7))
    q = np.linspace(0.01, 0.99, 99)
    q1 = np.quantile(sample[PRIME_V1], q)
    q2 = np.quantile(sample[PRIME_V2], q)
    ax.scatter(q1, q2, s=20, color=DEEP_BLUE)
    lo, hi = min(q1.min(), q2.min()), max(q1.max(), q2.max())
    ax.plot([lo, hi], [lo, hi], "--", color=GREY, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(f"Quantiles V1 ({V1})"); ax.set_ylabel(f"Quantiles V2 ({V2})")
    ax.set_title("QQ-plot — comparaison quantile par quantile")
    ax.legend(frameon=False); ax.grid(True, which="both", alpha=0.5)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_lorenz_impact(df: DataFrame, save=None):
    """Courbe de Lorenz de la concentration de l'impact absolu."""
    # collect uniquement |ecart_abs| — 1 colonne, sans danger jusqu'à plusieurs M lignes
    s = (df.select(F.abs("ecart_abs").alias("e"))
           .toPandas()["e"].sort_values(ascending=False).values)
    cum = np.cumsum(s) / s.sum()
    pct = np.arange(1, len(s)+1) / len(s)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(pct*100, cum*100, color=DEEP_BLUE, linewidth=2)
    ax.plot([0, 100], [0, 100], "--", color=GREY, label="Répartition uniforme")
    # point remarquable : concentration sur les 10% les plus impactants
    idx10 = int(0.10 * len(s))
    p10 = cum[idx10] * 100
    ax.scatter([10], [p10], color=DARK_BLUE, zorder=5)
    ax.annotate(f"10% des polices\nportent {p10:.0f}% de l'impact",
                (10, p10), xytext=(25, p10-10),
                arrowprops=dict(arrowstyle="->", color=GREY),
                fontsize=10, color="#333")
    ax.set_xlabel("% cumulé des polices (triées par impact décroissant)")
    ax.set_ylabel("% cumulé de l'impact absolu total")
    ax.set_title("Courbe de Lorenz — concentration de l'impact tarifaire")
    ax.legend(frameon=False); ax.grid(True, alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_garantie_impact(g_df: pd.DataFrame, save=None):
    """Barres : impact pondéré vs médian par garantie."""
    if g_df.empty: return None
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5*len(g_df))))
    y = np.arange(len(g_df))
    h = 0.38
    ax.barh(y - h/2, g_df["impact_pondere"]*100,   h,
            color=DEEP_BLUE, label="Impact pondéré (masse)")
    ax.barh(y + h/2, g_df["impact_equip_med"]*100, h,
            color=SKY,       label="Impact médian (police)")
    ax.set_yticks(y); ax.set_yticklabels(g_df["garantie"])
    ax.axvline(0, color=GREY, linewidth=1)
    ax.set_xlabel("Impact (%)")
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=1))
    ax.set_title("Impact par garantie  —  pondéré vs médian")
    ax.legend(frameon=False); ax.grid(True, axis="x", alpha=0.6)
    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_coef_tarifaire(sample: pd.DataFrame, save=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(sample[COEF_V1], bins=50, alpha=0.55, color=SKY,
            label=f"V1 (moy={sample[COEF_V1].mean():.3f})", edgecolor=WHITE)
    ax.hist(sample[COEF_V2], bins=50, alpha=0.55, color=DEEP_BLUE,
            label=f"V2 (moy={sample[COEF_V2].mean():.3f})", edgecolor=WHITE)
    ax.axvline(1.0, color=GREY, linestyle="--", label="coef = 1")
    ax.set_title("Distribution du coefficient tarifaire\n(prime AN / prime portefeuille)")
    ax.set_xlabel("Coefficient"); ax.set_ylabel("Nombre de polices")
    ax.legend(frameon=False); ax.grid(True, axis="y", alpha=0.6)

    ax = axes[1]
    d = sample["delta_coef"]
    ax.hist(d, bins=60, color=SKY, edgecolor=WHITE)
    ax.axvline(0, color=GREY, linewidth=1)
    ax.axvline(d.median(), color=DEEP_BLUE, linestyle="--",
               label=f"Médiane : {d.median():+.3f}")
    ax.set_title("Variation du coefficient tarifaire  (V2 − V1)")
    ax.set_xlabel("Δ coefficient"); ax.set_ylabel("Nombre de polices")
    ax.legend(frameon=False); ax.grid(True, axis="y", alpha=0.6)

    plt.tight_layout()
    if save: plt.savefig(save, dpi=150, bbox_inches="tight")
    return fig


# ===========================================================================
# 7. PIPELINE COMPLET
# ===========================================================================
def run_full_analysis(df_all: DataFrame, out_dir: str = "./outputs",
                      sample_size: int = 50_000):
    import os
    os.makedirs(out_dir, exist_ok=True)

    print("▶ Préparation du dataframe ...")
    df = prepare_df(df_all).cache()
    n = df.count()
    print(f"  {n:,} polices retenues.\n")

    print("▶ Metrics robustes ...")
    metrics = robust_metrics(df)
    print(metrics.to_string(index=False))
    metrics.to_csv(f"{out_dir}/metrics_robustes.csv", index=False)

    print("\n▶ Impact par garantie ...")
    g_df = garantie_impact(df)
    if not g_df.empty:
        print(g_df.round(4).to_string(index=False))
        g_df.to_csv(f"{out_dir}/impact_garanties.csv", index=False)

    print("\n▶ Ecart par décile de prime V1 ...")
    dec_df = ecart_par_decile(df)
    print(dec_df.round(4).to_string(index=False))
    dec_df.to_csv(f"{out_dir}/ecart_par_decile.csv", index=False)

    print(f"\n▶ Echantillonnage ({sample_size:,} polices) pour visuels ...")
    sample = sample_for_plot(df, n=sample_size)

    print("▶ Génération des visuels ...")
    plot_log_distribution (sample, save=f"{out_dir}/01_distribution_log.png")
    plot_ecart_relatif    (sample, metrics, save=f"{out_dir}/02_ecart_relatif.png")
    plot_scatter_log      (sample, save=f"{out_dir}/03_scatter_log.png")
    plot_ratio_vs_prime   (sample, save=f"{out_dir}/04_ratio_vs_prime.png")
    plot_boxplot_par_decile(dec_df, save=f"{out_dir}/05_boxplot_decile.png")
    plot_qqplot           (sample, save=f"{out_dir}/06_qqplot.png")
    plot_lorenz_impact    (df,     save=f"{out_dir}/07_lorenz.png")
    plot_garantie_impact  (g_df,   save=f"{out_dir}/08_garanties.png")
    plot_coef_tarifaire   (sample, save=f"{out_dir}/09_coef_tarifaire.png")

    print(f"\n✓ Rapport complet dans {out_dir}/")
    return {"metrics": metrics, "garanties": g_df, "deciles": dec_df, "df": df}


# ===========================================================================
# UTILISATION
# ===========================================================================
# from tarif_pyspark import run_full_analysis
# results = run_full_analysis(df_all, out_dir="./outputs_vt24_vt26")

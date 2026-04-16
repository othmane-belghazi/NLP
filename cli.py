# =============================================================================
# ÉTUDE ACTUARIELLE — PRICE TEST | SEGMENT JEUNES CONDUCTEURS (< 30 ANS)
# =============================================================================
# Auteur  : Équipe Tarification
# Objet   : Évaluation de l'impact du Price Test sur la rétention et la
#           rentabilité du segment Jeunes Conducteurs (2022–2025)
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. INITIALISATION SPARK
# ---------------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("PriceTest_JeunesConduct") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ---------------------------------------------------------------------------
# ÉTAPE 1 — RAPPEL DU CONTEXTE (affiché en console)
# ---------------------------------------------------------------------------
contexte = """
╔══════════════════════════════════════════════════════════════════════════════╗
║          ÉTUDE PRICE TEST — SEGMENT JEUNES CONDUCTEURS (< 30 ANS)           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MESURES TARIFAIRES APPLIQUÉES (variable : majoration_price_test)            ║
║  ┌─────────────┬───────────────────────────────────────────────────────┐    ║
║  │  Modalité   │  Signification                                         │    ║
║  ├─────────────┼───────────────────────────────────────────────────────┤    ║
║  │     -2      │  Réduction de 2% sur la majoration au renouvellement  │    ║
║  │     -1      │  Réduction de 1% sur la majoration au renouvellement  │    ║
║  │      0      │  Aucune variation (groupe témoin)                     │    ║
║  │     +1      │  Hausse de 1% sur la majoration au renouvellement     │    ║
║  │     +2      │  Hausse de 2% sur la majoration au renouvellement     │    ║
║  └─────────────┴───────────────────────────────────────────────────────┘    ║
║                                                                              ║
║  POPULATIONS ÉLIGIBLES AU PRICE TEST (variable : population_cible)          ║
║  • "Price Test"  : Population soumise au test tarifaire actif               ║
║  • "Psy Test"    : Groupe de contrôle psychologique                         ║
║  • "Non defini"  : Population hors périmètre du test                        ║
║                                                                              ║
║  PÉRIMÈTRE : Contrats avec âge < 30 ans | Millésimes 2022, 2023, 2024, 2025║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
print(contexte)


# ---------------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
# ---------------------------------------------------------------------------
# Adaptez les chemins selon votre environnement (HDFS, ADLS, S3, local…)

def load_base(year: int):
    """Charge et tague la base annuelle."""
    path = f"/data/assurance/portefeuille_{year}.parquet"   # ← À adapter
    # path = f"/mnt/data/portefeuille_{year}.csv"           # si CSV
    df = spark.read.parquet(path)
    # df = spark.read.csv(path, header=True, inferSchema=True)
    return df.withColumn("annee", F.lit(year))

df_2022 = load_base(2022)
df_2023 = load_base(2023)
df_2024 = load_base(2024)
df_2025 = load_base(2025)

df_raw = df_2022.unionByName(df_2023, allowMissingColumns=True) \
                .unionByName(df_2024, allowMissingColumns=True) \
                .unionByName(df_2025, allowMissingColumns=True)

print(f"[INFO] Données brutes chargées : {df_raw.count():,} lignes")


# ---------------------------------------------------------------------------
# 2. NETTOYAGE & TYPAGE
# ---------------------------------------------------------------------------
df_clean = df_raw \
    .withColumn("age", F.col("age").cast("integer")) \
    .withColumn("prime_ht",      F.col("prime ht").cast(DoubleType())) \
    .withColumn("prime_ttc",     F.col("prime ttc").cast(DoubleType())) \
    .withColumn("prime_ttc_av",  F.col("prime ttc avant renouvellement").cast(DoubleType())) \
    .withColumn("coef_tarifaire", F.col("coefficient tarifaire").cast(DoubleType())) \
    .withColumn("majoration_applique", F.col("majoration_applique").cast(DoubleType())) \
    .withColumn("elr",   F.col("elr").cast(DoubleType())) \
    .withColumn("resilie", F.col("resilie").cast("integer")) \
    .withColumn("prix_test_moda", F.col("majoration-price-test").cast("integer")) \
    .withColumn("population_cible", F.trim(F.col("population cible price test"))) \
    .withColumn("segment_client",   F.trim(F.col("segment de client"))) \
    .withColumn("date_renouvellement",
                F.to_date(F.col("date renouvellement"), "yyyy-MM-dd")) \
    .withColumn("date_emission_resiliation",
                F.to_date(F.col("date emission resiliation"), "yyyy-MM-dd"))


# ---------------------------------------------------------------------------
# 3. FILTRE JEUNES CONDUCTEURS (< 30 ANS)
# ---------------------------------------------------------------------------
df_jeunes = df_clean.filter(F.col("age") < 30)
print(f"[INFO] Après filtre âge < 30 : {df_jeunes.count():,} lignes")


# ---------------------------------------------------------------------------
# 4. CALCUL DES FENÊTRES DE RÉSILIATION
# ---------------------------------------------------------------------------
df_jeunes = df_jeunes.withColumn(
    "delta_jours",
    F.datediff(
        F.col("date_emission_resiliation"),
        F.col("date_renouvellement")
    )
)

# Résiliation dans la fenêtre : entre -30 j et +90 j autour du renouvellement
df_jeunes = df_jeunes \
    .withColumn(
        "resil_fenetree",
        F.when(
            (F.col("resilie") == 1) &
            (F.col("delta_jours").between(-30, 90)),
            1
        ).otherwise(0)
    ) \
    .withColumn(
        "resil_hors_fenetre",
        F.when(
            (F.col("resilie") == 1) &
            (
                (F.col("delta_jours") < -30) |
                (F.col("delta_jours") > 90)  |
                F.col("delta_jours").isNull()
            ),
            1
        ).otherwise(0)
    )


# ---------------------------------------------------------------------------
# ÉTAPE 2 — TABLEAU DE BORD GLOBAL PAR ANNÉE
# ---------------------------------------------------------------------------
print("\n" + "="*78)
print("  ÉTAPE 2 — VISION GLOBALE PAR ANNÉE")
print("="*78)

bilan_annee = df_jeunes.groupBy("annee").agg(
    F.count("contrat").alias("exposition"),
    F.sum("prime_ht").alias("ressource_ht"),
    F.avg("prime_ht").alias("prime_moy_ht"),
    F.avg("prime_ttc").alias("prime_moy_ttc"),
    F.avg("elr").alias("elr_moyen"),
    F.avg("majoration_applique").alias("maj_moy_applique"),
    F.avg("coef_tarifaire").alias("coef_tarifaire_moyen"),
    F.sum("resil_fenetree").alias("nb_resil_fenetre"),
    F.sum("resil_hors_fenetre").alias("nb_resil_hors"),
    F.sum("resilie").alias("nb_resil_total")
)

# Poids des jeunes par rapport au total portefeuille (toutes tranches d'âge)
total_par_annee = df_clean.groupBy("annee").agg(
    F.sum("prime_ht").alias("total_ht_portefeuille")
)

bilan_annee = bilan_annee \
    .join(total_par_annee, on="annee", how="left") \
    .withColumn(
        "poids_segment_pct",
        F.round(F.col("ressource_ht") / F.col("total_ht_portefeuille") * 100, 2)
    ) \
    .withColumn(
        "taux_resil_fenetre_pct",
        F.round(F.col("nb_resil_fenetre") / F.col("exposition") * 100, 2)
    ) \
    .withColumn(
        "taux_resil_hors_pct",
        F.round(F.col("nb_resil_hors") / F.col("exposition") * 100, 2)
    ) \
    .withColumn("ressource_ht_M", F.round(F.col("ressource_ht") / 1_000_000, 3)) \
    .withColumn("prime_moy_ht",   F.round("prime_moy_ht", 2)) \
    .withColumn("prime_moy_ttc",  F.round("prime_moy_ttc", 2)) \
    .withColumn("elr_moyen",      F.round("elr_moyen", 4)) \
    .withColumn("maj_moy_applique",    F.round("maj_moy_applique", 4)) \
    .withColumn("coef_tarifaire_moyen", F.round("coef_tarifaire_moyen", 4)) \
    .orderBy("annee")

print("\n[TABLEAU 1] Bilan global — Segment Jeunes Conducteurs (< 30 ans)\n")
bilan_annee.select(
    "annee", "exposition", "ressource_ht_M",
    "prime_moy_ht", "prime_moy_ttc",
    "poids_segment_pct", "elr_moyen",
    "maj_moy_applique", "coef_tarifaire_moyen",
    "taux_resil_fenetre_pct", "taux_resil_hors_pct"
).show(truncate=False)


# ---------------------------------------------------------------------------
# ÉTAPE 3 — ANALYSE PAR MODALITÉ DU PRICE TEST
# ---------------------------------------------------------------------------
print("\n" + "="*78)
print("  ÉTAPE 3 — ANALYSE PAR MODALITÉ PRICE TEST × ANNÉE")
print("="*78)

# Filtre uniquement sur la population "Price Test"
df_pt = df_jeunes.filter(F.col("population_cible") == "Price Test")

bilan_moda = df_pt.groupBy("annee", "prix_test_moda").agg(
    F.count("contrat").alias("exposition"),
    F.sum("prime_ht").alias("ressource_ht"),
    F.avg("majoration_applique").alias("maj_moy"),
    F.avg("elr").alias("elr_moyen"),
    F.sum("resil_fenetree").alias("nb_resil_fenetre"),
    F.sum("resil_hors_fenetre").alias("nb_resil_hors"),
    F.sum("resilie").alias("nb_resil_total")
).withColumn(
    "taux_resil_global_pct",
    F.round(F.col("nb_resil_total") / F.col("exposition") * 100, 2)
).withColumn(
    "taux_resil_fenetre_pct",
    F.round(F.col("nb_resil_fenetre") / F.col("exposition") * 100, 2)
).withColumn(
    "maj_moy", F.round("maj_moy", 4)
).withColumn(
    "elr_moyen", F.round("elr_moyen", 4)
).orderBy("annee", "prix_test_moda")

print("\n[TABLEAU 2] Bilan par Modalité du Price Test\n")
bilan_moda.show(40, truncate=False)


# ---------------------------------------------------------------------------
# ANALYSE POPULATION : Price Test vs Non défini vs Psy Test
# ---------------------------------------------------------------------------
bilan_population = df_jeunes.groupBy("annee", "population_cible").agg(
    F.count("contrat").alias("exposition"),
    F.avg("majoration_applique").alias("maj_moy"),
    F.sum("resil_fenetree").alias("nb_resil_fenetre"),
    F.sum("resil_hors_fenetre").alias("nb_resil_hors"),
    F.sum("resilie").alias("nb_resil_total")
).withColumn(
    "taux_resil_global_pct",
    F.round(F.col("nb_resil_total") / F.col("exposition") * 100, 2)
).withColumn(
    "taux_resil_fenetre_pct",
    F.round(F.col("nb_resil_fenetre") / F.col("exposition") * 100, 2)
).withColumn(
    "maj_moy", F.round("maj_moy", 4)
).orderBy("annee", "population_cible")

print("\n[TABLEAU 3] Comparaison Population (Price Test vs Autres)\n")
bilan_population.show(30, truncate=False)


# ---------------------------------------------------------------------------
# ANALYSE ÉLASTICITÉ-PRIX
# ---------------------------------------------------------------------------
print("\n" + "="*78)
print("  ANALYSE ÉLASTICITÉ-PRIX PAR MODALITÉ")
print("="*78)

# Arc-élasticité : % variation résiliation / % variation prime
w = Window.partitionBy("annee").orderBy("prix_test_moda")

elast_df = bilan_moda.select(
    "annee", "prix_test_moda", "exposition",
    "taux_resil_global_pct", "maj_moy"
).withColumn(
    "delta_resil",
    F.col("taux_resil_global_pct") - F.lag("taux_resil_global_pct", 1).over(w)
).withColumn(
    "delta_maj",
    F.col("maj_moy") - F.lag("maj_moy", 1).over(w)
).withColumn(
    "elasticite_arc",
    F.when(
        F.col("delta_maj") != 0,
        F.round((F.col("delta_resil") / F.col("delta_maj")), 4)
    ).otherwise(None)
).orderBy("annee", "prix_test_moda")

print("\n[TABLEAU 4] Élasticité-Prix arc par modalité et par année\n")
elast_df.show(40, truncate=False)


# ---------------------------------------------------------------------------
# IMPACT NET : GAIN PRIME vs PERTE PORTEFEUILLE
# ---------------------------------------------------------------------------
print("\n" + "="*78)
print("  IMPACT NET DU PRICE TEST (GAIN PRIME vs PERTE PORTEFEUILLE)")
print("="*78)

# Récupérer la prime moyenne et le taux de résiliation pour la modalité 0 (référence)
base_ref = bilan_moda.filter(F.col("prix_test_moda") == 0) \
    .select(
        "annee",
        F.col("ressource_ht").alias("ht_ref"),
        F.col("exposition").alias("expo_ref"),
        F.col("taux_resil_global_pct").alias("taux_ref")
    )

impact_net = bilan_moda \
    .join(base_ref, on="annee", how="left") \
    .withColumn(
        "gain_prime_estime",
        F.round((F.col("ressource_ht") - F.col("ht_ref")), 0)
    ) \
    .withColumn(
        "perte_expo_estime",
        F.round(
            (F.col("taux_resil_global_pct") - F.col("taux_ref"))
            / 100 * F.col("expo_ref"), 0
        )
    ) \
    .select(
        "annee", "prix_test_moda", "exposition",
        "ressource_ht", "gain_prime_estime",
        "taux_resil_global_pct", "taux_ref",
        "perte_expo_estime"
    ).orderBy("annee", "prix_test_moda")

print("\n[TABLEAU 5] Impact net estimé par modalité vs groupe témoin (modalité 0)\n")
impact_net.show(40, truncate=False)


# ===========================================================================
# VISUALISATIONS
# ===========================================================================
# Conversion en Pandas pour matplotlib
bilan_annee_pd   = bilan_annee.toPandas()
bilan_moda_pd    = bilan_moda.toPandas()
bilan_pop_pd     = bilan_population.toPandas()
elast_pd         = elast_df.toPandas()
impact_net_pd    = impact_net.toPandas()

ANNEES   = sorted(bilan_moda_pd["annee"].unique())
MODALITES = sorted(bilan_moda_pd["prix_test_moda"].unique())

# Palette couleurs (bleu ciel / actuarielle)
COLORS_MODA = {
    -2: "#1a6fa8",
    -1: "#4fa3d4",
     0: "#a8d5ea",
     1: "#f4a261",
     2: "#e76f51"
}
BLUE_MAIN   = "#1a6fa8"
BLUE_LIGHT  = "#87ceeb"
BLUE_MED    = "#4fa3d4"
ORANGE_ACC  = "#e76f51"
GREY_LINE   = "#2d2d2d"
GRAY_BG     = "#f7fbff"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.color": "#c5d9e8",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


# ─────────────────────────────────────────────────────────────────────────────
# VISUEL 1 — Exposition par modalité (barres) + Taux résil + Majoration moy
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "VISUEL 1 — Impact des Modalités du Price Test par Année\nSegment Jeunes Conducteurs (< 30 ans)",
    fontsize=13, fontweight="bold", color="#1a2a3a", y=1.01
)

for idx, (annee, ax) in enumerate(zip(ANNEES, axes.flatten())):
    d = bilan_moda_pd[bilan_moda_pd["annee"] == annee].sort_values("prix_test_moda")

    # Barres : exposition
    bars = ax.bar(
        d["prix_test_moda"].astype(str),
        d["exposition"],
        color=[COLORS_MODA.get(m, BLUE_LIGHT) for m in d["prix_test_moda"]],
        width=0.5, zorder=3, alpha=0.88
    )
    ax.set_ylabel("Exposition (nb contrats)", color=BLUE_MAIN)
    ax.set_facecolor(GRAY_BG)

    # Axe secondaire : taux résiliation
    ax2 = ax.twinx()
    ax2.plot(
        d["prix_test_moda"].astype(str),
        d["taux_resil_global_pct"],
        color=ORANGE_ACC, marker="o", linewidth=2.0,
        markersize=6, label="Taux résil. global (%)", zorder=5
    )

    # Axe secondaire : majoration moyenne
    ax2.plot(
        d["prix_test_moda"].astype(str),
        d["maj_moy"] * 100,  # passage en %
        color=GREY_LINE, marker="s", linewidth=1.5, linestyle="--",
        markersize=5, label="Majoration moy. (%)", zorder=5
    )
    ax2.set_ylabel("Taux résil. & Majoration (%)", fontsize=8)
    ax2.spines["top"].set_visible(False)

    ax.set_title(f"Année {annee}", fontweight="bold", fontsize=10, color="#1a2a3a")
    ax.set_xlabel("Modalité price test", fontsize=8)

    # Étiquettes sur barres
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + h * 0.01,
            f"{int(h):,}", ha="center", va="bottom", fontsize=7.5, color="#1a2a3a"
        )

    if idx == 0:
        ax2.legend(loc="upper left", fontsize=7.5, framealpha=0.7)

plt.tight_layout()
plt.savefig("/tmp/visuel1_modalites_price_test.png", dpi=150, bbox_inches="tight")
plt.show()
print("[OK] Visuel 1 sauvegardé.")


# ─────────────────────────────────────────────────────────────────────────────
# VISUEL 2 — Price Test vs Reste du portefeuille (Majoration + Résiliation)
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "VISUEL 2 — Price Test vs Autres Populations\nMajoration & Taux de Résiliation par Année",
    fontsize=13, fontweight="bold", color="#1a2a3a"
)

pop_colors = {
    "Price Test":  BLUE_MAIN,
    "Psy Test":    BLUE_MED,
    "Non defini":  "#b0c4de"
}

for pop in bilan_pop_pd["population_cible"].unique():
    sub = bilan_pop_pd[bilan_pop_pd["population_cible"] == pop].sort_values("annee")
    c = pop_colors.get(pop, "#999")

    ax1.plot(
        sub["annee"], sub["maj_moy"] * 100,
        marker="o", linewidth=2.2, color=c, label=pop
    )
    ax2.plot(
        sub["annee"], sub["taux_resil_global_pct"],
        marker="o", linewidth=2.2, color=c, label=pop
    )

ax1.set_title("Majoration Moyenne Appliquée (%)", fontweight="bold", fontsize=10)
ax1.set_ylabel("Majoration (%)")
ax1.set_xlabel("Année")
ax1.set_facecolor(GRAY_BG)
ax1.legend(fontsize=8)

ax2.set_title("Taux de Résiliation Global (%)", fontweight="bold", fontsize=10)
ax2.set_ylabel("Taux de résiliation (%)")
ax2.set_xlabel("Année")
ax2.set_facecolor(GRAY_BG)
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/tmp/visuel2_pt_vs_reste.png", dpi=150, bbox_inches="tight")
plt.show()
print("[OK] Visuel 2 sauvegardé.")


# ─────────────────────────────────────────────────────────────────────────────
# VISUEL 3 — ELR moyen par modalité × année (Heatmap)
# ─────────────────────────────────────────────────────────────────────────────
pivot_elr = bilan_moda_pd.pivot(
    index="prix_test_moda", columns="annee", values="elr_moyen"
)

fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(pivot_elr.values, cmap="RdYlGn_r", aspect="auto", vmin=0.6, vmax=1.2)
plt.colorbar(im, ax=ax, label="ELR moyen")

ax.set_xticks(range(len(pivot_elr.columns)))
ax.set_xticklabels(pivot_elr.columns)
ax.set_yticks(range(len(pivot_elr.index)))
ax.set_yticklabels([f"Modalité {m:+d}" for m in pivot_elr.index])

# Annotations dans chaque cellule
for i in range(len(pivot_elr.index)):
    for j in range(len(pivot_elr.columns)):
        val = pivot_elr.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=9, color="black", fontweight="bold")

ax.set_title(
    "VISUEL 3 — ELR Moyen par Modalité du Price Test & par Année\n"
    "Vert = rentable (ELR < 1), Rouge = déficitaire (ELR > 1)",
    fontsize=11, fontweight="bold", color="#1a2a3a"
)
ax.set_xlabel("Année")
plt.tight_layout()
plt.savefig("/tmp/visuel3_elr_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("[OK] Visuel 3 (Heatmap ELR) sauvegardé.")


# ─────────────────────────────────────────────────────────────────────────────
# VISUEL 4 — Impact Net : Gain prime vs Perte portefeuille par modalité
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle(
    "VISUEL 4 — Impact Net du Price Test vs Modalité 0 (Groupe Témoin)\n"
    "Gain de Prime & Perte d'Exposition Estimée par Année",
    fontsize=13, fontweight="bold", color="#1a2a3a"
)

for idx, (annee, ax) in enumerate(zip(ANNEES, axes.flatten())):
    d = impact_net_pd[impact_net_pd["annee"] == annee].sort_values("prix_test_moda")
    d = d[d["prix_test_moda"] != 0]

    moda_labels = [f"{m:+d}" for m in d["prix_test_moda"]]
    x = np.arange(len(moda_labels))
    width = 0.35

    bar1 = ax.bar(x - width/2, d["gain_prime_estime"], width,
                  label="Gain prime estimé (€)", color=BLUE_MED, alpha=0.85)
    ax2 = ax.twinx()
    bar2 = ax2.bar(x + width/2, d["perte_expo_estime"], width,
                   label="Perte exposition (nb contrats)", color=ORANGE_ACC, alpha=0.75)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Année {annee}", fontweight="bold", fontsize=10, color="#1a2a3a")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Modalité {l}" for l in moda_labels], fontsize=8)
    ax.set_ylabel("Gain prime (€)", color=BLUE_MED, fontsize=8)
    ax2.set_ylabel("Perte expo (contrats)", color=ORANGE_ACC, fontsize=8)
    ax.set_facecolor(GRAY_BG)
    ax2.spines["top"].set_visible(False)

    if idx == 0:
        lines = [bar1, bar2]
        labels = ["Gain prime estimé (€)", "Perte exposition (nb contrats)"]
        ax.legend(lines, labels, fontsize=7.5, loc="upper left", framealpha=0.7)

plt.tight_layout()
plt.savefig("/tmp/visuel4_impact_net.png", dpi=150, bbox_inches="tight")
plt.show()
print("[OK] Visuel 4 (Impact Net) sauvegardé.")


# ─────────────────────────────────────────────────────────────────────────────
# VISUEL 5 — Évolution de l'exposition et de la ressource HT (Jeunes)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 5))
fig.suptitle(
    "VISUEL 5 — Évolution du Portefeuille Jeunes Conducteurs (2022–2025)\n"
    "Exposition & Ressources HT",
    fontsize=12, fontweight="bold", color="#1a2a3a"
)

years_sorted = bilan_annee_pd.sort_values("annee")

bars = ax1.bar(
    years_sorted["annee"].astype(str),
    years_sorted["exposition"],
    color=BLUE_LIGHT, width=0.4, label="Exposition (nb contrats)", zorder=3
)
ax1.set_ylabel("Exposition (nb contrats)", color=BLUE_MAIN)
ax1.set_facecolor(GRAY_BG)

ax2_v5 = ax1.twinx()
ax2_v5.plot(
    years_sorted["annee"].astype(str),
    years_sorted["ressource_ht_M"],
    color=BLUE_MAIN, marker="D", linewidth=2.5, markersize=7,
    label="Ressources HT (M€)", zorder=5
)
ax2_v5.set_ylabel("Ressources HT (M€)", color=BLUE_MAIN)
ax2_v5.spines["top"].set_visible(False)

# Annotations barres
for bar in bars:
    h = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2, h + h * 0.01,
        f"{int(h):,}", ha="center", va="bottom", fontsize=9, color="#1a2a3a"
    )

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2_v5.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("/tmp/visuel5_exposition_ht.png", dpi=150, bbox_inches="tight")
plt.show()
print("[OK] Visuel 5 (Exposition & HT) sauvegardé.")


# ─────────────────────────────────────────────────────────────────────────────
# VISUEL 6 — Distribution des résiliations : fenêtrée vs hors fenêtre
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(ANNEES), figsize=(16, 5), sharey=True)
fig.suptitle(
    "VISUEL 6 — Taux de Résiliation : Dans la Fenêtre vs Hors Fenêtre\npar Modalité & Année",
    fontsize=12, fontweight="bold", color="#1a2a3a"
)

for idx, (annee, ax) in enumerate(zip(ANNEES, axes)):
    d = bilan_moda_pd[bilan_moda_pd["annee"] == annee].sort_values("prix_test_moda")
    moda_labels = [f"{m:+d}" for m in d["prix_test_moda"]]
    x = np.arange(len(moda_labels))
    w = 0.35

    ax.bar(x - w/2, d["taux_resil_fenetre_pct"], w,
           label="Dans la fenêtre", color=BLUE_MAIN, alpha=0.85)
    ax.bar(x + w/2, d.get("taux_resil_hors_pct", 0), w,
           label="Hors fenêtre", color=BLUE_LIGHT, alpha=0.85)

    ax.set_title(f"{annee}", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(moda_labels, fontsize=8)
    ax.set_xlabel("Modalité")
    ax.set_facecolor(GRAY_BG)

    if idx == 0:
        ax.set_ylabel("Taux de résiliation (%)")
        ax.legend(fontsize=7.5)

plt.tight_layout()
plt.savefig("/tmp/visuel6_resil_fenetre.png", dpi=150, bbox_inches="tight")
plt.show()
print("[OK] Visuel 6 (Résiliations fenêtre) sauvegardé.")

print("\n[DONE] Analyse complète terminée. Tous les visuels ont été sauvegardés dans /tmp/")
spark.stop()

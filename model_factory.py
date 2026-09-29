from pyspark.sql import functions as F

# ============================================================
# 1. Paramètres de calibration
# ============================================================

P_BAS = 0.10       # P10
P_HAUT = 0.95      # P95

MAJ_MIN = 1.14
MAJ_MAX = 1.30
AMPLITUDE = MAJ_MAX - MAJ_MIN


# ============================================================
# 2. Calcul de Cbas et Chaut sur l'ensemble de la population
# ============================================================

quantiles = (
    df
    .select(
        F.expr(
            f"percentile_approx(charges_Sinistre, {P_BAS}, 10000)"
        ).alias("C_bas"),

        F.expr(
            f"percentile_approx(charges_Sinistre, {P_HAUT}, 10000)"
        ).alias("C_haut")
    )
)

# Récupération des deux valeurs
params = quantiles.collect()[0]

C_bas = params["C_bas"]
C_haut = params["C_haut"]

print(f"C_bas  (P10) = {C_bas:,.2f}")
print(f"C_haut (P95) = {C_haut:,.2f}")


# ============================================================
# 3. Application de la nouvelle méthode
# ============================================================

df_new = (
    df

    # --------------------------------------------------------
    # Transformation logarithmique
    # --------------------------------------------------------
    .withColumn(
        "charge_log",
        F.log1p(F.col("charges_Sinistre"))
    )

    # --------------------------------------------------------
    # Bornes dans l'espace logarithmique
    # --------------------------------------------------------
    .withColumn(
        "log_C_bas",
        F.lit(float(__import__("math").log1p(C_bas)))
    )
    .withColumn(
        "log_C_haut",
        F.lit(float(__import__("math").log1p(C_haut)))
    )

    # --------------------------------------------------------
    # Normalisation x entre 0 et 1
    # --------------------------------------------------------
    .withColumn(
        "x_brut",
        (
            (F.col("charge_log") - F.col("log_C_bas"))
            /
            (F.col("log_C_haut") - F.col("log_C_bas"))
        )
    )

    # --------------------------------------------------------
    # Bornage [0,1]
    # --------------------------------------------------------
    .withColumn(
        "x",
        F.when(F.col("x_brut") < 0, 0.0)
         .when(F.col("x_brut") > 1, 1.0)
         .otherwise(F.col("x_brut"))
    )

    # --------------------------------------------------------
    # Smoothstep
    #
    # f(x) = 3x² - 2x³
    # --------------------------------------------------------
    .withColumn(
        "smoothstep",
        3 * F.pow(F.col("x"), 2)
        - 2 * F.pow(F.col("x"), 3)
    )

    # --------------------------------------------------------
    # Nouvelle majoration
    # --------------------------------------------------------
    .withColumn(
        "majoration_nouvelle",
        F.lit(MAJ_MIN)
        + F.lit(AMPLITUDE) * F.col("smoothstep")
    )

    # Arrondi éventuel à 4 décimales
    .withColumn(
        "majoration_nouvelle",
        F.round(F.col("majoration_nouvelle"), 4)
    )
)


# ============================================================
# 4. Comparaison avec l'ancienne majoration
# ============================================================

df_compare = (
    df_new

    .withColumn(
        "ecart",
        F.col("majoration_nouvelle")
        - F.col("majoration_ancienne")
    )

    .withColumn(
        "ecart_absolu",
        F.abs(F.col("ecart"))
    )

    .withColumn(
        "ecart_relatif",
        F.when(
            F.col("majoration_ancienne") != 0,
            F.col("ecart")
            / F.col("majoration_ancienne")
        )
    )

    .withColumn(
        "hausse_baisse",
        F.when(F.col("ecart") > 0, "HAUSSE")
         .when(F.col("ecart") < 0, "BAISSE")
         .otherwise("IDENTIQUE")
    )
)


# ============================================================
# 5. Vue individuelle
# ============================================================

df_compare.select(
    "ID",
    "charges_Sinistre",
    "majoration_ancienne",
    "majoration_nouvelle",
    "ecart",
    "ecart_absolu",
    "ecart_relatif",
    "hausse_baisse",
    "x"
).show(30, truncate=False)
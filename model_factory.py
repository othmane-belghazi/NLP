from pyspark.sql import functions as F

# ============================================================
# PARAMÈTRES
# ============================================================

P_BAS = 0.10       # seuil bas global : P10
P_HAUT = 0.95      # seuil haut global : P95

MAJ_MIN = 1.14
MAJ_MAX = 1.30

ALPHA = 2.0        # courbe convexe : x²


# ============================================================
# 1. CALCUL DES SEUILS GLOBAUX
#    IMPORTANT : pas de calcul par échéance
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

params = quantiles.collect()[0]

C_bas = float(params["C_bas"])
C_haut = float(params["C_haut"])

print(f"C_bas  = {C_bas:.2f} €")
print(f"C_haut = {C_haut:.2f} €")


# ============================================================
# 2. NORMALISATION DE LA CHARGE
# ============================================================

df_new = (
    df

    # Position de la charge entre C_bas et C_haut
    .withColumn(
        "x_brut",
        (
            F.col("charges_Sinistre") - F.lit(C_bas)
        )
        /
        F.lit(C_haut - C_bas)
    )

    # On borne entre 0 et 1
    .withColumn(
        "x",
        F.when(F.col("x_brut") <= 0, F.lit(0.0))
         .when(F.col("x_brut") >= 1, F.lit(1.0))
         .otherwise(F.col("x_brut"))
    )

    # ========================================================
    # 3. COURBE CONVEXE
    # ========================================================
    #
    # x = 0     -> 1.14
    # x = 0.5   -> 1.18
    # x = 1     -> 1.30
    #
    .withColumn(
        "x_alpha",
        F.pow(F.col("x"), F.lit(ALPHA))
    )

    # ========================================================
    # 4. CALCUL DE LA NOUVELLE MAJORATION
    # ========================================================

    .withColumn(
        "majoration_nouvelle",
        F.round(
            F.lit(MAJ_MIN)
            + (F.lit(MAJ_MAX) - F.lit(MAJ_MIN))
            * F.col("x_alpha"),
            4
        )
    )
)


# ============================================================
# 5. COMPARAISON AVEC L'ANCIENNE MAJORATION
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
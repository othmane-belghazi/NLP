import numpy as np
import pandas as pd


# Paramètres
COEF_MIN = 1.14
COEF_MAX = 1.30

# --------------------------------------------------
# 1. Population avec exactement 1 sinistre
# --------------------------------------------------
mask_1 = df["nb_sinistres"] == 1

# On travaille sur le montant
montant = df.loc[mask_1, "montant_sinistre"].clip(lower=0)

# --------------------------------------------------
# 2. Transformation log
# --------------------------------------------------
montant_log = np.log1p(montant)

# --------------------------------------------------
# 3. Rang percentile
# --------------------------------------------------
n = len(montant_log)

rang = montant_log.rank(method="average")

percentile = (rang - 0.5) / n

# --------------------------------------------------
# 4. Transformation en coefficient
# --------------------------------------------------
coef_1 = (
    COEF_MIN
    + (COEF_MAX - COEF_MIN) * percentile
)

# --------------------------------------------------
# 5. Règle finale
# --------------------------------------------------
df["coefficient_majoration"] = np.nan

# 1 sinistre
df.loc[mask_1, "coefficient_majoration"] = coef_1

# 2 sinistres ou plus
df.loc[
    df["nb_sinistres"] >= 2,
    "coefficient_majoration"
] = COEF_MAX

# Arrondi
df["coefficient_majoration"] = (
    df["coefficient_majoration"].round(3)
)
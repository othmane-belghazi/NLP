import numpy as np
import pandas as pd

# Paramètres
COEF_MIN = 1.14
COEF_MAX = 1.30

# 1. Population de calibration :
# uniquement les clients avec exactement 1 sinistre responsable
mask_1_sinistre = df["nb_sinistres"] == 1

montants_1 = df.loc[mask_1_sinistre, "montant_sinistre"].dropna()

# 2. Bornes robustes : P5 et P95
P5 = montants_1.quantile(0.05)
P95 = montants_1.quantile(0.95)

# 3. Calcul du score montant entre 0 et 1
score_montant = (
    (df["montant_sinistre"] - P5) / (P95 - P5)
).clip(0, 1)

# 4. Calcul du coefficient
df["coefficient_majoration"] = np.where(
    df["nb_sinistres"] >= 2,
    COEF_MAX,
    COEF_MIN + (COEF_MAX - COEF_MIN) * score_montant
)

# 5. Arrondi éventuel
df["coefficient_majoration"] = df["coefficient_majoration"].round(3)
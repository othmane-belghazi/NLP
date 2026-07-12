import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paramètres
# =========================
# df = votre portefeuille
# ebm = votre modèle EBM déjà entraîné
# feature_cols = variables utilisées par le modèle
# Hypothèse : Prime_Nm1 contient la prime avant renouvellement

chocs = [0.00, 0.01, 0.03, 0.05, 0.10]

# =========================
# Base portefeuille
# =========================
df_base = df.copy()

X_base = df_base[feature_cols]
proba_base = ebm.predict_probs(X_base)

# Si le modèle renvoie 2 colonnes, on prend la proba de résiliation (classe 1)
if len(np.shape(proba_base)) == 2:
    proba_base = proba_base[:, 1]

taux_base = np.mean(proba_base)
prime_base = np.mean(df_base["Prime_N"])

# =========================
# Scénarios de choc tarifaire
# =========================
resultats = []

for choc in chocs:
    df_s = df_base.copy()

    # Prime après choc construite à partir de Prime_Nm1
    df_s["Prime_N"] = df_s["Prime_Nm1"] * (1 + choc)

    # Variable tarifaire cohérente
    df_s["Delta_cotisation"] = df_s["Prime_N"] - df_s["Prime_Nm1"]

    # À adapter si votre définition métier de Majoration_N est différente
    df_s["Majoration_N"] = df_s["Delta_cotisation"]

    X_s = df_s[feature_cols]
    proba_s = ebm.predict_probs(X_s)

    if len(np.shape(proba_s)) == 2:
        proba_s = proba_s[:, 1]

    taux_s = np.mean(proba_s)
    prime_s = np.mean(df_s["Prime_N"])

    if choc == 0:
        elasticite = np.nan
    else:
        elasticite = ((taux_s - taux_base) / taux_base) / ((prime_s - prime_base) / prime_base)

    resultats.append({
        "choc_tarifaire": choc,
        "prime_moyenne": prime_s,
        "taux_resiliation_prevu": taux_s,
        "variation_taux_vs_base": (taux_s - taux_base) / taux_base if taux_base != 0 else np.nan,
        "elasticite_portefeuille": elasticite
    })

res = pd.DataFrame(resultats)

print("Taux de résiliation base :", round(taux_base, 6))
print("Prime moyenne base        :", round(prime_base, 6))
print()
print(res)

# =========================
# Visuel : évolution du taux de résiliation
# =========================
plt.figure(figsize=(9, 5))
plt.plot(res["choc_tarifaire"] * 100, res["taux_resiliation_prevu"] * 100, marker="o")
plt.axhline(taux_base * 100, linestyle="--")
plt.xlabel("Choc tarifaire (%)")
plt.ylabel("Taux de résiliation prédit (%)")
plt.title("Évolution du taux de résiliation prédit du portefeuille")
plt.grid(True)
plt.show()
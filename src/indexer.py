import pandas as pd

# --- 1. Charge tes données ---
# Colonnes attendues : tranche_prime, segment, ab_test, taux_resiliation
# ab_test en %, ex: 0, 1, 2, -1, -2   (PAS "0%", juste le nombre)
# taux_resiliation en %, ex: 2.6
df = pd.read_csv("mes_donnees.csv")

# Si l'AB test est écrit "0%", "+2%"... décommente la ligne suivante :
# df["ab_test"] = df["ab_test"].str.replace("%", "").str.replace("+", "").astype(float)

# --- 2. Indice prix et rétention ---
df["indice_prix"] = 100 + df["ab_test"]
df["retention"]   = 100 - df["taux_resiliation"]

# --- 3. Fonction élasticité d'arc (point milieu) ---
def arc_elasticite(R_test, R_ref, I_test, I_ref=100):
    num = (R_test - R_ref) / ((R_ref + R_test) / 2)   # variation rétention
    den = (I_test - I_ref) / ((I_ref + I_test) / 2)   # variation prix
    return num / den if den != 0 else None

# --- 4. Calcul par segment (et par tranche de prime) ---
# On compare chaque bras au bras 0% du MEME groupe
resultats = []
for (tranche, segment), g in df.groupby(["tranche_prime", "segment"]):
    ref = g[g["ab_test"] == 0]
    if ref.empty:
        continue  # pas de bras 0% -> pas de référence
    R_ref = ref["retention"].iloc[0]

    for _, ligne in g[g["ab_test"] != 0].iterrows():
        eps = arc_elasticite(ligne["retention"], R_ref, ligne["indice_prix"])
        resultats.append({
            "tranche_prime": tranche,
            "segment": segment,
            "ab_test": ligne["ab_test"],
            "resil_0%": ref["taux_resiliation"].iloc[0],
            "resil_test": ligne["taux_resiliation"],
            "elasticite": round(eps, 2) if eps else None,
        })

res = pd.DataFrame(resultats)
print(res)

# --- 5. (Optionnel) Élasticité moyenne par segment ---
print(res.groupby("segment")["elasticite"].mean().round(2))
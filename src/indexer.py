import pandas as pd

# stats_hm = ta DataFrame issue de .agg().toPandas()

# --- 1. Renommer pour simplifier ---
df = stats_hm.rename(columns={
    "POL_Segment_CourbesPrimeELR_simu": "segment",
    "tranche_prime": "tranche",
    "POL_MajorationPriceTest_simu": "majoration",
    "taux_resil": "resil",
    "nb_contrats": "nb",
})

# --- 2. Normaliser les échelles ---
# majoration -> points de % (0, 1, 2, -1, -2)
if df["majoration"].abs().max() <= 1:          # stockée en décimal (0.02) ?
    df["majoration"] = df["majoration"] * 100
df["majoration"] = df["majoration"].round(0).astype(int)

# résiliation : fraction (0.026) ou pourcentage (2.6) ? -> rétention = BASE - resil
BASE = 1 if df["resil"].max() <= 1 else 100
df["retention"] = BASE - df["resil"]

# --- 3. Fonction élasticité d'arc vs bras 0% ---
def elasticite_arc(g):
    ref = g[g["majoration"] == 0]
    if ref.empty:
        return pd.DataFrame()                  # pas de référence 0% -> on saute
    R0 = ref["retention"].iloc[0]
    lignes = []
    for _, r in g[g["majoration"] != 0].sort_values("majoration").iterrows():
        I = 100 + r["majoration"]              # indice prix (0% = 100)
        num = (r["retention"] - R0) / ((R0 + r["retention"]) / 2)
        den = (I - 100) / ((100 + I) / 2)
        lignes.append({**r.to_dict(),
                       "elasticite": round(num / den, 2) if den else None})
    return pd.DataFrame(lignes)

# --- 4. ÉLASTICITÉ PAR SEGMENT (tranches agrégées, pondéré par nb) ---
seg = (df.groupby(["segment", "majoration"])
         .apply(lambda x: pd.Series({
             "nb": x["nb"].sum(),
             "resil": (x["resil"] * x["nb"]).sum() / x["nb"].sum(),  # moyenne pondérée
         }), include_groups=False)
         .reset_index())
seg["retention"] = BASE - seg["resil"]

elast_segment = (seg.groupby("segment", group_keys=False)
                    .apply(elasticite_arc, include_groups=False)
                    .reset_index(drop=True))

# --- 5. ÉLASTICITÉ PAR TRANCHE DANS CHAQUE SEGMENT ---
elast_tranche = (df.groupby(["segment", "tranche"], group_keys=False)
                   .apply(elasticite_arc, include_groups=False)
                   .reset_index(drop=True))

# =================== RÉSULTATS ===================

# A) Tableau GLOBAL : élasticité par segment (segments en lignes, bras en colonnes)
tableau_global = elast_segment.pivot_table(
    index="segment", columns="majoration", values="elasticite")
print("=== ÉLASTICITÉ PAR SEGMENT ===")
print(tableau_global, "\n")

# B) Un tableau par segment : élasticité par tranche de prime
for s in elast_tranche["segment"].unique():
    sous = (elast_tranche[elast_tranche["segment"] == s]
            .pivot_table(index="tranche", columns="majoration", values="elasticite"))
    print(f"=== Segment : {s} — élasticité par tranche de prime ===")
    print(sous, "\n")
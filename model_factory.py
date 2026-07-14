import numpy as np
import pandas as pd

def elasticite_portefeuille(df, model, chocs=np.arange(-0.05, 0.06, 0.01)):
    resultats = []
    taux_base = model.predict(df).mean()          # taux de résiliation au tarif actuel
    retention_base = 1 - taux_base

    for choc in chocs:
        df_choc = df.copy()
        df_choc["prime"] = df["prime"] * (1 + choc)
        taux = model.predict(df_choc).mean()
        retention = 1 - taux
        # élasticité de la rétention (convention "demande")
        elast = ((retention - retention_base) / retention_base) / choc if choc != 0 else np.nan
        resultats.append({"choc_%": choc * 100,
                          "taux_resiliation_%": taux * 100,
                          "retention_%": retention * 100,
                          "elasticite": elast})

    return pd.DataFrame(resultats)

# Exemple d'utilisation
# res = elasticite_portefeuille(df_portefeuille, mon_modele)
# print(res)
# res.plot(x="choc_%", y="taux_resiliation_%")   # courbe de demande
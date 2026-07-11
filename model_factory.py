from pyspark.sql import functions as F
from functools import reduce

# 0. S'assurer que les dates sont bien en DateType (adapter le format si besoin)
date_cols = ["date_echeance"] + [f"date_demande_releve_{i}" for i in range(1, 11)]
for c in date_cols:
    df = df.withColumn(c, F.to_date(F.col(c)))

# 1. Date de référence = réception de la quittance = point de scoring
#    Aucune info postérieure ne doit entrer dans les features
df = df.withColumn("date_quittance", F.date_sub(F.col("date_echeance"), 30))

# 2. Début de la fenêtre : 6 mois avant la quittance
df = df.withColumn("debut_fenetre", F.add_months(F.col("date_quittance"), -6))

# 3. Au moins une demande dans [debut_fenetre, date_quittance]
releve_cols = [f"date_demande_releve_{i}" for i in range(1, 11)]
conditions = [
    F.col(c).isNotNull()
    & (F.col(c) >= F.col("debut_fenetre"))
    & (F.col(c) <= F.col("date_quittance"))
    for c in releve_cols
]
demande_dans_fenetre = reduce(lambda a, b: a | b, conditions)

df = df.withColumn(
    "releve_demande_6m_avant_quittance",
    demande_dans_fenetre.cast("integer")
)
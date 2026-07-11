from pyspark.sql import functions as F

# 1. Dates en DateType (adapter le format si besoin)
df = (
    df.withColumn("date_echeance", F.to_date("date_echeance"))
      .withColumn("date_demande_releve_1", F.to_date("date_demande_releve_1"))
)

# 2. Point de scoring = réception de la quittance = échéance − 30 j
df = df.withColumn("date_quittance", F.date_sub("date_echeance", 30))

# 3. Début de fenêtre = 6 mois avant la quittance
df = df.withColumn("debut_fenetre", F.add_months("date_quittance", -6))

# 4. Variable binaire : demande dans [debut_fenetre, date_quittance]
df = df.withColumn(
    "releve_demande_6m_avant_quittance",
    (
        F.col("date_demande_releve_1").isNotNull()
        & (F.col("date_demande_releve_1") >= F.col("debut_fenetre"))
        & (F.col("date_demande_releve_1") <= F.col("date_quittance"))
    ).cast("integer")
)
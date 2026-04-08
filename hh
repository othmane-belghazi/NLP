# Databricks notebook source
# MAGIC %md
# MAGIC # Analyse d'impact ELR & Majorations sur le Terme
# MAGIC ---

# COMMAND ----------

from pyspark.sql import functions as F, Window
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Préparation des données

# COMMAND ----------

# --- Adapter les noms de colonnes à ta base ---
# df = spark.table("ta_table")

# Calcul des deltas
df = (
    df
    .withColumn("delta_elr", F.col("elr_new_ajuste") - F.col("elr_old"))
    .withColumn("delta_elr_pct",
                (F.col("elr_new_ajuste") - F.col("elr_old")) / F.col("elr_old") * 100)
    .withColumn("delta_maj", F.col("majoration_new_ajuste") - F.col("majoration_old"))
    .withColumn("delta_maj_pct",
                F.when(F.col("majoration_old") == 0, F.lit(None))
                 .otherwise((F.col("majoration_new_ajuste") - F.col("majoration_old")) / F.col("majoration_old") * 100))
    .withColumn("delta_prime", F.col("prime_pure_new_ajuste") - F.col("prime_pure_old"))
    .withColumn("delta_prime_pct",
                (F.col("prime_pure_new_ajuste") - F.col("prime_pure_old")) / F.col("prime_pure_old") * 100)
)

# Buckets majoration
df = df.withColumn("bucket_maj",
    F.when(F.col("delta_maj_pct").isNull(), "maj_old = 0")
     .when(F.col("delta_maj_pct") <= -10, "baisse > 10%")
     .when(F.col("delta_maj_pct") <= -5,  "baisse 5-10%")
     .when(F.col("delta_maj_pct") <= 0,   "baisse 0-5%")
     .when(F.col("delta_maj_pct") <= 5,   "hausse 0-5%")
     .when(F.col("delta_maj_pct") <= 10,  "hausse 5-10%")
     .when(F.col("delta_maj_pct") <= 20,  "hausse 10-20%")
     .otherwise("hausse > 20%")
)

# Buckets prime
df = df.withColumn("bucket_prime",
    F.when(F.col("delta_prime_pct") <= -5, "baisse > 5%")
     .when(F.col("delta_prime_pct") <= 0,  "baisse 0-5%")
     .when(F.col("delta_prime_pct") <= 5,  "hausse 0-5%")
     .when(F.col("delta_prime_pct") <= 10, "hausse 5-10%")
     .when(F.col("delta_prime_pct") <= 20, "hausse 10-20%")
     .otherwise("hausse > 20%")
)

pdf_all = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Vue globale portefeuille

# COMMAND ----------

df.select(
    F.count("*").alias("nb_contrats"),
    F.round(F.sum("prime_pure_old"), 0).alias("prime_totale_old"),
    F.round(F.sum("prime_pure_new_ajuste"), 0).alias("prime_totale_new"),
    F.round(F.sum("delta_prime"), 0).alias("delta_total_eur"),
    F.round(F.sum("delta_prime") / F.sum("prime_pure_old") * 100, 2).alias("delta_total_pct"),
    F.round(F.mean("delta_maj_pct"), 2).alias("moy_delta_maj_pct"),
    F.round(F.percentile_approx("delta_maj_pct", 0.5), 2).alias("med_delta_maj_pct"),
).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Distribution des variations de majoration (histogramme)

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
palette = "#2C5F8A"

# 3a - Histo delta majoration %
data_maj = pdf_all["delta_maj_pct"].dropna()
axes[0].hist(data_maj, bins=50, color=palette, edgecolor="white", alpha=0.85)
axes[0].axvline(0, color="red", linestyle="--", linewidth=1.2)
axes[0].axvline(data_maj.median(), color="orange", linestyle="--", linewidth=1.2, label=f"médiane {data_maj.median():.1f}%")
axes[0].set_title("Distribution Δ Majoration (%)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Variation majoration (%)")
axes[0].set_ylabel("Nb contrats")
axes[0].legend()

# 3b - Histo delta ELR %
data_elr = pdf_all["delta_elr_pct"].dropna()
axes[1].hist(data_elr, bins=50, color="#5B9E6F", edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2)
axes[1].axvline(data_elr.median(), color="orange", linestyle="--", linewidth=1.2, label=f"médiane {data_elr.median():.1f}%")
axes[1].set_title("Distribution Δ ELR (%)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Variation ELR (%)")
axes[1].legend()

# 3c - Histo delta prime %
data_prime = pdf_all["delta_prime_pct"].dropna()
axes[2].hist(data_prime, bins=50, color="#D4853B", edgecolor="white", alpha=0.85)
axes[2].axvline(0, color="red", linestyle="--", linewidth=1.2)
axes[2].axvline(data_prime.median(), color="orange", linestyle="--", linewidth=1.2, label=f"médiane {data_prime.median():.1f}%")
axes[2].set_title("Distribution Δ Prime Pure (%)", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Variation prime pure (%)")
axes[2].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Nb contrats & impact € par bucket de majoration (bar chart)

# COMMAND ----------

bucket_order = ["baisse > 10%", "baisse 5-10%", "baisse 0-5%",
                "hausse 0-5%", "hausse 5-10%", "hausse 10-20%", "hausse > 20%", "maj_old = 0"]

stats_maj = (
    df.groupBy("bucket_maj")
    .agg(
        F.count("*").alias("nb_contrats"),
        F.round(F.mean("delta_maj_pct"), 2).alias("moy_delta_maj_pct"),
        F.round(F.mean("delta_prime_pct"), 2).alias("moy_delta_prime_pct"),
        F.round(F.sum("delta_prime"), 0).alias("impact_prime_eur"),
        F.round(F.mean("delta_prime"), 0).alias("moy_delta_prime_eur"),
    )
    .toPandas()
)
stats_maj["bucket_maj"] = pd.Categorical(stats_maj["bucket_maj"], categories=bucket_order, ordered=True)
stats_maj = stats_maj.sort_values("bucket_maj").dropna(subset=["bucket_maj"])

fig, ax1 = plt.subplots(figsize=(14, 6))

x = np.arange(len(stats_maj))
width = 0.4

# Barres nb contrats
bars1 = ax1.bar(x - width/2, stats_maj["nb_contrats"], width,
                color="#2C5F8A", alpha=0.85, label="Nb contrats")
ax1.set_ylabel("Nb contrats", color="#2C5F8A", fontsize=12)
ax1.tick_params(axis="y", labelcolor="#2C5F8A")

# Axe 2 : impact €
ax2 = ax1.twinx()
colors_eur = ["#D4853B" if v >= 0 else "#5B9E6F" for v in stats_maj["impact_prime_eur"]]
bars2 = ax2.bar(x + width/2, stats_maj["impact_prime_eur"], width,
                color=colors_eur, alpha=0.85, label="Impact prime (€)")
ax2.set_ylabel("Impact prime total (€)", color="#D4853B", fontsize=12)
ax2.tick_params(axis="y", labelcolor="#D4853B")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

ax1.set_xticks(x)
ax1.set_xticklabels(stats_maj["bucket_maj"], rotation=30, ha="right")
ax1.set_title("Répartition contrats & Impact prime par bucket de Majoration", fontsize=14, fontweight="bold")

fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.95))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Scatter : Δ Majoration vs Δ Prime (cohérence)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(10, 7))
sample = pdf_all.dropna(subset=["delta_maj_pct", "delta_prime_pct"])
if len(sample) > 5000:
    sample = sample.sample(5000, random_state=42)

ax.scatter(sample["delta_maj_pct"], sample["delta_prime_pct"],
           alpha=0.3, s=12, color="#2C5F8A", edgecolors="none")
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")

# Droite de tendance
z = np.polyfit(sample["delta_maj_pct"], sample["delta_prime_pct"], 1)
p = np.poly1d(z)
x_line = np.linspace(sample["delta_maj_pct"].min(), sample["delta_maj_pct"].max(), 100)
ax.plot(x_line, p(x_line), color="red", linewidth=1.5, linestyle="-",
        label=f"tendance (pente={z[0]:.2f})")

ax.set_xlabel("Δ Majoration (%)", fontsize=12)
ax.set_ylabel("Δ Prime Pure (%)", fontsize=12)
ax.set_title("Cohérence : variation Majoration vs variation Prime", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Boxplot Δ Prime par bucket de majoration

# COMMAND ----------

fig, ax = plt.subplots(figsize=(14, 6))

groups = []
labels = []
for b in bucket_order:
    vals = pdf_all.loc[pdf_all["bucket_maj"] == b, "delta_prime_pct"].dropna()
    if len(vals) > 0:
        groups.append(vals.values)
        labels.append(b)

bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.6)
for patch in bp["boxes"]:
    patch.set_facecolor("#2C5F8A")
    patch.set_alpha(0.6)
for median in bp["medians"]:
    median.set_color("red")
    median.set_linewidth(2)

ax.set_xticklabels(labels, rotation=30, ha="right")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_ylabel("Δ Prime Pure (%)", fontsize=12)
ax.set_title("Distribution Δ Prime par bucket de Majoration", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Heatmap : Majoration Old vs New (transition)

# COMMAND ----------

# Créer des buckets pour les niveaux absolus de majoration
for col_name, new_col in [("majoration_old", "level_maj_old"), ("majoration_new_ajuste", "level_maj_new")]:
    pdf_all[new_col] = pd.cut(
        pdf_all[col_name],
        bins=[-np.inf, 0, 0.05, 0.10, 0.20, 0.30, 0.50, np.inf],
        labels=["≤0%", "0-5%", "5-10%", "10-20%", "20-30%", "30-50%", ">50%"]
    )

transition = pd.crosstab(pdf_all["level_maj_old"], pdf_all["level_maj_new"], normalize="index") * 100

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(transition.values, cmap="YlOrRd", aspect="auto")

ax.set_xticks(range(len(transition.columns)))
ax.set_xticklabels(transition.columns, rotation=45, ha="right")
ax.set_yticks(range(len(transition.index)))
ax.set_yticklabels(transition.index)
ax.set_xlabel("Majoration NEW ajustée", fontsize=12)
ax.set_ylabel("Majoration OLD", fontsize=12)
ax.set_title("Matrice de transition Majoration Old → New (%)", fontsize=14, fontweight="bold")

# Annotations
for i in range(len(transition.index)):
    for j in range(len(transition.columns)):
        val = transition.values[i, j]
        if val > 1:
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color="white" if val > 40 else "black")

plt.colorbar(im, ax=ax, label="% des contrats (par ligne)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Waterfall : décomposition de l'impact prime (ELR vs Majoration)

# COMMAND ----------

# Approximation : impact ELR seul + impact majoration seul + effet croisé
prime_old_total = pdf_all["prime_pure_old"].sum()
prime_new_total = pdf_all["prime_pure_new_ajuste"].sum()

# Proxy : prime si on ne changeait que l'ELR (majoration old)
# prime_elr_only ≈ prime_old * (elr_new / elr_old) si prime = f(elr) * (1 + maj)
# Sinon on fait un calcul simplifié
pdf_all["ratio_elr"] = pdf_all["elr_new_ajuste"] / pdf_all["elr_old"]
pdf_all["prime_elr_only"] = pdf_all["prime_pure_old"] * pdf_all["ratio_elr"]

impact_elr = pdf_all["prime_elr_only"].sum() - prime_old_total
impact_maj = prime_new_total - pdf_all["prime_elr_only"].sum()
impact_total = prime_new_total - prime_old_total

labels = ["Prime Old", "Impact ELR", "Impact Majoration", "Prime New"]
values = [prime_old_total, impact_elr, impact_maj, prime_new_total]
bottoms = [0, prime_old_total, prime_old_total + impact_elr, 0]
colors = ["#2C5F8A",
          "#5B9E6F" if impact_elr >= 0 else "#D9534F",
          "#D4853B" if impact_maj >= 0 else "#D9534F",
          "#2C5F8A"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, [prime_old_total, impact_elr, impact_maj, prime_new_total],
              bottom=[0, prime_old_total, prime_old_total + impact_elr, 0],
              color=colors, edgecolor="white", width=0.5)

for bar, val in zip(bars, [prime_old_total, impact_elr, impact_maj, prime_new_total]):
    y_pos = bar.get_y() + bar.get_height() / 2
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
            f"{val:,.0f} €", ha="center", va="center", fontweight="bold", fontsize=11)

ax.set_title("Waterfall : décomposition impact ELR vs Majoration sur la prime",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Prime Pure (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Tableau récapitulatif (à exporter si besoin)

# COMMAND ----------

recap = (
    df.groupBy("bucket_maj")
    .agg(
        F.count("*").alias("nb_contrats"),
        F.round(F.mean("majoration_old"), 4).alias("moy_maj_old"),
        F.round(F.mean("majoration_new_ajuste"), 4).alias("moy_maj_new"),
        F.round(F.mean("delta_maj_pct"), 2).alias("moy_delta_maj_pct"),
        F.round(F.mean("delta_prime_pct"), 2).alias("moy_delta_prime_pct"),
        F.round(F.sum("delta_prime"), 0).alias("impact_prime_total_eur"),
        F.round(F.mean("prime_pure_old"), 0).alias("moy_prime_old"),
        F.round(F.mean("prime_pure_new_ajuste"), 0).alias("moy_prime_new"),
    )
    .orderBy("bucket_maj")
)
recap.show(truncate=False)

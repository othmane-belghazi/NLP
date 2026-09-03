# Databricks notebook source
# MAGIC %md
# MAGIC # Outil de segmentation et d'analyse
# MAGIC
# MAGIC **Pré-requis** : `df` (la base) et `model` (modèle déjà entraîné) existent déjà dans le notebook.
# MAGIC
# MAGIC **Utilisation**
# MAGIC 1. Exécuter les cellules 1 à 3 une première fois (initialisation, widgets, fonctions).
# MAGIC 2. Choisir les paramètres dans la barre de widgets en haut du notebook.
# MAGIC 3. Exécuter la cellule 4 « Exécution » autant de fois que nécessaire.
# MAGIC
# MAGIC **Syntaxe des ratios** (widget 5) : `nom = expression ; nom2 = expression2`
# MAGIC ex. `taux_util = solde / limite ; ratio = (a + b) / c`
# MAGIC (un nom de colonne avec espaces s'écrit entre backticks : `` `ma colonne` ``)

# COMMAND ----------

# MAGIC %md ## 1. Initialisation

# COMMAND ----------

import pandas as pd
import numpy as np

# Si df est un DataFrame Spark, on le convertit en pandas
if hasattr(df, "toPandas"):
    df = df.toPandas()

# Fonctions d'agrégation disponibles : ajouter une ligne ici pour en proposer une nouvelle
AGG_DISPONIBLES = {
    "moyenne":    "mean",
    "mediane":    "median",
    "somme":      "sum",
    "min":        "min",
    "max":        "max",
    "nb_obs":     "count",
    "ecart_type": "std",
    "p90":        lambda s: s.quantile(0.90),
}

# Une variable numérique ayant au plus ce nombre de valeurs distinctes est traitée comme discrète
SEUIL_DISCRET = 20

# COMMAND ----------

# MAGIC %md ## 2. Widgets (paramètres interactifs)

# COMMAND ----------

colonnes = [str(c) for c in df.columns]

# dbutils.widgets.removeAll()   # décommenter pour réinitialiser tous les widgets

dbutils.widgets.multiselect("seg_vars", colonnes[0], colonnes, "1. Variables de segmentation")
dbutils.widgets.text("n_bins", "5", "2. Nb de bins (variables continues)")
dbutils.widgets.multiselect("agg_vars", colonnes[0], colonnes, "3. Variables à agréger")
dbutils.widgets.multiselect("agg_funcs", "moyenne", list(AGG_DISPONIBLES), "4. Fonctions d'agrégation")
dbutils.widgets.text("ratios", "", "5. Ratios : nom = expr ; nom2 = expr2")
dbutils.widgets.dropdown("ratio_mode", "moyenne des ratios", ["moyenne des ratios", "ratio des sommes"], "5b. Mode de calcul des ratios")
dbutils.widgets.dropdown("utiliser_modele", "oui", ["oui", "non"], "6. Utiliser le modèle")
dbutils.widgets.text("model_features", "", "6b. Features du modèle (vide = auto)")
dbutils.widgets.dropdown("target", "(aucune)", ["(aucune)"] + colonnes, "7. Variable cible (taux réel)")

# COMMAND ----------

# MAGIC %md ## 3. Fonctions

# COMMAND ----------

def grouper(df, seg_cols):
    """Regroupement par segment, utilisé par toutes les fonctions ci-dessous."""
    return df.groupby(seg_cols, observed=True)


# ---------- Segmentation ----------

def creer_segments(df, seg_vars, n_bins):
    """Ajoute une colonne 'seg_<var>' par variable de segmentation.
    - variable continue  -> classes créées avec qcut
    - variable discrète  -> modalités telles quelles
    """
    df = df.copy()
    seg_cols = []
    for var in seg_vars:
        col = f"seg_{var}"
        s = df[var]
        est_continue = pd.api.types.is_numeric_dtype(s) and s.nunique() > SEUIL_DISCRET
        if est_continue:
            df[col] = pd.qcut(s, q=n_bins, duplicates="drop")
            df[col] = df[col].cat.add_categories("Manquant").fillna("Manquant")
        else:
            df[col] = s.astype("object").fillna("Manquant").astype(str)
        seg_cols.append(col)
    return df, seg_cols


# ---------- Probabilité du modèle (calculée une fois, ligne par ligne) ----------

def ajouter_proba_modele(df, model, features, target):
    """Ajoute la colonne 'proba_modele' = probabilité prédite par le modèle pour chaque ligne."""
    if not features:                                                  # 1) features saisies par l'utilisateur
        features = list(getattr(model, "feature_names_in_", []))      # 2) sinon celles connues du modèle (sklearn)
    if not features:                                                  # 3) sinon toutes les numériques hors cible
        features = [c for c in df.select_dtypes("number").columns if c != target]
    X = df[features]
    if hasattr(model, "predict_proba"):
        df["proba_modele"] = model.predict_proba(X)[:, 1]
    else:
        df["proba_modele"] = model.predict(X)
    return df


# ---------- Blocs d'indicateurs (chacun renvoie un DataFrame indexé par segment) ----------

def calculer_effectifs(df, seg_cols):
    """Nombre d'observations et poids (%) de chaque segment."""
    eff = grouper(df, seg_cols).size().rename("effectif").to_frame()
    eff["poids_pct"] = 100 * eff["effectif"] / eff["effectif"].sum()
    return eff


def calculer_taux_reel(df, seg_cols, target):
    """Taux réel observé = moyenne de la variable cible par segment."""
    if not target:
        return pd.DataFrame()
    return grouper(df, seg_cols)[target].mean().rename("taux_reel").to_frame()


def calculer_proba_moyenne(df, seg_cols):
    """Probabilité moyenne prédite par le modèle par segment."""
    if "proba_modele" not in df:
        return pd.DataFrame()
    return grouper(df, seg_cols)["proba_modele"].mean().to_frame()


def calculer_agregats(df, seg_cols, agg_vars, agg_funcs):
    """Une colonne '<variable>_<fonction>' par variable et par fonction choisie."""
    if not agg_vars or not agg_funcs:
        return pd.DataFrame()
    grp = grouper(df, seg_cols)
    res = pd.DataFrame()
    for var in agg_vars:
        for f in agg_funcs:
            res[f"{var}_{f}"] = grp[var].agg(AGG_DISPONIBLES[f])
    return res


def parser_ratios(texte):
    """'taux = a / b ; r = c / (d + e)'  ->  {'taux': 'a / b', 'r': 'c / (d + e)'}"""
    ratios = {}
    for morceau in texte.split(";"):
        if "=" in morceau:
            nom, expr = morceau.split("=", 1)
            ratios[nom.strip()] = expr.strip()
    return ratios


def calculer_ratios(df, seg_cols, ratios, mode):
    """Indicateurs calculés entre plusieurs variables, agrégés par segment.
    - 'moyenne des ratios' : ratio calculé ligne par ligne, puis moyenne par segment
    - 'ratio des sommes'   : sommes par segment, puis ratio appliqué sur les sommes
    """
    if not ratios:
        return pd.DataFrame()
    if mode == "moyenne des ratios":
        tmp = df[seg_cols].copy()
        for nom, expr in ratios.items():
            tmp[nom] = df.eval(expr)
        res = grouper(tmp, seg_cols)[list(ratios)].mean()
    else:
        sommes = grouper(df, seg_cols).sum(numeric_only=True)
        res = pd.DataFrame({nom: sommes.eval(expr) for nom, expr in ratios.items()})
    return res.replace([np.inf, -np.inf], np.nan)


# ---------- Assemblage ----------

def calculer_indicateurs(df, seg_cols, p):
    """Assemble tous les blocs : une ligne par segment, une colonne par indicateur.
    Pour ajouter un nouveau type d'indicateur : écrire une fonction qui renvoie
    un DataFrame indexé par segment et l'ajouter à la liste 'blocs'."""
    blocs = [
        calculer_effectifs(df, seg_cols),
        calculer_taux_reel(df, seg_cols, p["target"]),
        calculer_proba_moyenne(df, seg_cols),
        calculer_agregats(df, seg_cols, p["agg_vars"], p["agg_funcs"]),
        calculer_ratios(df, seg_cols, p["ratios"], p["ratio_mode"]),
    ]
    tableau = pd.concat([b for b in blocs if not b.empty], axis=1)

    # Écart taux réel - probabilité modèle, placé juste après proba_modele
    if "taux_reel" in tableau and "proba_modele" in tableau:
        position = tableau.columns.get_loc("proba_modele") + 1
        tableau.insert(position, "ecart_reel_modele", tableau["taux_reel"] - tableau["proba_modele"])
    return tableau


def construire_tableau(df, p):
    """Tableau final : une ligne par modalité de segmentation + une ligne TOTAL."""
    df_seg, seg_cols = creer_segments(df, p["seg_vars"], p["n_bins"])
    if p["model"] is not None:
        df_seg = ajouter_proba_modele(df_seg, p["model"], p["model_features"], p["target"])

    # Lignes par segment
    tableau = calculer_indicateurs(df_seg, seg_cols, p).reset_index()
    tableau[seg_cols] = tableau[seg_cols].astype(str)

    # Ligne TOTAL (même calcul, sur un segment unique)
    df_seg["_total"] = "TOTAL"
    total = calculer_indicateurs(df_seg, ["_total"], p).reset_index(drop=True)
    for c in seg_cols:
        total[c] = ""
    total[seg_cols[0]] = "TOTAL"

    tableau = pd.concat([tableau, total], ignore_index=True)
    return tableau.rename(columns={c: c[len("seg_"):] for c in seg_cols}).round(4)

# COMMAND ----------

# MAGIC %md ## 4. Exécution

# COMMAND ----------

def lire_liste(nom):
    return [v.strip() for v in dbutils.widgets.get(nom).split(",") if v.strip()]

target = dbutils.widgets.get("target")

params = {
    "seg_vars":       lire_liste("seg_vars"),
    "n_bins":         int(dbutils.widgets.get("n_bins")),
    "agg_vars":       lire_liste("agg_vars"),
    "agg_funcs":      lire_liste("agg_funcs"),
    "ratios":         parser_ratios(dbutils.widgets.get("ratios")),
    "ratio_mode":     dbutils.widgets.get("ratio_mode"),
    "model":          model if dbutils.widgets.get("utiliser_modele") == "oui" else None,
    "model_features": lire_liste("model_features"),
    "target":         None if target == "(aucune)" else target,
}

tableau = construire_tableau(df, params)
display(tableau)

# Utilisation sans widgets (ex. pour enchaîner plusieurs tableaux dans une boucle) :
# tableau_age = construire_tableau(df, {**params, "seg_vars": ["age"], "n_bins": 10})

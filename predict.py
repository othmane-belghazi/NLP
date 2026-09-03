# Databricks notebook source
# MAGIC %md
# MAGIC # Outil de segmentation et d'analyse
# MAGIC
# MAGIC **Pré-requis** : `df` (la base) et `model` (modèle déjà entraîné) existent déjà dans le notebook.
# MAGIC
# MAGIC **Utilisation** : exécuter les cellules 1 à 4, puis tout se pilote depuis le panneau
# MAGIC (aucune modification de code n'est nécessaire pour changer de tableau).
# MAGIC
# MAGIC **Syntaxe des ratios** : `nom = expression` (une par ligne)
# MAGIC ex. `taux_util = solde / limite`

# COMMAND ----------

# MAGIC %md ## 1. Initialisation

# COMMAND ----------

import traceback

import numpy as np
import pandas as pd
import ipywidgets as w
from IPython.display import display as ip_display, HTML

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

# Palette du thème
BLEU_FONCE, BLEU, BLEU_CLAIR, BLEU_PALE = "#0B4F79", "#1B7FC4", "#D6EAF8", "#F2F9FD"

# COMMAND ----------

# MAGIC %md ## 2. Fonctions de calcul

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
    """'taux = a / b' (une expression par ligne) -> {'taux': 'a / b'}"""
    ratios = {}
    for ligne in texte.replace(";", "\n").splitlines():
        if "=" in ligne:
            nom, expr = ligne.split("=", 1)
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

# MAGIC %md ## 3. Mise en forme du tableau (thème bleu clair)

# COMMAND ----------

def tableau_html(tableau, seg_cols):
    """Rend le tableau en HTML : en-tête bleu, lignes alternées, TOTAL en gras."""
    num_cols = [c for c in tableau.columns if c not in seg_cols]

    styles = [
        {"selector": "",
         "props": [("border-collapse", "collapse"), ("font-family", "Segoe UI, Arial, sans-serif"),
                   ("font-size", "13px"), ("width", "100%"),
                   ("box-shadow", "0 1px 4px rgba(11,79,121,.15)")]},
        {"selector": "thead th",
         "props": [("background-color", BLEU_FONCE), ("color", "white"), ("padding", "9px 12px"),
                   ("text-align", "right"), ("border-bottom", f"2px solid {BLEU}"),
                   ("position", "sticky"), ("top", "0")]},
        {"selector": "tbody td",
         "props": [("padding", "7px 12px"), ("text-align", "right"),
                   ("border-bottom", f"1px solid {BLEU_CLAIR}")]},
        {"selector": "tbody tr:nth-child(even)", "props": [("background-color", BLEU_PALE)]},
        {"selector": "tbody tr:hover", "props": [("background-color", BLEU_CLAIR)]},
    ]

    def ligne_total(row):
        est_total = str(row.iloc[0]) == "TOTAL"
        style = f"background-color:{BLEU_CLAIR};font-weight:600;border-top:2px solid {BLEU_FONCE};"
        return [style if est_total else ""] * len(row)

    st = (tableau.style
          .hide(axis="index")
          .set_table_styles(styles)
          .set_properties(subset=seg_cols, **{"text-align": "left", "font-weight": "600",
                                              "color": BLEU_FONCE})
          .format("{:,.4f}", subset=num_cols, na_rep="–")
          .apply(ligne_total, axis=1))

    if "effectif" in tableau:
        st = st.format("{:,.0f}", subset=["effectif"])
    for col in ["taux_reel", "proba_modele"]:                       # colonnes clés en dégradé bleu
        if col in tableau:
            st = st.background_gradient(cmap="Blues", subset=[col], vmin=0)

    return f'<div style="overflow:auto;max-height:600px;border-radius:6px;">{st.to_html()}</div>'


def carte(titre, contenu):
    """Encadré blanc avec titre bleu, utilisé pour chaque section du panneau."""
    entete = w.HTML(f'<div style="background:{BLEU_CLAIR};color:{BLEU_FONCE};font-weight:600;'
                    f'font-size:13px;padding:6px 10px;border-radius:4px;margin-bottom:6px;'
                    f'font-family:Segoe UI,Arial,sans-serif;">{titre}</div>')
    boite = w.VBox([entete, contenu])
    boite.layout = w.Layout(border=f"1px solid {BLEU_CLAIR}", padding="10px",
                            margin="0 8px 10px 0", border_radius="6px")
    return boite

# COMMAND ----------

# MAGIC %md ## 4. Panneau interactif

# COMMAND ----------

colonnes = [str(c) for c in df.columns]
num_cols_df = [c for c in colonnes if pd.api.types.is_numeric_dtype(df[c])]

liste = w.Layout(width="230px", height="150px")
champ = w.Layout(width="230px")

# --- contrôles
sel_seg    = w.SelectMultiple(options=colonnes, value=(colonnes[0],), layout=liste)
sl_bins    = w.IntSlider(value=5, min=2, max=20, description="Bins", continuous_update=False,
                         layout=champ, style={"description_width": "40px"})
sel_agg_v  = w.SelectMultiple(options=num_cols_df, layout=liste)
sel_agg_f  = w.SelectMultiple(options=list(AGG_DISPONIBLES), value=("moyenne",), layout=liste)
txt_ratios = w.Textarea(placeholder="taux_util = solde / limite\nautre = (a + b) / c",
                        layout=w.Layout(width="300px", height="90px"))
mode_ratio = w.ToggleButtons(options=["moyenne des ratios", "ratio des sommes"],
                             style={"button_width": "145px", "font_size": "11px"})
chk_model  = w.Checkbox(value=True, description="Appliquer le modèle", indent=False)
txt_feats  = w.Text(placeholder="features (vide = auto)", layout=champ)
dd_target  = w.Dropdown(options=["(aucune)"] + colonnes, layout=champ)

btn_gen    = w.Button(description="Générer le tableau", icon="play",
                      layout=w.Layout(width="190px", height="38px"))
btn_reset  = w.Button(description="Réinitialiser", icon="refresh",
                      layout=w.Layout(width="150px", height="38px"))
btn_csv    = w.Button(description="Exporter en CSV", icon="download",
                      layout=w.Layout(width="170px", height="38px"))
btn_gen.style.button_color, btn_gen.style.font_weight = BLEU_FONCE, "600"
btn_reset.style.button_color = BLEU_CLAIR
btn_csv.style.button_color = BLEU_CLAIR
sortie = w.Output()

titre = w.HTML(
    f'<div style="background:linear-gradient(90deg,{BLEU_FONCE},{BLEU});color:white;'
    f'padding:14px 18px;border-radius:6px;font-family:Segoe UI,Arial,sans-serif;">'
    f'<div style="font-size:19px;font-weight:600;">Outil de segmentation et d\'analyse</div>'
    f'<div style="font-size:12px;opacity:.85;">'
    f'{len(df):,} observations · {len(colonnes)} variables'.replace(",", " ") + '</div></div>')


# --- actions des boutons
def lire_parametres():
    """Lit l'état des contrôles et renvoie le dictionnaire de paramètres."""
    return {
        "seg_vars":       list(sel_seg.value),
        "n_bins":         sl_bins.value,
        "agg_vars":       list(sel_agg_v.value),
        "agg_funcs":      list(sel_agg_f.value),
        "ratios":         parser_ratios(txt_ratios.value),
        "ratio_mode":     mode_ratio.value,
        "model":          model if chk_model.value else None,
        "model_features": [v.strip() for v in txt_feats.value.split(",") if v.strip()],
        "target":         None if dd_target.value == "(aucune)" else dd_target.value,
    }


def au_clic_generer(_):
    global tableau                      # le résultat reste réutilisable dans les autres cellules
    sortie.clear_output()
    with sortie:
        p = lire_parametres()
        if not p["seg_vars"]:
            ip_display(HTML(f'<b style="color:{BLEU_FONCE}">Sélectionner au moins une variable '
                            f'de segmentation.</b>'))
            return
        try:
            tableau = construire_tableau(df, p)
            ip_display(HTML(tableau_html(tableau, p["seg_vars"])))
        except Exception:
            ip_display(HTML(f'<pre style="color:#B3261E;font-size:12px">{traceback.format_exc()}</pre>'))


def au_clic_reset(_):
    sel_seg.value, sel_agg_v.value, sel_agg_f.value = (colonnes[0],), (), ("moyenne",)
    sl_bins.value, txt_ratios.value, txt_feats.value = 5, "", ""
    mode_ratio.index, chk_model.value, dd_target.value = 0, True, "(aucune)"
    sortie.clear_output()


def au_clic_csv(_):
    with sortie:
        if "tableau" not in globals():
            ip_display(HTML("<b>Générer d'abord un tableau.</b>"))
            return
        chemin = "/dbfs/FileStore/tableau_segmentation.csv"
        tableau.to_csv(chemin, index=False, sep=";", decimal=",")
        ip_display(HTML(f'<div style="color:{BLEU_FONCE};font-size:13px">Enregistré : {chemin}<br>'
                        f'Téléchargement : /files/tableau_segmentation.csv</div>'))


btn_gen.on_click(au_clic_generer)
btn_reset.on_click(au_clic_reset)
btn_csv.on_click(au_clic_csv)

# --- assemblage du panneau
panneau = w.VBox([
    titre,
    w.HBox([
        carte("1 · Segmentation", w.VBox([sel_seg, sl_bins])),
        carte("2 · Variables à agréger", sel_agg_v),
        carte("3 · Agrégations", sel_agg_f),
    ]),
    w.HBox([
        carte("4 · Ratios calculés (nom = expression)", w.VBox([txt_ratios, mode_ratio])),
        carte("5 · Modèle et cible", w.VBox([chk_model, txt_feats, dd_target])),
    ]),
    w.HBox([btn_gen, btn_reset, btn_csv], layout=w.Layout(margin="0 0 12px 0")),
    sortie,
])
panneau.layout = w.Layout(padding="12px", border=f"1px solid {BLEU_CLAIR}", border_radius="8px")

ip_display(panneau)

# COMMAND ----------

# MAGIC %md ## 5. Utilisation sans le panneau (optionnel)

# COMMAND ----------

# Le dernier tableau généré est disponible dans la variable `tableau`.
# On peut aussi appeler la fonction directement, par exemple dans une boucle :
#
# for var in ["age", "region"]:
#     p = {"seg_vars": [var], "n_bins": 10, "agg_vars": ["revenu"], "agg_funcs": ["moyenne"],
#          "ratios": {}, "ratio_mode": "ratio des sommes", "model": model,
#          "model_features": [], "target": "defaut"}
#     display(construire_tableau(df, p))

# Databricks notebook source
# MAGIC %md
# MAGIC # Outil d'analyse actuarielle
# MAGIC
# MAGIC **Pré-requis** : `df` (la base) et `model` (modèle entraîné) existent déjà dans le notebook.
# MAGIC
# MAGIC Le modèle et la variable cible sont **fixés dans la cellule 1** (constantes `TARGET` et
# MAGIC `FEATURES`) : ils s'appliquent automatiquement à toutes les analyses.
# MAGIC
# MAGIC Exécuter les cellules 1 à 5, puis tout se pilote depuis le panneau à onglets.

# COMMAND ----------

# MAGIC %md ## 1. Initialisation et paramètres fixes

# COMMAND ----------

import traceback

import numpy as np
import pandas as pd
import ipywidgets as w
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display as ip_display, HTML

# ---------------------------------------------------------------- paramètres fixes
TARGET = "resiliation"      # <-- variable cible (taux réel observé). À adapter une seule fois.
FEATURES = []               # <-- features du modèle ; [] = déduites du modèle, sinon de df

SEUIL_DISCRET = 20          # au-delà de ce nb de modalités, une variable numérique est binée
MIN_OBS_DEFAUT = 30         # effectif minimum d'une cellule de heatmap pour être interprétée

# ---------------------------------------------------------------- charte graphique
NAVY, BLEU, BLEU_CLAIR, BLEU_PALE = "#14476B", "#2E7DAF", "#DCEAF4", "#F5F9FC"
GRIS, TEXTE, ROUGE, VERT = "#E3E8ED", "#1F2A37", "#B3261E", "#1E7B5E"
ECHELLE = ["#F5F9FC", "#DCEAF4", "#9EC6E0", "#5A9BC6", "#2E7DAF", "#14476B"]

pio.templates["actuariat"] = go.layout.Template(layout=dict(
    font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=TEXTE),
    colorway=[NAVY, BLEU, "#7FB3D3", "#A67C00", "#1E7B5E", "#8C5B9E"],
    plot_bgcolor="white", paper_bgcolor="white",
    xaxis=dict(gridcolor=GRIS, zerolinecolor=GRIS),
    yaxis=dict(gridcolor=GRIS, zerolinecolor=GRIS),
    margin=dict(l=60, r=30, t=60, b=50),
    title=dict(font=dict(size=15, color=NAVY)),
))
pio.templates.default = "actuariat"

# ---------------------------------------------------------------- préparation de la base
if hasattr(df, "toPandas"):
    df = df.toPandas()


def features_modele(df, model, features, target):
    """1) features imposées, 2) sinon celles connues du modèle, 3) sinon numériques hors cible."""
    if features:
        return features
    connues = list(getattr(model, "feature_names_in_", []))
    if connues:
        return connues
    return [c for c in df.select_dtypes("number").columns if c != target]


# La probabilité prédite est calculée une seule fois, ligne par ligne, puis réutilisée partout.
FEATURES = features_modele(df, model, FEATURES, TARGET)
X = df[FEATURES]
df["proba_modele"] = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)

# Fonctions d'agrégation disponibles : ajouter une ligne ici pour en proposer une nouvelle
AGG_DISPONIBLES = {
    "moyenne": "mean", "mediane": "median", "somme": "sum", "min": "min", "max": "max",
    "nb_obs": "count", "ecart_type": "std", "p90": lambda s: s.quantile(0.90),
}

# COMMAND ----------

# MAGIC %md ## 2. Fonctions de calcul

# COMMAND ----------

def grouper(d, seg_cols):
    return d.groupby(seg_cols, observed=True)


# ---------- Segmentation ----------

def creer_segments(d, seg_vars, n_bins):
    """Ajoute une colonne 'seg_<var>' : classes qcut si la variable est continue,
    modalités telles quelles sinon. Les manquants forment la modalité 'Manquant'."""
    d = d.copy()
    seg_cols = []
    for var in seg_vars:
        col = f"seg_{var}"
        s = d[var]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > SEUIL_DISCRET:
            d[col] = pd.qcut(s, q=n_bins, duplicates="drop")
            d[col] = d[col].cat.add_categories("Manquant").fillna("Manquant")
        else:
            d[col] = s.astype("object").fillna("Manquant").astype(str)
        seg_cols.append(col)
    return d, seg_cols


# ---------- Blocs d'indicateurs (chacun renvoie un DataFrame indexé par segment) ----------

def calculer_effectifs(d, seg_cols):
    eff = grouper(d, seg_cols).size().rename("effectif").to_frame()
    eff["poids_pct"] = eff["effectif"] / eff["effectif"].sum()
    return eff


def calculer_cible_et_proba(d, seg_cols):
    """Taux réel observé, probabilité moyenne prédite et écart entre les deux."""
    g = grouper(d, seg_cols)
    res = pd.DataFrame({"taux_reel": g[TARGET].mean(), "proba_modele": g["proba_modele"].mean()})
    res["ecart_reel_modele"] = res["taux_reel"] - res["proba_modele"]
    res["indice_vs_global"] = 100 * res["taux_reel"] / d[TARGET].mean()   # base 100 = moyenne globale
    return res


def calculer_agregats(d, seg_cols, agg_vars, agg_funcs):
    """Une colonne '<variable>_<fonction>' par variable et par fonction choisie."""
    if not agg_vars or not agg_funcs:
        return pd.DataFrame()
    g = grouper(d, seg_cols)
    return pd.DataFrame({f"{v}_{f}": g[v].agg(AGG_DISPONIBLES[f]) for v in agg_vars for f in agg_funcs})


def calculer_indicateurs(d, seg_cols, p):
    """Assemble les blocs : une ligne par segment, une colonne par indicateur.
    Pour ajouter un indicateur : écrire une fonction renvoyant un DataFrame indexé
    par segment et l'ajouter à la liste ci-dessous."""
    blocs = [
        calculer_effectifs(d, seg_cols),
        calculer_cible_et_proba(d, seg_cols),
        calculer_agregats(d, seg_cols, p["agg_vars"], p["agg_funcs"]),
    ]
    return pd.concat([b for b in blocs if not b.empty], axis=1)


def ajouter_total(d, seg_cols, tableau, calcul):
    """Ajoute une ligne TOTAL calculée avec la même fonction sur un segment unique."""
    d = d.copy()
    d["_total"] = "TOTAL"
    total = calcul(d, ["_total"]).reset_index(drop=True)
    for c in seg_cols:
        total[c] = ""
    total[seg_cols[0]] = "TOTAL"
    return pd.concat([tableau, total], ignore_index=True)


def construire_tableau(d, p):
    """Tableau de segmentation : une ligne par modalité + ligne TOTAL."""
    d, seg_cols = creer_segments(d, p["seg_vars"], p["n_bins"])
    tableau = calculer_indicateurs(d, seg_cols, p).reset_index()
    tableau[seg_cols] = tableau[seg_cols].astype(str)
    tableau = ajouter_total(d, seg_cols, tableau, lambda x, s: calculer_indicateurs(x, s, p))
    return tableau.rename(columns={c: c[4:] for c in seg_cols})


# ---------- Bloc Ratio : Somme(numérateur) / Somme(dénominateur) ----------

def construire_ratio(d, num, den, seg_vars, n_bins):
    """Ratio actuariel = Somme(num) / Somme(den) par segment.

    - lignes dont num ou den est manquant : exclues des sommes (et comptées à part) ;
    - dénominateur nul ou négatif : ratio non calculé (NaN) plutôt qu'une valeur infinie ;
    - indice base 100 : positionne chaque segment par rapport au ratio global.
    """
    d = d.copy()
    valides = d[num].notna() & d[den].notna()
    d["_num"], d["_den"] = d[num].where(valides), d[den].where(valides)

    def calcul(x, seg_cols):
        g = grouper(x, seg_cols)
        r = pd.DataFrame({
            "effectif":       g.size(),
            "obs_exclues":    g["_num"].apply(lambda s: s.isna().sum()),
            f"somme_{num}":   g["_num"].sum(min_count=1),
            f"somme_{den}":   g["_den"].sum(min_count=1),
            "proba_modele":   g["proba_modele"].mean(),
            "taux_reel":      g[TARGET].mean(),
        })
        den_sur = r[f"somme_{den}"].where(r[f"somme_{den}"] > 0)      # évite la division par 0
        r["ratio"] = (r[f"somme_{num}"] / den_sur).replace([np.inf, -np.inf], np.nan)
        return r

    d, seg_cols = creer_segments(d, seg_vars, n_bins) if seg_vars else (d.assign(_ens="Ensemble"), ["_ens"])
    tableau = calcul(d, seg_cols).reset_index()
    tableau[seg_cols] = tableau[seg_cols].astype(str)
    tableau = ajouter_total(d, seg_cols, tableau, calcul)

    ratio_global = tableau["ratio"].iloc[-1]
    if pd.notna(ratio_global) and ratio_global != 0:
        tableau.insert(tableau.columns.get_loc("ratio") + 1,
                       "indice_base100", 100 * tableau["ratio"] / ratio_global)
    return tableau.rename(columns={c: c[4:] for c in seg_cols if c.startswith("seg_")})


# ---------- Heatmap : croisement de deux variables ----------

METRIQUES = {
    "Probabilité moyenne prédite": ("proba_modele", "mean"),
    "Taux réel observé":           (TARGET, "mean"),
    "Écart réel - prédit":         ("_ecart", "mean"),
    "Effectif":                    ("proba_modele", "size"),
}


def construire_heatmap(d, var1, var2, metrique, n_bins, min_obs):
    """Croise deux variables et renvoie (valeurs, effectifs) sous forme de tableaux croisés.
    Les cellules dont l'effectif est inférieur à min_obs sont masquées (non interprétables)."""
    d = d.copy()
    d["_ecart"] = d[TARGET] - d["proba_modele"]
    d, seg_cols = creer_segments(d, [var1, var2], n_bins)
    col, fonc = METRIQUES[metrique]

    effectifs = d.pivot_table(index=seg_cols[0], columns=seg_cols[1], values=col,
                              aggfunc="size", observed=True)
    if fonc == "size":
        valeurs = effectifs
    else:
        valeurs = d.pivot_table(index=seg_cols[0], columns=seg_cols[1], values=col,
                                aggfunc=fonc, observed=True)
        valeurs = valeurs.where(effectifs >= min_obs)                 # masque les cellules trop fines
    return valeurs, effectifs.fillna(0)

# COMMAND ----------

# MAGIC %md ## 3. Mise en forme (tableaux et graphiques)

# COMMAND ----------

FORMATS = {"effectif": "{:,.0f}", "obs_exclues": "{:,.0f}", "poids_pct": "{:.1%}",
           "taux_reel": "{:.2%}", "proba_modele": "{:.2%}", "ecart_reel_modele": "{:+.2%}",
           "indice_vs_global": "{:.0f}", "indice_base100": "{:.0f}", "ratio": "{:,.4f}"}


def tableau_html(tableau, seg_cols):
    """Rendu HTML : en-tête sombre figé, lignes alternées, TOTAL souligné,
    dégradé bleu sur les colonnes clés, écarts colorés."""
    num_cols = [c for c in tableau.columns if c not in seg_cols]
    styles = [
        {"selector": "", "props": [("border-collapse", "collapse"), ("width", "100%"),
                                   ("font-family", "Segoe UI, Arial, sans-serif"), ("font-size", "12.5px")]},
        {"selector": "thead th", "props": [("background-color", NAVY), ("color", "white"),
                                           ("padding", "9px 12px"), ("text-align", "right"),
                                           ("font-weight", "600"), ("position", "sticky"), ("top", "0")]},
        {"selector": "tbody td", "props": [("padding", "7px 12px"), ("text-align", "right"),
                                           ("border-bottom", f"1px solid {GRIS}")]},
        {"selector": "tbody tr:nth-child(even)", "props": [("background-color", BLEU_PALE)]},
        {"selector": "tbody tr:hover", "props": [("background-color", BLEU_CLAIR)]},
    ]

    def ligne_total(row):
        est_total = str(row.iloc[0]) == "TOTAL"
        return [f"background-color:{BLEU_CLAIR};font-weight:600;border-top:2px solid {NAVY};"
                if est_total else ""] * len(row)

    def couleur_ecart(v):
        if pd.isna(v):
            return ""
        return f"color:{ROUGE};font-weight:600;" if v > 0 else f"color:{VERT};font-weight:600;"

    st = (tableau.style.hide(axis="index").set_table_styles(styles)
          .set_properties(subset=seg_cols, **{"text-align": "left", "font-weight": "600", "color": NAVY})
          .format("{:,.4f}", subset=num_cols, na_rep="–")
          .format({c: f for c, f in FORMATS.items() if c in tableau.columns}, na_rep="–")
          .apply(ligne_total, axis=1))
    for col in ["taux_reel", "proba_modele", "ratio"]:
        if col in tableau:
            st = st.background_gradient(cmap="Blues", subset=[col])
    if "ecart_reel_modele" in tableau:
        st = st.map(couleur_ecart, subset=["ecart_reel_modele"])
    return f'<div style="overflow:auto;max-height:520px;border:1px solid {GRIS};border-radius:6px;">{st.to_html()}</div>'


def figure_heatmap(valeurs, effectifs, metrique, afficher_n):
    """Heatmap annotée : valeur de la métrique et, si demandé, effectif de la cellule."""
    pct = metrique != "Effectif"
    fmt = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:,.0f}")
    texte = [[("" if pd.isna(v) else fmt(v)) + (f"<br><span style='font-size:9px'>n={int(n):,}</span>"
              if afficher_n and metrique != "Effectif" else "")
              for v, n in zip(lv, ln)] for lv, ln in zip(valeurs.values, effectifs.values)]

    fig = go.Figure(go.Heatmap(
        z=valeurs.values, x=[str(c) for c in valeurs.columns], y=[str(i) for i in valeurs.index],
        text=texte, texttemplate="%{text}", textfont={"size": 11},
        colorscale=ECHELLE, colorbar=dict(title=dict(text=metrique.split()[0], side="right"),
                                          tickformat=".1%" if pct else ","),
        hovertemplate="%{y} × %{x}<br>" + metrique + " : %{z:.4f}<extra></extra>"))
    fig.update_layout(title=f"{metrique} — croisement des deux variables",
                      xaxis_title=valeurs.columns.name, yaxis_title=valeurs.index.name,
                      height=110 + 46 * len(valeurs), xaxis=dict(side="top"))
    return fig


def afficher_figure(fig):
    """Affiche une figure Plotly (repli en HTML si le rendu natif n'est pas disponible)."""
    try:
        ip_display(fig)
    except Exception:
        ip_display(HTML(fig.to_html(full_html=False, include_plotlyjs="cdn")))


def carte(titre, contenu):
    """Encadré blanc avec bandeau de titre, utilisé pour chaque groupe de contrôles."""
    entete = w.HTML(f'<div style="color:{NAVY};font-weight:600;font-size:12px;letter-spacing:.4px;'
                    f'text-transform:uppercase;border-bottom:2px solid {BLEU_CLAIR};padding-bottom:5px;'
                    f'margin-bottom:8px;font-family:Segoe UI,Arial,sans-serif;">{titre}</div>')
    boite = w.VBox([entete, contenu])
    boite.layout = w.Layout(border=f"1px solid {GRIS}", padding="12px", margin="0 10px 10px 0",
                            border_radius="6px")
    return boite


def styler_bouton(b, principal=False):
    b.style.button_color = NAVY if principal else BLEU_CLAIR
    b.style.font_weight = "600"
    b.layout = w.Layout(width="190px", height="36px", margin="4px 8px 12px 0")
    return b

# COMMAND ----------

# MAGIC %md ## 4. Panneau d'analyse

# COMMAND ----------

colonnes = [c for c in df.columns if c != "proba_modele"]
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
seg_options = colonnes + ["proba_modele"]          # segmenter par score prédit est souvent utile

liste = w.Layout(width="240px", height="150px")
champ = w.Layout(width="240px")

# ---------------------------------------------------------------- bandeau et indicateurs globaux
def bloc_kpi(libelle, valeur, accent=BLEU):
    return (f'<div style="flex:1;background:white;border:1px solid {GRIS};border-left:4px solid {accent};'
            f'border-radius:5px;padding:10px 14px;margin-right:10px;">'
            f'<div style="font-size:11px;color:#5B6B7B;text-transform:uppercase;letter-spacing:.5px;">{libelle}</div>'
            f'<div style="font-size:20px;font-weight:600;color:{TEXTE};">{valeur}</div></div>')


ecart_global = df[TARGET].mean() - df["proba_modele"].mean()
entete = w.HTML(
    f'<div style="font-family:Segoe UI,Arial,sans-serif;">'
    f'<div style="background:{NAVY};color:white;padding:16px 20px;border-radius:6px 6px 0 0;">'
    f'<div style="font-size:19px;font-weight:600;">Outil d\'analyse actuarielle</div>'
    f'<div style="font-size:12px;opacity:.8;margin-top:2px;">Cible : {TARGET} · Modèle : '
    f'{type(model).__name__} · {len(FEATURES)} variables explicatives</div></div>'
    f'<div style="display:flex;padding:12px 0 4px 0;">'
    + bloc_kpi("Observations", f"{len(df):,}".replace(",", " "))
    + bloc_kpi("Taux réel observé", f"{df[TARGET].mean():.2%}")
    + bloc_kpi("Probabilité moyenne prédite", f"{df['proba_modele'].mean():.2%}")
    + bloc_kpi("Écart réel - prédit", f"{ecart_global:+.2%}", ROUGE if abs(ecart_global) > .01 else VERT)
    + '</div></div>')

# ---------------------------------------------------------------- onglet 1 : segmentation
sel_seg = w.SelectMultiple(options=seg_options, value=(seg_options[0],), layout=liste)
sl_bins = w.IntSlider(value=5, min=2, max=20, description="Bins", continuous_update=False,
                      layout=champ, style={"description_width": "45px"})
sel_agg_v = w.SelectMultiple(options=num_cols, layout=liste)
sel_agg_f = w.SelectMultiple(options=list(AGG_DISPONIBLES), value=("moyenne",), layout=liste)
btn_tab = styler_bouton(w.Button(description="Générer le tableau", icon="table"), True)
btn_csv = styler_bouton(w.Button(description="Exporter en CSV", icon="download"))
out_tab = w.Output()

onglet_seg = w.VBox([
    w.HBox([carte("1 · Variables de segmentation", w.VBox([sel_seg, sl_bins])),
            carte("2 · Variables à agréger", sel_agg_v),
            carte("3 · Fonctions d'agrégation", sel_agg_f)]),
    w.HBox([btn_tab, btn_csv]), out_tab])

# ---------------------------------------------------------------- onglet 2 : ratios
dd_num = w.Dropdown(options=num_cols, description="Numérateur", layout=champ,
                    style={"description_width": "95px"})
dd_den = w.Dropdown(options=num_cols, index=min(1, len(num_cols) - 1), description="Dénominateur",
                    layout=champ, style={"description_width": "95px"})
sel_seg_r = w.SelectMultiple(options=seg_options, layout=liste)
sl_bins_r = w.IntSlider(value=5, min=2, max=20, description="Bins", continuous_update=False,
                        layout=champ, style={"description_width": "45px"})
btn_ratio = styler_bouton(w.Button(description="Calculer le ratio", icon="calculator"), True)
out_ratio = w.Output()

onglet_ratio = w.VBox([
    w.HBox([carte("Ratio = Somme(numérateur) / Somme(dénominateur)",
                  w.VBox([dd_num, dd_den,
                          w.HTML(f'<div style="font-size:11px;color:#5B6B7B;margin-top:6px;">'
                                 f'Dénominateur nul et lignes incomplètes exclus du calcul.</div>')])),
            carte("Segmentation (optionnelle)", w.VBox([sel_seg_r, sl_bins_r]))]),
    w.HBox([btn_ratio]), out_ratio])

# ---------------------------------------------------------------- onglet 3 : visualisation
GRAPHIQUES = ["Histogramme", "Distribution selon la cible", "Boxplot par modalité",
              "Effectifs par modalité", "Prédit vs observé par classe",
              "Distribution des probabilités prédites"]
dd_var_g = w.Dropdown(options=seg_options, description="Variable", layout=champ,
                      style={"description_width": "70px"})
dd_type_g = w.Dropdown(options=GRAPHIQUES, description="Graphique", layout=w.Layout(width="300px"),
                       style={"description_width": "80px"})
sl_bins_g = w.IntSlider(value=10, min=2, max=30, description="Bins", continuous_update=False,
                        layout=champ, style={"description_width": "45px"})
btn_graph = styler_bouton(w.Button(description="Afficher le graphique", icon="bar-chart"), True)
out_graph = w.Output()

onglet_graph = w.VBox([
    w.HBox([carte("Variable analysée", w.VBox([dd_var_g, sl_bins_g])),
            carte("Type de visualisation", dd_type_g)]),
    w.HBox([btn_graph]), out_graph])

# ---------------------------------------------------------------- onglet 4 : heatmap
dd_h1 = w.Dropdown(options=seg_options, description="Variable 1", layout=champ,
                   style={"description_width": "85px"})
dd_h2 = w.Dropdown(options=seg_options, index=min(1, len(seg_options) - 1), description="Variable 2",
                   layout=champ, style={"description_width": "85px"})
dd_metrique = w.Dropdown(options=list(METRIQUES), description="Métrique", layout=w.Layout(width="330px"),
                         style={"description_width": "85px"})
sl_bins_h = w.IntSlider(value=5, min=2, max=12, description="Bins", continuous_update=False,
                        layout=champ, style={"description_width": "85px"})
sl_min_obs = w.IntSlider(value=MIN_OBS_DEFAUT, min=0, max=500, step=10, description="Obs. min",
                         continuous_update=False, layout=champ, style={"description_width": "85px"})
chk_n = w.Checkbox(value=True, description="Afficher les effectifs", indent=False)
btn_heat = styler_bouton(w.Button(description="Afficher la heatmap", icon="th"), True)
out_heat = w.Output()

onglet_heat = w.VBox([
    w.HBox([carte("Croisement", w.VBox([dd_h1, dd_h2, sl_bins_h])),
            carte("Métrique et lisibilité", w.VBox([dd_metrique, sl_min_obs, chk_n,
                  w.HTML('<div style="font-size:11px;color:#5B6B7B;margin-top:6px;">'
                         'Les cellules sous le seuil d\'effectif sont laissées vides.</div>')]))]),
    w.HBox([btn_heat]), out_heat])

# COMMAND ----------

# MAGIC %md ## 5. Actions et affichage du panneau

# COMMAND ----------

def message(txt, couleur=NAVY):
    ip_display(HTML(f'<div style="font-family:Segoe UI,Arial;font-size:13px;color:{couleur};'
                    f'padding:6px 0;">{txt}</div>'))


def executer(sortie, fonction):
    """Exécute une action dans une zone de sortie en affichant proprement les erreurs."""
    sortie.clear_output()
    with sortie:
        try:
            fonction()
        except Exception:
            ip_display(HTML(f'<pre style="color:{ROUGE};font-size:11.5px;white-space:pre-wrap">'
                            f'{traceback.format_exc()}</pre>'))


# ---------------------------------------------------------------- onglet segmentation
def action_tableau():
    global tableau
    if not sel_seg.value:
        return message("Sélectionner au moins une variable de segmentation.")
    p = {"seg_vars": list(sel_seg.value), "n_bins": sl_bins.value,
         "agg_vars": list(sel_agg_v.value), "agg_funcs": list(sel_agg_f.value)}
    tableau = construire_tableau(df, p)
    ip_display(HTML(tableau_html(tableau, p["seg_vars"])))


def action_csv():
    if "tableau" not in globals():
        return message("Générer d'abord un tableau.")
    chemin = "/dbfs/FileStore/tableau_segmentation.csv"
    tableau.to_csv(chemin, index=False, sep=";", decimal=",")
    message(f"Enregistré : {chemin} — téléchargement via /files/tableau_segmentation.csv")


# ---------------------------------------------------------------- onglet ratio
def action_ratio():
    global tableau_ratio
    if dd_num.value == dd_den.value:
        return message("Numérateur et dénominateur doivent être différents.")
    seg = list(sel_seg_r.value)
    tableau_ratio = construire_ratio(df, dd_num.value, dd_den.value, seg, sl_bins_r.value)
    seg_cols = seg if seg else ["_ens"]
    exclues = int(tableau_ratio["obs_exclues"].iloc[-1])
    ip_display(HTML(tableau_html(tableau_ratio, seg_cols)))
    message(f"Ratio {dd_num.value} / {dd_den.value} · indice base 100 = ratio global"
            + (f" · {exclues:,} observation(s) exclue(s) pour valeurs manquantes".replace(",", " ")
               if exclues else ""), "#5B6B7B")


# ---------------------------------------------------------------- onglet visualisation
def action_graphique():
    var, type_g, nb = dd_var_g.value, dd_type_g.value, sl_bins_g.value
    d, (seg,) = creer_segments(df, [var], nb)
    d["_classe"] = d[seg].astype(str)
    est_num = pd.api.types.is_numeric_dtype(df[var])

    if type_g == "Histogramme":
        fig = px.histogram(df, x=var, nbins=nb if est_num else None,
                           title=f"Distribution de {var}")
        fig.update_traces(marker_line_color="white", marker_line_width=1)

    elif type_g == "Distribution selon la cible":
        d["_cible"] = np.where(d[TARGET] == 1, "Résiliés", "Non résiliés")
        fig = px.histogram(d, x=var, color="_cible", barmode="overlay", opacity=.7,
                           nbins=nb if est_num else None, histnorm="percent",
                           title=f"Distribution de {var} selon la cible ({TARGET})")
        fig.update_layout(legend_title_text="")

    elif type_g == "Boxplot par modalité":
        if not est_num:
            return message("Le boxplot nécessite une variable numérique.")
        fig = px.box(d, x="_classe", y=var, points=False,
                     title=f"Dispersion de {var} par classe")
        fig.update_xaxes(title=var, tickangle=-30)

    elif type_g == "Effectifs par modalité":
        eff = d["_classe"].value_counts().sort_index()
        fig = px.bar(x=eff.index.astype(str), y=eff.values, text=eff.values,
                     title=f"Effectifs par modalité de {var}")
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(xaxis_title=var, yaxis_title="Effectif", xaxis_tickangle=-30)

    elif type_g == "Prédit vs observé par classe":
        g = d.groupby("_classe", observed=True).agg(
            reel=(TARGET, "mean"), predit=("proba_modele", "mean"), n=("proba_modele", "size")).sort_index()
        fig = go.Figure()
        fig.add_bar(x=g.index.astype(str), y=g["reel"], name="Taux réel observé",
                    marker_color=BLEU, customdata=g["n"],
                    hovertemplate="%{x}<br>Réel : %{y:.2%}<br>n = %{customdata:,}<extra></extra>")
        fig.add_scatter(x=g.index.astype(str), y=g["predit"], name="Probabilité prédite",
                        mode="lines+markers", line=dict(color=NAVY, width=3),
                        hovertemplate="%{x}<br>Prédit : %{y:.2%}<extra></extra>")
        fig.update_layout(title=f"Calibration du modèle par classe de {var}",
                          yaxis_tickformat=".1%", yaxis_title="Taux", xaxis_title=var,
                          xaxis_tickangle=-30, legend=dict(orientation="h", y=1.1))

    else:   # Distribution des probabilités prédites
        fig = px.histogram(d, x="proba_modele", color="_classe", nbins=40, barmode="overlay",
                           opacity=.65, histnorm="percent",
                           title=f"Distribution des probabilités prédites par classe de {var}")
        fig.update_layout(xaxis_tickformat=".0%", xaxis_title="Probabilité prédite",
                          legend_title_text=var)

    fig.update_layout(height=460)
    afficher_figure(fig)


# ---------------------------------------------------------------- onglet heatmap
def action_heatmap():
    global heat_valeurs, heat_effectifs
    if dd_h1.value == dd_h2.value:
        return message("Choisir deux variables différentes.")
    heat_valeurs, heat_effectifs = construire_heatmap(
        df, dd_h1.value, dd_h2.value, dd_metrique.value, sl_bins_h.value, sl_min_obs.value)
    afficher_figure(figure_heatmap(heat_valeurs, heat_effectifs, dd_metrique.value, chk_n.value))
    masquees = int(heat_valeurs.isna().sum().sum())
    if masquees:
        message(f"{masquees} cellule(s) masquée(s) : moins de {sl_min_obs.value} observations.", "#5B6B7B")


btn_tab.on_click(lambda _: executer(out_tab, action_tableau))
btn_csv.on_click(lambda _: executer(out_tab, action_csv))
btn_ratio.on_click(lambda _: executer(out_ratio, action_ratio))
btn_graph.on_click(lambda _: executer(out_graph, action_graphique))
btn_heat.on_click(lambda _: executer(out_heat, action_heatmap))

onglets = w.Tab(children=[onglet_seg, onglet_ratio, onglet_graph, onglet_heat])
for i, t in enumerate(["Segmentation", "Ratios", "Visualisation", "Heatmap"]):
    onglets.set_title(i, t)

panneau = w.VBox([entete, onglets])
panneau.layout = w.Layout(padding="14px", border=f"1px solid {GRIS}", border_radius="8px")
ip_display(panneau)

# COMMAND ----------

# MAGIC %md ## 6. Utilisation directe des fonctions (optionnel)

# COMMAND ----------

# Les derniers résultats restent disponibles : `tableau`, `tableau_ratio`, `heat_valeurs`.
#
# construire_tableau(df, {"seg_vars": ["age"], "n_bins": 10,
#                         "agg_vars": ["prime"], "agg_funcs": ["moyenne", "somme"]})
# construire_ratio(df, "sinistres", "primes", ["region"], 5)
# construire_heatmap(df, "age", "region", "Probabilité moyenne prédite", 5, 30)

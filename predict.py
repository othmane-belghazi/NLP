# Databricks notebook source
# MAGIC %md
# MAGIC # Outil d'analyse actuarielle — volumétrie 2 M de lignes
# MAGIC
# MAGIC **Principe** : Python pré-agrège la base en *cubes* compacts (quelques centaines de Ko),
# MAGIC le navigateur ne reçoit que ces cubes. Aucune ligne brute n'est transmise, ce qui évite
# MAGIC l'erreur *Command result size exceeds limit* et rend l'interface instantanée.
# MAGIC
# MAGIC **Conséquences de ce choix**
# MAGIC - segmentation sur **1 ou 2 variables** (croisements pré-calculés) ;
# MAGIC - variables continues découpées en 20 classes de quantiles, regroupables dans l'interface
# MAGIC   en 2, 4, 5, 10 ou 20 classes ;
# MAGIC - si `TARGET` est absente de la base, tous les indicateurs qui en dépendent
# MAGIC   (taux réel, écart, calibration) sont simplement retirés de l'outil.
# MAGIC
# MAGIC Exécuter les cellules 1 à 5. Le pré-calcul dure quelques dizaines de secondes sur 2 M de lignes.

# COMMAND ----------

# MAGIC %md ## 1. Paramètres

# COMMAND ----------

import json
import time

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- à adapter
TARGET = "resiliation"       # variable cible ; si absente de df, les indicateurs liés sont désactivés
FEATURES = []                # features du modèle ; [] = déduites du modèle

VARS_SEGMENTATION = []       # variables de segmentation / croisement ; [] = détection automatique
VARS_MESURE = []             # variables numériques agrégées et utilisées dans les ratios ; [] = auto

# ---------------------------------------------------------------- réglages de volumétrie
MAX_VARS = 10                # nb max de variables de segmentation (le poids du cube croît en V²)
MAX_MESURES = 6              # nb max de variables de mesure
MAX_MODALITES = 30           # variable catégorielle plus riche que cela -> écartée
N_CLASSES_FINES = 20         # classes de quantiles des variables continues (regroupables ensuite)
INCLURE_QUANTILES = True     # quartiles par classe (boxplots) ; coûte ~0,5 s par variable
LIMITE_MO = 6                # garde-fou sur la taille envoyée au navigateur

if hasattr(df, "toPandas"):
    print("Conversion Spark -> pandas (sélectionner en amont les colonnes utiles si la base est large)")
    df = df.toPandas()

A_CIBLE = TARGET in df.columns
if not A_CIBLE:
    print(f"[info] La cible '{TARGET}' est absente de la base : taux réel observé, écart "
          f"réel-prédit et graphiques de calibration seront désactivés.")

# COMMAND ----------

# MAGIC %md ## 2. Probabilité prédite

# COMMAND ----------

A_PROBA = False
try:
    if FEATURES:
        features = list(FEATURES)
    else:
        features = list(getattr(model, "feature_names_in_", [])) or \
                   [c for c in df.select_dtypes("number").columns if c != TARGET]
    absentes = [c for c in features if c not in df.columns]
    if absentes:
        raise ValueError(f"features absentes de df : {absentes}")

    X = df[features]
    df["proba_modele"] = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") \
        else model.predict(X)
    FEATURES, A_PROBA = features, True
    print(f"Probabilité prédite calculée sur {len(FEATURES)} features · "
          f"moyenne = {df['proba_modele'].mean():.2%}")

except NameError:
    print("[info] Aucun modèle nommé 'model' : les indicateurs de probabilité prédite sont désactivés.")
except Exception as e:
    nb_nan = int(df[features].isna().any(axis=1).sum()) if 'features' in dir() else 0
    print(f"[attention] Probabilité prédite non calculée ({e}).")
    if nb_nan:
        print(f"            {nb_nan} ligne(s) ont des valeurs manquantes parmi les features. "
              f"Utiliser un pipeline avec imputation.")

# COMMAND ----------

# MAGIC %md ## 3. Découpage des variables et pré-agrégation en cubes

# COMMAND ----------

def arrondir(x):
    """Arrondi à 6 chiffres significatifs : réduit fortement le poids du JSON."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return float(f"{float(x):.6g}")


def liste(a):
    return [arrondir(v) for v in np.asarray(a, dtype=float)]


# ---------------------------------------------------------------- choix des variables
def choisir_variables(d, target, vars_seg, vars_mes):
    """Détermine les variables de segmentation et de mesure, automatiquement si non imposées."""
    numeriques = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]

    if not vars_mes:
        vars_mes = [c for c in numeriques if c not in (target, "proba_modele")][:MAX_MESURES]
    if not vars_seg:
        categorielles = [c for c in d.columns
                         if c not in numeriques and d[c].nunique(dropna=True) <= MAX_MODALITES]
        vars_seg = (categorielles + [c for c in numeriques if c not in (target, "proba_modele")])[:MAX_VARS]

    # la cible et le score sont ajoutés comme variables de croisement : ils alimentent
    # les graphiques « distribution selon la cible » et « probabilités prédites par classe »
    for c in (target, "proba_modele"):
        if c in d.columns and c not in vars_seg:
            vars_seg.append(c)

    mesures = [c for c in vars_mes if c in d.columns]
    for c in (target, "proba_modele"):                 # taux réel et proba moyenne = des mesures
        if c in d.columns and c not in mesures:
            mesures.append(c)
    return vars_seg[:MAX_VARS + 2], mesures


VARS_SEGMENTATION, MESURES = choisir_variables(df, TARGET, list(VARS_SEGMENTATION), list(VARS_MESURE))
print("Segmentation / croisement :", ", ".join(VARS_SEGMENTATION))
print("Mesures                   :", ", ".join(MESURES))


# ---------------------------------------------------------------- découpage en classes
def decouper(serie, n_fines=N_CLASSES_FINES, max_modalites=MAX_MODALITES):
    """Renvoie (codes int16, métadonnées). La dernière classe est 'Manquant' si nécessaire.

    - catégorielle          -> une classe par modalité
    - numérique peu variée  -> une classe par valeur
    - numérique continue    -> 20 classes de quantiles, regroupables ensuite dans l'interface
    """
    if pd.api.types.is_numeric_dtype(serie):
        distinctes = serie.dropna().unique()
        if len(distinctes) <= n_fines:
            valeurs = np.sort(distinctes)
            codes = pd.Categorical(serie, categories=valeurs).codes.astype(np.int16)
            meta = {"type": "disc", "labels": [f"{v:g}" for v in valeurs]}
        else:
            classes, bornes = pd.qcut(serie, n_fines, duplicates="drop", labels=False, retbins=True)
            codes = np.where(classes.isna(), -1, classes).astype(np.int16)
            meta = {"type": "cont", "edges": liste(bornes),
                    "labels": [f"c{i + 1}" for i in range(len(bornes) - 1)]}
    else:
        s = serie.astype("object")
        modalites = sorted(map(str, s.dropna().unique()))[:max_modalites]
        codes = pd.Categorical(s.astype(str).where(s.notna()), categories=modalites).codes.astype(np.int16)
        meta = {"type": "cat", "labels": modalites}

    manquant = bool((codes < 0).any())
    if manquant:                                        # les manquants forment la dernière classe
        codes = np.where(codes < 0, len(meta["labels"]), codes).astype(np.int16)
        meta["labels"] = meta["labels"] + ["Manquant"]
    meta["manquant"] = manquant
    meta["nfin"] = len(meta["labels"]) - (1 if manquant else 0)
    return codes, meta


t0 = time.time()
CODES, VARS = {}, {}
for v in VARS_SEGMENTATION:
    CODES[v], VARS[v] = decouper(df[v])
print(f"Découpage : {time.time() - t0:.1f} s")


# ---------------------------------------------------------------- cube 1 : une variable
def cube_une_variable(d, codes, meta, mesures, avec_quantiles):
    """Par classe : effectif et, pour chaque mesure, somme / nb non nuls / min / max /
    somme des carrés (écart-type) et quartiles (boxplots)."""
    k = len(meta["labels"])
    g = pd.Series(codes, index=d.index)
    res = {"n": [int(x) for x in np.bincount(codes, minlength=k)], "mes": {}}

    for m in mesures:
        v = d[m].astype(float)
        nn = v.notna().to_numpy()
        vals = np.nan_to_num(v.to_numpy())
        bloc = {
            "s":  liste(np.bincount(codes, weights=vals, minlength=k)),
            "c":  [int(x) for x in np.bincount(codes, weights=nn, minlength=k)],
            "sq": liste(np.bincount(codes, weights=vals ** 2, minlength=k)),
        }
        agg = v.groupby(g, observed=True).agg(["min", "max"]).reindex(range(k))
        bloc["mn"], bloc["mx"] = liste(agg["min"]), liste(agg["max"])
        if avec_quantiles:
            q = v.groupby(g, observed=True).quantile([.25, .5, .75]).unstack().reindex(range(k))
            bloc["q"] = [[arrondir(x) for x in ligne] for ligne in q.to_numpy()]
        res["mes"][m] = bloc
    return res


t0 = time.time()
CUBE1 = {v: cube_une_variable(df, CODES[v], VARS[v], MESURES, INCLURE_QUANTILES)
         for v in VARS_SEGMENTATION}
print(f"Cube 1 variable  : {time.time() - t0:.1f} s")


# ---------------------------------------------------------------- cube 2 : croisements
def cube_deux_variables(d, c1, k1, c2, k2, mesures, mesures_nulles):
    """Croisement de deux variables : effectif et somme de chaque mesure par cellule.
    Stockage creux (cellules vides omises) et vectorisé par np.bincount."""
    cle = c1.astype(np.int32) * k2 + c2.astype(np.int32)
    n = np.bincount(cle, minlength=k1 * k2)
    plein = np.nonzero(n)[0]                            # seules les cellules non vides sont envoyées

    res = {"i": (plein // k2).tolist(), "j": (plein % k2).tolist(),
           "n": n[plein].astype(int).tolist(), "s": {}, "c": {}}
    for m in mesures:
        v = d[m].to_numpy(dtype=float)
        res["s"][m] = liste(np.bincount(cle, weights=np.nan_to_num(v), minlength=k1 * k2)[plein])
        if m in mesures_nulles:                         # nb de valeurs non nulles, si la mesure en a
            res["c"][m] = np.bincount(cle, weights=~np.isnan(v),
                                      minlength=k1 * k2)[plein].astype(int).tolist()
    return res


MESURES_NULLES = {m for m in MESURES if df[m].isna().any()}
t0 = time.time()
CUBE2 = {}
for a in range(len(VARS_SEGMENTATION)):
    for b in range(a + 1, len(VARS_SEGMENTATION)):
        v1, v2 = VARS_SEGMENTATION[a], VARS_SEGMENTATION[b]
        CUBE2[f"{v1}|{v2}"] = cube_deux_variables(
            df, CODES[v1], len(VARS[v1]["labels"]), CODES[v2], len(VARS[v2]["labels"]),
            MESURES, MESURES_NULLES)
print(f"Cube 2 variables : {time.time() - t0:.1f} s  ({len(CUBE2)} croisements)")


# ---------------------------------------------------------------- histogrammes à pas constant
def histogramme(serie, cible, n_barres=40):
    """Histogramme à pas constant, borné aux percentiles 0,5 % et 99,5 % : les valeurs
    extrêmes sont regroupées dans les barres d'extrémité plutôt que d'écraser l'échelle."""
    v = serie.dropna().to_numpy(dtype=float)
    if v.size == 0:
        return None
    bas, haut = np.percentile(v, [0.5, 99.5])
    if haut <= bas:
        bas, haut = float(v.min()), float(v.max()) + 1e-9
    bornes = np.linspace(bas, haut, n_barres + 1)
    coupe = np.clip(v, bas, haut)
    res = {"edges": liste(bornes), "n": np.histogram(coupe, bins=bornes)[0].tolist(),
           "extremes": [int((v < bas).sum()), int((v > haut).sum())],
           "manquants": int(serie.isna().sum())}
    if cible is not None:
        masque = cible.to_numpy()[serie.notna().to_numpy()] == 1
        res["n1"] = np.histogram(coupe[masque], bins=bornes)[0].tolist()
        res["n0"] = np.histogram(coupe[~masque], bins=bornes)[0].tolist()
    return res


t0 = time.time()
numeriques = [v for v in set(VARS_SEGMENTATION) | set(MESURES)
              if pd.api.types.is_numeric_dtype(df[v])]
HIST = {v: histogramme(df[v], df[TARGET] if A_CIBLE else None) for v in numeriques}
HIST = {k: v for k, v in HIST.items() if v}
print(f"Histogrammes     : {time.time() - t0:.1f} s")

# COMMAND ----------

# MAGIC %md ## 4. Assemblage du paquet envoyé au navigateur

# COMMAND ----------

donnees = {
    "n": int(len(df)),
    "vars": VARS, "cube1": CUBE1, "cube2": CUBE2, "hist": HIST,
    "mesures": MESURES,
    "cible": TARGET if A_CIBLE else None,
    "proba": "proba_modele" if A_PROBA else None,
    "quantiles": bool(INCLURE_QUANTILES),
    "nfin": N_CLASSES_FINES,
    "modele": type(model).__name__ if A_PROBA else None,
    "features": list(FEATURES) if A_PROBA else [],
}
payload = json.dumps(donnees, separators=(",", ":"), allow_nan=False)
taille = len(payload) / 1e6
print(f"Paquet : {taille:.2f} Mo · {len(VARS_SEGMENTATION)} variables · "
      f"{len(MESURES)} mesures · {len(CUBE2)} croisements")

if taille > LIMITE_MO:
    raise ValueError(
        f"Paquet trop volumineux ({taille:.1f} Mo > {LIMITE_MO} Mo) : le rendu risque l'erreur "
        f"« Command result size exceeds limit ».\nRéduire MAX_VARS (actuellement {MAX_VARS}), "
        f"MAX_MESURES ({MAX_MESURES}) ou MAX_MODALITES ({MAX_MODALITES}), "
        f"ou passer INCLURE_QUANTILES à False.")

# libère la mémoire des objets intermédiaires avant l'affichage
del CODES

# COMMAND ----------

# MAGIC %md ## 5. Application

# COMMAND ----------

APP = r"""
<div id="app">
<style>
  #app{--navy:#14476B;--bleu:#2E7DAF;--clair:#DCEAF4;--pale:#F5F9FC;--gris:#E3E8ED;
       --texte:#1F2A37;--doux:#5B6B7B;--rouge:#B3261E;--vert:#1E7B5E;
       font-family:'Segoe UI',Arial,sans-serif;color:var(--texte);font-size:13px;background:#fff;
       padding:14px;border:1px solid var(--gris);border-radius:8px;box-sizing:border-box}
  #app *{box-sizing:border-box}
  .hdr{background:var(--navy);color:#fff;padding:16px 20px;border-radius:6px}
  .hdr h1{margin:0;font-size:19px;font-weight:600}
  .hdr .sub{font-size:12px;opacity:.82;margin-top:3px}
  .kpis{display:flex;gap:10px;margin:12px 0}
  .kpi{flex:1;background:#fff;border:1px solid var(--gris);border-left:4px solid var(--bleu);
       border-radius:5px;padding:10px 14px}
  .kpi .lab{font-size:11px;color:var(--doux);text-transform:uppercase;letter-spacing:.5px}
  .kpi .val{font-size:20px;font-weight:600;margin-top:2px}
  .tabs{display:flex;gap:4px;border-bottom:2px solid var(--clair);margin-bottom:14px}
  .tab{padding:9px 22px;cursor:pointer;font-weight:600;font-size:13px;color:var(--doux);
       border:1px solid transparent;border-bottom:none;border-radius:5px 5px 0 0;margin-bottom:-2px}
  .tab:hover{background:var(--pale)}
  .tab.on{color:var(--navy);background:#fff;border-color:var(--clair);border-bottom:2px solid #fff}
  .stabs{display:flex;gap:6px;margin-bottom:12px}
  .stab{padding:6px 14px;font-size:12px;font-weight:600;color:var(--doux);background:var(--pale);
        border:1px solid var(--gris);border-radius:16px;cursor:pointer}
  .stab.on{background:var(--navy);color:#fff;border-color:var(--navy)}
  .panel,.spanel{display:none} .panel.on,.spanel.on{display:block}
  .row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}
  .card{border:1px solid var(--gris);border-radius:6px;padding:12px;min-width:225px}
  .card h3{margin:0 0 8px;font-size:11.5px;font-weight:600;color:var(--navy);
           text-transform:uppercase;letter-spacing:.4px;border-bottom:2px solid var(--clair);padding-bottom:5px}
  label{display:block;font-size:11.5px;color:var(--doux);margin:8px 0 3px}
  select,input[type=number]{width:100%;padding:5px 7px;border:1px solid var(--gris);border-radius:4px;
       font-family:inherit;font-size:12.5px;color:var(--texte);background:#fff}
  select[multiple]{height:132px}
  select:focus,input:focus{outline:none;border-color:var(--bleu)}
  .chk{display:flex;align-items:center;gap:6px;margin-top:9px;font-size:12px}
  .mini{padding:5px 12px;font-size:12px;margin-top:8px}
  .chips{margin-top:8px;display:flex;flex-wrap:wrap;gap:5px}
  .chip{background:var(--clair);color:var(--navy);border-radius:14px;padding:3px 9px;font-size:11.5px;
        font-weight:600;display:flex;align-items:center;gap:6px}
  .chip span{cursor:pointer;color:var(--rouge);font-weight:700}
  .btns{margin:12px 0}
  button{padding:9px 18px;border:none;border-radius:5px;font-family:inherit;font-size:13px;
         font-weight:600;cursor:pointer;margin-right:8px;background:var(--clair);color:var(--navy)}
  button.p{background:var(--navy);color:#fff}
  button:hover{opacity:.88}
  .note{font-size:11.5px;color:var(--doux);margin-top:8px;line-height:1.5}
  .msg{padding:8px 0;font-size:13px;color:var(--navy)}
  .err{color:var(--rouge);font-size:12px;white-space:pre-wrap}
  .tw{overflow:auto;max-height:520px;border:1px solid var(--gris);border-radius:6px;margin-top:4px}
  table{border-collapse:collapse;width:100%;font-size:12.5px}
  thead th{background:var(--navy);color:#fff;padding:9px 12px;text-align:right;font-weight:600;
           position:sticky;top:0;cursor:pointer;white-space:nowrap}
  thead th:hover{background:#0F3A57}
  thead th.seg{text-align:left}
  tbody td{padding:7px 12px;text-align:right;border-bottom:1px solid var(--gris);white-space:nowrap}
  tbody td.seg{text-align:left;font-weight:600;color:var(--navy)}
  tbody tr:nth-child(even){background:var(--pale)}
  tbody tr:hover{background:var(--clair)}
  tbody tr.tot{background:var(--clair);font-weight:600;border-top:2px solid var(--navy)}
  .chart{margin-top:10px;min-height:450px}
</style>

<div class="hdr"><h1>Outil d'analyse actuarielle</h1><div class="sub" id="sub"></div></div>
<div class="kpis" id="kpis"></div>

<div class="tabs">
  <div class="tab on" data-p="p1">Segmentation</div>
  <div class="tab" data-p="p2">Visualisation</div>
</div>

<!-- ======================================================== SEGMENTATION -->
<div class="panel on" id="p1">
  <div class="row">
    <div class="card"><h3>1 · Segmentation</h3>
      <select id="s_seg" multiple></select>
      <label>Nombre de classes (variables continues)</label>
      <select id="s_bins"></select>
      <div class="note">1 ou 2 variables. Ctrl+clic pour la seconde.</div>
    </div>
    <div class="card"><h3>2 · Indicateurs agrégés</h3>
      <label>Variables</label><select id="s_aggv" multiple style="height:96px"></select>
      <label>Fonctions</label><select id="s_aggf" multiple style="height:96px"></select>
    </div>
    <div class="card"><h3>3 · Ratios</h3>
      <label>Numérateur</label><select id="r_num"></select>
      <label>Dénominateur</label><select id="r_den"></select>
      <div class="chk"><input type="checkbox" id="r_indice"><span>Ajouter l'indice base 100</span></div>
      <button class="mini" id="b_addratio">Ajouter le ratio</button>
      <div class="chips" id="r_liste"></div>
      <div class="note">Somme(num.) / Somme(dén.) par segment. Dénominateur nul : cellule vide.</div>
    </div>
  </div>
  <div class="btns"><button class="p" id="b_tab">Générer le tableau</button>
    <button id="b_csv">Exporter en CSV</button></div>
  <div id="o_tab"></div>
</div>

<!-- ======================================================== VISUALISATION -->
<div class="panel" id="p2">
  <div class="stabs">
    <div class="stab on" data-s="v1">1 · Variables catégorielles</div>
    <div class="stab" data-s="v2">2 · Variables continues</div>
    <div class="stab" data-s="v3">3 · Heatmap</div>
  </div>

  <div class="spanel on" id="v1">
    <div class="row">
      <div class="card"><h3>Variable catégorielle</h3><select id="c_var"></select>
        <div class="note" id="c_info"></div></div>
      <div class="card" style="min-width:300px"><h3>Visualisation</h3><select id="c_type"></select>
        <label>Mesure associée</label><select id="c_mes"></select></div>
    </div>
    <div class="btns"><button class="p" id="b_cat">Afficher</button></div>
    <div id="o_cat" class="chart"></div>
  </div>

  <div class="spanel" id="v2">
    <div class="row">
      <div class="card"><h3>Variable continue</h3><select id="n_var"></select>
        <label>Nombre de classes</label><select id="n_bins"></select>
        <div class="note">Classes de quantiles ; les manquants forment une classe distincte.</div></div>
      <div class="card" style="min-width:300px"><h3>Visualisation</h3><select id="n_type"></select>
        <label>Mesure associée</label><select id="n_mes"></select>
        <div class="note" id="n_info"></div></div>
    </div>
    <div class="btns"><button class="p" id="b_num">Afficher</button></div>
    <div id="o_num" class="chart"></div>
  </div>

  <div class="spanel" id="v3">
    <div class="row">
      <div class="card"><h3>Croisement</h3>
        <label>Variable 1 (lignes)</label><select id="h_v1"></select>
        <label>Variable 2 (colonnes)</label><select id="h_v2"></select>
        <label>Nombre de classes</label><select id="h_bins"></select></div>
      <div class="card" style="min-width:320px"><h3>Métrique et lisibilité</h3>
        <label>Métrique</label><select id="h_metrique"></select>
        <label>Mesure associée</label><select id="h_mes"></select>
        <label>Effectif minimum par cellule</label><input type="number" id="h_min" value="30" min="0" step="10">
        <div class="chk"><input type="checkbox" id="h_n" checked><span>Afficher les effectifs</span></div>
      </div>
    </div>
    <div class="btns"><button class="p" id="b_heat">Afficher la heatmap</button></div>
    <div id="o_heat" class="chart"></div>
  </div>
</div>
</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
(function(){
"use strict";
const D = __DATA__;
const VARS=D.vars, C1=D.cube1, C2=D.cube2, HIST=D.hist, MESURES=D.mesures;
const CIBLE=D.cible, PROBA=D.proba, NTOT=D.n;
const NAVY="#14476B", BLEU="#2E7DAF", ROUGE="#B3261E", VERT="#1E7B5E", GRIS="#E3E8ED";
const ECHELLE=[[0,"#F5F9FC"],[.25,"#DCEAF4"],[.5,"#9EC6E0"],[.75,"#2E7DAF"],[1,"#14476B"]];
const DIVERGENTE=[[0,"#1E7B5E"],[.5,"#F7F7F7"],[1,"#B3261E"]];
const $ = id => document.getElementById(id);

/* ============================ 1. Formats ============================ */
const fmtInt = v => v==null||!isFinite(v) ? "–" : Math.round(v).toLocaleString("fr-FR");
const fmtPct = (v,d=2) => v==null||!isFinite(v) ? "–" : (v*100).toFixed(d).replace(".",",")+" %";
const fmtNum = (v,d=4) => v==null||!isFinite(v) ? "–" :
      v.toLocaleString("fr-FR",{minimumFractionDigits:d,maximumFractionDigits:d});
const fmtSigne = (v,d=2) => v==null||!isFinite(v) ? "–" : (v>=0?"+":"−")+fmtPct(Math.abs(v),d);
function borneLisible(x, etendue){
  const d = etendue>=100 ? 0 : etendue>=10 ? 1 : etendue>=1 ? 2 : 4;
  return x.toLocaleString("fr-FR",{minimumFractionDigits:d,maximumFractionDigits:d});
}

/* ============================ 2. Classes et regroupement ============================
   Les variables continues sont découpées en 20 classes de quantiles côté Python.
   Regrouper n classes consécutives redonne des classes de quantiles exactes,
   ce qui permet de choisir 2, 4, 5, 10 ou 20 classes sans recalcul serveur.        */
function diviseurs(n){ const d=[]; for(let k=2;k<=n;k++) if(n%k===0) d.push(k); return d; }

function grille(v, k){
  const m = VARS[v], nfin = m.nfin, manquant = m.manquant;
  if(m.type !== "cont" || !k || k >= nfin || nfin % k !== 0)
    return {labels:m.labels.slice(), map:m.labels.map((_,i)=>i), n:m.labels.length};

  const taille = nfin/k, map = [], labels = [];
  const e = m.edges, etendue = e[e.length-1]-e[0];
  for(let g=0;g<k;g++)
    labels.push((g===0?"[":"]")+borneLisible(e[g*taille],etendue)
                +" ; "+borneLisible(e[(g+1)*taille],etendue)+"]");
  for(let i=0;i<nfin;i++) map.push(Math.floor(i/taille));
  if(manquant){ map.push(k); labels.push("Manquant"); }
  return {labels, map, n:labels.length};
}

/* ============================ 3. Lecture des cubes ============================
   Une "cellule" agrège : effectif, et par mesure la somme, le nb de valeurs non
   nulles, le min, le max et la somme des carrés. Toutes ces quantités sont
   additives, donc regroupables sans repasser par les données brutes.              */
function celluleVide(){ return {n:0, m:{}}; }
function ajouter(dest, mes, s, c, mn, mx, sq){
  let b = dest.m[mes];
  if(!b) b = dest.m[mes] = {s:0, c:0, sq:0, mn:null, mx:null};
  b.s += s||0; b.c += c||0; b.sq += sq||0;
  if(mn!=null) b.mn = b.mn==null ? mn : Math.min(b.mn,mn);
  if(mx!=null) b.mx = b.mx==null ? mx : Math.max(b.mx,mx);
}

function lignes1(v, k){                                  // segmentation sur une variable
  const g = grille(v,k), cube = C1[v], out = [];
  for(let c=0;c<g.n;c++) out.push(Object.assign(celluleVide(), {cles:[g.labels[c]]}));
  cube.n.forEach((n,i)=>{
    const c = g.map[i]; if(c==null) return;
    out[c].n += n;
    MESURES.forEach(m=>{
      const b = cube.mes[m];
      ajouter(out[c], m, b.s[i], b.c[i], b.mn[i], b.mx[i], b.sq[i]);
    });
  });
  return out.filter(l=>l.n>0);
}

function lignes2(v1, v2, k1, k2){                        // segmentation croisée
  let cube = C2[v1+"|"+v2], inverse = false;
  if(!cube){ cube = C2[v2+"|"+v1]; inverse = true; }
  if(!cube) return null;
  const g1 = grille(v1,k1), g2 = grille(v2,k2), map = new Map();
  const ii = inverse ? cube.j : cube.i, jj = inverse ? cube.i : cube.j;
  for(let z=0;z<cube.n.length;z++){
    const a = g1.map[ii[z]], b = g2.map[jj[z]];
    if(a==null||b==null) continue;
    const cle = a+"|"+b;
    let cel = map.get(cle);
    if(!cel){ cel = Object.assign(celluleVide(), {cles:[g1.labels[a], g2.labels[b]], _o:[a,b]});
              map.set(cle,cel); }
    cel.n += cube.n[z];
    MESURES.forEach(m=>{
      const c = cube.c[m] ? cube.c[m][z] : cube.n[z];    // nb non nul = effectif si aucun manquant
      ajouter(cel, m, cube.s[m][z], c, null, null, null);
    });
  }
  return [...map.values()].sort((a,b)=> a._o[0]-b._o[0] || a._o[1]-b._o[1]);
}

function totalise(lignes, cles){                         // ligne TOTAL = somme des cellules
  const t = Object.assign(celluleVide(), {cles, total:true});
  lignes.forEach(l=>{
    t.n += l.n;
    MESURES.forEach(m=>{ const b=l.m[m]; if(b) ajouter(t,m,b.s,b.c,b.mn,b.mx,b.sq); });
  });
  return t;
}

/* ============================ 4. Agrégations ============================ */
const AGGS = {
  moyenne:    b => b && b.c ? b.s/b.c : NaN,
  somme:      b => b ? b.s : NaN,
  nb_obs:     b => b ? b.c : NaN,
  min:        b => b && b.mn!=null ? b.mn : NaN,
  max:        b => b && b.mx!=null ? b.mx : NaN,
  ecart_type: b => b && b.c>1 ? Math.sqrt(Math.max(0,(b.sq-b.s*b.s/b.c)/(b.c-1))) : NaN,
};
const AGGS_CROISEES = ["moyenne","somme","nb_obs"];       // seules additives sur le cube croisé
const moy = (cel,m) => AGGS.moyenne(cel.m[m]);

/* ============================ 5. Tableau de segmentation ============================ */
let ratios = [];                                          // [{num, den, indice}]

function construireTableau(segs, k, aggVars, aggFuncs){
  const croise = segs.length === 2;
  const lignes = croise ? lignes2(segs[0],segs[1],k,k) : lignes1(segs[0],k);
  if(!lignes) throw new Error("Croisement non pré-calculé pour ces deux variables.");
  const total = totalise(lignes, segs.map((_,i)=> i===0 ? "TOTAL" : ""));
  const toutes = [...lignes, total];

  const colonnes = [...segs, "effectif", "poids_pct"];
  if(CIBLE) colonnes.push("taux_reel");
  if(PROBA) colonnes.push("proba_modele");
  if(CIBLE && PROBA) colonnes.push("ecart_reel_modele");
  ratios.forEach(r=>{ colonnes.push(r.nom); if(r.indice) colonnes.push(r.nom+"_base100"); });
  aggVars.forEach(v=> aggFuncs.forEach(f=> colonnes.push(v+"_"+f)));

  const donnees = toutes.map(cel=>{
    const l = {_total:!!cel.total};
    segs.forEach((v,i)=> l[v]=cel.cles[i]);
    l.effectif = cel.n;
    l.poids_pct = cel.n/total.n;
    if(CIBLE) l.taux_reel = moy(cel,CIBLE);
    if(PROBA) l.proba_modele = moy(cel,PROBA);
    if(CIBLE && PROBA) l.ecart_reel_modele = moy(cel,CIBLE)-moy(cel,PROBA);
    ratios.forEach(r=>{
      const num=cel.m[r.num], den=cel.m[r.den];
      l[r.nom] = (den && den.s>0 && num) ? num.s/den.s : NaN;   // dénominateur nul -> pas de ratio
    });
    aggVars.forEach(v=> aggFuncs.forEach(f=> l[v+"_"+f] = AGGS[f](cel.m[v])));
    return l;
  });

  ratios.filter(r=>r.indice).forEach(r=>{                 // indice base 100 = ratio d'ensemble
    const global = donnees[donnees.length-1][r.nom];
    donnees.forEach(l=> l[r.nom+"_base100"] = (global && isFinite(global)) ? 100*l[r.nom]/global : NaN);
  });
  return {colonnes, lignes:donnees, seg:segs};
}

/* ============================ 6. Rendu des tableaux ============================ */
const DEGRADE = new Set(["taux_reel","proba_modele"]);
function typeColonne(c){
  if(c==="poids_pct"||c==="taux_reel"||c==="proba_modele") return "pct";
  if(c==="effectif"||c.endsWith("_nb_obs")) return "int";
  if(c==="ecart_reel_modele") return "signe";
  if(c.endsWith("_base100")) return "indice";
  return "num";
}
function formater(c,v){
  switch(typeColonne(c)){
    case "pct":   return fmtPct(v, c==="poids_pct"?1:2);
    case "int":   return fmtInt(v);
    case "signe": return fmtSigne(v);
    case "indice":return v==null||!isFinite(v) ? "–" : Math.round(v).toLocaleString("fr-FR");
    default:      return typeof v === "number" ? fmtNum(v) : (v==null?"–":v);
  }
}
function fond(t){
  const a=[245,249,252], b=[20,71,107], c=a.map((x,i)=>Math.round(x+(b[i]-x)*t));
  return {bg:`rgb(${c.join(",")})`, fg: t>.65 ? "#fff" : "inherit"};
}

function rendreTableau(res, cible){
  const {colonnes, lignes, seg} = res, bornes = {};
  const degrade = new Set([...DEGRADE, ...ratios.map(r=>r.nom)]);
  colonnes.forEach(c=>{
    if(!degrade.has(c)) return;
    const v = lignes.filter(l=>!l._total).map(l=>l[c]).filter(x=>x!=null&&isFinite(x));
    if(v.length) bornes[c] = [Math.min(...v), Math.max(...v)];
  });

  let h = '<div class="tw"><table><thead><tr>';
  colonnes.forEach((c,i)=> h += `<th class="${seg.includes(c)?'seg':''}" data-c="${i}">${c}</th>`);
  h += '</tr></thead><tbody>';
  lignes.forEach(l=>{
    h += `<tr class="${l._total?'tot':''}">`;
    colonnes.forEach(c=>{
      let style="";
      if(bornes[c] && !l._total && isFinite(l[c])){
        const [mn,mx]=bornes[c], t = mx>mn ? (l[c]-mn)/(mx-mn) : .5, col=fond(t);
        style = `background:${col.bg};color:${col.fg}`;
      }
      if(c==="ecart_reel_modele" && isFinite(l[c]))
        style = `color:${l[c]>0?ROUGE:VERT};font-weight:600`;
      h += `<td class="${seg.includes(c)?'seg':''}" style="${style}">${formater(c,l[c])}</td>`;
    });
    h += '</tr>';
  });
  $(cible).innerHTML = h + '</tbody></table></div>';

  let sens=1, dernier=-1;                                  // tri au clic sur l'en-tête
  $(cible).querySelectorAll("th").forEach(th=> th.onclick = ()=>{
    const i=+th.dataset.c, c=colonnes[i];
    sens = (i===dernier) ? -sens : 1; dernier=i;
    const corps = lignes.filter(l=>!l._total), tot = lignes.filter(l=>l._total);
    corps.sort((a,b)=>{
      const x=a[c], y=b[c];
      if(typeof x==="number" && typeof y==="number")
        return sens*((isFinite(x)?x:-Infinity)-(isFinite(y)?y:-Infinity));
      return sens*String(x).localeCompare(String(y),"fr");
    });
    rendreTableau({colonnes, lignes:[...corps,...tot], seg}, cible);
  });
}

/* ============================ 7. Graphiques ============================ */
const BASE = {
  font:{family:"Segoe UI, Arial, sans-serif", size:12, color:"#1F2A37"},
  plot_bgcolor:"#fff", paper_bgcolor:"#fff", height:460,
  margin:{l:70,r:30,t:70,b:80},
  xaxis:{gridcolor:GRIS,zerolinecolor:GRIS,automargin:true},
  yaxis:{gridcolor:GRIS,zerolinecolor:GRIS,automargin:true},
  title:{font:{size:15,color:NAVY}, x:.02, xanchor:"left", y:.97, yanchor:"top"},
};
const CONFIG={displayModeBar:true, displaylogo:false, responsive:true,
              modeBarButtonsToRemove:["lasso2d","select2d"]};

function tracer(cible, traces, layout){
  if(typeof Plotly === "undefined"){
    $(cible).innerHTML = '<div class="msg">Plotly n\'a pas pu être chargé depuis le CDN. '
      + 'Les tableaux restent disponibles.</div>'; return;
  }
  const l = JSON.parse(JSON.stringify(BASE));
  Object.keys(layout||{}).forEach(k=>{
    l[k] = (typeof layout[k]==="object" && !Array.isArray(layout[k]) && l[k])
           ? Object.assign(l[k], layout[k]) : layout[k];
  });
  Plotly.newPlot(cible, traces, l, CONFIG);
}
function message(txt, cible, ajouter){
  const h = '<div class="msg">'+txt+'</div>';
  if(ajouter) $(cible).insertAdjacentHTML("beforeend", h); else $(cible).innerHTML = h;
}

/* ---- graphiques communs aux deux familles de variables, alimentés par le cube 1 ---- */
function serie(v, k, calcul){
  const l = lignes1(v,k);
  return {x:l.map(c=>c.cles[0]), y:l.map(calcul), n:l.map(c=>c.n)};
}

function grEffectifs(v,k,cible){
  const s = serie(v,k,c=>c.n);
  tracer(cible,[{type:"bar",x:s.x,y:s.y,marker:{color:NAVY},
                 text:s.y.map(fmtInt),textposition:"outside",cliponaxis:false}],
    {title:{text:"Effectifs par modalité de "+v}, xaxis:{title:v,tickangle:-25}, yaxis:{title:"Effectif"}});
}
function grPareto(v,k,cible){
  const l = lignes1(v,k).sort((a,b)=>b.n-a.n);
  const x=l.map(c=>c.cles[0]), y=l.map(c=>c.n), tot=y.reduce((s,z)=>s+z,0);
  let cum=0; const cumule = y.map(z=> (cum+=z)/tot);
  tracer(cible,[
    {type:"bar",x,y,name:"Effectif",marker:{color:BLEU}},
    {type:"scatter",mode:"lines+markers",x,y:cumule,name:"Cumul",yaxis:"y2",
     line:{color:NAVY,width:3},hovertemplate:"%{x}<br>Cumul : %{y:.1%}<extra></extra>"}],
    {title:{text:"Concentration des effectifs de "+v+" (Pareto)"},
     xaxis:{title:v,tickangle:-25}, yaxis:{title:"Effectif"},
     yaxis2:{overlaying:"y",side:"right",tickformat:".0%",range:[0,1.02],gridcolor:"transparent"},
     legend:{orientation:"h",y:1.12}});
}
function grTaux(v,k,cible){
  const s = serie(v,k,c=>moy(c,CIBLE));
  tracer(cible,[{type:"bar",x:s.x,y:s.y,marker:{color:NAVY},customdata:s.n,
    text:s.y.map(z=>fmtPct(z,1)),textposition:"outside",cliponaxis:false,
    hovertemplate:"%{x}<br>Taux : %{y:.2%}<br>n = %{customdata:,}<extra></extra>"}],
    {title:{text:"Taux réel de "+CIBLE+" par modalité de "+v},
     xaxis:{title:v,tickangle:-25}, yaxis:{title:"Taux",tickformat:".1%"}});
}
function grCalibration(v,k,cible){
  const l = lignes1(v,k), x=l.map(c=>c.cles[0]);
  tracer(cible,[
    {type:"bar",x,y:l.map(c=>moy(c,CIBLE)),name:"Taux réel observé",marker:{color:BLEU},
     customdata:l.map(c=>c.n),
     hovertemplate:"%{x}<br>Réel : %{y:.2%}<br>n = %{customdata:,}<extra></extra>"},
    {type:"scatter",mode:"lines+markers",x,y:l.map(c=>moy(c,PROBA)),name:"Probabilité prédite",
     line:{color:NAVY,width:3},marker:{size:8},
     hovertemplate:"%{x}<br>Prédit : %{y:.2%}<extra></extra>"}],
    {title:{text:"Calibration du modèle par classe de "+v},
     xaxis:{title:v,tickangle:-25}, yaxis:{title:"Taux",tickformat:".1%"},
     legend:{orientation:"h",y:1.12}});
}
function grEcart(v,k,cible){
  const s = serie(v,k,c=>moy(c,CIBLE)-moy(c,PROBA));
  tracer(cible,[{type:"bar",x:s.x,y:s.y,customdata:s.n,
    marker:{color:s.y.map(z=>z>0?ROUGE:VERT)},
    hovertemplate:"%{x}<br>Écart : %{y:.2%}<br>n = %{customdata:,}<extra></extra>"}],
    {title:{text:"Écart taux réel - probabilité prédite par classe de "+v},
     xaxis:{title:v,tickangle:-25}, yaxis:{title:"Écart",tickformat:".1%",zeroline:true,
                                           zerolinecolor:"#9AA7B4",zerolinewidth:2}});
}
function grMoyenne(v,k,mes,cible){
  const s = serie(v,k,c=>moy(c,mes));
  tracer(cible,[{type:"bar",x:s.x,y:s.y,marker:{color:NAVY},customdata:s.n,
    hovertemplate:"%{x}<br>%{y:,.2f}<br>n = %{customdata:,}<extra></extra>"}],
    {title:{text:"Moyenne de "+mes+" par classe de "+v},
     xaxis:{title:v,tickangle:-25}, yaxis:{title:"Moyenne de "+mes}});
}
function grPoids(v,k,cible){
  const l = lignes1(v,k), tot = l.reduce((s,c)=>s+c.n,0);
  tracer(cible,[{type:"pie",labels:l.map(c=>c.cles[0]),values:l.map(c=>c.n),hole:.45,
    marker:{colors:["#14476B","#2E7DAF","#7FB3D3","#A9CCE3","#5B8AA6","#0F3A57","#9EC6E0","#C8DCE8"]},
    textinfo:"label+percent",hovertemplate:"%{label}<br>n = %{value:,}<extra></extra>"}],
    {title:{text:"Répartition de la population par "+v+" ("+fmtInt(tot)+" observations)"},
     height:470});
}

/* ---- variables continues : histogramme et boxplot depuis les pré-calculs dédiés ---- */
function grHistogramme(v,cible,parCible){
  const h = HIST[v];
  if(!h) return message("Histogramme non pré-calculé pour cette variable.",cible);
  const centres = h.edges.slice(0,-1).map((e,i)=>(e+h.edges[i+1])/2);
  const largeur = h.edges[1]-h.edges[0];
  const info = "Extrêmes regroupés aux bornes : "+fmtInt(h.extremes[0])+" en dessous, "
             + fmtInt(h.extremes[1])+" au-dessus · "+fmtInt(h.manquants)+" manquant(s)";
  if(parCible && h.n1){
    const t0=h.n0.reduce((s,z)=>s+z,0), t1=h.n1.reduce((s,z)=>s+z,0);
    tracer(cible,[
      {type:"bar",x:centres,y:h.n0.map(z=>z/t0),name:"Non "+CIBLE,marker:{color:BLEU},
       width:largeur,opacity:.65},
      {type:"bar",x:centres,y:h.n1.map(z=>z/t1),name:CIBLE,marker:{color:NAVY},
       width:largeur,opacity:.65}],
      {barmode:"overlay",title:{text:"Distribution de "+v+" selon "+CIBLE},
       xaxis:{title:v},yaxis:{title:"% de chaque population",tickformat:".1%"},
       legend:{orientation:"h",y:1.12},annotations:[noteBas(info)]});
  } else {
    tracer(cible,[{type:"bar",x:centres,y:h.n,marker:{color:NAVY,line:{color:"#fff",width:1}},
      width:largeur,hovertemplate:"%{x}<br>n = %{y:,}<extra></extra>"}],
      {title:{text:"Distribution de "+v},xaxis:{title:v},yaxis:{title:"Effectif"},
       annotations:[noteBas(info)]});
  }
}
function noteBas(txt){
  return {text:txt,showarrow:false,xref:"paper",yref:"paper",x:0,y:-.22,
          xanchor:"left",font:{size:10.5,color:"#5B6B7B"}};
}
function grBoxplot(v,mes,cible){
  if(!D.quantiles) return message("Quartiles non pré-calculés (INCLURE_QUANTILES = False).",cible);
  const m=VARS[v], cube=C1[v], b=cube.mes[mes], g=grille(v,null), traces=[];
  cube.n.forEach((n,i)=>{
    if(!n || !b.q[i] || b.q[i][1]==null) return;
    traces.push({type:"box",name:g.labels[i],q1:[b.q[i][0]],median:[b.q[i][1]],q3:[b.q[i][2]],
      lowerfence:[b.mn[i]],upperfence:[b.mx[i]],marker:{color:BLEU},line:{color:NAVY},
      showlegend:false});
  });
  if(!traces.length) return message("Pas de quartiles disponibles pour cette combinaison.",cible);
  tracer(cible,traces,{title:{text:"Dispersion de "+mes+" par classe de "+v+" (découpage fin)"},
    xaxis:{title:v,tickangle:-25},yaxis:{title:mes},
    annotations:[noteBas("Boîtes construites sur les 20 classes de quantiles : min, Q1, médiane, Q3, max.")]});
}
function grProbaParClasse(v,k,cible){
  const cube = C2[v+"|"+PROBA] || C2[PROBA+"|"+v];
  if(!cube) return message("Croisement avec le score non disponible.",cible);
  const inverse = !C2[v+"|"+PROBA];
  const gv=grille(v,k), gp=grille(PROBA,null);
  const ii = inverse?cube.j:cube.i, jj = inverse?cube.i:cube.j;
  const mat = gv.labels.map(()=>gp.labels.map(()=>0));
  for(let z=0;z<cube.n.length;z++){
    const a=gv.map[ii[z]], b=gp.map[jj[z]];
    if(a!=null&&b!=null) mat[a][b]+=cube.n[z];
  }
  const traces = mat.map((ligne,a)=>{
    const tot = ligne.reduce((s,z)=>s+z,0) || 1;
    return {type:"scatter",mode:"lines",x:gp.labels,y:ligne.map(z=>z/tot),
      name:gv.labels[a],line:{width:2},
      hovertemplate:"%{x}<br>%{y:.1%} de la classe<extra></extra>"};
  }).filter((t,a)=>mat[a].some(z=>z>0)).slice(0,10);
  tracer(cible,traces,{title:{text:"Répartition des probabilités prédites par classe de "+v},
    xaxis:{title:"Classe de probabilité prédite (déciles)",tickangle:-25},
    yaxis:{title:"% de la classe",tickformat:".0%"},legend:{title:{text:v}}});
}

/* ============================ 8. Heatmap ============================ */
function metriquesDispo(){
  const m = {};
  if(PROBA) m["Probabilité moyenne prédite"] = c=>moy(c,PROBA);
  if(CIBLE) m["Taux réel observé"] = c=>moy(c,CIBLE);
  if(CIBLE&&PROBA) m["Écart réel - prédit"] = c=>moy(c,CIBLE)-moy(c,PROBA);
  m["Effectif"] = c=>c.n;
  m["Moyenne d'une mesure"] = (c,mes)=>moy(c,mes);
  m["Ratio de deux mesures"] = (c,mes,den)=>{
    const a=c.m[mes], b=c.m[den]; return (b&&b.s>0&&a) ? a.s/b.s : NaN; };
  return m;
}

function dessinerHeatmap(v1,v2,nom,mes,den,k,minObs,afficherN){
  const cellules = lignes2(v1,v2,k,k);
  if(!cellules) return message("Croisement non pré-calculé pour ces deux variables.","o_heat");
  const g1=grille(v1,k), g2=grille(v2,k), fonction=metriquesDispo()[nom];
  const signe = nom==="Écart réel - prédit";
  const pct = ["Probabilité moyenne prédite","Taux réel observé","Écart réel - prédit"].includes(nom);
  const brut = nom==="Effectif" || nom==="Moyenne d'une mesure" || nom==="Ratio de deux mesures";

  const z=g1.labels.map(()=>g2.labels.map(()=>null));
  const txt=g1.labels.map(()=>g2.labels.map(()=>""));
  const eff=g1.labels.map(()=>g2.labels.map(()=>0));
  let masquees=0;
  cellules.forEach(c=>{
    const [a,b]=c._o; eff[a][b]=c.n;
    if(nom!=="Effectif" && c.n<minObs){ masquees++; return; }
    const val = fonction(c,mes,den);
    if(val==null||!isFinite(val)) return;
    z[a][b]=val;
    txt[a][b] = (signe?fmtSigne(val,1):pct?fmtPct(val,1):nom==="Effectif"?fmtInt(val):fmtNum(val,3))
              + (afficherN && nom!=="Effectif" ? "<br><span style='font-size:9px'>n="
                 +fmtInt(c.n)+"</span>" : "");
  });

  const garder = g1.labels.map((_,a)=> eff[a].some(x=>x>0));   // lignes vides retirées
  const y = g1.labels.filter((_,a)=>garder[a]);
  const zf = z.filter((_,a)=>garder[a]), tf = txt.filter((_,a)=>garder[a]);

  // marges calculées à partir de la longueur réelle des libellés : le titre ne peut plus
  // chevaucher l'axe, quel que soit le nombre de classes affichées
  const lgY = Math.max(...y.map(s=>s.length), 6);
  const lgX = Math.max(...g2.labels.map(s=>s.length), 6);
  const titre = nom + (nom.includes("mesure") ? " (" + mes + (nom.startsWith("Ratio")?" / "+den:"") + ")" : "")
              + " — " + v1 + " × " + v2;

  tracer("o_heat",[{
    type:"heatmap", z:zf, x:g2.labels, y, text:tf, texttemplate:"%{text}",
    textfont:{size:11}, hoverongaps:false, xgap:1, ygap:1,
    colorscale: signe?DIVERGENTE:ECHELLE, zmid: signe?0:undefined,
    colorbar:{title:{text:pct?"Taux":"Valeur",side:"right"}, tickformat:pct?".1%":",",
              thickness:14, len:.8, y:.42},
    hovertemplate:"%{y} × %{x}<br>"+nom+" : %{z}<extra></extra>"}],
    {title:{text:titre, y:.985, yanchor:"top", pad:{b:14}},
     xaxis:{title:{text:v2, standoff:16}, side:"bottom", tickangle:-25, automargin:true,
            gridcolor:"transparent", type:"category"},
     yaxis:{title:{text:v1, standoff:16}, autorange:"reversed", automargin:true,
            gridcolor:"transparent", type:"category"},
     height: Math.max(420, 150 + 44*y.length),
     margin:{l:Math.min(260, 45+6.6*lgY), r:70, t:96, b:Math.min(180, 70+5.5*lgX)}});

  if(masquees) message(masquees+" cellule(s) masquée(s) : moins de "+minObs+" observations.",
                       "o_heat", true);
}

/* ============================ 9. Export CSV ============================ */
function exporterCSV(res, fichier){
  const l = [res.colonnes.join(";")];
  res.lignes.forEach(r=> l.push(res.colonnes.map(c=>{
    const v=r[c];
    return typeof v==="number" ? (isFinite(v)?String(v).replace(".",","):"") : (v==null?"":v);
  }).join(";")));
  const blob = new Blob(["\ufeff"+l.join("\n")],{type:"text/csv;charset=utf-8;"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = fichier;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

/* ============================ 10. Interface ============================ */
function remplir(id, options, defaut){
  $(id).innerHTML = options.map(o=>`<option value="${o}">${o}</option>`).join("");
  if(defaut!==undefined && options.includes(defaut)) $(id).value = defaut;
}
const selection = id => [...$(id).selectedOptions].map(o=>o.value);
function proteger(f, cible){
  try{ f(); } catch(e){ $(cible).innerHTML = '<div class="err">'+(e&&e.stack?e.stack:e)+'</div>'; }
}

const segVars   = Object.keys(VARS);
const mesures   = MESURES.filter(m=> m!==CIBLE && m!==PROBA);
const mesTous   = MESURES;
const categorielles = segVars.filter(v=> VARS[v].type!=="cont");
const continues     = segVars.filter(v=> VARS[v].type==="cont");
const classesPossibles = diviseurs(D.nfin).filter(k=>k<=D.nfin);

// --- bandeau et indicateurs globaux
const totalGlobal = totalise(lignes1(segVars[0], null), ["TOTAL"]);
const tauxG  = CIBLE ? moy(totalGlobal,CIBLE) : null;
const probaG = PROBA ? moy(totalGlobal,PROBA) : null;
$("sub").textContent =
  (CIBLE ? "Cible : "+CIBLE : "Aucune cible dans la base")
  + (PROBA ? " · Modèle : "+D.modele+" ("+D.features.length+" features)" : " · Aucun modèle")
  + " · " + fmtInt(NTOT) + " observations pré-agrégées · " + segVars.length + " variables";

const kpis = [["Observations", fmtInt(NTOT), BLEU]];
if(CIBLE) kpis.push(["Taux réel observé", fmtPct(tauxG), BLEU]);
if(PROBA) kpis.push(["Probabilité moyenne prédite", fmtPct(probaG), BLEU]);
if(CIBLE&&PROBA) kpis.push(["Écart réel - prédit", fmtSigne(tauxG-probaG),
                            Math.abs(tauxG-probaG)>.01?ROUGE:VERT]);
if(!CIBLE) kpis.push(["Indicateurs de cible", "désactivés", "#9AA7B4"]);
$("kpis").innerHTML = kpis.map(([l,v,c])=>`<div class="kpi" style="border-left-color:${c}">
   <div class="lab">${l}</div><div class="val">${v}</div></div>`).join("");

// --- alimentation des listes
remplir("s_seg", segVars); $("s_seg").selectedIndex = 0;
remplir("s_bins", classesPossibles, 5);
remplir("s_aggv", mesTous); remplir("s_aggf", Object.keys(AGGS)); $("s_aggf").selectedIndex = 0;
remplir("r_num", mesTous); remplir("r_den", mesTous);
if(mesTous.length>1) $("r_den").selectedIndex = 1;

remplir("c_var", categorielles.length?categorielles:segVars);
remplir("n_var", continues.length?continues:segVars);
remplir("n_bins", classesPossibles, 5);
remplir("c_mes", mesures.length?mesures:mesTous); remplir("n_mes", mesures.length?mesures:mesTous);
remplir("h_v1", segVars); remplir("h_v2", segVars);
if(segVars.length>1) $("h_v2").selectedIndex = 1;
remplir("h_bins", classesPossibles, 5);
remplir("h_metrique", Object.keys(metriquesDispo()));
remplir("h_mes", mesTous);

const GR_CAT = ["Effectifs par modalité","Répartition (anneau)","Concentration (Pareto)",
  "Moyenne d'une mesure"].concat(CIBLE?["Taux réel par modalité"]:[])
  .concat(CIBLE&&PROBA?["Prédit vs observé","Écart réel - prédit"]:[]);
const GR_NUM = ["Histogramme","Effectifs par classe","Boxplot d'une mesure","Moyenne d'une mesure"]
  .concat(CIBLE?["Distribution selon la cible","Taux réel par classe"]:[])
  .concat(CIBLE&&PROBA?["Prédit vs observé","Écart réel - prédit"]:[])
  .concat(PROBA?["Probabilités prédites par classe"]:[]);
remplir("c_type", GR_CAT); remplir("n_type", GR_NUM);
$("c_info").textContent = categorielles.length
  ? categorielles.length+" variable(s) catégorielle(s) détectée(s)."
  : "Aucune variable catégorielle détectée.";
$("n_info").textContent = continues.length
  ? "Le boxplot utilise systématiquement le découpage fin en "+D.nfin+" classes."
  : "Aucune variable continue détectée.";

// --- onglets et sous-onglets
document.querySelectorAll("#app .tab").forEach(t=> t.onclick = ()=>{
  document.querySelectorAll("#app .tab").forEach(x=>x.classList.remove("on"));
  document.querySelectorAll("#app .panel").forEach(x=>x.classList.remove("on"));
  t.classList.add("on"); $(t.dataset.p).classList.add("on");
});
document.querySelectorAll("#app .stab").forEach(t=> t.onclick = ()=>{
  document.querySelectorAll("#app .stab").forEach(x=>x.classList.remove("on"));
  document.querySelectorAll("#app .spanel").forEach(x=>x.classList.remove("on"));
  t.classList.add("on"); $(t.dataset.s).classList.add("on");
});

// --- ratios : ajout, suppression, affichage
function rendreRatios(){
  $("r_liste").innerHTML = ratios.map((r,i)=>
    `<div class="chip">${r.nom}<span data-i="${i}">×</span></div>`).join("");
  $("r_liste").querySelectorAll("span").forEach(s=> s.onclick = ()=>{
    ratios.splice(+s.dataset.i,1); rendreRatios(); });
}
$("b_addratio").onclick = ()=>{
  const num=$("r_num").value, den=$("r_den").value;
  if(num===den) return message("Numérateur et dénominateur doivent être différents.","o_tab");
  const nom = num+"/"+den;
  if(!ratios.some(r=>r.nom===nom))
    ratios.push({nom, num, den, indice:$("r_indice").checked});
  rendreRatios();
};

// --- actions
let dernierTableau = null;
$("b_tab").onclick = ()=> proteger(()=>{
  let segs = selection("s_seg");
  if(!segs.length) return message("Sélectionner au moins une variable de segmentation.","o_tab");
  let avis = "";
  if(segs.length>2){ segs = segs.slice(0,2);
    avis = "Segmentation limitée à 2 variables (croisements pré-calculés) : "+segs.join(" × ")+". "; }
  let funcs = selection("s_aggf");
  if(segs.length===2){
    const retirees = funcs.filter(f=>!AGGS_CROISEES.includes(f));
    funcs = funcs.filter(f=>AGGS_CROISEES.includes(f));
    if(retirees.length) avis += "En croisement, "+retirees.join(", ")
      +" ne sont pas disponibles (non additifs) ; utiliser une seule variable.";
  }
  dernierTableau = construireTableau(segs, +$("s_bins").value, selection("s_aggv"), funcs);
  rendreTableau(dernierTableau, "o_tab");
  if(avis) message(avis, "o_tab", true);
}, "o_tab");

$("b_csv").onclick = ()=> dernierTableau ? exporterCSV(dernierTableau,"segmentation.csv")
                                         : message("Générer d'abord un tableau.","o_tab");

$("b_cat").onclick = ()=> proteger(()=>{
  const v=$("c_var").value, t=$("c_type").value, m=$("c_mes").value;
  if(t==="Effectifs par modalité")      return grEffectifs(v,null,"o_cat");
  if(t==="Répartition (anneau)")        return grPoids(v,null,"o_cat");
  if(t==="Concentration (Pareto)")      return grPareto(v,null,"o_cat");
  if(t==="Moyenne d'une mesure")        return grMoyenne(v,null,m,"o_cat");
  if(t==="Taux réel par modalité")      return grTaux(v,null,"o_cat");
  if(t==="Prédit vs observé")           return grCalibration(v,null,"o_cat");
  if(t==="Écart réel - prédit")         return grEcart(v,null,"o_cat");
}, "o_cat");

$("b_num").onclick = ()=> proteger(()=>{
  const v=$("n_var").value, t=$("n_type").value, k=+$("n_bins").value, m=$("n_mes").value;
  if(t==="Histogramme")                 return grHistogramme(v,"o_num",false);
  if(t==="Distribution selon la cible") return grHistogramme(v,"o_num",true);
  if(t==="Effectifs par classe")        return grEffectifs(v,k,"o_num");
  if(t==="Boxplot d'une mesure")        return grBoxplot(v,m,"o_num");
  if(t==="Moyenne d'une mesure")        return grMoyenne(v,k,m,"o_num");
  if(t==="Taux réel par classe")        return grTaux(v,k,"o_num");
  if(t==="Prédit vs observé")           return grCalibration(v,k,"o_num");
  if(t==="Écart réel - prédit")         return grEcart(v,k,"o_num");
  if(t==="Probabilités prédites par classe") return grProbaParClasse(v,k,"o_num");
}, "o_num");

$("b_heat").onclick = ()=> proteger(()=>{
  const v1=$("h_v1").value, v2=$("h_v2").value;
  if(v1===v2) return message("Choisir deux variables différentes.","o_heat");
  dessinerHeatmap(v1,v2,$("h_metrique").value,$("h_mes").value,$("r_den").value,
                  +$("h_bins").value, +$("h_min").value, $("h_n").checked);
}, "o_heat");

$("b_tab").click();
})();
</script>
"""

html = APP.replace("__DATA__", payload)

try:
    displayHTML(html)
except NameError:
    from IPython.display import HTML, display as _d
    _d(HTML(html))

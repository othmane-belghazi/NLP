# Databricks notebook source
# MAGIC %md
# MAGIC # Outil d'analyse actuarielle — version HTML / JavaScript
# MAGIC
# MAGIC Cette version **n'utilise pas ipywidgets** (à l'origine du blocage « Loading widget »).
# MAGIC Les données sont envoyées une fois au navigateur, puis toute l'interactivité
# MAGIC (segmentation, agrégations, ratios, graphiques, heatmap) est calculée en JavaScript.
# MAGIC
# MAGIC **Pré-requis** : `df` et `model` existent déjà. Régler `TARGET` dans la cellule 1,
# MAGIC puis exécuter les cellules 1 à 4 dans l'ordre.

# COMMAND ----------

# MAGIC %md ## 1. Paramètres et préparation des données

# COMMAND ----------

import json

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paramètres fixes
TARGET = "resiliation"      # <-- variable cible (taux réel observé). À adapter une seule fois.
FEATURES = []               # <-- features du modèle ; [] = déduites du modèle

MAX_LIGNES = 100_000        # au-delà, un échantillon aléatoire est envoyé au navigateur
MAX_MODALITES = 100         # les variables texte plus riches que cela sont écartées
SEUIL_DISCRET = 20          # au-delà de ce nb de valeurs, une variable numérique est binée
MIN_OBS_DEFAUT = 30         # effectif minimum d'une cellule de heatmap

if hasattr(df, "toPandas"):
    df = df.toPandas()


def features_modele(d, model, features, target):
    """1) features imposées, 2) sinon celles connues du modèle, 3) sinon numériques hors cible."""
    if features:
        return features
    connues = list(getattr(model, "feature_names_in_", []))
    return connues or [c for c in d.select_dtypes("number").columns if c != target]


# La probabilité prédite est calculée une seule fois, ligne par ligne
FEATURES = features_modele(df, model, FEATURES, TARGET)
X = df[FEATURES]
df["proba_modele"] = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)

print(f"{len(df):,} lignes · cible = {TARGET} · {len(FEATURES)} features · "
      f"proba moyenne = {df['proba_modele'].mean():.2%}".replace(",", " "))

# COMMAND ----------

# MAGIC %md ## 2. Encodage des données pour le navigateur

# COMMAND ----------

def encoder_donnees(d, target, max_lignes=MAX_LIGNES, max_modalites=MAX_MODALITES):
    """Transforme le DataFrame en dictionnaire compact (colonnes numériques ou encodées).

    - numérique  -> liste de nombres arrondis, None pour les manquants
    - catégoriel -> liste de codes entiers + liste des modalités (dictionnaire)
    Les colonnes trop riches en modalités (identifiants, dates libres) sont écartées.
    """
    echantillon = len(d) > max_lignes
    if echantillon:
        d = d.sample(max_lignes, random_state=0)

    cols, ecartees = {}, []
    for nom in d.columns:
        s = d[nom]
        if pd.api.types.is_numeric_dtype(s):
            v = s.astype(float).round(6)
            cols[str(nom)] = {"t": "num", "v": [None if pd.isna(x) else float(x) for x in v]}
        else:
            s = s.astype("object").where(s.notna(), "Manquant").astype(str)
            modalites = sorted(s.unique())
            if len(modalites) > max_modalites:
                ecartees.append(str(nom))
                continue
            index = {m: i for i, m in enumerate(modalites)}
            cols[str(nom)] = {"t": "cat", "l": modalites, "v": [index[x] for x in s]}

    return {
        "n": int(len(d)), "cols": cols, "target": target, "proba": "proba_modele",
        "echantillon": bool(echantillon), "n_total": int(len(df)),
        "seuil_discret": SEUIL_DISCRET, "min_obs": MIN_OBS_DEFAUT,
        "ecartees": ecartees, "modele": type(model).__name__, "features": list(FEATURES),
    }


donnees = encoder_donnees(df, TARGET)
payload = json.dumps(donnees, separators=(",", ":"), allow_nan=False)
print(f"{len(donnees['cols'])} variables encodées · {len(payload) / 1e6:.1f} Mo envoyés au navigateur"
      + (f" · colonnes écartées : {', '.join(donnees['ecartees'])}" if donnees["ecartees"] else ""))

# COMMAND ----------

# MAGIC %md ## 3. Interface HTML / JavaScript

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
  .hdr .sub{font-size:12px;opacity:.8;margin-top:3px}
  .kpis{display:flex;gap:10px;margin:12px 0}
  .kpi{flex:1;background:#fff;border:1px solid var(--gris);border-left:4px solid var(--bleu);
       border-radius:5px;padding:10px 14px}
  .kpi .lab{font-size:11px;color:var(--doux);text-transform:uppercase;letter-spacing:.5px}
  .kpi .val{font-size:20px;font-weight:600;margin-top:2px}
  .tabs{display:flex;gap:4px;border-bottom:2px solid var(--clair);margin-bottom:14px}
  .tab{padding:9px 20px;cursor:pointer;font-weight:600;font-size:13px;color:var(--doux);
       border:1px solid transparent;border-bottom:none;border-radius:5px 5px 0 0;margin-bottom:-2px}
  .tab:hover{background:var(--pale)}
  .tab.on{color:var(--navy);background:#fff;border-color:var(--clair);border-bottom:2px solid #fff}
  .panel{display:none} .panel.on{display:block}
  .row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start}
  .card{border:1px solid var(--gris);border-radius:6px;padding:12px;min-width:230px}
  .card h3{margin:0 0 8px;font-size:11.5px;font-weight:600;color:var(--navy);
           text-transform:uppercase;letter-spacing:.4px;border-bottom:2px solid var(--clair);padding-bottom:5px}
  label{display:block;font-size:11.5px;color:var(--doux);margin:8px 0 3px}
  select,input[type=number]{width:100%;padding:5px 7px;border:1px solid var(--gris);border-radius:4px;
       font-family:inherit;font-size:12.5px;color:var(--texte);background:#fff}
  select[multiple]{height:150px}
  select:focus,input:focus{outline:none;border-color:var(--bleu)}
  .chk{display:flex;align-items:center;gap:6px;margin-top:10px;font-size:12px;color:var(--texte)}
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
  .chart{margin-top:10px;min-height:440px}
</style>

<div class="hdr">
  <h1>Outil d'analyse actuarielle</h1>
  <div class="sub" id="sub"></div>
</div>
<div class="kpis" id="kpis"></div>

<div class="tabs">
  <div class="tab on" data-p="p1">Segmentation</div>
  <div class="tab" data-p="p2">Ratios</div>
  <div class="tab" data-p="p3">Visualisation</div>
  <div class="tab" data-p="p4">Heatmap</div>
</div>

<!-- ---------------------------------------------------------------- segmentation -->
<div class="panel on" id="p1">
  <div class="row">
    <div class="card"><h3>1 · Variables de segmentation</h3>
      <select id="s_seg" multiple></select>
      <label>Nombre de classes (variables continues)</label>
      <input type="number" id="s_bins" value="5" min="2" max="20">
    </div>
    <div class="card"><h3>2 · Variables à agréger</h3><select id="s_aggv" multiple></select>
      <div class="note">Ctrl+clic pour une sélection multiple.</div></div>
    <div class="card"><h3>3 · Fonctions d'agrégation</h3><select id="s_aggf" multiple></select></div>
  </div>
  <div class="btns"><button class="p" id="b_tab">Générer le tableau</button>
    <button id="b_csv1">Exporter en CSV</button></div>
  <div id="o_tab"></div>
</div>

<!-- ---------------------------------------------------------------- ratios -->
<div class="panel" id="p2">
  <div class="row">
    <div class="card" style="min-width:300px"><h3>Ratio = Somme(num.) / Somme(dén.)</h3>
      <label>Numérateur</label><select id="r_num"></select>
      <label>Dénominateur</label><select id="r_den"></select>
      <div class="note">Les lignes dont le numérateur ou le dénominateur est manquant sont exclues
        et comptabilisées. Un dénominateur nul ou négatif ne produit pas de ratio.</div>
    </div>
    <div class="card"><h3>Segmentation (optionnelle)</h3>
      <select id="r_seg" multiple></select>
      <label>Nombre de classes</label><input type="number" id="r_bins" value="5" min="2" max="20">
    </div>
  </div>
  <div class="btns"><button class="p" id="b_ratio">Calculer le ratio</button>
    <button id="b_csv2">Exporter en CSV</button></div>
  <div id="o_ratio"></div>
</div>

<!-- ---------------------------------------------------------------- visualisation -->
<div class="panel" id="p3">
  <div class="row">
    <div class="card"><h3>Variable analysée</h3><select id="g_var"></select>
      <label>Nombre de classes</label><input type="number" id="g_bins" value="10" min="2" max="30"></div>
    <div class="card" style="min-width:320px"><h3>Type de visualisation</h3><select id="g_type"></select>
      <div class="note">Graphiques interactifs : zoom, survol, export PNG.</div></div>
  </div>
  <div class="btns"><button class="p" id="b_graph">Afficher le graphique</button></div>
  <div id="o_graph" class="chart"></div>
</div>

<!-- ---------------------------------------------------------------- heatmap -->
<div class="panel" id="p4">
  <div class="row">
    <div class="card"><h3>Croisement</h3>
      <label>Variable 1 (lignes)</label><select id="h_v1"></select>
      <label>Variable 2 (colonnes)</label><select id="h_v2"></select>
      <label>Nombre de classes</label><input type="number" id="h_bins" value="5" min="2" max="12"></div>
    <div class="card" style="min-width:320px"><h3>Métrique et lisibilité</h3>
      <label>Métrique affichée</label><select id="h_metrique"></select>
      <label>Effectif minimum par cellule</label><input type="number" id="h_min" value="30" min="0" step="10">
      <div class="chk"><input type="checkbox" id="h_n" checked><span>Afficher les effectifs</span></div>
      <div class="note">Les cellules sous le seuil sont laissées vides : trop peu d'observations
        pour être interprétées.</div></div>
  </div>
  <div class="btns"><button class="p" id="b_heat">Afficher la heatmap</button></div>
  <div id="o_heat" class="chart"></div>
</div>
</div>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
(function(){
"use strict";
const D = __DATA__;                                  // données injectées depuis Python
const N = D.n, TARGET = D.target, PROBA = D.proba;
const NAVY="#14476B", BLEU="#2E7DAF", ROUGE="#B3261E", VERT="#1E7B5E", GRIS="#E3E8ED";
const ECHELLE=[[0,"#F5F9FC"],[.25,"#DCEAF4"],[.5,"#9EC6E0"],[.75,"#2E7DAF"],[1,"#14476B"]];
const $ = id => document.getElementById(id);

/* ============================ 1. Utilitaires ============================ */
const noms      = Object.keys(D.cols);
const estNum    = c => D.cols[c].t === "num";
const numeriques= noms.filter(estNum);
const fmtInt    = v => v==null||isNaN(v) ? "–" : Math.round(v).toLocaleString("fr-FR");
const fmtPct    = (v,d=2) => v==null||isNaN(v) ? "–" : (v*100).toFixed(d).replace(".",",")+" %";
const fmtNum    = (v,d=4) => v==null||isNaN(v) ? "–" :
                  v.toLocaleString("fr-FR",{minimumFractionDigits:d,maximumFractionDigits:d});

function quantile(triees, p){                        // interpolation linéaire, comme numpy
  if(!triees.length) return NaN;
  const h = (triees.length-1)*p, b = Math.floor(h);
  return triees[b] + (h-b)*((triees[b+1]!==undefined?triees[b+1]:triees[b]) - triees[b]);
}
function arrondiLisible(x, etendue){
  const d = etendue>=100 ? 0 : etendue>=10 ? 1 : etendue>=1 ? 2 : 4;
  return x.toLocaleString("fr-FR",{minimumFractionDigits:d,maximumFractionDigits:d});
}

/* ============================ 2. Découpage en classes ============================ */
/* Renvoie {labels:[...], codes:Int32Array} ; le code pointe vers l'indice du label.
   - catégoriel                     -> une classe par modalité
   - numérique peu de valeurs       -> une classe par valeur
   - numérique continu              -> classes de quantiles (équivalent de qcut)          */
function decouper(nom, nbins){
  const c = D.cols[nom];
  if(c.t === "cat") return {labels:c.l.slice(), codes:Int32Array.from(c.v)};

  const v = c.v, distinctes = new Set();
  for(let i=0;i<N && distinctes.size<=D.seuil_discret;i++) if(v[i]!=null) distinctes.add(v[i]);

  if(distinctes.size <= D.seuil_discret){            // variable discrète : valeurs telles quelles
    const vals = [...distinctes].sort((a,b)=>a-b);
    const labels = vals.map(x=>String(x)), index = new Map(vals.map((x,i)=>[x,i]));
    labels.push("Manquant");
    const codes = new Int32Array(N);
    for(let i=0;i<N;i++) codes[i] = v[i]==null ? labels.length-1 : index.get(v[i]);
    return {labels, codes};
  }

  const triees = []; for(let i=0;i<N;i++) if(v[i]!=null) triees.push(v[i]);
  triees.sort((a,b)=>a-b);
  let bornes = [];                                   // bornes de quantiles, doublons supprimés
  for(let k=0;k<=nbins;k++){
    const q = quantile(triees, k/nbins);
    if(!bornes.length || q > bornes[bornes.length-1]) bornes.push(q);
  }
  if(bornes.length < 2) bornes = [triees[0], triees[triees.length-1]+1e-9];
  const etendue = bornes[bornes.length-1]-bornes[0];
  const labels = [];
  for(let k=1;k<bornes.length;k++)
    labels.push("]"+arrondiLisible(bornes[k-1],etendue)+" ; "+arrondiLisible(bornes[k],etendue)+"]");
  labels.push("Manquant");
  const codes = new Int32Array(N);
  for(let i=0;i<N;i++){
    if(v[i]==null){ codes[i]=labels.length-1; continue; }
    let k=1; while(k<bornes.length-1 && v[i]>bornes[k]) k++;   // classe = première borne atteinte
    codes[i]=k-1;
  }
  return {labels, codes};
}

/* ============================ 3. Groupement et agrégations ============================ */
function grouper(decoupages){
  const map = new Map();
  for(let i=0;i<N;i++){
    let cle = "";
    for(const d of decoupages) cle += d.codes[i] + "|";
    let g = map.get(cle);
    if(!g){ g = {codes:decoupages.map(d=>d.codes[i]),
                 labels:decoupages.map(d=>d.labels[d.codes[i]]), idx:[]}; map.set(cle,g); }
    g.idx.push(i);
  }
  return [...map.values()].sort((a,b)=>{
    for(let k=0;k<a.codes.length;k++) if(a.codes[k]!==b.codes[k]) return a.codes[k]-b.codes[k];
    return 0;
  });
}

const AGGS = {
  moyenne:   v => v.reduce((s,x)=>s+x,0)/v.length,
  mediane:   v => quantile(v.slice().sort((a,b)=>a-b), .5),
  somme:     v => v.reduce((s,x)=>s+x,0),
  min:       v => Math.min(...v),
  max:       v => Math.max(...v),
  nb_obs:    v => v.length,
  ecart_type:v => { const m=v.reduce((s,x)=>s+x,0)/v.length;
                    return Math.sqrt(v.reduce((s,x)=>s+(x-m)*(x-m),0)/Math.max(1,v.length-1)); },
  p90:       v => quantile(v.slice().sort((a,b)=>a-b), .9),
};

function valeurs(idx, nom){                          // valeurs non manquantes d'un groupe
  const v = D.cols[nom].v, out = [];
  for(const i of idx) if(v[i]!=null) out.push(v[i]);
  return out;
}
function agreger(idx, nom, fonction){
  const v = valeurs(idx, nom);
  if(fonction === "nb_obs") return v.length;
  return v.length ? AGGS[fonction](v) : NaN;
}
const moyenne = (idx, nom) => agreger(idx, nom, "moyenne");

/* ============================ 4. Tableau de segmentation ============================ */
function tableauSegmentation(segVars, nbins, aggVars, aggFuncs){
  const decoupages = segVars.map(v=>decouper(v, nbins));
  const groupes = grouper(decoupages);
  const tous = Array.from({length:N}, (_,i)=>i);
  const cible = moyenne(tous, TARGET);

  const colonnes = [...segVars, "effectif", "poids_pct", "taux_reel", "proba_modele",
                    "ecart_reel_modele", "indice_vs_global"];
  aggVars.forEach(v => aggFuncs.forEach(f => colonnes.push(v+"_"+f)));

  const ligne = (g, libelles, total) => {
    const reel = moyenne(g.idx, TARGET), predit = moyenne(g.idx, PROBA);
    const r = {_total:total};
    segVars.forEach((v,k)=> r[v]=libelles[k]);
    r.effectif = g.idx.length;
    r.poids_pct = g.idx.length/N;
    r.taux_reel = reel;
    r.proba_modele = predit;
    r.ecart_reel_modele = reel - predit;
    r.indice_vs_global = cible ? 100*reel/cible : NaN;
    aggVars.forEach(v => aggFuncs.forEach(f => r[v+"_"+f] = agreger(g.idx, v, f)));
    return r;
  };

  const lignes = groupes.map(g => ligne(g, g.labels, false));
  const libTotal = segVars.map((_,k)=> k===0 ? "TOTAL" : "");
  lignes.push(ligne({idx:tous}, libTotal, true));
  return {colonnes, lignes, seg:segVars};
}

/* ============================ 5. Ratio actuariel ============================ */
function tableauRatio(num, den, segVars, nbins){
  const vn = D.cols[num].v, vd = D.cols[den].v;
  const decoupages = segVars.length ? segVars.map(v=>decouper(v,nbins))
                                    : [{labels:["Ensemble"], codes:new Int32Array(N)}];
  const seg = segVars.length ? segVars : ["perimetre"];
  const groupes = grouper(decoupages);

  const ligne = (idx, libelles, total) => {
    let sn=0, sd=0, valides=0, exclues=0;
    for(const i of idx){
      if(vn[i]==null || vd[i]==null){ exclues++; continue; }   // ligne incomplète : exclue
      sn += vn[i]; sd += vd[i]; valides++;
    }
    const r = {_total:total};
    seg.forEach((v,k)=> r[v]=libelles[k]);
    r.effectif = idx.length;
    r.obs_exclues = exclues;
    r["somme_"+num] = valides ? sn : NaN;
    r["somme_"+den] = valides ? sd : NaN;
    r.ratio = sd > 0 ? sn/sd : NaN;                            // dénominateur nul : pas de ratio
    r.taux_reel = moyenne(idx, TARGET);
    r.proba_modele = moyenne(idx, PROBA);
    return r;
  };

  const lignes = groupes.map(g => ligne(g.idx, g.labels, false));
  const tous = Array.from({length:N}, (_,i)=>i);
  if(segVars.length) lignes.push(ligne(tous, seg.map((_,k)=>k===0?"TOTAL":""), true));

  const global = lignes[lignes.length-1].ratio;                // indice base 100 = ratio d'ensemble
  lignes.forEach(l => l.indice_base100 = (global && isFinite(global)) ? 100*l.ratio/global : NaN);

  const colonnes = [...seg, "effectif", "obs_exclues", "somme_"+num, "somme_"+den,
                    "ratio", "indice_base100", "taux_reel", "proba_modele"];
  return {colonnes, lignes, seg, exclues:lignes[lignes.length-1].obs_exclues};
}

/* ============================ 6. Rendu des tableaux ============================ */
const PCT  = new Set(["poids_pct","taux_reel","proba_modele"]);
const ENT  = new Set(["effectif","obs_exclues"]);
const IND  = new Set(["indice_vs_global","indice_base100"]);
const DEGR = new Set(["taux_reel","proba_modele","ratio"]);

function formater(col, v){
  if(col === "ecart_reel_modele") return v==null||isNaN(v) ? "–" :
        (v>=0?"+":"") + fmtPct(v).replace("+","");
  if(PCT.has(col)) return fmtPct(v, col==="poids_pct"?1:2);
  if(ENT.has(col)) return fmtInt(v);
  if(IND.has(col)) return v==null||isNaN(v) ? "–" : Math.round(v).toLocaleString("fr-FR");
  return typeof v === "number" ? fmtNum(v) : (v==null?"–":v);
}
function couleurFond(t){                             // dégradé blanc -> bleu foncé
  const a=[245,249,252], b=[20,71,107];
  const c=a.map((x,i)=>Math.round(x+(b[i]-x)*t));
  return {bg:`rgb(${c.join(",")})`, fg: t>.65 ? "#fff" : "inherit"};
}

function rendreTableau(res, cible){
  const {colonnes, lignes, seg} = res;
  const bornes = {};
  DEGR.forEach(c => {
    if(!colonnes.includes(c)) return;
    const v = lignes.filter(l=>!l._total).map(l=>l[c]).filter(x=>x!=null&&!isNaN(x));
    if(v.length) bornes[c] = [Math.min(...v), Math.max(...v)];
  });

  let h = '<div class="tw"><table><thead><tr>';
  colonnes.forEach((c,i)=> h += `<th class="${seg.includes(c)?'seg':''}" data-c="${i}"
      title="Cliquer pour trier">${c}</th>`);
  h += '</tr></thead><tbody>';
  lignes.forEach(l => {
    h += `<tr class="${l._total?'tot':''}">`;
    colonnes.forEach(c => {
      let style = "";
      if(bornes[c] && !l._total && l[c]!=null && !isNaN(l[c])){
        const [mn,mx] = bornes[c], t = mx>mn ? (l[c]-mn)/(mx-mn) : .5;
        const col = couleurFond(t); style = `background:${col.bg};color:${col.fg}`;
      }
      if(c === "ecart_reel_modele" && l[c]!=null && !isNaN(l[c]))
        style = `color:${l[c]>0?ROUGE:VERT};font-weight:600`;
      h += `<td class="${seg.includes(c)?'seg':''}" style="${style}">${formater(c,l[c])}</td>`;
    });
    h += '</tr>';
  });
  $(cible).innerHTML = h + '</tbody></table></div>';

  let sens = 1, dernier = -1;                        // tri au clic sur l'en-tête
  $(cible).querySelectorAll("th").forEach(th => th.onclick = () => {
    const i = +th.dataset.c, c = colonnes[i];
    sens = (i === dernier) ? -sens : 1; dernier = i;
    const corps = lignes.filter(l=>!l._total), total = lignes.filter(l=>l._total);
    corps.sort((a,b)=>{
      const x=a[c], y=b[c];
      if(typeof x === "number" && typeof y === "number")
        return sens*((isNaN(x)?-Infinity:x)-(isNaN(y)?-Infinity:y));
      return sens*String(x).localeCompare(String(y),"fr");
    });
    rendreTableau({colonnes, lignes:[...corps,...total], seg}, cible);
  });
}

/* ============================ 7. Export CSV ============================ */
function exporterCSV(res, fichier){
  if(!res) return message("Générer d'abord un tableau.", res===undefined?"o_tab":"o_tab");
  const lignes = [res.colonnes.join(";")];
  res.lignes.forEach(l => lignes.push(res.colonnes
      .map(c => { const v=l[c]; return typeof v==="number"
          ? (isNaN(v)?"":String(v).replace(".",",")) : (v==null?"":v); }).join(";")));
  const blob = new Blob(["\ufeff"+lignes.join("\n")], {type:"text/csv;charset=utf-8;"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = fichier;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
}

/* ============================ 8. Graphiques ============================ */
const MISE_EN_PAGE = {
  font:{family:"Segoe UI, Arial, sans-serif", size:12, color:"#1F2A37"},
  plot_bgcolor:"#fff", paper_bgcolor:"#fff", height:450,
  margin:{l:65,r:30,t:60,b:70},
  xaxis:{gridcolor:GRIS,zerolinecolor:GRIS}, yaxis:{gridcolor:GRIS,zerolinecolor:GRIS},
  title:{font:{size:15,color:NAVY}},
};
const CONFIG = {displayModeBar:true, displaylogo:false, responsive:true,
                modeBarButtonsToRemove:["lasso2d","select2d"]};

function tracer(cible, traces, layout){
  if(typeof Plotly === "undefined"){
    $(cible).innerHTML = '<div class="msg">Plotly n\'a pas pu être chargé depuis le CDN '
      + '(réseau restreint). Les tableaux restent disponibles.</div>'; return;
  }
  Plotly.newPlot(cible, traces, Object.assign({}, MISE_EN_PAGE, layout), CONFIG);
}

const GRAPHIQUES = ["Histogramme","Distribution selon la cible","Boxplot par classe",
                    "Effectifs par modalité","Prédit vs observé par classe",
                    "Distribution des probabilités prédites"];

function dessinerGraphique(nom, type, nbins){
  const d = decouper(nom, nbins), groupes = grouper([d]);
  const libelles = groupes.map(g=>g.labels[0]);
  const num = estNum(nom), v = D.cols[nom].v;

  if(type === "Histogramme"){
    if(num) return tracer("o_graph",
      [{type:"histogram", x:v, nbinsx:nbins, marker:{color:NAVY, line:{color:"#fff",width:1}}}],
      {title:"Distribution de "+nom, xaxis:{title:nom}, yaxis:{title:"Effectif"}});
    return tracer("o_graph",
      [{type:"bar", x:libelles, y:groupes.map(g=>g.idx.length), marker:{color:NAVY}}],
      {title:"Distribution de "+nom, xaxis:{title:nom,tickangle:-30}, yaxis:{title:"Effectif"}});
  }

  if(type === "Distribution selon la cible"){
    const t = D.cols[TARGET].v, a=[], b=[];
    for(let i=0;i<N;i++){ if(v[i]==null) continue; (t[i]===1?a:b).push(v[i]); }
    if(num) return tracer("o_graph", [
      {type:"histogram", x:b, name:"Non résiliés", opacity:.65, nbinsx:nbins,
       histnorm:"percent", marker:{color:BLEU}},
      {type:"histogram", x:a, name:"Résiliés", opacity:.65, nbinsx:nbins,
       histnorm:"percent", marker:{color:NAVY}}],
      {barmode:"overlay", title:"Distribution de "+nom+" selon "+TARGET,
       xaxis:{title:nom}, yaxis:{title:"% de la population",ticksuffix:" %"},
       legend:{orientation:"h",y:1.12}});
    const part = groupes.map(g=>{ let s=0; for(const i of g.idx) if(t[i]===1) s++; return s/g.idx.length; });
    return tracer("o_graph",
      [{type:"bar", x:libelles, y:part, marker:{color:NAVY},
        text:part.map(p=>fmtPct(p,1)), textposition:"outside"}],
      {title:"Taux de "+TARGET+" par modalité de "+nom,
       xaxis:{title:nom,tickangle:-30}, yaxis:{title:"Taux",tickformat:".1%"}});
  }

  if(type === "Boxplot par classe"){
    if(!num) return message("Le boxplot nécessite une variable numérique.","o_graph");
    return tracer("o_graph", groupes.map(g=>({
        type:"box", y:g.idx.map(i=>v[i]).filter(x=>x!=null), name:g.labels[0],
        marker:{color:BLEU}, line:{color:NAVY}, boxpoints:false, showlegend:false})),
      {title:"Dispersion de "+nom+" par classe", xaxis:{tickangle:-30}, yaxis:{title:nom}});
  }

  if(type === "Effectifs par modalité"){
    const eff = groupes.map(g=>g.idx.length);
    return tracer("o_graph",
      [{type:"bar", x:libelles, y:eff, marker:{color:NAVY},
        text:eff.map(fmtInt), textposition:"outside"}],
      {title:"Effectifs par modalité de "+nom,
       xaxis:{title:nom,tickangle:-30}, yaxis:{title:"Effectif"}});
  }

  if(type === "Prédit vs observé par classe"){
    const reel = groupes.map(g=>moyenne(g.idx,TARGET));
    const predit = groupes.map(g=>moyenne(g.idx,PROBA));
    const eff = groupes.map(g=>g.idx.length);
    return tracer("o_graph", [
      {type:"bar", x:libelles, y:reel, name:"Taux réel observé", marker:{color:BLEU},
       customdata:eff, hovertemplate:"%{x}<br>Réel : %{y:.2%}<br>n = %{customdata:,}<extra></extra>"},
      {type:"scatter", mode:"lines+markers", x:libelles, y:predit, name:"Probabilité prédite",
       line:{color:NAVY,width:3}, marker:{size:8},
       hovertemplate:"%{x}<br>Prédit : %{y:.2%}<extra></extra>"}],
      {title:"Calibration du modèle par classe de "+nom,
       xaxis:{title:nom,tickangle:-30}, yaxis:{title:"Taux",tickformat:".1%"},
       legend:{orientation:"h",y:1.12}});
  }

  const p = D.cols[PROBA].v;                        // Distribution des probabilités prédites
  return tracer("o_graph", groupes.slice(0,8).map(g=>({
      type:"histogram", x:g.idx.map(i=>p[i]), name:g.labels[0], opacity:.6,
      nbinsx:40, histnorm:"percent"})),
    {barmode:"overlay", title:"Probabilités prédites par classe de "+nom,
     xaxis:{title:"Probabilité prédite",tickformat:".0%"},
     yaxis:{title:"% de la classe",ticksuffix:" %"}});
}

/* ============================ 9. Heatmap ============================ */
const METRIQUES = {
  "Probabilité moyenne prédite": idx => moyenne(idx, PROBA),
  "Taux réel observé":           idx => moyenne(idx, TARGET),
  "Écart réel - prédit":         idx => moyenne(idx, TARGET) - moyenne(idx, PROBA),
  "Effectif":                    idx => idx.length,
};

function dessinerHeatmap(v1, v2, metrique, nbins, minObs, afficherN){
  const d1 = decouper(v1,nbins), d2 = decouper(v2,nbins);
  const lignes = d1.labels, cols = d2.labels;
  const cellules = new Map();
  for(let i=0;i<N;i++){
    const cle = d1.codes[i]+"|"+d2.codes[i];
    if(!cellules.has(cle)) cellules.set(cle, []);
    cellules.get(cle).push(i);
  }
  const fonction = METRIQUES[metrique], pct = metrique !== "Effectif";
  const z=[], texte=[], eff=[]; let masquees=0;
  lignes.forEach((_,r)=>{
    const zr=[], tr=[], er=[];
    cols.forEach((__,c)=>{
      const idx = cellules.get(r+"|"+c) || [];
      er.push(idx.length);
      if(!idx.length || (metrique!=="Effectif" && idx.length < minObs)){
        if(idx.length) masquees++;
        zr.push(null); tr.push(""); return;
      }
      const val = fonction(idx);
      zr.push(val);
      tr.push((pct?fmtPct(val,1):fmtInt(val))
              + (afficherN && pct ? "<br><span style='font-size:9px'>n="+fmtInt(idx.length)+"</span>" : ""));
    });
    z.push(zr); texte.push(tr); eff.push(er);
  });

  const vides = lignes.map((_,r)=>eff[r].reduce((s,x)=>s+x,0)===0);   // lignes sans observation
  const gardees = lignes.map((l,r)=>r).filter(r=>!vides[r]);
  const zf = gardees.map(r=>z[r]), tf = gardees.map(r=>texte[r]), yf = gardees.map(r=>lignes[r]);

  tracer("o_heat", [{
    type:"heatmap", z:zf, x:cols, y:yf, text:tf, texttemplate:"%{text}",
    textfont:{size:11}, colorscale:ECHELLE, hoverongaps:false,
    colorbar:{title:{text:metrique.split(" ")[0],side:"right"}, tickformat:pct?".1%":","},
    hovertemplate:"%{y} × %{x}<br>"+metrique+" : %{z:.4f}<extra></extra>"}],
    {title:metrique+" — "+v1+" × "+v2,
     xaxis:{title:v2, side:"top", tickangle:-20}, yaxis:{title:v1, autorange:"reversed"},
     height:150+46*yf.length, margin:{l:150,r:30,t:90,b:40}});

  if(masquees) message(masquees+" cellule(s) masquée(s) : moins de "+minObs+" observations.",
                       "o_heat", true);
}

/* ============================ 10. Interface ============================ */
function message(txt, cible, ajouter){
  const html = '<div class="msg">'+txt+'</div>';
  if(ajouter) $(cible).insertAdjacentHTML("beforeend", html); else $(cible).innerHTML = html;
}
function remplir(id, options, defaut){
  $(id).innerHTML = options.map(o=>`<option value="${o}">${o}</option>`).join("");
  if(defaut!==undefined) $(id).value = defaut;
}
const selection = id => [...$(id).selectedOptions].map(o=>o.value);

function proteger(fonction, cible){                  // affiche les erreurs sans casser la page
  try{ fonction(); }
  catch(e){ $(cible).innerHTML = '<div class="err">'+(e && e.stack ? e.stack : e)+'</div>'; }
}

// --- bandeau et indicateurs globaux
const tous = Array.from({length:N},(_,i)=>i);
const tauxGlobal = moyenne(tous,TARGET), probaGlobale = moyenne(tous,PROBA);
const ecart = tauxGlobal - probaGlobale;
$("sub").textContent = "Cible : "+TARGET+" · Modèle : "+D.modele+" · "+D.features.length
  + " variables explicatives" + (D.echantillon
  ? " · échantillon de "+fmtInt(N)+" lignes sur "+fmtInt(D.n_total) : "");
$("kpis").innerHTML = [
  ["Observations", fmtInt(N), BLEU],
  ["Taux réel observé", fmtPct(tauxGlobal), BLEU],
  ["Probabilité moyenne prédite", fmtPct(probaGlobale), BLEU],
  ["Écart réel - prédit", (ecart>=0?"+":"−")+fmtPct(Math.abs(ecart)),
   Math.abs(ecart)>.01 ? ROUGE : VERT],
].map(([l,v,c]) => `<div class="kpi" style="border-left-color:${c}">
   <div class="lab">${l}</div><div class="val">${v}</div></div>`).join("");

// --- alimentation des listes
const segOptions = noms;
remplir("s_seg", segOptions); $("s_seg").selectedIndex = 0;
remplir("s_aggv", numeriques);
remplir("s_aggf", Object.keys(AGGS)); $("s_aggf").selectedIndex = 0;
remplir("r_num", numeriques); remplir("r_den", numeriques);
if(numeriques.length>1) $("r_den").selectedIndex = 1;
remplir("r_seg", segOptions);
remplir("g_var", segOptions); remplir("g_type", GRAPHIQUES);
remplir("h_v1", segOptions); remplir("h_v2", segOptions);
if(segOptions.length>1) $("h_v2").selectedIndex = 1;
remplir("h_metrique", Object.keys(METRIQUES));
$("h_min").value = D.min_obs;

// --- onglets
document.querySelectorAll("#app .tab").forEach(t => t.onclick = () => {
  document.querySelectorAll("#app .tab").forEach(x=>x.classList.remove("on"));
  document.querySelectorAll("#app .panel").forEach(x=>x.classList.remove("on"));
  t.classList.add("on"); $(t.dataset.p).classList.add("on");
});

// --- actions
let resSeg = null, resRatio = null;

$("b_tab").onclick = () => proteger(() => {
  const seg = selection("s_seg");
  if(!seg.length) return message("Sélectionner au moins une variable de segmentation.","o_tab");
  resSeg = tableauSegmentation(seg, +$("s_bins").value, selection("s_aggv"), selection("s_aggf"));
  rendreTableau(resSeg, "o_tab");
}, "o_tab");

$("b_ratio").onclick = () => proteger(() => {
  const num = $("r_num").value, den = $("r_den").value;
  if(num === den) return message("Numérateur et dénominateur doivent être différents.","o_ratio");
  resRatio = tableauRatio(num, den, selection("r_seg"), +$("r_bins").value);
  rendreTableau(resRatio, "o_ratio");
  message("Ratio "+num+" / "+den+" · indice base 100 = ratio d'ensemble"
    + (resRatio.exclues ? " · "+fmtInt(resRatio.exclues)+" observation(s) exclue(s)" : ""),
    "o_ratio", true);
}, "o_ratio");

$("b_graph").onclick = () => proteger(() =>
  dessinerGraphique($("g_var").value, $("g_type").value, +$("g_bins").value), "o_graph");

$("b_heat").onclick = () => proteger(() => {
  if($("h_v1").value === $("h_v2").value)
    return message("Choisir deux variables différentes.","o_heat");
  dessinerHeatmap($("h_v1").value, $("h_v2").value, $("h_metrique").value,
                  +$("h_bins").value, +$("h_min").value, $("h_n").checked);
}, "o_heat");

$("b_csv1").onclick = () => resSeg ? exporterCSV(resSeg,"segmentation.csv")
                                   : message("Générer d'abord un tableau.","o_tab");
$("b_csv2").onclick = () => resRatio ? exporterCSV(resRatio,"ratios.csv")
                                     : message("Calculer d'abord un ratio.","o_ratio");

$("b_tab").click();                                  // premier tableau affiché d'emblée
})();
</script>
"""

html = APP.replace("__DATA__", payload)

try:
    displayHTML(html)                                  # Databricks
except NameError:                                      # Jupyter / autre environnement
    from IPython.display import HTML, display as _d
    _d(HTML(html))

# COMMAND ----------

# MAGIC %md ## 4. Si les graphiques ne s'affichent pas
# MAGIC
# MAGIC L'interface fonctionne sans réseau ; seuls les graphiques utilisent Plotly via un CDN.
# MAGIC Si votre navigateur bloque `cdn.plot.ly`, les tableaux restent disponibles et un message
# MAGIC l'indique dans les onglets graphiques.
# MAGIC
# MAGIC Pour servir Plotly depuis le cluster plutôt que depuis le CDN :
# MAGIC
# MAGIC ```python
# MAGIC import plotly, shutil, os
# MAGIC src = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
# MAGIC shutil.copy(src, "/dbfs/FileStore/plotly.min.js")
# MAGIC # puis remplacer l'URL du CDN par "/files/plotly.min.js" dans la balise <script> ci-dessus
# MAGIC ```

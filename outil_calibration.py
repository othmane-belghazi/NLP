# -*- coding: utf-8 -*-
"""
Outil interactif de calibration des courbes ELR × Prime
=======================================================

Usage dans un Jupyter Notebook :

    from outil_calibration import generer_base_test, afficher_outil, PARAMS_INITIAUX

    df = generer_base_test(8000)          # ou votre propre DataFrame
    afficher_outil(df, PARAMS_INITIAUX, echeance="01/01/2027", annee=2026)

Le DataFrame doit contenir au minimum les colonnes :
    - Population_courbe : 'A' … 'F'
    - ELR               : valeur E de la formule
    - PrimeSS           : valeur P de la formule (sert aussi au tranchage 200→1600)
    - Prime_avant       : prime avant majoration
Colonnes optionnelles utilisées par les fonctions d'ajustement d'exemple :
    - Anciennete, Segment_sensible

Formule :
    Valeur              = (a·log(E) + b) × (l + log(P·m) + n)      (log népérien)
    Majoration initiale = min(max, max(min, Valeur))
    Majoration finale   = ajustements 1→5 appliqués en chaîne
    Nouvelle prime      = Prime_avant × Majoration finale
    Ressource finale    = Σ nouvelles primes / Σ anciennes primes
"""

import json
import html as _html
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Paramètres initiaux par population
# ---------------------------------------------------------------------------

PARAMS_INITIAUX = {
    "A": {"a": 0.080, "b": 1.020, "l": 0.30, "m": 0.0020, "n": 0.65, "min": 0.95, "max": 1.35},
    "B": {"a": 0.060, "b": 1.000, "l": 0.25, "m": 0.0022, "n": 0.70, "min": 0.93, "max": 1.30},
    "C": {"a": 0.100, "b": 1.050, "l": 0.35, "m": 0.0018, "n": 0.60, "min": 0.97, "max": 1.40},
    "D": {"a": 0.050, "b": 0.980, "l": 0.20, "m": 0.0025, "n": 0.75, "min": 0.92, "max": 1.28},
    "E": {"a": 0.090, "b": 1.030, "l": 0.32, "m": 0.0019, "n": 0.62, "min": 0.96, "max": 1.38},
    "F": {"a": 0.070, "b": 1.010, "l": 0.28, "m": 0.0021, "n": 0.68, "min": 0.94, "max": 1.32},
}

POPULATIONS = ["A", "B", "C", "D", "E", "F"]

# Bornes des sliders (min, max, pas)
BORNES_SLIDERS = {
    "a":   (-0.30, 0.30, 0.005),
    "b":   (0.50, 1.50, 0.005),
    "l":   (-1.00, 1.00, 0.01),
    "m":   (0.0005, 0.0060, 0.0001),
    "n":   (-1.00, 1.50, 0.01),
    "min": (0.80, 1.20, 0.005),
    "max": (1.00, 1.80, 0.005),
}

# Tranches de prime : 200 → 1600, pas de 50  (28 tranches)
TRANCHE_MIN, TRANCHE_MAX, TRANCHE_PAS = 200, 1600, 50


# ---------------------------------------------------------------------------
# 2. Fonctions d'ajustement (référence Python)
#    >>> Remplacez le corps de ces 5 fonctions par les vôtres. <<<
#    Chaque fonction reçoit la majoration courante et la ligne du contrat,
#    et renvoie la majoration ajustée. Elles sont appliquées dans l'ordre 1→5.
#    IMPORTANT : le miroir JavaScript (AJUSTEMENTS_JS plus bas) doit rester
#    identique pour que l'outil interactif reproduise exactement vos règles.
# ---------------------------------------------------------------------------

def ajustement_1_segment_sensible(maj, contrat):
    """Plafonne la majoration à 1.10 pour les segments sensibles."""
    if contrat.get("Segment_sensible", 0) == 1:
        return min(maj, 1.10)
    return maj


def ajustement_2_anciennete(maj, contrat):
    """Réduit de 20 % l'écart à 1 pour les contrats d'ancienneté ≥ 10 ans."""
    if contrat.get("Anciennete", 0) >= 10:
        return 1.0 + (maj - 1.0) * 0.80
    return maj


def ajustement_3_sinistralite(maj, contrat):
    """Ajoute +0.02 si l'ELR dépasse 1.20."""
    if contrat["ELR"] > 1.20:
        return maj + 0.02
    return maj


def ajustement_4_petite_prime(maj, contrat):
    """Plafonne à 1.15 les contrats dont la PrimeSS est inférieure à 300."""
    if contrat["PrimeSS"] < 300:
        return min(maj, 1.15)
    return maj


def ajustement_5_bornage_final(maj, contrat):
    """Bornage réglementaire final [0.85 ; 1.60]."""
    return min(1.60, max(0.85, maj))


AJUSTEMENTS = [
    ajustement_1_segment_sensible,
    ajustement_2_anciennete,
    ajustement_3_sinistralite,
    ajustement_4_petite_prime,
    ajustement_5_bornage_final,
]

# Miroir JavaScript des 5 ajustements — à maintenir identique aux fonctions
# Python ci-dessus. `maj` est la majoration courante, `c` le contrat
# ({elr, prime, primeAvant, anc, seg}).
AJUSTEMENTS_JS = r"""
const AJUSTEMENTS = [
  // 1. Segment sensible : plafond 1.10
  (maj, c) => c.seg === 1 ? Math.min(maj, 1.10) : maj,
  // 2. Ancienneté ≥ 10 ans : écart à 1 réduit de 20 %
  (maj, c) => c.anc >= 10 ? 1.0 + (maj - 1.0) * 0.80 : maj,
  // 3. ELR > 1.20 : +0.02
  (maj, c) => c.elr > 1.20 ? maj + 0.02 : maj,
  // 4. PrimeSS < 300 : plafond 1.15
  (maj, c) => c.prime < 300 ? Math.min(maj, 1.15) : maj,
  // 5. Bornage final [0.85 ; 1.60]
  (maj, c) => Math.min(1.60, Math.max(0.85, maj)),
];
"""


# ---------------------------------------------------------------------------
# 3. Base de test
# ---------------------------------------------------------------------------

def generer_base_test(n=8000, graine=42):
    """Génère une base de contrats synthétique pour tester l'outil."""
    rng = np.random.default_rng(graine)
    poids = [0.24, 0.20, 0.17, 0.15, 0.13, 0.11]
    pop = rng.choice(POPULATIONS, size=n, p=poids)

    prime = np.exp(rng.normal(6.35, 0.45, n))                 # ~ 400–1200
    prime = np.clip(prime, TRANCHE_MIN, TRANCHE_MAX - 1).round(2)

    elr = np.exp(rng.normal(-0.12, 0.28, n))                  # ~ 0.55–1.45
    elr = np.clip(elr, 0.30, 2.20).round(4)

    prime_avant = (prime * rng.uniform(0.95, 1.05, n)).round(2)
    anciennete = rng.integers(0, 26, n)
    segment = (rng.random(n) < 0.12).astype(int)

    return pd.DataFrame({
        "Num_contrat": [f"CT{100000 + i}" for i in range(n)],
        "Population_courbe": pop,
        "ELR": elr,
        "PrimeSS": prime,
        "Prime_avant": prime_avant,
        "Anciennete": anciennete,
        "Segment_sensible": segment,
    })


# ---------------------------------------------------------------------------
# 4. Calcul de référence Python (vectorisé) — pour validation hors outil
# ---------------------------------------------------------------------------

def calculer_resultats(df, params):
    """Renvoie une copie du df avec Majoration_finale et Nouvelle_prime,
    ainsi que la ressource finale globale."""
    out = df.copy()
    maj = np.empty(len(df))
    for popu, p in params.items():
        masque = (df["Population_courbe"] == popu).to_numpy()
        if not masque.any():
            continue
        E = df.loc[masque, "ELR"].to_numpy(float)
        P = df.loc[masque, "PrimeSS"].to_numpy(float)
        val = (p["a"] * np.log(E) + p["b"]) * (p["l"] + np.log(P * p["m"]) + p["n"])
        maj[masque] = np.minimum(p["max"], np.maximum(p["min"], val))
    out["Majoration_initiale"] = maj

    finales = []
    for i, ligne in enumerate(df.to_dict("records")):
        v = maj[i]
        for f in AJUSTEMENTS:
            v = f(v, ligne)
        finales.append(v)
    out["Majoration_finale"] = finales
    out["Nouvelle_prime"] = out["Prime_avant"] * out["Majoration_finale"]
    ressource = out["Nouvelle_prime"].sum() / out["Prime_avant"].sum()
    return out, ressource


# ---------------------------------------------------------------------------
# 5. Construction de l'outil HTML interactif
# ---------------------------------------------------------------------------

def construire_html(df, params=None, echeance="01/01/2027", annee=2026):
    """Construit le document HTML autonome de l'outil (tout le recalcul est
    fait côté navigateur pour une mise à jour fluide des sliders)."""
    params = params or PARAMS_INITIAUX

    colonnes_requises = {"Population_courbe", "ELR", "PrimeSS", "Prime_avant"}
    manquantes = colonnes_requises - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans la base : {sorted(manquantes)}")

    donnees = {
        "pop":   df["Population_courbe"].astype(str).tolist(),
        "elr":   [round(float(x), 4) for x in df["ELR"]],
        "prime": [round(float(x), 2) for x in df["PrimeSS"]],
        "pav":   [round(float(x), 2) for x in df["Prime_avant"]],
        "anc":   [int(x) for x in df["Anciennete"]] if "Anciennete" in df else [0] * len(df),
        "seg":   [int(x) for x in df["Segment_sensible"]] if "Segment_sensible" in df else [0] * len(df),
    }

    tranches = list(range(TRANCHE_MIN, TRANCHE_MAX, TRANCHE_PAS))
    labels = [f"{t}–{t + TRANCHE_PAS}" for t in tranches]

    html_doc = (
        _TEMPLATE
        .replace("__DATA__", json.dumps(donnees, separators=(",", ":")))
        .replace("__PARAMS__", json.dumps(params))
        .replace("__BORNES__", json.dumps(BORNES_SLIDERS))
        .replace("__LABELS__", json.dumps(labels))
        .replace("__TR_MIN__", str(TRANCHE_MIN))
        .replace("__TR_PAS__", str(TRANCHE_PAS))
        .replace("__NB_TR__", str(len(tranches)))
        .replace("__ECHEANCE__", _html.escape(str(echeance)))
        .replace("__ANNEE__", _html.escape(str(annee)))
        .replace("__VOLUME__", f"{len(df):,}".replace(",", " "))
        .replace("__AJUSTEMENTS_JS__", AJUSTEMENTS_JS)
    )
    return html_doc


def afficher_outil(df=None, params=None, echeance="01/01/2027", annee=2026, hauteur=920):
    """Affiche l'outil directement dans une cellule Jupyter (iframe isolée)."""
    from IPython.display import HTML
    if df is None:
        df = generer_base_test()
    doc = construire_html(df, params, echeance, annee)
    iframe = (
        f'<iframe srcdoc="{_html.escape(doc, quote=True)}" '
        f'style="width:100%;height:{hauteur}px;border:none;border-radius:10px;" '
        f'sandbox="allow-scripts"></iframe>'
    )
    return HTML(iframe)


def sauvegarder_html(chemin, df=None, params=None, echeance="01/01/2027", annee=2026):
    """Sauvegarde l'outil dans un fichier HTML autonome (ouvrable au navigateur)."""
    if df is None:
        df = generer_base_test()
    doc = construire_html(df, params, echeance, annee)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(doc)
    return chemin


# ---------------------------------------------------------------------------
# 6. Template HTML / CSS / JS
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibration ELR × Prime</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --papier:#EDF0F3; --panneau:#FFFFFF; --encre:#16232E; --sourdine:#5F6E7B;
  --trait:#DCE3E9; --accent:#0E7C86; --accent-fonce:#0A5A62;
  --reference:#C7802B; --barre:#C6D0D9; --pastille:#F2F5F7;
  --mono:'IBM Plex Mono','SFMono-Regular',Consolas,'Liberation Mono',monospace;
  --sans:'IBM Plex Sans','Segoe UI',system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--papier);color:var(--encre);font-family:var(--sans);
  font-size:13.5px;line-height:1.45;padding:14px}
.cadre{max-width:1240px;margin:0 auto}

/* ---------- bandeau ---------- */
.bandeau{display:flex;flex-wrap:wrap;align-items:flex-end;gap:16px;
  border-bottom:2px solid var(--encre);padding-bottom:12px;margin-bottom:14px}
.bandeau h1{font-size:19px;font-weight:600;letter-spacing:.2px}
.bandeau h1 span{color:var(--accent)}
.sous-titre{color:var(--sourdine);font-size:12px;margin-top:2px}
.faits{display:flex;gap:10px;margin-left:auto;flex-wrap:wrap}
.fait{background:var(--panneau);border:1px solid var(--trait);border-radius:8px;
  padding:6px 12px;min-width:96px}
.fait .etq{font-size:10.5px;text-transform:uppercase;letter-spacing:.8px;color:var(--sourdine)}
.fait .val{font-family:var(--mono);font-size:14.5px;font-weight:600;margin-top:1px}

/* ---------- grille ---------- */
.grille{display:grid;grid-template-columns:322px 1fr;gap:14px}
@media(max-width:960px){.grille{grid-template-columns:1fr}}
.panneau{background:var(--panneau);border:1px solid var(--trait);border-radius:10px;padding:14px}

/* ---------- formule (élément signature) ---------- */
.formule{background:var(--encre);color:#EAF1F5;border-radius:10px;
  padding:13px 14px;font-family:var(--mono);font-size:12.6px;line-height:1.9;margin-bottom:13px}
.formule .titre-f{font-family:var(--sans);font-size:10.5px;letter-spacing:1px;
  text-transform:uppercase;color:#8FA6B5;margin-bottom:6px}
.formule b{font-weight:600;color:#FFD9A0}
.co{display:inline-block;border-radius:4px;padding:0 4px;font-weight:600;
  background:rgba(14,124,134,.35);color:#7FE0E8;transition:background .25s}
.co.flash{background:#0E7C86;color:#fff}
.formule .clamp{color:#B9C9D4}

/* ---------- population ---------- */
.section-titre{font-size:10.5px;text-transform:uppercase;letter-spacing:1px;
  color:var(--sourdine);margin:2px 0 8px}
.pops{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;margin-bottom:13px}
.pop{border:1px solid var(--trait);background:var(--pastille);border-radius:7px;
  padding:7px 0;text-align:center;font-family:var(--mono);font-weight:600;font-size:14px;
  cursor:pointer;transition:all .15s;position:relative}
.pop:hover{border-color:var(--accent)}
.pop.actif{background:var(--accent);border-color:var(--accent-fonce);color:#fff}
.pop .n{display:block;font-family:var(--sans);font-weight:400;font-size:9.5px;
  color:var(--sourdine);margin-top:1px}
.pop.actif .n{color:#CFEDEF}
.pop.modif::after{content:'';position:absolute;top:4px;right:5px;width:6px;height:6px;
  border-radius:50%;background:var(--reference)}

/* ---------- sliders ---------- */
.curseur{display:grid;grid-template-columns:34px 1fr 76px;align-items:center;
  gap:9px;padding:5px 0}
.curseur label{font-family:var(--mono);font-weight:600;font-size:13.5px;color:var(--accent-fonce)}
.curseur input[type=range]{width:100%;accent-color:var(--accent);height:22px;cursor:pointer}
.curseur input[type=number]{width:100%;font-family:var(--mono);font-size:12px;
  border:1px solid var(--trait);border-radius:6px;padding:4px 6px;color:var(--encre);background:#FBFCFD}
.curseur input[type=number]:focus{outline:2px solid var(--accent);border-color:transparent}
.separ{border-top:1px dashed var(--trait);margin:8px 0}

.boutons{display:flex;gap:8px;margin-top:12px}
button{font-family:var(--sans);font-size:12.5px;border-radius:7px;padding:8px 12px;
  cursor:pointer;border:1px solid var(--trait);background:var(--pastille);color:var(--encre);
  transition:all .15s;flex:1}
button:hover{border-color:var(--accent);color:var(--accent-fonce)}
button.principal{background:var(--encre);border-color:var(--encre);color:#fff}
button.principal:hover{background:var(--accent-fonce);border-color:var(--accent-fonce);color:#fff}

/* ---------- métriques ---------- */
.metriques{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px;margin-bottom:12px}
.metrique{background:var(--panneau);border:1px solid var(--trait);border-radius:10px;
  padding:10px 13px;border-top:3px solid var(--barre)}
.metrique.saillante{border-top-color:var(--accent)}
.metrique.ref{border-top-color:var(--reference)}
.metrique .etq{font-size:10.5px;text-transform:uppercase;letter-spacing:.7px;color:var(--sourdine)}
.metrique .val{font-family:var(--mono);font-size:19px;font-weight:600;margin-top:3px}
.metrique .det{font-family:var(--mono);font-size:11px;color:var(--sourdine);margin-top:2px}
.delta-pos{color:var(--accent-fonce)} .delta-neg{color:#A54B21}

/* ---------- graphique ---------- */
.zone-graph{position:relative;height:430px}
.legende-note{display:flex;gap:16px;align-items:center;margin-top:9px;
  font-size:11.5px;color:var(--sourdine);flex-wrap:wrap}
.pastille-leg{display:inline-block;width:18px;height:0;border-top:3px solid;
  vertical-align:middle;margin-right:5px;border-radius:2px}
.pastille-barre{display:inline-block;width:11px;height:11px;background:var(--barre);
  border-radius:3px;margin-right:5px;vertical-align:-1px}
@media (prefers-reduced-motion: reduce){*{transition:none !important}}
</style>
</head>
<body>
<div class="cadre">

  <div class="bandeau">
    <div>
      <h1>Calibration des courbes <span>ELR × Prime</span></h1>
      <div class="sous-titre">Majoration tarifaire par population — outil de calibrage interactif</div>
    </div>
    <div class="faits">
      <div class="fait"><div class="etq">Échéance</div><div class="val">__ECHEANCE__</div></div>
      <div class="fait"><div class="etq">Année</div><div class="val">__ANNEE__</div></div>
      <div class="fait"><div class="etq">Volume base</div><div class="val">__VOLUME__ contrats</div></div>
    </div>
  </div>

  <div class="grille">
    <!-- ==================== colonne de gauche ==================== -->
    <div>
      <div class="formule">
        <div class="titre-f">Formule de majoration</div>
        Valeur = (<span class="co" id="fa">a</span>·log(E) + <span class="co" id="fb">b</span>)
        × (<span class="co" id="fl">l</span> + log(P·<span class="co" id="fm">m</span>)
        + <span class="co" id="fn">n</span>)<br>
        <span class="clamp">Maj. initiale = min(<span class="co" id="fmax">max</span>,
        max(<span class="co" id="fmin">min</span>, Valeur))</span><br>
        <span class="clamp">Maj. finale&nbsp;&nbsp; = ajustements 1→5</span>
      </div>

      <div class="panneau">
        <div class="section-titre">Population calibrée</div>
        <div class="pops" id="pops"></div>

        <div class="section-titre">Coefficients de l'équation</div>
        <div id="sliders"></div>

        <div class="boutons">
          <button id="btn-reset-pop" class="principal">Réinitialiser la population</button>
          <button id="btn-reset-tout">Tout réinitialiser</button>
        </div>
      </div>
    </div>

    <!-- ==================== colonne principale ==================== -->
    <div>
      <div class="metriques">
        <div class="metrique saillante">
          <div class="etq">Ressource finale — population</div>
          <div class="val" id="m-res-pop">–</div>
          <div class="det" id="m-res-pop-delta"></div>
        </div>
        <div class="metrique ref">
          <div class="etq">Référence — population</div>
          <div class="val" id="m-res-pop-ref">–</div>
          <div class="det">coefficients initiaux</div>
        </div>
        <div class="metrique saillante">
          <div class="etq">Ressource finale — portefeuille</div>
          <div class="val" id="m-res-glob">–</div>
          <div class="det" id="m-res-glob-delta"></div>
        </div>
        <div class="metrique">
          <div class="etq">Σ primes — population</div>
          <div class="val" id="m-primes">–</div>
          <div class="det" id="m-primes-det"></div>
        </div>
        <div class="metrique">
          <div class="etq">Contrats — population</div>
          <div class="val" id="m-nb">–</div>
          <div class="det" id="m-nb-det"></div>
        </div>
      </div>

      <div class="panneau">
        <div class="zone-graph"><canvas id="graphique"></canvas></div>
        <div class="legende-note">
          <span><span class="pastille-barre"></span>Nombre de contrats (axe gauche)</span>
          <span><span class="pastille-leg" style="border-color:#C7802B;border-top-style:dashed"></span>Courbe de référence — coefficients initiaux (axe droit)</span>
          <span><span class="pastille-leg" style="border-color:#0E7C86"></span>Courbe calibrée — coefficients courants (axe droit)</span>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
"use strict";
/* ============================== données ============================== */
const BRUT      = __DATA__;
const PARAMS_INIT = __PARAMS__;
const BORNES    = __BORNES__;
const LABELS    = __LABELS__;
const TR_MIN = __TR_MIN__, TR_PAS = __TR_PAS__, NB_TR = __NB_TR__;
const POPS = Object.keys(PARAMS_INIT);
const CLES = ["a","b","l","m","n","min","max"];

/* Contrats restructurés + index par population (une seule fois) */
const CONTRATS = BRUT.pop.map((p,i)=>({
  popu:p, elr:BRUT.elr[i], prime:BRUT.prime[i], pav:BRUT.pav[i],
  anc:BRUT.anc[i], seg:BRUT.seg[i],
  logE:Math.log(BRUT.elr[i]), logP:Math.log(BRUT.prime[i]),
  tranche:Math.min(NB_TR-1, Math.max(0, Math.floor((BRUT.prime[i]-TR_MIN)/TR_PAS)))
}));
const PAR_POP = {};
POPS.forEach(p=>PAR_POP[p]=[]);
CONTRATS.forEach(c=>{ if(PAR_POP[c.popu]) PAR_POP[c.popu].push(c); });

/* ====================== fonctions d'ajustement ======================= */
__AJUSTEMENTS_JS__

/* =========================== moteur de calcul ======================== */
function majContrat(c, p, logM){
  // Valeur = (a·log(E)+b) × (l + log(P·m) + n) ; log(P·m)=log(P)+log(m)
  let v = (p.a*c.logE + p.b) * (p.l + c.logP + logM + p.n);
  v = Math.min(p.max, Math.max(p.min, v));           // majoration initiale
  for(let k=0;k<AJUSTEMENTS.length;k++) v = AJUSTEMENTS[k](v, c);
  return v;                                           // majoration finale
}

/* Agrégats d'une population : par tranche + totaux */
function calculerPop(popu, p){
  const lst = PAR_POP[popu], logM = Math.log(p.m);
  const nvT = new Float64Array(NB_TR), avT = new Float64Array(NB_TR), nbT = new Int32Array(NB_TR);
  let nv=0, av=0;
  for(let i=0;i<lst.length;i++){
    const c = lst[i];
    const np = c.pav * majContrat(c, p, logM);
    nv += np; av += c.pav;
    nvT[c.tranche]+=np; avT[c.tranche]+=c.pav; nbT[c.tranche]++;
  }
  const courbe = Array.from({length:NB_TR}, (_,t)=> avT[t]>0 ? nvT[t]/avT[t] : null);
  return {courbe, nbT:Array.from(nbT), nv, av, ressource: av>0?nv/av:null, nb:lst.length};
}

function ressourceGlobale(tousParams){
  let nv=0, av=0;
  for(const popu of POPS){
    const p = tousParams[popu], logM = Math.log(p.m), lst = PAR_POP[popu];
    for(let i=0;i<lst.length;i++){
      const c=lst[i]; nv += c.pav*majContrat(c,p,logM); av += c.pav;
    }
  }
  return {ressource:nv/av, nv, av};
}

/* ============================ état courant =========================== */
const copie = o => JSON.parse(JSON.stringify(o));
let params = copie(PARAMS_INIT);
let popActive = POPS[0];

/* Références figées (coefficients initiaux) — calculées une fois */
const REF_POP = {};
POPS.forEach(p=>REF_POP[p]=calculerPop(p, PARAMS_INIT[p]));
const REF_GLOB = ressourceGlobale(PARAMS_INIT);
const NB_TOTAL = CONTRATS.length;

/* =========================== mise en forme =========================== */
const fmtNb   = new Intl.NumberFormat('fr-FR');
const fmtEur  = new Intl.NumberFormat('fr-FR',{maximumFractionDigits:0});
const fmtMaj  = v => v==null?'–':v.toLocaleString('fr-FR',{minimumFractionDigits:3,maximumFractionDigits:3});
const fmtCoef = (k,v)=> k==='m' ? v.toFixed(4) : v.toFixed(3);

/* ============================ graphique ============================== */
const ctx = document.getElementById('graphique').getContext('2d');
const graphique = new Chart(ctx, {
  data:{
    labels: LABELS,
    datasets:[
      { type:'line', label:'Courbe calibrée', data:[], yAxisID:'yMaj', order:0,
        borderColor:'#0E7C86', backgroundColor:'#0E7C86', borderWidth:2.5,
        pointRadius:2.5, pointHoverRadius:5, tension:.3, spanGaps:true },
      { type:'line', label:'Courbe de référence', data:[], yAxisID:'yMaj', order:1,
        borderColor:'#C7802B', borderDash:[7,4], borderWidth:2,
        pointRadius:0, pointHoverRadius:4, tension:.3, spanGaps:true },
      { type:'bar', label:'Nombre de contrats', data:[], yAxisID:'yNb', order:2,
        backgroundColor:'#C6D0D9', borderRadius:3, barPercentage:.82 }
    ]
  },
  options:{
    responsive:true, maintainAspectRatio:false, animation:{duration:220},
    interaction:{mode:'index', intersect:false},
    plugins:{
      legend:{display:false},
      tooltip:{
        backgroundColor:'#16232E', titleFont:{family:"'IBM Plex Mono',monospace"},
        bodyFont:{family:"'IBM Plex Mono',monospace", size:12}, padding:10,
        callbacks:{
          title: it => 'Tranche de prime ' + it[0].label + ' €',
          label: it => it.dataset.yAxisID==='yMaj'
              ? ' ' + it.dataset.label + ' : ' + fmtMaj(it.parsed.y)
              : ' Contrats : ' + fmtNb.format(it.parsed.y)
        }
      }
    },
    scales:{
      x:{ title:{display:true, text:'Tranche de prime (PrimeSS, €)',
            font:{size:11.5}, color:'#5F6E7B'},
          grid:{display:false},
          ticks:{font:{family:"'IBM Plex Mono',monospace", size:10},
            maxRotation:60, minRotation:60, color:'#5F6E7B'} },
      yNb:{ position:'left', beginAtZero:true,
          title:{display:true, text:'Nombre de contrats', font:{size:11.5}, color:'#5F6E7B'},
          grid:{color:'#EAEEF2'},
          ticks:{font:{family:"'IBM Plex Mono',monospace", size:10.5}, color:'#5F6E7B'} },
      yMaj:{ position:'right',
          title:{display:true, text:'Majoration moyenne (Σ nouvelles / Σ anciennes)',
            font:{size:11.5}, color:'#0A5A62'},
          grid:{drawOnChartArea:false},
          ticks:{font:{family:"'IBM Plex Mono',monospace", size:10.5}, color:'#0A5A62',
            callback:v=>fmtMaj(v)} }
    }
  }
});

/* ======================= construction interface ====================== */
const elPops = document.getElementById('pops');
POPS.forEach(p=>{
  const d = document.createElement('div');
  d.className='pop'; d.id='pop-'+p;
  d.innerHTML = p + '<span class="n">'+fmtNb.format(PAR_POP[p].length)+'</span>';
  d.title = 'Population '+p+' — '+fmtNb.format(PAR_POP[p].length)+' contrats';
  d.addEventListener('click', ()=>changerPopulation(p));
  elPops.appendChild(d);
});

const elSliders = document.getElementById('sliders');
const refs = {};   // refs[k] = {range, num, badge}
CLES.forEach((k,idx)=>{
  if(k==='min'){ const s=document.createElement('div'); s.className='separ'; elSliders.appendChild(s); }
  const [mn,mx,pas] = BORNES[k];
  const ligne = document.createElement('div');
  ligne.className='curseur';
  ligne.innerHTML =
    '<label for="sl-'+k+'">'+k+'</label>'+
    '<input type="range" id="sl-'+k+'" min="'+mn+'" max="'+mx+'" step="'+pas+'" aria-label="Coefficient '+k+'">'+
    '<input type="number" id="nu-'+k+'" min="'+mn+'" max="'+mx+'" step="'+pas+'" aria-label="Valeur du coefficient '+k+'">';
  elSliders.appendChild(ligne);
  const range = ligne.querySelector('input[type=range]');
  const num   = ligne.querySelector('input[type=number]');
  refs[k] = {range, num, badge:document.getElementById('f'+k)};
  range.addEventListener('input', ()=>appliquerCoef(k, parseFloat(range.value)));
  num.addEventListener('change', ()=>{
    let v = parseFloat(num.value);
    if(isNaN(v)) v = params[popActive][k];
    v = Math.min(mx, Math.max(mn, v));
    appliquerCoef(k, v);
  });
});

document.getElementById('btn-reset-pop').addEventListener('click', ()=>{
  params[popActive] = copie(PARAMS_INIT[popActive]);
  synchroniserControles(); planifier();
});
document.getElementById('btn-reset-tout').addEventListener('click', ()=>{
  params = copie(PARAMS_INIT);   // la population sélectionnée est conservée
  synchroniserControles(); planifier();
});

/* ========================= logique interactive ======================= */
function appliquerCoef(k, v){
  params[popActive][k] = v;
  refs[k].range.value = v; refs[k].num.value = fmtCoef(k, v);
  refs[k].badge.textContent = k+'='+fmtCoef(k,v);
  refs[k].badge.classList.add('flash');
  clearTimeout(refs[k]._t);
  refs[k]._t = setTimeout(()=>refs[k].badge.classList.remove('flash'), 260);
  planifier();
}

function synchroniserControles(){
  const p = params[popActive];
  CLES.forEach(k=>{
    refs[k].range.value = p[k];
    refs[k].num.value = fmtCoef(k, p[k]);
    refs[k].badge.textContent = k+'='+fmtCoef(k, p[k]);
  });
  POPS.forEach(q=>{
    const el = document.getElementById('pop-'+q);
    el.classList.toggle('actif', q===popActive);
    el.classList.toggle('modif', JSON.stringify(params[q])!==JSON.stringify(PARAMS_INIT[q]));
  });
}

function changerPopulation(p){
  popActive = p;
  synchroniserControles();
  planifier(true);
}

/* recalcul planifié sur requestAnimationFrame → fluide pendant le drag */
let enAttente = false, avecStructure = false;
function planifier(structure){
  avecStructure = avecStructure || !!structure;
  if(enAttente) return;
  enAttente = true;
  requestAnimationFrame(()=>{ enAttente=false; const s=avecStructure; avecStructure=false; rafraichir(s); });
}

function rafraichir(structure){
  const res  = calculerPop(popActive, params[popActive]);
  const ref  = REF_POP[popActive];
  const glob = ressourceGlobale(params);

  /* graphique */
  graphique.data.datasets[0].data = res.courbe;
  if(structure){
    graphique.data.datasets[1].data = ref.courbe;
    graphique.data.datasets[2].data = ref.nbT;
  }
  graphique.update(structure ? undefined : 'none');

  /* métriques */
  const dPop  = res.ressource - ref.ressource;
  const dGlob = glob.ressource - REF_GLOB.ressource;
  poser('m-res-pop',  fmtMaj(res.ressource));
  poserDelta('m-res-pop-delta', dPop);
  poser('m-res-pop-ref', fmtMaj(ref.ressource));
  poser('m-res-glob', fmtMaj(glob.ressource));
  poserDelta('m-res-glob-delta', dGlob);
  poser('m-primes', fmtEur.format(res.nv)+' €');
  poser('m-primes-det', 'avant : '+fmtEur.format(res.av)+' €');
  poser('m-nb', fmtNb.format(res.nb));
  poser('m-nb-det', 'portefeuille : '+fmtNb.format(NB_TOTAL));
  synchroniserControles();
}

function poser(id, txt){ document.getElementById(id).textContent = txt; }
function poserDelta(id, d){
  const el = document.getElementById(id);
  const signe = d>=0?'+':'−';
  const pts = Math.abs(d).toLocaleString('fr-FR',{minimumFractionDigits:3,maximumFractionDigits:3});
  const pct = (Math.abs(d)*100).toLocaleString('fr-FR',{minimumFractionDigits:2,maximumFractionDigits:2});
  el.textContent = signe+pts+' ('+signe+pct+' %) vs référence';
  el.className = 'det '+(d>=0?'delta-pos':'delta-neg');
}

/* ============================ démarrage ============================== */
synchroniserControles();
planifier(true);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generer_base_test(8000)
    df.to_csv("base_test_contrats.csv", index=False)
    resultats, ressource = calculer_resultats(df, PARAMS_INITIAUX)
    print(f"Base test : {len(df)} contrats")
    print(f"Ressource finale (référence Python) : {ressource:.6f}")
    print(sauvegarder_html("outil_calibration.html", df))

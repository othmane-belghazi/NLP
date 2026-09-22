<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Bilan des mesures tarifaires 2026</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root{
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px);
  padding-bottom:env(safe-area-inset-bottom,0px);
  --page:#E9ECF1; --page-ink:#3A4250;
  --navy:#0A2240; --blue:#1F4E9A; --blue-soft:#C9D5E8;
  --ink:#1C2430; --grey:#6B7483; --rule:#D9DEE6; --rule-strong:#9AA3B1;
  --good:#1B7A48; --bad:#B42318; --target:#B42318;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){ --page:#12161D; --page-ink:#AEB6C3; }
}
:root[data-theme="dark"]{ --page:#12161D; --page-ink:#AEB6C3; }
*,*::before,*::after{box-sizing:inherit}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
html,body{margin:0;background:var(--page);color:var(--page-ink)}
body{font-family:"Source Sans 3",Arial,Helvetica,sans-serif;padding:24px 16px}
.wrap{max-width:1280px;margin:0 auto;position:relative}
.slide{
  width:1280px;height:720px;background:#fff;color:var(--ink);
  transform-origin:top left;position:absolute;top:0;left:0;
  padding:40px 52px 30px;display:flex;flex-direction:column;
  border:1px solid #CDD3DC;
}
.tracker{font-size:13px;color:var(--grey);display:flex;justify-content:space-between;margin-bottom:14px}
h1{font-family:"Source Serif 4",Georgia,serif;font-weight:600;font-size:30px;line-height:1.2;color:var(--navy);margin:0 0 8px;max-width:1060px;letter-spacing:-.2px}
.sub{font-size:15px;color:var(--grey);margin:0 0 18px;padding-bottom:14px;border-bottom:2px solid var(--navy)}
.body{display:grid;grid-template-columns:1fr 272px;gap:36px;flex:1;min-height:0}

.tbl{display:grid;grid-template-columns:150px 140px 150px 190px 1fr;align-content:start}
.th{font-size:13px;font-weight:700;color:var(--ink);padding:0 10px 8px 0;border-bottom:1px solid var(--rule-strong);line-height:1.25}
.th span{display:block;font-weight:400;color:var(--grey);font-size:12px}
.td{display:flex;align-items:center;gap:8px;height:52px;border-bottom:1px solid var(--rule);font-size:15px;padding-right:10px;font-variant-numeric:tabular-nums}
.name{font-weight:600}
.tot{border-top:1.5px solid var(--navy);border-bottom:none;font-weight:700;height:56px}
.bar{height:12px;background:var(--blue);border-radius:0 2px 2px 0}
.val{min-width:44px;white-space:nowrap}
.d{font-size:12.5px;display:inline-flex;align-items:center;gap:3px;font-weight:600}
.d svg{width:9px;height:9px}
.good{color:var(--good)} .bad{color:var(--bad)}
.elr{position:relative;width:170px;height:22px;flex:none}
.elr .b{position:absolute;left:0;top:4px;height:14px;border-radius:0 2px 2px 0}
.elr .t{position:absolute;left:50%;top:-8px;bottom:-8px;border-left:1.5px dashed var(--target)}
.hi{background:var(--blue)} .lo{background:var(--blue-soft)}
.axis{grid-column:5;font-size:12px;color:var(--target);padding-left:calc(85px - 38px);padding-top:6px}

.take{border-left:3px solid var(--navy);padding:2px 0 0 20px}
.take h2{font-family:"Source Serif 4",Georgia,serif;font-size:19px;font-weight:600;color:var(--navy);margin:0 0 14px}
.take ol{margin:0;padding:0;list-style:none;counter-reset:k}
.take li{counter-increment:k;position:relative;padding-left:30px;margin-bottom:16px;font-size:14.5px;line-height:1.45}
.take li::before{content:counter(k);position:absolute;left:0;top:1px;width:20px;height:20px;border-radius:50%;background:var(--navy);color:#fff;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center}
.take li b{color:var(--navy)}
.decision{margin-top:8px;padding:12px 14px;background:#F1F4F9;font-size:13.5px;line-height:1.45}
.decision b{display:block;color:var(--navy);margin-bottom:3px}

.legend{display:flex;gap:22px;font-size:12px;color:var(--grey);margin-top:14px;align-items:center;flex-wrap:wrap}
.legend i{display:inline-block;width:12px;height:10px;margin-right:6px;vertical-align:-1px}
.foot{margin-top:auto;display:flex;justify-content:space-between;align-items:flex-end;font-size:11.5px;color:var(--grey);border-top:1px solid var(--rule);padding-top:8px}
.note{font-size:12.5px;color:var(--page-ink);text-align:center;margin:12px auto 0;max-width:640px}
@media print{
  @page{size:1280px 720px;margin:0}
  body{padding:0;background:#fff}
  .slide{transform:none!important;border:none}
  .wrap{height:720px!important}
  .note{display:none}
}
</style>
</head>
<body>
<div class="wrap" id="wrap">
<section class="slide" id="slide" aria-label="Bilan des mesures tarifaires 2026">
  <div class="tracker"><span>Tarification IARD, bilan des mesures 2026</span><span>Comité de direction</span></div>
  <h1>L'ELR repasse sous la cible et la ressource progresse de 7 %, au prix d'une résiliation en hausse sur toutes les mesures</h1>
  <p class="sub">Indicateurs 2026 par mesure tarifaire et écart par rapport à 2025</p>

  <div class="body">
    <div>
      <div class="tbl" id="tbl" role="table" aria-label="Indicateurs par mesure tarifaire">
        <div class="th" role="columnheader">Mesure tarifaire</div>
        <div class="th" role="columnheader">Part des contrats<span>2026</span></div>
        <div class="th" role="columnheader">Taux de résiliation<span>2026, écart en pts</span></div>
        <div class="th" role="columnheader">Ressource HT<span>2026, variation vs 2025</span></div>
        <div class="th" role="columnheader">ELR<span>2026, écart en pts</span></div>
      </div>
      <div class="legend">
        <span><i style="background:var(--blue)"></i>ELR au-dessus de la cible</span>
        <span><i style="background:var(--blue-soft)"></i>ELR sous la cible</span>
        <span><i style="border-left:1.5px dashed var(--target);width:1px;height:12px"></i>Cible ELR 75 %</span>
        <span class="good">▲▼ amélioration</span>
        <span class="bad">▲▼ dégradation</span>
      </div>
    </div>

    <aside class="take">
      <h2>Ce qu'il faut retenir</h2>
      <ol>
        <li>L'ELR portefeuille atteint <b>73,6 %</b> (−2,7 pts), sous la cible de 75 %, porté par les hausses forte et ciblée.</li>
        <li>La résiliation progresse sur <b>toutes les mesures</b> (+0,9 pt au global), jusqu'à +2,3 pts sur la hausse forte.</li>
        <li>Gel tarifaire et fidélité <b>dégradent l'ELR</b> de 4 à 5 pts alors que la fidélité gagne 19 % de ressource.</li>
      </ol>
      <div class="decision"><b>Décision attendue</b>Reconduire ou ajuster le gel et la remise fidélité pour 2027.</div>
    </aside>
  </div>

  <div class="foot">
    <span>Source : données internes ; résiliation 2026 observée à fin août. ELR portefeuille pondéré par la ressource HT, résiliation par les contrats. Données illustratives.</span>
    <span>3</span>
  </div>
</section>
</div>
<p class="note">Pour PowerPoint : capture d'écran de la slide, ou impression en PDF (format paysage) puis insertion.</p>

<script>
const data = [
  // mesure, part contrats 26, résil 25, résil 26, ressource 25, ressource 26, ELR 25, ELR 26
  ["Indexation",    33,  7.5,  8.0, 42,  45,   68, 66],
  ["Hausse ciblée", 24, 10.5, 11.8, 30,  35,   74, 70],
  ["Hausse forte",  10, 15.2, 17.5, 18,  16,   88, 80],
  ["Gel tarifaire", 14,  5.8,  6.2, 14,  13,   82, 86],
  ["Fidélité",      12,  4.2,  4.5,  8,   9.5, 74, 79],
  ["Repricing",      7, 19.0, 21.0,  7,   8.5, 96, 91],
];
const CIBLE = 75, ELR_MIN = 50, ELR_MAX = 100;

const fr = (x, d = 1) => x.toLocaleString("fr-FR", {minimumFractionDigits: d, maximumFractionDigits: d});
const up = '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M5 1 9.5 9h-9z" fill="currentColor"/></svg>';
const dn = '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M5 9 .5 1h9z" fill="currentColor"/></svg>';

// Écart : lowerIsBetter = true pour résiliation et ELR
function delta(ecart, lowerIsBetter, d, unite) {
  const bon = lowerIsBetter ? ecart < 0 : ecart > 0;
  const cls = ecart === 0 ? "" : (bon ? "good" : "bad");
  const ico = ecart > 0 ? up : (ecart < 0 ? dn : "");
  const s = ecart > 0 ? "+" : (ecart < 0 ? "−" : "");
  return `<span class="d ${cls}">${ico}${s}${fr(Math.abs(ecart), d)}${unite}</span>`;
}

const maxPart = Math.max(...data.map(r => r[1]));
const tbl = document.getElementById("tbl");

function row(r, total) {
  const [nom, part, re0, re1, rs0, rs1, e0, e1] = r;
  const c = total ? " tot" : "";
  const wPart = total ? 0 : part / maxPart * 70;
  const wElr = (Math.min(e1, ELR_MAX) - ELR_MIN) / (ELR_MAX - ELR_MIN) * 170;
  const var_rs = (rs1 / rs0 - 1) * 100;
  tbl.insertAdjacentHTML("beforeend", `
    <div class="td name${c}" role="cell">${nom}</div>
    <div class="td${c}" role="cell">${total ? "" : `<div class="bar" style="width:${wPart}px"></div>`}<span>${fr(part, 0)} %</span></div>
    <div class="td${c}" role="cell"><span class="val">${fr(re1)} %</span>${delta(re1 - re0, true, 1, "")}</div>
    <div class="td${c}" role="cell"><span class="val" style="min-width:74px">${fr(rs1)} M€</span>${delta(var_rs, false, 0, " %")}</div>
    <div class="td${c}" role="cell">
      <div class="elr"><div class="b ${e1 > CIBLE ? "hi" : "lo"}" style="width:${wElr}px"></div><div class="t"></div></div>
      <span class="val">${fr(e1, total ? 1 : 0)} %</span>${delta(e1 - e0, true, total ? 1 : 0, "")}
    </div>`);
}

data.forEach(r => row(r, false));

// Portefeuille : résiliation pondérée par les contrats, ELR pondéré par la ressource
const partTot = data.reduce((s, r) => s + r[1], 0);
const rs0 = data.reduce((s, r) => s + r[4], 0), rs1 = data.reduce((s, r) => s + r[5], 0);
// part contrats 2025 (pour pondérer la résiliation 2025)
const part25 = [35, 22, 12, 15, 10, 6];
const re0 = data.reduce((s, r, i) => s + r[2] * part25[i], 0) / part25.reduce((a, b) => a + b, 0);
const re1 = data.reduce((s, r) => s + r[3] * r[1], 0) / partTot;
const e0 = data.reduce((s, r) => s + r[6] * r[4], 0) / rs0;
const e1 = data.reduce((s, r) => s + r[7] * r[5], 0) / rs1;
row(["Portefeuille", partTot, re0, re1, rs0, rs1, e0, e1], true);
tbl.insertAdjacentHTML("beforeend", `<div class="axis" style="grid-column:5">Cible 75 %</div>`);

// Mise à l'échelle de la slide 1280x720 selon la largeur disponible
const wrap = document.getElementById("wrap"), slide = document.getElementById("slide");
function fit() {
  const s = Math.min(1, wrap.clientWidth / 1280);
  slide.style.transform = `scale(${s})`;
  wrap.style.height = (720 * s) + "px";
}
window.addEventListener("resize", fit);
fit();
</script>
</body>
</html>

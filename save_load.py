"""
Visuels style McKinsey - mesures tarifaires, 2025 vs 2026
    1. Matrice résiliation x ELR (bubble chart)
    2. Mekko : part des contrats x ELR
    3. Dumbbell : résiliation observée vs prédite
    4. One-pager : tableau-graphique de synthèse

Dépendances : pip install pandas matplotlib openpyxl
Lancement   : python visuels_mesures_tarifaires.py
Sortie      : dossier ./visuels (PNG pour les mails, SVG éditable dans PowerPoint)
"""
import logging
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# =============================================================================
# 1. DONNÉES
# =============================================================================
# Remplacez ce bloc par vos données, par exemple :
#   df = pd.read_excel("mesures_tarifaires.xlsx")
# Format long : une ligne par (mesure, année).
# Colonnes : mesure, annee, part_contrats (%), ressource_ht (M€),
#            resil_obs (%), resil_pred (%), elr (%)
# Si vos taux sont en décimal (0.66), multipliez-les par 100.
mesures = ["Indexation", "Hausse ciblée", "Hausse forte",
           "Gel tarifaire", "Fidélité", "Repricing"]
df = pd.DataFrame({
    "mesure": mesures * 2,
    "annee": [2025] * 6 + [2026] * 6,
    "part_contrats": [35, 22, 12, 15, 10, 6,
                      33, 24, 10, 14, 12, 7],
    "ressource_ht": [42, 30, 18, 14, 8, 7,
                     45, 35, 16, 13, 9.5, 8.5],
    "resil_obs": [7.5, 10.5, 15.2, 5.8, 4.2, 19.0,
                  8.0, 11.8, 17.5, 6.2, 4.5, 21.0],
    "resil_pred": [7.8, 9.8, 13.0, 6.0, 5.0, 16.5,
                   7.6, 10.2, 14.0, 6.1, 4.8, 18.0],
    "elr": [68, 74, 88, 82, 74, 96,
            66, 70, 80, 86, 79, 91],
})

# =============================================================================
# 2. PARAMÈTRES
# =============================================================================
AN_N1, AN_N = 2025, 2026
ELR_CIBLE = 75       # % : ligne cible
SEUIL_RESIL = 10     # % : séparation verticale des quadrants
SEUIL_ECART = 1.0    # pt : écart observé/prédit jugé significatif
SOURCE = ("Source : données internes ; résiliation 2026 observée à fin août ; "
          "analyse équipe Tarification")
DOSSIER = Path("visuels")
FORMATS = ["png", "svg"]

# Titres-messages : à réécrire selon ce que disent VOS chiffres
TITRES = {
    "matrice": ("Hausse forte et repricing gagnent 5 à 8 pts d'ELR, au prix "
                "d'une résiliation supérieure au modèle",
                f"Taux de résiliation observé vs ELR par mesure tarifaire, "
                f"{AN_N1} → {AN_N}"),
    "mekko": ("4 mesures sur 6 restent au-dessus de l'ELR cible, mais ne pèsent "
              "que 43 % des contrats",
              "ELR par mesure tarifaire ; épaisseur des barres = part des contrats"),
    "dumbbell": ("Le modèle sous-estime de 3 pts la résiliation des hausses "
                 "les plus fortes",
                 "Taux de résiliation observé vs prédit par mesure tarifaire, en %"),
    "onepager": ("Ressource HT +7 % et ELR −2,7 pts, portés par l'indexation "
                 "et la hausse ciblée",
                 f"Synthèse par mesure tarifaire, {AN_N1} → {AN_N}"),
}

# Position des étiquettes de la matrice si elles se chevauchent :
# "droite" (défaut), "gauche", "haut" ou "bas"
POSITION_LABEL = {"Indexation": "gauche", "Hausse forte": "bas",
                  "Repricing": "bas"}

# Palette : une couleur d'accent, des gris, rouge/vert réservés aux alertes
ACCENT = "#185FA5"
TEXTE = "#1A1A1A"
GRIS_FONCE = "#5F5E5A"
GRIS = "#888780"
GRIS_CLAIR = "#C9C7BE"
GRIS_LIGNE = "#E8E6DF"
ROUGE = "#C8372D"
VERT = "#2E7D32"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "text.color": TEXTE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GRIS,
    "axes.labelcolor": GRIS_FONCE,
    "xtick.color": GRIS_FONCE,
    "ytick.color": GRIS_FONCE,
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",  # texte éditable dans PowerPoint
})


# =============================================================================
# 3. OUTILS
# =============================================================================
def fr(x, d=1):
    """Nombre au format français : 1 234,5"""
    return f"{x:,.{d}f}".replace(",", "\u202f").replace(".", ",")


def signe(x, d=1):
    """Écart signé : +3,5 / −2,0"""
    s = "+" if x > 0 else ("−" if x < 0 else "")
    return f"{s}{fr(abs(x), d)}"


def pct(v, _=None):
    return f"{v:.0f} %"


def habiller(fig, cle):
    """Titre-message, sous-titre et source, comme sur une slide."""
    titre, sous_titre = TITRES[cle]
    w, h = fig.get_size_inches()
    lignes = textwrap.wrap(titre, width=int(w * 8))
    fig.text(0.01, 1 - 0.15 / h, "\n".join(lignes), fontsize=15,
             fontweight="bold", va="top", ha="left", linespacing=1.15)
    y_sous = 1 - (0.15 + len(lignes) * 0.27 + 0.1) / h
    fig.text(0.01, y_sous, sous_titre, fontsize=10.5, color=GRIS_FONCE,
             va="top", ha="left")
    fig.subplots_adjust(top=y_sous - 0.65 / h)
    fig.text(0.01, 0.005, SOURCE, fontsize=8, color=GRIS,
             va="bottom", ha="left")


def pivot(d):
    return d.pivot(index="mesure", columns="annee")


def portefeuille(d, an):
    """Agrégats portefeuille : résiliation pondérée par les contrats,
    ELR pondéré par la ressource (idéalement : sinistres / primes acquises)."""
    x = d[d.annee == an]
    w_c, w_r = x.part_contrats, x.ressource_ht
    return {
        "part_contrats": w_c.sum(),
        "ressource_ht": w_r.sum(),
        "resil_obs": np.average(x.resil_obs, weights=w_c),
        "resil_pred": np.average(x.resil_pred, weights=w_c),
        "elr": np.average(x.elr, weights=w_r),
    }


# =============================================================================
# 4. VISUEL 1 : MATRICE RÉSILIATION x ELR
# =============================================================================
def graphe_matrice(d):
    p = pivot(d)
    x0, y0 = p["resil_obs"][AN_N1], p["elr"][AN_N1]
    x1, y1 = p["resil_obs"][AN_N], p["elr"][AN_N]
    xp = p["resil_pred"][AN_N]
    taille = p["ressource_ht"][AN_N] / p["ressource_ht"][AN_N].max() * 3000

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(top=0.84, bottom=0.17, left=0.07, right=0.97)

    ax.axhline(ELR_CIBLE, color=ROUGE, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.axvline(SEUIL_RESIL, color=GRIS, lw=1, ls=(0, (4, 3)), zorder=1)

    for m in p.index:
        ax.plot([x0[m], x1[m]], [y0[m], y1[m]], color=GRIS, lw=1.2, zorder=2)
        ax.plot([xp[m], x1[m]], [y1[m], y1[m]], color=TEXTE, lw=0.8,
                ls=(0, (2, 2)), zorder=2)
    ax.scatter(x0, y0, s=28, color=GRIS, zorder=3)
    ax.scatter(x1, y1, s=taille, color=ACCENT, alpha=0.88, lw=0, zorder=4)
    ax.scatter(xp, y1, s=50, facecolor="none", edgecolor=TEXTE, lw=1.3,
               zorder=5)

    for m in p.index:
        r = np.sqrt(taille[m]) / 2 + 5
        pos = POSITION_LABEL.get(m, "droite")
        dx, dy, ha, va = {"droite": (r, 0, "left", "center"),
                          "gauche": (-r, 0, "right", "center"),
                          "haut": (0, r, "center", "bottom"),
                          "bas": (0, -r, "center", "top")}[pos]
        ax.annotate(m, (x1[m], y1[m]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=10.5, zorder=6)

    xs = pd.concat([x0, x1, xp])
    ys = pd.concat([y0, y1])
    ax.set_xlim(0, xs.max() * 1.15)
    ax.set_ylim(min(ys.min(), ELR_CIBLE) - 6, max(ys.max(), ELR_CIBLE) + 6)

    style_q = dict(transform=ax.transAxes, fontsize=10, color=GRIS,
                   style="italic")
    ax.text(0.01, 0.98, "Fidèle mais déficitaire", va="top", **style_q)
    ax.text(0.99, 0.98, "À redresser", va="top", ha="right", **style_q)
    ax.text(0.01, 0.02, "Cœur rentable", va="bottom", **style_q)
    ax.text(0.99, 0.02, "Rentable mais fuite", va="bottom", ha="right",
            **style_q)
    ax.text(0.01, ELR_CIBLE, f"ELR cible {ELR_CIBLE} %",
            transform=ax.get_yaxis_transform(), va="bottom", fontsize=9,
            color=ROUGE)

    ax.xaxis.set_major_formatter(FuncFormatter(pct))
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_xlabel(f"Taux de résiliation observé {AN_N}")
    ax.set_ylabel(f"ELR {AN_N}")

    leg = [
        Line2D([0], [0], marker="o", ls="none", markerfacecolor=ACCENT,
               markeredgecolor="none", markersize=13,
               label=f"{AN_N} (taille = ressource HT)"),
        Line2D([0], [0], marker="o", ls="none", markerfacecolor=GRIS,
               markeredgecolor="none", markersize=6, label=str(AN_N1)),
        Line2D([0], [0], marker="o", ls="none", markerfacecolor="none",
               markeredgecolor=TEXTE, markersize=7,
               label=f"Résiliation prédite {AN_N}"),
    ]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=3, frameon=False)
    habiller(fig, "matrice")
    return fig


# =============================================================================
# 5. VISUEL 2 : MEKKO PART DES CONTRATS x ELR
# =============================================================================
def graphe_mekko(d):
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.12, right=0.97,
                        wspace=0.45)
    xmax = d["elr"].max() * 1.15

    for ax, an in zip(axes, [AN_N1, AN_N]):
        x = d[d.annee == an].sort_values("elr", ascending=False)
        total = x.part_contrats.sum()
        y = 0.0
        for _, r in x.iterrows():
            h = r.part_contrats / total * 100
            haut = r.elr > ELR_CIBLE
            coul = ACCENT if haut else GRIS_CLAIR
            ax.barh(y + h / 2, r.elr, height=h - 0.8, color=coul, lw=0)
            ax.text(1.5, y + h / 2, f"{fr(r.part_contrats, 0)} %",
                    va="center", fontsize=9,
                    color="white" if haut else TEXTE)
            ax.text(-1.5, y + h / 2, r.mesure, ha="right", va="center",
                    fontsize=10)
            ax.text(r.elr + 1.5, y + h / 2, f"{fr(r.elr, 0)} %",
                    va="center", fontsize=10, fontweight="bold", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
            y += h

        ax.set_ylim(100, 0)
        ax.set_xlim(0, xmax)
        ax.axvline(ELR_CIBLE, color=ROUGE, lw=1, ls=(0, (4, 3)))
        ax.text(ELR_CIBLE, 101.5, f"ELR cible {ELR_CIBLE} %", ha="center",
                va="top", fontsize=9, color=ROUGE)
        ax.set_title(str(an), loc="left", fontsize=12, fontweight="bold",
                     pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["bottom"].set_visible(False)

    leg = [Patch(color=ACCENT, label="ELR au-dessus de la cible"),
           Patch(color=GRIS_CLAIR, label="ELR sous la cible")]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.03),
               ncol=2, frameon=False)
    habiller(fig, "mekko")
    return fig


# =============================================================================
# 6. VISUEL 3 : DUMBBELL RÉSILIATION OBSERVÉE vs PRÉDITE
# =============================================================================
def graphe_dumbbell(d):
    p = pivot(d)
    ecart = p["resil_obs"] - p["resil_pred"]
    ordre = ecart[AN_N].sort_values().index  # plus gros écart en haut
    xmax = max(p["resil_obs"].values.max(), p["resil_pred"].values.max()) * 1.25

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.13, right=0.97,
                        wspace=0.12)

    for ax, an in zip(axes, [AN_N1, AN_N]):
        for i, m in enumerate(ordre):
            o, pr = p["resil_obs"][an][m], p["resil_pred"][an][m]
            e = o - pr
            c = ROUGE if abs(e) > SEUIL_ECART else GRIS
            ax.axhline(i, color=GRIS_LIGNE, lw=0.8, zorder=0)
            ax.plot([pr, o], [i, i], color=c, lw=3, solid_capstyle="butt",
                    zorder=2)
            ax.scatter(pr, i, s=75, facecolor="white", edgecolor=c, lw=1.6,
                       zorder=3)
            ax.scatter(o, i, s=75, color=c, zorder=4)
            ax.text(max(o, pr) + xmax * 0.035, i, f"{signe(e)} pt",
                    va="center", fontsize=9, color=c,
                    fontweight="bold" if c == ROUGE else "normal")
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.7, len(ordre) - 0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(pct))
        ax.set_title(str(an), loc="left", fontsize=12, fontweight="bold",
                     pad=10)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    axes[0].set_yticks(range(len(ordre)))
    axes[0].set_yticklabels(ordre, fontsize=10.5, color=TEXTE)

    leg = [
        Line2D([0], [0], marker="o", ls="none", markerfacecolor=GRIS,
               markeredgecolor=GRIS, markersize=8, label="Observé"),
        Line2D([0], [0], marker="o", ls="none", markerfacecolor="none",
               markeredgecolor=GRIS, markersize=8, label="Prédit"),
        Line2D([0], [0], color=ROUGE, lw=3,
               label=f"Écart > {fr(SEUIL_ECART)} pt"),
    ]
    fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.03),
               ncol=3, frameon=False)
    habiller(fig, "dumbbell")
    return fig


# =============================================================================
# 7. VISUEL 4 : ONE-PAGER TABLEAU-GRAPHIQUE
# =============================================================================
def graphe_onepager(d):
    p = pivot(d)
    ordre = list(p["ressource_ht"][AN_N].sort_values(ascending=False).index)
    n = len(ordre)
    tot = {an: portefeuille(d, an) for an in (AN_N1, AN_N)}

    # Colonnes (axe x de 0 à 100)
    X_NOM, X_PART, W_PART = 0, 16, 8
    X_RESS, W_RESS, X_DRESS = 30, 9, 52
    X_RESIL, W_RESIL, X_RESIL_TXT = 56, 14, 71.5
    X_ELR, X_DELR = 84, 100

    fig, ax = plt.subplots(figsize=(13, 0.55 * (n + 2) + 1.8))
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.01, right=0.99)
    ax.set_xlim(0, 100)
    ax.set_ylim(n + 0.7, -1.1)
    ax.axis("off")

    entetes = [(X_NOM, "Mesure tarifaire", "left"),
               (X_PART, f"Part contrats {AN_N}", "left"),
               (X_RESS, f"Ressource HT {AN_N} (| = {AN_N1})", "left"),
               (X_DRESS, f"Δ vs {AN_N1}", "right"),
               (X_RESIL, f"Résiliation {AN_N} : observée ● / prédite ○", "left"),
               (X_ELR, f"ELR {AN_N1} → {AN_N}", "left")]
    for x, t, ha in entetes:
        ax.text(x, -0.9, t, fontsize=9, color=GRIS, ha=ha, va="bottom")
    ax.plot([0, 100], [-0.6, -0.6], color=GRIS, lw=0.8)

    max_part = p["part_contrats"][AN_N].max()
    max_ress = p["ressource_ht"].values.max()
    max_resil = max(p["resil_obs"][AN_N].max(), p["resil_pred"][AN_N].max()) * 1.05

    def ligne(i, nom, part, r0, r1, o, pr, e0, e1, gras=False, barres=True):
        fw = "bold" if gras else "normal"
        ax.text(X_NOM, i, nom, va="center", fontsize=10.5, fontweight=fw)
        # Part contrats
        wp = part / max_part * W_PART if barres else 0
        if barres:
            ax.barh(i, wp, left=X_PART, height=0.36, color=ACCENT, lw=0)
        ax.text(X_PART + wp + 0.6, i, f"{fr(part, 0)} %", va="center",
                fontsize=10, fontweight=fw)
        # Ressource HT (barre N + repère N-1)
        wr = r1 / max_ress * W_RESS if barres else 0
        if barres:
            ax.barh(i, wr, left=X_RESS, height=0.36, color=ACCENT, lw=0)
            xr0 = X_RESS + r0 / max_ress * W_RESS
            ax.plot([xr0, xr0], [i - 0.27, i + 0.27], color=TEXTE, lw=1.6)
        fin = max(wr, r0 / max_ress * W_RESS) if barres else 0
        ax.text(X_RESS + fin + 0.6, i, f"{fr(r1)} M€", va="center",
                fontsize=10, fontweight=fw)
        dr = (r1 / r0 - 1) * 100
        ax.text(X_DRESS, i, f"{signe(dr, 0)} %", va="center", ha="right",
                fontsize=10, fontweight=fw, color=VERT if dr >= 0 else ROUGE)
        # Résiliation observée vs prédite (mini-dumbbell)
        c = ROUGE if abs(o - pr) > SEUIL_ECART else GRIS
        xo = X_RESIL + o / max_resil * W_RESIL
        xp_ = X_RESIL + pr / max_resil * W_RESIL
        ax.plot([X_RESIL, X_RESIL + W_RESIL], [i, i], color=GRIS_LIGNE, lw=1,
                zorder=0)
        ax.plot([xp_, xo], [i, i], color=c, lw=2.5, zorder=1)
        ax.scatter(xp_, i, s=40, facecolor="white", edgecolor=c, lw=1.3,
                   zorder=2)
        ax.scatter(xo, i, s=40, color=c, zorder=3)
        ax.text(X_RESIL_TXT, i, f"{fr(o)} / {fr(pr)} %", va="center",
                fontsize=10, fontweight=fw, color=ROUGE if c == ROUGE else TEXTE)
        # ELR N-1 -> N
        de = e1 - e0
        ax.text(X_ELR, i, f"{fr(e0, 0)} → {fr(e1, 0)} %", va="center",
                fontsize=10, fontweight=fw)
        ax.text(X_DELR, i, f"{signe(de)} pts", va="center", ha="right",
                fontsize=10, fontweight="bold",
                color=VERT if de <= 0 else ROUGE)

    for i, m in enumerate(ordre):
        ligne(i, m, p["part_contrats"][AN_N][m],
              p["ressource_ht"][AN_N1][m], p["ressource_ht"][AN_N][m],
              p["resil_obs"][AN_N][m], p["resil_pred"][AN_N][m],
              p["elr"][AN_N1][m], p["elr"][AN_N][m])
        if i < n - 1:
            ax.plot([0, 100], [i + 0.5, i + 0.5], color=GRIS_LIGNE, lw=0.8)

    ax.plot([0, 100], [n - 0.45, n - 0.45], color=GRIS, lw=0.8)
    t0, t1 = tot[AN_N1], tot[AN_N]
    ligne(n + 0.1, "Portefeuille", t1["part_contrats"],
          t0["ressource_ht"], t1["ressource_ht"],
          t1["resil_obs"], t1["resil_pred"], t0["elr"], t1["elr"],
          gras=True, barres=False)

    habiller(fig, "onepager")
    return fig


# =============================================================================
# 8. EXPORT
# =============================================================================
if __name__ == "__main__":
    DOSSIER.mkdir(exist_ok=True)
    visuels = [("1_matrice_resiliation_elr", graphe_matrice),
               ("2_mekko_contrats_elr", graphe_mekko),
               ("3_dumbbell_resiliation", graphe_dumbbell),
               ("4_onepager_synthese", graphe_onepager)]
    for nom, fonction in visuels:
        fig = fonction(df)
        for ext in FORMATS:
            fig.savefig(DOSSIER / f"{nom}.{ext}")
        plt.close(fig)
        print(f"OK  {DOSSIER / nom}.[{', '.join(FORMATS)}]")

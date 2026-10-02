"""Matrice des sinistres : % de contrats en texte, charge totale en couleur.

Installation : pip install pandas matplotlib
Contrats : une ligne par contrat, colonnes contrat_id, segment.
Sinistres : une ligne par sinistre, colonnes contrat_id, nature, type,
etat, responsabilite, charge. Charge numérique, même devise et même base.
"""
import argparse
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib import colormaps


def generer_matrice(contrats, sinistres, segment, sortie="matrice.html"):
    for df, cols in [(contrats, ["contrat_id", "segment"]),
                     (sinistres, ["contrat_id", "nature", "type", "etat",
                                  "responsabilite", "charge"])]:
        if not set(cols).issubset(df.columns):
            raise ValueError(f"Colonnes requises : {cols}")
        if df[cols].isna().any().any():
            raise ValueError("Renseigner toutes les colonnes requises.")
    if contrats["contrat_id"].duplicated().any():
        raise ValueError("La table contrats doit avoir une ligne par contrat.")
    if not sinistres["contrat_id"].isin(contrats["contrat_id"]).all():
        raise ValueError("Certains sinistres ont un contrat absent de la table contrats.")

    # Dénominateur : tous les contrats du segment, même sans sinistre.
    ids = contrats.loc[contrats["segment"].eq(segment), "contrat_id"]
    total = ids.nunique()
    if not total:
        raise ValueError(f"Aucun contrat pour le segment {segment!r}.")
    s = sinistres.loc[sinistres["contrat_id"].isin(ids)].copy()
    if s.empty:
        raise ValueError("Aucun sinistre dans ce segment pour construire la matrice.")
    s["charge"] = pd.to_numeric(s["charge"], errors="raise")
    if not np.isfinite(s["charge"]).all():
        raise ValueError("Les charges doivent être des nombres finis.")

    dimensions = ["nature", "type", "etat", "responsabilite"]
    groupes = s.groupby(dimensions, observed=True).agg(
        contrats=("contrat_id", "nunique"), charge=("charge", "sum"))

    # Un contrat est compté une seule fois dans chaque case.
    pourcentages = (100 * groupes["contrats"] / total).unstack(
        ["etat", "responsabilite"]).fillna(0)
    charges = groupes["charge"].unstack(["etat", "responsabilite"]).reindex(
        index=pourcentages.index, columns=pourcentages.columns).fillna(0)

    # Couleur = charge totale ; une seule échelle pour toute la matrice.
    minimum = min(0.0, float(charges.min().min()))
    maximum = max(0.0, float(charges.max().max()))
    palette = colormaps["Blues"]

    format_fr = lambda x: f"{x:,.0f}".replace(",", " ")
    entetes = '<tr><th rowspan="2">Nature</th><th rowspan="2">Type</th>'
    for etat in pourcentages.columns.get_level_values(0).unique():
        nombre = sum(col[0] == etat for col in pourcentages.columns)
        entetes += f'<th colspan="{nombre}">État : {escape(str(etat))}</th>'
    entetes += '</tr><tr>' + ''.join(
        f'<th>Responsabilité : {escape(str(resp))}</th>'
        for _, resp in pourcentages.columns) + '</tr>'
    lignes = []
    for (nature, typ), valeurs in pourcentages.iterrows():
        ligne = f'<tr><th>{escape(str(nature))}</th><th>{escape(str(typ))}</th>'
        for col, pct in valeurs.items():
            charge = charges.loc[(nature, typ), col]
            niveau = (charge-minimum)/(maximum-minimum) if maximum>minimum else 0
            couleur = to_hex(palette(0.08 + 0.52 * niveau))
            texte = f'{pct:.1f} %'.replace('.', ',')
            detail = f'Charge totale : {format_fr(charge)} €'
            ligne += (f'<td style="background:{couleur}" title="{detail}" '
                      f'aria-label="{texte}. {detail}">{texte}</td>')
        lignes.append(ligne + '</tr>')
    tableau = '<table><thead>' + entetes + '</thead><tbody>' + ''.join(lignes) + '</tbody></table>'

    page = f"""<!doctype html>
<html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Matrice des sinistres</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#111;background:#fff}}
table{{border-collapse:collapse}} th,td{{padding:12px;border:1px solid #ddd;text-align:center}}
th{{background:#f3f4f6}} .matrice{{overflow-x:auto}}</style>
<h2>Segment : {escape(str(segment))}</h2>
<p>Chaque case = % des {format_fr(total)} contrats du segment concernés.</p>
<div class="matrice">{tableau}</div>
<p>Bleu clair → bleu foncé : charge totale de {format_fr(minimum)}
à {format_fr(maximum)} €. Survol d’une case : charge exacte.</p>
<p>Un contrat peut apparaître dans plusieurs cases : les pourcentages
ne s’additionnent donc pas nécessairement à 100 %.</p>
</html>"""
    Path(sortie).write_text(page, encoding="utf-8")
    return pourcentages, charges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrats", required=True)
    parser.add_argument("--sinistres", required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--sep", default=";", help="Séparateur CSV")
    parser.add_argument("--decimal", default=",", help="Séparateur décimal")
    parser.add_argument("--sortie", default="matrice.html")
    args = parser.parse_args()
    c = pd.read_csv(args.contrats, sep=args.sep,
                    dtype={"contrat_id": str, "segment": str})
    s = pd.read_csv(args.sinistres, sep=args.sep, decimal=args.decimal,
                    dtype={k: str for k in ["contrat_id", "nature", "type",
                                             "etat", "responsabilite"]})
    generer_matrice(c, s, args.segment, args.sortie)
    print(f"Tableau généré : {args.sortie}")

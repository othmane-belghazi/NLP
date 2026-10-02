"""Matrice depuis un seul DataFrame : % de contrats et charge en couleur.
Installation : pip install pandas matplotlib
Utilisation : pourcentages, charges = generer_matrice(df)
id doit identifier le contrat. La charge de chaque sinistre doit figurer
une seule fois ; les charges répétées après une jointure seraient additionnées.
"""
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import to_hex


def generer_matrice(df, sortie="matrice.html", colonnes=None):
    # Modifier ici les noms de colonnes si nécessaire.
    noms = {
        "id": "id", "etat": "etat", "nature": "nature",
        "responsabilite": "responsabilité",
        "type": "type sinistre", "charge": "charge",
    }
    if colonnes:
        noms.update(colonnes)
    if len(set(noms.values())) != len(noms):
        raise ValueError("Chaque variable doit correspondre à une colonne distincte.")
    manquantes = set(noms.values()) - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes absentes : {sorted(manquantes)}")
    s = df[list(noms.values())].copy()
    s.columns = list(noms)
    if s["id"].isna().any():
        raise ValueError("Chaque ligne doit avoir un identifiant de contrat.")
    total = s["id"].nunique()
    if total == 0:
        raise ValueError("Le DataFrame ne contient aucun contrat.")

    dimensions = ["nature", "type", "etat", "responsabilite"]
    # Les contrats sans sinistre restent dans le dénominateur si présents dans df.
    s = s.loc[~s[dimensions + ["charge"]].isna().all(axis=1)].copy()
    if s.empty:
        raise ValueError("Aucun sinistre à représenter.")
    for col in dimensions:
        s[col] = s[col].astype("string").fillna("Non renseigné")
    s["charge"] = pd.to_numeric(s["charge"], errors="raise")
    if not np.isfinite(s["charge"]).all():
        raise ValueError("Chaque sinistre doit avoir une charge numérique renseignée.")

    groupes = s.groupby(dimensions, observed=True).agg(
        contrats=("id", "nunique"), charge=("charge", "sum"))
    pourcentages = (100 * groupes["contrats"] / total).unstack(
        ["etat", "responsabilite"]).fillna(0).sort_index(axis=1)
    charges = groupes["charge"].unstack(["etat", "responsabilite"]).reindex(
        index=pourcentages.index, columns=pourcentages.columns).fillna(0)

    # Une échelle commune de charge pour toutes les cases.
    bas = min(0.0, float(charges.min().min()))
    haut = max(0.0, float(charges.max().max()))
    palette = colormaps["Blues"]
    euros = lambda x: f"{x:,.2f} €".replace(",", " ").replace(".", ",")

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
            niveau = (charge - bas) / (haut - bas) if haut > bas else 0
            couleur = to_hex(palette(0.08 + 0.52 * niveau))
            texte = f'{pct:.1f} %'.replace('.', ',')
            detail = f'Charge totale : {euros(charge)}'
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
<h2>Matrice des sinistres</h2>
<p>Chaque case : % des {total} contrats distincts présents dans df.</p>
<div class="matrice">{tableau}</div>
<p>Bleu clair → bleu foncé : charge totale de {euros(bas)} à {euros(haut)}.
Survol d’une case : charge exacte.</p>
<p>Un contrat peut apparaître dans plusieurs cases : les pourcentages
ne s’additionnent donc pas nécessairement à 100 %.</p>
</html>"""
    Path(sortie).write_text(page, encoding="utf-8")
    return pourcentages, charges

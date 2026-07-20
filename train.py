# -*- coding: utf-8 -*-
"""
generer_rapport.py
==================
Générateur du rapport HTML de monitoring des renouvellements auto.

Principe
--------
Chaque contrôle, table ou visuel est produit par UNE fonction Python.
Les fonctions sont enregistrées dans le PIPELINE (en bas du fichier),
section par section, dans l'ordre d'affichage souhaité.

Pour AJOUTER un contrôle   : écrire une fonction @checkpoint et l'ajouter au PIPELINE.
Pour DÉPLACER un contrôle  : déplacer sa ligne dans la liste du PIPELINE.
Pour RETIRER un contrôle   : retirer sa ligne du PIPELINE (la fonction peut rester).

Le script exécute toutes les fonctions, assemble la structure RAPPORT
et l'injecte dans le fichier HTML entre les marqueurs :
    // ==DEBUT_DONNEES==   et   // ==FIN_DONNEES==

Usage
-----
    python generer_rapport.py
(adapter CHEMIN_MODELE / CHEMIN_SORTIE si besoin, et brancher vos
DataFrames réels dans charger_donnees()).
"""

import json
import re
from datetime import datetime

CHEMIN_MODELE = "rapport_renouvellements.html"   # modèle (avec marqueurs)
CHEMIN_SORTIE = "rapport_renouvellements.html"   # fichier mis à jour (peut être identique)

# ----------------------------------------------------------------------
# 1. CHARGEMENT DES DONNÉES
#    Brancher ici vos sources réelles (SQL, parquet, csv, ...).
#    Le dictionnaire retourné est passé à chaque fonction du pipeline.
# ----------------------------------------------------------------------
def charger_donnees():
    # Exemple : return {"df": pd.read_parquet("renouvellements_202606.parquet")}
    return {}


# ----------------------------------------------------------------------
# 2. AIDES DE CONSTRUCTION
#    Chaque fonction du pipeline retourne un dict via l'une de ces aides.
# ----------------------------------------------------------------------
def checkpoint(libelle, statut, valeur, seuil, commentaire=""):
    """statut : 'ok' | 'alerte' | 'ko'"""
    assert statut in ("ok", "alerte", "ko"), f"statut invalide : {statut}"
    return {"_type": "checkpoint", "libelle": libelle, "statut": statut,
            "valeur": valeur, "seuil": seuil, "commentaire": commentaire}

def table(titre, colonnes, lignes):
    return {"_type": "table", "titre": titre, "colonnes": colonnes,
            "lignes": [[str(v) for v in l] for l in lignes]}

def visuel(titre, type_graph, labels, series):
    """type_graph : 'line' | 'bar' | 'doughnut'
       series : liste de {"nom": str, "valeurs": [nombres]}"""
    return {"_type": "visuel", "titre": titre, "type": type_graph,
            "labels": labels, "series": series}

def statut_seuil(valeur, seuil_ok, seuil_alerte=None, sens="max"):
    """Aide facultative : déduit le statut d'une valeur numérique.
    sens='max' -> KO si valeur > seuil ; sens='min' -> KO si valeur < seuil."""
    depasse = (valeur > seuil_ok) if sens == "max" else (valeur < seuil_ok)
    if not depasse:
        return "ok"
    if seuil_alerte is not None:
        grave = (valeur > seuil_alerte) if sens == "max" else (valeur < seuil_alerte)
        return "ko" if grave else "alerte"
    return "ko"


# ----------------------------------------------------------------------
# 3. FONCTIONS DE CALCUL — exemples pour la section GLOBALE
#    Remplacer le contenu par vos vrais calculs (df = donnees["df"], etc.)
# ----------------------------------------------------------------------
def verif_volumetrie(donnees):
    nb = 48_217
    ecart_pct = 1.8
    return checkpoint(
        libelle="Volumétrie du portefeuille vs mois précédent",
        statut=statut_seuil(abs(ecart_pct), seuil_ok=5),
        valeur=f"{nb:,}".replace(",", " "),
        seuil="écart < 5 %",
        commentaire=f"Écart de {ecart_pct:+.1f} % par rapport au mois précédent.",
    )

def verif_taux_renouvellement(donnees):
    taux = 87.4
    return checkpoint(
        libelle="Taux de renouvellement global",
        statut=statut_seuil(taux, seuil_ok=85, sens="min"),
        valeur=f"{taux:.1f} %",
        seuil="≥ 85 %",
    )

def table_synthese_mensuelle(donnees):
    return table(
        titre="Synthèse mensuelle des renouvellements",
        colonnes=["Mois", "À renouveler", "Renouvelés", "Taux (%)"],
        lignes=[["Mai 2026", "47 370", "41 290", "87,2"],
                ["Juin 2026", "48 217", "42 142", "87,4"]],
    )

def visuel_evolution_taux(donnees):
    return visuel(
        titre="Évolution du taux de renouvellement",
        type_graph="line",
        labels=["Janv.", "Févr.", "Mars", "Avr.", "Mai", "Juin"],
        series=[{"nom": "Taux (%)", "valeurs": [86.5, 86.3, 87.1, 87.0, 87.2, 87.4]},
                {"nom": "Objectif (%)", "valeurs": [85] * 6}],
    )

# ... écrire ici les fonctions des autres sections (agents, courtier, ...)


# ----------------------------------------------------------------------
# 4. PIPELINE — l'ordre des fonctions = l'ordre d'affichage
# ----------------------------------------------------------------------
PIPELINE = {
    "globale": {
        "libelle": "Globale",
        "description": "Portefeuille auto complet — tous réseaux confondus.",
        "kpis": lambda d: [
            {"label": "Contrats à renouveler", "valeur": "48 217"},
            {"label": "Taux de renouvellement", "valeur": "87,4 %"},
        ],
        "fonctions": [
            verif_volumetrie,
            verif_taux_renouvellement,
            table_synthese_mensuelle,
            visuel_evolution_taux,
        ],
    },
    "agents":   {"libelle": "Agents",   "description": "Réseau des agents généraux.",            "kpis": lambda d: [], "fonctions": []},
    "courtier": {"libelle": "Courtier", "description": "Portefeuille du réseau de courtage.",     "kpis": lambda d: [], "fonctions": []},
    "annexe":   {"libelle": "Annexe",   "description": "Canaux annexes et partenariats.",         "kpis": lambda d: [], "fonctions": []},
    "hors_ko":  {"libelle": "Hors KO",  "description": "Périmètre hors rejets techniques (KO).",  "kpis": lambda d: [], "fonctions": []},
}


# ----------------------------------------------------------------------
# 5. ASSEMBLAGE ET INJECTION — ne pas modifier
# ----------------------------------------------------------------------
def assembler(donnees):
    rapport = {
        "meta": {
            "periode": "Juin 2026",  # à paramétrer
            "genere_le": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "ordre_sections": list(PIPELINE.keys()),
        "sections": {},
    }
    for cle, cfg in PIPELINE.items():
        section = {
            "libelle": cfg["libelle"],
            "description": cfg["description"],
            "kpis": cfg["kpis"](donnees),
            "checkpoints": [], "tables": [], "visuels": [],
        }
        for fonction in cfg["fonctions"]:
            resultat = fonction(donnees)
            resultat["source_fonction"] = fonction.__name__
            genre = resultat.pop("_type")
            if genre == "checkpoint":
                resultat["fonction"] = resultat.pop("source_fonction")
                section["checkpoints"].append(resultat)
            elif genre == "table":
                section["tables"].append(resultat)
            elif genre == "visuel":
                section["visuels"].append(resultat)
        rapport["sections"][cle] = section
    return rapport


def injecter(rapport):
    with open(CHEMIN_MODELE, encoding="utf-8") as f:
        html = f.read()
    bloc = ("// ==DEBUT_DONNEES==\nconst RAPPORT = "
            + json.dumps(rapport, ensure_ascii=False, indent=2)
            + ";\n// ==FIN_DONNEES==")
    html, n = re.subn(r"// ==DEBUT_DONNEES==.*?// ==FIN_DONNEES==",
                      lambda _: bloc, html, flags=re.S)
    if n != 1:
        raise RuntimeError("Marqueurs de données introuvables dans le modèle HTML.")
    with open(CHEMIN_SORTIE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Rapport généré : {CHEMIN_SORTIE}")


if __name__ == "__main__":
    donnees = charger_donnees()
    injecter(assembler(donnees))

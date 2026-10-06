"""Scoring de contrats sinistres, par etapes. PySpark 3.5+.

Entree : UNE ligne par contrat et une meme date d'arrete pour tous les compteurs.
Les slots 1 a n contiennent tous les sinistres recents, sans trou, jusqu'a 4.
Les slots inutilises ont un statut NULL ou vide.
Les charges sont celles des sinistres sur 12 mois (pas 13 mois).
Les compteurs anciens concernent la fenetre 12-24 mois, pas un cumul sur 24 mois.
Un historique incomplet reste NULL et ne doit pas etre remplace par zero.

Colonnes attendues, ou a fournir dans correspondance :
id_contrat, nb_resp_12m, nb_non_resp_12m, nb_total_12_24m, nb_resp_12_24m,
statut_1, responsabilite_1, charge_1, ... jusqu'a statut_4, responsabilite_4, charge_4.

Statuts : EN COURS/ENCOURS/OUVERT ou CLOS/CLOTURE/FERME, accents acceptes.
Responsabilites : TOTAL/TOTALE/RESPONSABLE, PARTIEL/PARTIELLE,
NON RESPONSABLE/NR, ou un taux numerique selon unite_responsabilite.
Exemple en mode pourcentage : 50 ou '50%' = 0.5. 1 = 1%, PAS 100%.
PARTIELLE sans taux reel utilise responsabilite_partielle, hypothese configurable.
Charges : valeurs numeriques en euros, ou chaines numeriques avec point decimal.

Sortie : donnees canoniques, variables retenues, cinq scores, contributions,
coefficient_majoration, majoration_pct, statut_calcul et motifs_blocage.
HORS_PERIMETRE : coefficient 1, majoration 0, score global NULL.
NON_CALCULABLE : coefficient et score NULL. Aucun calcul sur les seuls 4 premiers
sinistres lorsque les compteurs indiquent davantage de sinistres.

Les poids, seuils et forfaits sont des hypotheses illustratives non calibrees.
Pas de UDF Python, pas de collect() ou d'ecriture automatique de la base.
"""

import math

from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F


POIDS_PAR_DEFAUT = {
    "statut": 0.10,
    "responsabilite": 0.25,
    "charge": 0.40,
    "nombre_12m": 0.175,
    "historique": 0.075,
}


def _col(nom):
    # Les backticks permettent aussi les noms de colonnes avec espaces ou points.
    return F.col("`" + nom.replace("`", "``") + "`")


def _nombre(nom):
    # TRY_CAST laisse les valeurs non convertibles a NULL, meme en mode ANSI.
    identifiant = "`" + nom.replace("`", "``") + "`"
    return F.expr("try_cast(" + identifiant + " as double)")


def _fini(colonne):
    return colonne.isNotNull() & (~F.isnan(colonne)) & (F.abs(colonne) != F.lit(float("inf")))


def _compteur_valide(colonne):
    return _fini(colonne) & (colonne >= 0) & (colonne == F.floor(colonne))


def _texte_code(nom):
    texte = F.upper(F.trim(_col(nom).cast("string")))
    texte = F.translate(texte, "AEIOU\u00c0\u00c1\u00c2\u00c4\u00c8\u00c9\u00ca\u00cb\u00cc\u00cd\u00ce\u00cf\u00d2\u00d3\u00d4\u00d6\u00d9\u00da\u00db\u00dc\u00c7", "AEIOUAAAAEEEEIIIIOOOOUUUUC")
    return F.regexp_replace(texte, r"[\s_\-]+", "")


def _borne_01(colonne):
    # LEAST/GREATEST ignorent les NULL : ce test evite de transformer un NULL en 1.
    return F.when(_fini(colonne), F.least(F.lit(1.0), F.greatest(F.lit(0.0), colonne)))


def calculer_scoring(
    df: DataFrame,
    *,
    charge_en_cours: float,
    correspondance=None,
    unite_responsabilite="pourcentage",
    responsabilite_partielle=0.50,
    poids_non_responsable=0.50,
    seuil_nombre_12m=2.0,
    seuil_historique=2.0,
    charge_echelle=1000.0,
    charge_reference=15000.0,
    historique_resp_cumul_24m=False,
    poids=None,
):
    """correspondance = {nom_canonique: nom_reel_dans_df}.

    charge_en_cours est un forfait PAR sinistre ouvert, pas par contrat.
    Si historique_resp_cumul_24m=True, le compteur responsable ancien fourni
    est un cumul 0-24 mois : on en soustrait les responsables 0-12 mois.
    Le compteur nb_total_12_24m reste toujours un compteur de la fenetre 12-24.
    Les zeros historiques sont acceptes si l'assurance et l'observation couvrent
    reellement la periode complete. Le code ne peut pas le deduire des compteurs.
    """

    # ETAPE 1 : verifier les parametres avant de construire les expressions Spark.
    p = dict(POIDS_PAR_DEFAUT if poids is None else poids)
    if set(p) != set(POIDS_PAR_DEFAUT):
        raise ValueError("Les poids doivent porter sur les cinq composantes attendues.")
    if any(not math.isfinite(float(v)) or v < 0 for v in p.values()):
        raise ValueError("Les poids doivent etre finis et positifs ou nuls.")
    if not math.isclose(sum(p.values()), 1.0, abs_tol=1e-9):
        raise ValueError("La somme des poids doit etre egale a 1.")
    if unite_responsabilite not in {"pourcentage", "taux"}:
        raise ValueError("unite_responsabilite doit etre 'pourcentage' ou 'taux'.")
    if not (0 < responsabilite_partielle < 1):
        raise ValueError("responsabilite_partielle doit etre strictement entre 0 et 1.")
    if not (0 <= poids_non_responsable <= 1):
        raise ValueError("poids_non_responsable doit etre entre 0 et 1.")
    for valeur in (seuil_nombre_12m, seuil_historique, charge_echelle, charge_reference):
        if not math.isfinite(float(valeur)) or valeur <= 0:
            raise ValueError("Les seuils et echelles doivent etre finis et strictement positifs.")
    if not math.isfinite(float(charge_en_cours)) or charge_en_cours < 0:
        raise ValueError("Le forfait en cours doit etre fini et positif ou nul.")

    # ETAPE 2 : renommer les colonnes pour rendre le reste du code lisible.
    compteurs = ["nb_resp_12m", "nb_non_resp_12m", "nb_total_12_24m", "nb_resp_12_24m"]
    colonnes = ["id_contrat"] + compteurs
    for j in range(1, 5):
        colonnes += [f"statut_{j}", f"responsabilite_{j}", f"charge_{j}"]
    noms = {nom: nom for nom in colonnes}
    if correspondance is not None:
        if set(correspondance) - set(noms):
            raise ValueError("La correspondance contient des noms canoniques inconnus.")
        noms.update(correspondance)
    manquantes = [source for source in noms.values() if source not in df.columns]
    if manquantes:
        raise ValueError("Colonnes manquantes : " + ", ".join(manquantes))
    if len(set(noms.values())) != len(noms):
        raise ValueError("Chaque variable doit correspondre a une colonne source distincte.")
    d = df.select(*[_col(noms[nom]).alias(nom) for nom in colonnes])

    # ETAPE 3 : convertir les compteurs sans remplacer les inconnus par zero.
    for nom in compteurs:
        d = d.withColumn(nom, _nombre(nom))
    d = d.withColumn("nb_sinistres_12m", _col("nb_resp_12m") + _col("nb_non_resp_12m"))

    # ETAPE 4 : distinguer un compteur ancien 12-24 mois d'un cumul 0-24 mois.
    d = d.withColumn("nb_resp_historique_source", _col("nb_resp_12_24m"))
    if historique_resp_cumul_24m:
        d = d.withColumn("nb_resp_12_24m", _col("nb_resp_12_24m") - _col("nb_resp_12m"))

    # ETAPE 5 : deduire le nombre ancien de sinistres non responsables.
    d = d.withColumn("nb_non_resp_12_24m", _col("nb_total_12_24m") - _col("nb_resp_12_24m"))

    # ETAPE 6 : identifier les contrats eligibles a la nouvelle majoration.
    recent_valide = _compteur_valide(_col("nb_resp_12m")) & _compteur_valide(_col("nb_non_resp_12m"))
    condition_eligibilite = (_col("nb_resp_12m") >= 1) | (_col("nb_non_resp_12m") >= 2)
    d = d.withColumn("eligible", F.when(recent_valide, condition_eligibilite))

    # ETAPE 7 : verifier qu'un contrat apparait une seule fois dans l'entree.
    fenetre_contrat = Window.partitionBy("id_contrat")
    d = d.withColumn("_sc_nb_lignes_id", F.count(F.lit(1)).over(fenetre_contrat))

    erreurs_details = []
    somme_encours = F.lit(0.0)
    somme_responsabilite = F.lit(0.0)
    somme_charge = F.lit(0.0)
    nombre_resp_detail = F.lit(0.0)
    nombre_nr_detail = F.lit(0.0)
    nombre_partielle_estimee = F.lit(0)

    # ETAPE 8 : appliquer la meme preparation aux quatre sinistres disponibles.
    for j in range(1, 5):
        actif = _col("nb_sinistres_12m") >= j

        # 8a : normaliser le statut de ce sinistre.
        statut = _texte_code(f"statut_{j}")
        encours = F.when(statut.isin("ENCOURS", "OUVERT"), F.lit(1.0))
        encours = encours.when(statut.isin("CLOS", "CLOTURE", "FERME"), F.lit(0.0))
        d = d.withColumn(f"encours_{j}", F.when(actif, encours).otherwise(F.lit(0.0)))

        # 8b : lire les taux numeriques, y compris les chaines '50%' ou '50,0%'.
        brut = F.trim(_col(f"responsabilite_{j}").cast("string"))
        brut = F.regexp_replace(brut, "%", "")
        brut = F.regexp_replace(brut, ",", ".")
        d = d.withColumn(f"_sc_resp_num_{j}", brut)
        d = d.withColumn(f"_sc_resp_num_{j}", _nombre(f"_sc_resp_num_{j}"))
        diviseur = 100.0 if unite_responsabilite == "pourcentage" else 1.0
        contient_pct = _col(f"responsabilite_{j}").cast("string").contains("%")
        taux = F.when(contient_pct, _col(f"_sc_resp_num_{j}") / 100.0)
        taux = taux.otherwise(_col(f"_sc_resp_num_{j}") / diviseur)

        # 8c : traduire aussi les categories de responsabilite.
        code_resp = _texte_code(f"responsabilite_{j}")
        partielle = code_resp.isin("PARTIEL", "PARTIELLE")
        resp = F.when(code_resp.isin("TOTAL", "TOTALE", "RESPONSABLE", "R"), F.lit(1.0))
        resp = resp.when(code_resp.isin("NONRESPONSABLE", "NONRESP", "NR"), F.lit(0.0))
        resp = resp.when(partielle, F.lit(float(responsabilite_partielle)))
        resp = resp.otherwise(taux)
        d = d.withColumn(f"responsabilite_norm_{j}", F.when(actif, resp).otherwise(F.lit(0.0)))

        # 8d : remplacer la charge ouverte par le forfait, garder la charge close.
        d = d.withColumn(f"charge_{j}", _nombre(f"charge_{j}"))
        charge = F.when(_col(f"encours_{j}") == 1, F.lit(float(charge_en_cours)))
        charge = charge.when(_col(f"encours_{j}") == 0, _col(f"charge_{j}"))
        d = d.withColumn(f"charge_retenue_{j}", F.when(actif, charge).otherwise(F.lit(0.0)))

        # 8e : une responsabilite ou un statut inconnu bloque un slot actif.
        statut_ok = _col(f"encours_{j}").isNotNull()
        resp_ok = _fini(_col(f"responsabilite_norm_{j}")) & _col(f"responsabilite_norm_{j}").between(0.0, 1.0)
        charge_ok = _fini(_col(f"charge_retenue_{j}")) & (_col(f"charge_retenue_{j}") >= 0)
        slot_ok = statut_ok & resp_ok & charge_ok
        erreurs_details.append(F.when(actif & (~slot_ok), F.lit(f"SINISTRE_{j}_INCOMPLET_OU_INVALIDE")))

        # 8f : un statut renseigne au-dela du nombre declare signale une incoherence.
        statut_present = _col(f"statut_{j}").isNotNull() & (F.trim(_col(f"statut_{j}").cast("string")) != "")
        erreurs_details.append(F.when((~actif) & statut_present, F.lit(f"SINISTRE_{j}_HORS_COMPTEUR")))

        # 8g : cumuler les informations, chaque sinistre actif compte une fois.
        somme_encours = somme_encours + _col(f"encours_{j}")
        somme_responsabilite = somme_responsabilite + _col(f"responsabilite_norm_{j}")
        somme_charge = somme_charge + _col(f"charge_retenue_{j}")
        nombre_resp_detail = nombre_resp_detail + F.when(actif & (_col(f"responsabilite_norm_{j}") > 0), 1.0).otherwise(0.0)
        nombre_nr_detail = nombre_nr_detail + F.when(actif & (_col(f"responsabilite_norm_{j}") == 0), 1.0).otherwise(0.0)
        nombre_partielle_estimee = nombre_partielle_estimee + F.when(actif & partielle, 1).otherwise(0)

    # ETAPE 9 : creer les agregats de contrat avant de normaliser les scores.
    d = d.withColumn("nb_encours_12m", somme_encours)
    d = d.withColumn("_sc_somme_responsabilite", somme_responsabilite)
    d = d.withColumn("charge_retenue_12m", somme_charge)
    d = d.withColumn("nb_resp_detail_12m", nombre_resp_detail)
    d = d.withColumn("nb_non_resp_detail_12m", nombre_nr_detail)
    d = d.withColumn("nb_responsabilites_partielles_estimees", nombre_partielle_estimee)

    # ETAPE 10 : verifier les compteurs, le detail et les limites de la base.
    ancien_valide = _compteur_valide(_col("nb_total_12_24m")) & _compteur_valide(_col("nb_resp_12_24m")) & _compteur_valide(_col("nb_resp_historique_source"))
    ancien_valide = ancien_valide & (_col("nb_non_resp_12_24m") >= 0)
    details_concordants = (_col("nb_resp_detail_12m") == _col("nb_resp_12m")) & (_col("nb_non_resp_detail_12m") == _col("nb_non_resp_12m"))
    id_invalide = _col("id_contrat").isNull() | (F.trim(_col("id_contrat").cast("string")) == "")
    erreurs = [F.when(id_invalide, "ID_CONTRAT_MANQUANT")]
    erreurs.append(F.when(_col("_sc_nb_lignes_id") > 1, "ID_CONTRAT_DUPLIQUE"))
    erreurs.append(F.when(~recent_valide, "COMPTEURS_12M_INVALIDES"))
    erreurs.append(F.when(_col("eligible") & (~ancien_valide), "HISTORIQUE_INCOMPLET_OU_INCOHERENT"))
    erreurs.append(F.when(_col("eligible") & (_col("nb_sinistres_12m") > 4), "PLUS_DE_4_SINISTRES_DETAIL_INCOMPLET"))
    erreurs.append(F.when(_col("eligible") & (_col("nb_sinistres_12m") <= 4) & (~details_concordants), "COMPTEURS_ET_RESPONSABILITES_INCOHERENTS"))
    erreurs.append(F.when(_col("eligible") & (~_fini(_col("charge_retenue_12m"))), "CHARGE_AGREGEE_INVALIDE"))
    erreurs += [F.when(_col("eligible"), erreur) for erreur in erreurs_details]
    d = d.withColumn("motifs_blocage", F.concat_ws("; ", *erreurs))

    # ETAPE 11 : classer le contrat avant de produire une majoration.
    d = d.withColumn("statut_calcul", F.when(_col("motifs_blocage") != "", "NON_CALCULABLE").when(~_col("eligible"), "HORS_PERIMETRE").otherwise("CALCULE"))
    calculable = _col("statut_calcul") == "CALCULE"
    denominateur = F.when(_col("nb_sinistres_12m") > 0, _col("nb_sinistres_12m"))

    # ETAPE 12 : score du statut = proportion de dossiers ouverts sur 12 mois.
    d = d.withColumn("score_statut", F.when(calculable, _col("nb_encours_12m") / denominateur))

    # ETAPE 13 : score de responsabilite = moyenne de tous les taux recents.
    d = d.withColumn("score_responsabilite", F.when(calculable, _col("_sc_somme_responsabilite") / denominateur))

    # ETAPE 14 : score de charge = logarithme de la somme des charges retenues.
    facteur_charge = math.log1p(charge_reference / charge_echelle)
    score_charge = F.log1p(_col("charge_retenue_12m") / float(charge_echelle)) / facteur_charge
    d = d.withColumn("score_charge", F.when(calculable, _borne_01(score_charge)))

    # ETAPE 15 : score du nombre de sinistres sur les 12 derniers mois.
    d = d.withColumn("nb_pondere_12m", _col("nb_resp_12m") + float(poids_non_responsable) * _col("nb_non_resp_12m"))
    score_nombre = _col("nb_pondere_12m") / float(seuil_nombre_12m)
    d = d.withColumn("score_nombre_12m", F.when(calculable, _borne_01(score_nombre)))

    # ETAPE 16 : score d'historique, uniquement sur la fenetre 12-24 mois.
    d = d.withColumn("nb_pondere_12_24m", _col("nb_resp_12_24m") + float(poids_non_responsable) * _col("nb_non_resp_12_24m"))
    score_historique = _col("nb_pondere_12_24m") / float(seuil_historique)
    d = d.withColumn("score_historique", F.when(calculable, _borne_01(score_historique)))

    # ETAPE 17 : afficher chaque contribution au score, sans aucun ELR.
    d = d.withColumn("contribution_statut", float(p["statut"]) * _col("score_statut"))
    d = d.withColumn("contribution_responsabilite", float(p["responsabilite"]) * _col("score_responsabilite"))
    d = d.withColumn("contribution_charge", float(p["charge"]) * _col("score_charge"))
    d = d.withColumn("contribution_nombre_12m", float(p["nombre_12m"]) * _col("score_nombre_12m"))
    d = d.withColumn("contribution_historique", float(p["historique"]) * _col("score_historique"))

    # ETAPE 18 : additionner les contributions, conserver la precision de calcul.
    score_global = _col("contribution_statut") + _col("contribution_responsabilite") + _col("contribution_charge") + _col("contribution_nombre_12m") + _col("contribution_historique")
    d = d.withColumn("score_global", F.when(calculable, _borne_01(score_global)))
    d = d.withColumn("score_sur_100", F.round(100.0 * _col("score_global"), 4))

    # ETAPE 19 : appliquer la borne metier uniquement aux contrats calculables.
    coefficient = F.lit(1.10) + F.lit(0.20) * _col("score_global")
    coefficient = F.when(calculable, coefficient)
    coefficient = coefficient.when(_col("statut_calcul") == "HORS_PERIMETRE", F.lit(1.0))
    d = d.withColumn("coefficient_majoration", coefficient)

    # ETAPE 20 : fournir aussi le taux en pourcentage pour la restitution.
    d = d.withColumn("majoration_pct", F.round(100.0 * (_col("coefficient_majoration") - 1.0), 4))
    d = d.withColumn("version_scoring", F.lit("SCORING_SINISTRES_V1_SANS_ELR"))

    # ETAPE 21 : masquer une somme partielle des charges si le detail est incomplet.
    d = d.withColumn("charge_retenue_12m", F.when(calculable, _col("charge_retenue_12m")))

    # ETAPE 22 : retirer les colonnes techniques, garder les autres pour audit.
    techniques = [nom for nom in d.columns if nom.startswith("_sc_")]
    return d.drop(*techniques)


# UTILISATION dans votre notebook, apres avoir cree votre DataFrame df :
#
# df_score = calculer_scoring(
#     df,
#     charge_en_cours=3000.0,              # Exemple de forfait PAR dossier ouvert.
#     unite_responsabilite="pourcentage", # 50 signifie 50%, 100 signifie 100%.
# )
#
# df_score.select(
#     "id_contrat",
#     "eligible",
#     "statut_calcul",
#     "score_statut",
#     "score_responsabilite",
#     "score_charge",
#     "score_nombre_12m",
#     "score_historique",
#     "score_sur_100",
#     "coefficient_majoration",
#     "majoration_pct",
#     "motifs_blocage",
# ).show(truncate=False)
#
# # Si vos noms reels sont differents, donnez une correspondance explicite :
# correspondance = {
#     "id_contrat": "ID contrat",
#     "nb_resp_12m": "votre_colonne_responsables_12m",
#     "nb_non_resp_12m": "votre_colonne_non_responsables_12m",
#     "nb_total_12_24m": "votre_colonne_total_12_24m",
#     "nb_resp_12_24m": "votre_colonne_responsables_12_24m",
#     "statut_1": "votre_colonne_statut_sinistre_1",
#     "responsabilite_1": "votre_colonne_responsabilite_sinistre_1",
#     "charge_1": "votre_colonne_charge_sinistre_1",
#     # Completer de meme pour les slots 2, 3 et 4.
# }
#
# df_score = calculer_scoring(df, charge_en_cours=3000.0, correspondance=correspondance)

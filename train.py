from IPython.display import HTML, display
from html import escape
from math import sqrt

# ── Données à remplacer par les vôtres ──────────────────────────────
segments = [
    dict(
        nom="Responsable + paiement",
        elr=112, resiliation=18,
        ressource_2026=0.90, ressource_2025=1.20,
        part_contrats=12, etiquette="dessus"
    ),
    dict(
        nom="NR répétés + paiement",
        elr=88, resiliation=15,
        ressource_2026=0.72, ressource_2025=0.80,
        part_contrats=8, etiquette="droite"
    ),
    dict(
        nom="Responsable seul",
        elr=103, resiliation=9,
        ressource_2026=3.90, ressource_2025=3.50,
        part_contrats=34, etiquette="dessous"
    ),
    dict(
        nom="NR répétés seuls",
        elr=78, resiliation=7,
        ressource_2026=3.00, ressource_2025=2.70,
        part_contrats=28, etiquette="dessous"
    ),
    dict(
        nom="Paiement seul",
        elr=66, resiliation=20,
        ressource_2026=1.20, ressource_2025=1.60,
        part_contrats=18, etiquette="droite"
    ),
]

# ── Style et dimensions du visuel ──────────────────────────────────
LARGEUR, HAUTEUR = 1600, 1020

MARINE = "#062D47"
BLEU   = "#006C9C"
BAISSE = "#B76557"
GRIS   = "#61727D"
GRILLE = "#DFE6EA"

# Bornes conservées pour reproduire exactement la disposition.
ELR_MIN, ELR_MAX = 54.5, 123.5
RESILIATION_MAX = 25

def format_nombre(valeur):
    return f"{valeur:.2f}".replace(".", ",")

def format_ecart(segment):
    ecart = segment["ressource_2026"] - segment["ressource_2025"]
    signe = "+" if ecart >= 0 else "−"
    return signe + format_nombre(abs(ecart))

def position_x(elr):
    return 190 + (elr - ELR_MIN) / (ELR_MAX - ELR_MIN) * 1310

def position_y(taux):
    return 820 - taux / RESILIATION_MAX * 580

def texte(contenu, x, y, taille=24, couleur=MARINE,
          ancrage="start", police="Arial", graisse=400):
    return (
        f'<text x="{x}" y="{y}" fill="{couleur}" '
        f'font-family="{police}" font-size="{taille}" '
        f'font-weight="{graisse}" text-anchor="{ancrage}">'
        f'{escape(str(contenu))}</text>'
    )

def ligne(x1, y1, x2, y2, couleur=GRILLE, epaisseur=1):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{couleur}" stroke-width="{epaisseur}"/>'
    )

def cercle(cx, cy, rayon, couleur):
    return (
        f'<circle cx="{cx}" cy="{cy}" '
        f'r="{rayon}" fill="{couleur}"/>'
    )

# ── Construction du graphique ──────────────────────────────────────
svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" '
    f'viewBox="0 0 {LARGEUR} {HAUTEUR}" '
    f'width="100%" role="img" '
    f'aria-label="ELR, résiliation et évolution de la ressource HT" '
    f'style="display:block; height:auto;">',
    f'<rect width="{LARGEUR}" height="{HAUTEUR}" fill="white"/>'
]

# Titre
svg += [
    texte(
        "Performance des segments et évolution de la ressource HT",
        70, 76, taille=43, police="Georgia"
    ),
    texte("2026 vs 2025 · Exemple illustratif", 73, 120, 23, GRIS),
    ligne(70, 150, 1530, 150, MARINE, 2),
]

# Légende
svg += [
    cercle(81, 187, 8, BLEU),
    texte("Hausse de ressource", 101, 195, 22, GRIS),
    cercle(401, 187, 8, BAISSE),
    texte("Baisse de ressource", 421, 195, 22, GRIS),
    texte(
        "Surface des bulles = part des contrats",
        1530, 195, 22, GRIS, "end"
    ),
]

# Grille et graduations de l'axe vertical
for valeur in [0, 5, 10, 15, 20, 25]:
    y = position_y(valeur)
    svg += [
        ligne(190, y, 1500, y),
        texte(valeur, 163, y + 8, 23, GRIS, "end"),
    ]

# Graduations de l'axe horizontal
for valeur in [60, 70, 80, 90, 100, 110, 120]:
    x = position_x(valeur)
    svg += [
        ligne(x, 820, x, 827, MARINE),
        texte(valeur, x, 856, 23, GRIS, "middle"),
    ]

# Titres des axes
svg += [
    ligne(190, 820, 1500, 820, MARINE, 1.4),
    texte("ELR (%)", 845, 901, 25, MARINE, "middle"),
    '<g transform="rotate(-90 85 535)">',
    texte("Taux de résiliation (%)", 85, 535, 25, MARINE, "middle"),
    "</g>",
]

# Bulles et étiquettes
for segment in segments:
    cx = position_x(segment["elr"])
    cy = position_y(segment["resiliation"])

    # La surface, et non le rayon, est proportionnelle à la part.
    rayon = 64 * sqrt(segment["part_contrats"] / 34)

    ecart = segment["ressource_2026"] - segment["ressource_2025"]
    couleur = BLEU if ecart >= 0 else BAISSE

    svg += [
        cercle(cx, cy, rayon, couleur),
        texte(
            f'{segment["part_contrats"]} %',
            cx, cy + 9, 26, "white", "middle"
        ),
    ]

    # Placement des étiquettes
    label_x = cx
    label_y = cy + rayon + 37
    ancrage = "middle"

    if segment["etiquette"] == "dessus":
        label_y = cy - rayon - 57

    elif segment["etiquette"] == "droite":
        label_x = cx + rayon + 23
        label_y = cy - 10
        ancrage = "start"

    montant = format_nombre(segment["ressource_2026"])
    variation = format_ecart(segment)

    svg += [
        texte(
            segment["nom"], label_x, label_y,
            25, MARINE, ancrage, graisse=500
        ),
        texte(
            f"{montant} M€ / {variation} M€",
            label_x, label_y + 33, 24, GRIS, ancrage
        ),
    ]

# Notes de lecture
svg += [
    ligne(70, 934, 1530, 934),
    texte(
        "Étiquettes : ressource HT 2026 en M€ / "
        "écart absolu avec 2025 en M€",
        72, 966, 21, GRIS
    ),
    texte(
        "Données fictives. Parts des contrats sur la population analysée. "
        "NR répétés = plus de 2 sinistres non responsables.",
        72, 998, 19, GRIS
    ),
    "</svg>",
]

# ── Affichage HTML dans le notebook ────────────────────────────────
html_graphique = (
    '<div style="width:100%; max-width:1600px; '
    'margin:0 auto; background:white;">'
    + "".join(svg)
    + "</div>"
)

display(HTML(html_graphique))
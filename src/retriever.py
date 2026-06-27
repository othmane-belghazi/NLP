"""
lapse_eda_report.py
====================
Générateur de rapport HTML de qualité de données & analyse exploratoire (EDA)
orienté MODÉLISATION DE LA RÉSILIATION (lapse) en assurance.

Utilisation rapide
------------------
    from lapse_eda_report import LapseEDAReport

    rep = LapseEDAReport(
        df,
        target="resiliation",          # flag binaire 0/1 (1 = résilié)
        cat_cols=None,                 # None -> auto-détection
        num_cols=None,                 # None -> auto-détection
        date_col=None,                 # ex: "date_effet" -> active la stabilité temporelle
        id_cols=["id_police"],         # colonnes à exclure de l'analyse
        title="Rapport EDA - Résiliation",
    )
    rep.generate("rapport_resiliation.html")

Le rapport HTML est autonome (images encodées en base64, aucune dépendance externe).

Dépendances : pandas, numpy, matplotlib, seaborn, scipy
"""

from __future__ import annotations

import base64
import html
import io
import warnings
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  PALETTE / STYLE GRAPHIQUE (identité visuelle "actuariel / institutionnel")
# --------------------------------------------------------------------------- #
INK = "#1b2a3a"        # encre / texte
SLATE = "#3d5a73"      # bleu ardoise (exposition / barres)
TEAL = "#0e7c7b"       # teal secondaire
SIGNAL = "#c8553d"     # rouge brique = SIGNAL de résiliation (risque)
AMBER = "#dd9933"      # alerte
GRID = "#dfe4ea"
BG = "#ffffff"
MUTED = "#6b7a8d"

PLOT_RC = {
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 110,
}


def _fig_to_base64(fig) -> str:
    """Convertit une figure matplotlib en balise <img> base64 autonome."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f'<img class="plot" src="data:image/png;base64,{b64}" loading="lazy"/>'


def _esc(x) -> str:
    return html.escape(str(x))


def _fmt(x, pct=False, dec=2):
    """Formatage compact et robuste des nombres."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    if pct:
        return f"{x*100:.{dec}f}%"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", " ")
    if isinstance(x, (float, np.floating)):
        if abs(x) >= 1000:
            return f"{x:,.0f}".replace(",", " ")
        return f"{x:.{dec}f}"
    return _esc(x)


def _human_bytes(n: float) -> str:
    for unit in ["o", "Ko", "Mo", "Go"]:
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} To"


# --------------------------------------------------------------------------- #
#  CLASSE PRINCIPALE
# --------------------------------------------------------------------------- #
class LapseEDAReport:
    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        cat_cols=None,
        num_cols=None,
        date_col=None,
        id_cols=None,
        title="Rapport EDA — Résiliation",
        max_cat_modalities=30,
        high_card_threshold=50,
        plot_sample=50000,
        n_bins=10,
        random_state=42,
    ):
        self.df = df.copy()
        self.target = target
        self.date_col = date_col
        self.id_cols = list(id_cols or [])
        self.title = title
        self.max_cat_modalities = max_cat_modalities
        self.high_card_threshold = high_card_threshold
        self.plot_sample = plot_sample
        self.n_bins = n_bins
        self.random_state = random_state
        self.alerts = []

        if target not in df.columns:
            raise ValueError(f"Cible '{target}' absente du DataFrame.")

        # --- Cible : on force un flag binaire 0/1 ---
        y = self.df[target]
        if y.dtype == bool:
            self.df[target] = y.astype(int)
        uniq = self.df[target].dropna().unique()
        if not set(np.unique(uniq)).issubset({0, 1}):
            # Tentative de mapping si 2 modalités texte type Oui/Non
            if self.df[target].nunique(dropna=True) == 2:
                vals = sorted(self.df[target].dropna().unique(), key=str)
                mapping = {vals[0]: 0, vals[1]: 1}
                self.alerts.append(
                    ("info", f"Cible recodée : {mapping} (1 = positif = résilié).")
                )
                self.df[target] = self.df[target].map(mapping)
            else:
                raise ValueError(
                    "La cible doit être binaire (0/1). "
                    f"Modalités trouvées : {uniq[:10]}"
                )
        self.target_rate = self.df[target].mean()

        # --- Détection automatique des types ---
        exclude = set(self.id_cols + [target] + ([date_col] if date_col else []))
        feats = [c for c in self.df.columns if c not in exclude]

        if num_cols is None:
            num_cols = [
                c for c in feats
                if pd.api.types.is_numeric_dtype(self.df[c])
                and self.df[c].nunique(dropna=True) > 10
            ]
        if cat_cols is None:
            cat_cols = [
                c for c in feats
                if c not in num_cols
            ]
        self.num_cols = [c for c in num_cols if c in self.df.columns]
        self.cat_cols = [c for c in cat_cols if c in self.df.columns]

        # Normalisation des catégorielles en chaîne + remplissage MISSING explicite
        for c in self.cat_cols:
            self.df[c] = self.df[c].astype("object")

        if date_col and date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")

    # ------------------------------------------------------------------ #
    #  OUTILS STATISTIQUES
    # ------------------------------------------------------------------ #
    @staticmethod
    def _information_value(feature: pd.Series, y: pd.Series, bins=10):
        """IV/WoE. Variable continue -> binning quantile ; sinon par modalité."""
        s = feature.copy()
        df = pd.DataFrame({"x": s, "y": y}).dropna(subset=["y"])
        if pd.api.types.is_numeric_dtype(df["x"]) and df["x"].nunique() > bins:
            df["grp"] = pd.qcut(df["x"], q=bins, duplicates="drop")
        else:
            df["grp"] = df["x"].fillna("MISSING").astype(str)
        grp = df.groupby("grp", observed=True)["y"].agg(["count", "sum"])
        grp.columns = ["total", "bad"]
        grp["good"] = grp["total"] - grp["bad"]
        tot_bad = max(grp["bad"].sum(), 1)
        tot_good = max(grp["good"].sum(), 1)
        grp["dist_bad"] = grp["bad"] / tot_bad
        grp["dist_good"] = grp["good"] / tot_good
        eps = 1e-6
        grp["woe"] = np.log(
            (grp["dist_good"] + eps) / (grp["dist_bad"] + eps)
        )
        grp["iv"] = (grp["dist_good"] - grp["dist_bad"]) * grp["woe"]
        return float(grp["iv"].sum()), grp

    @staticmethod
    def _iv_label(iv):
        if iv < 0.02:
            return "inutile", MUTED
        if iv < 0.1:
            return "faible", SLATE
        if iv < 0.3:
            return "moyen", TEAL
        if iv < 0.5:
            return "fort", SIGNAL
        return "suspect (à vérifier)", AMBER

    @staticmethod
    def _wilson_ci(k, n, z=1.96):
        """Intervalle de confiance de Wilson pour une proportion."""
        if n == 0:
            return (np.nan, np.nan)
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        marge = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return (max(0, centre - marge), min(1, centre + marge))

    def _plot_df(self):
        """Échantillon pour les graphiques lourds (perf sur gros volumes)."""
        if len(self.df) > self.plot_sample:
            return self.df.sample(self.plot_sample, random_state=self.random_state)
        return self.df

    # ================================================================== #
    #  SECTION 1 — VUE D'ENSEMBLE
    # ================================================================== #
    def _section_overview(self):
        n, p = self.df.shape
        mem = self.df.memory_usage(deep=True).sum()
        n_dup = int(self.df.duplicated().sum())
        n_resil = int(self.df[self.target].sum())
        lo, hi = self._wilson_ci(n_resil, n)
        dtypes = self.df.dtypes.astype(str).value_counts()

        cards = [
            ("Polices (lignes)", _fmt(n), "observations"),
            ("Variables (colonnes)", _fmt(p), f"{len(self.num_cols)} num · {len(self.cat_cols)} cat"),
            ("Empreinte mémoire", _human_bytes(mem), f"≈ {_human_bytes(mem/n)} / ligne"),
            ("Doublons", _fmt(n_dup), _fmt(n_dup / n, pct=True) + " du portefeuille"),
        ]
        kpi = '<div class="kpi-grid">'
        for lab, val, sub in cards:
            kpi += (
                f'<div class="kpi"><div class="kpi-label">{lab}</div>'
                f'<div class="kpi-value">{val}</div>'
                f'<div class="kpi-sub">{sub}</div></div>'
            )
        # carte signature : taux de résiliation global
        kpi += (
            f'<div class="kpi kpi-signal"><div class="kpi-label">Taux de résiliation portefeuille</div>'
            f'<div class="kpi-value">{_fmt(self.target_rate, pct=True)}</div>'
            f'<div class="kpi-sub">{_fmt(n_resil)} résiliés · IC95% [{_fmt(lo,pct=True)} – {_fmt(hi,pct=True)}]</div></div>'
        )
        kpi += "</div>"

        types_rows = "".join(
            f"<tr><td>{_esc(t)}</td><td class='num'>{_fmt(int(c))}</td></tr>"
            for t, c in dtypes.items()
        )
        types_tbl = (
            "<table class='mini'><thead><tr><th>Type</th><th>Nb colonnes</th></tr></thead>"
            f"<tbody>{types_rows}</tbody></table>"
        )
        return self._wrap_section(
            "1", "overview", "Vue d'ensemble",
            kpi
            + "<div class='two-col'><div>" + types_tbl + "</div>"
            + f"<div class='note'>Période d'analyse : "
            + (f"{self.df[self.date_col].min():%d/%m/%Y} → {self.df[self.date_col].max():%d/%m/%Y}"
               if self.date_col else "non renseignée")
            + f"<br>Cible : <code>{_esc(self.target)}</code> (1 = résilié)."
            + "</div></div>",
        )

    # ================================================================== #
    #  SECTION 2 — QUALITÉ DES DONNÉES
    # ================================================================== #
    def _section_quality(self):
        rows = []
        n = len(self.df)
        for c in self.df.columns:
            s = self.df[c]
            n_miss = int(s.isna().sum())
            pct_miss = n_miss / n
            n_uni = int(s.nunique(dropna=True))
            mem = s.memory_usage(deep=True)
            role = ("cible" if c == self.target else
                    "id" if c in self.id_cols else
                    "date" if c == self.date_col else
                    "num" if c in self.num_cols else
                    "cat" if c in self.cat_cols else "—")
            # % modalité MISSING pour les catégorielles
            pct_missing_mod = ""
            if c in self.cat_cols:
                pct_missing_mod = _fmt(pct_miss, pct=True)
            rows.append((c, role, str(s.dtype), n_uni, n_miss, pct_miss,
                         pct_missing_mod, _human_bytes(mem)))

            # Alertes automatiques
            if pct_miss > 0.30 and role in ("num", "cat"):
                self.alerts.append(("warn", f"<code>{c}</code> : {_fmt(pct_miss,pct=True)} de valeurs manquantes."))
            if role == "cat" and n_uni > self.high_card_threshold:
                self.alerts.append(("warn", f"<code>{c}</code> : cardinalité élevée ({n_uni} modalités)."))
            if n_uni <= 1 and role in ("num", "cat"):
                self.alerts.append(("warn", f"<code>{c}</code> : variable constante (à retirer)."))

        rows.sort(key=lambda r: r[5], reverse=True)
        body = ""
        for c, role, dt, nu, nm, pm, pmm, mem in rows:
            bar = self._missbar(pm)
            body += (
                f"<tr><td class='colname'>{_esc(c)}</td>"
                f"<td><span class='tag tag-{role}'>{role}</span></td>"
                f"<td>{_esc(dt)}</td>"
                f"<td class='num'>{_fmt(nu)}</td>"
                f"<td class='num'>{_fmt(nm)}</td>"
                f"<td class='num'>{bar}</td>"
                f"<td class='num'>{pmm}</td>"
                f"<td class='num muted'>{mem}</td></tr>"
            )
        tbl = (
            "<table class='data'><thead><tr>"
            "<th>Variable</th><th>Rôle</th><th>Type</th><th>Nb modalités</th>"
            "<th>NaN</th><th>% manquant</th><th>% MISSING (cat)</th><th>Mémoire</th>"
            "</tr></thead><tbody>" + body + "</tbody></table>"
        )

        # Heatmap des valeurs manquantes (motifs)
        miss_cols = [c for c in self.df.columns
                     if self.df[c].isna().any() and c not in self.id_cols]
        heat = ""
        if miss_cols:
            with plt.rc_context(PLOT_RC):
                d = self._plot_df()[miss_cols].isna()
                fig, ax = plt.subplots(figsize=(min(11, 1 + 0.5 * len(miss_cols)), 3.2))
                sns.heatmap(d.T, cbar=False, cmap=["#eef2f6", SIGNAL], ax=ax)
                ax.set_title("Cartographie des valeurs manquantes (lignes = variables)")
                ax.set_xlabel("Observations"); ax.set_xticks([]); ax.set_yticklabels(
                    [_esc(c) for c in miss_cols], rotation=0, fontsize=8)
                heat = _fig_to_base64(fig)
        else:
            heat = "<p class='note'>Aucune valeur manquante détectée. ✓</p>"

        return self._wrap_section(
            "2", "quality", "Qualité des données",
            "<p class='lead'>Type, taille, cardinalité, mémoire et manquants par variable. "
            "Le bandeau récapitule type de données, espace mémoire et représentation des modalités.</p>"
            + tbl + heat,
        )

    # ================================================================== #
    #  SECTION 3 — VARIABLES CONTINUES
    # ================================================================== #
    def _continuous_stats(self, s):
        x = s.dropna()
        if len(x) == 0:
            return None
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((x < lo) | (x > hi)).sum())
        mean = x.mean(); std = x.std()
        return {
            "count": int(len(x)), "n_miss": int(s.isna().sum()),
            "mean": mean, "std": std,
            "cv": std / mean if mean else np.nan,
            "min": x.min(), "q1": q1, "median": med, "q3": q3, "max": x.max(),
            "iqr": iqr, "skew": stats.skew(x), "kurt": stats.kurtosis(x),
            "n_out": n_out, "pct_out": n_out / len(x),
        }

    def _section_continuous(self):
        if not self.num_cols:
            return ""
        # Tableau de statistiques descriptives
        cols = ["count", "n_miss", "mean", "std", "cv", "min", "q1", "median",
                "q3", "max", "iqr", "skew", "kurt", "n_out", "pct_out"]
        labels = ["N", "NaN", "Moy.", "Éc.-type", "CV", "Min", "Q1", "Médiane",
                  "Q3", "Max", "IQR", "Skew", "Kurt.", "Outliers", "% outl."]
        body = ""
        stats_map = {}
        for c in self.num_cols:
            st = self._continuous_stats(self.df[c])
            if st is None:
                continue
            stats_map[c] = st
            tds = f"<td class='colname'>{_esc(c)}</td>"
            for k in cols:
                v = st[k]
                if k in ("skew", "kurt"):
                    cls = "num warn-cell" if abs(v) > 1 else "num"
                    tds += f"<td class='{cls}'>{_fmt(v)}</td>"
                elif k == "pct_out":
                    tds += f"<td class='num'>{_fmt(v, pct=True)}</td>"
                elif k == "cv":
                    tds += f"<td class='num'>{_fmt(v)}</td>"
                else:
                    tds += f"<td class='num'>{_fmt(v)}</td>"
            body += f"<tr>{tds}</tr>"
            if abs(st["skew"]) > 1:
                self.alerts.append(("info", f"<code>{c}</code> : distribution asymétrique (skew={_fmt(st['skew'])}), une transformation log peut aider."))
        head = "<th>Variable</th>" + "".join(f"<th>{l}</th>" for l in labels)
        tbl = (f"<table class='data'><thead><tr>{head}</tr></thead>"
               f"<tbody>{body}</tbody></table>")

        # Distributions : histogramme+KDE & boxplot côte à côte
        plots = ""
        pdf = self._plot_df()
        for c in self.num_cols:
            with plt.rc_context(PLOT_RC):
                fig, (ax1, ax2) = plt.subplots(
                    1, 2, figsize=(9, 2.9), gridspec_kw={"width_ratios": [3, 1]})
                x = pdf[c].dropna()
                sns.histplot(x, kde=True, ax=ax1, color=SLATE,
                             edgecolor="white", line_kws={"color": SIGNAL})
                st = stats_map[c]
                ax1.axvline(st["median"], color=SIGNAL, ls="--", lw=1.2, label="médiane")
                ax1.axvline(st["mean"], color=TEAL, ls=":", lw=1.2, label="moyenne")
                ax1.legend(fontsize=8, frameon=False)
                ax1.set_title(f"Distribution — {c}"); ax1.set_ylabel("Fréquence")
                sns.boxplot(y=x, ax=ax2, color=SLATE, width=0.5,
                            flierprops=dict(marker="o", markersize=3,
                                            markerfacecolor=SIGNAL, alpha=0.4))
                ax2.set_title("Boxplot"); ax2.set_ylabel("")
                plots += f"<div class='plot-card'>{_fig_to_base64(fig)}</div>"

        return self._wrap_section(
            "3", "continuous", "Variables continues",
            "<p class='lead'>Statistiques descriptives (min, max, médiane, IQR, skewness, "
            "kurtosis, coefficient de variation) puis distribution + boxplot par variable. "
            "Les valeurs de skew/kurtosis &gt; 1 en valeur absolue sont surlignées.</p>"
            + "<div class='scroll'>" + tbl + "</div>"
            + "<div class='plot-row'>" + plots + "</div>",
        )

    # ================================================================== #
    #  SECTION 4 — VARIABLES CATÉGORIELLES
    # ================================================================== #
    def _section_categorical(self):
        if not self.cat_cols:
            return ""
        blocks = ""
        n = len(self.df)
        for c in self.cat_cols:
            s = self.df[c]
            vc = s.fillna("MISSING").astype(str).value_counts(dropna=False)
            n_uni = s.nunique(dropna=True)
            pct_missing = s.isna().mean()
            shown = vc.head(self.max_cat_modalities)
            rows = ""
            for mod, cnt in shown.items():
                pct = cnt / n
                is_miss = mod == "MISSING"
                rows += (
                    f"<tr class='{'miss-row' if is_miss else ''}'>"
                    f"<td>{_esc(mod)}</td>"
                    f"<td class='num'>{_fmt(int(cnt))}</td>"
                    f"<td class='num'>{_fmt(pct, pct=True)}</td>"
                    f"<td>{self._propbar(pct)}</td></tr>"
                )
            extra = ""
            if len(vc) > self.max_cat_modalities:
                extra = f"<p class='note'>… {len(vc)-self.max_cat_modalities} modalités supplémentaires (queue) non affichées.</p>"
            tbl = (
                "<table class='data compact'><thead><tr><th>Modalité</th><th>N</th>"
                "<th>%</th><th>Représentation</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>{extra}"
            )
            # bar chart distribution
            with plt.rc_context(PLOT_RC):
                top = shown[::-1]
                fig, ax = plt.subplots(figsize=(4.6, max(2.2, 0.32 * len(top) + 0.6)))
                colors = [SIGNAL if m == "MISSING" else SLATE for m in top.index]
                ax.barh([_esc(m)[:24] for m in top.index], top.values, color=colors)
                ax.set_title(f"{c} — {n_uni} modalités · MISSING {_fmt(pct_missing,pct=True)}")
                ax.set_xlabel("Effectif")
                plot = _fig_to_base64(fig)
            blocks += (
                f"<div class='cat-block'><h3>{_esc(c)}</h3>"
                f"<div class='two-col'><div class='scroll'>{tbl}</div>"
                f"<div class='plot-card'>{plot}</div></div></div>"
            )
        return self._wrap_section(
            "4", "categorical", "Variables catégorielles",
            "<p class='lead'>Pour chaque variable : représentation de chaque modalité "
            "(effectif et %), part de la modalité <code>MISSING</code> (surlignée), "
            "cardinalité et histogramme de répartition.</p>" + blocks,
        )

    # ================================================================== #
    #  SECTION 5 — CORRÉLATIONS
    # ================================================================== #
    @staticmethod
    def _cramers_v(x, y):
        cm = pd.crosstab(x, y)
        if cm.size == 0 or cm.shape[0] < 2 or cm.shape[1] < 2:
            return np.nan
        chi2 = stats.chi2_contingency(cm)[0]
        n = cm.sum().sum()
        phi2 = chi2 / n
        r, k = cm.shape
        phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        rc = r - (r - 1) ** 2 / (n - 1)
        kc = k - (k - 1) ** 2 / (n - 1)
        denom = min(kc - 1, rc - 1)
        return np.sqrt(phi2c / denom) if denom > 0 else np.nan

    def _section_correlation(self):
        out = "<p class='lead'>Corrélations de Pearson et Spearman entre variables numériques, " \
              "association de Cramér's V entre variables catégorielles, et paires redondantes.</p>"
        num = self.num_cols
        if len(num) >= 2:
            corr_p = self.df[num].corr(method="pearson")
            corr_s = self.df[num].corr(method="spearman")
            with plt.rc_context(PLOT_RC):
                fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 0.5 * len(num))))
                for ax, corr, name in zip(axes, [corr_p, corr_s], ["Pearson", "Spearman"]):
                    sns.heatmap(corr, annot=len(num) <= 12, fmt=".2f", cmap="RdBu_r",
                                center=0, vmin=-1, vmax=1, ax=ax, square=True,
                                cbar_kws={"shrink": 0.6},
                                annot_kws={"size": 7})
                    ax.set_title(f"Corrélation {name}")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
                out += _fig_to_base64(fig)
            # Paires fortement corrélées
            pairs = []
            for i in range(len(num)):
                for j in range(i + 1, len(num)):
                    r = corr_p.iloc[i, j]
                    if abs(r) >= 0.7:
                        pairs.append((num[i], num[j], r))
            if pairs:
                pairs.sort(key=lambda t: -abs(t[2]))
                rows = "".join(
                    f"<tr><td>{_esc(a)}</td><td>{_esc(b)}</td>"
                    f"<td class='num warn-cell'>{_fmt(r)}</td></tr>"
                    for a, b, r in pairs
                )
                out += ("<h3>Paires redondantes (|r| ≥ 0,70)</h3>"
                        "<table class='data compact'><thead><tr><th>Variable A</th>"
                        "<th>Variable B</th><th>Pearson r</th></tr></thead>"
                        f"<tbody>{rows}</tbody></table>")
                for a, b, r in pairs:
                    self.alerts.append(("info", f"Multicolinéarité : <code>{a}</code> ↔ <code>{b}</code> (r={_fmt(r)})."))
        else:
            out += "<p class='note'>Moins de 2 variables numériques : matrice non calculée.</p>"

        # Cramér's V catégorielles
        cats = [c for c in self.cat_cols if self.df[c].nunique() <= self.high_card_threshold]
        if len(cats) >= 2:
            mat = pd.DataFrame(index=cats, columns=cats, dtype=float)
            for i, a in enumerate(cats):
                for b in cats[i:]:
                    v = 1.0 if a == b else self._cramers_v(self.df[a], self.df[b])
                    mat.loc[a, b] = mat.loc[b, a] = v
            with plt.rc_context(PLOT_RC):
                fig, ax = plt.subplots(figsize=(min(9, 1 + 0.6 * len(cats)),
                                                min(8, 1 + 0.6 * len(cats))))
                sns.heatmap(mat.astype(float), annot=len(cats) <= 12, fmt=".2f",
                            cmap="Greens", vmin=0, vmax=1, ax=ax, square=True,
                            cbar_kws={"shrink": 0.6}, annot_kws={"size": 7})
                ax.set_title("Association catégorielle — Cramér's V")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
                out += _fig_to_base64(fig)
        return self._wrap_section("5", "correlation", "Corrélations & associations", out)

    # ================================================================== #
    #  SECTION 6 — TAUX DE RÉSILIATION (cœur du rapport)
    # ================================================================== #
    def _dual_chart(self, labels, exposure_pct, rate, counts, title, xlabel,
                    rotate=False):
        """Barres = part du portefeuille ; courbe = taux de résiliation."""
        with plt.rc_context(PLOT_RC):
            fig, ax1 = plt.subplots(figsize=(7.6, 3.4))
            xpos = np.arange(len(labels))
            ax1.bar(xpos, exposure_pct, color=SLATE, alpha=0.55, width=0.7,
                    label="% du portefeuille")
            ax1.set_ylabel("% du portefeuille", color=SLATE)
            ax1.set_ylim(0, max(exposure_pct) * 1.25 if len(exposure_pct) else 1)
            ax1.set_xticks(xpos)
            ax1.set_xticklabels([str(l)[:18] for l in labels],
                                rotation=45 if rotate else 0,
                                ha="right" if rotate else "center", fontsize=8)
            ax1.grid(False)
            ax2 = ax1.twinx()
            ax2.plot(xpos, rate, color=SIGNAL, marker="o", lw=2, ms=5,
                     label="taux de résiliation")
            ax2.axhline(self.target_rate, color=INK, ls="--", lw=1.1,
                        label=f"moyenne portefeuille ({_fmt(self.target_rate,pct=True)})")
            ax2.set_ylabel("Taux de résiliation", color=SIGNAL)
            ax2.set_ylim(0, max(max(rate) * 1.25, self.target_rate * 1.6) if len(rate) else 1)
            ax2.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
            ax2.grid(False)
            # annotations effectifs
            for x, r, c in zip(xpos, rate, counts):
                ax2.annotate(f"{r*100:.1f}%", (x, r), textcoords="offset points",
                             xytext=(0, 7), ha="center", fontsize=7, color=SIGNAL)
            ax1.set_title(title)
            ax1.set_xlabel(xlabel)
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False,
                       loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
            return _fig_to_base64(fig)

    def _section_target(self):
        out = ("<p class='lead'>Croisement de la cible avec chaque variable. "
               "Les <b>barres</b> donnent la part du portefeuille (exposition) de chaque "
               "modalité / tranche ; la <b>courbe rouge</b> donne le taux de résiliation ; "
               "la <b>ligne pointillée</b> rappelle la moyenne portefeuille. "
               "Un écart fort à la moyenne = variable discriminante.</p>")
        y = self.df[self.target]
        n = len(self.df)

        # --- Catégorielles : par modalité ---
        if self.cat_cols:
            out += "<h3>Croisement avec les variables catégorielles (par modalité)</h3>"
            cards = ""
            for c in self.cat_cols:
                s = self.df[c].fillna("MISSING").astype(str)
                g = pd.DataFrame({"x": s, "y": y}).groupby("x", observed=True)["y"].agg(
                    ["count", "mean", "sum"])
                g = g.sort_values("mean", ascending=False).head(self.max_cat_modalities)
                labels = g.index.tolist()
                exp = (g["count"] / n * 100).tolist()
                rate = g["mean"].tolist()
                cnt = g["count"].tolist()
                img = self._dual_chart(labels, exp, rate, cnt,
                                       f"Résiliation × {c}", c, rotate=len(labels) > 4)
                iv, _ = self._information_value(self.df[c], y)
                lab, col = self._iv_label(iv)
                cards += (f"<div class='plot-card'>{img}"
                          f"<div class='iv-badge' style='color:{col}'>IV = {_fmt(iv)} · {lab}</div></div>")
            out += "<div class='plot-row'>" + cards + "</div>"

        # --- Continues : par tranche (quantiles) ---
        if self.num_cols:
            out += "<h3>Croisement avec les variables continues (par tranche)</h3>"
            cards = ""
            for c in self.num_cols:
                x = self.df[c]
                try:
                    binned = pd.qcut(x, q=self.n_bins, duplicates="drop")
                except Exception:
                    binned = pd.cut(x, bins=self.n_bins)
                tmp = pd.DataFrame({"bin": binned, "y": y})
                # tranche MISSING explicite
                miss_mask = x.isna()
                g = tmp.dropna(subset=["bin"]).groupby("bin", observed=True)["y"].agg(
                    ["count", "mean"])
                labels = [f"{iv.left:.0f}–{iv.right:.0f}" if abs(iv.right) >= 10
                          else f"{iv.left:.2f}–{iv.right:.2f}" for iv in g.index]
                exp = (g["count"] / n * 100).tolist()
                rate = g["mean"].tolist()
                cnt = g["count"].tolist()
                if miss_mask.any():
                    labels.append("MISSING")
                    exp.append(miss_mask.sum() / n * 100)
                    rate.append(y[miss_mask].mean())
                    cnt.append(int(miss_mask.sum()))
                img = self._dual_chart(labels, exp, rate, cnt,
                                       f"Résiliation × {c} (déciles)", c, rotate=True)
                iv, _ = self._information_value(x, y, bins=self.n_bins)
                lab, col = self._iv_label(iv)
                cards += (f"<div class='plot-card'>{img}"
                          f"<div class='iv-badge' style='color:{col}'>IV = {_fmt(iv)} · {lab}</div></div>")
            out += "<div class='plot-row'>" + cards + "</div>"

        return self._wrap_section("6", "target", "Taux de résiliation & pouvoir discriminant", out)

    # ================================================================== #
    #  SECTION 7 — POUVOIR PRÉDICTIF (Information Value) [proposition]
    # ================================================================== #
    def _section_predictive(self):
        y = self.df[self.target]
        rows_data = []
        for c in self.num_cols + self.cat_cols:
            try:
                iv, _ = self._information_value(self.df[c], y, bins=self.n_bins)
            except Exception:
                iv = np.nan
            kind = "num" if c in self.num_cols else "cat"
            rows_data.append((c, kind, iv))
        rows_data.sort(key=lambda t: (-(t[2] if not np.isnan(t[2]) else -1)))

        # Graphique classement IV
        with plt.rc_context(PLOT_RC):
            top = rows_data[:25][::-1]
            fig, ax = plt.subplots(figsize=(8, max(2.5, 0.32 * len(top) + 0.5)))
            ivs = [t[2] for t in top]
            cols = [self._iv_label(v)[1] for v in ivs]
            ax.barh([t[0] for t in top], ivs, color=cols)
            for thr, lab in [(0.02, "inutile"), (0.1, "faible"),
                             (0.3, "moyen"), (0.5, "fort")]:
                ax.axvline(thr, color=MUTED, ls=":", lw=0.8)
            ax.set_title("Classement des variables par Information Value (IV)")
            ax.set_xlabel("IV")
            img = _fig_to_base64(fig)

        body = ""
        for c, kind, iv in rows_data:
            lab, col = self._iv_label(iv)
            body += (f"<tr><td class='colname'>{_esc(c)}</td>"
                     f"<td><span class='tag tag-{kind}'>{kind}</span></td>"
                     f"<td class='num'>{_fmt(iv)}</td>"
                     f"<td style='color:{col};font-weight:600'>{lab}</td></tr>")
        tbl = ("<table class='data'><thead><tr><th>Variable</th><th>Type</th>"
               "<th>IV</th><th>Pouvoir prédictif</th></tr></thead>"
               f"<tbody>{body}</tbody></table>")
        return self._wrap_section(
            "7", "predictive", "Pouvoir prédictif — Information Value",
            "<p class='lead'>L'<b>Information Value</b> (IV) mesure le pouvoir discriminant "
            "de chaque variable vis-à-vis de la résiliation (référence du scoring assurantiel). "
            "Grille usuelle : &lt;0,02 inutile · 0,02–0,1 faible · 0,1–0,3 moyen · "
            "0,3–0,5 fort · &gt;0,5 suspect (risque de fuite de cible / sur-ajustement).</p>"
            + img + "<div class='scroll'>" + tbl + "</div>",
        )

    # ================================================================== #
    #  SECTION 8 — STABILITÉ TEMPORELLE (si date fournie) [proposition]
    # ================================================================== #
    def _section_temporal(self):
        if not self.date_col or self.date_col not in self.df.columns:
            return ""
        d = self.df[[self.date_col, self.target]].dropna(subset=[self.date_col])
        if d.empty:
            return ""
        d["periode"] = d[self.date_col].dt.to_period("M").dt.to_timestamp()
        g = d.groupby("periode")[self.target].agg(["count", "mean"])
        with plt.rc_context(PLOT_RC):
            fig, ax1 = plt.subplots(figsize=(9, 3.4))
            ax1.bar(g.index, g["count"], width=20, color=SLATE, alpha=0.45,
                    label="exposition")
            ax1.set_ylabel("Effectif", color=SLATE)
            ax2 = ax1.twinx()
            ax2.plot(g.index, g["mean"], color=SIGNAL, marker="o", lw=2,
                     label="taux de résiliation")
            ax2.axhline(self.target_rate, color=INK, ls="--", lw=1)
            ax2.set_ylabel("Taux de résiliation", color=SIGNAL)
            ax2.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
            ax1.set_title("Évolution du taux de résiliation dans le temps")
            ax1.grid(False); ax2.grid(False)
            img = _fig_to_base64(fig)
        return self._wrap_section(
            "8", "temporal", "Stabilité temporelle",
            "<p class='lead'>Évolution mensuelle de l'exposition et du taux de résiliation. "
            "Permet de détecter une dérive (drift) du portefeuille ou de la cible — clé pour "
            "le découpage apprentissage / validation et la robustesse du modèle.</p>" + img,
        )

    # ================================================================== #
    #  SECTION 9 — SYNTHÈSE & RECOMMANDATIONS
    # ================================================================== #
    def _section_recommendations(self):
        # dédoublonnage des alertes en gardant l'ordre
        seen, uniq = set(), []
        for kind, msg in self.alerts:
            if msg not in seen:
                seen.add(msg); uniq.append((kind, msg))
        order = {"warn": 0, "info": 1}
        uniq.sort(key=lambda a: order.get(a[0], 2))
        items = "".join(
            f"<li class='alert alert-{k}'>{m}</li>" for k, m in uniq
        ) or "<li class='alert alert-info'>Aucun point d'attention automatique détecté.</li>"
        checklist = """
        <h3>Pistes de pré-traitement avant modélisation</h3>
        <ul class='check'>
          <li>Retirer les variables constantes / quasi-constantes et les identifiants.</li>
          <li>Traiter les variables à IV &gt; 0,5 (risque de fuite de cible / data leakage).</li>
          <li>Regrouper les modalités rares (&lt; 1 %) et la modalité MISSING si pertinente.</li>
          <li>Envisager une transformation (log, Yeo-Johnson) sur les variables très asymétriques.</li>
          <li>Gérer la multicolinéarité (paires |r| ≥ 0,7) — VIF ou sélection.</li>
          <li>Encoder via WoE les catégorielles à forte cardinalité plutôt qu'un one-hot massif.</li>
          <li>Vérifier l'équilibre de la cible et prévoir une stratégie (pondération / ré-échantillonnage).</li>
        </ul>"""
        return self._wrap_section(
            "9", "reco", "Synthèse & recommandations",
            "<p class='lead'>Points d'attention détectés automatiquement et pistes de "
            "pré-traitement.</p><ul class='alerts'>" + items + "</ul>" + checklist,
        )

    # ------------------------------------------------------------------ #
    #  PETITS COMPOSANTS HTML
    # ------------------------------------------------------------------ #
    @staticmethod
    def _missbar(pct):
        w = int(round(pct * 100))
        color = SIGNAL if pct > 0.3 else (AMBER if pct > 0.1 else SLATE)
        return (f"<span class='barwrap'><span class='bar' style='width:{w}%;"
                f"background:{color}'></span></span>"
                f"<span class='barval'>{_fmt(pct, pct=True)}</span>")

    @staticmethod
    def _propbar(pct):
        w = max(1, int(round(pct * 100)))
        return (f"<span class='barwrap'><span class='bar' style='width:{w}%;"
                f"background:{TEAL}'></span></span>")

    def _wrap_section(self, num, anchor, title, content):
        return (f"<section id='{anchor}'><div class='sec-head'>"
                f"<span class='sec-num'>{num}</span>"
                f"<h2>{_esc(title)}</h2></div>{content}</section>")

    # ------------------------------------------------------------------ #
    #  ASSEMBLAGE / CSS
    # ------------------------------------------------------------------ #
    def _css(self):
        return f"""
:root{{--ink:{INK};--slate:{SLATE};--teal:{TEAL};--signal:{SIGNAL};
--amber:{AMBER};--grid:{GRID};--muted:{MUTED};--bg:{BG};}}
*{{box-sizing:border-box;}}
body{{margin:0;font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',
Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:#f4f6f8;
line-height:1.5;font-size:14px;}}
.layout{{display:grid;grid-template-columns:248px 1fr;}}
/* --- TOC latérale --- */
nav.toc{{position:sticky;top:0;height:100vh;overflow-y:auto;background:var(--ink);
color:#cdd6e0;padding:26px 18px;}}
nav.toc .brand{{font-weight:700;font-size:15px;color:#fff;letter-spacing:.3px;
line-height:1.25;margin-bottom:4px;}}
nav.toc .brand small{{display:block;font-weight:400;font-size:11px;color:#8aa0b6;
margin-top:6px;letter-spacing:.5px;text-transform:uppercase;}}
nav.toc ol{{list-style:none;padding:0;margin:24px 0 0;counter-reset:s;}}
nav.toc li a{{display:flex;gap:10px;align-items:baseline;padding:9px 10px;
border-radius:7px;color:#cdd6e0;text-decoration:none;font-size:13px;transition:.15s;}}
nav.toc li a:hover{{background:rgba(255,255,255,.08);color:#fff;}}
nav.toc li a .n{{color:{TEAL};font-weight:700;font-variant-numeric:tabular-nums;
min-width:16px;}}
nav.toc .foot{{margin-top:26px;font-size:11px;color:#6b7f94;border-top:1px solid #2c3e50;
padding-top:14px;}}
/* --- contenu --- */
main{{padding:0 38px 80px;max-width:1180px;}}
header.hero{{padding:46px 0 26px;border-bottom:3px solid var(--ink);margin-bottom:8px;}}
header.hero .eyebrow{{text-transform:uppercase;letter-spacing:2px;font-size:11px;
color:var(--teal);font-weight:700;}}
header.hero h1{{font-size:30px;margin:8px 0 6px;letter-spacing:-.5px;}}
header.hero .meta{{color:var(--muted);font-size:13px;}}
section{{background:#fff;border:1px solid var(--grid);border-radius:12px;
padding:26px 28px;margin:22px 0;box-shadow:0 1px 2px rgba(20,40,60,.04);}}
.sec-head{{display:flex;align-items:center;gap:14px;margin-bottom:14px;
border-bottom:1px solid var(--grid);padding-bottom:12px;}}
.sec-num{{background:var(--ink);color:#fff;width:32px;height:32px;border-radius:8px;
display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px;
flex-shrink:0;}}
h2{{font-size:20px;margin:0;letter-spacing:-.3px;}}
h3{{font-size:15px;margin:24px 0 10px;color:var(--ink);
border-left:3px solid var(--teal);padding-left:10px;}}
p.lead{{color:#42566a;margin:0 0 16px;max-width:80ch;}}
p.note{{color:var(--muted);font-size:12.5px;font-style:italic;}}
code{{background:#eef2f6;padding:1px 6px;border-radius:5px;font-size:12.5px;
color:var(--ink);font-family:'SFMono-Regular',Consolas,monospace;}}
/* KPI */
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:14px;margin-bottom:18px;}}
.kpi{{background:#f8fafc;border:1px solid var(--grid);border-radius:10px;padding:16px;}}
.kpi-label{{font-size:11.5px;color:var(--muted);text-transform:uppercase;
letter-spacing:.4px;}}
.kpi-value{{font-size:25px;font-weight:700;margin:6px 0 2px;
font-variant-numeric:tabular-nums;}}
.kpi-sub{{font-size:11.5px;color:var(--muted);}}
.kpi-signal{{background:linear-gradient(135deg,#fbeae6,#fff);border-color:{SIGNAL};}}
.kpi-signal .kpi-value{{color:var(--signal);}}
/* tables */
table{{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0 4px;}}
table.data th,table.mini th{{background:var(--ink);color:#fff;text-align:left;
padding:8px 10px;font-weight:600;position:sticky;top:0;white-space:nowrap;}}
table.data td,table.mini td{{padding:7px 10px;border-bottom:1px solid #eef2f6;}}
table.data tbody tr:hover{{background:#f6f9fc;}}
table.compact td,table.compact th{{padding:5px 8px;}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;}}
td.colname{{font-weight:600;}}
td.muted{{color:var(--muted);}}
.warn-cell{{color:var(--signal);font-weight:600;}}
.miss-row{{background:#fdf2ef;}}
.scroll{{max-height:520px;overflow:auto;border:1px solid var(--grid);border-radius:8px;}}
.mini{{width:auto;}}
.tag{{font-size:10.5px;padding:2px 8px;border-radius:20px;font-weight:600;
text-transform:uppercase;letter-spacing:.3px;}}
.tag-num{{background:#e3effb;color:#1d5a96;}}
.tag-cat{{background:#e6f4f1;color:#0e7c7b;}}
.tag-cible{{background:#fbeae6;color:{SIGNAL};}}
.tag-id{{background:#eee;color:#666;}}
.tag-date{{background:#f3eafd;color:#6c3fb5;}}
/* barres in-cell */
.barwrap{{display:inline-block;width:70px;height:8px;background:#eef2f6;
border-radius:4px;overflow:hidden;vertical-align:middle;}}
.bar{{display:block;height:100%;}}
.barval{{font-size:11px;color:var(--muted);margin-left:7px;
font-variant-numeric:tabular-nums;}}
/* plots */
img.plot{{max-width:100%;height:auto;border-radius:8px;display:block;}}
.plot-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
gap:18px;margin-top:8px;}}
.plot-card{{background:#fff;border:1px solid var(--grid);border-radius:10px;
padding:12px;}}
.iv-badge{{font-size:12px;font-weight:600;text-align:center;margin-top:6px;}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start;}}
.cat-block{{border-top:1px solid var(--grid);padding-top:14px;margin-top:8px;}}
.cat-block h3{{margin-top:6px;}}
/* alerts */
ul.alerts{{list-style:none;padding:0;}}
.alert{{padding:10px 14px;border-radius:8px;margin-bottom:8px;font-size:13px;
border-left:4px solid;}}
.alert-warn{{background:#fff7ed;border-color:var(--amber);}}
.alert-info{{background:#eef6f9;border-color:var(--teal);}}
ul.check{{font-size:13px;color:#42566a;}}
ul.check li{{margin-bottom:6px;}}
.note{{margin-top:8px;}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr;}}
nav.toc{{position:static;height:auto;}}.two-col{{grid-template-columns:1fr;}}}}
@media print{{nav.toc{{display:none;}}.layout{{grid-template-columns:1fr;}}
section{{break-inside:avoid;box-shadow:none;}}}}
"""

    def generate(self, output_path="rapport_resiliation.html"):
        sns.set_style("whitegrid")
        sections = [
            ("overview", "Vue d'ensemble", self._section_overview()),
            ("quality", "Qualité des données", self._section_quality()),
            ("continuous", "Variables continues", self._section_continuous()),
            ("categorical", "Variables catégorielles", self._section_categorical()),
            ("correlation", "Corrélations", self._section_correlation()),
            ("target", "Taux de résiliation", self._section_target()),
            ("predictive", "Pouvoir prédictif", self._section_predictive()),
            ("temporal", "Stabilité temporelle", self._section_temporal()),
            ("reco", "Synthèse & recommandations", self._section_recommendations()),
        ]
        # la section recos doit être recalculée APRÈS les autres (alertes cumulées)
        sections[-1] = ("reco", "Synthèse & recommandations", self._section_recommendations())

        toc = "".join(
            f"<li><a href='#{a}'><span class='n'>{i}</span>{_esc(t)}</a></li>"
            for i, (a, t, c) in enumerate(sections, 1) if c
        )
        body = "".join(c for a, t, c in sections if c)
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        n, p = self.df.shape
        doc = f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(self.title)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{self._css()}</style></head>
<body><div class="layout">
<nav class="toc"><div class="brand">{_esc(self.title)}<small>Data Quality &amp; EDA</small></div>
<ol>{toc}</ol>
<div class="foot">Généré le {now}<br>{_fmt(n)} lignes · {p} variables<br>
Cible : {_esc(self.target)}</div></nav>
<main><header class="hero"><div class="eyebrow">Assurance · Modélisation de la résiliation</div>
<h1>{_esc(self.title)}</h1>
<div class="meta">{_fmt(n)} polices · {len(self.num_cols)} variables continues · """\
f"""{len(self.cat_cols)} variables catégorielles · taux de résiliation {_fmt(self.target_rate,pct=True)} · {now}</div>
</header>{body}</main></div></body></html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(doc)
        return output_path

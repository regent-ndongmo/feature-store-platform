"""
Feature Store Platform — Interface Tkinter Ultra-Premium
Graphiques, loaders, tooltips pédagogiques, API avec curl live.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import threading, os, sys, json, time, io, math
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.cleaner import DataCleaner
from core.engineer import FeatureEngineer
from store.feature_store import FeatureStore
from api.server import start_api, get_api_url

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────────────
# PALETTE & HELPERS
# ──────────────────────────────────────────────────────────────────────────────
BG      = "#0f0f1a"
SURFACE = "#181828"
CARD    = "#1e1e32"
CARD2   = "#252540"
ACCENT  = "#7c6af7"
ACCT2   = "#5bc4f5"
SUCCESS = "#3dd68c"
WARN    = "#f7a84a"
DANGER  = "#f76a6a"
PINK    = "#f06fa8"
TEXT    = "#e8e8f5"
SUBTEXT = "#7878a8"
BORDER  = "#2e2e50"
WHITE   = "#ffffff"

FONT_TITLE  = ("Segoe UI", 14, "bold")
FONT_HEAD   = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_CODE   = ("Consolas", 9)
FONT_CODE_B = ("Consolas", 10, "bold")

# Descriptions pédagogiques pour chaque opération
CLEAN_INFO = {
    "drop_duplicates":   ("🔁 Supprimer les doublons", "Cherche et supprime les lignes identiques dans le dataset. Utile quand les données ont été fusionnées ou dupliquées par erreur.", "scatter"),
    "drop_all_na_rows":  ("🗑️ Supprimer lignes vides", "Retire les lignes où TOUTES les colonnes sont vides. Évite des lignes fantômes qui faussent les calculs.", "bar"),
    "drop_column":       ("❌ Supprimer une colonne", "Retire complètement une colonne inutile (ex: URL, description textuelle, ID interne).", "bar"),
    "drop_na_rows":      ("🚫 Supprimer lignes avec NaN", "Supprime les lignes où une colonne spécifique est vide. Recommandé si peu de valeurs manquantes.", "bar"),
    "fill_mean":         ("📊 Remplir avec la moyenne", "Remplace les valeurs manquantes par la moyenne de la colonne. Bon pour les données numériques sans outliers.", "hist"),
    "fill_median":       ("📏 Remplir avec la médiane", "Remplace les valeurs manquantes par la valeur centrale. Résistant aux valeurs extrêmes (outliers).", "hist"),
    "fill_mode":         ("🎯 Remplir avec le mode", "Remplace les valeurs manquantes par la valeur la plus fréquente. Idéal pour les colonnes catégorielles (texte).", "bar"),
    "fill_value":        ("✏️ Remplir avec une valeur fixe", "Remplace les valeurs manquantes par une valeur que vous choisissez (ex: 0, 'inconnu', 'N/A').", "bar"),
    "clip_outliers":     ("✂️ Écrêter les valeurs extrêmes", "Limite les valeurs aberrantes entre les percentiles 1% et 99%. Empêche un prix de 1 000 000€ de fausser le modèle.", "box"),
    "cast_numeric":      ("🔢 Convertir en numérique", "Transforme une colonne texte en nombres. Ex: '2015' (texte) → 2015 (nombre). Les valeurs non convertibles deviennent NaN.", "hist"),
    "lowercase_str":     ("🔤 Minuscules et nettoyage", "Met tout le texte en minuscules et retire les espaces superflus. 'Toyota' et 'TOYOTA' deviennent 'toyota'.", "bar"),
}

ENGINEER_INFO = {
    "log_transform":  ("📉 Transformation logarithmique", "Applique log(1+x) à une colonne. Compresse les grandes valeurs et rend les distributions asymétriques plus symétriques. Idéal pour les prix.", "hist_compare"),
    "normalize":      ("📐 Normalisation min-max", "Ramène toutes les valeurs entre 0 et 1. Indispensable pour les algorithmes sensibles à l'échelle (KNN, réseaux de neurones).", "line_scale"),
    "standardize":    ("⚖️ Standardisation (Z-score)", "Centre les données autour de 0 avec un écart-type de 1. Recommandé pour la régression et les SVM.", "hist_compare"),
    "binarize":       ("🔘 Binarisation", "Transforme une colonne numérique en 0 ou 1 selon un seuil. Ex: prix > 15000 → 1 (cher), sinon 0 (pas cher).", "bar_binary"),
    "label_encode":   ("🏷️ Encodage catégoriel", "Convertit du texte en nombres entiers. 'gas'→0, 'diesel'→1, 'electric'→2. Requis par la plupart des algorithmes ML.", "bar"),
    "bin":            ("📦 Découpage en intervalles", "Regroupe les valeurs en N tranches. Ex: km en tranches 'peu', 'moyen', 'beaucoup'. Simplifie les distributions complexes.", "bar"),
    "ratio":          ("➗ Ratio entre deux colonnes", "Calcule col1 / col2. Ex: prix/odometer donne un 'coût par km', feature souvent très informative.", "scatter"),
    "difference":     ("➖ Différence entre colonnes", "Calcule col1 − col2. Capture la relation relative entre deux mesures.", "line_scale"),
    "product":        ("✖️ Produit entre colonnes", "Calcule col1 × col2. Crée des interactions entre features, utiles pour les modèles linéaires.", "scatter"),
    "age_from_year":  ("🗓️ Âge à partir de l'année", "Calcule année_actuelle − colonne_année. Ex: 2024 − 2018 = 6 ans. Bien plus utile que l'année brute pour un modèle.", "hist_compare"),
    "custom_expr":    ("💻 Expression personnalisée", "Écrivez votre propre formule avec pandas eval(). Ex: price / (odometer + 1). Pour les utilisateurs avancés.", "scatter"),
}

plt.rcParams.update({
    "figure.facecolor": CARD,
    "axes.facecolor": CARD2,
    "axes.edgecolor": BORDER,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": SUBTEXT,
    "ytick.color": SUBTEXT,
    "grid.color": BORDER,
    "grid.alpha": 0.5,
})


def fig_to_photoimage(fig, w=None, h=None):
    """Convertit une figure matplotlib en PhotoImage Tkinter via PIL."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=96)
    buf.seek(0)
    img = Image.open(buf)
    if w and h:
        img = img.resize((w, h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


def make_rounded_image(w, h, radius, bg_color, border_color=None):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = _hex_to_rgb(bg_color) + (255,)
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=fill)
    if border_color:
        bc = _hex_to_rgb(border_color) + (255,)
        draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, outline=bc, width=1)
    return img


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def darker(color, factor=0.8):
    r, g, b = _hex_to_rgb(color)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


# ──────────────────────────────────────────────────────────────────────────────
# WIDGET HELPERS
# ──────────────────────────────────────────────────────────────────────────────

class SmartButton(tk.Frame):
    """Bouton avec loader animé et effet hover."""
    def __init__(self, parent, text, command, color=ACCENT, icon="", width=None, **kw):
        super().__init__(parent, bg=parent["bg"])
        self._cmd = command
        self._color = color
        self._text = text
        self._icon = icon
        self._loading = False
        self._dots = 0

        self.btn = tk.Button(
            self, text=f"{icon}  {text}" if icon else text,
            command=self._run, bg=color, fg=WHITE,
            relief="flat", activebackground=darker(color, 0.85),
            activeforeground=WHITE, cursor="hand2",
            font=("Segoe UI", 10, "bold"),
            padx=16, pady=8, bd=0,
            **({} if width is None else {"width": width}),
        )
        self.btn.pack(fill="x" if width is None else "none")
        self.btn.bind("<Enter>", self._on_enter)
        self.btn.bind("<Leave>", self._on_leave)

    def _on_enter(self, e):
        if not self._loading:
            self.btn.config(bg=darker(self._color, 0.85))

    def _on_leave(self, e):
        if not self._loading:
            self.btn.config(bg=self._color)

    def _run(self):
        self.set_loading(True)
        def worker():
            try:
                self._cmd()
            finally:
                self.after(0, lambda: self.set_loading(False))
        threading.Thread(target=worker, daemon=True).start()

    def set_loading(self, state: bool):
        self._loading = state
        if state:
            self.btn.config(state="disabled", bg=darker(self._color, 0.6))
            self._animate()
        else:
            self._loading = False
            self._dots = 0
            self.btn.config(state="normal", bg=self._color,
                            text=f"{self._icon}  {self._text}" if self._icon else self._text)

    def _animate(self):
        if not self._loading:
            return
        dots = "●" * (self._dots % 4) + "○" * (3 - self._dots % 4)
        self.btn.config(text=f"  {dots}  ")
        self._dots += 1
        self.after(280, self._animate)


class Tooltip:
    """Tooltip élégant avec fond sombre."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwin = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
        widget.bind("<ButtonPress>", self.hide)

    def show(self, e=None):
        if self.tipwin or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tipwin = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(bg=BORDER)
        outer = tk.Frame(tw, bg=CARD2, padx=12, pady=8)
        outer.pack(padx=1, pady=1)
        tk.Label(outer, text=self.text, bg=CARD2, fg=TEXT,
                 font=FONT_SMALL, justify="left",
                 wraplength=320).pack()

    def hide(self, e=None):
        if self.tipwin:
            self.tipwin.destroy()
            self.tipwin = None


class SectionCard(tk.Frame):
    """Carte section avec titre pill et séparateur."""
    def __init__(self, parent, title, icon="", color=ACCENT, **kw):
        super().__init__(parent, bg=CARD, bd=0,
                         highlightbackground=BORDER, highlightthickness=1, **kw)
        header = tk.Frame(self, bg=CARD)
        header.pack(fill="x", padx=12, pady=(10, 6))
        pill = tk.Label(header, text=f" {icon} ", bg=color, fg=WHITE,
                        font=("Segoe UI", 9, "bold"), padx=6, pady=2)
        pill.pack(side="left")
        tk.Label(header, text=f"  {title}", bg=CARD, fg=TEXT,
                 font=FONT_HEAD).pack(side="left")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=12)

    def body(self, padx=12, pady=10):
        f = tk.Frame(self, bg=CARD)
        f.pack(fill="both", expand=True, padx=padx, pady=pady)
        return f


class NaNBadge(tk.Label):
    def __init__(self, parent, count, total, **kw):
        pct = count / total * 100 if total else 0
        color = DANGER if pct > 30 else WARN if pct > 10 else SUCCESS
        super().__init__(parent, text=f"  NaN: {count} ({pct:.0f}%)  ",
                         bg=color, fg=WHITE, font=("Segoe UI", 8, "bold"),
                         padx=4, **kw)


# ──────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _make_empty_chart(msg="Chargez un dataset pour voir le graphique", w=480, h=200):
    fig, ax = plt.subplots(figsize=(w/96, h/96))
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color=SUBTEXT, fontsize=11,
            style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.5)
    return fig


def render_clean_chart(df, col, action, w=500, h=220):
    """Génère un graphique pertinent selon l'action de nettoyage."""
    if df is None or col not in df.columns:
        return _make_empty_chart(w=w, h=h)

    series = df[col]
    is_num = pd.api.types.is_numeric_dtype(series)
    fig, axes = plt.subplots(1, 2 if is_num and action in ("fill_mean","fill_median","clip_outliers") else 1,
                              figsize=(w/96, h/96))
    ax = axes[0] if isinstance(axes, np.ndarray) else axes
    n_nan = series.isna().sum()
    color_main = "#7c6af7"
    color_nan  = "#f76a6a"

    if action in ("fill_mean", "fill_median", "fill_mode", "fill_value", "drop_na_rows") and is_num:
        vals = series.dropna()
        ax.hist(vals, bins=25, color=color_main, alpha=0.85, edgecolor="none")
        if action == "fill_mean":
            ax.axvline(vals.mean(), color=SUCCESS, lw=2, ls="--", label=f"Moyenne: {vals.mean():.1f}")
        elif action == "fill_median":
            ax.axvline(vals.median(), color=WARN, lw=2, ls="--", label=f"Médiane: {vals.median():.1f}")
        ax.set_title(f"Distribution de '{col}'  ({n_nan} NaN)", fontsize=9, pad=6)
        ax.set_ylabel("Fréquence", fontsize=8)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8, framealpha=0.3)

    elif action == "clip_outliers" and is_num:
        vals = series.dropna()
        p1, p99 = vals.quantile(0.01), vals.quantile(0.99)
        outliers = vals[(vals < p1) | (vals > p99)]
        normal   = vals[(vals >= p1) & (vals <= p99)]
        ax.hist(normal, bins=25, color=color_main, alpha=0.85, label=f"Normal ({len(normal)})", edgecolor="none")
        ax.hist(outliers, bins=10, color=color_nan, alpha=0.9, label=f"Outliers ({len(outliers)})", edgecolor="none")
        ax.axvline(p1, color=WARN, lw=1.5, ls=":")
        ax.axvline(p99, color=WARN, lw=1.5, ls=":", label=f"Seuils: [{p1:.0f}, {p99:.0f}]")
        ax.set_title(f"Outliers dans '{col}'", fontsize=9, pad=6)
        ax.legend(fontsize=8, framealpha=0.3)

        if isinstance(axes, np.ndarray):
            ax2 = axes[1]
            ax2.boxplot([vals], patch_artist=True,
                        boxprops=dict(facecolor=color_main, alpha=0.6),
                        medianprops=dict(color=SUCCESS, lw=2),
                        whiskerprops=dict(color=SUBTEXT),
                        capprops=dict(color=SUBTEXT),
                        flierprops=dict(marker="o", color=color_nan, markersize=4, alpha=0.6))
            ax2.set_title("Boxplot", fontsize=9, pad=6)
            ax2.set_xticks([])

    elif action in ("drop_column", "drop_duplicates", "drop_all_na_rows", "drop_na_rows", "lowercase_str", "cast_numeric"):
        # Barre NaN / valides
        vals = [len(series) - n_nan, n_nan]
        labels = ["Valides", "Manquants"]
        colors = [SUCCESS, color_nan]
        bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="none")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(v), ha="center", va="bottom", fontsize=10, color=TEXT)
        ax.set_title(f"Qualité des données : '{col}'", fontsize=9, pad=6)
        ax.set_ylabel("Nb lignes", fontsize=8)

    elif not is_num:
        # Barres catégorielles
        vc = series.value_counts().head(10)
        colors_cat = [color_main] * len(vc)
        ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color=colors_cat[::-1], edgecolor="none")
        ax.set_title(f"Top valeurs : '{col}'", fontsize=9, pad=6)
        ax.set_xlabel("Fréquence", fontsize=8)
    else:
        ax.hist(series.dropna(), bins=25, color=color_main, alpha=0.85, edgecolor="none")
        ax.set_title(f"Distribution : '{col}'", fontsize=9, pad=6)

    fig.tight_layout(pad=0.8)
    return fig


def render_engineer_chart(df, col, col2, action, new_col, threshold="0", w=500, h=220):
    """Graphique AVANT / APRÈS pour le feature engineering."""
    if df is None or col not in df.columns:
        return _make_empty_chart(w=w, h=h)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w/96, h/96))
    color_before = SUBTEXT
    color_after  = ACCENT

    series = df[col].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype("category").cat.codes

    ax1.hist(series, bins=25, color=color_before, alpha=0.8, edgecolor="none")
    ax1.set_title(f"Avant : '{col}'", fontsize=9, pad=6)
    ax1.set_ylabel("Fréquence", fontsize=8)

    try:
        if action == "log_transform":
            after = np.log1p(series.clip(lower=0))
        elif action == "normalize":
            after = (series - series.min()) / (series.max() - series.min() + 1e-9)
        elif action == "standardize":
            after = (series - series.mean()) / (series.std() + 1e-9)
        elif action == "binarize":
            thr = float(threshold) if threshold else 0
            after = (series > thr).astype(int)
        elif action == "age_from_year":
            after = datetime.now().year - series
        elif action == "bin":
            bins = max(2, int(threshold) if threshold.isdigit() else 4)
            after = pd.cut(series, bins=bins, labels=False)
        elif action == "label_encode":
            after = df[col].astype("category").cat.codes
        elif action in ("ratio", "difference", "product") and col2 and col2 in df.columns:
            s2 = pd.to_numeric(df[col2], errors="coerce").dropna()
            s1 = series.loc[s2.index]
            if action == "ratio":     after = s1 / (s2.replace(0, np.nan))
            elif action == "difference": after = s1 - s2
            else:                         after = s1 * s2
        else:
            after = series
        after = after.dropna()
        ax2.hist(after, bins=25, color=color_after, alpha=0.85, edgecolor="none")
        ax2.set_title(f"Après : '{new_col or action}'", fontsize=9, pad=6)
    except Exception as e:
        ax2.text(0.5, 0.5, f"Aperçu\nnon disponible", ha="center", va="center",
                 transform=ax2.transAxes, color=SUBTEXT, fontsize=10, style="italic")
        ax2.set_xticks([]); ax2.set_yticks([])

    fig.tight_layout(pad=0.8)
    return fig


def render_store_overview(groups, w=500, h=200):
    """Graphique résumant les groupes dans le store."""
    if not groups:
        return _make_empty_chart("Aucun groupe enregistré dans le store", w=w, h=h)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w/96, h/96))
    names    = [g["name"][:14] for g in groups]
    versions = [g["version"] for g in groups]
    n_feats  = [len(g["feature_cols"]) for g in groups]
    colors   = [ACCENT, ACCT2, SUCCESS, WARN, PINK, DANGER] * 5
    ax1.barh(names, n_feats, color=colors[:len(names)], edgecolor="none")
    ax1.set_title("Features par groupe", fontsize=9, pad=6)
    ax1.set_xlabel("Nb features", fontsize=8)
    ax2.bar(names, versions, color=colors[:len(names)], edgecolor="none")
    ax2.set_title("Versions", fontsize=9, pad=6)
    ax2.set_ylabel("Version", fontsize=8)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=20, labelsize=8)
    fig.tight_layout(pad=0.8)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ──────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🗄️  Feature Store Platform")
        self.geometry("1350x850")
        self.minsize(1100, 720)
        self.configure(bg=BG)

        self.df_raw         = None
        self.df_clean       = None
        self.df_engineered  = None
        self.cleaner        = DataCleaner()
        self.engineer       = FeatureEngineer()
        self.store          = FeatureStore()
        self._chart_photos  = {}  # keep references to avoid GC

        self._setup_styles()
        self._build_header()
        self._build_notebook()
        self._build_statusbar()

        threading.Thread(target=start_api, daemon=True).start()
        self.after(900, lambda: (
            self.api_dot.config(bg=SUCCESS),
            self._status(f"✅  API démarrée — {get_api_url()}", SUCCESS)
        ))

    # ── Styles ─────────────────────────────────────────────────────────────────
    def _setup_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=SURFACE, foreground=SUBTEXT,
                    padding=[20, 10], font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", CARD)],
              foreground=[("selected", ACCENT)],
              font=[("selected", ("Segoe UI", 10, "bold"))])
        s.configure("Treeview", background=CARD2, fieldbackground=CARD2,
                    foreground=TEXT, rowheight=24, font=FONT_CODE)
        s.configure("Treeview.Heading", background=CARD, foreground=ACCT2,
                    font=("Segoe UI", 9, "bold"), relief="flat")
        s.map("Treeview", background=[("selected", ACCENT)])
        s.configure("TEntry", fieldbackground=CARD2, foreground=TEXT,
                    insertcolor=TEXT, padding=6)
        s.configure("TCombobox", fieldbackground=CARD2, foreground=TEXT,
                    selectbackground=ACCENT, padding=4)
        s.map("TCombobox", fieldbackground=[("readonly", CARD2)])
        s.configure("Vertical.TScrollbar", background=SURFACE, troughcolor=BG,
                    arrowcolor=SUBTEXT)
        s.configure("Horizontal.TScrollbar", background=SURFACE, troughcolor=BG,
                    arrowcolor=SUBTEXT)

    # ── Header ─────────────────────────────────────────────────────────────────
    def _build_header(self):
        h = tk.Frame(self, bg=SURFACE, height=60)
        h.pack(fill="x")
        h.pack_propagate(False)
        # Logo + titre
        left = tk.Frame(h, bg=SURFACE)
        left.pack(side="left", padx=20, pady=10)
        tk.Label(left, text="🗄️", bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 18)).pack(side="left")
        tblock = tk.Frame(left, bg=SURFACE)
        tblock.pack(side="left", padx=8)
        tk.Label(tblock, text="Feature Store Platform",
                 bg=SURFACE, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tk.Label(tblock, text="Préparez, transformez et stockez vos données ML",
                 bg=SURFACE, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w")
        # Badges
        right = tk.Frame(h, bg=SURFACE)
        right.pack(side="right", padx=20)
        self.api_dot = tk.Label(right, text="●", bg=WARN, fg=WHITE,
                                font=FONT_SMALL, padx=8, pady=4)
        self.api_dot.pack(side="right", padx=4)
        tk.Label(right, text="API", bg=SURFACE, fg=SUBTEXT,
                 font=FONT_SMALL).pack(side="right")
        # Steps progress
        self._step_labels = []
        steps = ["① Charger", "② Nettoyer", "③ Transformer", "④ Stocker", "⑤ Utiliser"]
        progress = tk.Frame(h, bg=SURFACE)
        progress.pack(side="left", padx=30, pady=16)
        for i, step in enumerate(steps):
            lbl = tk.Label(progress, text=step, bg=SURFACE,
                           fg=SUBTEXT if i > 0 else ACCENT,
                           font=("Segoe UI", 9, "bold" if i == 0 else "normal"),
                           padx=6)
            lbl.pack(side="left")
            self._step_labels.append(lbl)
            if i < len(steps) - 1:
                tk.Label(progress, text="→", bg=SURFACE, fg=BORDER,
                         font=FONT_SMALL).pack(side="left")

    def _highlight_step(self, idx):
        for i, lbl in enumerate(self._step_labels):
            if i == idx:
                lbl.config(fg=ACCENT, font=("Segoe UI", 9, "bold"))
            elif i < idx:
                lbl.config(fg=SUCCESS, font=("Segoe UI", 9))
            else:
                lbl.config(fg=SUBTEXT, font=("Segoe UI", 9))

    # ── Notebook ───────────────────────────────────────────────────────────────
    def _build_notebook(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=0, pady=0)
        tabs = []
        for title in ["📂  Chargement", "🧹  Nettoyage",
                      "⚙️  Feature Engineering", "🗄️  Feature Store", "🌐  API / Endpoint"]:
            f = tk.Frame(self.nb, bg=BG)
            self.nb.add(f, text=f"  {title}  ")
            tabs.append(f)
        self.tab_load, self.tab_clean, self.tab_feat, self.tab_store, self.tab_api = tabs
        self.nb.bind("<<NotebookTabChanged>>",
                     lambda e: self._highlight_step(self.nb.index("current")))
        self._build_tab_load()
        self._build_tab_clean()
        self._build_tab_feat()
        self._build_tab_store()
        self._build_tab_api()

    # ── Status bar ─────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        sb = tk.Frame(self, bg=CARD, height=30)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        self.status_dot = tk.Label(sb, text="●", bg=CARD, fg=SUBTEXT,
                                   font=("Segoe UI", 10))
        self.status_dot.pack(side="left", padx=(12, 4), pady=6)
        self.status_var = tk.StringVar(value="Prêt. Commencez par charger un dataset.")
        tk.Label(sb, textvariable=self.status_var, bg=CARD, fg=SUBTEXT,
                 font=FONT_SMALL, anchor="w").pack(side="left")
        self.row_count_var = tk.StringVar(value="")
        tk.Label(sb, textvariable=self.row_count_var, bg=CARD,
                 fg=SUBTEXT, font=FONT_SMALL).pack(side="right", padx=16)

    def _status(self, msg, color=SUBTEXT):
        self.status_var.set(msg)
        self.status_dot.config(fg=color)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — CHARGEMENT
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_load(self):
        p = self.tab_load
        left = tk.Frame(p, bg=BG, width=340)
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        left.pack_propagate(False)
        right = tk.Frame(p, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(8, 16), pady=16)

        # Hero card
        hero = tk.Frame(left, bg=CARD2, bd=0,
                        highlightbackground=ACCENT, highlightthickness=1)
        hero.pack(fill="x", pady=(0, 12))
        tk.Label(hero, text="👋  Bienvenue !", bg=CARD2, fg=TEXT,
                 font=FONT_HEAD, anchor="w", padx=14).pack(fill="x", pady=(14, 2))
        intro = ("Cette plateforme vous guide pas à pas :\n\n"
                 "  1. Chargez vos données CSV ou Excel\n"
                 "  2. Nettoyez les valeurs manquantes\n"
                 "  3. Créez de nouvelles features\n"
                 "  4. Sauvegardez dans le Feature Store\n"
                 "  5. Récupérez via l'API pour votre modèle")
        tk.Label(hero, text=intro, bg=CARD2, fg=SUBTEXT, font=FONT_SMALL,
                 justify="left", padx=14, anchor="w").pack(fill="x", pady=(2, 14))

        # Source card
        sec = SectionCard(left, "Source de données", "📂", ACCENT)
        sec.pack(fill="x", pady=(0, 10))
        b = sec.body()
        SmartButton(b, "Charger CSV / Excel / Parquet", self._load_file,
                    ACCENT, "📁").pack(fill="x", pady=3)
        tk.Label(b, text="— ou utilisez notre dataset d'exemple —",
                 bg=CARD, fg=SUBTEXT, font=FONT_SMALL).pack(pady=4)
        SmartButton(b, "🚗  Dataset Véhicules (300 lignes)", self._load_sample,
                    ACCT2).pack(fill="x", pady=3)

        # Info card
        sec2 = SectionCard(left, "Infos dataset", "ℹ️", ACCENT)
        sec2.pack(fill="x", pady=(0, 10))
        b2 = sec2.body(pady=8)
        self.info_text = scrolledtext.ScrolledText(
            b2, bg=SURFACE, fg=TEXT, height=10,
            font=FONT_CODE, relief="flat", bd=0, state="disabled",
            selectbackground=ACCENT
        )
        self.info_text.pack(fill="both")

        SmartButton(left, "▶  Continuer : Nettoyage →",
                    lambda: self.nb.select(1), SUCCESS, "").pack(fill="x", pady=4)

        # Preview
        sec3 = SectionCard(right, "Aperçu des données brutes", "👁️", ACCT2)
        sec3.pack(fill="both", expand=True)
        b3 = sec3.body()
        self.preview_tree = self._make_tree(b3)

    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Données", "*.csv *.xlsx *.xls *.parquet"), ("Tous", "*.*")])
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".csv":        df = pd.read_csv(path)
            elif ext in (".xlsx", ".xls"): df = pd.read_excel(path)
            elif ext == ".parquet":  df = pd.read_parquet(path)
            else:
                messagebox.showerror("Format non supporté", f"Extension '{ext}' non reconnue.")
                return
            self._set_dataframe(df, os.path.basename(path))
        except Exception as e:
            messagebox.showerror("Erreur de chargement", str(e))

    def _load_sample(self):
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            "vehicle_id": range(n),
            "year":       np.random.randint(2000, 2024, n).astype(float),
            "price":      np.random.randint(2000, 45000, n).astype(float),
            "odometer":   np.random.randint(5000, 250000, n).astype(float),
            "manufacturer": np.random.choice(["toyota","ford","honda","chevrolet","bmw","volkswagen"], n),
            "condition":  np.random.choice(["excellent","good","fair","like new","salvage",None], n),
            "cylinders":  np.random.choice([4, 6, 8, None], n),
            "fuel":       np.random.choice(["gas","diesel","electric","hybrid"], n),
            "transmission": np.random.choice(["automatic","manual"], n),
            "drive":      np.random.choice(["4wd","fwd","rwd",None], n),
            "state":      np.random.choice(["ca","tx","fl","ny","az"], n),
        })
        for col in ["year","odometer","condition","cylinders","drive"]:
            idx = np.random.choice(df.index, 25, replace=False)
            df.loc[idx, col] = None
        self._set_dataframe(df, "vehicles_sample.csv")

    def _set_dataframe(self, df, label=""):
        self.df_raw = df
        self.df_clean = df.copy()
        self.df_engineered = df.copy()
        self.cleaner.clear_rules()
        self.engineer.clear()
        self._refresh_preview(df, self.preview_tree)
        self._refresh_info(df, label)
        self._refresh_all_combos()
        self.row_count_var.set(f"  {len(df):,} lignes · {len(df.columns)} colonnes  ")
        self._status(f"✅  Chargé : {label}", SUCCESS)
        self._highlight_step(0)
        self.after(200, lambda: self.nb.select(1))

    def _refresh_info(self, df, label=""):
        lines = [
            f"Fichier   : {label}",
            f"Lignes    : {len(df):,}",
            f"Colonnes  : {len(df.columns)}",
            f"Mémoire   : {df.memory_usage(deep=True).sum()/1024:.1f} Ko",
            "",
            f"{'Colonne':<22} {'Type':<10} {'NaN':>6}  {'%':>5}",
            "─" * 48,
        ]
        for col in df.columns:
            n = df[col].isna().sum()
            pct = n / len(df) * 100
            flag = "⚠️ " if pct > 30 else ""
            lines.append(f"{flag}{col:<22} {str(df[col].dtype):<10} {n:>6}  {pct:>4.0f}%")
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "\n".join(lines))
        self.info_text.configure(state="disabled")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — NETTOYAGE
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_clean(self):
        p = self.tab_clean
        left = tk.Frame(p, bg=BG, width=370)
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        left.pack_propagate(False)
        right = tk.Frame(p, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(8, 16), pady=16)

        # ── Explication ──
        tip = tk.Frame(left, bg=CARD2,
                       highlightbackground=ACCT2, highlightthickness=1)
        tip.pack(fill="x", pady=(0, 10))
        tk.Label(tip, text="💡  Le nettoyage", bg=CARD2, fg=ACCT2,
                 font=("Segoe UI", 10, "bold"), padx=12).pack(anchor="w", pady=(10, 2))
        tk.Label(tip,
                 text=("Avant d'entraîner un modèle ML, vos données doivent\n"
                       "être propres. Ici vous définissez les règles qui seront\n"
                       "appliquées dans l'ordre. Le graphique explique l'effet."),
                 bg=CARD2, fg=SUBTEXT, font=FONT_SMALL,
                 justify="left", padx=12, anchor="w").pack(fill="x", pady=(0, 10))

        # ── Règle ──
        sec = SectionCard(left, "Nouvelle règle de nettoyage", "🧹", ACCENT)
        sec.pack(fill="x", pady=(0, 8))
        b = sec.body()

        r0 = tk.Frame(b, bg=CARD); r0.pack(fill="x", pady=2)
        tk.Label(r0, text="Action :", bg=CARD, fg=SUBTEXT, font=FONT_SMALL, width=12, anchor="w").pack(side="left")
        self.clean_action = ttk.Combobox(r0, state="readonly", width=26,
                                          values=list(CLEAN_INFO.keys()))
        self.clean_action.set("fill_median")
        self.clean_action.pack(side="left", padx=4)
        self.clean_action.bind("<<ComboboxSelected>>", self._on_clean_action_change)

        r1 = tk.Frame(b, bg=CARD); r1.pack(fill="x", pady=2)
        tk.Label(r1, text="Colonne :", bg=CARD, fg=SUBTEXT, font=FONT_SMALL, width=12, anchor="w").pack(side="left")
        self.clean_col = ttk.Combobox(r1, state="readonly", width=26)
        self.clean_col.pack(side="left", padx=4)
        self.clean_col.bind("<<ComboboxSelected>>", self._on_clean_action_change)

        r2 = tk.Frame(b, bg=CARD); r2.pack(fill="x", pady=2)
        tk.Label(r2, text="Valeur fixe :", bg=CARD, fg=SUBTEXT, font=FONT_SMALL, width=12, anchor="w").pack(side="left")
        self.clean_val = tk.StringVar()
        self.clean_val_entry = ttk.Entry(r2, textvariable=self.clean_val, width=28)
        self.clean_val_entry.pack(side="left", padx=4)

        # Description dynamique
        self.clean_desc_var = tk.StringVar(value="Sélectionnez une action pour voir son explication.")
        self.clean_desc_lbl = tk.Label(b, textvariable=self.clean_desc_var,
                                       bg=CARD, fg=SUBTEXT, font=FONT_SMALL,
                                       wraplength=300, justify="left", anchor="w")
        self.clean_desc_lbl.pack(fill="x", pady=(4, 6))

        SmartButton(b, "➕  Ajouter cette règle", self._add_clean_rule, ACCENT).pack(fill="x", pady=3)

        # ── Queue ──
        sec2 = SectionCard(left, "Règles en attente (ordre d'exécution)", "📋", WARN)
        sec2.pack(fill="x", pady=(0, 8))
        b2 = sec2.body(pady=8)
        self.clean_listbox = tk.Listbox(
            b2, bg=SURFACE, fg=TEXT, selectbackground=ACCENT,
            font=FONT_CODE, height=7, relief="flat", bd=0,
            selectforeground=WHITE, activestyle="none"
        )
        self.clean_listbox.pack(fill="x")
        brow = tk.Frame(b2, bg=CARD); brow.pack(fill="x", pady=(6, 0))
        SmartButton(brow, "🗑  Supprimer", self._remove_clean_rule, DANGER).pack(side="left", padx=(0, 4))
        SmartButton(brow, "🔄  Tout effacer", self._clear_clean_rules, WARN).pack(side="left")

        SmartButton(left, "▶  Appliquer le nettoyage", self._apply_cleaning, SUCCESS, "🧹").pack(fill="x", pady=4)
        SmartButton(left, "Continuer : Feature Engineering →",
                    lambda: self.nb.select(2), ACCT2).pack(fill="x")

        # ── Graphique + log + preview ──
        top_right = tk.Frame(right, bg=BG)
        top_right.pack(fill="x", pady=(0, 8))

        sec_chart = SectionCard(top_right, "Visualisation — effet de l'opération sélectionnée", "📊", ACCENT)
        sec_chart.pack(fill="x")
        b_chart = sec_chart.body(pady=8)
        self.clean_chart_lbl = tk.Label(b_chart, bg=CARD, anchor="center")
        self.clean_chart_lbl.pack(fill="x")
        self._update_clean_chart()

        mid_right = tk.Frame(right, bg=BG, height=110)
        mid_right.pack(fill="x", pady=(0, 8))
        mid_right.pack_propagate(False)
        sec_log = SectionCard(mid_right, "Log de nettoyage", "📝", SUCCESS)
        sec_log.pack(fill="both", expand=True)
        bl = sec_log.body(pady=6)
        self.clean_log = scrolledtext.ScrolledText(
            bl, bg=SURFACE, fg=ACCT2, font=FONT_CODE,
            relief="flat", bd=0, height=4, state="disabled"
        )
        self.clean_log.pack(fill="both", expand=True)

        sec_prev = SectionCard(right, "Aperçu après nettoyage", "👁️", ACCT2)
        sec_prev.pack(fill="both", expand=True)
        bp = sec_prev.body()
        self.clean_tree = self._make_tree(bp)

    def _on_clean_action_change(self, e=None):
        action = self.clean_action.get()
        col    = self.clean_col.get()
        if action in CLEAN_INFO:
            _, desc, _ = CLEAN_INFO[action]
            self.clean_desc_var.set(desc)
        self._update_clean_chart()

    def _update_clean_chart(self, event=None):
        action = self.clean_action.get() if hasattr(self, "clean_action") else ""
        col    = self.clean_col.get() if hasattr(self, "clean_col") else ""
        df     = self.df_raw
        fig    = render_clean_chart(df, col, action, w=680, h=180)
        photo  = fig_to_photoimage(fig, w=680, h=180)
        plt.close(fig)
        self._chart_photos["clean"] = photo
        if hasattr(self, "clean_chart_lbl"):
            self.clean_chart_lbl.config(image=photo)

    def _add_clean_rule(self):
        action = self.clean_action.get()
        col    = self.clean_col.get()
        val    = self.clean_val.get()
        rule   = {"action": action, "column": col, "value": val}
        self.cleaner.add_rule(rule)
        icon, _, _ = CLEAN_INFO.get(action, ("", "", ""))
        label = f"{icon}  {action}" + (f"  [{col}]" if col else "") + (f" = '{val}'" if val and action == "fill_value" else "")
        self.clean_listbox.insert("end", f"  {label}")
        self._status(f"Règle ajoutée : {action}", ACCENT)

    def _remove_clean_rule(self):
        sel = self.clean_listbox.curselection()
        if sel:
            self.clean_listbox.delete(sel[0])
            self.cleaner.rules.pop(sel[0])

    def _clear_clean_rules(self):
        self.cleaner.clear_rules()
        self.clean_listbox.delete(0, "end")

    def _apply_cleaning(self):
        if self.df_raw is None:
            messagebox.showwarning("Pas de données", "Chargez d'abord un dataset.")
            return
        time.sleep(0.3)  # let loader show
        self.df_clean = self.cleaner.apply(self.df_raw)
        self.df_engineered = self.df_clean.copy()
        log = "\n".join(self.cleaner.get_log()) or "Aucune transformation appliquée."
        self.clean_log.configure(state="normal")
        self.clean_log.delete("1.0", "end")
        self.clean_log.insert("1.0", log)
        self.clean_log.configure(state="disabled")
        self._refresh_preview(self.df_clean, self.clean_tree)
        self._refresh_all_combos()
        self._update_clean_chart()
        n_rules = len(self.cleaner.rules)
        self._status(f"✅  {n_rules} règle(s) appliquée(s) — {len(self.df_clean)} lignes, {len(self.df_clean.columns)} colonnes", SUCCESS)
        self._highlight_step(1)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_feat(self):
        p = self.tab_feat
        left = tk.Frame(p, bg=BG, width=380)
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        left.pack_propagate(False)
        right = tk.Frame(p, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(8, 16), pady=16)

        # Intro
        tip = tk.Frame(left, bg=CARD2,
                       highlightbackground=PINK, highlightthickness=1)
        tip.pack(fill="x", pady=(0, 10))
        tk.Label(tip, text="💡  Feature Engineering", bg=CARD2, fg=PINK,
                 font=("Segoe UI", 10, "bold"), padx=12).pack(anchor="w", pady=(10, 2))
        tk.Label(tip,
                 text=("Créez de nouvelles colonnes à partir des existantes.\n"
                       "Ex: log(prix), âge depuis l'année, normalisation...\n"
                       "Ces nouvelles features améliorent souvent le modèle."),
                 bg=CARD2, fg=SUBTEXT, font=FONT_SMALL,
                 justify="left", padx=12, anchor="w").pack(fill="x", pady=(0, 10))

        sec = SectionCard(left, "Nouvelle transformation", "⚙️", PINK)
        sec.pack(fill="x", pady=(0, 8))
        b = sec.body()

        rows_cfg = [
            ("Action :",      "fe_action",  "combobox",  list(ENGINEER_INFO.keys())),
            ("Colonne 1 :",   "fe_col",     "combobox2", []),
            ("Colonne 2 :",   "fe_col2",    "combobox2", []),
            ("Nom résultat :", "fe_new_col", "entry",     "new_feature"),
            ("Paramètre :",   "fe_param",   "entry",     ""),
        ]
        for label, attr, kind, vals in rows_cfg:
            row = tk.Frame(b, bg=CARD); row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=CARD, fg=SUBTEXT, font=FONT_SMALL, width=14, anchor="w").pack(side="left")
            if "entry" in kind:
                sv = tk.StringVar(value=vals)
                w = ttk.Entry(row, textvariable=sv, width=26)
                w.pack(side="left", padx=4)
                setattr(self, attr, sv)
            else:
                cb = ttk.Combobox(row, state="readonly", width=25, values=vals)
                if attr == "fe_action": cb.set("log_transform")
                cb.pack(side="left", padx=4)
                setattr(self, attr, cb)
                if attr == "fe_action":
                    cb.bind("<<ComboboxSelected>>", self._on_fe_action_change)
                else:
                    cb.bind("<<ComboboxSelected>>", self._on_fe_action_change)

        # param hint
        self.fe_param_hint = tk.Label(b, text="", bg=CARD, fg=SUBTEXT,
                                      font=("Segoe UI", 8, "italic"))
        self.fe_param_hint.pack(anchor="w", pady=(0, 2))

        self.fe_desc_var = tk.StringVar(value="Sélectionnez une transformation pour voir son explication.")
        tk.Label(b, textvariable=self.fe_desc_var, bg=CARD, fg=SUBTEXT,
                 font=FONT_SMALL, wraplength=310, justify="left", anchor="w").pack(fill="x", pady=(4, 6))

        SmartButton(b, "➕  Ajouter transformation", self._add_fe_rule, PINK, "").pack(fill="x", pady=3)

        sec2 = SectionCard(left, "Transformations en attente", "📋", WARN)
        sec2.pack(fill="x", pady=(0, 8))
        b2 = sec2.body(pady=8)
        self.fe_listbox = tk.Listbox(
            b2, bg=SURFACE, fg=TEXT, selectbackground=ACCENT,
            font=FONT_CODE, height=6, relief="flat", bd=0,
            selectforeground=WHITE, activestyle="none"
        )
        self.fe_listbox.pack(fill="x")
        brow2 = tk.Frame(b2, bg=CARD); brow2.pack(fill="x", pady=(6, 0))
        SmartButton(brow2, "🗑  Supprimer", self._remove_fe_rule, DANGER).pack(side="left", padx=(0, 4))
        SmartButton(brow2, "🔄  Effacer", self._clear_fe_rules, WARN).pack(side="left")

        SmartButton(left, "▶  Appliquer le feature engineering", self._apply_fe, SUCCESS, "⚙️").pack(fill="x", pady=4)
        SmartButton(left, "Continuer : Feature Store →",
                    lambda: self.nb.select(3), ACCT2).pack(fill="x")

        # Graphique avant/après
        sec_chart = SectionCard(right, "Visualisation — Avant ↔ Après la transformation", "📊", PINK)
        sec_chart.pack(fill="x", pady=(0, 8))
        bc = sec_chart.body(pady=8)
        self.fe_chart_lbl = tk.Label(bc, bg=CARD, anchor="center")
        self.fe_chart_lbl.pack(fill="x")
        self._update_fe_chart()

        sec_log = SectionCard(right, "Log Feature Engineering", "📝", SUCCESS)
        sec_log.pack(fill="x", pady=(0, 8))
        bl = sec_log.body(pady=6)
        mid_frame = tk.Frame(bl, bg=CARD, height=90)
        mid_frame.pack(fill="x")
        mid_frame.pack_propagate(False)
        self.fe_log = scrolledtext.ScrolledText(
            mid_frame, bg=SURFACE, fg=ACCT2, font=FONT_CODE,
            relief="flat", bd=0, state="disabled"
        )
        self.fe_log.pack(fill="both", expand=True)

        sec_prev = SectionCard(right, "Aperçu après feature engineering", "👁️", ACCT2)
        sec_prev.pack(fill="both", expand=True)
        bp = sec_prev.body()
        self.fe_tree = self._make_tree(bp)

    def _on_fe_action_change(self, e=None):
        action = self.fe_action.get()
        if action in ENGINEER_INFO:
            _, desc, _ = ENGINEER_INFO[action]
            self.fe_desc_var.set(desc)
            hints = {
                "binarize": "Paramètre = seuil numérique (ex: 15000)",
                "bin": "Paramètre = nombre d'intervalles (ex: 4)",
                "custom_expr": "Paramètre = expression (ex: price / odometer)",
            }
            self.fe_param_hint.config(text=hints.get(action, ""))
        self._update_fe_chart()

    def _update_fe_chart(self, event=None):
        if not hasattr(self, "fe_action"):
            return
        action  = self.fe_action.get()
        col     = self.fe_col.get() if hasattr(self, "fe_col") else ""
        col2    = self.fe_col2.get() if hasattr(self, "fe_col2") else ""
        new_col = self.fe_new_col.get() if hasattr(self, "fe_new_col") else ""
        thr     = self.fe_param.get() if hasattr(self, "fe_param") else ""
        df      = self.df_clean if self.df_clean is not None else self.df_raw
        fig     = render_engineer_chart(df, col, col2, action, new_col, thr, w=680, h=185)
        photo   = fig_to_photoimage(fig, w=680, h=185)
        plt.close(fig)
        self._chart_photos["fe"] = photo
        self.fe_chart_lbl.config(image=photo)

    def _add_fe_rule(self):
        action  = self.fe_action.get()
        col     = self.fe_col.get()
        col2    = self.fe_col2.get()
        new_col = self.fe_new_col.get() or f"{action}_{col}"
        param   = self.fe_param.get()
        t = {"action": action, "column": col, "column2": col2, "new_col": new_col}
        if param:
            if action == "binarize": t["threshold"] = param
            elif action == "bin":    t["bins"] = param
            elif action == "custom_expr": t["expression"] = param
        self.engineer.add_transformation(t)
        icon, _, _ = ENGINEER_INFO.get(action, ("", "", ""))
        self.fe_listbox.insert("end", f"  {icon}  {action}({col}) → {new_col}")
        self._update_fe_chart()

    def _remove_fe_rule(self):
        sel = self.fe_listbox.curselection()
        if sel:
            self.fe_listbox.delete(sel[0])
            self.engineer.transformations.pop(sel[0])

    def _clear_fe_rules(self):
        self.engineer.clear()
        self.fe_listbox.delete(0, "end")

    def _apply_fe(self):
        src = self.df_clean if self.df_clean is not None else self.df_raw
        if src is None:
            messagebox.showwarning("Pas de données", "Chargez d'abord un dataset.")
            return
        time.sleep(0.3)
        self.df_engineered = self.engineer.apply(src)
        log = "\n".join(self.engineer.get_log()) or "Aucune transformation."
        self.fe_log.configure(state="normal")
        self.fe_log.delete("1.0", "end")
        self.fe_log.insert("1.0", log)
        self.fe_log.configure(state="disabled")
        self._refresh_preview(self.df_engineered, self.fe_tree)
        self._refresh_all_combos()
        self._update_fe_chart()
        self._status(f"✅  Feature engineering appliqué — {len(self.df_engineered.columns)} colonnes", SUCCESS)
        self._highlight_step(2)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — FEATURE STORE
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_store(self):
        p = self.tab_store
        left = tk.Frame(p, bg=BG, width=380)
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        left.pack_propagate(False)
        right = tk.Frame(p, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(8, 16), pady=16)

        tip = tk.Frame(left, bg=CARD2,
                       highlightbackground=SUCCESS, highlightthickness=1)
        tip.pack(fill="x", pady=(0, 10))
        tk.Label(tip, text="💡  Le Feature Store", bg=CARD2, fg=SUCCESS,
                 font=("Segoe UI", 10, "bold"), padx=12).pack(anchor="w", pady=(10, 2))
        tk.Label(tip,
                 text=("Sauvegardez vos features transformées de manière\n"
                       "centralisée. Chaque sauvegarde est versionnée.\n"
                       "L'API vous permet de les récupérer pour l'entraînement."),
                 bg=CARD2, fg=SUBTEXT, font=FONT_SMALL,
                 justify="left", padx=12, anchor="w").pack(fill="x", pady=(0, 10))

        sec = SectionCard(left, "Sauvegarder dans le Store", "💾", SUCCESS)
        sec.pack(fill="x", pady=(0, 8))
        b = sec.body()

        for lbl, attr, default in [
            ("Nom du groupe :", "store_name",  "my_features"),
            ("Description :",  "store_desc",  ""),
        ]:
            row = tk.Frame(b, bg=CARD); row.pack(fill="x", pady=2)
            tk.Label(row, text=lbl, bg=CARD, fg=SUBTEXT, font=FONT_SMALL, width=14, anchor="w").pack(side="left")
            sv = tk.StringVar(value=default)
            ttk.Entry(row, textvariable=sv, width=24).pack(side="left", padx=4)
            setattr(self, attr, sv)

        row2 = tk.Frame(b, bg=CARD); row2.pack(fill="x", pady=2)
        tk.Label(row2, text="Colonne entité :", bg=CARD, fg=SUBTEXT, font=FONT_SMALL, width=14, anchor="w").pack(side="left")
        self.store_entity = ttk.Combobox(row2, state="readonly", width=22)
        self.store_entity.pack(side="left", padx=4)
        Tooltip(self.store_entity, "La colonne entité est l'identifiant unique de chaque ligne (ex: vehicle_id, user_id). C'est la clé de récupération des features.")

        tk.Label(b, text="Features à stocker (Ctrl+clic = multi-sélection) :",
                 bg=CARD, fg=SUBTEXT, font=FONT_SMALL, anchor="w").pack(fill="x", pady=(6, 2))
        lf = tk.Frame(b, bg=SURFACE, bd=0,
                      highlightbackground=BORDER, highlightthickness=1)
        lf.pack(fill="x")
        self.feat_listbox = tk.Listbox(
            lf, bg=SURFACE, fg=TEXT, selectbackground=ACCENT,
            selectforeground=WHITE, selectmode="multiple",
            font=FONT_CODE, height=8, relief="flat", bd=0,
            activestyle="none", exportselection=False
        )
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self.feat_listbox.yview)
        self.feat_listbox.configure(yscrollcommand=vsb.set)
        self.feat_listbox.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        SmartButton(b, "💾  Sauvegarder dans le Store", self._save_to_store, SUCCESS, "").pack(fill="x", pady=(10, 3))

        sec2 = SectionCard(left, "Groupes enregistrés", "🗄️", ACCT2)
        sec2.pack(fill="both", expand=True, pady=(0, 8))
        b2 = sec2.body(pady=8)
        gf = tk.Frame(b2, bg=SURFACE,
                      highlightbackground=BORDER, highlightthickness=1)
        gf.pack(fill="both", expand=True)
        self.groups_listbox = tk.Listbox(
            gf, bg=SURFACE, fg=TEXT, selectbackground=ACCENT,
            selectforeground=WHITE, font=FONT_CODE, relief="flat", bd=0,
            activestyle="none"
        )
        self.groups_listbox.pack(fill="both", expand=True, padx=4, pady=4)
        self.groups_listbox.bind("<<ListboxSelect>>", self._on_group_select)

        brow3 = tk.Frame(b2, bg=CARD); brow3.pack(fill="x", pady=(4, 0))
        SmartButton(brow3, "🔄  Rafraîchir", self._refresh_store_groups, ACCT2).pack(side="left", padx=(0, 4))
        SmartButton(brow3, "🗑  Supprimer", self._delete_group, DANGER).pack(side="left")

        SmartButton(left, "Continuer : API / Endpoint →",
                    lambda: self.nb.select(4), ACCENT).pack(fill="x")

        # Right : overview chart + details + preview
        sec_chart = SectionCard(right, "Vue d'ensemble du Feature Store", "📊", SUCCESS)
        sec_chart.pack(fill="x", pady=(0, 8))
        bc = sec_chart.body(pady=8)
        self.store_chart_lbl = tk.Label(bc, bg=CARD, anchor="center")
        self.store_chart_lbl.pack(fill="x")

        sec3 = SectionCard(right, "Détails du groupe sélectionné", "📋", ACCT2)
        sec3.pack(fill="x", pady=(0, 8))
        b3 = sec3.body(pady=8)
        details_frame = tk.Frame(b3, bg=CARD, height=90)
        details_frame.pack(fill="x"); details_frame.pack_propagate(False)
        self.group_detail = scrolledtext.ScrolledText(
            details_frame, bg=SURFACE, fg=TEXT, font=FONT_CODE,
            relief="flat", bd=0, state="disabled"
        )
        self.group_detail.pack(fill="both", expand=True)

        sec4 = SectionCard(right, "Aperçu des features stockées", "👁️", ACCT2)
        sec4.pack(fill="both", expand=True)
        b4 = sec4.body()
        self.store_tree = self._make_tree(b4)

        self._refresh_store_groups()

    def _save_to_store(self):
        if self.df_engineered is not None:
            df = self.df_engineered
        elif self.df_clean is not None:
            df = self.df_clean
        else:
            df = self.df_raw
        if df is None:
            messagebox.showwarning("Pas de données", "Chargez d'abord un dataset.")
            return
        name = self.store_name.get().strip()
        if not name:
            messagebox.showwarning("Nom manquant", "Entrez un nom pour le groupe.")
            return
        entity_col = self.store_entity.get()
        if not entity_col or entity_col not in df.columns:
            messagebox.showwarning("Entité manquante", "Sélectionnez une colonne entité valide.")
            return
        sel = self.feat_listbox.curselection()
        if not sel:
            messagebox.showwarning("Features manquantes", "Sélectionnez au moins une feature.")
            return
        feature_cols = [self.feat_listbox.get(i) for i in sel]
        time.sleep(0.4)
        result = self.store.save_feature_group(
            name=name, df=df, entity_col=entity_col,
            feature_cols=feature_cols, description=self.store_desc.get()
        )
        self._refresh_store_groups()
        self._refresh_store_chart()
        self._status(f"✅  '{name}' v{result['version']} sauvegardé — {result['rows']} lignes, {len(feature_cols)} features", SUCCESS)
        self._highlight_step(3)

    def _refresh_store_groups(self):
        self.groups_listbox.delete(0, "end")
        for g in self.store.list_feature_groups():
            n = len(g["feature_cols"])
            self.groups_listbox.insert("end", f"  📦  {g['name']}  v{g['version']}  [{n} features]")
        self._refresh_store_chart()
        self._refresh_api_groups()

    def _refresh_store_chart(self):
        groups = self.store.list_feature_groups()
        fig = render_store_overview(groups, w=640, h=185)
        photo = fig_to_photoimage(fig, w=640, h=185)
        plt.close(fig)
        self._chart_photos["store"] = photo
        self.store_chart_lbl.config(image=photo)

    def _on_group_select(self, e=None):
        sel = self.groups_listbox.curselection()
        if not sel: return
        raw = self.groups_listbox.get(sel[0])
        name = raw.strip().split("  ")[1].strip()
        meta = self.store.get_metadata(name)
        if not meta: return
        lines = [
            f"Nom        : {meta['name']}",
            f"Version    : v{meta['version']}",
            f"Description: {meta['description'] or '—'}",
            f"Entité     : {meta['entity_col']}",
            f"Créé le    : {meta['created_at'][:19]}",
            f"",
            f"Features ({len(meta['feature_cols'])}) :",
        ] + [f"  · {f}" for f in meta["feature_cols"]]
        self.group_detail.configure(state="normal")
        self.group_detail.delete("1.0", "end")
        self.group_detail.insert("1.0", "\n".join(lines))
        self.group_detail.configure(state="disabled")
        try:
            df = self.store.get_features(name)
            self._refresh_preview(df.head(60), self.store_tree)
        except Exception:
            pass

    def _delete_group(self):
        sel = self.groups_listbox.curselection()
        if not sel: return
        raw = self.groups_listbox.get(sel[0])
        name = raw.strip().split("  ")[1].strip()
        if messagebox.askyesno("Confirmer suppression",
                               f"Supprimer définitivement le groupe\n'{name}' ?"):
            self.store.delete_feature_group(name)
            self._refresh_store_groups()
            self._status(f"🗑  Groupe '{name}' supprimé", DANGER)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — API / ENDPOINT
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_api(self):
        p = self.tab_api
        left = tk.Frame(p, bg=BG, width=420)
        left.pack(side="left", fill="y", padx=(16, 8), pady=16)
        left.pack_propagate(False)
        right = tk.Frame(p, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(8, 16), pady=16)

        # Intro
        tip = tk.Frame(left, bg=CARD2,
                       highlightbackground=ACCT2, highlightthickness=1)
        tip.pack(fill="x", pady=(0, 10))
        tk.Label(tip, text="💡  Endpoint Feature Store", bg=CARD2, fg=ACCT2,
                 font=("Segoe UI", 10, "bold"), padx=12).pack(anchor="w", pady=(10, 2))
        tk.Label(tip,
                 text=("L'API REST permet à votre code Python (notebook,\n"
                       "script) de récupérer les features pour entraîner\n"
                       "votre modèle ML, sans re-calculer quoi que ce soit."),
                 bg=CARD2, fg=SUBTEXT, font=FONT_SMALL,
                 justify="left", padx=12, anchor="w").pack(fill="x", pady=(0, 10))

        # URL badge
        url_card = tk.Frame(left, bg=CARD,
                            highlightbackground=SUCCESS, highlightthickness=1)
        url_card.pack(fill="x", pady=(0, 10))
        tk.Label(url_card, text="🌐  Serveur actif :", bg=CARD, fg=SUBTEXT,
                 font=FONT_SMALL, padx=12).pack(anchor="w", pady=(8, 2))
        tk.Label(url_card, text=get_api_url(), bg=CARD, fg=SUCCESS,
                 font=FONT_CODE_B, padx=12).pack(anchor="w", pady=(0, 8))

        # Endpoints reference
        sec_ref = SectionCard(left, "Endpoints disponibles", "📌", ACCT2)
        sec_ref.pack(fill="x", pady=(0, 10))
        b_ref = sec_ref.body(pady=8)
        for method, path, desc, color in [
            ("GET",    "/api/feature-groups",          "Lister tous les groupes",   SUCCESS),
            ("GET",    "/api/feature-groups/<nom>",    "Métadonnées d'un groupe",   ACCT2),
            ("POST",   "/api/feature-groups/<nom>/fetch", "Récupérer des features", ACCENT),
            ("DELETE", "/api/feature-groups/<nom>",    "Supprimer un groupe",       DANGER),
            ("GET",    "/health",                      "Statut de l'API",           WARN),
        ]:
            row = tk.Frame(b_ref, bg=CARD2,
                           highlightbackground=BORDER, highlightthickness=1)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f" {method} ", bg=color, fg=WHITE,
                     font=("Consolas", 8, "bold"), padx=4, pady=3).pack(side="left")
            tk.Label(row, text=f" {path}", bg=CARD2, fg=ACCT2,
                     font=("Consolas", 9)).pack(side="left")
            tk.Label(row, text=f"  {desc}", bg=CARD2, fg=SUBTEXT,
                     font=FONT_SMALL).pack(side="left")

        # Requête
        sec_req = SectionCard(left, "Tester l'API", "🔬", ACCENT)
        sec_req.pack(fill="x", pady=(0, 8))
        b_req = sec_req.body()

        row_g = tk.Frame(b_req, bg=CARD); row_g.pack(fill="x", pady=2)
        tk.Label(row_g, text="Groupe :", bg=CARD, fg=SUBTEXT, font=FONT_SMALL, width=14, anchor="w").pack(side="left")
        self.api_group = ttk.Combobox(row_g, state="readonly", width=22)
        self.api_group.pack(side="left", padx=4)
        self.api_group.bind("<<ComboboxSelected>>", self._on_api_group_select)

        row_f = tk.Frame(b_req, bg=CARD); row_f.pack(fill="x", pady=2)
        tk.Label(row_f, text="Features :", bg=CARD, fg=SUBTEXT, font=FONT_SMALL, width=14, anchor="w").pack(side="left")
        self.api_features = tk.StringVar()
        fe = ttk.Entry(row_f, textvariable=self.api_features, width=24)
        fe.pack(side="left", padx=4)
        Tooltip(fe, "Noms des features séparés par des virgules.\nLaissez vide pour toutes les features.\nEx: price, odometer, fuel")

        row_e = tk.Frame(b_req, bg=CARD); row_e.pack(fill="x", pady=2)
        tk.Label(row_e, text="Entity IDs :", bg=CARD, fg=SUBTEXT, font=FONT_SMALL, width=14, anchor="w").pack(side="left")
        self.api_entity_ids = tk.StringVar()
        ee = ttk.Entry(row_e, textvariable=self.api_entity_ids, width=24)
        ee.pack(side="left", padx=4)
        Tooltip(ee, "IDs des entités séparés par des virgules.\nLaissez vide pour toutes les lignes.\nEx: 0, 1, 2, 10")

        btn_row = tk.Frame(b_req, bg=CARD); btn_row.pack(fill="x", pady=(8, 2))
        SmartButton(btn_row, "📋  List", self._api_list, ACCT2).pack(side="left", padx=(0, 4))
        SmartButton(btn_row, "📥  Fetch", self._api_fetch, ACCENT).pack(side="left", padx=(0, 4))
        SmartButton(btn_row, "💊  Health", self._api_health, SUCCESS).pack(side="left")

        SmartButton(left, "🔄  Rafraîchir la liste des groupes",
                    self._refresh_api_groups, WARN).pack(fill="x", pady=4)

        # Python snippet
        sec_py = SectionCard(left, "Utiliser l'API depuis Python", "🐍", PINK)
        sec_py.pack(fill="x")
        bp = sec_py.body(pady=8)
        snippet_frame = tk.Frame(bp, bg=SURFACE); snippet_frame.pack(fill="x")
        self.py_snippet = scrolledtext.ScrolledText(
            snippet_frame, bg=SURFACE, fg=ACCT2, font=("Consolas", 8),
            relief="flat", bd=0, height=5, state="disabled"
        )
        self.py_snippet.pack(fill="x")
        self._update_py_snippet()

        # Right : curl + response
        sec_curl = SectionCard(right, "Requête cURL générée", "📡", WARN)
        sec_curl.pack(fill="x", pady=(0, 8))
        bc = sec_curl.body(pady=8)
        curl_toolbar = tk.Frame(bc, bg=CARD); curl_toolbar.pack(fill="x", pady=(0, 4))
        tk.Label(curl_toolbar, text="$ ", bg=CARD, fg=SUCCESS,
                 font=FONT_CODE_B).pack(side="left")
        self.curl_display = scrolledtext.ScrolledText(
            bc, bg=SURFACE, fg=WARN, font=FONT_CODE,
            height=5, relief="flat", bd=0
        )
        self.curl_display.pack(fill="x")
        self.curl_display.insert("1.0", "# La commande cURL apparaîtra ici après chaque requête")
        SmartButton(bc, "📋  Copier le cURL", self._copy_curl, WARN).pack(anchor="e", pady=(4, 0))

        sec_resp = SectionCard(right, "Réponse de l'API (JSON)", "📨", ACCENT)
        sec_resp.pack(fill="both", expand=True, pady=(0, 0))
        br = sec_resp.body()
        resp_toolbar = tk.Frame(br, bg=CARD); resp_toolbar.pack(fill="x", pady=(0, 4))
        self.resp_status_lbl = tk.Label(resp_toolbar, text="", bg=CARD,
                                        fg=SUBTEXT, font=FONT_CODE)
        self.resp_status_lbl.pack(side="left")
        self.resp_time_lbl   = tk.Label(resp_toolbar, text="", bg=CARD,
                                        fg=SUBTEXT, font=FONT_SMALL)
        self.resp_time_lbl.pack(side="right")
        self.api_response = scrolledtext.ScrolledText(
            br, bg=SURFACE, fg=TEXT, font=FONT_CODE,
            relief="flat", bd=0, selectbackground=ACCENT
        )
        self.api_response.pack(fill="both", expand=True)
        self.api_response.insert("1.0", "# La réponse JSON apparaîtra ici...")

        self._refresh_api_groups()

    def _on_api_group_select(self, e=None):
        self._update_py_snippet()
        self._build_curl_preview()

    def _build_curl_preview(self, method="POST", body=None):
        group = self.api_group.get()
        url_base = get_api_url()
        if not group:
            return
        if method == "POST":
            features_raw = self.api_features.get().strip()
            ids_raw      = self.api_entity_ids.get().strip()
            features = [f.strip() for f in features_raw.split(",") if f.strip()] if features_raw else None
            entity_ids = [i.strip() for i in ids_raw.split(",") if i.strip()] if ids_raw else None
            payload = json.dumps({"features": features, "entity_ids": entity_ids}, indent=2)
            curl = (f"curl -X POST \\\n"
                    f"  {url_base}/api/feature-groups/{group}/fetch \\\n"
                    f"  -H 'Content-Type: application/json' \\\n"
                    f"  -d '{payload}'")
        elif method == "GET_LIST":
            curl = f"curl {url_base}/api/feature-groups"
        elif method == "GET_HEALTH":
            curl = f"curl {url_base}/health"
        else:
            curl = f"curl {url_base}/api/feature-groups/{group}"
        self.curl_display.delete("1.0", "end")
        self.curl_display.insert("1.0", curl)

    def _copy_curl(self):
        curl = self.curl_display.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(curl)
        self._status("📋  cURL copié dans le presse-papiers", ACCT2)

    def _update_py_snippet(self):
        group = self.api_group.get() or "mon_groupe"
        url = get_api_url()
        snippet = (f"import urllib.request, json, pandas as pd\n"
                   f"\n"
                   f"payload = json.dumps({{'features': None}}).encode()\n"
                   f"req = urllib.request.Request(\n"
                   f"  '{url}/api/feature-groups/{group}/fetch',\n"
                   f"  data=payload, method='POST',\n"
                   f"  headers={{'Content-Type': 'application/json'}}\n"
                   f")\n"
                   f"data = json.loads(urllib.request.urlopen(req).read())\n"
                   f"df = pd.DataFrame(data['data'])")
        self.py_snippet.configure(state="normal")
        self.py_snippet.delete("1.0", "end")
        self.py_snippet.insert("1.0", snippet)
        self.py_snippet.configure(state="disabled")

    def _api_list(self):
        import urllib.request, urllib.error
        self._build_curl_preview("GET_LIST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(f"{get_api_url()}/api/feature-groups", timeout=3) as r:
                data = json.loads(r.read())
            self._show_response(data, 200, time.time() - t0)
        except Exception as e:
            self._show_response({"error": str(e)}, 500, 0)

    def _api_fetch(self):
        import urllib.request, urllib.error
        group = self.api_group.get()
        if not group:
            messagebox.showwarning("Groupe manquant", "Sélectionnez un groupe.")
            return
        features_raw = self.api_features.get().strip()
        ids_raw      = self.api_entity_ids.get().strip()
        features   = [f.strip() for f in features_raw.split(",") if f.strip()] if features_raw else None
        entity_ids = [i.strip() for i in ids_raw.split(",") if i.strip()] if ids_raw else None
        self._build_curl_preview("POST")
        payload = json.dumps({"features": features, "entity_ids": entity_ids}).encode()
        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{get_api_url()}/api/feature-groups/{group}/fetch",
                data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
            self._show_response(data, 200, time.time() - t0)
        except urllib.error.HTTPError as e:
            body = json.loads(e.read())
            self._show_response(body, e.code, time.time() - t0)
        except Exception as e:
            self._show_response({"error": str(e)}, 500, 0)

    def _api_health(self):
        import urllib.request
        self._build_curl_preview("GET_HEALTH")
        t0 = time.time()
        try:
            with urllib.request.urlopen(f"{get_api_url()}/health", timeout=3) as r:
                data = json.loads(r.read())
            self._show_response(data, 200, time.time() - t0)
        except Exception as e:
            self._show_response({"error": str(e)}, 500, 0)

    def _show_response(self, data, status_code, elapsed):
        text = json.dumps(data, indent=2, default=str)
        color = SUCCESS if status_code == 200 else DANGER
        status_icon = "✅" if status_code == 200 else "❌"
        self.resp_status_lbl.config(
            text=f" {status_icon}  HTTP {status_code} ",
            bg=color, fg=WHITE, padx=6, pady=2)
        self.resp_time_lbl.config(text=f"⏱  {elapsed*1000:.0f} ms")
        self.api_response.configure(fg=SUCCESS if status_code == 200 else DANGER)
        self.api_response.delete("1.0", "end")
        self.api_response.insert("1.0", text)
        self._status(f"📡  Réponse {status_code} — {elapsed*1000:.0f} ms — {len(text)} chars",
                     SUCCESS if status_code == 200 else DANGER)
        self._highlight_step(4)

    def _refresh_api_groups(self):
        try:
            groups = self.store.list_feature_groups()
            names = [g["name"] for g in groups]
            self.api_group["values"] = names
            if names and not self.api_group.get():
                self.api_group.set(names[0])
            self._update_py_snippet()
            self._build_curl_preview()
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # SHARED HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    def _make_tree(self, parent):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="both", expand=True)
        vsb = ttk.Scrollbar(frame, orient="vertical")
        hsb = ttk.Scrollbar(frame, orient="horizontal")
        tree = ttk.Treeview(frame, yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                             show="headings")
        vsb.configure(command=tree.yview)
        hsb.configure(command=tree.xview)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        return tree

    def _refresh_preview(self, df, tree, max_rows=100):
        tree.delete(*tree.get_children())
        if df is None or df.empty:
            return
        cols = list(df.columns)
        tree["columns"] = cols
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=max(90, min(160, len(str(col)) * 10)), minwidth=70)
        for i, (_, row) in enumerate(df.head(max_rows).iterrows()):
            vals = [("" if (isinstance(v, float) and math.isnan(v)) else str(v)) for v in row]
            tag = "even" if i % 2 == 0 else "odd"
            tree.insert("", "end", values=vals, tags=(tag,))
        tree.tag_configure("even", background=CARD)
        tree.tag_configure("odd",  background=CARD2)

    def _refresh_all_combos(self):
        if self.df_engineered is not None:
            df = self.df_engineered
        elif self.df_clean is not None:
            df = self.df_clean
        else:
            df = self.df_raw
        if df is None:
            return
        cols = list(df.columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

        if hasattr(self, "clean_col"):
            self.clean_col["values"] = cols
        if hasattr(self, "fe_col"):
            self.fe_col["values"] = cols
            self.fe_col2["values"] = cols
        if hasattr(self, "store_entity"):
            self.store_entity["values"] = cols
        if hasattr(self, "feat_listbox"):
            self.feat_listbox.delete(0, "end")
            for c in cols:
                self.feat_listbox.insert("end", c)


if __name__ == "__main__":
    app = App()
    app.mainloop()

"""Matplotlib defaults and the fixed per-method identity.

Centralizes the conventions so every figure looks the same: vector
PDF with embedded TrueType fonts (no Type-3), a serif body font, column-width
sizing, and a colorblind-safe palette that also survives grayscale printing
(methods stay distinguishable by marker/linestyle, not color alone). Importing
this module selects a non-interactive backend -- the analysis pipeline is batch
only, with no interactive step -- and `apply()` installs the rcParams.

The per-method visual identity (`method_style`) gives each run id one consistent
color + marker + linestyle + label used across every figure, so a method reads
the same everywhere and the cumulative-time ladder is legible at a glance.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # batch rendering only; the pipeline never displays a figure
import matplotlib.pyplot as plt  # noqa: E402

# column widths (inches)
WIDTH_SINGLE = 3.3
WIDTH_FULL = 7.0

# Okabe-Ito colorblind-safe palette (also separable in grayscale).
PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "purple": "#CC79A7",
}

# One fixed identity per run id, reused across every figure. Unknown run ids
# (e.g. external baselines) fall through to `_FALLBACK_STYLE`.
_METHOD_STYLE = {
    "scan":          {"color": PALETTE["black"],     "marker": "o", "linestyle": "-"},
    "kd":            {"color": PALETTE["skyblue"],    "marker": "s", "linestyle": "--"},
    "kd_agg":        {"color": PALETTE["blue"],       "marker": "D", "linestyle": "--"},
    "adkd":          {"color": PALETTE["orange"],     "marker": "^", "linestyle": "-."},
    "adkd_agg":      {"color": PALETTE["vermilion"],  "marker": "v", "linestyle": "-."},
    "adkd_sampling": {"color": PALETTE["green"],      "marker": "P", "linestyle": ":"},
    "a3i":           {"color": PALETTE["purple"],     "marker": "*", "linestyle": "-"},
}
_FALLBACK_STYLE = {"color": PALETTE["yellow"], "marker": "x", "linestyle": "-"}


def method_style(method: str) -> dict:
    """Fixed color/marker/linestyle/label for a run id (unknown -> a fallback)."""
    base = _METHOD_STYLE.get(method, _FALLBACK_STYLE)
    return {**base, "label": method}


_RCPARAMS = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,      # PNG previews; PDF/SVG output stays vector
}


def apply() -> None:
    """Install the figure rcParams. Idempotent; safe to call per figure."""
    plt.rcParams.update(_RCPARAMS)


def figure_size(width: str = "single", aspect: float = 0.75):
    """(w, h) inches for a `single`- or `full`-column figure at the given aspect."""
    w = WIDTH_FULL if width == "full" else WIDTH_SINGLE
    return (w, w * aspect)


apply()

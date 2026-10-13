"""Estilo matplotlib unico para figuras publicables del proyecto CCUS columna 1D."""
from __future__ import annotations

from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccus_columna.constantes import T_FIN_EXTRACCION, T_FIN_INYECCION, T_FIN  # parametros del proyecto

import matplotlib as mpl
import matplotlib.pyplot as plt

# Paleta colorblind-safe (Okabe-Ito)
COLORES = {
    "azul": "#0072B2",
    "naranja": "#E69F00",
    "verde": "#009E73",
    "rojo": "#D55E00",
    "morado": "#CC79A7",
    "celeste": "#56B4E9",
    "negro": "#000000",
    "gris": "#999999",
    "amarillo": "#F0E442",
}

ETAPAS = [
    ("Extraccion", 0.0, T_FIN_EXTRACCION, "#56B4E9"),
    ("Inyeccion CO2", T_FIN_EXTRACCION, T_FIN_INYECCION, "#E69F00"),
    ("Observacion", T_FIN_INYECCION, T_FIN, "#009E73"),
]


def aplicar_estilo() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def guardar(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, facecolor="white")
        paths.append(p)
    return paths


def sombrear_etapas(ax, ymax=None, ymin=None, alpha=0.12, label=True) -> None:
    for i, (nombre, t0, t1, color) in enumerate(ETAPAS):
        ax.axvspan(t0, t1, color=color, alpha=alpha, zorder=0)
        if label and ymax is not None:
            y = ymax if ymin is None else ymin + 0.92 * (ymax - ymin)
            ax.text(
                (t0 + t1) / 2,
                y,
                nombre,
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                alpha=0.9,
            )

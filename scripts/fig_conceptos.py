"""Figuras conceptuales didacticas (no dependen de corridas PHREEQC)."""
from __future__ import annotations

from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccus_columna.constantes import T_FIN_EXTRACCION, T_FIN_INYECCION, T_FIN  # parametros del proyecto

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

from estilo_publicacion import COLORES, ETAPAS, aplicar_estilo, guardar


def _caja(ax, xy, w, h, text, fc="#F7F7F7", ec="#333333", fontsize=9):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor=ec, facecolor=fc, mutation_aspect=0.3,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def fig_acople_cirrus_phreeqc(out_dir: Path) -> None:
    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Acople multifasico-geoquimico: la saturacion es dato", loc="left", pad=12)

    _caja(ax, (0.2, 1.3), 2.0, 1.4, "Cirrus\n(simulador\nmultifasico)", fc="#E8F4FC", ec=COLORES["azul"])
    _caja(ax, (3.0, 1.3), 2.2, 1.4, "Mapa S,P →\nm_w, V_gas, P_CO2\n(poro reactivo)", fc="#FFF4E0", ec=COLORES["naranja"])
    _caja(ax, (6.0, 1.3), 2.0, 1.4, "PHREEQC\n(quimica +\ncinetica)", fc="#E8F8F2", ec=COLORES["verde"])
    _caja(ax, (8.4, 1.3), 1.4, 1.4, "pH, C,\ncalcita,\nΔφ, I_alm", fc="#F5E8F0", ec=COLORES["morado"])

    for x0, x1 in ((2.2, 3.0), (5.2, 6.0), (8.0, 8.4)):
        ax.annotate("", xy=(x1, 2.0), xytext=(x0, 2.0),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#333333"))

    ax.text(5.0, 0.45,
            "Principio: Sg, Sw, So y P(t) vienen de Cirrus. PHREEQC no inventa saturaciones;\n"
            "solo integra el sistema geoquimico forzado por el mapa de saturacion.",
            ha="center", va="center", fontsize=9, color="#333333")

    guardar(fig, out_dir, "FIG01_acople_cirrus_phreeqc")
    plt.close(fig)


def fig_columna_y_etapas(out_dir: Path) -> None:
    aplicar_estilo()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.0, 1.4]})

    # Panel A: columna esquematica
    ax = axes[0]
    ax.set_xlim(0, 3)
    ax.set_ylim(9.5, 16.5)
    ax.set_xlabel("")
    ax.set_ylabel("Indice vertical k (arriba = sello)")
    ax.set_title("A. Columna 1D junto al pozo", loc="left")
    # sello
    ax.add_patch(Rectangle((0.8, 9.7), 1.4, 0.8, facecolor="#D55E00", alpha=0.35, edgecolor=COLORES["rojo"], lw=1.5))
    ax.text(1.5, 10.1, "k=10 sello\n(Esmeraldas)", ha="center", va="center", fontsize=8)
    # reservorio cells
    for k, y in zip(range(11, 16), np.linspace(10.7, 15.2, 5)):
        ax.add_patch(Rectangle((0.8, y), 1.4, 0.85, facecolor="#0072B2", alpha=0.18 + 0.05 * (k - 11),
                               edgecolor=COLORES["azul"], lw=1.0))
        label = f"k={k}" + (" inyeccion" if k == 15 else "")
        ax.text(1.5, y + 0.42, label, ha="center", va="center", fontsize=8)
    ax.annotate("frente CO2\nsube al sello", xy=(2.3, 10.5), xytext=(2.55, 13.5),
                fontsize=8, color=COLORES["naranja"],
                arrowprops=dict(arrowstyle="->", color=COLORES["naranja"]))
    ax.set_xticks([])

    # Panel B: timeline
    ax = axes[1]
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 3)
    ax.set_yticks([])
    ax.set_xlabel("Tiempo (años)")
    ax.set_title("B. Etapas operativas (70 años)", loc="left")
    for nombre, t0, t1, color in ETAPAS:
        ax.add_patch(Rectangle((t0, 1.0), t1 - t0, 1.0, facecolor=color, alpha=0.45, edgecolor=color, lw=1.2))
        ax.text((t0 + t1) / 2, 1.5, nombre, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.axvline(T_FIN_EXTRACCION, color="#333", ls="--", lw=0.8)
    ax.axvline(T_FIN_INYECCION, color="#333", ls="--", lw=0.8)
    ax.text(T_FIN_EXTRACCION, 0.35, f"{T_FIN_EXTRACCION:.1f} a", ha="center", fontsize=8)
    ax.text(T_FIN_INYECCION, 0.35, f"{T_FIN_INYECCION:.1f} a", ha="center", fontsize=8)
    ax.text(35, 2.45, "Forzante: perfiles Sg, Sw, So, P por celda (Cirrus)", ha="center", fontsize=8, color="#444")

    fig.tight_layout()
    guardar(fig, out_dir, "FIG02_columna_etapas")
    plt.close(fig)


def fig_dos_metodologias(out_dir: Path) -> None:
    aplicar_estilo()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # met1
    ax = axes[0]
    ax.set_title("met1 — celdas encadenadas", loc="left")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ys = [4.0, 2.8, 1.6]
    labels = ["k=15 (inyeccion)", "k=14 … k=11", "k=10 (sello)"]
    for y, lab in zip(ys, labels):
        _caja(ax, (0.4, y - 0.35), 2.2, 0.7, f"PHREEQC batch\n{lab}", fc="#E8F4FC", ec=COLORES["azul"], fontsize=8)
    for y0, y1 in ((3.65, 3.15), (2.45, 1.95)):
        ax.annotate("agua reaccionada\nsi Sg receptor > 0", xy=(1.5, y1), xytext=(1.5, y0),
                    fontsize=7, ha="center", color=COLORES["naranja"],
                    arrowprops=dict(arrowstyle="->", color=COLORES["naranja"]))
    ax.text(3.6, 2.5, "Estado entre\npasos vía\nDUMP / INCLUDE", ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="round", fc="#FFF4E0", ec=COLORES["naranja"]))

    # met2
    ax = axes[1]
    ax.set_title("met2 — TRANSPORT nativo PHREEQC", loc="left")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    _caja(ax, (0.5, 1.2), 4.0, 2.6,
          "Columna 1D advectivo-dispersiva\n"
          "cells / shifts / lengths / a_L\n\n"
          "Adveccion por hops (frente Cirrus)\n"
          "Dispersividad desde ancho de pluma\n"
          "(DC-029)",
          fc="#E8F8F2", ec=COLORES["verde"], fontsize=9)
    ax.annotate("", xy=(4.3, 2.5), xytext=(0.7, 2.5),
                arrowprops=dict(arrowstyle="->", lw=2, color=COLORES["verde"]))
    ax.text(2.5, 0.55, "Misma forzante Sg/P; distinto acople entre celdas", ha="center", fontsize=8)

    fig.tight_layout()
    guardar(fig, out_dir, "FIG03_dos_metodologias")
    plt.close(fig)


def main():
    # Default: relative to this file → figuras/publicacion
    here = Path(__file__).resolve()
    # Prefer project root figures when installed under scripts/
    candidates = [
        here.parents[1] / "figuras" / "publicacion",
        Path("figuras/publicacion"),
        Path("/workspace/ccus-pub/figuras_publicacion"),
    ]
    out = candidates[0]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            out = c
            break
        except Exception:
            continue
    fig_acople_cirrus_phreeqc(out)
    fig_columna_y_etapas(out)
    fig_dos_metodologias(out)
    print(f"Figuras conceptuales en: {out}")


if __name__ == "__main__":
    main()

"""GIFs met1 estilo sg_carbono: perfil C/Sg (izq) + columna por colores (der).

Variantes:
  - SI geoquimico: Calcite (reservorio) / Quartz (sello k=10) — DC-021
  - Delta phi / phi0 (%) con silicatos cinéticos en V_min
  - Permeabilidad efectiva K = K0 * (phi/phi0)^3  (Kozeny-Carman simplificado)

Uso:
  python comparacion/gifs_met1.py --malla malla_original --tasa <Q>
  python comparacion/gifs_met1.py --solo si
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from ccus_columna.constantes import TASA_REF, TASA_REFINADA_K_ESPECIAL  # parametros del proyecto
from ccus_columna.constantes import (
    K_INYECCION,
    K_RES_MD,
    K_RES_MD_REFINADA_50K,
    K_SELLO,
    K_SELLO_MD,
    K_SELLO_MD_REFINADA_50K,
)
from ccus_columna.mineralogia import phi0
from ccus_columna.postproceso import _fila_final_paso, calcular_delta_phi, leer_selected_output
from ccus_columna.rutas import dir_figuras_corrida

from sg_carbono import (
    LABEL_Q,
    MET1,
    _etapa,
    _perfil_en_t,
    ks_columna_met1,
    tabla_carbono_met1,
)

_FASE_A_SI = {
    "Calcite": "si_calcite",
    "Quartz": "si_quartz",
    "Pyrite": "si_pyrite",
    "Dolomite": "si_dolomite",
    "Dawsonite": "si_dawsonite",
    "Ankerite": "si_ankerite",
    "Gypsum": "si_gypsum",
    "K-feldspar": "si_k-feldspar",
    "Albite": "si_albite",
    "Kaolinite": "si_kaolinite",
    "Illite": "si_illite",
    "Chlorite(14A)": "si_chlorite(14a)",
    "Ca-Montmorillonite": "si_ca-montmorillonite",
    "K-mica": "si_k-mica",
    "Siderite": "si_siderite",
}


def _k0_md(malla: str, tasa: int, k: int) -> float:
    if malla == "malla_refinada" and tasa == TASA_REFINADA_K_ESPECIAL:
        return K_SELLO_MD_REFINADA_50K if k == K_SELLO else K_RES_MD_REFINADA_50K
    return K_SELLO_MD if k == K_SELLO else K_RES_MD


def mineral_si_por_celda(k: int) -> tuple[str, str]:
    """SI geoquimico: Calcite reservorio, Quartz sello (DC-021)."""
    if k == K_SELLO:
        return "Quartz", "si_quartz"
    return "Calcite", "si_calcite"


def tabla_si_geoquimico(run_dir: Path, ks: list[int]) -> pd.DataFrame:
    meta = {k: mineral_si_por_celda(k) for k in ks}

    def _si(row: pd.Series, k: int) -> float:
        _, col = meta[k]
        v = row.get(col, float("nan"))
        return float(v) if v is not None and v == v else float("nan")

    df = _tabla_desde_hitos(run_dir, ks, _si)
    df["fase"] = df["k"].map(lambda k: meta[int(k)][0])
    return df


def tabla_porosidad(run_dir: Path, ks: list[int], malla: str,
                    incluir_cinetica: bool = True) -> pd.DataFrame:
    df = _tabla_desde_hitos(
        run_dir, ks,
        lambda row, k: 100.0 * calcular_delta_phi(
            row, k=k, malla=malla, incluir_cinetica=incluir_cinetica,
        ) / phi0(k),
    )
    return df.rename(columns={"valor": "dphi_pct"})


def _tabla_desde_hitos(
    run_dir: Path,
    ks: list[int],
    valor_fn: Callable[[pd.Series, int], float],
) -> pd.DataFrame:
    run_dir = Path(run_dir)
    pasos_map: dict[int, float] = {}
    if (run_dir / "pasos.json").exists():
        raw = json.loads((run_dir / "pasos.json").read_text(encoding="utf-8"))
        pasos_map = {int(p["paso"]): float(p["t_fin"]) for p in raw}
    filas = []
    pat = re.compile(r"k(\d+)_p(\d+)")
    for path in sorted((run_dir / "hitos").glob("k*_p*.txt")):
        m = pat.search(path.name)
        if not m:
            continue
        k, paso = int(m.group(1)), int(m.group(2))
        if k not in ks:
            continue
        t = pasos_map.get(paso, float("nan"))
        if not np.isfinite(t):
            continue
        row = _fila_final_paso(leer_selected_output(path))
        filas.append({"tiempo_anos": t, "k": k, "valor": valor_fn(row, k)})
    return pd.DataFrame(filas).sort_values(["tiempo_anos", "k"])


def tabla_permeabilidad(
    run_dir: Path, malla: str, tasa: int, ks: list[int],
) -> pd.DataFrame:
    df = tabla_porosidad(run_dir, ks, malla, incluir_cinetica=True)
    phi_ref = df["k"].map(lambda k: phi0(int(k)))
    df["phi"] = phi_ref * (1.0 + df["dphi_pct"] / 100.0)
    k0 = df["k"].map(lambda k: _k0_md(malla, tasa, int(k)))
    ratio = (df["phi"] / phi_ref).clip(lower=0.05)
    df["K_md"] = k0 * ratio ** 3
    return df


def gif_met1_perfil_columna(
    tab_c: pd.DataFrame,
    tab_col: pd.DataFrame,
    col_valor: str,
    ks: list[int],
    malla: str,
    tasa: int,
    k_fondo: int,
    out_path: Path,
    *,
    titulo_base: str,
    col_label: str,
    cmap,
    norm,
    n_frames: int = 60,
    fps: int = 6,
    cbar_ticks: list[float] | None = None,
) -> Path:
    """Panel izq C/Sg + panel der columna de color (escala fija)."""
    ks_arr = np.array(ks, dtype=float)
    t_end = float(tab_c["tiempo_anos"].max())
    t_anim = np.linspace(0.0, t_end, n_frames)

    x_f = max(float(np.nanpercentile(tab_c["f_C"], 99)) * 1.08, 0.01)
    x_sg = max(float(tab_c["Sg_cirrus_pct"].max()) * 1.08, 5.0)
    y_lo, y_hi = ks_arr.min() - 0.5, ks_arr.max() + 0.5

    fig, (ax, ax_r) = plt.subplots(
        1, 2, figsize=(11.2, 5.4),
        gridspec_kw={"width_ratios": [4.2, 0.75], "wspace": 0.06},
    )
    axb = ax.twiny()
    line_f, = ax.plot([], [], "o-", color="#1565c0", ms=8, lw=2.2,
                      label=r"$\Delta n_C / n_{total}$ (proxy)")
    line_s, = axb.plot([], [], "s--", color="#e65100", ms=7, lw=1.8,
                       label="Sg Cirrus")
    fill = {"poly": None}

    ax.set_xlim(0, x_f)
    ax.set_ylim(y_hi, y_lo)
    ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.8, alpha=0.8)
    ax.axhline(k_fondo, color="#6a1b9a", ls=":", lw=0.8, alpha=0.8)
    ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.8, alpha=0.8)
    ax.set_xlabel(r"Proxy $\Delta n_C / n_{total}$   (C extra vs extraccion)")
    ax.set_ylabel("Celda k (sello arriba)")
    ax.grid(True, alpha=0.3)
    axb.set_xlim(0, x_sg)
    axb.set_xlabel("Sg Cirrus [% poro]", color="#e65100")
    axb.tick_params(axis="x", colors="#e65100")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(y_hi, y_lo)
    ax_r.set_xticks([])
    ax_r.set_xlabel(col_label, fontsize=9)
    ax_r.tick_params(axis="y", labelleft=False)
    for kk, c in ((K_INYECCION, "#1565c0"), (k_fondo, "#6a1b9a"), (K_SELLO, "#2e7d32")):
        ax_r.axhline(kk, color=c, ls=":", lw=0.8, alpha=0.5)

    v0 = float(np.nanmedian(tab_col[col_valor]))
    im = ax_r.imshow(
        np.full((len(ks), 1), v0),
        aspect="auto", cmap=cmap, norm=norm,
        extent=[0, 1, y_lo, y_hi], origin="upper", interpolation="nearest",
    )
    cax = make_axes_locatable(ax_r).append_axes("right", size="18%", pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(col_label, fontsize=8)
    if cbar_ticks:
        cbar.set_ticks(cbar_ticks)

    titulo = fig.suptitle("", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.82, bottom=0.14, wspace=0.18)

    def update(frame: int):
        tt = float(t_anim[frame])
        f = _perfil_en_t(tab_c, "f_C", tt, ks)
        sg = _perfil_en_t(tab_c, "Sg_cirrus_pct", tt, ks)
        col = _perfil_en_t(tab_col, col_valor, tt, ks)
        line_f.set_data(f, ks_arr)
        line_s.set_data(sg, ks_arr)
        if fill["poly"] is not None:
            fill["poly"].remove()
        fill["poly"] = ax.fill_betweenx(
            ks_arr, 0, np.nan_to_num(f, nan=0.0), color="#90caf9", alpha=0.35,
        )
        im.set_data(np.round(col, 4).reshape(-1, 1))
        s_calc = _perfil_en_t(tab_c, "S_calc_pct", tt, [K_INYECCION])[0]
        titulo.set_text(
            f"{titulo_base}  |  {malla} q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t = {tt:.1f} a ({_etapa(tt)})  |  k{k_fondo}->k10  "
            rf"|  $S_{{calc}}$ k15={s_calc:.1f}%"
        )
        return line_f, line_s, titulo

    from matplotlib import animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def generar_gifs_met1(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    run_dir: Path | None = None,
    out_dir: Path | None = None,
    n_frames: int = 60,
    fps: int = 6,
    solo: str | None = None,
) -> list[Path]:
    run_dir = Path(run_dir) if run_dir else MET1 / f"{malla}_q{tasa}_completo"
    out_dir = Path(out_dir) if out_dir else dir_figuras_corrida("met1", run_dir)
    ks = ks_columna_met1(malla, tasa, run_dir)
    k_fondo = ks[0]
    tab_c = tabla_carbono_met1(run_dir, malla, tasa)

    tab_si = tabla_si_geoquimico(run_dir, ks)
    si_vmin = float(np.floor(np.nanmin(tab_si["valor"].to_numpy()) * 10) / 10) - 0.2
    si_vmax = float(np.ceil(np.nanmax(tab_si["valor"].to_numpy()) * 10) / 10) + 0.2
    if si_vmax - si_vmin < 0.5:
        si_vmax = si_vmin + 0.5
    si_levels = np.linspace(si_vmin, si_vmax, 24)
    if si_vmin < 0 < si_vmax:
        norm_si = TwoSlopeNorm(vmin=si_vmin, vcenter=0.0, vmax=si_vmax)
        si_ticks = list(np.linspace(si_vmin, si_vmax, 5))
    else:
        norm_si = BoundaryNorm(si_levels, plt.cm.RdYlBu_r.N, clip=True)
        si_ticks = list(si_levels[::4])

    tab_phi = tabla_porosidad(run_dir, ks, malla, incluir_cinetica=True)
    phi_vmax = max(float(np.nanpercentile(tab_phi["dphi_pct"], 99)) * 1.1, 0.05)
    phi_levels = np.linspace(0.0, phi_vmax, 24)

    tab_k = tabla_permeabilidad(run_dir, malla, tasa, ks)
    k_min = float(tab_k["K_md"].min()) * 0.995
    k_max = float(tab_k["K_md"].max()) * 1.005
    k_levels = np.linspace(k_min, k_max, 24)

    specs = [
        ("si", tab_si, "valor",
         f"met1 — SI Calcite (res) / Quartz (sello)",
         "SI Calcite / Quartz", plt.cm.RdYlBu_r, norm_si,
         "gif_si_calcite_met1.gif", si_ticks),
        ("porosidad", tab_phi, "dphi_pct",
         "met1 — cambio de porosidad",
         r"$\Delta\phi / \phi_0$ (%)", plt.cm.YlOrRd,
         BoundaryNorm(phi_levels, plt.cm.YlOrRd.N, clip=True),
         "gif_porosidad_met1.gif", list(phi_levels[::4])),
        ("permeabilidad", tab_k, "K_md",
         r"met1 — $K_{eff}$ (Kozeny-Carman)",
         r"$K_{eff}$ (mD)", plt.cm.plasma,
         BoundaryNorm(k_levels, plt.cm.plasma.N, clip=True),
         "gif_permeabilidad_met1.gif", list(k_levels[::4])),
    ]

    paths: list[Path] = []
    for key, tab, col, titulo, clab, cmap, norm, fname, ticks in specs:
        if solo and solo != key:
            continue
        p = gif_met1_perfil_columna(
            tab_c, tab, col, ks, malla, tasa, k_fondo, out_dir / fname,
            titulo_base=titulo, col_label=clab, cmap=cmap, norm=norm,
            n_frames=n_frames, fps=fps, cbar_ticks=ticks,
        )
        tab.to_csv(out_dir / fname.replace(".gif", ".csv"), index=False, float_format="%.6g")
        paths.append(p)
    return paths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="GIFs met1: SI, porosidad, permeabilidad")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--solo", choices=("si", "porosidad", "permeabilidad"), default=None)
    args = ap.parse_args()
    run = args.run_dir or MET1 / f"{args.malla}_q{args.tasa}_completo"
    for p in generar_gifs_met1(
        args.malla, args.tasa, run, n_frames=args.n_frames, fps=args.fps, solo=args.solo,
    ):
        print(p)

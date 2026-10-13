"""GIF: caudal CO2 por celda desde Sg Cirrus — Q = V_poro * d(Sg)/dt (DC-023).

Uso:
  python comparacion/gif_caudal_inyeccion.py --malla malla_original --tasa <Q>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from ccus_columna.constantes import TASA_REF, TASAS_M3D, etiqueta_tasa  # parametros del proyecto
from ccus_columna.constantes import (
    K_INYECCION,
    K_INYECCION_FONDO,
    K_SELLO,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.inyeccion_cirrus import tabla_volumen_escenario
from ccus_columna.rutas import dir_figuras_corrida

LABEL_Q = {q: etiqueta_tasa(q) for q in TASAS_M3D}


def _etapa(t: float) -> str:
    if t < T_FIN_EXTRACCION:
        return "extraccion"
    if t < T_FIN_INYECCION:
        return "inyeccion"
    return "shut-in"


def _perfil_en_t(tab: pd.DataFrame, col: str, t_anos: float, ks: list[int]) -> np.ndarray:
    out = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        sub = tab[tab["k"].astype(int) == int(k)].sort_values("tiempo_anos")
        if sub.empty:
            continue
        out[i] = float(np.interp(
            t_anos,
            sub["tiempo_anos"].to_numpy(dtype=float),
            sub[col].to_numpy(dtype=float),
            left=float(sub[col].iloc[0]),
            right=float(sub[col].iloc[-1]),
        ))
    return out


def gif_caudal_inyeccion(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    n_frames: int = 60,
    fps: int = 6,
    out_path: Path | None = None,
    variante: str = "completo",
) -> Path:
    tab = tabla_volumen_escenario(malla, tasa)
    ks = sorted(tab["k"].unique(), reverse=True)
    ks_arr = np.array(ks, dtype=float)
    k_fondo = max(ks)
    t_end = float(tab["tiempo_anos"].max())
    t_anim = np.linspace(0.0, t_end, n_frames)

    iny = tab[tab["etapa_inyeccion"]]
    q_inst = iny["Q_desde_Sg_m3d"].clip(lower=0.0)
    q_max = max(float(np.nanpercentile(q_inst, 99)) * 1.15, 0.5)
    q_levels = np.linspace(0.0, q_max, 28)
    cmap = plt.cm.YlOrRd
    norm = BoundaryNorm(q_levels, cmap.N, clip=True)

    # Q media constante por celda (referencia inyeccion pareja)
    q_med_ref = {
        int(k): float(g["Q_media_inyeccion_m3d"].iloc[0])
        for k, g in tab.groupby("k")
    }

    out_path = out_path or (
        dir_figuras_corrida("met1", f"{malla}_q{tasa}_{variante}")
        / "gif_caudal_inyeccion_cirrus.gif"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_lo, y_hi = ks_arr.min() - 0.5, ks_arr.max() + 0.5

    fig, (ax, ax_c) = plt.subplots(
        1, 2, figsize=(11.0, 5.6),
        gridspec_kw={"width_ratios": [4.0, 0.7], "wspace": 0.08},
    )
    line_q, = ax.plot([], [], "o-", color="#c62828", ms=7, lw=2.0,
                      label=r"$Q = V_{poro}\, dS_g/dt$ (instantaneo)")
    line_med, = ax.plot([], [], "s", color="#1565c0", ms=6, lw=0,
                        label=r"$Q_{media}$ inyeccion")
    fill = {"p": None}
    vline_sum = {"l": None}

    ax.set_xlim(0, q_max * 1.08)
    ax.set_ylim(y_hi, y_lo)
    for kk, c in (
        (K_INYECCION, "#1565c0"),
        (K_INYECCION_FONDO, "#6a1b9a"),
        (K_SELLO, "#2e7d32"),
    ):
        ax.axhline(kk, color=c, ls=":", lw=0.8, alpha=0.75)
    ax.set_xlabel(r"Caudal desde $S_g$ (m$^3$/d)")
    ax.set_ylabel("Celda k (sello arriba)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(y_hi, y_lo)
    ax_c.set_xticks([])
    ax_c.tick_params(axis="y", labelleft=False)
    for kk, c in (
        (K_INYECCION, "#1565c0"),
        (K_INYECCION_FONDO, "#6a1b9a"),
        (K_SELLO, "#2e7d32"),
    ):
        ax_c.axhline(kk, color=c, ls=":", lw=0.8, alpha=0.5)

    v0 = 0.0
    im = ax_c.imshow(
        np.full((len(ks), 1), v0), aspect="auto", cmap=cmap, norm=norm,
        extent=[0, 1, y_lo, y_hi], origin="upper", interpolation="nearest",
    )
    cax = make_axes_locatable(ax_c).append_axes("right", size="14%", pad=0.06)
    cb = fig.colorbar(im, cax=cax, label=r"$Q$ desde $S_g$ (m$^3$/d)")
    cb.ax.tick_params(labelsize=7)

    tit = fig.suptitle("", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.92, top=0.82, bottom=0.10)

    def update(fr: int):
        tt = float(t_anim[fr])
        q = np.clip(_perfil_en_t(tab, "Q_desde_Sg_m3d", tt, ks), 0.0, None)
        q_col = np.clip(_perfil_en_t(tab, "Q_desde_Sg_m3d", tt, ks), 0.0, None)
        q_med = np.array([q_med_ref.get(int(k), 0.0) for k in ks], dtype=float)
        mask_perf = (ks_arr >= K_INYECCION) & (ks_arr <= K_INYECCION_FONDO)
        q_sum = float(np.nansum(q[mask_perf])) if mask_perf.any() else 0.0

        line_q.set_data(q, ks_arr)
        line_med.set_data(q_med, ks_arr)
        if vline_sum["l"] is not None:
            vline_sum["l"].remove()
        vline_sum["l"] = ax.axvline(
            q_sum, color="#616161", ls="--", lw=1.2, alpha=0.8,
        )
        if fill["p"] is not None:
            fill["p"].remove()
        fill["p"] = ax.fill_betweenx(ks_arr, 0, q, color="#ffcdd2", alpha=0.45)
        im.set_data(q_col.reshape(-1, 1))
        tit.set_text(
            f"Caudal CO$_2$ desde $S_g$ (celda adyacente)  |  {malla} q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t = {tt:.1f} a ({_etapa(tt)})  |  suma k15-30 = {q_sum:.2f} m$^3$/d "
            f"(pozo {tasa} m$^3$/d)"
        )
        return line_q, line_med, tit

    from matplotlib import animation
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

    csv_out = out_path.with_suffix(".csv")
    tab[[
        "tiempo_anos", "k", "Sg", "V_gas_m3", "Q_desde_Sg_m3d",
        "Q_media_inyeccion_m3d", "V_poro_m3",
    ]].to_csv(csv_out, index=False, float_format="%.6g")
    return out_path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    p = gif_caudal_inyeccion(args.malla, args.tasa, args.n_frames, args.fps)
    print(p)
    print(p.with_suffix(".csv"))


if __name__ == "__main__":
    main()

"""GIF: permeabilidad de roca met1 — potencia vs Carman-Kozeny completo.

    potencia:   K/K0 = (phi/phi0)^3
    completo:   K/K0 = (phi/phi0)^3 * ((1-phi0)/(1-phi))^2

phi = phi0 + delta_phi (minerales PHREEQC). Escala lineal en %.

Salida: figuras/met1/{malla}/q{tasa}/{variante}/gif_delta_permeabilidad_met1.gif

Uso:
  python comparacion/gif_delta_permeabilidad_met1.py --malla malla_original --tasa <Q>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))
if str(_RAIZ / "comparacion") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "comparacion"))

from ccus_columna.constantes import TASA_REF  # parametros del proyecto
from ccus_columna.constantes import K_INYECCION, K_SELLO, T_FIN_EXTRACCION, T_FIN_INYECCION
from ccus_columna.permeabilidad import tabla_delta_permeabilidad
from ccus_columna.rutas import dir_figuras_corrida

from sg_carbono import LABEL_Q, MET1, _perfil_en_t, ks_columna_met1

K_PCT_REF = 100.0


def _etapa(t: float) -> str:
    if t < T_FIN_EXTRACCION:
        return "extraccion"
    if t < T_FIN_INYECCION:
        return "inyeccion"
    return "shut-in"


def _ratio_a_pct(ratio: np.ndarray) -> np.ndarray:
    return np.asarray(ratio, dtype=float) * 100.0


def _fmt_pct(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    if abs(v - 100.0) < 0.005:
        return "100.00%"
    if abs(v - 100.0) < 1.0:
        return f"{v:.3f}%"
    return f"{v:.2f}%"


def _rango_lineal_pct(*arrays: np.ndarray) -> tuple[float, float]:
    allv = np.concatenate([np.asarray(a, dtype=float) for a in arrays])
    lo = float(np.nanmin(allv))
    hi = float(np.nanmax(allv))
    pad = max(0.15, 0.25 * (hi - lo))
    lo = min(lo - pad, 99.85)
    hi = max(hi + pad, 100.15)
    d = max(100.0 - lo, hi - 100.0)
    return 100.0 - d, 100.0 + d


def gif_delta_permeabilidad_met1(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    run_dir: Path | None = None,
    out_dir: Path | None = None,
    n_frames: int = 60,
    fps: int = 6,
) -> Path:
    run_dir = Path(run_dir) if run_dir else MET1 / f"{malla}_q{tasa}_completo"
    out_dir = Path(out_dir) if out_dir else dir_figuras_corrida("met1", run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gif_delta_permeabilidad_met1.gif"

    ks = ks_columna_met1(malla, tasa, run_dir)
    k_fondo = ks[0]
    tab = tabla_delta_permeabilidad(run_dir, malla, tasa, ks)
    tab["K_pct_kc"] = tab["ratio_kc"] * 100.0
    tab["K_pct_kc_full"] = tab["ratio_kc_full"] * 100.0

    ks_arr = np.array(ks, dtype=float)
    k_ticks = [int(k) for k in ks]
    t_end = float(tab["tiempo_anos"].max())
    t_anim = np.linspace(0.0, t_end, n_frames)
    y_lo, y_hi = ks_arr.min() - 0.5, ks_arr.max() + 0.5

    pct_all_pot = _ratio_a_pct(tab["ratio_kc"].to_numpy())
    pct_all_full = _ratio_a_pct(tab["ratio_kc_full"].to_numpy())
    x_lo, x_hi = _rango_lineal_pct(pct_all_pot, pct_all_full)
    norm = Normalize(vmin=x_lo, vmax=x_hi)
    cmap = plt.cm.viridis

    fig, (ax, ax_pot, ax_full) = plt.subplots(
        1, 3, figsize=(13.5, 6.0),
        gridspec_kw={"width_ratios": [4.2, 0.55, 0.55], "wspace": 0.12},
    )
    line_pot, = ax.plot(
        [], [], "o-", color="#1565c0", ms=6, lw=2.0,
        label=r"Potencia $(\phi/\phi_0)^3$",
    )
    line_full, = ax.plot(
        [], [], "s-", color="#c62828", ms=5, lw=1.8,
        label=r"Carman-Kozeny $(\phi/\phi_0)^3((1-\phi_0)/(1-\phi))^2$",
    )
    ax.axvline(K_PCT_REF, color="#616161", ls="--", lw=1.0, alpha=0.7, label="100 % ($K_0$)")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_hi, y_lo)
    ax.set_yticks(k_ticks)
    ax.set_yticklabels([str(k) for k in k_ticks])
    for kk, c in ((K_INYECCION, "#1565c0"), (k_fondo, "#6a1b9a"), (K_SELLO, "#2e7d32")):
        ax.axhline(kk, color=c, ls=":", lw=0.8, alpha=0.75)
    ax.set_xlabel(r"$K / K_0$ (%)   disolucion $\rightarrow$   |   $\leftarrow$ precipitacion")
    ax.set_ylabel("Celda k (sello arriba)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")

    labels_pot: list = []
    labels_full: list = []

    for ax_r, tit in ((ax_pot, "Potencia"), (ax_full, "C-K full")):
        ax_r.set_xlim(0, 1)
        ax_r.set_ylim(y_hi, y_lo)
        ax_r.set_xticks([])
        ax_r.set_yticks(k_ticks)
        ax_r.set_yticklabels([])
        ax_r.set_title(tit, fontsize=9)
        ax_r.tick_params(axis="y", labelleft=False)
        for kk, c in ((K_INYECCION, "#1565c0"), (k_fondo, "#6a1b9a"), (K_SELLO, "#2e7d32")):
            ax_r.axhline(kk, color=c, ls=":", lw=0.8, alpha=0.5)

    im_pot = ax_pot.imshow(
        np.full((len(ks), 1), K_PCT_REF), aspect="auto", cmap=cmap, norm=norm,
        extent=[0, 1, y_lo, y_hi], origin="upper", interpolation="nearest",
    )
    im_full = ax_full.imshow(
        np.full((len(ks), 1), K_PCT_REF), aspect="auto", cmap=cmap, norm=norm,
        extent=[0, 1, y_lo, y_hi], origin="upper", interpolation="nearest",
    )
    cax = make_axes_locatable(ax_full).append_axes("right", size="12%", pad=0.08)
    cb = fig.colorbar(im_full, cax=cax)
    cb.set_label(r"$K / K_0$ (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    titulo = fig.suptitle("", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.80, bottom=0.12)

    def _clear_labels():
        for t in labels_pot + labels_full:
            t.remove()
        labels_pot.clear()
        labels_full.clear()

    def update(frame: int):
        tt = float(t_anim[frame])
        pct_pot = _ratio_a_pct(_perfil_en_t(tab, "ratio_kc", tt, ks))
        pct_full = _ratio_a_pct(_perfil_en_t(tab, "ratio_kc_full", tt, ks))
        line_pot.set_data(pct_pot, ks_arr)
        line_full.set_data(pct_full, ks_arr)
        im_pot.set_data(np.round(pct_pot, 6).reshape(-1, 1))
        im_full.set_data(np.round(pct_full, 6).reshape(-1, 1))

        _clear_labels()
        span = x_hi - x_lo
        for i, k in enumerate(ks_arr):
            xp, xf = float(pct_pot[i]), float(pct_full[i])
            labels_pot.append(ax.text(
                xp + 0.02 * span, k, _fmt_pct(xp), fontsize=5.5, color="#0d47a1",
                va="center", ha="left", clip_on=True,
            ))
            labels_full.append(ax.text(
                xf - 0.02 * span, k - 0.22, _fmt_pct(xf), fontsize=5.5, color="#b71c1c",
                va="center", ha="right", clip_on=True,
            ))

        titulo.set_text(
            f"met1 — potencia vs Carman-Kozeny  |  "
            f"{malla} q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t = {tt:.1f} a ({_etapa(tt)})  |  k{k_fondo}->k10"
        )
        return line_pot, line_full, im_pot, im_full, titulo

    from matplotlib import animation
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

    tab.to_csv(out_dir / "delta_permeabilidad_met1.csv", index=False, float_format="%.6g")
    return out_path


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="GIF permeabilidad potencia + Carman-Kozeny")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    run = args.run_dir or MET1 / f"{args.malla}_q{args.tasa}_completo"
    p = gif_delta_permeabilidad_met1(args.malla, args.tasa, run, n_frames=args.n_frames, fps=args.fps)
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

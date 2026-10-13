"""GIFs transporte met1 vs met2.

Problema del proxy Sg_calc (DC-016): C(4) equilibra rapido con CO2(g) mientras
Sg de Cirrus crece despacio → Sg_calc(met1) se adelanta y satura antes. Por eso:

  - Panel / curva met1 = Sg Cirrus (el transporte de met1 ES la saturacion)
  - Panel / curva met2 = Sg_calc desde C(4) (proxy calibrado, DC-016)
  - Overlay: ambas + opcional Sg_calc(met1) en gris (solo diagnostico)

Uso:
  python comparacion/gif_transporte_co2.py --todas
  python comparacion/gif_transporte_co2.py --malla malla_original --tasa <Q>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

_RAIZ = Path(__file__).resolve().parents[1]
_COMP = Path(__file__).resolve().parent
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))
if str(_COMP) not in sys.path:
    sys.path.insert(0, str(_COMP))

from ccus_columna.constantes import TASA_REF, TASAS_M3D, etiqueta_tasa  # parametros del proyecto
from ccus_columna import datos
from ccus_columna.constantes import (
    K_INYECCION,
    K_SELLO,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.rutas import dir_figuras_comparacion
from sg_aparente import baseline_c0, calibrar_c_star, sg_calc_en_t, sg_max_celdas

MET1 = _RAIZ / "met1_celdas_encadenadas" / "corridas"
MET2 = _RAIZ / "met2_transporte_phreeqc" / "corridas"
LABEL_Q = {q: etiqueta_tasa(q) for q in TASAS_M3D}
KS_RAMA = list(range(K_INYECCION, K_SELLO - 1, -1))  # 15..10


def _etapa(t: float) -> str:
    if t < T_FIN_EXTRACCION:
        return "extraccion"
    if t < T_FIN_INYECCION:
        return "inyeccion"
    return "shut-in"


def _sg_rama(malla: str, tasa: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = datos.perfil(malla, tasa, "Sg")
    t = p["tiempo_anos"].to_numpy(dtype=float)
    ks = np.array(KS_RAMA, dtype=float)
    Z = p[[f"k{int(k)}" for k in ks]].to_numpy(dtype=float)
    return t, ks, Z


def _c_perfil(malla: str, tasa: int, met: str) -> pd.DataFrame:
    if met == "met1":
        path = MET1 / f"{malla}_q{tasa}_completo" / "perfil_quimica_met1.csv"
    else:
        path = MET2 / f"{malla}_q{tasa}_reactivo" / "perfil_quimica_met2.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df[(df["k"] >= K_SELLO) & (df["k"] <= K_INYECCION)].copy()


def _perfil_c_en_t(df: pd.DataFrame, t_anos: float, ks: np.ndarray) -> np.ndarray:
    out = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        sub = df[df["k"].astype(int) == int(k)].sort_values("tiempo_anos")
        if sub.empty:
            continue
        out[i] = float(np.interp(
            t_anos,
            sub["tiempo_anos"].to_numpy(dtype=float),
            sub["C_total_mol_kgw"].to_numpy(dtype=float),
            left=float(sub["C_total_mol_kgw"].iloc[0]),
            right=float(sub["C_total_mol_kgw"].iloc[-1]),
        ))
    return out


def _sg_en_t(t_vec: np.ndarray, Sg: np.ndarray, t_anos: float) -> np.ndarray:
    out = np.empty(Sg.shape[1])
    for j in range(Sg.shape[1]):
        out[j] = float(np.interp(t_anos, t_vec, Sg[:, j]))
    return out


def _prep_sg_calc(malla: str, tasa: int):
    t_sg, ks, Sg = _sg_rama(malla, tasa)
    df1 = _c_perfil(malla, tasa, "met1")
    df2 = _c_perfil(malla, tasa, "met2")
    sg_max = sg_max_celdas(malla, tasa, KS_RAMA)
    c0_1 = baseline_c0(df1, KS_RAMA)
    c0_2 = baseline_c0(df2, KS_RAMA)
    c_star = calibrar_c_star(df1, malla, tasa, c0=c0_1, sg_max=sg_max)
    xlim = max(float(np.nanmax(Sg)) * 1.08, max(sg_max.values()) * 1.05, 0.15)
    t_end = min(
        float(t_sg.max()),
        float(df1["tiempo_anos"].max()),
        float(df2["tiempo_anos"].max()),
    )
    return {
        "t_sg": t_sg,
        "ks": ks,
        "Sg": Sg,
        "df1": df1,
        "df2": df2,
        "sg_max": sg_max,
        "c0_1": c0_1,
        "c0_2": c0_2,
        "c_star": c_star,
        "xlim": xlim,
        "t_end": t_end,
    }


def gif_sg_calc_met1_vs_met2(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    n_frames: int = 70,
    fps: int = 7,
    out_dir: Path | None = None,
) -> Path:
    """Dos paneles: Sg Cirrus (met1) | Sg_calc met2 — misma escala."""
    p = _prep_sg_calc(malla, tasa)
    ks = p["ks"]
    t_anim = np.linspace(0.0, p["t_end"], n_frames)
    out_dir = Path(out_dir) if out_dir else dir_figuras_comparacion(malla, tasa)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.4), sharey=True)
    line1, = ax1.plot([], [], "o-", color="#e65100", ms=8, lw=2.2,
                      label="Sg Cirrus (fuerza met1)")
    line2, = ax2.plot([], [], "o-", color="#1565c0", ms=8, lw=2.2,
                      label="Sg_calc met2 (desde C)")
    line_ref, = ax2.plot([], [], "s--", color="#e65100", ms=5, lw=1.3, alpha=0.75,
                         label="Sg Cirrus (ref)")
    fills = {"1": None, "2": None}

    for ax in (ax1, ax2):
        ax.set_xlim(0, p["xlim"])
        ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
        ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.9, alpha=0.8)
        ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.9, alpha=0.8)
        ax.set_xlabel(r"$S_g$   (misma escala)")
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Celda k (sello arriba)")
    ax1.set_title("met1 — transporte = Sg Cirrus")
    ax2.set_title("met2 — Sg_calc desde C(4)")
    ax1.legend(fontsize=7, loc="lower right")
    ax2.legend(fontsize=7, loc="lower right")

    titulo = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    def update(frame: int):
        tt = float(t_anim[frame])
        sg = _sg_en_t(p["t_sg"], p["Sg"], tt)
        s2 = sg_calc_en_t(p["df2"], tt, ks, p["c0_2"], p["c_star"], p["sg_max"])
        line1.set_data(sg, ks)
        line2.set_data(s2, ks)
        line_ref.set_data(sg, ks)
        for key, ax, vals, color in (
            ("1", ax1, sg, "#ffcc80"),
            ("2", ax2, s2, "#90caf9"),
        ):
            if fills[key] is not None:
                fills[key].remove()
            fills[key] = ax.fill_betweenx(
                ks, 0, np.nan_to_num(vals, nan=0.0), color=color, alpha=0.35,
            )
        titulo.set_text(
            f"Transporte comparable en $S_g$  |  {malla}  q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t = {tt:.1f} a  ({_etapa(tt)})  |  C*={p['c_star']:.4f}"
        )
        return line1, line2, line_ref, titulo

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )
    path = out_dir / "gif_transporte_sg_calc_met1_met2.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

    meta = out_dir / "sg_calc_calibracion.txt"
    meta.write_text(
        f"malla={malla}\ntasa={tasa}\nC_star={p['c_star']:.8f}\n"
        f"nota=met1 panel usa Sg Cirrus; met2 usa Sg_calc desde C\n"
        f"sg_max={p['sg_max']}\nc0_met1={p['c0_1']}\nc0_met2={p['c0_2']}\n",
        encoding="utf-8",
    )
    return path


def gif_sg_calc_overlay(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    n_frames: int = 70,
    fps: int = 7,
    out_dir: Path | None = None,
) -> Path:
    """Un eje: Sg Cirrus, Sg_calc met2 (+ Sg_calc met1 diagnostico)."""
    p = _prep_sg_calc(malla, tasa)
    ks = p["ks"]
    t_anim = np.linspace(0.0, p["t_end"], n_frames)
    out_dir = Path(out_dir) if out_dir else dir_figuras_comparacion(malla, tasa)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    line_sg, = ax.plot([], [], "o-", color="#e65100", ms=8, lw=2.4,
                       label="Sg Cirrus (met1)")
    line2, = ax.plot([], [], "s-", color="#1565c0", ms=7, lw=2.4,
                     label="Sg_calc met2")
    line1d, = ax.plot([], [], "^:", color="#9e9e9e", ms=5, lw=1.2, alpha=0.7,
                      label="Sg_calc met1 (C→Sg, diagnostico)")

    ax.set_xlim(0, p["xlim"])
    ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.9)
    ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.9)
    ax.set_xlabel(r"$S_g$ / $S_{g,\mathrm{calc}}$")
    ax.set_ylabel("Celda k (sello arriba)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    titulo = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    def update(frame: int):
        tt = float(t_anim[frame])
        line_sg.set_data(_sg_en_t(p["t_sg"], p["Sg"], tt), ks)
        line2.set_data(
            sg_calc_en_t(p["df2"], tt, ks, p["c0_2"], p["c_star"], p["sg_max"]), ks,
        )
        line1d.set_data(
            sg_calc_en_t(p["df1"], tt, ks, p["c0_1"], p["c_star"], p["sg_max"]), ks,
        )
        titulo.set_text(
            f"met1 (Sg) vs met2 (Sg_calc)  |  {malla} q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t={tt:.1f} a ({_etapa(tt)})"
        )
        return line_sg, line2, line1d, titulo

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    path = out_dir / "gif_transporte_sg_calc_overlay.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def gif_met1_vs_met2(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    n_frames: int = 70,
    fps: int = 7,
    out_dir: Path | None = None,
) -> Path:
    """Dos paneles C(4) con la misma escala + Sg de referencia (opcional)."""
    t_sg, ks, Sg = _sg_rama(malla, tasa)
    df1 = _c_perfil(malla, tasa, "met1")
    df2 = _c_perfil(malla, tasa, "met2")

    c_max = max(
        float(df1["C_total_mol_kgw"].max()),
        float(df2["C_total_mol_kgw"].max()),
        0.002,
    ) * 1.08
    sg_max = max(float(np.nanmax(Sg)) * 1.05, 0.15)

    t_end = min(float(t_sg.max()), float(df1["tiempo_anos"].max()),
                float(df2["tiempo_anos"].max()))
    t_anim = np.linspace(0.0, t_end, n_frames)

    out_dir = Path(out_dir) if out_dir else dir_figuras_comparacion(malla, tasa)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.4), sharey=True)
    line1, = ax1.plot([], [], "o-", color="#c62828", ms=8, lw=2.2,
                      label="C(4) met1 (saturaciones)")
    line2, = ax2.plot([], [], "o-", color="#1565c0", ms=8, lw=2.2,
                      label="C(4) met2 (TRANSPORT)")
    ax1b = ax1.twiny()
    ax2b = ax2.twiny()
    line_sg1, = ax1b.plot([], [], "s--", color="#e65100", ms=5, lw=1.3, alpha=0.75,
                          label="Sg Cirrus")
    line_sg2, = ax2b.plot([], [], "s--", color="#e65100", ms=5, lw=1.3, alpha=0.75,
                          label="Sg Cirrus")

    fills = {"1": None, "2": None}

    for ax, axb in ((ax1, ax1b), (ax2, ax2b)):
        ax.set_xlim(0, c_max)
        ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
        ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.9, alpha=0.8)
        ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.9, alpha=0.8)
        ax.set_xlabel("C(4)  [mol/kgw]   ← misma escala")
        ax.grid(True, alpha=0.3)
        axb.set_xlim(0, sg_max)
        axb.set_xlabel(r"$S_g$ Cirrus  (referencia)", color="#e65100", fontsize=8)
        axb.tick_params(axis="x", colors="#e65100", labelsize=8)

    ax1.set_ylabel("Celda k (sello arriba)")
    ax1.set_title("met1 — transporte por saturaciones")
    ax2.set_title("met2 — TRANSPORT PHREEQC 1D")

    h1, l1 = ax1.get_legend_handles_labels()
    h1b, l1b = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h1b, l1 + l1b, fontsize=7, loc="lower right")
    h2, l2 = ax2.get_legend_handles_labels()
    h2b, l2b = ax2b.get_legend_handles_labels()
    ax2.legend(h2 + h2b, l2 + l2b, fontsize=7, loc="lower right")

    titulo = fig.suptitle("", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    def update(frame: int):
        tt = float(t_anim[frame])
        c1 = _perfil_c_en_t(df1, tt, ks)
        c2 = _perfil_c_en_t(df2, tt, ks)
        sg = _sg_en_t(t_sg, Sg, tt)
        line1.set_data(c1, ks)
        line2.set_data(c2, ks)
        line_sg1.set_data(sg, ks)
        line_sg2.set_data(sg, ks)
        for key, ax, vals, color in (
            ("1", ax1, c1, "#ef9a9a"),
            ("2", ax2, c2, "#90caf9"),
        ):
            if fills[key] is not None:
                fills[key].remove()
            fills[key] = ax.fill_betweenx(
                ks, 0, np.nan_to_num(vals, nan=0.0), color=color, alpha=0.35,
            )
        titulo.set_text(
            f"CO$_2$ acuoso C(4) comparable  |  {malla}  q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t = {tt:.1f} a  ({_etapa(tt)})"
        )
        return line1, line2, line_sg1, line_sg2, titulo

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )
    path = out_dir / "gif_transporte_met1_met2.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def gif_overlay(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    n_frames: int = 70,
    fps: int = 7,
    out_dir: Path | None = None,
) -> Path:
    """C(4) met1 y met2 superpuestos."""
    t_sg, ks, Sg = _sg_rama(malla, tasa)
    df1 = _c_perfil(malla, tasa, "met1")
    df2 = _c_perfil(malla, tasa, "met2")
    c_max = max(
        float(df1["C_total_mol_kgw"].max()),
        float(df2["C_total_mol_kgw"].max()),
        0.002,
    ) * 1.08
    sg_max = max(float(np.nanmax(Sg)) * 1.05, 0.15)
    t_end = min(float(t_sg.max()), float(df1["tiempo_anos"].max()),
                float(df2["tiempo_anos"].max()))
    t_anim = np.linspace(0.0, t_end, n_frames)

    out_dir = Path(out_dir) if out_dir else dir_figuras_comparacion(malla, tasa)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    axb = ax.twiny()
    line1, = ax.plot([], [], "o-", color="#c62828", ms=8, lw=2.4, label="met1 C(4)")
    line2, = ax.plot([], [], "s-", color="#1565c0", ms=7, lw=2.4, label="met2 C(4)")
    line_sg, = axb.plot([], [], "--", color="#e65100", lw=1.5, alpha=0.8,
                        label="Sg Cirrus")

    ax.set_xlim(0, c_max)
    ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.9)
    ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.9)
    ax.set_xlabel("C(4) [mol/kgw]")
    ax.set_ylabel("Celda k (sello arriba)")
    ax.grid(True, alpha=0.3)
    axb.set_xlim(0, sg_max)
    axb.set_xlabel(r"$S_g$ (referencia)", color="#e65100", fontsize=9)
    axb.tick_params(axis="x", colors="#e65100")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    titulo = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def update(frame: int):
        tt = float(t_anim[frame])
        line1.set_data(_perfil_c_en_t(df1, tt, ks), ks)
        line2.set_data(_perfil_c_en_t(df2, tt, ks), ks)
        line_sg.set_data(_sg_en_t(t_sg, Sg, tt), ks)
        titulo.set_text(
            f"met1 vs met2 — C(4)  |  {malla} q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t={tt:.1f} a ({_etapa(tt)})"
        )
        return line1, line2, line_sg, titulo

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    path = out_dir / "gif_transporte_overlay.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def tasas_disponibles(malla: str) -> list[int]:
    out = []
    for p2 in sorted(MET2.glob(f"{malla}_q*_reactivo")):
        mid = p2.name[len(malla) + 1 :]
        q = mid.split("_")[0][1:]
        if not q.isdigit():
            continue
        tasa = int(q)
        p1 = MET1 / f"{malla}_q{tasa}_completo" / "perfil_quimica_met1.csv"
        if p1.exists() and (p2 / "perfil_quimica_met2.csv").exists():
            out.append(tasa)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="GIF transporte en Sg_calc (desde C(4)) met1 vs met2")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--todas", action="store_true")
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--tambien-c", action="store_true",
                    help="Generar tambien los GIF en C(4)")
    args = ap.parse_args()

    tasas = tasas_disponibles(args.malla) if args.todas else [args.tasa]
    if not tasas:
        print(f"Sin pares met1/met2 con CSV en {args.malla}")
        raise SystemExit(1)

    for q in tasas:
        print(f"=== {args.malla} q={q} ===")
        out = dir_figuras_comparacion(args.malla, q)
        print(f"  {gif_sg_calc_met1_vs_met2(args.malla, q, args.n_frames, args.fps, out)}")
        print(f"  {gif_sg_calc_overlay(args.malla, q, args.n_frames, args.fps, out)}")
        if args.tambien_c:
            print(f"  {gif_met1_vs_met2(args.malla, q, args.n_frames, args.fps, out)}")
            print(f"  {gif_overlay(args.malla, q, args.n_frames, args.fps, out)}")

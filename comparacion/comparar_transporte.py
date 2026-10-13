"""Comparacion del transporte: met1 (saturaciones) vs met2 (TRANSPORT PHREEQC).

Figura principal CMP_transporte.png:
  - heatmaps C(4) lado a lado
  - perfiles verticales C(k) a tiempos fijos
  - llegada del frente quimico (C o pH) vs frente Sg de Cirrus

Uso:
  python comparacion/comparar_transporte.py --todas
  python comparacion/comparar_transporte.py --malla malla_original --tasa <Q>
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from ccus_columna import datos
from ccus_columna.constantes import (
    K_INYECCION,
    K_SELLO,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.postproceso import matriz_perfil_quimica
from ccus_columna.rutas import dir_figuras_comparacion

MET1 = _RAIZ / "met1_celdas_encadenadas" / "corridas"
MET2 = _RAIZ / "met2_transporte_phreeqc" / "corridas"

# Umbral: C sube respecto al basal nativo (~6e-4)
C_UMBRAL = 0.0015
# Tiempos (a) para perfiles verticales
T_PERFILES = (15.0, 25.0, 40.0, 60.0)


def _cargar(malla: str, tasa: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    p1 = MET1 / f"{malla}_q{tasa}_completo" / "perfil_quimica_met1.csv"
    p2 = MET2 / f"{malla}_q{tasa}_reactivo" / "perfil_quimica_met2.csv"
    if not p1.exists():
        raise FileNotFoundError(p1)
    if not p2.exists():
        raise FileNotFoundError(p2)
    return pd.read_csv(p1), pd.read_csv(p2)


def _t_llegada_c(df: pd.DataFrame, umbral: float = C_UMBRAL) -> pd.DataFrame:
    """Primer tiempo con C(4) > umbral por celda."""
    filas = []
    for k, g in df.groupby("k"):
        g = g.sort_values("tiempo_anos")
        hit = g[g["C_total_mol_kgw"] > umbral]
        if hit.empty:
            continue
        filas.append({"k": int(k), "t_anos": float(hit["tiempo_anos"].iloc[0])})
    return pd.DataFrame(filas)


def _perfil_en_t(df: pd.DataFrame, t_anos: float) -> pd.DataFrame:
    """Perfil C(k) en el tiempo mas cercano a t_anos."""
    ts = df["tiempo_anos"].unique()
    t0 = float(ts[np.argmin(np.abs(ts - t_anos))])
    sub = df[df["tiempo_anos"] == t0].sort_values("k", ascending=False)
    return sub.assign(t_ref=t0)


def figura_transporte(malla: str, tasa: int, out_dir: Path | None = None) -> Path:
    p1, p2 = _cargar(malla, tasa)
    out_dir = Path(out_dir) if out_dir else dir_figuras_comparacion(malla, tasa)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.28)

    # --- fila 1: heatmaps C ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    axd = fig.add_subplot(gs[0, 2])

    zs = []
    for df in (p1, p2):
        _, _, Z = matriz_perfil_quimica(df, "C")
        zs.append(Z)
    finite = np.concatenate([z[np.isfinite(z)].ravel() for z in zs])
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 0.01

    for ax, df, titulo in [
        (ax1, p1, "met1 — saturaciones\nC(4) mol/kgw"),
        (ax2, p2, "met2 — TRANSPORT PHREEQC\nC(4) mol/kgw"),
    ]:
        t, ks, Z = matriz_perfil_quimica(df, "C")
        im = ax.pcolormesh(t, ks, Z.T, shading="auto", cmap="YlOrRd",
                           vmin=vmin, vmax=vmax)
        ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
        ax.axhline(K_INYECCION, color="#1565c0", ls="--", lw=0.9)
        ax.axhline(K_SELLO, color="#2e7d32", ls="--", lw=0.9)
        ax.axvline(T_FIN_EXTRACCION, color="0.4", ls=":", lw=0.8)
        ax.axvline(T_FIN_INYECCION, color="0.4", ls=":", lw=0.8)
        ax.set_xlabel("Tiempo (a)")
        ax.set_ylabel("k")
        ax.set_title(titulo, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Delta C (met2 - met1) interpolado
    filas = []
    for k in sorted(p1["k"].unique()):
        a = p1[p1["k"] == k].sort_values("tiempo_anos")
        b = p2[p2["k"] == k].sort_values("tiempo_anos")
        if b.empty:
            continue
        bi = np.interp(a["tiempo_anos"], b["tiempo_anos"], b["C_total_mol_kgw"],
                       left=np.nan, right=np.nan)
        filas.append(pd.DataFrame({
            "tiempo_anos": a["tiempo_anos"].values,
            "k": int(k),
            "dC": bi - a["C_total_mol_kgw"].values,
        }))
    if filas:
        dC = pd.concat(filas, ignore_index=True)
        piv = dC.pivot_table(index="tiempo_anos", columns="k", values="dC")
        t = piv.index.to_numpy()
        ks = np.array(sorted(piv.columns, reverse=True))
        Z = np.column_stack([piv[k].values for k in ks])
        lim = float(np.nanmax(np.abs(Z[np.isfinite(Z)]))) if np.isfinite(Z).any() else 1e-3
        lim = max(lim, 1e-6)
        im = axd.pcolormesh(t, ks, Z.T, shading="auto", cmap="coolwarm",
                            vmin=-lim, vmax=lim)
        axd.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
        axd.set_title("ΔC (met2 − met1)", fontsize=10)
        axd.set_xlabel("Tiempo (a)")
        fig.colorbar(im, ax=axd, fraction=0.046, pad=0.02)

    # --- fila 2 izq: perfiles C(k) a tiempos fijos ---
    axp = fig.add_subplot(gs[1, 0])
    estilos = {
        15.0: (":", 1.0),
        25.0: ("--", 1.2),
        40.0: ("-", 1.4),
        60.0: ("-", 1.8),
    }
    for t_ref in T_PERFILES:
        s1 = _perfil_en_t(p1, t_ref)
        s2 = _perfil_en_t(p2, t_ref)
        ls, lw = estilos.get(t_ref, ("-", 1.2))
        axp.plot(s1["C_total_mol_kgw"], s1["k"], color="#c62828", ls=ls, lw=lw,
                 label=f"met1 @{t_ref:.0f}a")
        axp.plot(s2["C_total_mol_kgw"], s2["k"], color="#1565c0", ls=ls, lw=lw,
                 label=f"met2 @{t_ref:.0f}a")
    axp.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.7)
    axp.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.7)
    axp.invert_yaxis()
    axp.set_xlabel("C(4) mol/kgw")
    axp.set_ylabel("Celda k")
    axp.set_title("Perfil vertical C(k)", fontsize=10)
    axp.legend(fontsize=6, ncol=2, loc="best")
    axp.grid(True, alpha=0.25)

    # --- fila 2 centro: llegada frente ---
    axf = fig.add_subplot(gs[1, 1])
    r = datos.resumen_frente()
    r = r[(r["malla"] == malla) & (r["tasa_m3d"] == tasa)].dropna(
        subset=["t_primer_gas_anos"])
    # solo rama k15..k10
    r = r[(r["k"] >= K_SELLO) & (r["k"] <= K_INYECCION)].sort_values("k")
    axf.plot(r["t_primer_gas_anos"], r["k"], "ko-", ms=5, lw=1.2,
             label="Sg Cirrus (dato)")
    for df, lab, c, m in [
        (p1, "met1 C>umbral", "#c62828", "s"),
        (p2, "met2 C>umbral", "#1565c0", "D"),
    ]:
        ll = _t_llegada_c(df)
        if ll.empty:
            continue
        axf.plot(ll["t_anos"], ll["k"], color=c, marker=m, ls="--",
                 ms=6, lw=1.2, label=lab)
    axf.axvline(T_FIN_EXTRACCION, color="0.5", ls=":", lw=0.8)
    axf.axvline(T_FIN_INYECCION, color="0.5", ls=":", lw=0.8)
    axf.set_xlabel("Tiempo de llegada (a)")
    axf.set_ylabel("Celda k")
    axf.invert_yaxis()
    axf.set_title(f"Frente quimico vs gas\n(C > {C_UMBRAL:g} mol/kgw)", fontsize=10)
    axf.legend(fontsize=7, loc="best")
    axf.grid(True, alpha=0.25)

    # --- fila 2 der: series C en k15 y k10 ---
    axs = fig.add_subplot(gs[1, 2])
    for k, estilo_k in ((K_INYECCION, "-"), (K_SELLO, "--")):
        for df, lab, c in ((p1, "met1", "#c62828"), (p2, "met2", "#1565c0")):
            sub = df[df["k"].astype(int) == k].sort_values("tiempo_anos")
            if sub.empty:
                continue
            axs.plot(sub["tiempo_anos"], sub["C_total_mol_kgw"], color=c,
                     ls=estilo_k, lw=1.4,
                     label=f"{lab} k{k}")
    axs.axvspan(0, T_FIN_EXTRACCION, color="#bdbdbd", alpha=0.12, zorder=0)
    axs.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#90caf9", alpha=0.10, zorder=0)
    axs.axvspan(T_FIN_INYECCION, 70, color="#ffe082", alpha=0.10, zorder=0)
    axs.set_xlabel("Tiempo (a)")
    axs.set_ylabel("C(4) mol/kgw")
    axs.set_title("C en inyeccion (—) y sello (--)", fontsize=10)
    axs.legend(fontsize=6, loc="best", ncol=2)
    axs.grid(True, alpha=0.25)

    fig.suptitle(
        f"Transporte: saturaciones (met1) vs TRANSPORT PHREEQC (met2)\n"
        f"{malla}  q={tasa} m3/d",
        fontsize=12, y=0.995,
    )
    path = out_dir / "CMP_transporte.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def pares_disponibles(malla: str = "malla_original") -> list[int]:
    tasas = []
    for p1 in sorted(MET1.glob(f"{malla}_q*_completo")):
        mid = p1.name[len(malla) + 1 :]
        q = mid.split("_")[0][1:]
        if not q.isdigit():
            continue
        tasa = int(q)
        p2 = MET2 / f"{malla}_q{tasa}_reactivo" / "perfil_quimica_met2.csv"
        if (p1 / "perfil_quimica_met1.csv").exists() and p2.exists():
            tasas.append(tasa)
    return tasas


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Comparar transporte met1 vs met2")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=None)
    ap.add_argument("--todas", action="store_true")
    args = ap.parse_args()

    if args.todas or args.tasa is None:
        tasas = pares_disponibles(args.malla)
        if not tasas:
            print(f"Sin pares completo/reactivo para {args.malla}")
            raise SystemExit(1)
        for q in tasas:
            p = figura_transporte(args.malla, q)
            print(f"  {p}")
    else:
        p = figura_transporte(args.malla, args.tasa)
        print(f"  {p}")

"""Comparacion interna met1: mismas malla, distintas tasas Q.

Salidas en figuras/comparacion/met1_tasas/{malla}/:
  M1T_ph.png, M1T_carbono.png, M1T_calcita.png  — heatmaps por tasa
  M1T_series_inyeccion.png, M1T_series_sello.png — pH y C vs t (una curva/Q)
  M1T_resumen_final.png — barras pH/C/Δcalcita a t≈70 a en k15 y k10
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

from ccus_columna.constantes import (
    TASAS_M3D,
    K_INYECCION,
    K_SELLO,
    RAIZ,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.postproceso import anotar_delta_calcita, matriz_perfil_quimica

MET1_CORRIDAS = RAIZ / "met1_celdas_encadenadas" / "corridas"
OUT_ROOT = RAIZ / "figuras" / "comparacion" / "met1_tasas"

# Colores fijos por tasa (misma paleta en todas las figuras)
_COLOR_Q = dict(zip(TASAS_M3D, ("#1565c0", "#00838f", "#ef6c00", "#c62828")))
_HEATMAPS = (
    ("pH", "M1T_ph.png", "RdYlBu_r", "pH", 1.0, "pH"),
    ("C", "M1T_carbono.png", "YlOrRd", "C(4) mol/kgw", 1.0, "mol/kgw"),
    ("delta_calcita", "M1T_calcita.png", "RdBu", "Δ calcita (mol)", 1.0, "mol"),
)


def _label_q(tasa: int) -> str:
    if tasa >= 1000:
        return f"q={tasa // 1000}k"
    return f"q={tasa}"


def corridas_met1_completo(malla: str) -> list[tuple[int, Path]]:
    """[(tasa, run_dir)] con perfil CSV, ordenadas por tasa."""
    out = []
    for p in sorted(MET1_CORRIDAS.glob(f"{malla}_q*_completo")):
        csv = p / "perfil_quimica_met1.csv"
        if not csv.exists():
            continue
        # malla_original_q<Q>_completo
        mid = p.name[len(malla) + 1 :]  # q<Q>_completo
        if not mid.startswith("q") or "_completo" not in mid:
            continue
        q = mid.split("_", 1)[0][1:]
        if not q.isdigit():
            continue
        out.append((int(q), p))
    out.sort(key=lambda x: x[0])
    return out


def _cargar(run_dir: Path, malla: str) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "perfil_quimica_met1.csv")
    return anotar_delta_calcita(df, malla=malla)


def _bandas(ax):
    ax.axvspan(0, T_FIN_EXTRACCION, color="#bdbdbd", alpha=0.12, zorder=0)
    ax.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#90caf9", alpha=0.10, zorder=0)
    ax.axvspan(T_FIN_INYECCION, 70, color="#ffe082", alpha=0.10, zorder=0)


def _heatmaps_por_tasa(perfiles: dict[int, pd.DataFrame], malla: str,
                       variable: str, fname: str, cmap: str, titulo: str,
                       escala: float, cbar: str, out_dir: Path) -> Path:
    tasas = sorted(perfiles)
    n = len(tasas)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n + 1.5, 4.2), sharey=True,
                             constrained_layout=True)
    if n == 1:
        axes = [axes]

    zs = []
    for q in tasas:
        _, _, Z = matriz_perfil_quimica(perfiles[q], variable)
        zs.append(Z * escala)
    finite = np.concatenate([z[np.isfinite(z)].ravel() for z in zs if np.isfinite(z).any()])
    vmin = vmax = None
    if finite.size:
        if variable == "delta_calcita":
            vmax = max(float(np.nanmax(np.abs(finite))), 1e-9)
            vmin = -vmax
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))

    im = None
    for ax, q, Z in zip(axes, tasas, zs):
        t, ks, _ = matriz_perfil_quimica(perfiles[q], variable)
        im = ax.pcolormesh(t, ks, Z.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
        ax.set_xlabel("Tiempo (a)")
        ax.set_title(_label_q(q))
        ax.axhline(K_INYECCION, color="#1565c0", ls="--", lw=0.8)
        ax.axhline(K_SELLO, color="#2e7d32", ls="--", lw=0.8)
    axes[0].set_ylabel("Celda k (sello arriba)")
    if im is not None:
        fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02).set_label(cbar, fontsize=8)
    fig.suptitle(f"met1 — {titulo} por tasa — {malla}", y=1.02)
    path = out_dir / fname
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _series_celda(perfiles: dict[int, pd.DataFrame], k: int, malla: str,
                  out_path: Path, etiqueta: str) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for q, df in sorted(perfiles.items()):
        sub = df[df["k"].astype(int) == int(k)].sort_values("tiempo_anos")
        if sub.empty:
            continue
        c = _COLOR_Q.get(q, "#424242")
        lab = _label_q(q)
        axes[0].plot(sub["tiempo_anos"], sub["pH"], color=c, lw=1.5, label=lab)
        if "C_total_mol_kgw" in sub.columns:
            axes[1].plot(sub["tiempo_anos"], sub["C_total_mol_kgw"],
                         color=c, lw=1.5, label=lab)
    for ax in axes:
        _bandas(ax)
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("pH")
    axes[1].set_ylabel("C(4) mol/kgw")
    axes[1].set_xlabel("Tiempo (a)")
    fig.suptitle(f"met1 — {etiqueta} k={k} — {malla}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _resumen_final(perfiles: dict[int, pd.DataFrame], malla: str,
                   out_path: Path) -> Path:
    """Barras a t max ≈ 70 a en inyeccion y sello."""
    filas = []
    for q, df in perfiles.items():
        t_max = float(df["tiempo_anos"].max())
        for k, rol in ((K_INYECCION, "inyeccion"), (K_SELLO, "sello")):
            sub = df[(df["k"].astype(int) == k) & (df["tiempo_anos"] >= t_max - 0.6)]
            if sub.empty:
                continue
            row = sub.sort_values("tiempo_anos").iloc[-1]
            filas.append({
                "tasa": q,
                "k": k,
                "rol": rol,
                "pH": float(row["pH"]),
                "C": float(row.get("C_total_mol_kgw", np.nan)),
                "dcal": float(row.get("delta_calcita_mol", np.nan)),
            })
    if not filas:
        return out_path
    tab = pd.DataFrame(filas)
    tasas = sorted(tab["tasa"].unique())
    x = np.arange(len(tasas))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, ylab in [
        (axes[0], "pH", "pH a ~70 a"),
        (axes[1], "C", "C(4) mol/kgw a ~70 a"),
        (axes[2], "dcal", "Δ calcita (mol) a ~70 a"),
    ]:
        for i, (rol, color) in enumerate([("inyeccion", "#1565c0"), ("sello", "#2e7d32")]):
            ys = []
            for q in tasas:
                sub = tab[(tab["tasa"] == q) & (tab["rol"] == rol)]
                ys.append(float(sub[col].iloc[0]) if len(sub) else np.nan)
            ax.bar(x + (i - 0.5) * w, ys, width=w, label=rol, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([_label_q(q) for q in tasas])
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"met1 — estado final por tasa — {malla}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def comparar_tasas_met1(malla: str) -> list[Path]:
    pares = corridas_met1_completo(malla)
    if len(pares) < 2:
        raise RuntimeError(
            f"Hace falta al menos 2 corridas completo con CSV para {malla}; "
            f"hay {len(pares)}: {[q for q, _ in pares]}"
        )
    perfiles = {q: _cargar(run, malla) for q, run in pares}
    out_dir = OUT_ROOT / malla
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for var, fname, cmap, titulo, escala, cbar in _HEATMAPS:
        col = {"pH": "pH", "C": "C_total_mol_kgw",
               "delta_calcita": "delta_calcita_mol"}[var]
        if any(col not in df.columns for df in perfiles.values()):
            continue
        paths.append(_heatmaps_por_tasa(
            perfiles, malla, var, fname, cmap, titulo, escala, cbar, out_dir,
        ))

    paths.append(_series_celda(
        perfiles, K_INYECCION, malla,
        out_dir / "M1T_series_inyeccion.png", "inyeccion",
    ))
    paths.append(_series_celda(
        perfiles, K_SELLO, malla,
        out_dir / "M1T_series_sello.png", "sello",
    ))
    paths.append(_resumen_final(perfiles, malla, out_dir / "M1T_resumen_final.png"))

    # CSV compacto: ultimo valor por tasa/k
    filas = []
    for q, df in perfiles.items():
        t_max = float(df["tiempo_anos"].max())
        for k in sorted(df["k"].unique()):
            sub = df[(df["k"] == k) & (df["tiempo_anos"] >= t_max - 0.6)]
            if sub.empty:
                continue
            r = sub.sort_values("tiempo_anos").iloc[-1]
            filas.append({
                "malla": malla, "tasa": q, "k": int(k),
                "tiempo_anos": float(r["tiempo_anos"]),
                "pH": float(r["pH"]),
                "C_total_mol_kgw": float(r.get("C_total_mol_kgw", np.nan)),
                "delta_calcita_mol": float(r.get("delta_calcita_mol", np.nan)),
                "delta_phi": float(r.get("delta_phi", np.nan)),
            })
    csv_path = out_dir / "met1_tasas_final.csv"
    pd.DataFrame(filas).to_csv(csv_path, index=False, float_format="%.6f")
    paths.append(csv_path)
    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Comparar tasas Q dentro de met1")
    ap.add_argument("--malla", default=None,
                    help="malla_original o malla_refinada (si omite: ambas)")
    ap.add_argument("--todas", action="store_true",
                    help="original + refinada (igual que omitir --malla)")
    args = ap.parse_args()
    mallas = []
    if args.malla:
        mallas = [args.malla]
    else:
        mallas = ["malla_original", "malla_refinada"]
    any_ok = False
    for m in mallas:
        try:
            print(f"=== met1 tasas — {m} ===")
            for p in comparar_tasas_met1(m):
                print(f"  {p}")
            any_ok = True
        except RuntimeError as e:
            print(f"  (omitida) {e}")
    if not any_ok:
        raise SystemExit(1)

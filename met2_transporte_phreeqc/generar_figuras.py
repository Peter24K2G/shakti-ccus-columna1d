"""Figuras de resultados met2 (heatmaps k-t de quimica TRANSPORT y porosidad)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from ccus_columna.constantes import K_INYECCION, K_SELLO, PHI, T_FIN_EXTRACCION, T_FIN_INYECCION
from ccus_columna.postproceso import (
    anotar_delta_calcita,
    guardar_perfil,
    matriz_perfil_quimica,
    perfil_desde_met2,
)
from ccus_columna.rutas import dir_figuras_corrida, malla_desde_run


def _bandas_etapa(ax):
    ax.axvspan(0, T_FIN_EXTRACCION, color="#bdbdbd", alpha=0.12, zorder=0)
    ax.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#90caf9", alpha=0.10, zorder=0)
    ax.axvspan(T_FIN_INYECCION, 70, color="#ffe082", alpha=0.10, zorder=0)


def _detectar_archivo(run_dir: Path) -> str:
    for name in ("transporte_reactivo.txt", "transporte_trazador.txt"):
        if (run_dir / name).exists():
            return name
    raise FileNotFoundError(f"Sin salida TRANSPORT en {run_dir}")


def _guardar_heatmap(df: pd.DataFrame, variable: str, titulo: str, cmap: str,
                     path: Path, vmin=None, vmax=None, cbar_label: str | None = None,
                     escala: float = 1.0) -> Path:
    t, ks, Z = matriz_perfil_quimica(df, variable)
    Z = Z * escala
    k_iny = int(df["k"].astype(int).max())
    k_sello = int(df["k"].astype(int).min())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.pcolormesh(t, ks, Z.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axhline(k_iny, color="#1565c0", ls="--", lw=1, label=f"inyeccion k{k_iny}")
    if k_sello == K_SELLO or k_sello in ks:
        ax.axhline(k_sello, color="#2e7d32", ls="--", lw=1, label=f"sello k{k_sello}")
    _bandas_etapa(ax)
    ax.set_xlabel("Tiempo (años)")
    ax.set_ylabel("Celda k (sello arriba)")
    ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    ax.set_title(titulo)
    ax.legend(loc="lower right", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def figuras_met2(run_dir: Path, out_dir: Path | None = None,
                 usar_csv: bool = False) -> list[Path]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else dir_figuras_corrida("met2", run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "perfil_quimica_met2.csv"
    if not (usar_csv and csv_path.exists()):
        archivo = _detectar_archivo(run_dir)
        guardar_perfil(perfil_desde_met2(run_dir, archivo), csv_path)

    df = anotar_delta_calcita(pd.read_csv(csv_path), malla=malla_desde_run(run_dir))
    # Figuras solo rama al sello k15→k10 (el TRANSPORT puede incluir perforacion)
    df = df[(df["k"] >= K_SELLO) & (df["k"] <= K_INYECCION)]
    if "phi" not in df.columns and "delta_phi" in df.columns:
        df["phi"] = PHI + df["delta_phi"]
    tag = run_dir.name
    paths: list[Path] = []

    for var, fname, cmap, titulo in [
        ("pH", "M2_ph.png", "RdYlBu_r", f"pH — met2 TRANSPORT ({tag})"),
        ("C", "M2_carbono.png", "YlOrRd", f"C(4) (mol/kgw) — met2 ({tag})"),
    ]:
        p = _guardar_heatmap(df, var, titulo, cmap, out_dir / fname)
        paths.append(p)

    if not df["calcita_mol"].isna().all() and "pct_calcita" in df.columns:
        t, ks, Z = matriz_perfil_quimica(df, "pct_calcita")
        finite = Z[np.isfinite(Z)]
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1e-6
        vmax = max(vmax, 1e-6)
        p = _guardar_heatmap(
            df, "pct_calcita",
            f"Cambio de calcita — met2 ({tag})\n"
            "% respecto al inventario inicial m0  (Δ < 0 disolucion)",
            "RdBu", out_dir / "M2_calcita.png",
            vmin=-vmax, vmax=vmax,
            cbar_label="Δ calcita / m0 (%)",
        )
        paths.append(p)

    if not df["calcita_mol"].isna().all() and "delta_phi" in df.columns:
        t, ks, Z = matriz_perfil_quimica(df, "delta_phi")
        Z_pct = 100.0 * Z / PHI
        finite = Z_pct[np.isfinite(Z_pct)]
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1e-6
        vmax = max(vmax, 1e-6)
        vmin = -vmax if finite.size and float(np.nanmin(finite)) < 0 else 0.0
        p = _guardar_heatmap(
            df, "delta_phi",
            f"Cambio de porosidad — met2 ({tag})\n"
            f"dphi > 0 disolucion neta; phi0 = {PHI}",
            "YlOrRd", out_dir / "M2_delta_phi.png",
            vmin=vmin, vmax=vmax,
            cbar_label="dphi / phi0 (%)",
            escala=100.0 / PHI,
        )
        paths.append(p)

    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--usar-csv", action="store_true",
                    help="No reparsear TRANSPORT; usa perfil_quimica_met2.csv existente")
    args = ap.parse_args()
    for p in figuras_met2(args.run_dir, usar_csv=args.usar_csv):
        print(p)

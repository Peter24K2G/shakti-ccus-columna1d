"""Figuras de resultados met1 (heatmaps k-t de quimica y porosidad)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from ccus_columna.constantes import K_SELLO, PHI, T_FIN_EXTRACCION, T_FIN_INYECCION
from ccus_columna import datos
from ccus_columna.geometria import factor_moles_campo
from ccus_columna.mapa_saturacion import estado_poro
from ccus_columna.mineralogia import m_water_ref
from ccus_columna.postproceso import (
    anotar_delta_calcita,
    guardar_perfil,
    matriz_perfil_quimica,
    perfil_desde_met1,
)
from ccus_columna.rutas import dir_figuras_corrida, malla_desde_run, parse_id_corrida

R_L_ATM = 0.08205746
T_K = 60.0 + 273.15
MOL_A_T = 44.01e-6  # mol CO2 -> t


def _celdas_extremos(df: pd.DataFrame) -> tuple[int, int]:
    """Origen de la rama (k max) y sello (k min). No asume k=15."""
    ks = df["k"].dropna().astype(int)
    if ks.empty:
        raise ValueError("perfil_quimica sin columna k")
    return int(ks.max()), int(ks.min())


def _bandas_etapa(ax):
    ax.axvspan(0, T_FIN_EXTRACCION, color="#bdbdbd", alpha=0.12, zorder=0)
    ax.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#90caf9", alpha=0.10, zorder=0)
    ax.axvspan(T_FIN_INYECCION, 70, color="#ffe082", alpha=0.10, zorder=0)


def _heatmap(df: pd.DataFrame, variable: str, titulo: str, cmap: str,
             vmin=None, vmax=None, cbar_label: str | None = None,
             escala: float = 1.0) -> plt.Figure:
    t, ks, Z = matriz_perfil_quimica(df, variable)
    Z = Z * escala
    k_iny, k_sello = _celdas_extremos(df)
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
    return fig


def _serie_celda(df: pd.DataFrame, k: int) -> pd.DataFrame:
    sub = df[df["k"].astype(int) == int(k)].sort_values("tiempo_anos")
    return sub.dropna(subset=["tiempo_anos", "pH"])


def _series_sello(df: pd.DataFrame, titulo: str) -> plt.Figure:
    k_iny, k_sello = _celdas_extremos(df)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    k_up = _serie_celda(df, k_iny)
    k_down = _serie_celda(df, k_sello)

    if k_up.empty:
        axes[0].text(0.5, 0.5, f"sin datos k={k_iny}", ha="center", va="center",
                     transform=axes[0].transAxes, color="#9e9e9e")
        axes[1].text(0.5, 0.5, f"sin datos k={k_iny}", ha="center", va="center",
                     transform=axes[1].transAxes, color="#9e9e9e")
    else:
        axes[0].plot(k_up["tiempo_anos"], k_up["pH"], color="#c62828", lw=1.5)
        axes[1].plot(k_up["tiempo_anos"], k_up["pct_calcita"], color="#6a1b9a", lw=1.5)

    if k_down.empty:
        axes[2].text(0.5, 0.5, f"sin datos k={k_sello}", ha="center", va="center",
                     transform=axes[2].transAxes, color="#9e9e9e")
    else:
        axes[2].plot(k_down["tiempo_anos"], k_down["pH"], color="#2e7d32", lw=1.5)

    for ax in axes:
        _bandas_etapa(ax)
    axes[0].set_ylabel(f"pH k{k_iny}")
    axes[1].set_ylabel(f"Δ calcita k{k_iny} (%)")
    axes[2].set_ylabel(f"pH k{k_sello} (sello)")
    axes[2].set_xlabel("Tiempo (años)")
    fig.suptitle(titulo, y=1.01)
    fig.tight_layout()
    return fig


def _tabla_co2_atrapado(df: pd.DataFrame, malla: str, tasa: int) -> pd.DataFrame:
    """Inventario CO2 por (t,k): Sg Cirrus + moles gas/sol/min en vol. campo."""
    ks = sorted(int(k) for k in df["k"].unique())
    sg_max = {k: float(datos.celda(malla, tasa, k)["Sg"].max()) for k in ks}
    filas = []
    for _, row in df.iterrows():
        k = int(row["k"])
        t = float(row["tiempo_anos"])
        f = factor_moles_campo(malla, k)
        s = datos.interpolar_celda(malla, tasa, k, t)
        est = estado_poro(
            s["Sg"], s["Sw"], s["So"], s["P_bar"], sg_max[k],
            usar_p_real=True, k=k, malla=malla,
        )
        n_sol = float(row["C_total_mol_kgw"]) * m_water_ref(k, malla) * f
        n_gas = float(est["p_co2_atm"] * est["v_gas"] / (R_L_ATM * T_K)) * f
        dcal = float(row.get("delta_calcita_mol", 0.0) or 0.0)
        n_min = max(0.0, dcal) * f
        i_alm = n_sol + n_gas + n_min
        filas.append({
            "tiempo_anos": t,
            "k": k,
            "Sg_pct": 100.0 * s["Sg"],
            "n_gas_t": n_gas * MOL_A_T,
            "n_sol_t": n_sol * MOL_A_T,
            "n_min_t": n_min * MOL_A_T,
            "I_alm_t": i_alm * MOL_A_T,
        })
    return pd.DataFrame(filas).sort_values(["tiempo_anos", "k"])


def _heatmap_co2(tab: pd.DataFrame, col: str, ax, titulo: str, cmap: str,
                 vmin=None, vmax=None, cbar_label: str = ""):
    piv = tab.pivot_table(index="tiempo_anos", columns="k", values=col)
    t = piv.index.to_numpy()
    ks = np.array(sorted(piv.columns, reverse=True))
    Z = np.column_stack([piv[k].to_numpy() for k in ks])
    im = ax.pcolormesh(t, ks, Z.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    k_iny, k_sello = int(ks.max()), int(ks.min())
    ax.axhline(k_iny, color="#1565c0", ls=":", lw=0.8)
    ax.axhline(k_sello, color="#2e7d32", ls=":", lw=0.8)
    ax.axvline(T_FIN_EXTRACCION, color="0.45", ls=":", lw=0.7)
    ax.axvline(T_FIN_INYECCION, color="0.45", ls=":", lw=0.7)
    ax.set_xlabel("Tiempo (años)")
    ax.set_ylabel("Celda k")
    ax.set_title(titulo, fontsize=10)
    return im, cbar_label


def _fig_co2_atrapado(df: pd.DataFrame, malla: str, tasa: int, tag: str) -> plt.Figure:
    """CO2 atrapado: Sg gas (Cirrus) + inventario total I_alm (vol. campo)."""
    tab = _tabla_co2_atrapado(df, malla, tasa)
    vmax_sg = max(float(tab["Sg_pct"].max()) * 1.05, 5.0)
    vmax_i = max(float(tab["I_alm_t"].max()) * 1.05, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    im1, _ = _heatmap_co2(
        tab, "Sg_pct", axes[0],
        r"$S_g$ Cirrus (% del poro) — fase gas",
        "YlOrRd", vmin=0, vmax=vmax_sg, cbar_label="% poro",
    )
    fig.colorbar(im1, ax=axes[0], fraction=0.046).set_label("% poro", fontsize=8)
    im2, _ = _heatmap_co2(
        tab, "I_alm_t", axes[1],
        r"$I_{alm}$ total (t CO$_2$ / celda)",
        "YlOrRd", vmin=0, vmax=vmax_i, cbar_label="t CO2",
    )
    fig.colorbar(im2, ax=axes[1], fraction=0.046).set_label("t CO2 / celda", fontsize=8)
    fig.suptitle(
        f"CO$_2$ atrapado — met1 ({tag})\n"
        r"Izquierda: saturacion gas Cirrus;  derecha: $I_{alm}=n_{sol}+n_{gas}+n_{min}$ "
        r"(vol. campo; domina $n_{gas}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def figuras_met1(run_dir: Path, out_dir: Path | None = None,
                 usar_csv: bool = False) -> list[Path]:
    """Genera PNG en figuras/met1/{malla}/q{tasa}/{variante}/."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else dir_figuras_corrida("met1", run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "perfil_quimica_met1.csv"
    hitos = run_dir / "hitos"
    if usar_csv and csv_path.exists():
        pass
    elif hitos.is_dir() and any(hitos.glob("k*_p*.txt")):
        guardar_perfil(perfil_desde_met1(run_dir), csv_path)
    elif not csv_path.exists():
        guardar_perfil(perfil_desde_met1(run_dir), csv_path)

    df = anotar_delta_calcita(pd.read_csv(csv_path), malla=malla_desde_run(run_dir))
    if "phi" not in df.columns and "delta_phi" in df.columns:
        df["phi"] = PHI + df["delta_phi"]
    tag = run_dir.name
    paths: list[Path] = []

    for var, fname, cmap, titulo in [
        ("pH", "M1_ph.png", "RdYlBu_r", f"pH — met1 ({tag})"),
        ("C", "M1_carbono.png", "YlOrRd", f"C(4) disuelto (mol/kgw) — met1 ({tag})"),
    ]:
        fig = _heatmap(df, var, titulo, cmap)
        p = out_dir / fname
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    parsed = parse_id_corrida(run_dir.name)
    if parsed is not None:
        malla, tasa, _ = parsed
        fig = _fig_co2_atrapado(df, malla, tasa, tag)
        p = out_dir / "M1_co2_atrapado.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    t, ks, Z = matriz_perfil_quimica(df, "pct_calcita")
    finite = Z[np.isfinite(Z)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1e-6
    vmax = max(vmax, 1e-6)
    fig = _heatmap(
        df, "pct_calcita",
        f"Cambio de calcita — met1 ({tag})\n"
        "% respecto al inventario inicial m0  (Δ < 0 disolucion)",
        "RdBu", vmin=-vmax, vmax=vmax,
        cbar_label="Δ calcita / m0 (%)",
    )
    p = out_dir / "M1_calcita.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    t, ks, Z = matriz_perfil_quimica(df, "delta_phi")
    Z_pct = 100.0 * Z / PHI
    finite = Z_pct[np.isfinite(Z_pct)]
    vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1e-6
    vmax = max(vmax, 1e-6)
    vmin = -vmax if finite.size and float(np.nanmin(finite)) < 0 else 0.0
    fig = _heatmap(
        df, "delta_phi",
        f"Cambio de porosidad — met1 ({tag})\n"
        f"dphi > 0 disolucion neta (aumenta porosidad); phi0 = {PHI}",
        "YlOrRd", vmin=vmin, vmax=vmax,
        cbar_label="dphi / phi0 (%)",
        escala=100.0 / PHI,
    )
    p = out_dir / "M1_delta_phi.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig = _series_sello(df, f"Química inyección y sello — met1 ({tag})")
    p = out_dir / "M1_sello.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--usar-csv", action="store_true",
                    help="No reparsear hitos; usa perfil_quimica_met1.csv existente")
    args = ap.parse_args()
    for p in figuras_met1(args.run_dir, usar_csv=args.usar_csv):
        print(p)

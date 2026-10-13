"""Comparacion met1 vs met2: CSV unificado y figuras lado a lado + diferencia."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from ccus_columna.constantes import TASA_REF  # parametros del proyecto
from ccus_columna import datos
from ccus_columna.constantes import K_INYECCION, K_SELLO, PHI
from ccus_columna.postproceso import anotar_delta_calcita, matriz_perfil_quimica
from ccus_columna.rutas import dir_figuras_comparacion, malla_desde_run

OUT_DIR = Path(__file__).resolve().parent

# (variable en perfil, nombre archivo, cmap, titulo corto, escala, cbar)
_HEATMAPS = (
    ("pH", "CMP_ph.png", "RdYlBu_r", "pH", 1.0, "pH"),
    ("C", "CMP_carbono.png", "YlOrRd", "C(4) mol/kgw", 1.0, "mol/kgw"),
    ("pct_calcita", "CMP_calcita.png", "RdBu", "Δ calcita / m0 (%)", 1.0, "%"),
    ("delta_phi", "CMP_delta_phi.png", "YlOrRd", "Δφ / φ₀ (%)", 100.0 / PHI, "%"),
)


def cargar_perfiles(run_met1: Path, run_met2: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    p1 = pd.read_csv(run_met1 / "perfil_quimica_met1.csv")
    p2 = pd.read_csv(run_met2 / "perfil_quimica_met2.csv")
    malla = malla_desde_run(run_met1) or malla_desde_run(run_met2)
    p1 = anotar_delta_calcita(p1, malla=malla)
    p2 = anotar_delta_calcita(p2, malla=malla)
    # Comparacion visual solo k15→k10
    p1 = p1[(p1["k"] >= K_SELLO) & (p1["k"] <= K_INYECCION)].copy()
    p2 = p2[(p2["k"] >= K_SELLO) & (p2["k"] <= K_INYECCION)].copy()
    p1["met"] = "met1"
    p2["met"] = "met2"
    return p1, p2


def _tabla_delta(p1: pd.DataFrame, p2: pd.DataFrame, col: str) -> pd.DataFrame:
    """met2 interpolado a tiempos de met1; delta = met2 - met1."""
    filas = []
    for k in sorted(p1["k"].unique()):
        a = p1[p1["k"] == k].sort_values("tiempo_anos")
        b = p2[p2["k"] == k].sort_values("tiempo_anos")
        if b.empty or col not in a.columns or col not in b.columns:
            continue
        bi = np.interp(a["tiempo_anos"], b["tiempo_anos"], b[col],
                       left=np.nan, right=np.nan)
        filas.append(pd.DataFrame({
            "tiempo_anos": a["tiempo_anos"].values,
            "k": int(k),
            "variable": col,
            "met1": a[col].values,
            "met2": bi,
            "delta_met2_menos_met1": bi - a[col].values,
        }))
    if not filas:
        return pd.DataFrame()
    return pd.concat(filas, ignore_index=True)


def _col_perfil(variable: str) -> str:
    return {
        "pH": "pH",
        "C": "C_total_mol_kgw",
        "delta_calcita": "delta_calcita_mol",
        "pct_calcita": "pct_calcita",
        "delta_phi": "delta_phi",
    }.get(variable, variable)


def _heatmap_trio(p1: pd.DataFrame, p2: pd.DataFrame, delta: pd.DataFrame,
                  variable: str, cmap: str, titulo: str, escala: float,
                  cbar: str, malla: str, tasa: int, path: Path) -> Path:
    col = _col_perfil(variable)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    vmin = vmax = None
    zs = []
    for df in (p1, p2):
        _, _, Z = matriz_perfil_quimica(df, variable)
        zs.append(Z * escala)
    finite = np.concatenate([z[np.isfinite(z)].ravel() for z in zs if np.isfinite(z).any()])
    if finite.size:
        if variable in ("delta_calcita", "pct_calcita") or (variable == "delta_phi" and float(np.nanmin(finite)) < 0):
            vmax = float(np.nanmax(np.abs(finite)))
            vmax = max(vmax, 1e-9)
            vmin = -vmax
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))

    for ax, df, lab in [
        (axes[0], p1, f"met1 — {titulo}"),
        (axes[1], p2, f"met2 — {titulo}"),
    ]:
        t, ks, Z = matriz_perfil_quimica(df, variable)
        im = ax.pcolormesh(t, ks, (Z * escala).T, shading="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax)
        ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
        ax.set_xlabel("Tiempo (a)")
        ax.set_ylabel("k")
        ax.set_title(lab)
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(cbar, fontsize=8)

    d = delta[delta["variable"] == col] if not delta.empty else pd.DataFrame()
    if not d.empty:
        piv = d.pivot_table(index="tiempo_anos", columns="k",
                            values="delta_met2_menos_met1")
        t = piv.index.to_numpy()
        ks = np.array(sorted(piv.columns, reverse=True))
        Z = np.column_stack([piv[k].values for k in ks]) * escala
        fin = Z[np.isfinite(Z)]
        lim = float(np.nanmax(np.abs(fin))) if fin.size else 1.0
        lim = max(lim, 1e-9)
        if variable == "pH":
            lim = min(lim, 1.0)
        im = axes[2].pcolormesh(t, ks, Z.T, shading="auto", cmap="coolwarm",
                                vmin=-lim, vmax=lim)
        axes[2].set_ylim(ks.max() + 0.5, ks.min() - 0.5)
        axes[2].set_title(f"Δ (met2 − met1)")
        axes[2].set_xlabel("Tiempo (a)")
        fig.colorbar(im, ax=axes[2], fraction=0.046).set_label(cbar, fontsize=8)
    else:
        axes[2].text(0.5, 0.5, "sin overlap", ha="center", va="center",
                     transform=axes[2].transAxes)

    fig.suptitle(f"Comparacion met1 vs met2 — {malla} q={tasa}", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _series_celdas(p1: pd.DataFrame, p2: pd.DataFrame, malla: str, tasa: int,
                   path: Path) -> Path:
    """Series temporales en k inyeccion y sello: pH y C."""
    ks = [K_INYECCION, K_SELLO]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    vars_ = [("pH", "pH"), ("C_total_mol_kgw", "C(4) mol/kgw")]
    for j, k in enumerate(ks):
        for i, (col, ylab) in enumerate(vars_):
            ax = axes[i, j]
            for df, lab, c, ls in [
                (p1, "met1", "#c62828", "-"),
                (p2, "met2", "#1565c0", "--"),
            ]:
                sub = df[df["k"].astype(int) == k].sort_values("tiempo_anos")
                if sub.empty or col not in sub.columns:
                    continue
                ax.plot(sub["tiempo_anos"], sub[col], color=c, ls=ls, lw=1.4, label=lab)
            ax.set_title(f"k={k}" + (" (inyeccion)" if k == K_INYECCION else " (sello)"))
            ax.set_ylabel(ylab)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.25)
    for ax in axes[1, :]:
        ax.set_xlabel("Tiempo (a)")
    fig.suptitle(f"Series met1 vs met2 — {malla} q={tasa}", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _frente_quimico(p1: pd.DataFrame, p2: pd.DataFrame, malla: str, tasa: int,
                    path: Path) -> Path:
    r = datos.resumen_frente()
    r = r[(r["malla"] == malla) & (r["tasa_m3d"] == tasa)].dropna(
        subset=["t_primer_gas_anos"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(r["t_primer_gas_anos"], r["k"], "ko-", label="Frente Sg (Cirrus)", ms=4)
    for df, lab, c in [(p1, "met1 pH min < 7.5", "#c62828"),
                       (p2, "met2 pH min < 7.5", "#1565c0")]:
        hit = df.groupby("k")["pH"].min().reset_index()
        hit = hit[hit["pH"] < 7.5]
        if hit.empty:
            continue
        # primer tiempo en que pH baja de 7.5 en esa celda
        tq = []
        kk = []
        for k in hit["k"]:
            sub = df[df["k"] == k].sort_values("tiempo_anos")
            bajo = sub[sub["pH"] < 7.5]
            if not bajo.empty:
                tq.append(float(bajo["tiempo_anos"].iloc[0]))
                kk.append(int(k))
        if tq:
            ax.scatter(tq, kk, c=c, s=40, label=lab, zorder=3)
    ax.set_xlabel("Tiempo (a)")
    ax.set_ylabel("Celda k")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_title(f"Llegada frente quimico vs gas — {malla} q={tasa}")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def comparar(run_met1: Path, run_met2: Path, malla: str, tasa: int) -> list[Path]:
    """CSV + figuras CMP_* en figuras/comparacion/{malla}/q{tasa}/."""
    run_met1, run_met2 = Path(run_met1), Path(run_met2)
    for run, name in ((run_met1, "perfil_quimica_met1.csv"),
                       (run_met2, "perfil_quimica_met2.csv")):
        if not (run / name).exists():
            raise FileNotFoundError(f"Falta {run / name}")

    p1, p2 = cargar_perfiles(run_met1, run_met2)
    fig_dir = dir_figuras_comparacion(malla, tasa)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cols = [c for c in ("pH", "calcita_mol", "C_total_mol_kgw", "I_alm_mol",
                        "delta_phi", "delta_calcita_mol", "pct_calcita")
            if c in p1.columns and c in p2.columns]
    deltas = [_tabla_delta(p1, p2, c) for c in cols]
    comp = pd.concat([d for d in deltas if not d.empty], ignore_index=True)
    out_csv = fig_dir / "comparacion_resumen.csv"
    comp.to_csv(out_csv, index=False, float_format="%.6f")
    # copia en comparacion/ con nombre por tasa (no pisar otras)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(OUT_DIR / f"comparacion_{malla}_q{tasa}.csv",
                index=False, float_format="%.6f")

    paths: list[Path] = [out_csv]
    for var, fname, cmap, titulo, escala, cbar in _HEATMAPS:
        col = _col_perfil(var)
        if col not in p1.columns or col not in p2.columns:
            continue
        d = _tabla_delta(p1, p2, col)
        paths.append(_heatmap_trio(
            p1, p2, d, var, cmap, titulo, escala, cbar, malla, tasa,
            fig_dir / fname,
        ))

    paths.append(_series_celdas(p1, p2, malla, tasa, fig_dir / "CMP_series.png"))
    paths.append(_frente_quimico(p1, p2, malla, tasa, fig_dir / "CMP_frente.png"))
    try:
        from comparar_transporte import figura_transporte
        paths.append(figura_transporte(malla, tasa, out_dir=fig_dir))
    except Exception as exc:  # noqa: BLE001 — no tumbar el resto de CMP
        print(f"  (aviso) CMP_transporte omitida: {exc}")
    return paths


def pares_disponibles() -> list[tuple[str, int, Path, Path]]:
    """(malla, tasa, run_met1, run_met2) con ambos perfiles CSV."""
    met1_root = _RAIZ / "met1_celdas_encadenadas" / "corridas"
    met2_root = _RAIZ / "met2_transporte_phreeqc" / "corridas"
    out = []
    for m1 in sorted(met1_root.glob("malla_*_completo")):
        # malla_original_q<Q>_completo
        parts = m1.name.rsplit("_", 2)  # ..., q<Q>, completo
        if len(parts) < 3 or parts[-1] != "completo":
            continue
        q = parts[-2]  # q<Q>
        if not q.startswith("q") or not q[1:].isdigit():
            continue
        tasa = int(q[1:])
        malla = "_".join(parts[:-2])  # malla_original
        m2 = met2_root / f"{malla}_q{tasa}_reactivo"
        if (m1 / "perfil_quimica_met1.csv").exists() and (
                m2 / "perfil_quimica_met2.csv").exists():
            out.append((malla, tasa, m1, m2))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Comparar met1 vs met2")
    ap.add_argument("--met1", type=Path, default=None)
    ap.add_argument("--met2", type=Path, default=None)
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--todas", action="store_true",
                    help="comparar todos los pares completo/reactivo con CSV")
    args = ap.parse_args()

    if args.todas:
        pares = pares_disponibles()
        if not pares:
            print("No hay pares met1 completo + met2 reactivo con perfil CSV")
            raise SystemExit(1)
        for malla, tasa, m1, m2 in pares:
            print(f"=== {malla} q={tasa} ===")
            for p in comparar(m1, m2, malla, tasa):
                print(f"  {p}")
    else:
        if args.met1 is None or args.met2 is None:
            ap.error("indique --met1 y --met2, o use --todas")
        for p in comparar(args.met1, args.met2, args.malla, args.tasa):
            print(f"  {p}")

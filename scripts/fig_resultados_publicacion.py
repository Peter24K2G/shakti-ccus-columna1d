"""Figuras de resultados publicables a partir de perfil_quimica_met*.csv canonicos."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from estilo_publicacion import COLORES, aplicar_estilo, guardar, sombrear_etapas

ROOT = Path(__file__).resolve().parents[1]

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccus_columna.constantes import TASA_REF  # parametros del proyecto
OUT = ROOT / "figuras" / "publicacion"


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    # fuzzy contains
    for c in df.columns:
        cl = c.lower()
        for name in candidates:
            if name.lower() in cl:
                return c
    return None


def load_perfil(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    t = _pick_col(df, ["tiempo_anos", "time_years", "t_anos", "tiempo", "time"])
    k = _pick_col(df, ["k", "celda", "cell", "celda_k"])
    if t is None or k is None:
        raise SystemExit(f"No encuentro columnas t/k en {path}: {list(df.columns)}")
    df = df.rename(columns={t: "t", k: "k"})
    return df


def heatmap(ax, df, var, title, cmap="viridis", center=None):
    col = _pick_col(df, [var] if isinstance(var, str) else var)
    if col is None:
        ax.set_title(f"{title}\n(sin columna)")
        ax.axis("off")
        return None
    pivot = df.pivot_table(index="k", columns="t", values=col, aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)  # sello arriba
    data = pivot.values
    if center is not None:
        vmax = np.nanmax(np.abs(data - center))
        norm = colors.TwoSlopeNorm(vmin=center - vmax, vcenter=center, vmax=center + vmax)
    else:
        norm = None
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min() - 0.5, pivot.index.max() + 0.5],
        origin="upper",
    )
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("t (años)")
    ax.set_ylabel("k")
    return im


def fig_heatmaps_met1(df: pd.DataFrame, out: Path, tag: str) -> None:
    aplicar_estilo()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    specs = [
        (axes[0, 0], ["ph", "pH"], "pH", "RdYlBu_r", 7),
        (axes[0, 1], ["C_total_mol_kgw", "c4", "C(4)", "carbono"], "C(4) (mol/kgw)", "YlOrRd", None),
        (axes[1, 0], ["calcita_mol", "calcita", "delta_calcita"], "Calcita (mol)", "YlGn", None),
        (axes[1, 1], ["delta_phi", "dphi"], "Δφ / φ0 (%)", "coolwarm", 0),
    ]
    for ax, names, title, cmap, center in specs:
        im = heatmap(ax, df, names, title, cmap=cmap, center=center)
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"met1 — evolucion geoquimica de la columna ({tag})", fontsize=12, y=1.02)
    guardar(fig, out, f"FIG04_met1_heatmaps_{tag}")
    plt.close(fig)


def fig_series_inyeccion_sello(df: pd.DataFrame, out: Path, tag: str) -> None:
    aplicar_estilo()
    ph = _pick_col(df, ["ph", "pH"])
    c = _pick_col(df, ["C_total_mol_kgw", "c4", "C(4)", "carbono"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for ax, var, ylab in ((axes[0], ph, "pH"), (axes[1], c, "C(4) (mol/kgw)")):
        if var is None:
            ax.set_visible(False)
            continue
        for k, color, lab in ((15, COLORES["azul"], "k=15 inyeccion"), (10, COLORES["rojo"], "k=10 sello")):
            sub = df[df["k"] == k].sort_values("t")
            if sub.empty:
                # try nearest
                ks = df["k"].unique()
                if len(ks) == 0:
                    continue
                k2 = min(ks, key=lambda x: abs(x - k))
                sub = df[df["k"] == k2].sort_values("t")
                lab = f"k={k2}"
            ax.plot(sub["t"], sub[var], color=color, lw=1.8, label=lab)
        ymin, ymax = ax.get_ylim()
        sombrear_etapas(ax, ymax=ymax, ymin=ymin, label=(ax is axes[0]))
        ax.set_ylabel(ylab)
        ax.legend(loc="best")
    axes[-1].set_xlabel("t (años)")
    fig.suptitle(f"Serie en inyeccion vs sello ({tag})", fontsize=12)
    guardar(fig, out, f"FIG05_series_k15_k10_{tag}")
    plt.close(fig)


def fig_met1_vs_met2(df1: pd.DataFrame, df2: pd.DataFrame, out: Path, tag: str) -> None:
    aplicar_estilo()
    var_names = ["ph", "pH"]
    v1 = _pick_col(df1, var_names)
    v2 = _pick_col(df2, var_names)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    for ax, df, title, cmap in (
        (axes[0], df1, "met1 pH", "RdYlBu_r"),
        (axes[1], df2, "met2 pH", "RdYlBu_r"),
    ):
        im = heatmap(ax, df, ["ph", "pH"], title, cmap=cmap, center=7)
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # delta on common grid if possible
    ax = axes[2]
    if v1 and v2:
        p1 = df1.pivot_table(index="k", columns="t", values=v1, aggfunc="mean")
        p2 = df2.pivot_table(index="k", columns="t", values=v2, aggfunc="mean")
        # align
        common_t = p1.columns.intersection(p2.columns)
        common_k = p1.index.intersection(p2.index)
        if len(common_t) and len(common_k):
            d = p1.loc[common_k, common_t] - p2.loc[common_k, common_t]
            d = d.sort_index(ascending=False)
            vmax = np.nanmax(np.abs(d.values)) or 1
            im = ax.imshow(
                d.values,
                aspect="auto",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                extent=[common_t.min(), common_t.max(), common_k.min() - 0.5, common_k.max() + 0.5],
                origin="upper",
            )
            ax.set_title("ΔpH (met1 − met2)", loc="left", fontsize=10)
            ax.set_xlabel("t (años)")
            ax.set_ylabel("k")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")
            ax.set_title("Sin grilla comun t/k")
    else:
        ax.axis("off")
    fig.suptitle(f"Comparacion met1 vs met2 — pH ({tag})", fontsize=12)
    guardar(fig, out, f"FIG06_met1_vs_met2_ph_{tag}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--met1", type=Path, required=True)
    ap.add_argument("--met2", type=Path, default=None)
    ap.add_argument("--tag", default=f"q{TASA_REF}")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or OUT
    out.mkdir(parents=True, exist_ok=True)
    df1 = load_perfil(args.met1)
    print("met1 cols:", list(df1.columns))
    fig_heatmaps_met1(df1, out, args.tag)
    fig_series_inyeccion_sello(df1, out, args.tag)
    if args.met2 and args.met2.exists():
        df2 = load_perfil(args.met2)
        print("met2 cols:", list(df2.columns))
        fig_met1_vs_met2(df1, df2, out, args.tag)
    print("OK ->", out)


if __name__ == "__main__":
    main()

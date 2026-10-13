"""Diagnóstico met1 vs met2(dc018) vs met2(fugacidad_met1) — FIG07 3 vías."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccus_columna.constantes import TASA_REF, T_FIN_EXTRACCION, T_FIN_INYECCION, T_FIN  # parametros del proyecto
DEFAULT_MET1 = ROOT / f"met1_celdas_encadenadas/corridas/malla_original_q{TASA_REF}_completo_anual/perfil_quimica_met1.csv"
DEFAULT_MET2 = ROOT / f"met2_transporte_phreeqc/corridas/malla_original_q{TASA_REF}_reactivo_dc018/perfil_quimica_met2.csv"
DEFAULT_FUG = ROOT / f"met2_transporte_phreeqc/corridas/malla_original_q{TASA_REF}_reactivo_fugmet1/perfil_quimica_met2.csv"
DEFAULT_OUT = ROOT / "figuras" / "publicacion"

COLS = ["pH", "C_total_mol_kgw", "calcita_mol", "I_alm_mol"]
LABELS = {
    "pH": "pH",
    "C_total_mol_kgw": r"$C$ total (mol/kgw)",
    "calcita_mol": "Calcite (mol)",
    "I_alm_mol": r"$I_{\mathrm{alm}}$ (mol)",
}
COLORS = {
    "met1": "#0072B2",
    "met2_dc018": "#E69F00",
    "met2_fugmet1": "#009E73",
}


def series_at(df: pd.DataFrame, k: int, cols: list[str]) -> pd.DataFrame:
    sub = df[df["k"] == k].sort_values("tiempo_anos")
    keep = ["tiempo_anos"] + [c for c in cols if c in sub.columns]
    return sub[keep]


def metrics_at_70(df: pd.DataFrame, k: int) -> dict:
    s = series_at(df, k, COLS)
    if s.empty:
        return {}
    row = s.iloc[(s["tiempo_anos"] - 70).abs().argmin()]
    out = {"t": float(row["tiempo_anos"])}
    for c in COLS:
        if c in row:
            out[c] = float(row[c])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="FIG07 three-way met1 / met2 dc018 / met2 fugmet1")
    ap.add_argument("--met1", type=Path, default=DEFAULT_MET1)
    ap.add_argument("--met2", type=Path, default=DEFAULT_MET2)
    ap.add_argument("--met2-fug", type=Path, default=DEFAULT_FUG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    series = [("met1", args.met1, COLORS["met1"])]
    if args.met2.exists():
        series.append(("met2_dc018", args.met2, COLORS["met2_dc018"]))
    else:
        raise SystemExit(f"missing met2: {args.met2}")

    if args.met2_fug.exists():
        series.append(("met2_fugmet1", args.met2_fug, COLORS["met2_fugmet1"]))
        print(f"3-way: including {args.met2_fug}")
    else:
        print(f"NOTE: met2-fug absent ({args.met2_fug}); plotting 2-way only")

    frames = {name: pd.read_csv(path) for name, path, _ in series}

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    pairs = [
        (axes[0, 0], "pH"),
        (axes[0, 1], "C_total_mol_kgw"),
        (axes[1, 0], "calcita_mol"),
        (axes[1, 1], "I_alm_mol"),
    ]
    for ax, col in pairs:
        for name, _path, color in series:
            df = frames[name]
            for k, ls in ((15, "-"), (10, "--")):
                s = series_at(df, k, [col])
                if col not in s.columns or s.empty:
                    continue
                ax.plot(
                    s["tiempo_anos"],
                    s[col],
                    ls=ls,
                    color=color,
                    lw=1.6,
                    label=f"{name} k={k}",
                )
        for t0, t1, c in ((0, T_FIN_EXTRACCION, "#56B4E9"), (T_FIN_EXTRACCION, T_FIN_INYECCION, "#F0E442"), (T_FIN_INYECCION, T_FIN, "#CC79A7")):
            ax.axvspan(t0, t1, color=c, alpha=0.07, zorder=0)
        ax.set_ylabel(LABELS[col])
        ax.legend(fontsize=6.5, ncol=2, loc="best")
    axes[1, 0].set_xlabel("t (a)")
    axes[1, 1].set_xlabel("t (a)")
    n = len(series)
    fig.suptitle(
        f"Method comparison ({'3' if n == 3 else '2'}-way) — q50k original",
        fontsize=12,
    )
    stem = "FIG07_diagnostico_brecha_3vias" if n == 3 else "FIG07_diagnostico_brecha_met1_met2"
    fig.savefig(args.out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_series": n,
        "paths": {name: str(path) for name, path, _ in series},
        "at_70a": {name: {str(k): metrics_at_70(frames[name], k) for k in (15, 10)} for name, _, _ in series},
    }
    json_path = args.out_dir / "diagnostico_brecha_3vias.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("WROTE", args.out_dir / f"{stem}.pdf")
    print("WROTE", json_path)
    print(json.dumps(summary["at_70a"], indent=2))


if __name__ == "__main__":
    main()

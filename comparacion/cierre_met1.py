"""Cierre analitico met1 — snapshots, balance C, sensibilidad dP, sintesis 0D (DC-022).

Uso:
  python comparacion/cierre_met1.py --malla malla_original --tasa <Q>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))
sys.path.insert(0, str(_RAIZ / "met1_celdas_encadenadas"))

from ccus_columna.constantes import TASA_REF  # parametros del proyecto
from ccus_columna import datos
from ccus_columna.constantes import (
    BAR_A_ATM,
    K_INYECCION,
    K_SELLO,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.geometria import factor_moles_campo
from ccus_columna.mapa_saturacion import estado_poro
from ccus_columna.mineralogia import m_water_ref, v_poro
from ccus_columna.postproceso import _fila_final_paso, leer_selected_output
from ccus_columna.rutas import dir_figuras_comparacion

from fig_co2_atrapado import LABEL_Q, MET1
from sg_carbono import _perfil_en_t, ks_columna_met1, tabla_carbono_met1

import transferencia as tr

R_L_ATM = 0.08205746
T_K = 60.0 + 273.15
I_ALM_0D_50K = 0.285  # mol/celda 1 L @ 70 a (docs/00_CONTEXTO_LAPAZ_0D.md)
MINERALES_SEC = ("dawsonite", "ankerite", "pyrite", "calcite")
SNAPSHOT_T = (10.0, 30.0, 70.0)
DP_UMBRALES = (0.2, 0.5, 1.0)


def _tabla_desde_hitos(
    run_dir: Path, ks: list[int], valor_fn,
) -> pd.DataFrame:
    pasos_map = {}
    if (run_dir / "pasos.json").exists():
        raw = json.loads((run_dir / "pasos.json").read_text(encoding="utf-8"))
        pasos_map = {int(p["paso"]): float(p["t_fin"]) for p in raw}
    filas = []
    pat = re.compile(r"k(\d+)_p(\d+)")
    for path in sorted((run_dir / "hitos").glob("k*_p*.txt")):
        m = pat.search(path.name)
        if not m:
            continue
        k, paso = int(m.group(1)), int(m.group(2))
        if k not in ks:
            continue
        t = pasos_map.get(paso, float("nan"))
        if not np.isfinite(t):
            continue
        row = _fila_final_paso(leer_selected_output(path))
        filas.append({"tiempo_anos": t, "k": k, **valor_fn(row, k)})
    return pd.DataFrame(filas).sort_values(["tiempo_anos", "k"])


def tabla_minerales_sec(run_dir: Path, ks: list[int]) -> pd.DataFrame:
    def _vals(row, k):
        out = {}
        for m in MINERALES_SEC:
            out[f"{m}_mol"] = float(row.get(m, row.get(m.capitalize(), 0.0)) or 0.0)
        return out

    return _tabla_desde_hitos(run_dir, ks, _vals)


def sensibilidad_dp(malla: str, tasa: int, ks: list[int],
                    umbrales: tuple[float, ...] = DP_UMBRALES) -> pd.DataFrame:
    filas = []
    for umbral in umbrales:
        n_extra = 0
        for idx in range(1, len(ks)):
            k_orig, k_dest = ks[idx - 1], ks[idx]
            df_o = datos.celda(malla, tasa, k_orig)
            for _, row in df_o.iterrows():
                t = float(row["tiempo_anos"])
                sg_act = tr.transferencia_activa(malla, tasa, k_orig, k_dest, t, "sg")
                dp_act = tr.transferencia_activa(
                    malla, tasa, k_orig, k_dest, t, "sg_o_dp", dp_umbral_bar=umbral,
                )
                if dp_act and not sg_act:
                    n_extra += 1
        filas.append({"dp_umbral_bar": umbral, "pasos_extra_mix": n_extra})
    return pd.DataFrame(filas)


def auditoria_balance(tab_c: pd.DataFrame, malla: str, tasa: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """C por celda: PHREEQC vs capacidad gas Cirrus (receta 1 L)."""
    ks = sorted(tab_c["k"].unique())
    sg_max = {k: float(datos.celda(malla, tasa, k)["Sg"].max()) for k in ks}
    filas = []
    for _, row in tab_c.iterrows():
        k, t = int(row["k"]), float(row["tiempo_anos"])
        s = datos.interpolar_celda(malla, tasa, k, t)
        est = estado_poro(
            s["Sg"], s["Sw"], s["So"], s["P_bar"], sg_max[k],
            usar_p_real=True, k=k, malla=malla,
        )
        n_gas_cirrus = float(est["p_co2_atm"] * est["v_gas"] / (R_L_ATM * T_K))
        n_c_phreeqc = float(row["n_C_delta"])
        f = factor_moles_campo(malla, k)
        filas.append({
            "tiempo_anos": t, "k": k,
            "n_C_delta": n_c_phreeqc,
            "n_gas_cirrus": n_gas_cirrus,
            "n_C_delta_campo": n_c_phreeqc * f,
            "n_gas_cirrus_campo": n_gas_cirrus * f,
            "residual_pq": n_c_phreeqc - n_gas_cirrus,
            "residual_campo": (n_c_phreeqc - n_gas_cirrus) * f,
            "Sg_cirrus": s["Sg"],
        })
    df = pd.DataFrame(filas)
    g = df.groupby("tiempo_anos", as_index=False)[
        ["n_C_delta_campo", "n_gas_cirrus_campo", "residual_campo"]
    ].sum()
    return df, g


def fig_snapshots(
    tab_sg: pd.DataFrame,
    tab_dp: pd.DataFrame,
    perf_sg: pd.DataFrame,
    perf_dp: pd.DataFrame,
    ks: list[int],
    malla: str,
    tasa: int,
    out_path: Path,
) -> Path:
    """Perfiles verticales k30->k10 en tiempos clave."""
    ks_arr = np.array(ks, dtype=float)
    tiempos = SNAPSHOT_T
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex="col")
    vars_ = [
        ("f_C", r"Proxy $f_C$", tab_sg, tab_dp),
        ("S_calc_pct", r"$S_{calc}$ [% poro]", tab_sg, tab_dp),
    ]
    for r, (col, ylab, tb, tm) in enumerate(vars_):
        for c, t in enumerate(tiempos):
            ax = axes[r, c]
            v_sg = _perfil_en_t(tb, col, t, ks)
            v_dp = _perfil_en_t(tm, col, t, ks)
            ax.plot(v_sg, ks_arr, "k--", lw=1.5, label="sg only")
            ax.plot(v_dp, ks_arr, color="#1565c0", lw=2, label="sg_o_dp")
            for kk, cc in ((K_INYECCION, "#1565c0"), (K_SELLO, "#2e7d32")):
                ax.axhline(kk, color=cc, ls=":", lw=0.7, alpha=0.6)
            ax.set_ylim(ks_arr.max() + 0.5, ks_arr.min() - 0.5)
            ax.set_title(f"t = {t:.0f} a", fontsize=10)
            ax.set_ylabel("k" if c == 0 else "")
            ax.grid(True, alpha=0.25)
            if r == 0 and c == 2:
                ax.legend(fontsize=7, loc="lower right")
        axes[r, 0].set_ylabel(f"{ylab}\nk (sello arriba)")
    # pH from perfil row 2 equivalent - use row 1 for pH in second row... 
    # Actually use 3rd row via separate - we have 2 rows. Add pH to second row by replacing S_calc with pH in a separate figure
    fig.suptitle(
        f"Snapshots columna met1 — {malla} q={LABEL_Q.get(tasa, tasa)}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_snapshots_ph(
    perf_sg: pd.DataFrame,
    perf_dp: pd.DataFrame,
    ks: list[int],
    malla: str,
    tasa: int,
    out_path: Path,
) -> Path:
    ks_arr = np.array(ks, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for c, t in enumerate(SNAPSHOT_T):
        ax = axes[c]
        for perf, sty, lab in ((perf_sg, "k--", "sg"), (perf_dp, "-", "sg_o_dp")):
            sub = perf[perf["k"].isin(ks)].copy()
            ph = []
            for k in ks:
                sk = sub[sub["k"] == k].sort_values("tiempo_anos")
                if sk.empty:
                    ph.append(np.nan)
                else:
                    ph.append(float(np.interp(t, sk["tiempo_anos"], sk["pH"])))
            ax.plot(ph, ks_arr, sty, lw=1.8 if lab == "sg_o_dp" else 1.2,
                    color="#1565c0" if lab == "sg_o_dp" else "k", label=lab)
        ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.7, alpha=0.5)
        ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.7, alpha=0.5)
        ax.set_title(f"t = {t:.0f} a")
        ax.set_xlabel("pH")
        ax.grid(True, alpha=0.25)
        if c == 2:
            ax.legend(fontsize=8)
    axes[0].set_ylabel("k (sello arriba)")
    axes[0].set_ylim(ks_arr.max() + 0.5, ks_arr.min() - 0.5)
    fig.suptitle(f"pH columna — {malla} q={LABEL_Q.get(tasa, tasa)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_minerales_sec(
    tab_sg: pd.DataFrame,
    tab_dp: pd.DataFrame,
    malla: str,
    tasa: int,
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, mineral in zip(axes, ("dawsonite", "ankerite", "pyrite")):
        col = f"{mineral}_mol"
        for tab, sty, lab in ((tab_sg, "k--", "sg only"), (tab_dp, "-", "sg_o_dp")):
            sub = tab[tab["k"] == K_INYECCION].sort_values("tiempo_anos")
            ax.plot(sub["tiempo_anos"], sub[col], sty, lw=1.8 if "dp" in lab else 1.2,
                    color="#1565c0" if "dp" in lab else "k", label=lab)
        ax.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#fff3e0", alpha=0.35)
        ax.set_ylabel(f"{mineral} (mol/L)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Tiempo (a)")
    axes[0].set_title(
        f"Minerales secundarios k={K_INYECCION} — {malla} q={LABEL_Q.get(tasa, tasa)}",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_dispersion_masa(df_bal: pd.DataFrame, malla: str, tasa: int, out_path: Path) -> Path:
    """Cirrus n_gas vs PHREEQC n_C_delta (receta 1 L, inyeccion)."""
    sub = df_bal[
        (df_bal["tiempo_anos"] >= T_FIN_EXTRACCION) & (df_bal["Sg_cirrus"] > 1e-6)
    ].copy()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(
        sub["n_gas_cirrus"],
        sub["n_C_delta"],
        c=sub["k"], cmap="viridis", s=14, alpha=0.55, edgecolors="none",
    )
    lim = max(float(sub["n_C_delta"].max()), float(sub["n_gas_cirrus"].max())) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="1:1")
    ax.set_xlabel(r"$n_{gas}$ Cirrus (mol/L receta)")
    ax.set_ylabel(r"$\Delta n_C$ PHREEQC (mol/L receta)")
    ax.set_title(f"Dispersion flujo masico C — {malla} q={LABEL_Q.get(tasa, tasa)}")
    fig.colorbar(sc, ax=ax, label="celda k")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_balance_columna(g: pd.DataFrame, malla: str, tasa: int, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    t = g["tiempo_anos"]
    ax.plot(t, g["n_gas_cirrus_campo"], "k--", lw=1.5, label=r"$\Sigma n_{gas}$ Cirrus")
    ax.plot(t, g["n_C_delta_campo"], color="#1565c0", lw=2, label=r"$\Sigma \Delta n_C$ PHREEQC")
    ax.plot(t, g["residual_campo"], color="#c62828", lw=1, alpha=0.8, label="residual")
    ax.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#fff3e0", alpha=0.35)
    ax.set_xlabel("Tiempo (a)")
    ax.set_ylabel("mol C (campo, suma columna)")
    ax.set_title(f"Balance de masa C — columna k30->k10 ({malla} q={LABEL_Q.get(tasa, tasa)})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_sintesis_0d(
    perf_sg: pd.DataFrame,
    perf_dp: pd.DataFrame,
    malla: str,
    tasa: int,
    out_path: Path,
) -> Path:
    """k15 I_alm vs referencia batch 0D (tasa de referencia)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for perf, sty, lab in ((perf_sg, "k--", "met1 sg (columna)"), (perf_dp, "-", "met1 sg_o_dp")):
        sub = perf[perf["k"] == K_INYECCION].sort_values("tiempo_anos")
        ax.plot(sub["tiempo_anos"], sub["I_alm_mol"], sty, lw=1.8 if "dp" in lab else 1.2,
                label=lab)
    ax.axhline(I_ALM_0D_50K, color="#6a1b9a", ls=":", lw=1.5,
               label=f"0D batch ref. ({I_ALM_0D_50K} mol @ 70 a)")
    ax.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#fff3e0", alpha=0.35)
    ax.set_xlabel("Tiempo (a)")
    ax.set_ylabel(r"$I_{alm}$ (mol/L receta, k=15)")
    ax.set_title(f"Sintesis met1 vs 0D — {malla} q={LABEL_Q.get(tasa, tasa)}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def fig_sensibilidad_dp(df: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["dp_umbral_bar"].astype(str), df["pasos_extra_mix"], color="#1565c0", alpha=0.85)
    ax.set_xlabel(r"Umbral $\Delta P$ (bar)")
    ax.set_ylabel("Pasos extra MIX (sin Sg en destino)")
    ax.set_title("Sensibilidad activacion sg_o_dp")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Cierre analitico met1 (DC-022)")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--run-base", type=Path, default=MET1 / f"malla_original_q{TASA_REF}_completo")
    ap.add_argument("--run-dp", type=Path, default=MET1 / f"malla_original_q{TASA_REF}_completo_dp")
    args = ap.parse_args()

    run_sg = Path(args.run_base)
    run_dp = Path(args.run_dp)
    out = dir_figuras_comparacion(args.malla, args.tasa) / "cierre"
    out.mkdir(parents=True, exist_ok=True)

    ks = ks_columna_met1(args.malla, args.tasa, run_dp)
    tab_sg = tabla_carbono_met1(run_sg, args.malla, args.tasa)
    tab_dp = tabla_carbono_met1(run_dp, args.malla, args.tasa)
    min_sg = tabla_minerales_sec(run_sg, ks)
    min_dp = tabla_minerales_sec(run_dp, ks)
    perf_sg = pd.read_csv(run_sg / "perfil_quimica_met1.csv")
    perf_dp = pd.read_csv(run_dp / "perfil_quimica_met1.csv")

    paths = []
    paths.append(fig_snapshots(
        tab_sg, tab_dp, perf_sg, perf_dp, ks, args.malla, args.tasa,
        out / "snapshots_fc_scalc.png",
    ))
    paths.append(fig_snapshots_ph(
        perf_sg, perf_dp, ks, args.malla, args.tasa, out / "snapshots_ph.png",
    ))
    paths.append(fig_minerales_sec(min_sg, min_dp, args.malla, args.tasa, out / "minerales_sec_k15.png"))

    df_bal, g_bal = auditoria_balance(tab_dp, args.malla, args.tasa)
    df_bal.to_csv(out / "balance_carbono_celdas.csv", index=False, float_format="%.6g")
    g_bal.to_csv(out / "balance_carbono_columna.csv", index=False, float_format="%.6g")
    paths.append(fig_balance_columna(g_bal, args.malla, args.tasa, out / "balance_carbono_columna.png"))

    paths.append(fig_dispersion_masa(df_bal, args.malla, args.tasa, out / "dispersion_masa_c.png"))

    df_dp = sensibilidad_dp(args.malla, args.tasa, ks)
    df_dp.to_csv(out / "sensibilidad_dp.csv", index=False)
    paths.append(fig_sensibilidad_dp(df_dp, out / "sensibilidad_dp.png"))

    paths.append(fig_sintesis_0d(perf_sg, perf_dp, args.malla, args.tasa, out / "sintesis_0d_k15.png"))

    # Resumen JSON
    t70_sg = perf_sg[perf_sg["k"] == K_INYECCION].sort_values("tiempo_anos")
    i70_sg = float(t70_sg["I_alm_mol"].iloc[-1]) if not t70_sg.empty else float("nan")
    t70_dp = perf_dp[perf_dp["k"] == K_INYECCION].sort_values("tiempo_anos")
    i70_dp = float(t70_dp["I_alm_mol"].iloc[-1]) if not t70_dp.empty else float("nan")
    bal70 = g_bal.iloc[-1]
    resumen = {
        "I_alm_k15_70a_sg": i70_sg,
        "I_alm_k15_70a_dp": i70_dp,
        "I_alm_0d_ref_50k": I_ALM_0D_50K,
        "balance_70a_mol_campo": {
            "n_C_phreeqc": float(bal70["n_C_delta_campo"]),
            "n_gas_cirrus": float(bal70["n_gas_cirrus_campo"]),
            "residual": float(bal70["residual_campo"]),
        },
        "sensibilidad_dp": df_dp.to_dict(orient="records"),
    }
    (out / "resumen_cierre_met1.json").write_text(
        json.dumps(resumen, indent=2), encoding="utf-8",
    )

    for p in paths:
        print(p)
    print(f"  CSV/JSON -> {out}")


if __name__ == "__main__":
    main()

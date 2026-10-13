"""CO2 atrapado por celda y tiempo, escalado al volumen Cirrus (DC-015).

En PHREEQC la receta es 1 L (= celda original 50x50x20 m). Moles de campo:

    n_campo = n_phreeqc * factor_moles_campo(malla, k)

    n_sol = C(4) * m_water_ref * factor
    n_gas = P_CO2 * V_gas / (R T) * factor     (Sg/P Cirrus)
    n_min = max(0, delta_calcita) * factor      (precipitacion)
    I_alm = n_sol + n_gas + n_min

Uso:
  python comparacion/fig_co2_atrapado.py --todas
  python comparacion/fig_co2_atrapado.py --malla malla_original --tasa <Q>
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

from ccus_columna.constantes import TASA_REF, TASAS_M3D, etiqueta_tasa  # parametros del proyecto
from ccus_columna import datos
from ccus_columna.constantes import (
    K_INYECCION,
    K_SELLO,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.geometria import factor_moles_campo, v_bulk_m3
from ccus_columna.mapa_saturacion import estado_poro
from ccus_columna.mineralogia import m_water_ref
from ccus_columna.postproceso import anotar_delta_calcita
from ccus_columna.rutas import dir_figuras_comparacion

MET1 = _RAIZ / "met1_celdas_encadenadas" / "corridas"
MET2 = _RAIZ / "met2_transporte_phreeqc" / "corridas"
LABEL_Q = {q: etiqueta_tasa(q) for q in TASAS_M3D}
R_L_ATM_MOL_K = 0.08205746
T_K = 60.0 + 273.15
MOL_A_T = 44.01e-6  # mol CO2 -> t


def _cargar(malla: str, tasa: int, met: str) -> pd.DataFrame:
    if met == "met1":
        path = MET1 / f"{malla}_q{tasa}_completo" / "perfil_quimica_met1.csv"
    else:
        path = MET2 / f"{malla}_q{tasa}_reactivo" / "perfil_quimica_met2.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = anotar_delta_calcita(pd.read_csv(path), malla)
    return df[df["k"] >= K_SELLO].copy()


def tabla_co2_celdas(df: pd.DataFrame, malla: str, tasa: int, met: str) -> pd.DataFrame:
    """Una fila por (t, k) con moles de campo y volumen geometrico."""
    ks = sorted(int(k) for k in df["k"].unique())
    sg_max = {
        k: float(datos.celda(malla, tasa, k)["Sg"].max())
        for k in ks
    }
    filas = []
    for _, row in df.iterrows():
        k = int(row["k"])
        t = float(row["tiempo_anos"])
        f = factor_moles_campo(malla, k)
        mw = m_water_ref(k, malla)
        n_sol_pq = float(row["C_total_mol_kgw"]) * mw
        s = datos.interpolar_celda(malla, tasa, k, t)
        est = estado_poro(
            s["Sg"], s["Sw"], s["So"], s["P_bar"], sg_max.get(k, 1.0),
            usar_p_real=True, k=k, malla=malla,
        )
        n_gas_pq = float(est["p_co2_atm"] * est["v_gas"] / (R_L_ATM_MOL_K * T_K))
        dcal = float(row.get("delta_calcita_mol", 0.0) or 0.0)
        n_min_pq = max(0.0, dcal)
        n_sol = n_sol_pq * f
        n_gas = n_gas_pq * f
        n_min = n_min_pq * f
        filas.append({
            "met": met,
            "tiempo_anos": t,
            "k": k,
            "V_bulk_m3": v_bulk_m3(malla, k),
            "factor_campo": f,
            "n_sol_mol": n_sol,
            "n_gas_mol": n_gas,
            "n_min_mol": n_min,
            "I_alm_mol": n_sol + n_gas + n_min,
            "n_sol_t": n_sol * MOL_A_T,
            "n_gas_t": n_gas * MOL_A_T,
            "n_min_t": n_min * MOL_A_T,
            "I_alm_t": (n_sol + n_gas + n_min) * MOL_A_T,
            "C_mol_kgw": float(row["C_total_mol_kgw"]),
            "pct_calcita": float(row.get("pct_calcita", np.nan) or np.nan),
            "Sg": s["Sg"],
            "P_bar": s["P_bar"],
        })
    return pd.DataFrame(filas).sort_values(["tiempo_anos", "k"])


def serie_columna(tab: pd.DataFrame, ks: list[int] | None = None) -> pd.DataFrame:
    """Suma sobre celdas vs tiempo."""
    sub = tab if ks is None else tab[tab["k"].isin(ks)]
    g = sub.groupby("tiempo_anos", as_index=False)[
        ["n_sol_mol", "n_gas_mol", "n_min_mol", "I_alm_mol",
         "n_sol_t", "n_gas_t", "n_min_t", "I_alm_t"]
    ].sum()
    return g.sort_values("tiempo_anos")


def _heatmap_i_alm(tab: pd.DataFrame, ax, titulo: str):
    piv = tab.pivot_table(index="tiempo_anos", columns="k", values="I_alm_t")
    t = piv.index.to_numpy()
    ks = np.array(sorted(piv.columns, reverse=True))
    Z = np.column_stack([piv[k].to_numpy() for k in ks])
    im = ax.pcolormesh(t, ks, Z.T, shading="auto", cmap="YlOrRd")
    ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.9)
    ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.9)
    ax.set_xlabel("tiempo [a]")
    ax.set_ylabel("k")
    ax.set_title(titulo)
    return im


def fig_co2_atrapado(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    out_dir: Path | None = None,
) -> list[Path]:
    df1 = _cargar(malla, tasa, "met1")
    df2 = _cargar(malla, tasa, "met2")
    t1 = tabla_co2_celdas(df1, malla, tasa, "met1")
    t2 = tabla_co2_celdas(df2, malla, tasa, "met2")

    out_dir = Path(out_dir) if out_dir else dir_figuras_comparacion(malla, tasa)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # CSV por celda/tiempo
    csv_celdas = out_dir / "co2_atrapado_celdas.csv"
    pd.concat([t1, t2], ignore_index=True).to_csv(
        csv_celdas, index=False, float_format="%.6g",
    )
    paths.append(csv_celdas)

    ks_comun = sorted(set(t1["k"]) & set(t2["k"]))
    s1 = serie_columna(t1, ks_comun)
    s2 = serie_columna(t2, ks_comun)
    csv_sum = out_dir / "co2_atrapado.csv"
    out = s1.rename(columns=lambda c: f"met1_{c}" if c != "tiempo_anos" else c)
    out = out.merge(
        s2.rename(columns=lambda c: f"met2_{c}" if c != "tiempo_anos" else c),
        on="tiempo_anos", how="outer",
    ).sort_values("tiempo_anos")
    out.to_csv(csv_sum, index=False, float_format="%.6g")
    paths.append(csv_sum)

    # --- figura 1: stacked suma columna (t) ---
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, s, titulo, color in (
        (axes[0], s1, "met1 (saturaciones)", "#c62828"),
        (axes[1], s2, "met2 (TRANSPORT)", "#1565c0"),
    ):
        tt = s["tiempo_anos"].to_numpy()
        ax.stackplot(
            tt, s["n_sol_t"], s["n_gas_t"], s["n_min_t"],
            labels=[r"$n_{sol}$", r"$n_{gas}$", r"$n_{min}$"],
            colors=["#00838f", "#ef6c00", "#6a1b9a"], alpha=0.85,
        )
        ax.plot(tt, s["I_alm_t"], color=color, lw=1.8, label=r"$I_{alm}$")
        ax.axvline(T_FIN_EXTRACCION, color="0.5", ls=":", lw=0.9)
        ax.axvline(T_FIN_INYECCION, color="0.5", ls=":", lw=0.9)
        ax.set_xlabel("tiempo [a]")
        ax.set_title(titulo)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel(r"t CO$_2$ (suma celdas comunes, vol. campo)")
    vref = v_bulk_m3(malla, K_INYECCION)
    fig.suptitle(
        f"CO$_2$ atrapado (vol. campo)  |  {malla} q={LABEL_Q.get(tasa, tasa)}  "
        f"|  celda ref {vref:.0f} m$^3$  |  ks={ks_comun[0]}..{ks_comun[-1]}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = out_dir / "CMP_co2_atrapado.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- figura 2: heatmap I_alm por celda ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=False)
    im1 = _heatmap_i_alm(t1, axes[0], "met1 — $I_{alm}$ por celda")
    im2 = _heatmap_i_alm(t2, axes[1], "met2 — $I_{alm}$ por celda")
    for ax, im in ((axes[0], im1), (axes[1], im2)):
        fig.colorbar(im, ax=ax, fraction=0.046).set_label(r"t CO$_2$ / celda", fontsize=8)
    fig.suptitle(
        f"$I_{{alm}}(k,t)$ vol. campo  |  {malla} q={LABEL_Q.get(tasa, tasa)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out_dir / "CMP_co2_atrapado_celdas.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- figura 3: series por celda (comunes) ---
    ks_plot = [k for k in range(K_INYECCION, K_SELLO - 1, -1) if k in ks_comun]
    ncols = 3
    nrows = int(np.ceil(len(ks_plot) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.8 * nrows), sharex=True)
    axes = np.atleast_2d(axes)
    for i, k in enumerate(ks_plot):
        ax = axes[i // ncols, i % ncols]
        a = t1[t1["k"] == k].sort_values("tiempo_anos")
        b = t2[t2["k"] == k].sort_values("tiempo_anos")
        ax.plot(a["tiempo_anos"], a["I_alm_t"], color="#c62828", lw=1.5, label="met1")
        ax.plot(b["tiempo_anos"], b["I_alm_t"], color="#1565c0", lw=1.5, label="met2")
        ax.axvline(T_FIN_EXTRACCION, color="0.5", ls=":", lw=0.7)
        ax.axvline(T_FIN_INYECCION, color="0.5", ls=":", lw=0.7)
        ax.set_title(f"k={k}  V={v_bulk_m3(malla, k):.0f} m$^3$", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
        ax.set_ylabel(r"t CO$_2$")
    for j in range(len(ks_plot), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("tiempo [a]")
    fig.suptitle(
        f"$I_{{alm}}$ por celda (vol. campo)  |  {malla} q={LABEL_Q.get(tasa, tasa)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = out_dir / "CMP_co2_atrapado_series_k.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


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
    ap = argparse.ArgumentParser(description="CO2 atrapado por celda/tiempo (vol. campo)")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--todas", action="store_true")
    args = ap.parse_args()
    tasas = tasas_disponibles(args.malla) if args.todas else [args.tasa]
    for q in tasas:
        print(f"=== {args.malla} q={q} ===")
        for p in fig_co2_atrapado(args.malla, q):
            print(f"  {p}")

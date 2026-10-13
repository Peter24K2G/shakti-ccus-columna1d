"""Proxy visual Sg_calc a partir de C(4) (DC-016).

C(4) no es Sg: incluye disolucion mineral. No hay inversion termodinamica
limpia de D-018. Para comparar transporte met1 vs met2 en la misma magnitud
que Cirrus se usa:

    Sg_calc(k,t) = Sg_max(k) * clip( (C(k,t) - C0(k)) / C_star , 0, 1 )

C0(k) = mediana de C en extraccion (por serie).
C_star se calibra con met1 vs Sg de Cirrus (minimos cuadrados en la pluma).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ccus_columna import datos
from ccus_columna.constantes import K_INYECCION, K_SELLO, SG_UMBRAL_GAS, T_FIN_EXTRACCION

KS_RAMA = list(range(K_INYECCION, K_SELLO - 1, -1))


def sg_max_celdas(malla: str, tasa: int, ks: list[int] | None = None) -> dict[int, float]:
    ks = ks or KS_RAMA
    out: dict[int, float] = {}
    for k in ks:
        out[int(k)] = float(datos.celda(malla, tasa, int(k))["Sg"].max())
    return out


def baseline_c0(df: pd.DataFrame, ks: list[int] | None = None) -> dict[int, float]:
    """Mediana de C(4) en extraccion por celda (propia de la serie)."""
    ks = ks or sorted(int(k) for k in df["k"].unique())
    out: dict[int, float] = {}
    for k in ks:
        sub = df[(df["k"].astype(int) == int(k)) & (df["tiempo_anos"] <= T_FIN_EXTRACCION)]
        if sub.empty:
            sub = df[df["k"].astype(int) == int(k)]
        if sub.empty:
            out[int(k)] = 0.0
            continue
        out[int(k)] = float(sub["C_total_mol_kgw"].median())
    return out


def calibrar_c_star(
    df_met1: pd.DataFrame,
    malla: str,
    tasa: int,
    *,
    c0: dict[int, float] | None = None,
    sg_max: dict[int, float] | None = None,
) -> float:
    """C_star que minimiza RMSE de Sg_calc(met1) vs Sg Cirrus en la pluma."""
    c0 = c0 or baseline_c0(df_met1)
    sg_max = sg_max or sg_max_celdas(malla, tasa)

    cex_list: list[float] = []
    sg_list: list[float] = []
    smax_list: list[float] = []
    for _, row in df_met1.iterrows():
        k = int(row["k"])
        if k not in sg_max:
            continue
        t = float(row["tiempo_anos"])
        sg = datos.interpolar_celda(malla, tasa, k, t)["Sg"]
        if sg <= SG_UMBRAL_GAS:
            continue
        cex = max(float(row["C_total_mol_kgw"]) - c0.get(k, 0.0), 0.0)
        cex_list.append(cex)
        sg_list.append(sg)
        smax_list.append(sg_max[k])

    if not cex_list:
        return 0.02

    cex = np.asarray(cex_list, dtype=float)
    sg = np.asarray(sg_list, dtype=float)
    smax = np.asarray(smax_list, dtype=float)

    def rmse(c_star: float) -> float:
        f = np.clip(cex / max(c_star, 1e-12), 0.0, 1.0)
        return float(np.sqrt(((sg - smax * f) ** 2).mean()))

    lo = max(float(np.percentile(cex, 50)), 1e-4)
    hi = max(float(np.percentile(cex, 99)), lo * 1.5)
    best_c, best_r = lo, rmse(lo)
    for c in np.linspace(lo, hi, 120):
        r = rmse(float(c))
        if r < best_r:
            best_r, best_c = r, float(c)
    return float(best_c)


def sg_calc_en_t(
    df: pd.DataFrame,
    t_anos: float,
    ks: np.ndarray | list[int],
    c0: dict[int, float],
    c_star: float,
    sg_max: dict[int, float],
) -> np.ndarray:
    """Perfil Sg_calc(k) interpolado en t_anos."""
    out = np.full(len(ks), np.nan)
    c_star = max(float(c_star), 1e-12)
    for i, k in enumerate(ks):
        k = int(k)
        sub = df[df["k"].astype(int) == k].sort_values("tiempo_anos")
        if sub.empty:
            continue
        c = float(
            np.interp(
                t_anos,
                sub["tiempo_anos"].to_numpy(dtype=float),
                sub["C_total_mol_kgw"].to_numpy(dtype=float),
                left=float(sub["C_total_mol_kgw"].iloc[0]),
                right=float(sub["C_total_mol_kgw"].iloc[-1]),
            )
        )
        f = min(1.0, max(0.0, (c - c0.get(k, 0.0)) / c_star))
        out[i] = sg_max.get(k, 0.0) * f
    return out

"""Volumen y caudal de CO2 por celda desde Sg Cirrus (DC-023).

Los CSV son de la celda **adyacente** al pozo; el intervalo perforado es k15..k30.
El caudal en cada celda NO es la tasa del pozo: se obtiene de
la saturacion:

    V_gas_m3(t) = Sg(t) * V_poro_m3
    Q_desde_Sg_m3d = d(V_gas_m3)/dt = V_poro_m3 * d(Sg)/dt   [m3/d]

Q_media_inyeccion_m3d: tasa media constante en la etapa de inyeccion (T_FIN_EXTRACCION-T_FIN_INYECCION) (inyeccion pareja).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ccus_columna import datos
from ccus_columna.cli import tasas_disponibles
from ccus_columna.constantes import (
    K_INYECCION,
    K_INYECCION_FONDO,
    MALLAS,
    SEG_POR_ANO,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.geometria import v_bulk_m3
from ccus_columna.mapa_saturacion import fracciones_reactivas
from ccus_columna.mineralogia import phi0

DIAS_POR_ANO = SEG_POR_ANO / 86400.0
DIAS_INYECCION = (T_FIN_INYECCION - T_FIN_EXTRACCION) * DIAS_POR_ANO


def v_poro_m3(malla: str, k: int) -> float:
    return phi0(k) * v_bulk_m3(malla, k)


def _baseline_sg(df: pd.DataFrame) -> float:
    ext = df[df["tiempo_anos"] <= T_FIN_EXTRACCION + 0.01]
    if ext.empty:
        return float(df["Sg"].iloc[0])
    return float(ext["Sg"].median())


def _sg_suavizado(sg: pd.Series, ventana: int = 3) -> pd.Series:
    """Mediana movil para derivada estable con nodos CSV irregulares."""
    return sg.astype(float).rolling(ventana, center=True, min_periods=1).median()


def _q_desde_derivada_sg(
    tiempos: pd.Series, sg: pd.Series, v_poro: float,
) -> pd.Series:
    """Q [m3/d] = V_poro * d(Sg)/dt con dt en anos."""
    dt_a = tiempos.astype(float).diff().clip(lower=1e-12)
    dsg = sg.diff().fillna(0.0)
    return v_poro * dsg / dt_a / DIAS_POR_ANO


def _q_media_inyeccion(df: pd.DataFrame, v_poro: float) -> float:
    """Caudal medio en ventana de inyeccion (Sg parejo en el tiempo)."""
    iny = df[
        (df["tiempo_anos"] >= T_FIN_EXTRACCION - 1e-9)
        & (df["tiempo_anos"] <= T_FIN_INYECCION + 1e-9)
    ]
    if len(iny) < 2:
        return 0.0
    t0, t1 = float(iny["tiempo_anos"].iloc[0]), float(iny["tiempo_anos"].iloc[-1])
    sg0, sg1 = float(iny["Sg"].iloc[0]), float(iny["Sg"].iloc[-1])
    dt_a = max(t1 - t0, 1e-12)
    return v_poro * (sg1 - sg0) / dt_a / DIAS_POR_ANO


def tabla_volumen_celda(malla: str, tasa: int, k: int) -> pd.DataFrame:
    """Serie temporal: volumen y caudal desde Sg (sin repartir tasa del pozo)."""
    df = datos.celda(malla, tasa, k).copy()
    vp = v_poro_m3(malla, k)
    sg0 = _baseline_sg(df)
    sg_s = _sg_suavizado(df["Sg"])

    sg_eff = []
    for _, row in df.iterrows():
        _, sg_e = fracciones_reactivas(float(row["Sg"]), float(row["Sw"]), float(row["So"]))
        sg_eff.append(sg_e)

    q_med = _q_media_inyeccion(df, vp)

    df = df.assign(
        malla=malla,
        tasa_m3d=tasa,
        k=k,
        V_poro_m3=vp,
        V_bulk_m3=v_bulk_m3(malla, k),
        phi=phi0(k),
        sg_eff=sg_eff,
        Sg_suav=sg_s,
        V_gas_m3=sg_s * vp,
        V_gas_reactivo_m3=np.array(sg_eff, dtype=float) * vp,
        Sg_baseline=sg0,
        dV_gas_m3=sg_s * vp - sg0 * vp,
        Q_desde_Sg_m3d=_q_desde_derivada_sg(df["tiempo_anos"], sg_s, vp),
        Q_media_inyeccion_m3d=q_med,
        en_intervalo_perforado=(k >= K_INYECCION) & (k <= K_INYECCION_FONDO),
    )
    df["etapa_inyeccion"] = (
        (df["tiempo_anos"] >= T_FIN_EXTRACCION - 1e-9)
        & (df["tiempo_anos"] <= T_FIN_INYECCION + 1e-9)
    )
    # Alias retrocompatible
    df["Q_gas_m3d"] = df["Q_desde_Sg_m3d"]
    df["dV_gas_m3_paso"] = df["V_gas_m3"].diff().fillna(0.0)
    return df


def tabla_volumen_escenario(malla: str, tasa: int,
                            ks: list[int] | None = None) -> pd.DataFrame:
    if ks is None:
        ks = datos.celdas_disponibles(malla, tasa)
    return pd.concat(
        [tabla_volumen_celda(malla, tasa, k) for k in ks],
        ignore_index=True,
    )


def resumen_escenario(malla: str, tasa: int,
                      tab: pd.DataFrame | None = None) -> pd.DataFrame:
    if tab is None:
        tab = tabla_volumen_escenario(malla, tasa)
    iny = tab[tab["etapa_inyeccion"]].copy()
    filas = []
    for k, g in tab.groupby("k"):
        gi = iny[iny["k"] == k]
        q_med = float(g["Q_media_inyeccion_m3d"].iloc[0])
        filas.append({
            "malla": malla,
            "tasa_m3d": tasa,
            "k": int(k),
            "en_intervalo_perforado": bool(
                K_INYECCION <= int(k) <= K_INYECCION_FONDO
            ),
            "V_poro_m3": float(g["V_poro_m3"].iloc[0]),
            "Sg_baseline": float(g["Sg_baseline"].iloc[0]),
            "Sg_max": float(g["Sg"].max()),
            "V_gas_max_m3": float(g["V_gas_m3"].max()),
            "V_gas_fin_iny_m3": float(gi["V_gas_m3"].iloc[-1]) if len(gi) else 0.0,
            "dV_gas_acum_iny_m3": float(
                gi["V_gas_m3"].iloc[-1] - gi["V_gas_m3"].iloc[0]
            ) if len(gi) >= 2 else 0.0,
            "Q_desde_Sg_media_iny_m3d": float(gi["Q_desde_Sg_m3d"].mean()) if len(gi) else 0.0,
            "Q_desde_Sg_max_iny_m3d": float(gi["Q_desde_Sg_m3d"].max()) if len(gi) else 0.0,
            "Q_media_inyeccion_m3d": q_med,
        })
    return pd.DataFrame(filas).sort_values("k", ascending=False)


def resumen_columna_inyeccion(malla: str, tasa: int, tab: pd.DataFrame) -> dict:
    iny = tab[tab["etapa_inyeccion"] & tab["en_intervalo_perforado"]].copy()
    if iny.empty:
        return {"tasa_nominal_m3d": tasa}
    q_inst = iny.groupby("tiempo_anos")["Q_desde_Sg_m3d"].sum()
    q_med_celdas = iny.groupby("k")["Q_media_inyeccion_m3d"].first().sum()
    return {
        "malla": malla,
        "tasa_nominal_pozo_m3d": tasa,
        "Q_desde_Sg_suma_media_m3d": float(q_inst.mean()),
        "Q_desde_Sg_suma_max_m3d": float(q_inst.max()),
        "Q_media_inyeccion_suma_k15_30_m3d": float(q_med_celdas),
        "ratio_suma_vs_tasa_pozo": float(q_inst.mean() / tasa) if tasa > 0 else float("nan"),
        "V_gas_acum_intervalo_fin_m3": float(iny.groupby("k")["V_gas_m3"].last().sum()),
        "nota": "Q_desde_Sg = V_poro*d(Sg)/dt; suma << tasa pozo (celda adyacente)",
    }


def exportar_todos(out_raiz: Path | None = None) -> list[Path]:
    from ccus_columna.constantes import RAIZ

    out_raiz = Path(out_raiz) if out_raiz else RAIZ / "datos" / "derivados" / "volumen_gas_cirrus"
    paths: list[Path] = []
    resumenes = []
    resumenes_col = []

    for malla in MALLAS:
        for tasa in tasas_disponibles(malla):
            tab = tabla_volumen_escenario(malla, tasa)
            out_dir = out_raiz / malla / f"q{tasa}"
            out_dir.mkdir(parents=True, exist_ok=True)

            p_full = out_dir / "volumen_gas_todas_celdas.csv"
            tab.to_csv(p_full, index=False, float_format="%.6g")
            paths.append(p_full)

            res = resumen_escenario(malla, tasa, tab)
            p_res = out_dir / "resumen_por_celda.csv"
            res.to_csv(p_res, index=False, float_format="%.6g")
            paths.append(p_res)
            resumenes.append(res)

            resumenes_col.append(resumen_columna_inyeccion(malla, tasa, tab))

    p_global = out_raiz / "resumen_columna_perforada.csv"
    pd.DataFrame(resumenes_col).to_csv(p_global, index=False, float_format="%.6g")
    paths.append(p_global)

    p_all = out_raiz / "resumen_todas_celdas.csv"
    pd.concat(resumenes, ignore_index=True).to_csv(p_all, index=False, float_format="%.6g")
    paths.append(p_all)
    return paths

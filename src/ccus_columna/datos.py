"""Loader oficial de los datos Cirrus exportados a CSV.

Todo consumo de datos del proyecto pasa por aqui (DC-002). Los CSV se generan
con scripts/exportar_xlsx_a_csv.py; su estructura esta documentada en
docs/02_DICCIONARIO_DATOS.md.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ccus_columna.constantes import (
    DATOS_CSV,
    K_INYECCION,
    K_INYECCION_FONDO,
    K_SELLO,
    SG_UMBRAL_GAS,
)


def _qdir(malla: str, tasa: int):
    d = DATOS_CSV / malla / f"q{tasa}"
    if not d.is_dir():
        disponibles = sorted(p.name for m in DATOS_CSV.iterdir() if m.is_dir() for p in m.iterdir())
        raise FileNotFoundError(f"No existe {d}. Disponibles: {disponibles}")
    return d


@lru_cache(maxsize=None)
def celda(malla: str, tasa: int, k: int) -> pd.DataFrame:
    """Serie temporal de una celda: tiempo_anos, etapa, P_bar, Sg, So, Sw."""
    return pd.read_csv(_qdir(malla, tasa) / f"celda_k{k:02d}.csv")


@lru_cache(maxsize=None)
def perfil(malla: str, tasa: int, variable: str) -> pd.DataFrame:
    """Matriz t x k de una variable ('Sg', 'So', 'Sw' o 'P')."""
    if variable not in ("Sg", "So", "Sw", "P"):
        raise ValueError(f"variable invalida: {variable}")
    return pd.read_csv(_qdir(malla, tasa) / f"perfil_{variable}.csv")


@lru_cache(maxsize=None)
def resumen_frente() -> pd.DataFrame:
    """Llegadas del frente de gas por malla/tasa/celda (t_primer_gas, Sg_max...)."""
    return pd.read_csv(DATOS_CSV / "resumen_frente_sg.csv")


def celdas_disponibles(malla: str, tasa: int) -> list[int]:
    return sorted(int(p.stem.split("_k")[1]) for p in _qdir(malla, tasa).glob("celda_k*.csv"))


def orden_frente(malla: str, tasa: int, solo_con_gas: bool = True) -> list[int]:
    """Celdas ordenadas por primera llegada del gas (base de DC-004).

    Las celdas que nunca ven gas quedan al final (o se omiten si
    solo_con_gas=True).
    """
    r = resumen_frente()
    r = r[(r["malla"] == malla) & (r["tasa_m3d"] == tasa)].copy()
    con = r.dropna(subset=["t_primer_gas_anos"]).sort_values(["t_primer_gas_anos", "k"])
    orden = [int(x) for x in con["k"]]
    if not solo_con_gas:
        orden += [int(x) for x in r[r["t_primer_gas_anos"].isna()].sort_values("k")["k"]]
    return orden


def rama_al_sello(malla: str, tasa: int) -> list[int]:
    """Celdas desde la zona de inyeccion hacia el sello k=10 (rama v1 de met1).

    Si k=15 ve gas, se usa como origen (consistente con la fase 0D). Si no,
    el origen es la celda de primer gas. Asi, cuando varias celdas reciben
    gas en el mismo nodo (tasas altas), no se arranca en k14 por el desempate
    de orden_frente (t, k).
    """
    orden = orden_frente(malla, tasa)
    if not orden:
        return list(range(K_INYECCION, K_SELLO - 1, -1))
    k0 = K_INYECCION if K_INYECCION in orden else orden[0]
    tope = min(k0, K_SELLO)
    return list(range(k0, tope - 1, -1))


def columna_pozo_sello(
    malla: str,
    tasa: int,
    k_fondo_max: int = K_INYECCION_FONDO,
) -> list[int]:
    """Columna 1D met2: fondo del intervalo perforado → sello (DC-017).

    Incluye celdas con gas hasta min(k_fondo_max, k_max_con_gas), siempre
    al menos desde K_INYECCION, y sube hasta K_SELLO. Orden: k descendente
    (celda PHREEQC 1 = mas profunda = entrada del TRANSPORT).
    """
    orden = orden_frente(malla, tasa, solo_con_gas=True)
    if not orden:
        k_deep = min(k_fondo_max, K_INYECCION)
    else:
        abajo = [int(k) for k in orden if int(k) <= int(k_fondo_max)]
        k_deep = max(abajo) if abajo else K_INYECCION
        k_deep = max(k_deep, K_INYECCION)
        k_deep = min(k_deep, int(k_fondo_max))
    return list(range(int(k_deep), K_SELLO - 1, -1))


def columna_met1_sello(k_fondo: int = K_INYECCION_FONDO) -> list[int]:
    """Columna met1: k30 (fondo perforado) -> k10 (sello), transporte hacia arriba.

    Orden de corrida: celda profunda primero; MIX va de k mayor a k menor
    (aguas abajo -> aguas arriba / sello).
    """
    return list(range(int(k_fondo), K_SELLO - 1, -1))


def interpolar_celda(malla: str, tasa: int, k: int, t_anos: float) -> dict[str, float]:
    """Sg, So, Sw, P_bar interpolados linealmente en t_anos."""
    df = celda(malla, tasa, k)
    t = df["tiempo_anos"].to_numpy()
    return {v: float(np.interp(t_anos, t, df[v].to_numpy())) for v in ("Sg", "So", "Sw", "P_bar")}


def hay_gas(malla: str, tasa: int, k: int, t_anos: float) -> bool:
    """Condicion de activacion de transferencia met1: Sg(t) > umbral."""
    return interpolar_celda(malla, tasa, k, t_anos)["Sg"] > SG_UMBRAL_GAS

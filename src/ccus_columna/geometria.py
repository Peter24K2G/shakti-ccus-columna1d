"""Geometria de celda Cirrus (DC-014 dz, DC-015 dx-dy).

malla original: 50 m x 50 m; malla refinada: 20 m x 20 m.
Reservorio dz=20 m; sello k=10 dz=17 m.

PHREEQC no integra decenas de miles de m3 de agua (alcalinidad no converge). El reactor
usa la receta 1 L de la celda original de reservorio (50x50x20 m); las demas
celdas conservan la *razon* de volumenes (sello 17/20, refinada (20/50)^2).
v_bulk_m3() sigue siendo el volumen geometrico Cirrus; factor_moles_campo()
pasa moles del .pqi a moles de la celda de campo.
"""

from __future__ import annotations

from ccus_columna.constantes import (
    DX_MALLA_ORIGINAL_M,
    DX_MALLA_REFINADA_M,
    DY_MALLA_ORIGINAL_M,
    DY_MALLA_REFINADA_M,
    DZ_M,
    DZ_SELLO_M,
    K_SELLO,
    M3_A_L,
    V_BULK_RECETA_L,
)


def dx_m(malla: str | None) -> float:
    if malla == "malla_refinada":
        return DX_MALLA_REFINADA_M
    return DX_MALLA_ORIGINAL_M


def dy_m(malla: str | None) -> float:
    if malla == "malla_refinada":
        return DY_MALLA_REFINADA_M
    return DY_MALLA_ORIGINAL_M


def dz_m(k: int | None) -> float:
    """Espesor de la celda Cirrus k."""
    if k is not None and int(k) == int(K_SELLO):
        return DZ_SELLO_M
    return DZ_M


def v_bulk_ref_m3() -> float:
    """Celda de referencia = reservorio malla original (50 x 50 x 20 m)."""
    return DX_MALLA_ORIGINAL_M * DY_MALLA_ORIGINAL_M * DZ_M


def v_bulk_m3(malla: str | None, k: int | None) -> float:
    """Volumen bulk geometrico dx*dy*dz [m3]. Receta 1 L si malla is None."""
    if malla is None:
        return V_BULK_RECETA_L / M3_A_L
    return dx_m(malla) * dy_m(malla) * dz_m(k)


def v_bulk_l(malla: str | None, k: int | None) -> float:
    """Volumen del reactor PHREEQC [L]: 1 L = celda original de reservorio.

    Poner dx*dy*dz en litros (~5e7 L) rompe la convergencia de alcalinidad.
    """
    if malla is None:
        return V_BULK_RECETA_L
    return V_BULK_RECETA_L * v_bulk_m3(malla, k) / v_bulk_ref_m3()


def escala_volumen(malla: str | None, k: int | None) -> float:
    """V_reactor / V_receta (1 L). Original k!=10 -> 1.0."""
    return v_bulk_l(malla, k) / V_BULK_RECETA_L


def factor_moles_campo(malla: str | None, k: int | None) -> float:
    """n_campo = n_phreeqc * factor. 1 L de receta -> V_bulk de la celda de campo."""
    v_pq_m3 = v_bulk_l(malla, k) / M3_A_L
    if v_pq_m3 <= 0:
        return 1.0
    return v_bulk_m3(malla, k) / v_pq_m3

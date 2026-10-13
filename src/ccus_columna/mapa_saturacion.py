"""Mapa saturacion Cirrus -> estado del reactor (D-021 + D-018 / DC-003).

Port de referencia_lapaz_0d/src_lapaz/saturacion.py (solo el nucleo S->estado).
"""

from __future__ import annotations

from ccus_columna.constantes import BAR_A_ATM
from ccus_columna.mineralogia import m_water_ref, v_poro


def fracciones_reactivas(sg: float, sw: float, so: float = 0.0) -> tuple[float, float]:
    """(sw_eff, sg_eff) en poro libre de crudo inerte (D-021)."""
    so = max(0.0, min(1.0, so))
    poro_reactivo = max(0.0, 1.0 - so)
    sg_sw = sg + sw
    if sg_sw <= 1e-12:
        return poro_reactivo, 0.0
    return poro_reactivo * sw / sg_sw, poro_reactivo * sg / sg_sw


def volumenes_poro(sg: float, sw: float, so: float = 0.0,
                   k: int | None = None,
                   malla: str | None = None) -> tuple[float, float]:
    """(m_water_kg, v_gas_L) con V_bulk de la celda (DC-015)."""
    sw_eff, sg_eff = fracciones_reactivas(sg, sw, so)
    return m_water_ref(k, malla) * sw_eff, v_poro(k, malla) * sg_eff


def p_co2_fugacidad_proporcional(
    sg: float,
    sg_max: float,
    p_ref: float = 80.0,
) -> float:
    """P_CO2 = P_ref * min(1, Sg/Sg_max). P_ref en atm (DC-003: P_celda*BAR_A_ATM)."""
    if sg <= 1e-8 or sg_max <= 1e-8:
        return 0.0
    return p_ref * min(1.0, sg / sg_max)


def estado_poro(
    sg: float,
    sw: float,
    so: float,
    p_bar: float,
    sg_max: float,
    *,
    usar_p_real: bool = True,
    k: int | None = None,
    malla: str | None = None,
) -> dict[str, float]:
    """Estado completo del poro reactivo en un instante."""
    m_water, v_gas = volumenes_poro(sg, sw, so, k=k, malla=malla)
    p_ref = p_bar * BAR_A_ATM if usar_p_real else 80.0
    return {
        "m_water": m_water,
        "v_gas": v_gas,
        "p_co2_atm": p_co2_fugacidad_proporcional(sg, sg_max, p_ref),
        "sg": sg,
        "sw": sw,
        "so": so,
        "p_bar": p_bar,
        "p_ref_atm": p_ref,
        "sg_max": sg_max,
    }

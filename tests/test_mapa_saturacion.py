"""Tests del mapa de saturacion compartido."""

from ccus_columna.mapa_saturacion import (
    estado_poro,
    fracciones_reactivas,
    p_co2_fugacidad_proporcional,
    volumenes_poro,
)


def test_fracciones_sin_crudo():
    sw_e, sg_e = fracciones_reactivas(0.4, 0.6, 0.0)
    assert abs(sw_e - 0.6) < 1e-12
    assert abs(sg_e - 0.4) < 1e-12


def test_volumenes_sw1():
    from ccus_columna.constantes import M_WATER_REF_KG
    m, v = volumenes_poro(0.0, 1.0, 0.0)
    assert abs(m - M_WATER_REF_KG) < 1e-12
    assert abs(v - 0.0) < 1e-12


def test_volumenes_con_gas():
    from ccus_columna.constantes import M_WATER_REF_KG, V_PORO_L
    m, v = volumenes_poro(0.5, 0.5, 0.0)
    assert abs(m - 0.5 * M_WATER_REF_KG) < 1e-12
    assert abs(v - 0.5 * V_PORO_L) < 1e-12


def test_volumenes_sello_usa_phi_mayor():
    m15, _ = volumenes_poro(0.0, 1.0, 0.0, k=15)
    m10, _ = volumenes_poro(0.0, 1.0, 0.0, k=10)
    assert m10 > m15


def test_fugacidad():
    assert p_co2_fugacidad_proporcional(0.0, 0.5) == 0.0
    assert abs(p_co2_fugacidad_proporcional(0.25, 0.5, 80.0) - 40.0) < 1e-9
    assert abs(p_co2_fugacidad_proporcional(0.5, 0.5, 80.0) - 80.0) < 1e-9


def test_estado_poro_p_real():
    e = estado_poro(0.4, 0.5, 0.1, p_bar=200.0, sg_max=0.5, usar_p_real=True)
    assert e["m_water"] > 0 and e["v_gas"] > 0
    assert e["p_co2_atm"] > 0

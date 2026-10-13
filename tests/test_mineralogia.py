"""Tests de mineralogia del sello (XRD Esmeraldas, DC-011)."""

import pytest

from ccus_columna.constantes import K_SELLO, PHI, PHI_SELLO
from ccus_columna.mineralogia import (
    MOLES_SELLO,
    inventario_eq,
    inventario_kin,
    m_water_ref,
    phi0,
    wt_promedio_xrd,
)
from ccus_columna.quimica import bloque_kinetics_inicial, equilibrium_rocas
from ccus_columna.constantes import (  # noqa: E402  parametros del proyecto
    DX_MALLA_ORIGINAL_M, DX_MALLA_REFINADA_M, DZ_M, DZ_SELLO_M, PHI,
    T_FIN_EXTRACCION as _T_EXT, T_FIN_INYECCION as _T_INY,
    TASA_REF, TASAS_M3D, TASAS_POR_MALLA,
)
from ccus_columna.mineralogia import INVENTARIO_EQ_RESERVORIO as _INV_EQ  # noqa: E402
from ccus_columna.parametros import ES_EJEMPLO, P  # noqa: E402

CAL0 = _INV_EQ["Calcite"]
PIR0 = _INV_EQ["Pyrite"]
PR = P.get("pruebas", {})  # rangos esperados (dependen de los datos reales)
TASA_SOLO_ORIGINAL = next(q for q in TASAS_POR_MALLA["malla_original"]
                          if q not in TASAS_POR_MALLA["malla_refinada"])
solo_reales = pytest.mark.skipif(ES_EJEMPLO, reason="requiere parametros reales (datos/parametros.yaml)")



def test_phi_sello_distinta():
    assert abs(phi0(15) - PHI) < 1e-15
    assert abs(phi0(K_SELLO) - PHI_SELLO) < 1e-15
    assert m_water_ref(K_SELLO) > m_water_ref(15)


@solo_reales
def test_sello_cuarzo_dominante():
    wt = wt_promedio_xrd()
    assert wt["Quartz"] > 0.75
    assert MOLES_SELLO["Quartz"] > 20.0
    assert MOLES_SELLO["Calcite"] < CAL0
    assert MOLES_SELLO["Kaolinite"] > 0.3
    assert MOLES_SELLO["K-mica"] > 0.2


def test_inventario_sello_vs_reservorio():
    eq15 = inventario_eq(15)
    eq10 = inventario_eq(10)
    assert abs(eq15["Calcite"] - CAL0) < 1e-12
    assert eq10["Calcite"] == MOLES_SELLO["Calcite"]
    assert eq10["Dolomite"] == MOLES_SELLO["Dolomite"]
    assert eq10["Pyrite"] == 0.0
    kin10 = inventario_kin(10)
    assert "K-mica" in kin10 and "Quartz" in kin10
    assert "Calcite" not in kin10
    assert "Siderite" not in kin10


def test_volumen_celda_dc015():
    from ccus_columna.geometria import (
        escala_volumen,
        factor_moles_campo,
        v_bulk_l,
        v_bulk_m3,
        v_bulk_ref_m3,
    )

    assert abs(v_bulk_m3("malla_original", 15) - v_bulk_ref_m3()) < 1e-9
    assert abs(v_bulk_l("malla_original", 15) - 1.0) < 1e-12
    assert abs(v_bulk_l("malla_original", 10) - DZ_SELLO_M / DZ_M) < 1e-12
    assert abs(v_bulk_l("malla_refinada", 15) - (DX_MALLA_REFINADA_M / DX_MALLA_ORIGINAL_M) ** 2) < 1e-12
    s = escala_volumen("malla_original", 15)
    assert abs(inventario_eq(15, "malla_original")["Calcite"] - CAL0 * s) < 1e-6
    m15 = m_water_ref(15, "malla_original")
    m10 = m_water_ref(10, "malla_original")
    assert m10 < m15  # sello: dz y phi propios
    m15r = m_water_ref(15, "malla_refinada")
    assert abs(m15r / m15 - (DX_MALLA_REFINADA_M / DX_MALLA_ORIGINAL_M) ** 2) < 1e-12
    # 1 L de receta representa dx*dy*dz de k15 original
    assert abs(factor_moles_campo("malla_original", 15) - DX_MALLA_ORIGINAL_M ** 2 * DZ_M * 1000.0) < 1.0


def test_bloques_pqi_sello():
    eq = equilibrium_rocas(1, 10)
    assert "Dolomite" in eq
    assert "CAL0" not in eq
    kin = bloque_kinetics_inicial(1, 10)
    assert "K-mica" in kin
    assert "Siderite" not in kin
    kin15 = bloque_kinetics_inicial(1, 15)
    assert "Siderite" in kin15
    assert "K-mica" not in kin15

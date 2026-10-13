"""Tests estimadores de permeabilidad de roca (DC-024 / DC-026)."""

import pytest

from ccus_columna.permeabilidad import (
    SIW_DEFAULT,
    delta_k_pct,
    krg_wyllie,
    krw_corey,
    krw_wyllie,
    ratio_kozeny_carman,
)
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



def test_kozeny_carman_neutral():
    assert abs(ratio_kozeny_carman(PHI, PHI) - 1.0) < 1e-9
    assert abs(ratio_kozeny_carman(PHI, PHI, completo=True) - 1.0) < 1e-9


def test_kozeny_carman_disolucion_aumenta_k():
    r = ratio_kozeny_carman(PHI + 0.004, PHI, completo=False)
    r_full = ratio_kozeny_carman(PHI + 0.004, PHI, completo=True)
    assert r > 1.0
    assert r_full > r


def test_kozeny_carman_precipitacion_baja_k():
    r = ratio_kozeny_carman(PHI - 0.006, PHI, completo=False)
    assert r < 1.0


def test_krw_corey_sw0():
    assert krw_corey(SIW_DEFAULT, SIW_DEFAULT) == 0.0


def test_krw_corey_sw1():
    assert abs(krw_corey(1.0, SIW_DEFAULT) - 1.0) < 1e-9


def test_krw_wyllie_sw0():
    assert krw_wyllie(SIW_DEFAULT, SIW_DEFAULT) == 0.0


def test_krg_wyllie_positivo():
    assert krg_wyllie(0.3, SIW_DEFAULT) > 0.0


def test_delta_k_pct():
    assert abs(delta_k_pct(1.1) - 10.0) < 1e-9

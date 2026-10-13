"""Tests de validacion de malla/tasa y resolucion de corridas."""

import pytest

from ccus_columna.cli import resolver_dir_corrida, validar_malla_tasa
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



def test_malla_invalida():
    with pytest.raises(SystemExit):
        validar_malla_tasa("malla_inventada", TASA_REF)


from ccus_columna.constantes import DATOS_CSV as _DATOS_CSV  # noqa: E402
_mixto = pytest.mark.skipif(
    ES_EJEMPLO and _DATOS_CSV.exists(),
    reason="datos/csv reales con parametros de ejemplo")


@_mixto
def test_tasa_100k_no_existe_en_refinada():
    with pytest.raises(SystemExit):
        validar_malla_tasa("malla_refinada", TASA_SOLO_ORIGINAL)


@_mixto
def test_malla_tasa_ok():
    validar_malla_tasa("malla_original", TASA_REF)
    validar_malla_tasa("malla_refinada", TASA_REF)


def test_placeholder_id_rechazado(tmp_path):
    with pytest.raises(SystemExit) as e:
        resolver_dir_corrida("corridas/<id>", tmp_path)
    msg = str(e.value)
    assert "placeholder" in msg.lower() or "REAL" in msg

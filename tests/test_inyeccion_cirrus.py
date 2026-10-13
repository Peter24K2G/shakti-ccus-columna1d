"""Tests volumen gas desde Sg Cirrus."""

import pytest

from ccus_columna.inyeccion_cirrus import (
    resumen_columna_inyeccion,
    tabla_volumen_celda,
    tabla_volumen_escenario,
    v_poro_m3,
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
from ccus_columna.constantes import DATOS_CSV as _DATOS_CSV  # noqa: E402
requiere_datos = pytest.mark.skipif(
    ES_EJEMPLO or not _DATOS_CSV.exists(),
    reason="requiere datos Cirrus (datos/csv) y parametros reales")
solo_reales = pytest.mark.skipif(ES_EJEMPLO, reason="requiere parametros reales (datos/parametros.yaml)")



@requiere_datos
def test_v_poro_original_k15():
    vp = v_poro_m3("malla_original", 15)
    lo, hi = PR["v_poro_k15_m3"]
    assert lo < vp < hi


@requiere_datos
def test_q_desde_sg_formula():
    tab = tabla_volumen_celda("malla_original", TASA_REF, 15)
    iny = tab[tab["etapa_inyeccion"]]
    assert iny["Q_desde_Sg_m3d"].max() > 0
    # Caudal desde Sg << tasa pozo (celda adyacente)
    assert iny["Q_desde_Sg_m3d"].max() < TASA_REF


@requiere_datos
def test_q_media_inyeccion_constante():
    tab = tabla_volumen_celda("malla_original", TASA_REF, 15)
    q = tab["Q_media_inyeccion_m3d"].iloc[0]
    assert q > 0
    assert q < 100  # orden ~1 m3/d por celda


@requiere_datos
def test_resumen_columna_menor_que_tasa_pozo():
    tab = tabla_volumen_escenario("malla_original", TASA_REF)
    rc = resumen_columna_inyeccion("malla_original", TASA_REF, tab)
    assert rc["Q_desde_Sg_suma_media_m3d"] < rc["tasa_nominal_pozo_m3d"]
    assert rc["ratio_suma_vs_tasa_pozo"] < 0.01

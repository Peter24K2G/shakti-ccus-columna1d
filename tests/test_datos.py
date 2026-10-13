"""Tests del loader compartido de datos (requieren datos/csv/ generado)."""

import pandas as pd
import pytest

from ccus_columna import datos
from ccus_columna.constantes import DATOS_CSV
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


pytestmark = requiere_datos


def test_celda_columnas_y_rangos():
    df = datos.celda("malla_original", TASA_REF, 15)
    assert list(df.columns) == ["tiempo_anos", "etapa", "P_bar", "Sg", "So", "Sw"]
    assert len(df) == PR["n_nodos_tiempo"]
    assert df["tiempo_anos"].iloc[0] == 0.0 and df["tiempo_anos"].iloc[-1] == 70.0
    for v in ("Sg", "So", "Sw"):
        assert df[v].between(0, 1).all()
    # cierre de saturaciones
    assert ((df["Sg"] + df["So"] + df["Sw"]) - 1.0).abs().max() < 1e-3


def test_perfil_forma():
    p = datos.perfil("malla_original", TASA_REF, "Sg")
    assert p.shape == (PR["n_nodos_tiempo"], 27)  # tiempo + 26 celdas
    assert "k10" in p.columns and "k35" in p.columns


def test_refinada_sin_100k():
    with pytest.raises(FileNotFoundError):
        datos.celda("malla_refinada", TASA_SOLO_ORIGINAL, 15)


def test_orden_frente_50k():
    orden = datos.orden_frente("malla_original", TASA_REF)
    assert orden[0] == 15                    # el gas nace en k15
    assert 10 in orden                       # el sello llega a ver gas
    assert 30 not in orden                   # las profundas no


def test_rama_al_sello():
    assert datos.rama_al_sello("malla_original", TASA_REF) == [15, 14, 13, 12, 11, 10]


def test_columna_pozo_sello_incluye_perforacion():
    col = datos.columna_pozo_sello("malla_original", TASA_REF)
    assert col[0] == 25 and col[-1] == 10
    assert 15 in col and 20 in col


def test_columna_met1_sello_siempre_k30():
    col = datos.columna_met1_sello()
    assert col[0] == 30 and col[-1] == 10
    assert len(col) == 21


def test_rama_al_sello_500k_incluye_k15():
    """A la tasa maxima varias celdas ven gas al iniciar la inyeccion; la rama debe partir de k15, no k14."""
    rama = datos.rama_al_sello("malla_original", TASAS_M3D[-1])
    assert rama[0] == 15
    assert rama[-1] == 10
    assert 15 in rama and 14 in rama


def test_interpolacion_y_hay_gas():
    s = datos.interpolar_celda("malla_original", TASA_REF, 15, 30.0)
    assert s["Sg"] > 0.1 and s["P_bar"] > 200
    assert not datos.hay_gas("malla_original", TASA_REF, 10, 15.0)  # frente no llego
    assert datos.hay_gas("malla_original", TASA_REF, 10, 30.0)


def test_etapas_en_datos():
    df = datos.celda("malla_original", TASAS_M3D[2], 20)
    assert set(df["etapa"].unique()) >= {"Extracción", "Inyección"}

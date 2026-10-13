"""Tests del post-proceso compartido."""

import pytest

from pathlib import Path

import pandas as pd

from ccus_columna.postproceso import (
    PERFIL_COLS,
    anotar_delta_calcita,
    calcular_delta_phi,
    perfil_desde_met1,
    perfil_desde_met2,
)
from ccus_columna.rutas import dir_figuras_comparacion, dir_figuras_corrida, parse_id_corrida
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


RAIZ = Path(__file__).resolve().parents[1]


def test_delta_phi_disolucion_calcita():
    row = pd.Series({"calcite": (CAL0 - 0.001), "pyrite": PIR0})
    dphi = calcular_delta_phi(row, incluir_cinetica=False)
    expected = -(((CAL0 - 0.001) * 0.0369 + PIR0 * 0.0239)
                 - (CAL0 * 0.0369 + PIR0 * 0.0239))
    assert abs(dphi - expected) < 1e-12
    assert dphi > 0  # disolucion aumenta porosidad


def test_delta_phi_incluye_cinetica():
    row_intacto = pd.Series({
        "calcite": CAL0, "pyrite": PIR0,
        "k_quartz": 0.0, "dk_quartz": 0.0,
    })
    d_eq = calcular_delta_phi(row_intacto, k=15, incluir_cinetica=False)
    d_kin = calcular_delta_phi(row_intacto, k=15, incluir_cinetica=True)
    assert abs(d_kin - d_eq) < 1e-12  # k_*=0 no implica disolucion total

    row_dis = pd.Series({
        "calcite": CAL0, "pyrite": PIR0,
        "k_quartz": 0.0, "dk_quartz": -0.5,
    })
    d_dis = calcular_delta_phi(row_dis, k=15, incluir_cinetica=True)
    assert d_dis > d_eq


def test_delta_phi_desde_columna_perfil():
    row = pd.Series({"calcita_mol": CAL0})
    assert abs(calcular_delta_phi(row)) < 1e-12


def test_delta_calcita_vs_inventario():
    df = pd.DataFrame({"k": [15, 10], "calcita_mol": [CAL0, 0.20]})
    out = anotar_delta_calcita(df)
    assert abs(out.loc[0, "delta_calcita_mol"]) < 1e-12
    assert out.loc[1, "delta_calcita_mol"] < 0


def test_parse_id_y_carpetas_figuras():
    assert parse_id_corrida(f"malla_original_q{TASA_REF}_completo_anual") == (
        "malla_original", TASA_REF, "completo_anual",
    )
    d = dir_figuras_corrida("met1", Path(f"malla_original_q{TASA_REF}_completo_anual"))
    assert d.as_posix().endswith(f"figuras/met1/malla_original/q{TASA_REF}/completo_anual")
    c = dir_figuras_comparacion("malla_original", TASA_REF)
    assert c.as_posix().endswith(f"figuras/comparacion/malla_original/q{TASA_REF}")


def test_perfil_met1_smoke():
    run = RAIZ / "met1_celdas_encadenadas" / "corridas" / f"malla_original_q{TASA_REF}_smoke_k15"
    if not run.exists():
        return
    df = perfil_desde_met1(run)
    assert not df.empty
    assert list(df.columns) == list(PERFIL_COLS)
    assert df["k"].iloc[0] == 15


def test_perfil_met2_trazador():
    run = RAIZ / "met2_transporte_phreeqc" / "corridas" / f"malla_original_q{TASA_REF}_trazador"
    if not run.exists():
        return
    df = perfil_desde_met2(run)
    assert not df.empty
    assert "pH" in df.columns
    assert df["k"].min() == 10
    assert df["k"].max() <= 15
    assert set(df["k"].astype(int).unique()).issubset(set(range(10, 16)))

"""Tests smoke met1 (generacion .pqi; PHREEQC opcional)."""

import sys
from pathlib import Path

import pytest

MET1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MET1))
sys.path.insert(0, str(MET1.parent / "src"))

from config import CorridaConfig  # noqa: E402
import estado as est  # noqa: E402
import transferencia as tr  # noqa: E402
import motor  # noqa: E402
from ccus_columna.runner import PHREEQC_EXE  # noqa: E402
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



def test_config_id_y_guardado(tmp_path):
    cfg = CorridaConfig(malla="malla_original", tasa_m3d=TASA_REF)
    assert cfg.id_corrida() == f"malla_original_q{TASA_REF}_tau_residencia"
    p = cfg.guardar(tmp_path)
    assert p.exists()


@requiere_datos
def test_orden_de_flujo_rama_sello():
    orden = tr.orden_de_flujo("malla_original", TASA_REF, dominio="rama_sello")
    assert orden[0] == 15
    assert orden[-1] == 10
    assert orden == sorted(orden, reverse=True)


def test_orden_de_flujo_columna_pozo():
    orden = tr.orden_de_flujo("malla_original", TASA_REF, dominio="columna_pozo")
    assert orden[0] == 30
    assert orden[-1] == 10
    assert 15 in orden and 25 in orden


@requiere_datos
def test_tau_residencia_positivo_hacia_el_sello():
    tau = tr.tau_residencia_anos("malla_original", TASA_REF, 15, 14)
    assert tau is not None and tau > 0


@requiere_datos
def test_fraccion_cero_sin_gas():
    f = tr.fraccion_transferida("tau_residencia", 0.5, "malla_original", TASA_REF,
                                15, 14, t_anos=5.0)
    assert f == 0.0


@requiere_datos
def test_fraccion_activa_con_gas():
    f = tr.fraccion_transferida("tau_residencia", 0.5, "malla_original", TASA_REF,
                                15, 14, t_anos=30.0)
    assert 0.0 < f <= 1.0


@requiere_datos
def test_activacion_dp_sin_gas():
    # k25->k24 antes de gas en destino pero con gradiente P
    assert tr.transferencia_activa(
        "malla_original", TASA_REF, 25, 24, 12.0, "sg_o_dp",
    )
    assert not tr.transferencia_activa(
        "malla_original", TASA_REF, 25, 24, 12.0, "sg",
    )


@requiere_datos
def test_mapa_saturacion_celda_inyeccion():
    cfg = CorridaConfig()
    m = motor.mapa_saturacion_celda(cfg, 15, 30.0)
    from ccus_columna.mineralogia import m_water_ref, v_poro
    assert 0 < m["m_water"] < m_water_ref(15, cfg.malla)
    assert 0 < m["v_gas"] < v_poro(15, cfg.malla)
    assert m["p_co2_atm"] > 0


@requiere_datos
def test_pasos_de_tiempo():
    cfg = CorridaConfig()
    pasos = motor.pasos_de_tiempo(cfg)
    assert len(pasos) == PR["n_nodos_tiempo"] - 1
    assert pasos[0][0] == 0.0 and abs(pasos[-1][1] - 70.0) < 1e-9


@requiere_datos
def test_pasos_max_y_tmax():
    cfg = CorridaConfig(max_pasos=3, t_max_anos=20.0)
    pasos = motor.pasos_de_tiempo(cfg)
    assert len(pasos) == 3
    assert pasos[-1][1] <= 20.0 + 1e-9


@requiere_datos
def test_pqi_paso_celda_contiene_bloques_clave(tmp_path):
    cfg = CorridaConfig(celdas=[15], transferir=False, etiqueta="test")
    run_dir = tmp_path
    (run_dir / "estado").mkdir()
    (run_dir / "hitos").mkdir()
    texto = motor.pqi_paso_celda(
        cfg, 15, 0, 0.0, 1.0, run_dir, None, None, 0.0,
    )
    assert "PHASES" in texto
    assert "KINETICS" in texto
    assert "DUMP" in texto
    assert "SOLUTION_MODIFY" in texto
    assert "-mass_water" in texto


@requiere_datos
def test_bloque_inicial_tiene_rocas():
    t = est.bloque_inicial(k=15)
    assert "Calcite" in t and "KINETICS" in t and "Agua_Nativa" in t
    t10 = est.bloque_inicial(k=10)
    assert "K-mica" in t10 and "Dolomite" in t10
    assert "Siderite" not in t10


@requiere_datos
@pytest.mark.skipif(not PHREEQC_EXE.exists(), reason="phreeqc.exe no instalado")
def test_smoke_phreeqc_k15():
    cfg = motor.config_smoke()
    cfg.max_pasos = 2
    run_dir = motor.correr(cfg)
    assert (run_dir / "OK.txt").exists()
    assert (run_dir / "hitos_k15.csv").exists() or list((run_dir / "hitos").glob("k15_*.txt"))

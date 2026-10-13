"""Tests smoke met2 (generacion .pqi; PHREEQC opcional)."""

import sys
from pathlib import Path

import pytest

MET2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MET2))
sys.path.insert(0, str(MET2.parent / "src"))

import calibracion as cal  # noqa: E402
import generador as gen  # noqa: E402
from ccus_columna.constantes import (  # noqa: E402
    T_FIN,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
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



@requiere_datos
def test_geometria_sello_y_difusion():
    rama = [15, 14, 13, 12, 11, 10]
    assert cal.longitudes_m(rama) == [DZ_M] * 5 + [DZ_SELLO_M]
    assert abs(cal.longitud_hops_m(rama) - (4 * DZ_M + DZ_SELLO_M)) < 1e-9
    assert abs(cal.dz_shift_m(rama) - 19.4) < 1e-9
    col = cal.celdas_transport("malla_original", TASA_REF)
    assert col[0] == 25 and col[-1] == 10 and len(col) == 16
    texto = gen.pqi_reactivo("malla_original", TASA_REF)
    lengths = " ".join("20" if k != 10 else "17" for k in col)
    assert f"-lengths               {lengths}" in texto
    assert "-diffusion_coefficient" in texto
    assert "1e-09" in texto or "1e-9" in texto
    from ccus_columna.mineralogia import inventario_eq, m_water_ref
    assert f"{m_water_ref(15, 'malla_original'):.6f}" in texto
    assert f"{inventario_eq(15, 'malla_original')['Calcite']:.6g}" in texto


@requiere_datos
def test_velocidad_frente_positiva():
    v = cal.velocidad_frente_m_por_ano("malla_original", TASA_REF)
    assert len(v) >= 4
    assert all(x > 0 for x in v.values())
    lo, hi = PR["v_media_m_por_ano"]
    assert lo < cal.v_media_m_por_ano("malla_original", TASA_REF) < hi


@requiere_datos
def test_velocidad_cero_sin_inyeccion():
    """Extraccion y observacion: P no empuja al sello y no hay inyeccion."""
    p_ext = cal.parametros_transport("malla_original", TASA_REF, 0.0, T_FIN_EXTRACCION)
    assert p_ext["v_m_por_ano"] == 0.0
    assert p_ext["flow_direction"] == "diffusion_only"
    p_obs = cal.parametros_transport("malla_original", TASA_REF, T_FIN_INYECCION, T_FIN)
    assert p_obs["v_m_por_ano"] == 0.0
    assert p_obs["flow_direction"] == "diffusion_only"


@requiere_datos
def test_velocidad_inyeccion_desde_sg():
    """Durante el avance del frente, v sale de Sg (concentracion) y P."""
    t_sello = cal.t_llegada_sello_anos("malla_original", TASA_REF)
    p = cal.parametros_transport("malla_original", TASA_REF, T_FIN_EXTRACCION, t_sello)
    assert p["flow_direction"] == "forward"
    lo, hi = PR["v_frente_inyeccion_m_por_ano"]
    assert lo < p["v_m_por_ano"] < hi
    assert p["shifts"] >= 1


@requiere_datos
def test_parametros_transport_ventana_inyeccion():
    p = cal.parametros_transport("malla_original", TASA_REF, _T_EXT, _T_INY)
    assert p["shifts"] >= 1
    assert p["time_step_s"] > 0
    assert abs(p["shifts"] * p["dt_shift_anos"] - (_T_INY - _T_EXT)) < p["dt_shift_anos"]


@requiere_datos
def test_ventanas_nodos_csv_como_excel():
    """Un paso por intervalo de P/Sg; sin adveccion en extraccion/observacion."""
    t = cal.nodos_tiempo_csv("malla_original", TASA_REF)
    pasos = cal.ventanas_nodos_csv("malla_original", TASA_REF)
    assert len(t) == PR["n_nodos_tiempo"]
    assert len(pasos) == len(t) - 1
    assert pasos[0]["nombre"] == "extraccion"
    assert pasos[-1]["nombre"] == "observacion"
    assert all(p["flow_direction"] == "diffusion_only"
               for p in pasos if p["nombre"] == "extraccion")
    n_fwd = sum(p["shifts"] for p in pasos if p["flow_direction"] == "forward")
    assert 3 <= n_fwd <= 40  # rama al sello + tramos perforados


@requiere_datos
def test_ventanas_operativas_con_observacion_partida():
    vents = cal.ventanas_operativas("malla_original", TASA_REF)
    nombres = [v[2] for v in vents]
    assert nombres[0] == "extraccion"
    assert any(n.startswith("frente_") for n in nombres)
    assert "inyeccion" in nombres
    assert all(n == "observacion" for n in nombres if n == "observacion")
    assert nombres[-1] == "observacion"
    assert vents[0][0] == 0.0 and vents[-1][1] == T_FIN
    # hops k15→k10: 5 tramos
    hops = cal.hops_hacia_sello("malla_original", TASA_REF)
    assert len(hops) == 5
    assert hops[0][:2] == (15, 14) and hops[-1][:2] == (11, 10)


@requiere_datos
def test_a_L_calibrada_frente():
    a_L = cal.a_L_calibrada_m("malla_original", TASA_REF)
    assert 1.0 < a_L < 15.0
    disp = cal.dispersividades_m(
        cal.celdas_transport("malla_original", TASA_REF),
        "malla_original", TASA_REF, calibrar=True,
    )
    assert all(abs(d - a_L) < 1e-9 for d in disp)


@requiere_datos
def test_si_co2_activo_solo_con_gas():
    rama = [15, 14, 13, 12, 11, 10]
    antes = gen.si_co2_por_celda("malla_original", TASA_REF, rama, 5.0)
    assert all(v is None for v in antes.values())
    durante = gen.si_co2_por_celda("malla_original", TASA_REF, rama, 30.0)
    assert durante[15] is not None and durante[15] > 0
    despues_sello = gen.si_co2_por_celda("malla_original", TASA_REF, [10], 15.0)
    assert despues_sello[10] is None


@requiere_datos
def test_pqi_reactivo_corto_solo_inyeccion():
    vents = [(T_FIN_EXTRACCION, T_FIN_EXTRACCION + 2.0, "inyeccion")]
    texto = gen.pqi_reactivo("malla_original", TASA_REF, ventanas=vents)
    assert texto.count("TRANSPORT") == 1
    assert "diffusion_only" not in texto
    assert "-cvode" in texto
    assert "Ventana 0 extraccion" not in texto
    eq1 = texto.split("EQUILIBRIUM_PHASES 1")[1].split("EQUILIBRIUM_PHASES 2")[0]
    assert "-10.000000" not in eq1


def test_plantilla_semilla_existe():
    seed = MET2 / "plantillas" / "transporte_base.pqi"
    texto = seed.read_text(encoding="utf-8")
    assert "TRANSPORT" in texto and "-boundary_conditions" in texto


@requiere_datos
def test_pqi_trazador_tiene_transport():
    texto = gen.pqi_trazador("malla_original", TASA_REF)
    assert "TRANSPORT" in texto
    assert "SOLUTION 1" in texto
    assert "KINETICS" not in texto
    assert "diffusion_only" in texto


@requiere_datos
def test_pqi_reactivo_sin_co2_fuera_de_inyeccion():
    texto = gen.pqi_reactivo("malla_original", TASA_REF)
    assert "TRANSPORT" in texto
    assert "-cvode" in texto
    assert "-runge_kutta" not in texto
    assert "Calcite" in texto
    assert "K-mica" in texto
    assert "Dolomite" in texto
    n_tr = texto.count("TRANSPORT")
    assert 3 <= n_tr <= 25  # hops del frente (DC-029) + observacion partida + rampas
    assert "diffusion_only" in texto
    assert "EQUILIBRIUM_PHASES_MODIFY" in texto
    assert "rampa CVODE" in texto
    assert f"CO2(g)            {gen.SI_CO2_SUAVE:.6f}" in texto
    assert "observacion" in texto
    assert "inyeccion" in texto
    assert "Ventana 0 extraccion" in texto
    i_ext = texto.find("Ventana 0 extraccion")
    i_kin = texto.find("KINETICS 1")
    assert 0 <= i_ext < i_kin
    v0 = texto.split("Ventana 0 extraccion")[1].split("rampa CVODE")[0]
    assert "EQUILIBRIUM_PHASES 0" not in v0
    assert "diffusion_only" in v0
    vobs = texto.split("observacion")[-1]
    assert "EQUILIBRIUM_PHASES 0" not in vobs.split("TRANSPORT")[0]


@requiere_datos
@pytest.mark.skipif(not PHREEQC_EXE.exists(), reason="phreeqc.exe no instalado")
def test_smoke_phreeqc_trazador():
    run_dir = gen.correr("malla_original", TASA_REF, modo="trazador")
    assert (run_dir / "OK.txt").exists()
    assert (run_dir / "transporte_trazador.txt").exists()

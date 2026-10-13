"""Generador del .pqi TRANSPORT multi-ventana (met2) y ejecucion de corridas."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccus_columna.constantes import TASA_REF, TASAS_M3D  # parametros del proyecto
from ccus_columna import datos, quimica, runner
from ccus_columna.cli import listar_corridas, resolver_dir_corrida, validar_malla_tasa
from ccus_columna.constantes import (
    BAR_A_ATM,
    D_DIFUSION_M2S,
    K_INYECCION,
    SG_UMBRAL_GAS,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.geometria import escala_volumen
from ccus_columna.mineralogia import m_water_ref
from ccus_columna.postproceso import guardar_perfil, perfil_desde_met2

import calibracion as cal

PLANTILLAS = Path(__file__).resolve().parent / "plantillas"
RAIZ_CORRIDAS = Path(__file__).resolve().parent / "corridas"


SI_CO2_APAGADO = -10.0  # no usar en reactivo: deja pH~10.7 y CVODE muere
# Buffer sin pluma: SI elegido para C(4)~0.002 mol/kgw (agua nativa + calcita,
# mismo orden que met1). SI=-2.5 dejaba C~0.00063 y pH~7 (DC-018).
SI_CO2_SUAVE = -1.62
# dc018: buffer suave cuando no hay pluma (DC-018).
# fugacidad_met1: misma P_CO2= P*min(1,Sg/Sg_max) cuando hay gas;
#   sin pluma no se impone el buffer SI=-1.62 (alinea mejor a GAS_PHASE met1).
FORZANTE_FUGACIDAD = "dc018"  # o "fugacidad_met1"
MOLES_CO2_PLUMA = 10.0
MOLES_CO2_SUAVE = 1.0


def ventanas_por_etapa(malla: str = "malla_original",
                       tasa: int = TASA_REF) -> list[tuple[float, float, str]]:
    """Etapas gruesas (para v por etapa). El .pqi usa nodos del CSV."""
    return cal.ventanas_operativas(malla, tasa)


def _resolver_pasos(malla: str, tasa: int, ventanas) -> list[dict]:
    """Lista de pasos TRANSPORT. Por defecto: 4 etapas (rapido)."""
    if ventanas is None:
        ventanas = ventanas_por_etapa(malla, tasa)
    if ventanas and isinstance(ventanas[0], dict):
        return list(ventanas)
    out = []
    for item in _normalizar_ventanas(malla, tasa, ventanas):
        t_a, t_b, nombre = item
        p = cal.parametros_transport(malla, tasa, t_a, t_b)
        out.append({"t_a": t_a, "t_b": t_b, "nombre": nombre, **p})
    return out


def _si_firma(celdas: list[int], si_map: dict[int, float | None],
              forzar: bool) -> tuple:
    if not forzar:
        return tuple((k, None) for k in celdas)
    return tuple(
        (k, None if si_map.get(k) is None else round(float(si_map[k]), 1))
        for k in celdas
    )


def _co2_si_moles(si: float | None, apagar: bool = False,
                  malla: str | None = None, k: int | None = None) -> tuple[float, float]:
    """SI y moles de CO2(g). apagar=True: sin fase (extraccion, sin cinetica)."""
    if apagar:
        return SI_CO2_APAGADO, 0.0
    escala = escala_volumen(malla, k)
    if si is None:
        if FORZANTE_FUGACIDAD == "fugacidad_met1":
            # Sin buffer DC-018. SI=-10 + 0 moles alcaliniza y mata CVODE;
            # usar SI bajo pero numerico-estable (C acuoso << 0.002 de SI=-1.62).
            return -4.0, max(1e-6, 1e-4 * escala)
        return SI_CO2_SUAVE, MOLES_CO2_SUAVE * escala
    return float(si), MOLES_CO2_PLUMA * escala


def etapa_con_inyeccion(nombre: str, t_mid: float) -> bool:
    """CO2 entra por el pozo solo durante inyeccion (no 0-10 a ni 50-70 a)."""
    if nombre in ("extraccion", "observacion"):
        return False
    return T_FIN_EXTRACCION <= t_mid < T_FIN_INYECCION


def si_entrada_pozo(si_col: dict[int, float | None], celdas: list[int]) -> float | None:
    """SI de SOLUTION 0: max fugacidad en el intervalo perforado (k>=15).

    El fondo de la columna puede no tener gas al inicio; la inyeccion Cirrus
    ocurre en k15..k30 (DC-017).
    """
    vals = [
        float(si_col[k])
        for k in celdas
        if k >= K_INYECCION and si_col.get(k) is not None
    ]
    return max(vals) if vals else None


def si_co2_por_celda(malla: str, tasa: int, celdas: list[int], t_anos: float) -> dict[int, float | None]:
    """SI de CO2(g) por celda en t (mapa de fugacidad compartido con met1)."""
    r = datos.resumen_frente()
    out: dict[int, float | None] = {}
    for k in celdas:
        s = datos.interpolar_celda(malla, tasa, k, t_anos)
        fila = r[(r["malla"] == malla) & (r["tasa_m3d"] == tasa) & (r["k"] == k)]
        sg_max = float(fila["Sg_max"].iloc[0]) if len(fila) else 1.0
        if s["Sg"] <= SG_UMBRAL_GAS or sg_max <= SG_UMBRAL_GAS:
            out[k] = None
            continue
        f_co2 = s["P_bar"] * BAR_A_ATM * min(1.0, s["Sg"] / sg_max)
        out[k] = math.log10(max(f_co2, 1e-10))
    return out


def _eq_rocas_y_co2(celdas: list[int], si_map: dict[int, float | None] | None = None,
                    apagar: bool = False, malla: str | None = None) -> str:
    """Rocas + CO2(g). Extraccion: apagado. Inyeccion: pluma o buffer suave."""
    bloques = []
    for i, k in enumerate(celdas, start=1):
        si = None if si_map is None else si_map.get(k)
        si_val, moles = _co2_si_moles(si, apagar=apagar, malla=malla, k=k)
        bloques.append(
            f"{quimica.equilibrium_rocas(i, k, malla=malla)}\n"
            f"    CO2(g)            {si_val:.6f}  {moles:.6g}\n"
        )
    return "\n".join(bloques)


def _eq_modify_co2(celdas: list[int], si_map: dict[int, float | None],
                   forzar_inyeccion: bool, malla: str | None = None) -> str:
    """Actualiza SI de CO2(g) sin resetear moles de calcita.

    Nunca SI=-10 con 0 moles en el reactivo: eso alcaliniza y CVODE falla
    en ~1e-7 s (celda sin pluma → celda con pluma).
    """
    bloques = []
    for i, k in enumerate(celdas, start=1):
        # fugacidad_met1: mantener mapa P_CO2 tambien en observacion
        # (el gas libre sigue en el poro tras el cierre de inyeccion).
        usar_mapa = forzar_inyeccion or (FORZANTE_FUGACIDAD == "fugacidad_met1")
        si = si_map.get(k) if usar_mapa else None
        si_val, moles = _co2_si_moles(si, malla=malla, k=k)
        bloques.append(f"""EQUILIBRIUM_PHASES_MODIFY {i}
    -component CO2(g)
        -si     {si_val:.6f}
        -moles  {moles:.6g}
""")
    return "\n".join(bloques) + "END\n"


def _kinetics_columna(celdas: list[int], malla: str | None = None) -> str:
    bloques = []
    for i, k in enumerate(celdas, start=1):
        bloques.append(quimica.bloque_kinetics_inicial(
            i, k, cvode=True, rapido=False,
            cvode_steps=50000, bad_step_max=100000,
            malla=malla,
        ))
    return "\n".join(bloques)


def _rampa_cvode(celdas: list[int], dias: int = 7,
                 malla: str | None = None, tasa: int | None = None) -> str:
    """Pasos de 1 dia, sin adveccion, para que CVODE arranque sin saltos de anos."""
    return (
        f"# rampa CVODE {dias} d diffusion_only\n"
        + _transport_block(celdas, dias, 86400.0, dias, "diffusion_only",
                           malla=malla, tasa=tasa)
        + "\n"
    )


def _ventanas_pqi(pasos: list[dict], malla: str, tasa: int, celdas: list[int],
                  start_iv: int, prev_firma, prev_si_in) -> tuple[str, object, object]:
    bloques: list[str] = []
    for j, w in enumerate(pasos):
        iv = start_iv + j
        t_a, t_b, nombre = w["t_a"], w["t_b"], w["nombre"]
        t_mid = 0.5 * (t_a + t_b)
        iny = etapa_con_inyeccion(nombre, t_mid)
        si_col = si_co2_por_celda(malla, tasa, celdas, t_mid)
        si_in = si_entrada_pozo(si_col, celdas) if iny else None
        if nombre in ("extraccion", "observacion"):
            si_in = None
        si_in_key = None if si_in is None else round(float(si_in), 1)
        forzar_celdas = nombre != "extraccion"
        firma = _si_firma(celdas, si_col, forzar_celdas)
        extra = []
        if firma != prev_firma:
            extra.append(_eq_modify_co2(celdas, si_col, forzar_celdas, malla=malla))
            prev_firma = firma
        if si_in_key != prev_si_in:
            extra.append(_entrada_brina(si_in, malla=malla, k_in=celdas[0]))
            prev_si_in = si_in_key
        bloques.append(
            f"# --- Ventana {iv} {nombre}: {t_a:.3f}-{t_b:.3f} a "
            f"| v={w['v_m_por_ano']:.3f} m/a shifts={w['shifts']} "
            f"{w['flow_direction']} ---\n"
            f"{''.join(extra)}"
            f"{_transport_block(celdas, w['shifts'], w['time_step_s'], 1, w['flow_direction'], malla=malla, tasa=tasa)}\n"
        )
    return "".join(bloques), prev_firma, prev_si_in


def _pasos_reactivo(malla: str, tasa: int, ventanas) -> list[dict]:
    """TRANSPORT reactivo: 0-10 a y 50-70 a sin CO2 por el pozo ni adveccion."""
    return _resolver_pasos(malla, tasa, ventanas)


def _soluciones_columna(celdas: list[int], malla: str | None = None) -> str:
    return "\n".join(
        quimica.agua_nativa_default(i, f"Agua_k{k}", k=k, malla=malla)
        for i, k in enumerate(celdas, start=1)
    )


def _lista_m(vals: list[float]) -> str:
    return " ".join(f"{x:.6g}" for x in vals)


def _transport_block(celdas: list[int], shifts: int, time_step_s: float,
                     punch_freq: int, flow_direction: str = "forward",
                     malla: str | None = None, tasa: int | None = None) -> str:
    n = len(celdas)
    return f"""TRANSPORT
    -cells                 {n}
    -shifts                {shifts}
    -time_step             {time_step_s:.6g}
    -flow_direction        {flow_direction}
    -boundary_conditions   flux flux
    -lengths               {_lista_m(cal.longitudes_m(celdas))}
    -dispersivities        {_lista_m(cal.dispersividades_m(celdas, malla, tasa))}
    -diffusion_coefficient {D_DIFUSION_M2S:.6g}
    -punch_cells           1-{n}
    -punch_frequency       {punch_freq}
END"""


def _entrada_brina(si_co2: float | None, cl_boost: float = 0.0,
                   malla: str | None = None, k_in: int | None = None) -> str:
    """SOLUTION 0: agua nativa (+Cl trazador) equilibrada con CO2 si aplica."""
    cl0 = quimica._B["mg_l"]["Cl"]
    cl = float(cl0) * (1.0 + cl_boost)
    p_atm = 10 ** si_co2 if si_co2 is not None else 100.0
    agua = m_water_ref(k_in, malla)
    sol = quimica.solucion_nativa(0, "Brina_entrada", agua, max(p_atm, 1.0))
    if cl_boost > 0:
        sol = sol.replace(f"Cl        {cl0}", f"Cl        {cl:.2f}")
    partes = [sol, "SAVE solution 0", "END"]
    if si_co2 is not None:
        _, moles_co2 = _co2_si_moles(si_co2, malla=malla, k=k_in)
        partes.append(f"""USE solution 0
EQUILIBRIUM_PHASES 0
    CO2(g)    {si_co2:.6f}  {moles_co2:.6g}
SAVE solution 0
END""")
    return "\n".join(partes) + "\n"


def _normalizar_ventanas(malla: str, tasa: int, ventanas) -> list[tuple[float, float, str]]:
    if not ventanas:
        return ventanas_por_etapa(malla, tasa)
    out = []
    for item in ventanas:
        if len(item) == 3:
            out.append((float(item[0]), float(item[1]), str(item[2])))
        else:
            t_a, t_b = float(item[0]), float(item[1])
            t_mid = 0.5 * (t_a + t_b)
            if t_mid < T_FIN_EXTRACCION:
                nombre = "extraccion"
            elif t_mid >= T_FIN_INYECCION:
                nombre = "observacion"
            else:
                nombre = "inyeccion"
            out.append((t_a, t_b, nombre))
    return out


def pqi_trazador(malla: str, tasa: int,
                 ventanas: list | None = None,
                 archivo_out: str = "transporte_trazador.txt") -> str:
    """M2.2: columna SIN minerales; Cl elevado en la entrada solo en inyeccion."""
    celdas = cal.celdas_transport(malla, tasa)
    pasos = _resolver_pasos(malla, tasa, ventanas)
    k0 = celdas[0]

    bloques_v = []
    prev_si = object()
    for iv, w in enumerate(pasos):
        t_a, t_b, nombre = w["t_a"], w["t_b"], w["nombre"]
        t_mid = 0.5 * (t_a + t_b)
        iny = etapa_con_inyeccion(nombre, t_mid)
        si_col = si_co2_por_celda(malla, tasa, celdas, t_mid) if (iny or FORZANTE_FUGACIDAD == "fugacidad_met1") else {}
        si_in = si_entrada_pozo(si_col, celdas) if iny else None
        extra = ""
        if si_in != prev_si:
            extra = _entrada_brina(si_in, cl_boost=1.0 if si_in is not None else 0.0,
                                  malla=malla, k_in=celdas[0]) + "\n"
            prev_si = si_in
        bloques_v.append(
            f"# --- Ventana {iv} {nombre}: {t_a:.3f}-{t_b:.3f} a "
            f"| v={w['v_m_por_ano']:.3f} m/a {w['flow_direction']} ---\n"
            f"{extra}"
            f"{_transport_block(celdas, w['shifts'], w['time_step_s'], 1, w['flow_direction'], malla=malla, tasa=tasa)}\n"
        )

    return f"""# met2 trazador {malla} q={tasa} celdas={celdas}
# pasos={len(pasos)} (etapas operativas)
{quimica.FASES_SECUNDARIAS}

{_soluciones_columna(celdas, malla)}
END

{quimica.selected_output_transport(archivo_out, con_kinetics=False)}

USER_PUNCH 1
    -headings   k_cirrus C_mol_kgw Cl_mol_kgw
    -start
    10 IF (CELL_NO < 1) THEN GOTO 100
    20 kmap = {k0} - (CELL_NO - 1)
    30 IF (kmap < 10) THEN GOTO 100
    40 PUNCH kmap, TOT("C"), TOT("Cl")
   100 REM
    -end

{''.join(bloques_v)}
"""


def pqi_reactivo(malla: str, tasa: int,
                 ventanas: list | None = None,
                 archivo_out: str = "transporte_reactivo.txt") -> str:
    """Columna reactiva. CVODE desde la inyeccion, con rampa de dias.

    0-T_FIN_EXTRACCION: sin KINETICS, sin CO2 de pozo, buffer suave (pH plano).
    Luego 7 d de CVODE a SI suave y 7 d ya con pluma, y despues los
    TRANSPORT largos. 50.4-70 a: pozo sin CO2; gas residual si Sg>0.
    Columna = intervalo perforado (hasta k<=30) → sello (DC-017).
    """
    celdas = cal.celdas_transport(malla, tasa)
    pasos = _pasos_reactivo(malla, tasa, ventanas)
    if not pasos:
        raise ValueError("met2 reactivo sin pasos TRANSPORT")
    rate_params, rates, _ = quimica.bloques_roca()
    k0 = celdas[0]

    ext = [w for w in pasos if w["nombre"] == "extraccion"]
    resto = [w for w in pasos if w["nombre"] != "extraccion"]
    hay_ext = bool(ext)

    w0 = pasos[0]
    t_ref = 0.5 * (float(w0["t_a"]) + float(w0["t_b"]))
    iny0 = etapa_con_inyeccion(w0["nombre"], t_ref)
    if iny0:
        si_col0 = si_co2_por_celda(malla, tasa, celdas, t_ref)
        si_in0 = si_entrada_pozo(si_col0, celdas)
    else:
        si_col0 = {k: None for k in celdas}
        si_in0 = None

    prev_firma = _si_firma(celdas, si_col0, forzar=iny0)
    prev_si_in = None if si_in0 is None else round(float(si_in0), 1)
    txt_ext, prev_firma, prev_si_in = _ventanas_pqi(
        ext, malla, tasa, celdas, 0, prev_firma, prev_si_in,
    )

    rampa = ""
    if hay_ext and resto:
        w1 = resto[0]
        t_mid1 = 0.5 * (float(w1["t_a"]) + float(w1["t_b"]))
        si_col1 = si_co2_por_celda(malla, tasa, celdas, t_mid1)
        si_in1 = si_entrada_pozo(si_col1, celdas)
        rampa = (
            f"{_kinetics_columna(celdas, malla)}\nEND\n\n"
            f"{_rampa_cvode(celdas, malla=malla, tasa=tasa)}\n"
            f"{_eq_modify_co2(celdas, si_col1, True, malla=malla)}"
            f"{_entrada_brina(si_in1, malla=malla, k_in=celdas[0])}"
            f"{_rampa_cvode(celdas, malla=malla, tasa=tasa)}\n"
        )
        prev_firma = _si_firma(celdas, si_col1, forzar=True)
        prev_si_in = None if si_in1 is None else round(float(si_in1), 1)
        kin_despues = ""
    else:
        kin_despues = f"{_kinetics_columna(celdas, malla)}\nEND\n\n" if hay_ext else ""

    txt_resto, _, _ = _ventanas_pqi(
        resto, malla, tasa, celdas, len(ext), prev_firma, prev_si_in,
    )
    kin_header = "" if hay_ext else _kinetics_columna(celdas, malla)

    return f"""# met2 reactivo {malla} q={tasa} celdas={celdas} forzante={FORZANTE_FUGACIDAD}
# pasos={len(pasos)} t={float(pasos[0]['t_a']):.3f}-{float(pasos[-1]['t_b']):.3f} a
# extraccion: sin cinetica ni CO2 de pozo; rampa CVODE; inyeccion; observacion
# DC-017: columna perforacion->sello (k_fondo={k0} .. k10)
{quimica.FASES_SECUNDARIAS}

{rate_params}

RATES
{rates}

{_entrada_brina(si_in0, malla=malla, k_in=celdas[0])}

{_soluciones_columna(celdas, malla)}
{_eq_rocas_y_co2(celdas, si_col0, apagar=False, malla=malla)}
{kin_header}
END

{quimica.selected_output_transport(archivo_out)}

USER_PUNCH 1
    -headings   k_cirrus C_mol_kgw
    -start
    10 IF (CELL_NO < 1) THEN GOTO 100
    20 kmap = {k0} - (CELL_NO - 1)
    30 IF (kmap < 10) THEN GOTO 100
    40 PUNCH kmap, TOT("C")
   100 REM
    -end

{txt_ext}{rampa if hay_ext else kin_despues}{txt_resto}
"""


def exportar_resultados(run_dir: Path, modo: str) -> Path:
    archivo = "transporte_reactivo.txt" if "reactivo" in modo else "transporte_trazador.txt"
    df = perfil_desde_met2(run_dir, archivo)
    return guardar_perfil(df, run_dir / "perfil_quimica_met2.csv")


def _texto_params(malla: str, tasa: int, modo: str, pasos: list) -> str:
    pasos = _resolver_pasos(malla, tasa, pasos)
    lines = [
        f"malla={malla}",
        f"tasa={tasa}",
        f"modo={modo}",
        f"n_pasos={len(pasos)}",
        "",
    ]
    n_et = Counter(w["nombre"] for w in pasos)
    n_fwd = Counter(
        w["nombre"] for w in pasos if w["flow_direction"] == "forward"
    )
    sh_fwd = {}
    for w in pasos:
        if w["flow_direction"] == "forward":
            sh_fwd[w["nombre"]] = sh_fwd.get(w["nombre"], 0) + w["shifts"]
    for nombre, n in n_et.items():
        lines.append(
            f"{nombre}: {n} ventanas, shifts_adveccion={sh_fwd.get(nombre, 0)}, "
            f"nodos_con_flujo={n_fwd.get(nombre, 0)}"
        )
    return "\n".join(lines) + "\n"


def _rmtree_reintento(path: Path, intentos: int = 8, espera_s: float = 1.5) -> None:
    """Borra corrida previa; en Windows el .pqi/.pqo puede estar bloqueado."""
    import os
    import time
    ultimo: Exception | None = None
    for i in range(intentos):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            ultimo = exc
            time.sleep(espera_s * (1.0 + 0.25 * i))
        except OSError as exc:
            ultimo = exc
            time.sleep(espera_s)
    basura = path.with_name(path.name + f"_lock_{os.getpid()}")
    try:
        path.rename(basura)
        print(
            f"Aviso: no se pudo borrar {path.name} (archivo en uso). "
            f"Renombrado a {basura.name}. Cierre la otra corrida PHREEQC "
            f"o espere a que termine, luego borre la carpeta a mano.",
            flush=True,
        )
        return
    except OSError:
        pass
    raise PermissionError(
        f"No se puede sobrescribir {path}: hay otra corrida usando esos archivos. "
        f"Espere a que termine o use --etiqueta otra_corrida.\nDetalle: {ultimo}"
    ) from ultimo


def correr(malla: str, tasa: int, modo: str = "trazador",
           etiqueta: str = "", reanudar: bool = False) -> Path:
    """Genera .pqi, ejecuta PHREEQC y guarda en corridas/."""
    tag = etiqueta or modo
    run_dir = RAIZ_CORRIDAS / f"{malla}_q{tasa}_{tag}"
    ok_file = run_dir / "OK.txt"
    if reanudar and ok_file.exists():
        print(f"Ya existe corrida OK: {run_dir}")
        exportar_resultados(run_dir, modo)
        return run_dir

    if run_dir.exists():
        _rmtree_reintento(run_dir)
    run_dir.mkdir(parents=True)

    todos = ventanas_por_etapa(malla, tasa)

    if modo == "trazador":
        vents = todos
        texto = pqi_trazador(malla, tasa, ventanas=vents,
                             archivo_out="transporte_trazador.txt")
        name = "modelo_trazador"
    elif modo == "trazador_corto":
        vents = [(T_FIN_EXTRACCION, min(T_FIN_INYECCION, T_FIN_EXTRACCION + 15.0))]
        texto = pqi_trazador(malla, tasa, ventanas=vents,
                             archivo_out="transporte_trazador.txt")
        name = "modelo_trazador"
    elif modo == "reactivo":
        vents = todos
        texto = pqi_reactivo(malla, tasa, ventanas=vents,
                             archivo_out="transporte_reactivo.txt")
        name = "modelo_reactivo"
    elif modo == "reactivo_corto":
        vents = [(T_FIN_EXTRACCION, T_FIN_EXTRACCION + 2.0, "inyeccion")]
        texto = pqi_reactivo(malla, tasa, ventanas=vents,
                             archivo_out="transporte_reactivo.txt")
        name = "modelo_reactivo_corto"
    else:
        raise ValueError(f"modo desconocido: {modo}")

    (run_dir / "params.txt").write_text(
        _texto_params(malla, tasa, modo, vents), encoding="utf-8",
    )
    n_tr = texto.count("TRANSPORT")
    print(f"PHREEQC {name}: {n_tr} bloque(s) TRANSPORT. "
          "No hay progreso tipo met1; mire LastWriteTime del .pqo / .txt.",
          flush=True)
    res = runner.run_in_dir(texto, run_dir, name=name)
    if not res.ok:
        raise RuntimeError("PHREEQC fallo:\n" + "\n".join(res.errors[:10]))
    csv_path = exportar_resultados(run_dir, modo)
    (run_dir / "OK.txt").write_text(
        res.summary() + f"\nperfil={csv_path.name}\n", encoding="utf-8",
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Correr met2 TRANSPORT")
    ap.add_argument("--modo", choices=(
        "trazador", "trazador_corto", "reactivo", "reactivo_corto",
    ), default="trazador")
    ap.add_argument("--malla", default="malla_original",
                    help="malla_original o malla_refinada")
    ap.add_argument("--tasa", type=int, default=TASA_REF,
                    help="tasas disponibles: " + ", ".join(str(q) for q in TASAS_M3D))
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--etiqueta", default="",
                    help="sufijo de carpeta (default=modo). Evita pisar una corrida en curso.")
    ap.add_argument("--forzante", choices=("dc018", "fugacidad_met1"), default="dc018",
                    help="dc018=buffer SI suave; fugacidad_met1=mapa P_CO2 sin buffer suave")
    ap.add_argument("--figuras", action="store_true")
    ap.add_argument("--listar", action="store_true",
                    help="listar corridas existentes y salir")
    ap.add_argument("--solo-generar", action="store_true",
                    help="escribe .pqi sin ejecutar PHREEQC")
    ap.add_argument("--solo-exportar", type=str, default=None,
                    help="nombre o ruta de una corrida existente (NO use <id>)")
    args = ap.parse_args(argv)

    if args.listar:
        print(f"Corridas en {RAIZ_CORRIDAS}:")
        print(listar_corridas(RAIZ_CORRIDAS))
        return 0

    if args.solo_exportar:
        run_dir = resolver_dir_corrida(args.solo_exportar, RAIZ_CORRIDAS)
        exportar_resultados(run_dir, args.modo)
        if args.figuras:
            from generar_figuras import figuras_met2
            for p in figuras_met2(run_dir):
                print(f"  figura: {p}")
        print(f"Exportado -> {run_dir / 'perfil_quimica_met2.csv'}")
        return 0

    validar_malla_tasa(args.malla, args.tasa)
    global FORZANTE_FUGACIDAD
    FORZANTE_FUGACIDAD = args.forzante

    if args.solo_generar:
        run_dir = RAIZ_CORRIDAS / f"{args.malla}_q{args.tasa}_{args.modo}_gen"
        run_dir.mkdir(parents=True, exist_ok=True)
        etapas = ventanas_por_etapa(args.malla, args.tasa)
        if args.modo == "trazador":
            texto = pqi_trazador(args.malla, args.tasa, ventanas=etapas)
            name = "modelo_trazador.pqi"
        elif args.modo == "trazador_corto":
            vents = [(T_FIN_EXTRACCION, min(T_FIN_INYECCION, T_FIN_EXTRACCION + 15.0))]
            texto = pqi_trazador(args.malla, args.tasa, ventanas=vents)
            name = "modelo_trazador.pqi"
        elif args.modo == "reactivo_corto":
            vents = [(T_FIN_EXTRACCION, T_FIN_EXTRACCION + 2.0, "inyeccion")]
            texto = pqi_reactivo(args.malla, args.tasa, ventanas=vents)
            name = "modelo_reactivo_corto.pqi"
        else:
            texto = pqi_reactivo(args.malla, args.tasa, ventanas=etapas)
            name = "modelo_reactivo.pqi"
        (run_dir / name).write_text(texto, encoding="utf-8")
        print(f"Generado {run_dir / name} ({len(texto)} chars)")
        return 0

    etiq = args.etiqueta
    if not etiq and args.forzante == "fugacidad_met1" and "reactivo" in args.modo:
        etiq = f"{args.modo}_fugmet1"
    run_dir = correr(args.malla, args.tasa, args.modo,
                     etiqueta=etiq, reanudar=args.reanudar)
    if args.figuras:
        from generar_figuras import figuras_met2
        for p in figuras_met2(run_dir):
            print(f"  figura: {p}")
    print(f"OK -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

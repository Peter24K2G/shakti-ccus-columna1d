"""Reglas de acople entre celdas: orden de flujo y fraccion transferida.

Decisiones DC-004 (orden) y DC-005 (fraccion f) en docs/decisiones.md.
El principio: la cinematica sale de los DATOS (t_primer_gas del frente Cirrus),
no de un modelo de flujo propio.
"""

from __future__ import annotations

from ccus_columna import datos
from ccus_columna.constantes import DZ_M, K_INYECCION_FONDO, K_SELLO, SG_UMBRAL_GAS

# Tau por celda cuando la profunda no tiene t_primer_gas (solo sube hacia k10)
_TAU_CELDA_ANOS = 0.5


def transferencia_activa(
    malla: str,
    tasa: int,
    k_origen: int,
    k_destino: int,
    t_anos: float,
    activacion_f: str = "sg",
    sg_umbral: float = SG_UMBRAL_GAS,
    dp_umbral_bar: float = 0.5,
) -> bool:
    """Condicion de activacion del MIX (DC-005 / DC-021).

    sg: receptor con Sg > umbral (legacy).
    sg_o_dp: ademas permite transferencia si P(origen) > P(destino) + umbral
    (empuje hidraulico de salmuera hacia el sello, aun sin gas en destino).
    """
    s_dest = datos.interpolar_celda(malla, tasa, k_destino, t_anos)
    if s_dest["Sg"] > sg_umbral:
        return True
    if activacion_f == "sg":
        return False
    if activacion_f == "sg_o_dp":
        if k_origen <= k_destino:
            return False
        s_orig = datos.interpolar_celda(malla, tasa, k_origen, t_anos)
        return (s_orig["P_bar"] - s_dest["P_bar"]) > dp_umbral_bar
    raise ValueError(f"activacion_f desconocida: {activacion_f}")


def orden_de_flujo(malla: str, tasa: int, dominio: str = "rama_sello") -> list[int]:
    """Celdas en el orden en que se corren y encadenan (DC-004 / DC-019).

    dominio:
      - rama_sello: k15 -> k10 (v1)
      - columna_pozo: k30 -> k10, transporte hacia sello (met1 DC-020)
      - frente: orden por t_primer_gas (legacy)
    """
    if dominio == "columna_pozo":
        return datos.columna_met1_sello(K_INYECCION_FONDO)
    if dominio == "rama_sello":
        return datos.rama_al_sello(malla, tasa)
    if dominio == "frente":
        return datos.orden_frente(malla, tasa)
    raise ValueError(f"dominio desconocido: {dominio}")


def tau_residencia_anos(malla: str, tasa: int, k_origen: int, k_destino: int) -> float | None:
    """Tiempo de transito del frente entre dos celdas vecinas (DC-005 opcion 1).

    k_origen > k_destino (mas profunda -> hacia sello k10).
    tau = t_primer_gas(destino) - t_primer_gas(origen). Si origen no ve gas pero
    destino si, usa (k_origen - k_destino) * _TAU_CELDA_ANOS (DC-020).
    """
    if k_origen <= k_destino:
        return None
    r = datos.resumen_frente()
    r = r[(r["malla"] == malla) & (r["tasa_m3d"] == tasa)].set_index("k")["t_primer_gas_anos"]
    t0, t1 = r.get(k_origen), r.get(k_destino)
    if t1 is None or t1 != t1:
        return None
    if t0 is None or t0 != t0:
        return max((k_origen - k_destino) * _TAU_CELDA_ANOS, 1e-6)
    return max(float(t1) - float(t0), 1e-6)


def fraccion_transferida(regla: str, dt_anos: float, malla: str, tasa: int,
                         k_origen: int, k_destino: int, t_anos: float,
                         f_constante: float = 0.25,
                         activacion_f: str = "sg",
                         sg_umbral: float = SG_UMBRAL_GAS,
                         dp_umbral_bar: float = 0.5) -> float:
    """Fraccion f de agua reaccionada de k_origen que entra a k_destino en dt."""
    if not transferencia_activa(
        malla, tasa, k_origen, k_destino, t_anos,
        activacion_f, sg_umbral, dp_umbral_bar,
    ):
        return 0.0
    if regla == "constante":
        return f_constante
    if regla == "tau_residencia":
        tau = tau_residencia_anos(malla, tasa, k_origen, k_destino)
        if tau is None:
            return 0.0
        return min(1.0, dt_anos / tau)
    if regla == "delta_sg":
        # TODO(M1.6): f proporcional al dSg del receptor en el paso, normalizado
        # por Sg_max de la celda. Definir constante de proporcionalidad.
        raise NotImplementedError("DC-005 opcion 2")
    raise ValueError(f"regla desconocida: {regla}")


def bloque_mix(f: float, numero_efluente: int = 99, numero_local: int = 1) -> str:
    """MIX PHREEQC: f de efluente aguas arriba + (1-f) de agua local.

    Mismo artificio que el 'mix superficie' de la fase 0D, pero el aporte es el
    agua REACCIONADA de la celda previa (no brina fresca de superficie).
    Por convencion met1: local=1, efluente=99 (ver estado.py).
    """
    if f <= 1e-12:
        return ""
    f = min(1.0, f)
    return f"""MIX {numero_local}
    {numero_efluente}    {f:.6f}
    {numero_local}    {1.0 - f:.6f}
SAVE solution {numero_local}
END
"""

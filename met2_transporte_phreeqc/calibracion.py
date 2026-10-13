"""Calibracion de parametros TRANSPORT desde P y Sg de Cirrus (DC-007/DC-012/DC-017).

Velocidad hacia el sello en la columna perforacion→sello (k_fondo..k10):

1. Signo Darcy desde el perfil de presion: si el flujo acuoso va hacia el
   productor (extraccion / shut-in), v_TRANSPORT = 0.
2. Magnitud desde el campo de Sg (concentracion de CO2):
       v = mediana de  -(dSg/dt) / (dSg/dz)    (adveccion 1D del pluma)
3. Respaldo: v_frente = dz(destino) / (t_primer_gas_j - t_primer_gas_i).

Por debajo de V_MIN_TRANSPORTE_M_A se usa diffusion_only (D molecular
1e-9 m2/s; sin adveccion). Con v>0: forward = adveccion + a_L*v + D.
Observacion se parte en subventanas de 5 a para refrescar SI residual.
"""

from __future__ import annotations

import math

import numpy as np

from ccus_columna import datos
from ccus_columna.geometria import dz_m
from ccus_columna.constantes import (
    DZ_M,
    G_MS2,
    K_INYECCION,
    K_RES_MD,
    K_RES_MD_REFINADA_50K,
    TASA_REFINADA_K_ESPECIAL,
    K_SELLO,
    K_SELLO_MD,
    K_SELLO_MD_REFINADA_50K,
    MD_A_M2,
    MU_BRINA_PA_S,
    PHI,
    RHO_BRINA,
    SEG_POR_ANO,
    SG_UMBRAL_GAS,
    T_FIN,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
    V_MIN_TRANSPORTE_M_A,
)

RHO_KGM3 = RHO_BRINA * 1000.0
HYDRO_PA_M = RHO_KGM3 * G_MS2
DSG_DZ_MIN = 1e-5  # 1/m; evita v disparada en plateaus
# Cinetica PK2004 no integra 2-10 a en un solo shift (RK se cuelga; CVODE
# tambien si el paso es enorme). En diffusion_only se parte el tiempo.
DT_KIN_MAX_ANOS = 2.0


def celdas_transport(malla: str, tasa: int) -> list[int]:
    """Columna PHREEQC met2: intervalo perforado → sello (DC-017)."""
    return datos.columna_pozo_sello(malla, tasa)


def longitudes_m(celdas: list[int]) -> list[float]:
    return [dz_m(k) for k in celdas]


def longitud_hops_m(celdas: list[int]) -> float:
    """Camino entrada -> sello: suma de dz de las celdas de destino."""
    if len(celdas) < 2:
        return dz_m(celdas[0]) if celdas else DZ_M
    return float(sum(dz_m(k) for k in celdas[1:]))


def dz_shift_m(celdas: list[int]) -> float:
    """dz efectivo por shift PHREEQC (media de los hops)."""
    hops = [dz_m(k) for k in celdas[1:]]
    if not hops:
        return DZ_M
    return float(sum(hops) / len(hops))


def ancho_frente_sg_m(malla: str, tasa: int,
                      fraccion_lo: float = 0.1,
                      fraccion_hi: float = 0.9) -> dict:
    """Ancho espacial del frente Sg Cirrus al llegar al sello (DC-029).

    Mide la distancia entre Sg = f_lo·Sg_max y f_hi·Sg_max a lo largo de la
    columna transport en t ≈ t_primer_gas(k10).
    """
    rama = celdas_transport(malla, tasa)
    Sg = datos.perfil(malla, tasa, "Sg")
    t = Sg["tiempo_anos"].to_numpy(dtype=float)
    t_sello = t_llegada_sello_anos(malla, tasa)
    i = int(np.argmin(np.abs(t - t_sello)))
    row = Sg.iloc[i]
    z = 0.0
    zs: list[float] = []
    sgs: list[float] = []
    for k in rama:
        zs.append(z + 0.5 * dz_m(k))
        sgs.append(float(row[f"k{k}"]))
        z += dz_m(k)
    zs_a = np.asarray(zs, dtype=float)
    sgs_a = np.asarray(sgs, dtype=float)
    smax = max(float(sgs_a.max()), 1e-6)
    lo, hi = fraccion_lo * smax, fraccion_hi * smax
    z_lo = z_hi = None
    for za, sa, zb, sb in zip(zs_a[:-1], sgs_a[:-1], zs_a[1:], sgs_a[1:]):
        if abs(sb - sa) < 1e-12:
            continue
        for thr, slot in ((lo, "lo"), (hi, "hi")):
            if (sa - thr) * (sb - thr) <= 0:
                zc = float(za + (thr - sa) / (sb - sa) * (zb - za))
                if slot == "lo":
                    z_lo = zc
                else:
                    z_hi = zc
    ancho = abs(z_hi - z_lo) if (z_lo is not None and z_hi is not None) else float("nan")
    rama_sello = [k for k in rama if k <= K_INYECCION]
    if len(rama_sello) < 2:
        rama_sello = rama
    x_travel = longitud_hops_m(rama_sello)
    # Frente gaussiano 1D: L_(0.1-0.9) ≈ 3.92 sqrt(a_L · x)
    a_L = float("nan")
    if np.isfinite(ancho) and ancho > 0 and x_travel > 0:
        a_L = (ancho / 3.92) ** 2 / x_travel
    return {
        "t_perfil": float(t[i]),
        "Sg_max": smax,
        "ancho_m": float(ancho) if np.isfinite(ancho) else float("nan"),
        "x_travel_m": float(x_travel),
        "a_L_m": float(a_L) if np.isfinite(a_L) else float("nan"),
        "z_m": zs,
        "Sg": sgs,
        "celdas": list(rama),
    }


def a_L_calibrada_m(malla: str, tasa: int,
                    a_L_min: float = 0.5,
                    a_L_max: float = 20.0) -> float:
    """Dispersividad longitudinal calibrada al ancho del frente Cirrus.

    Si no hay frente medible, cae a 0.1·dz medio (DC-014).
    """
    info = ancho_frente_sg_m(malla, tasa)
    a_L = info["a_L_m"]
    if not np.isfinite(a_L) or a_L <= 0:
        return dispersividad_inicial_m(dz_shift_m(celdas_transport(malla, tasa)))
    return float(min(max(a_L, a_L_min), a_L_max))


def dispersividades_m(celdas: list[int],
                      malla: str | None = None,
                      tasa: int | None = None,
                      *,
                      calibrar: bool = True) -> list[float]:
    """a_L por celda. Por defecto calibrada al frente Cirrus (DC-029).

    Sin malla/tasa (o calibrar=False): a_L = 0.1 dz (semilla DC-014).
    Con calibracion: a_L unica (mismo valor en todas las celdas) escalada
    al ancho 0.1–0.9 de Sg al llegar al sello.
    """
    Ls = longitudes_m(celdas)
    if not calibrar or malla is None or tasa is None:
        return [0.1 * L for L in Ls]
    a_L = a_L_calibrada_m(malla, tasa)
    return [a_L for _ in Ls]

def distancia_centros_m(k_a: int, k_b: int) -> float:
    return 0.5 * (dz_m(k_a) + dz_m(k_b))


def permeabilidad_m2(malla: str, tasa: int, k: int) -> float:
    """K de portada Cirrus. La malla refinada en TASA_REFINADA_K_ESPECIAL usa otro par (no mezclar con original)."""
    if malla == "malla_refinada" and tasa == TASA_REFINADA_K_ESPECIAL:
        k_md = K_SELLO_MD_REFINADA_50K if k == K_SELLO else K_RES_MD_REFINADA_50K
    else:
        k_md = K_SELLO_MD if k == K_SELLO else K_RES_MD
    return k_md * MD_A_M2


def velocidad_frente_m_por_ano(malla: str, tasa: int) -> dict[tuple[int, int], float]:
    """{(k_origen, k_destino): v [m/a]} a lo largo de la columna transport."""
    rama = celdas_transport(malla, tasa)
    r = datos.resumen_frente()
    r = r[(r["malla"] == malla) & (r["tasa_m3d"] == tasa)].set_index("k")["t_primer_gas_anos"]
    v = {}
    for k0, k1 in zip(rama[:-1], rama[1:]):
        t0, t1 = r.get(k0), r.get(k1)
        if t0 is None or t1 is None or np.isnan(t0) or np.isnan(t1) or t1 <= t0:
            continue
        v[(k0, k1)] = dz_m(k1) / (t1 - t0)
    return v


def v_media_m_por_ano(malla: str, tasa: int) -> float:
    v = velocidad_frente_m_por_ano(malla, tasa)
    if not v:
        raise RuntimeError(f"Sin tramos con frente para {malla} q={tasa}")
    return float(np.mean(list(v.values())))


def t_llegada_sello_anos(malla: str, tasa: int) -> float:
    """Primer gas en k=10; si no llega, fin de inyeccion."""
    r = datos.resumen_frente()
    fila = r[(r["malla"] == malla) & (r["tasa_m3d"] == tasa) & (r["k"] == K_SELLO)]
    if fila.empty:
        return T_FIN_INYECCION
    t = fila["t_primer_gas_anos"].iloc[0]
    if t is None or np.isnan(t):
        return T_FIN_INYECCION
    return float(t)


def hops_hacia_sello(malla: str, tasa: int) -> list[tuple[int, int, float, float, float]]:
    """Tramos (k_origen, k_destino, t0, t1, v_m_a) hacia el sello con t_primer_gas.

    Solo hops con t1 > t0 (frente que avanza). v = dz(destino)/(t1-t0).
    Orden temporal de llegada.
    """
    rama = celdas_transport(malla, tasa)
    # Solo la rama buoyant k15→k10 (llegadas monotonas hacia el sello)
    rama_sello = [k for k in rama if k <= K_INYECCION]
    if len(rama_sello) < 2:
        rama_sello = rama
    r = datos.resumen_frente()
    r = r[(r["malla"] == malla) & (r["tasa_m3d"] == tasa)].set_index("k")["t_primer_gas_anos"]
    hops = []
    for k0, k1 in zip(rama_sello[:-1], rama_sello[1:]):
        t0, t1 = r.get(k0), r.get(k1)
        if t0 is None or t1 is None or np.isnan(t0) or np.isnan(t1) or t1 <= t0:
            continue
        v = dz_m(k1) / (float(t1) - float(t0))
        hops.append((int(k0), int(k1), float(t0), float(t1), float(v)))
    hops.sort(key=lambda h: h[2])
    return hops


def ventanas_operativas(malla: str, tasa: int,
                        *,
                        frente_por_hops: bool = True) -> list[tuple[float, float, str]]:
    """Extraccion / frente / inyeccion / observacion (subventanas 5 a).

    Si frente_por_hops=True (DC-029), la ventana del frente se parte en un
    tramo por hop k_i→k_{i+1} con v = dz/Δt_primer_gas, para que las
    llegadas PHREEQC coincidan con Cirrus celda a celda.
    Observacion partida para actualizar SI de CO2(g) residual (DC-017).
    """
    t_sello = t_llegada_sello_anos(malla, tasa)
    t_sello = min(max(t_sello, T_FIN_EXTRACCION + 1.0), T_FIN_INYECCION)
    out: list[tuple[float, float, str]] = [(0.0, T_FIN_EXTRACCION, "extraccion")]
    if frente_por_hops:
        hops = hops_hacia_sello(malla, tasa)
        t_cursor = T_FIN_EXTRACCION
        for k0, k1, t0, t1, _v in hops:
            ta = max(t_cursor, t0)
            tb = min(t1, t_sello)
            if tb <= ta + 1e-9:
                continue
            out.append((ta, tb, f"frente_{k0}_{k1}"))
            t_cursor = tb
        if t_cursor + 1e-6 < t_sello:
            out.append((t_cursor, t_sello, "inyeccion_frente"))
    else:
        out.append((T_FIN_EXTRACCION, t_sello, "inyeccion_frente"))
    out.append((t_sello, T_FIN_INYECCION, "inyeccion"))
    t = T_FIN_INYECCION
    while t < T_FIN - 1e-9:
        t2 = min(t + 5.0, T_FIN)
        out.append((t, t2, "observacion"))
        t = t2
    return out

def _muestras_ventana(t: np.ndarray, t_ini: float, t_fin: float) -> np.ndarray:
    """Indices i>=1 cuyo intervalo [t[i-1], t[i]] solapa [t_ini, t_fin]."""
    idx = []
    for i in range(1, len(t)):
        if t[i] < t_ini or t[i - 1] > t_fin:
            continue
        idx.append(i)
    return np.array(idx, dtype=int)


def v_darcy_hacia_sello_m_por_ano(malla: str, tasa: int,
                                  t_ini: float, t_fin: float) -> float:
    """Velocidad intersticial media (m/a) hacia el sello. Negativa = hacia el pozo.

    u = -(K/mu) (dP/dz_up + rho g); z_up de entrada (fondo) a k_sello.
    """
    rama = celdas_transport(malla, tasa)
    k_up = rama[0]
    k_top = rama[-1]
    P = datos.perfil(malla, tasa, "P")
    Sw = datos.perfil(malla, tasa, "Sw")
    t = P["tiempo_anos"].to_numpy()
    idx = _muestras_ventana(t, t_ini, t_fin)
    if len(idx) == 0:
        return 0.0

    dz = sum(distancia_centros_m(ka, kb) for ka, kb in zip(rama[:-1], rama[1:]))
    k_si = float(np.mean([permeabilidad_m2(malla, tasa, k) for k in rama]))
    vs = []
    for i in idx:
        p_up = 0.5 * (P[f"k{k_up}"].iloc[i] + P[f"k{k_up}"].iloc[i - 1])
        p_top = 0.5 * (P[f"k{k_top}"].iloc[i] + P[f"k{k_top}"].iloc[i - 1])
        sw = 0.5 * (
            0.5 * (Sw[f"k{k_up}"].iloc[i] + Sw[f"k{k_up}"].iloc[i - 1])
            + 0.5 * (Sw[f"k{k_top}"].iloc[i] + Sw[f"k{k_top}"].iloc[i - 1])
        )
        grad_pa_m = ((p_up - p_top) / dz) * 1.0e5  # bar/m -> Pa/m
        u = (k_si / MU_BRINA_PA_S) * (grad_pa_m - HYDRO_PA_M)  # Darcy, m/s
        v = u / (PHI * max(sw, 0.05))
        vs.append(v * SEG_POR_ANO)
    return float(np.mean(vs))


def v_sg_m_por_ano(malla: str, tasa: int,
                   t_ini: float, t_fin: float) -> float | None:
    """v (m/a) del pluma de CO2: mediana de -(dSg/dt)/(dSg/dz) hacia el sello."""
    rama = celdas_transport(malla, tasa)
    Sg = datos.perfil(malla, tasa, "Sg")
    t = Sg["tiempo_anos"].to_numpy()
    idx = _muestras_ventana(t, t_ini, t_fin)
    vals: list[float] = []
    for i in idx:
        dt = t[i] - t[i - 1]
        if dt <= 1e-12:
            continue
        for ka, kb in zip(rama[:-1], rama[1:]):
            sga = 0.5 * (Sg[f"k{ka}"].iloc[i] + Sg[f"k{ka}"].iloc[i - 1])
            sgb = 0.5 * (Sg[f"k{kb}"].iloc[i] + Sg[f"k{kb}"].iloc[i - 1])
            dsg_dz = (sgb - sga) / distancia_centros_m(ka, kb)
            dsg_dt = 0.5 * (
                (Sg[f"k{ka}"].iloc[i] - Sg[f"k{ka}"].iloc[i - 1])
                + (Sg[f"k{kb}"].iloc[i] - Sg[f"k{kb}"].iloc[i - 1])
            ) / dt
            if abs(dsg_dz) < DSG_DZ_MIN:
                continue
            v = -dsg_dt / dsg_dz
            if v > 0.0:
                vals.append(float(v))
    if not vals:
        return None
    return float(np.median(vals))


def hay_gas_en_ventana(malla: str, tasa: int, t_ini: float, t_fin: float) -> bool:
    Sg = datos.perfil(malla, tasa, "Sg")
    t = Sg["tiempo_anos"].to_numpy()
    mask = (t >= t_ini) & (t <= t_fin)
    if not mask.any():
        mask = (t >= t_ini - 0.6) & (t <= t_fin + 0.6)
    col = f"k{K_INYECCION}"
    if col not in Sg.columns:
        col = f"k{celdas_transport(malla, tasa)[0]}"
    return bool(Sg.loc[mask, col].max() > SG_UMBRAL_GAS)


def velocidad_ventana_m_por_ano(malla: str, tasa: int,
                                t_ini: float, t_fin: float) -> float:
    """Velocidad de TRANSPORT (m/a) hacia el sello en [t_ini, t_fin].

    Extraccion y observacion: P indica flujo hacia el productor -> 0.
    Inyeccion: magnitud del frente de Sg; si no hay muestras, v_frente media.
    """
    v_darcy = v_darcy_hacia_sello_m_por_ano(malla, tasa, t_ini, t_fin)
    gas = hay_gas_en_ventana(malla, tasa, t_ini, t_fin)
    if (not gas) or (v_darcy <= 0.0 and t_fin <= T_FIN_EXTRACCION + 1e-6):
        return 0.0
    if t_ini + 1e-6 >= T_FIN_INYECCION:
        return 0.0

    v_sg = v_sg_m_por_ano(malla, tasa, t_ini, t_fin)
    t_sello = t_llegada_sello_anos(malla, tasa)
    if (abs(t_ini - T_FIN_EXTRACCION) < 0.6
            and abs(t_fin - t_sello) < 1.5):
        rama = celdas_transport(malla, tasa)
        # Solo hops hacia el sello desde k15 (pluma buoyant); no exigir
        # recorrer todo el intervalo perforado en la ventana del frente.
        rama_sello = [k for k in rama if k <= K_INYECCION]
        if len(rama_sello) < 2:
            rama_sello = rama
        v_col = longitud_hops_m(rama_sello) / max(t_fin - t_ini, 1e-6)
        if v_sg is not None:
            return max(v_sg, v_col)
        return v_col

    if v_sg is not None and v_sg >= V_MIN_TRANSPORTE_M_A:
        return v_sg
    try:
        return v_media_m_por_ano(malla, tasa)
    except RuntimeError:
        return 0.0


def nombre_etapa(t_mid: float, t_sello: float,
                 hops: list[tuple[int, int, float, float, float]] | None = None) -> str:
    if t_mid < T_FIN_EXTRACCION:
        return "extraccion"
    if t_mid >= T_FIN_INYECCION:
        return "observacion"
    if hops:
        for k0, k1, t0, t1, _v in hops:
            if t0 - 1e-9 <= t_mid < t1 + 1e-9 and t_mid < t_sello:
                return f"frente_{k0}_{k1}"
    if t_mid < t_sello:
        return "inyeccion_frente"
    return "inyeccion"


def nodos_tiempo_csv(malla: str, tasa: int) -> np.ndarray:
    """Tiempos del Excel/CSV de presion (misma malla de t que Sg)."""
    t = datos.perfil(malla, tasa, "P")["tiempo_anos"].to_numpy(dtype=float)
    t = np.unique(t)
    t.sort()
    return t


def v_por_etapa(malla: str, tasa: int) -> dict[str, float]:
    """Una v por etapa operativa (no se recalcula en cada nodo del CSV).

    En tramos frente_k0_k1 usa exactamente dz(k1)/(t1-t0) (DC-029).
    """
    out: dict[str, float] = {}
    hops = hops_hacia_sello(malla, tasa)
    hop_v = {f"frente_{k0}_{k1}": v for k0, k1, _t0, _t1, v in hops}
    for t_a, t_b, nombre in ventanas_operativas(malla, tasa):
        if nombre in hop_v:
            out[nombre] = hop_v[nombre]
        else:
            out[nombre] = velocidad_ventana_m_por_ano(malla, tasa, t_a, t_b)
    return out

def _paso_transport(dur: float, v: float, travel: float,
                    dz_m: float) -> tuple[dict, float]:
    """Un intervalo del CSV: quimica siempre; adveccion solo si se acumula dz.

    PHREEQC no puede hacer medio shift: el residuo de distancia se guarda en
    `travel` (m) para el siguiente nodo.
    """
    dur = max(dur, 1e-12)
    if v < V_MIN_TRANSPORTE_M_A:
        return {
            "v_m_por_ano": 0.0,
            "time_step_s": dur * SEG_POR_ANO,
            "shifts": 1,
            "dt_shift_anos": dur,
            "flow_direction": "diffusion_only",
        }, travel

    travel = travel + v * dur
    n_sh = int(travel // dz_m)
    travel = travel - n_sh * dz_m
    if n_sh <= 0:
        return {
            "v_m_por_ano": v,
            "time_step_s": dur * SEG_POR_ANO,
            "shifts": 1,
            "dt_shift_anos": dur,
            "flow_direction": "diffusion_only",
        }, travel
    dt = dur / n_sh
    return {
        "v_m_por_ano": v,
        "time_step_s": dt * SEG_POR_ANO,
        "shifts": n_sh,
        "dt_shift_anos": dt,
        "flow_direction": "forward",
    }, travel


def ventanas_nodos_csv(malla: str, tasa: int,
                       t_max: float | None = None) -> list[dict]:
    """Un paso TRANSPORT por intervalo del Excel (P/Sg), 184 nodos → 183 pasos."""
    t = nodos_tiempo_csv(malla, tasa)
    if t_max is not None:
        t = t[t <= t_max + 1e-9]
        if len(t) == 0:
            return []
        if t[-1] < t_max:
            t = np.append(t, t_max)
    t_sello = t_llegada_sello_anos(malla, tasa)
    t_sello = min(max(t_sello, T_FIN_EXTRACCION + 1.0), T_FIN_INYECCION)
    hops = hops_hacia_sello(malla, tasa)
    v_et = v_por_etapa(malla, tasa)
    travel = 0.0
    out: list[dict] = []
    # En hops del frente, el dz de shift es el del destino (1 celda / hop)
    dz_default = dz_shift_m(celdas_transport(malla, tasa))
    hop_dz = {f"frente_{k0}_{k1}": dz_m(k1) for k0, k1, _a, _b, _v in hops}
    for t_a, t_b in zip(t[:-1], t[1:]):
        dur = float(t_b - t_a)
        if dur <= 1e-12:
            continue
        t_mid = 0.5 * (float(t_a) + float(t_b))
        nombre = nombre_etapa(t_mid, t_sello, hops)
        v = v_et.get(nombre, 0.0)
        dz_sh = hop_dz.get(nombre, dz_default)
        p, travel = _paso_transport(dur, v, travel, dz_sh)
        out.append({"t_a": float(t_a), "t_b": float(t_b), "nombre": nombre, **p})
    return out

def filtrar_ventanas_corto(pasos: list[dict], anos_ext: float = 2.0,
                           anos_iny: float = 2.0) -> list[dict]:
    """Nodos CSV de extraccion hasta anos_ext y de inyeccion los primeros anos_iny."""
    t_iny_fin = T_FIN_EXTRACCION + anos_iny
    out = []
    for p in pasos:
        if p["t_b"] <= anos_ext + 1e-9:
            out.append(p)
        elif p["t_a"] + 1e-9 >= T_FIN_EXTRACCION and p["t_b"] <= t_iny_fin + 1e-9:
            out.append(p)
    return out


def parametros_transport(malla: str, tasa: int, t_ini_anos: float,
                         t_fin_anos: float, dz_shift: float | None = None) -> dict:
    """-time_step (s), -shifts y -flow_direction para [t_ini, t_fin]."""
    dur = max(t_fin_anos - t_ini_anos, 1e-9)
    # Si el intervalo coincide con un hop del frente, usar v exacta (DC-029)
    v = None
    dz_hop = None
    for k0, k1, t0, t1, v_hop in hops_hacia_sello(malla, tasa):
        if abs(t_ini_anos - t0) < 0.6 and abs(t_fin_anos - t1) < 0.6:
            v = v_hop
            dz_hop = dz_m(k1)
            break
        # intervalo contenido en el hop (nodos CSV)
        if t_ini_anos + 1e-9 >= t0 and t_fin_anos - 1e-9 <= t1:
            v = v_hop
            dz_hop = dz_m(k1)
            break
    if v is None:
        v = velocidad_ventana_m_por_ano(malla, tasa, t_ini_anos, t_fin_anos)
    rama = celdas_transport(malla, tasa)
    dz_sh = dz_shift if dz_shift is not None else (dz_hop if dz_hop is not None else dz_shift_m(rama))
    p, _ = _paso_transport(dur, v, 0.0, dz_sh)
    if p["flow_direction"] == "forward":
        return p
    n = max(1, int(math.ceil(dur / DT_KIN_MAX_ANOS)))
    dt = dur / n
    return {
        "v_m_por_ano": 0.0,
        "time_step_s": dt * SEG_POR_ANO,
        "shifts": n,
        "dt_shift_anos": dt,
        "flow_direction": "diffusion_only",
    }


def dispersividad_inicial_m(dz_m: float = DZ_M) -> float:
    """Semilla a_L = dz/10 (mezcla mecanica)."""
    return dz_m / 10.0

"""Permeabilidad de roca a partir del cambio mineralogico (DC-024 / DC-026 / DC-027).

phi sale de PHREEQC: delta_phi = -(V_min - V_min_ref) / V_bulk (DC-010 / DC-021).

1. Potencia (KC simplificado):
       K/K0 = (phi / phi0)^3

2. Carman-Kozeny completo:
       K/K0 = (phi / phi0)^3 * ((1 - phi0) / (1 - phi))^2

K0 Cirrus: sello k=10 vs reservorio. phi0: receta de la celda.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ccus_columna.constantes import (
    K_RES_MD,
    K_RES_MD_REFINADA_50K,
    TASA_REFINADA_K_ESPECIAL,
    K_SELLO,
    K_SELLO_MD,
    K_SELLO_MD_REFINADA_50K,
)
from ccus_columna.mineralogia import phi0
from ccus_columna.postproceso import _fila_final_paso, calcular_delta_phi, leer_selected_output

SIW_DEFAULT = 0.203
COREY_N_DEFAULT = 2.0
KRW_RATIO_MAX_DEFAULT = 10.0
SG_MIN_KRW_DEFAULT = 1e-3
SW_REF_MARGIN_DEFAULT = 0.05
KRW_REF_FLOOR_DEFAULT = 1e-3


def k0_md(malla: str, tasa: int, k: int) -> float:
    if malla == "malla_refinada" and tasa == TASA_REFINADA_K_ESPECIAL:
        return K_SELLO_MD_REFINADA_50K if k == K_SELLO else K_RES_MD_REFINADA_50K
    return K_SELLO_MD if k == K_SELLO else K_RES_MD


def ratio_kozeny_carman(phi: float, phi0_val: float, *, completo: bool = False) -> float:
    """K/K0 por alteracion de porosidad (minerales)."""
    phi = max(float(phi), 1e-5)
    p0 = max(float(phi0_val), 1e-5)
    r = phi / p0
    if not completo:
        return r ** 3
    return r ** 3 * ((1.0 - p0) / (1.0 - phi)) ** 2


def _sw_eff(sw: float, swc: float, sgr: float = 0.0) -> float:
    denom = max(1.0 - swc - sgr, 1e-6)
    return max(0.0, min(1.0, (float(sw) - swc) / denom))


def _sw_norm(sw: float, siw: float) -> float:
    return _sw_eff(sw, swc=siw, sgr=0.0)


def krw_corey(
    sw: float,
    swc: float = SIW_DEFAULT,
    sgr: float = 0.0,
    n: float = COREY_N_DEFAULT,
) -> float:
    """krw Corey (permeabilidad relativa al agua; no es K de roca)."""
    return _sw_eff(sw, swc, sgr) ** float(n)


def krw_brooks_corey(
    sw: float,
    swc: float = SIW_DEFAULT,
    sgr: float = 0.0,
    lambda_bc: float = 0.5,
) -> float:
    lam = max(float(lambda_bc), 1e-6)
    exp = (2.0 + 3.0 * lam) / lam
    return _sw_eff(sw, swc, sgr) ** exp


def krw_wyllie(sw: float, siw: float = SIW_DEFAULT) -> float:
    return _sw_norm(sw, siw) ** 4


def krg_wyllie(sg: float, siw: float = SIW_DEFAULT) -> float:
    sg = max(float(sg), 0.0)
    denom = max(1.0 - siw, 1e-6)
    term1 = sg / denom
    term2 = max(0.0, (1.0 - sg - siw) / denom)
    return max(0.0, term1 ** 2 * (1.0 - term2 ** 2))


def _krw_ratio_corey(
    sw: float,
    sw_ref: float,
    sg: float,
    *,
    siw: float = SIW_DEFAULT,
    n: float = COREY_N_DEFAULT,
    sg_min: float = SG_MIN_KRW_DEFAULT,
    sw_ref_margin: float = SW_REF_MARGIN_DEFAULT,
    krw_ref_floor: float = KRW_REF_FLOOR_DEFAULT,
    max_ratio: float = KRW_RATIO_MAX_DEFAULT,
) -> float:
    if float(sg) <= sg_min:
        return 1.0
    krw = krw_corey(sw, swc=siw, n=n)
    krw_ref = krw_corey(sw_ref, swc=siw, n=n)
    if sw_ref <= siw + sw_ref_margin:
        krw_ref_eff = max(krw_ref, krw_ref_floor)
    else:
        krw_ref_eff = max(krw_ref, 1e-12)
    if krw_ref_eff <= 1e-12 and krw <= 1e-12:
        return 1.0
    ratio = krw / krw_ref_eff
    return min(max(ratio, 1.0 / max_ratio), max_ratio)


def ratio_kc_corey(
    phi: float,
    phi_ref: float,
    sw: float,
    sw_ref: float,
    sg: float,
    *,
    siw: float = SIW_DEFAULT,
    n: float = COREY_N_DEFAULT,
    sg_min: float = SG_MIN_KRW_DEFAULT,
    sw_ref_margin: float = SW_REF_MARGIN_DEFAULT,
    krw_ref_floor: float = KRW_REF_FLOOR_DEFAULT,
    max_ratio: float = KRW_RATIO_MAX_DEFAULT,
) -> float:
    """K efectiva (KC x krw). No usar para K de roca."""
    k_phi = ratio_kozeny_carman(phi, phi_ref, completo=False)
    r_krw = _krw_ratio_corey(
        sw, sw_ref, sg,
        siw=siw, n=n, sg_min=sg_min,
        sw_ref_margin=sw_ref_margin,
        krw_ref_floor=krw_ref_floor,
        max_ratio=max_ratio,
    )
    return k_phi * r_krw


def ratio_kc_sw(
    phi: float,
    phi_ref: float,
    sw: float,
    sw_ref: float,
    *,
    siw: float = SIW_DEFAULT,
    sg: float = 0.0,
    **kwargs,
) -> float:
    return ratio_kc_corey(phi, phi_ref, sw, sw_ref, sg, siw=siw, **kwargs)


def delta_k_pct(ratio: float) -> float:
    """Cambio de permeabilidad [%]: 100 * (K/K0 - 1)."""
    return 100.0 * (float(ratio) - 1.0)


def tabla_delta_permeabilidad(
    run_dir: Path,
    malla: str,
    tasa: int,
    ks: list[int],
) -> pd.DataFrame:
    """Serie (t, k) de K de roca: potencia y Carman-Kozeny completo."""
    run_dir = Path(run_dir)
    pasos_map: dict[int, float] = {}
    if (run_dir / "pasos.json").exists():
        raw = json.loads((run_dir / "pasos.json").read_text(encoding="utf-8"))
        pasos_map = {int(p["paso"]): float(p["t_fin"]) for p in raw}

    filas = []
    pat = re.compile(r"k(\d+)_p(\d+)")
    for path in sorted((run_dir / "hitos").glob("k*_p*.txt")):
        m = pat.search(path.name)
        if not m:
            continue
        k, paso = int(m.group(1)), int(m.group(2))
        if k not in ks:
            continue
        t = pasos_map.get(paso, float("nan"))
        if not np.isfinite(t):
            continue
        row = _fila_final_paso(leer_selected_output(path))
        p0 = phi0(k)
        dphi = calcular_delta_phi(row, k=k, malla=malla, incluir_cinetica=True)
        phi = p0 + dphi
        k0 = k0_md(malla, tasa, k)
        r_kc = ratio_kozeny_carman(phi, p0, completo=False)
        r_full = ratio_kozeny_carman(phi, p0, completo=True)

        filas.append({
            "tiempo_anos": t,
            "k": k,
            "phi": phi,
            "phi0": p0,
            "dphi": dphi,
            "dphi_pct": 100.0 * dphi / p0 if p0 > 0 else 0.0,
            "K0_md": k0,
            "K_kc_md": k0 * r_kc,
            "K_kc_full_md": k0 * r_full,
            "ratio_kc": r_kc,
            "ratio_kc_full": r_full,
            "K_pct_kc": 100.0 * r_kc,
            "K_pct_kc_full": 100.0 * r_full,
            "dK_pct_kc": delta_k_pct(r_kc),
            "dK_pct_kc_full": delta_k_pct(r_full),
        })
    return pd.DataFrame(filas).sort_values(["tiempo_anos", "k"])

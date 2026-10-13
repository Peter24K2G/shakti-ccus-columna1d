"""Mineralogia por celda: reservorio (D-020) vs sello k=10 (XRD Esmeraldas, DC-011).

Conversion wt% XRD -> moles PHREEQC:

    1. Promedio aritmetico de las muestras (celdas vacias = 0).
    2. Microclina + ortoclasa -> K-feldspar; esmectita -> Ca-Montmorillonite;
       moscovita -> K-mica (nombre de phreeqc.dat; no existe "Muscovite").
    3. Normalizar a suma 100 % (la fraccion no reportada no se modela).
    4. rho_grano = 1 / sum(w_i / rho_i),  rho_i = MW_i / Vm_i
    5. masa_solida = rho_grano * V_bulk * (1 - phi)
    6. moles_i = w_i * masa_solida / MW_i

V_bulk PHREEQC: 1 L = celda original 50x50x20 m; las demas celdas en razon
geometrica (DC-015). v_bulk_m3() es el volumen Cirrus real.
Unidades: MW en g/mol, Vm en cm3/mol, V_bulk en L, masa en g.
"""

from __future__ import annotations

from ccus_columna.constantes import (
    K_SELLO,
    PHI,
    PHI_SELLO,
    RHO_BRINA,
    V_BULK_RECETA_L,
)
from ccus_columna.geometria import escala_volumen, v_bulk_l
from ccus_columna.parametros import P as _P

# XRD Fm. Esmeraldas (sello): % peso por muestra (datos del proyecto, YAML externo).
XRD_ESMERALDAS: tuple[dict[str, float], ...] = tuple(
    {k: float(v) for k, v in m.items()} for m in _P["xrd_sello"]
)

# XRD -> fase PHREEQC (phreeqc.dat / PHASES propias)
XRD_A_PHREEQC: dict[str, str] = {
    "Cuarzo": "Quartz",
    "Microclina": "K-feldspar",
    "Ortoclasa": "K-feldspar",
    "Albita": "Albite",
    "Calcita": "Calcite",
    "Dolomita": "Dolomite",
    "Pirita": "Pyrite",
    "Ilita": "Illite",
    "Esmectita": "Ca-Montmorillonite",
    "Clorita": "Chlorite(14A)",
    "Caolinita": "Kaolinite",
    "Moscovita": "K-mica",
}

# MW g/mol, Vm cm3/mol (Helgeson / Robie; Vm cinetica alineada a D-021)
PROPS: dict[str, tuple[float, float]] = {
    "Quartz": (60.0843, 22.688),
    "K-feldspar": (278.3315, 108.72),
    "Albite": (262.2230, 100.07),
    "Calcite": (100.0869, 36.934),
    "Dolomite": (184.4009, 64.34),
    "Pyrite": (119.975, 23.94),
    "Illite": (389.34, 140.71),
    "Ca-Montmorillonite": (367.0, 130.0),
    "Chlorite(14A)": (714.0, 142.0),
    "Kaolinite": (258.1604, 99.52),
    "K-mica": (398.3079, 140.71),
}

EQ_FASES = ("Calcite", "Pyrite", "Dawsonite", "Ankerite", "Gypsum", "Dolomite")

# Inventarios del reservorio (moles receta 1 L, D-020): datos del proyecto (YAML externo).
INVENTARIO_EQ_RESERVORIO: dict[str, float] = {
    k: float(v) for k, v in _P["inventario_eq_reservorio"].items()
}
INVENTARIO_KIN_RESERVORIO: dict[str, float] = {
    k: float(v) for k, v in _P["inventario_kin_reservorio"].items()
}

def es_sello(k: int | None) -> bool:
    return k is not None and int(k) == int(K_SELLO)


def phi0(k: int | None = None) -> float:
    return PHI_SELLO if es_sello(k) else PHI


def m_water_ref(k: int | None = None, malla: str | None = None) -> float:
    return phi0(k) * RHO_BRINA * v_bulk_l(malla, k)


def v_poro(k: int | None = None, malla: str | None = None) -> float:
    return phi0(k) * v_bulk_l(malla, k)


def _escala(malla: str | None, k: int | None) -> float:
    return escala_volumen(malla, k)


def wt_promedio_xrd(muestras: tuple[dict[str, float], ...] = XRD_ESMERALDAS
                    ) -> dict[str, float]:
    """Promedio de muestras XRD, mapeado a fases PHREEQC y normalizado a 1."""
    claves = list(XRD_A_PHREEQC)
    acc: dict[str, float] = {XRD_A_PHREEQC[c]: 0.0 for c in claves}
    n = len(muestras)
    for muestra in muestras:
        for col, fase in XRD_A_PHREEQC.items():
            acc[fase] += float(muestra.get(col, 0.0) or 0.0) / n
    total = sum(acc.values())
    if total <= 0:
        raise ValueError("XRD sin minerales")
    return {fase: w / total for fase, w in acc.items() if w > 0}


def moles_desde_wt(wt: dict[str, float], phi: float,
                   v_bulk_l: float = V_BULK_RECETA_L) -> dict[str, float]:
    """wt (fraccion 0-1) -> moles en el volumen solido (1-phi)*V_bulk."""
    v_solido_cm3 = (1.0 - phi) * v_bulk_l * 1000.0
    inv_rho = 0.0
    for fase, w in wt.items():
        if fase not in PROPS or w <= 0:
            continue
        mw, vm = PROPS[fase]
        rho = mw / vm  # g/cm3
        inv_rho += w / rho
    rho_grano = 1.0 / inv_rho
    masa_g = rho_grano * v_solido_cm3
    moles: dict[str, float] = {}
    for fase, w in wt.items():
        if fase not in PROPS or w <= 0:
            continue
        mw, _vm = PROPS[fase]
        moles[fase] = w * masa_g / mw
    return moles


def _moles_sello() -> dict[str, float]:
    return moles_desde_wt(wt_promedio_xrd(), PHI_SELLO)


MOLES_SELLO: dict[str, float] = _moles_sello()


def inventario_eq(k: int | None = None, malla: str | None = None) -> dict[str, float]:
    s = _escala(malla, k)
    if not es_sello(k):
        return {fase: n * s for fase, n in INVENTARIO_EQ_RESERVORIO.items()}
    inv = {fase: 0.0 for fase in EQ_FASES}
    inv["Calcite"] = MOLES_SELLO.get("Calcite", 0.0) * s
    inv["Dolomite"] = MOLES_SELLO.get("Dolomite", 0.0) * s
    inv["Pyrite"] = MOLES_SELLO.get("Pyrite", 0.0) * s
    return inv


def inventario_kin(k: int | None = None, malla: str | None = None) -> dict[str, float]:
    s = _escala(malla, k)
    if not es_sello(k):
        return {m: n * s for m, n in INVENTARIO_KIN_RESERVORIO.items() if n > 0}
    skip = {"Calcite", "Dolomite", "Pyrite"}
    return {m: n * s for m, n in MOLES_SELLO.items() if m not in skip and n > 0}


def resumen_sello(malla: str | None = None) -> str:
    wt = wt_promedio_xrd()
    inv = inventario_eq(K_SELLO, malla) | inventario_kin(K_SELLO, malla)
    v = v_bulk_l(malla, K_SELLO)
    lineas = [
        f"Sello k={K_SELLO}: Fm. Esmeraldas, phi0={PHI_SELLO}, V_bulk={v:.4g} L",
        "wt% promedio (2 muestras, normalizado):",
    ]
    for fase, w in sorted(wt.items(), key=lambda x: -x[1]):
        mol = inv.get(fase, 0.0)
        lineas.append(f"  {fase:22s}  {100 * w:6.2f} wt%   {mol:8.4g} mol")
    return "\n".join(lineas)

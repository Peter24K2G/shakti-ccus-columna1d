"""Parametros y RATES Palandri & Kharaka (2004) para minerales del reactor La Paz.

Fuente: Palandri & Kharaka (2004), USGS OFR 2004-1068; valores alineados con
`phreeqc_rates.dat` (PHREEQC 3.8.6). Implementacion via RATE_PARAMETERS_PK +
funcion BASIC RATE_PK (sin reimplementar la ley en BASIC).

RSA (PARM 1): area reactiva [m2] calibrada al reactor batch 1 L.
"""

from __future__ import annotations

from dataclasses import dataclass

PK_REF = "Palandri & Kharaka (2004), USGS OFR 2004-1068; phreeqc_rates.dat 3.8.6"
CALCITE_RSA_RAPIDA_MULT = 10.0  # D-021: RSA x10 (x100 agresivo; usar con cautela)


@dataclass(frozen=True)
class MineralPK:
    """Parametros PK2004 + inventario del reactor."""

    m: float
    area: float
    rsa_nota: str
    log_k_acid: float
    Ea_acid: float
    n_H: float
    log_k_neut: float
    Ea_neut: float
    log_k_basic: float = -30.0
    Ea_basic: float = 0.0
    n_OH: float = 0.0
    tabla_pk: str = "PK2004"

    def parms_pqi(self) -> str:
        """Linea -parms KINETICS: solo RSA [m2] (constantes en RATE_PARAMETERS_PK)."""
        return f"{self.area:.6g}"

    def linea_rate_parameters(self, mineral: str) -> str:
        name = _RATE_PK_NAMES.get(mineral, mineral)
        return (
            f"{name}  "
            f"{self.log_k_acid:.6g} {self.Ea_acid:.6g} {self.n_H:.6g}  "
            f"{self.log_k_neut:.6g} {self.Ea_neut:.6g}  "
            f"{self.log_k_basic:.6g} {self.Ea_basic:.6g} {self.n_OH:.6g}"
        )


# Clave KINETICS/PHASES -> nombre en RATE_PK (si difiere)
_RATE_PK_NAMES: dict[str, str] = {
    "Ca-Montmorillonite": "montmorillonite",
    "Illite": "Illite",
}


from ccus_columna.parametros import P as _P

# m [mol, receta 1 L] y RSA calibrada [m2] son datos del proyecto (YAML externo).
_INV = _P["cinetica_inventario"]

MINERALES_PK: dict[str, MineralPK] = {
    "Quartz": MineralPK(
        m=_INV["Quartz"]["m"], area=_INV["Quartz"]["area"],
        rsa_nota="RSA calibrada reactor 1 L; Tabla 4 PK2004",
        log_k_acid=-30.0, Ea_acid=0.0, n_H=0.0,
        log_k_neut=-13.4, Ea_neut=90.9,
        tabla_pk="Tabla 4 (cuarzo)",
    ),
    "K-feldspar": MineralPK(
        m=_INV["K-feldspar"]["m"], area=_INV["K-feldspar"]["area"],
        rsa_nota="RSA calibrada; Tabla 15 PK2004",
        log_k_acid=-10.06, Ea_acid=51.7, n_H=0.5,
        log_k_neut=-12.41, Ea_neut=38.0,
        log_k_basic=-21.2, Ea_basic=94.1, n_OH=-0.823,
        tabla_pk="Tabla 15 (K-feldspar)",
    ),
    "Albite": MineralPK(
        m=_INV["Albite"]["m"], area=_INV["Albite"]["area"],
        rsa_nota="RSA calibrada; Tabla 1 PK2004",
        log_k_acid=-10.16, Ea_acid=65.0, n_H=0.317,
        log_k_neut=-12.56, Ea_neut=65.0,
        log_k_basic=-15.6, Ea_basic=66.5, n_OH=-0.471,
        tabla_pk="Tabla 1 (albita)",
    ),
    "Chlorite(14A)": MineralPK(
        m=_INV["Chlorite(14A)"]["m"], area=_INV["Chlorite(14A)"]["area"],
        rsa_nota="RSA calibrada; chlorite(14A) PK2004",
        log_k_acid=-11.11, Ea_acid=88.0, n_H=0.5,
        log_k_neut=-12.52, Ea_neut=88.0,
        tabla_pk="Tabla 30 (clorita)",
    ),
    "Illite": MineralPK(
        m=_INV["Illite"]["m"], area=_INV["Illite"]["area"],
        rsa_nota="RSA calibrada; proxy muscovita PK2004 Tabla 28",
        log_k_acid=-11.85, Ea_acid=22.0, n_H=0.37,
        log_k_neut=-13.55, Ea_neut=22.0,
        log_k_basic=-14.55, Ea_basic=22.0, n_OH=-0.22,
        tabla_pk="Tabla 28 (muscovita, proxy illita)",
    ),
    "Kaolinite": MineralPK(
        m=_INV["Kaolinite"]["m"], area=_INV["Kaolinite"]["area"],
        rsa_nota="RSA calibrada; Tabla 29 PK2004",
        log_k_acid=-11.31, Ea_acid=65.9, n_H=0.777,
        log_k_neut=-13.18, Ea_neut=22.2,
        log_k_basic=-17.05, Ea_basic=17.9, n_OH=-0.472,
        tabla_pk="Tabla 29 (caolinita)",
    ),
    "Ca-Montmorillonite": MineralPK(
        m=_INV["Ca-Montmorillonite"]["m"], area=_INV["Ca-Montmorillonite"]["area"],
        rsa_nota="RSA calibrada; montmorillonita PK2004 Tabla 29",
        log_k_acid=-12.71, Ea_acid=48.0, n_H=0.22,
        log_k_neut=-14.41, Ea_neut=48.0,
        log_k_basic=-14.41, Ea_basic=48.0, n_OH=-0.13,
        tabla_pk="Tabla 29 (montmorillonita)",
    ),
    "Siderite": MineralPK(
        m=_INV["Siderite"]["m"], area=_INV["Siderite"]["area"],
        rsa_nota="RSA calibrada; carbonato Fe (extrap. PK/Chou 1984)",
        log_k_acid=-8.94, Ea_acid=53.1, n_H=0.5,
        log_k_neut=-10.75, Ea_neut=79.9,
        tabla_pk="Carbonato hierro (no tabulado PK)",
    ),
    "K-mica": MineralPK(
        m=_INV["K-mica"]["m"], area=_INV["K-mica"]["area"],
        rsa_nota="RSA especifica = illita (m2/mol); Tabla 28 PK2004 muscovita",
        log_k_acid=-11.85, Ea_acid=22.0, n_H=0.37,
        log_k_neut=-13.55, Ea_neut=22.0,
        log_k_basic=-14.55, Ea_basic=22.0, n_OH=-0.22,
        tabla_pk="Tabla 28 (muscovita); fase phreeqc.dat = K-mica",
    ),
    "Calcite": MineralPK(
        m=_INV["Calcite"]["m"], area=_INV["Calcite"]["area"],
        rsa_nota="RSA calibrada reactor 1 L (D-007); Tabla 7 PK2004",
        log_k_acid=-5.19, Ea_acid=23.5, n_H=1.0,
        log_k_neut=-10.48, Ea_neut=35.4,
        tabla_pk="Tabla 7 (calcita)",
    ),
}

# Nombres en RATE_PARAMETERS_PK (deben coincidir con RATE_PK argument)
for _k, _v in list(MINERALES_PK.items()):
    if _k == "Ca-Montmorillonite":
        _RATE_PK_NAMES[_k] = "montmorillonite"
    elif _k == "Chlorite(14A)":
        _RATE_PK_NAMES[_k] = "chlorite(14A)"
    elif _k == "Illite":
        _RATE_PK_NAMES[_k] = "Illite"
    else:
        _RATE_PK_NAMES[_k] = _k


def rsa_m2(mineral: str, moles: float) -> float:
    """Area reactiva [m2] escalada con el inventario (misma RSA especifica 0D)."""
    p = MINERALES_PK[mineral]
    if moles <= 0:
        return 0.0
    if p.m > 0:
        return p.area * (moles / p.m)
    return p.area * moles  # p.area ya es m2/mol (K-mica)


def kinetics_items_inventario(moles: dict[str, float],
                              calcite_cinetica: bool = False,
                              calcite_rsa_mult: float = 1.0) -> str:
    """Items KINETICS a partir de un inventario {mineral: moles}."""
    lineas = []
    for m, p in MINERALES_PK.items():
        if m == "Calcite" and not calcite_cinetica:
            continue
        n = float(moles.get(m, 0.0))
        if n <= 0:
            continue
        area = rsa_m2(m, n)
        if m == "Calcite":
            area *= calcite_rsa_mult
        lineas.append(f"    {m}\n        -m {n:.6g}; -parms {area:.6g}")
    return "\n".join(lineas)


def rate_parameters_pqi(excluir: set[str] | None = None,
                        calcite_cinetica: bool = False,
                        calcite_rsa_mult: float = 1.0) -> str:
    """Bloque RATE_PARAMETERS_PK para el .pqi."""
    if excluir is None:
        excluir = set() if calcite_cinetica else {"Calcite"}
    elif not calcite_cinetica:
        excluir = set(excluir) | {"Calcite"}
    lineas = [
        "RATE_PARAMETERS_PK",
        "# Acid log_k E nH+ | Neutral log_k E | Base log_k E nOH-",
        f"# Fuente: {PK_REF}",
    ]
    for m, p in MINERALES_PK.items():
        if m in excluir:
            continue
        area = p.area * (calcite_rsa_mult if m == "Calcite" else 1.0)
        lineas.append(f"# {m}: RSA={area} m2; {p.tabla_pk}; {p.rsa_nota}")
        lineas.append(
            f"{_RATE_PK_NAMES.get(m, m)}  "
            f"{p.log_k_acid:.6g} {p.Ea_acid:.6g} {p.n_H:.6g}  "
            f"{p.log_k_neut:.6g} {p.Ea_neut:.6g}  "
            f"{p.log_k_basic:.6g} {p.Ea_basic:.6g} {p.n_OH:.6g}"
        )
    return "\n".join(lineas)


_RATE_PK_BODY = """{mineral}
-start
 10 ratepk = RATE_PK("{rate_name}")
 20 area = PARM(1)
 30 SRmin = SR("{mineral}")
 40 rate = ratepk * area * (1 - SRmin)
 50 moles = rate * TIME
 60 IF (moles > M) THEN moles = M
 70 IF (moles < -M) THEN moles = -M
 80 SAVE moles
-end"""


def rate_block(mineral: str) -> str:
    if mineral not in MINERALES_PK:
        raise KeyError(f"Mineral {mineral} sin parametros PK2004")
    return _RATE_PK_BODY.format(
        mineral=mineral,
        rate_name=_RATE_PK_NAMES[mineral],
    )


def rates_pqi(excluir: set[str] | None = None,
              calcite_cinetica: bool = False) -> str:
    if excluir is None:
        excluir = set() if calcite_cinetica else {"Calcite"}
    elif not calcite_cinetica:
        excluir = set(excluir) | {"Calcite"}
    return "\n\n".join(
        rate_block(m) for m in MINERALES_PK if m not in excluir
    )


def bloques_cinetica_pk(excluir: set[str] | None = None,
                        calcite_cinetica: bool = False,
                        calcite_rsa_mult: float = 1.0) -> tuple[str, str, str]:
    """RATE_PARAMETERS_PK, RATES, items KINETICS."""
    if excluir is None:
        excluir = set() if calcite_cinetica else {"Calcite"}
    elif not calcite_cinetica:
        excluir = set(excluir) | {"Calcite"}
    items = kinetics_items_inventario(
        {m: p.m for m, p in MINERALES_PK.items() if m not in excluir},
        calcite_cinetica=True,  # Calcite ya filtrada en excluir
        calcite_rsa_mult=calcite_rsa_mult,
    )
    return (
        rate_parameters_pqi(excluir, calcite_cinetica, calcite_rsa_mult),
        rates_pqi(excluir, calcite_cinetica),
        items,
    )


def kinetics_items_pqi(excluir: set[str] | None = None,
                       calcite_cinetica: bool = False,
                       calcite_rsa_mult: float = 1.0) -> str:
    return bloques_cinetica_pk(excluir, calcite_cinetica, calcite_rsa_mult)[2]


CINETICA_CALIBRADA = {
    m: {"m": p.m, "area": p.area, "ea": p.Ea_neut * 1000,
        "k25": 10 ** p.log_k_neut}
    for m, p in MINERALES_PK.items()
}

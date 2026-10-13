"""Bloques PHREEQC comunes a ambas metodologias.

Quimica identica a la fase 0D (D-020): agua nativa calibrada de la Fm. La Paz,
fases secundarias no estandar (dawsonite, ankerita) y el inventario mineral del
reactor. Cinetica Palandri & Kharaka 2004 en ccus_columna.kinetics_pk2004.
"""

from __future__ import annotations

from ccus_columna import kinetics_pk2004 as pk
from ccus_columna import mineralogia
from ccus_columna.parametros import P

# Agua nativa calibrada (Fm. La Paz): composicion = dato del proyecto (YAML
# externo, seccion "brina"). {numero}, {nombre}, {agua_kg}, {presion_atm} se
# formatean por celda/paso.
_B = P["brina"]


def _lineas_brina(b: dict = _B) -> str:
    out = [
        f"    temp      {float(b['temp_c']):.1f}",
        f"    pH        {b['pH']}",
        f"    pe        {b['pe']}",
        "    units     mg/L",
        f"    density   {b['density']}",
    ]
    for el, val in b["mg_l"].items():
        out.append(f"    {el:9s} {val}")
    return "\n".join(out)


AGUA_NATIVA = ("SOLUTION {numero} {nombre}\n"
               + _lineas_brina().replace("{", "{{").replace("}", "}}")
               + "\n    -water    {agua_kg:.6f}\n    -pressure {presion_atm:.2f}")

# Fases no incluidas en phreeqc.dat
FASES_SECUNDARIAS = """PHASES
Dawsonite
    NaAlCO3(OH)2 + 3 H+ = Al+3 + HCO3- + Na+ + 2 H2O
    log_k 4.3464
    -delta_h -76.3549 kJ/mol
    -analytic -113.93 -0.023487 7175.8 40.9 121.89
Ankerite
    CaFe(CO3)2 = Ca+2 + Fe+2 + 2 CO3-2
    log_k -20.8732
    -analytic 6743.141 2.308961 -252723.633 -2681.493 9.661065e6 -0.000880753
    -Vm 66.06"""

# Inventario de equilibrio por celda (calcita = buffer primario, D-020).
# Preferir equilibrium_rocas(k): el sello usa XRD Esmeraldas (DC-011).
EQUILIBRIUM_ROCAS = "EQUILIBRIUM_PHASES {numero} Rocas_Rapidas\n" + "\n".join(
    f"    {fase:16s}  0.0     {mol}" for fase, mol in P["inventario_eq_reservorio"].items()
)

SI_EXPORT = (
    "Calcite Siderite Dolomite Pyrite Gypsum CO2(g) Quartz Dawsonite Ankerite "
    "K-feldspar Albite Kaolinite Illite Chlorite(14A) Ca-Montmorillonite K-mica"
)

EXP_EQ = "Calcite Pyrite Dawsonite Ankerite Gypsum Dolomite"
EXP_KIN = (
    "Siderite Quartz K-feldspar Albite Chlorite(14A) Illite "
    "Kaolinite Ca-Montmorillonite K-mica"
)


def solucion_nativa(numero: int | str, nombre: str, agua_kg: float,
                    presion_atm: float = 100.0) -> str:
    return AGUA_NATIVA.format(numero=numero, nombre=nombre, agua_kg=agua_kg,
                              presion_atm=presion_atm)


def equilibrium_rocas(numero: int | str, k: int | None = None,
                      malla: str | None = None) -> str:
    """EQUILIBRIUM_PHASES del inventario de la celda k (DC-011 / DC-015)."""
    inv = mineralogia.inventario_eq(k, malla)
    lineas = [f"EQUILIBRIUM_PHASES {numero} Rocas_k{k if k is not None else numero}"]
    for fase in mineralogia.EQ_FASES:
        lineas.append(f"    {fase:16s}  0.0     {inv.get(fase, 0.0):.6g}")
    return "\n".join(lineas)


def bloques_roca() -> tuple[str, str, str]:
    """RATE_PARAMETERS_PK, RATES (cuerpo), items KINETICS (indentados)."""
    return pk.bloques_cinetica_pk(calcite_cinetica=False)


def prefijo_definiciones(convergence_tol: float = 1e-12) -> str:
    """PHASES + tasas PK + KNOBS (necesario en cada .pqi batch independiente)."""
    rate_params, rates, _ = bloques_roca()
    return f"""{FASES_SECUNDARIAS}

{rate_params}

RATES
{rates}

KNOBS
    -pe_step_size    5
    -diagonal_scale  true
    -convergence_tolerance  {convergence_tol:.0e}
"""


def bloque_kinetics_inicial(numero: int | str = 1, k: int | None = None,
                            cvode: bool = False,
                            cvode_steps: int = 5000,
                            bad_step_max: int = 20000,
                            rapido: bool = False,
                            malla: str | None = None) -> str:
    """KINETICS inicial. met2 reactivo usa cvode=True (RK se cuelga en años)."""
    items = pk.kinetics_items_inventario(mineralogia.inventario_kin(k, malla))
    if cvode:
        flags = (
            "    -cvode\n"
            f"    -cvode_steps    {cvode_steps}\n"
            f"    -bad_step_max   {bad_step_max}\n"
        )
    elif rapido:
        flags = "    -runge_kutta   3\n    -bad_step_max  5\n"
    else:
        flags = ""
    return f"""KINETICS {numero} Rocas_Cineticas
{flags}{items}
"""


def selected_output_batch(archivo: str) -> str:
    return f"""SELECTED_OUTPUT 1
    -file           {archivo}
    -reset          false
    -time           true
    -solution       true
    -pH             true
    -alkalinity     true
    -totals         Na Mg Ca K Fe Cl S(6) C(4)
    -equilibrium    {EXP_EQ}
    -kinetics       {EXP_KIN}
    -si             {SI_EXPORT}
"""


def selected_output_transport(archivo: str, con_kinetics: bool = True) -> str:
    kin = f"    -kinetics       {EXP_KIN}\n" if con_kinetics else ""
    eq = ("    -equilibrium_phases Calcite Dawsonite Ankerite Gypsum Pyrite Dolomite\n"
          if con_kinetics else "")
    return f"""SELECTED_OUTPUT 1
    -file           {archivo}
    -reset          false
    -distance       true
    -time           true
    -step           true
    -solution       true
    -pH             true
    -alkalinity     true
    -totals         Na Mg Ca Fe Cl C(4)
{eq}{kin}    -si             Calcite Dawsonite Ankerite Gypsum CO2(g)
"""


def agua_nativa_default(numero: int | str, nombre: str,
                        presion_atm: float = 100.0,
                        k: int | None = None,
                        malla: str | None = None) -> str:
    return solucion_nativa(numero, nombre, mineralogia.m_water_ref(k, malla),
                           presion_atm)

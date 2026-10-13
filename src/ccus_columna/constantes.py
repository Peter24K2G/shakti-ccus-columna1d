"""Constantes de la fase columna 1D (heredadas de la fase 0D donde aplica)."""

from __future__ import annotations

from pathlib import Path

from ccus_columna.parametros import P

_R = P["roca"]; _E = P["etapas"]; _C = P["columna"]; _D = P["darcy"]

# --- raiz del proyecto (CCUS/) ---
RAIZ = Path(__file__).resolve().parents[2]
DATOS_CSV = RAIZ / "datos" / "csv"

# --- reactor receta 1 L (D-001/D-021); los moles de celda se escalan con dx*dy*dz (DC-015) ---
V_BULK_RECETA_L = 1.0     # volumen de la receta D-020 / XRD
V_BULK_L = V_BULK_RECETA_L  # alias: tests y receta 1 L
PHI = float(_R["phi"])          # porosidad inicial reservorio (D-001)
PHI_SELLO = float(_R["phi_sello"])  # porosidad promedio sello (Fm. Esmeraldas, DC-011)
RHO_BRINA = float(_R["rho_brina_kg_l"])  # kg/L a T de reservorio
V_PORO_L = PHI * V_BULK_L
M_WATER_REF_KG = PHI * RHO_BRINA * V_BULK_L  # kg a Sw=1 (receta 1 L)
V_PORO_SELLO_L = PHI_SELLO * V_BULK_L
M_WATER_SELLO_KG = PHI_SELLO * RHO_BRINA * V_BULK_L

# --- etapas operativas Cirrus (anos; desde parametros) ---
T_FIN_EXTRACCION = float(_E["t_fin_extraccion_anos"])
T_FIN_INYECCION = float(_E["t_fin_inyeccion_anos"])
T_FIN = float(_E["t_fin_anos"])

# --- columna ---
K_SELLO = int(_C["k_sello"])     # celda del sello (tope de la columna)
K_FONDO = int(_C["k_fondo"])
K_INYECCION = int(_C["k_inyeccion"])  # celda donde primero aparece gas
# Intervalo perforado Cirrus (DC-017): inyeccion ~k15..k30; met2 extiende
# la columna 1D desde el fondo de gas (<=30) hasta el sello.
K_INYECCION_FONDO = int(_C["k_inyeccion_fondo"])
DZ_M = float(_C["dz_m"])         # espesor reservorio (k != sello)
DZ_SELLO_M = float(_C["dz_sello_m"])  # espesor sello (DC-014)
DX_MALLA_ORIGINAL_M = float(_C["dx_malla_original_m"])
DY_MALLA_ORIGINAL_M = float(_C["dy_malla_original_m"])
DX_MALLA_REFINADA_M = float(_C["dx_malla_refinada_m"])
DY_MALLA_REFINADA_M = float(_C["dy_malla_refinada_m"])
M3_A_L = 1000.0           # 1 m3 = 1000 L
# Difusion molecular aparente CO2(aq)/iones en brina ~60 C (m2/s).
# En 10 a: sqrt(2Dt) ~ 0.8 m; en 20 a ~ 1.1 m; en 70 a ~ 2.1 m  (<< dz).
# La mezcla entre celdas en inyeccion la marca la dispersividad a_L = 0.1 dz.
D_DIFUSION_M2S = 1.0e-9

# --- Darcy (portadas Cirrus; desde parametros) ---
K_RES_MD = float(_D["k_res_md"])
K_SELLO_MD = float(_D["k_sello_md"])
# Par K alternativo para un escenario particular de la malla refinada
TASA_REFINADA_K_ESPECIAL = int(_D["refinada_especial"]["tasa_m3d"])
K_RES_MD_REFINADA_50K = float(_D["refinada_especial"]["k_res_md"])
K_SELLO_MD_REFINADA_50K = float(_D["refinada_especial"]["k_sello_md"])
MD_A_M2 = 9.869233e-16
MU_BRINA_PA_S = 4.7e-4    # brina ~60 C
G_MS2 = 9.81
V_MIN_TRANSPORTE_M_A = 0.05  # por debajo: diffusion_only (DC-012)

# --- conversiones ---
BAR_A_ATM = 0.986923
SEG_POR_ANO = 86400 * 365.25

# --- umbrales ---
SG_UMBRAL_GAS = 1e-3      # Sg minimo para considerar que "hay gas" en la celda

TASAS_M3D = tuple(int(q) for q in P["escenarios"]["tasas_m3d"])
TASAS_POR_MALLA = {m: tuple(int(q) for q in v) for m, v in P["escenarios"]["tasas_por_malla"].items()}
TASA_REF = TASAS_M3D[0]   # escenario de referencia (por defecto en CLI)


def etiqueta_tasa(q: int) -> str:
    return f"{int(q) // 1000}k"
MALLAS = ("malla_original", "malla_refinada")

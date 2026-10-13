"""Libreria compartida de la fase columna 1D CCUS (Fm. La Paz).

Modulos:
    constantes         — escalas del reactor, etapas, rutas
    cli                — validar malla/tasa y resolver carpetas de corrida
    datos              — loader oficial de los CSV Cirrus
    quimica            — bloques PHREEQC comunes
    kinetics_pk2004    — tasas Palandri & Kharaka 2004
    mineralogia        — inventario por celda (reservorio D-020 / sello DC-011)
    geometria          — dx, dy, dz y V_bulk de celda Cirrus (DC-015)
    mapa_saturacion    — S -> (m_water, V_gas, P_CO2)
    postproceso        — CSV perfil_quimica y delta_phi
    rutas              — carpetas de figuras por tipo de corrida
    runner             — ejecucion de phreeqc.exe
"""

from ccus_columna import (  # noqa: F401
    cli,
    constantes,
    datos,
    geometria,
    kinetics_pk2004,
    mineralogia,
    postproceso,
    quimica,
    rutas,
    runner,
)
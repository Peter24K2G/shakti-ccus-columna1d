"""Carga de parametros del proyecto (datos de sitio) desde un YAML externo.

Los valores de sitio (Cirrus, XRD, inventarios minerales, composicion de brina,
etc.) NO se distribuyen con el codigo. Orden de busqueda:

1. variable de entorno ``SHAKTI_PARAMS`` (ruta a un YAML);
2. ``<raiz>/datos/parametros.yaml`` (ignorado por git);
3. ``<raiz>/config/parametros_ejemplo.yaml`` (valores FICTICIOS, solo para
   probar la instalacion; ``ES_EJEMPLO`` queda en True).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]
RUTA_EJEMPLO = RAIZ / "config" / "parametros_ejemplo.yaml"
RUTA_REAL = RAIZ / "datos" / "parametros.yaml"


def ruta_parametros() -> tuple[Path, bool]:
    env = os.environ.get("SHAKTI_PARAMS")
    if env:
        return Path(env), False
    if RUTA_REAL.exists():
        return RUTA_REAL, False
    return RUTA_EJEMPLO, True


RUTA, ES_EJEMPLO = ruta_parametros()
with open(RUTA, encoding="utf-8") as _f:
    P: dict = yaml.safe_load(_f)

if ES_EJEMPLO or P.get("ficticio", False):
    ES_EJEMPLO = True
    warnings.warn(
        f"Usando parametros de EJEMPLO (ficticios): {RUTA}. "
        "Defina SHAKTI_PARAMS o cree datos/parametros.yaml.",
        stacklevel=2,
    )

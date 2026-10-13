"""Verifica que el entorno esta listo para trabajar en el proyecto.

Uso: python scripts/verificar_entorno.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

OK, FAIL = "[ok]  ", "[FALTA]"


def check(nombre: str, cond: bool, ayuda: str = "") -> bool:
    print(f"{OK if cond else FAIL} {nombre}" + (f"  -> {ayuda}" if not cond and ayuda else ""))
    return cond


def main() -> int:
    todo_ok = True

    print("== Python y librerias ==")
    todo_ok &= check(f"python {sys.version.split()[0]} >= 3.10", sys.version_info >= (3, 10))
    for mod in ("pandas", "numpy", "openpyxl", "matplotlib", "pytest"):
        try:
            importlib.import_module(mod)
            check(mod, True)
        except ImportError:
            todo_ok &= check(mod, False, "pip install -r requirements.txt")

    print("\n== PHREEQC ==")
    from ccus_columna.runner import DATABASE, DATABASE_ALT, PHREEQC_EXE
    todo_ok &= check(f"phreeqc.exe ({PHREEQC_EXE})", PHREEQC_EXE.exists(),
                     "instalar PHREEQC 3.8.6 batch de USGS")
    todo_ok &= check("phreeqc.dat", DATABASE.exists() or DATABASE_ALT.exists())

    print("\n== Datos ==")
    xlsx = list(RAIZ.glob("*.xlsx"))
    todo_ok &= check(f"xlsx crudos en raiz ({len(xlsx)}/4)", len(xlsx) == 4)
    csv_dir = RAIZ / "datos" / "csv"
    tiene_csv = (csv_dir / "resumen_frente_sg.csv").exists()
    todo_ok &= check("datos/csv/ exportado", tiene_csv,
                     "python scripts/exportar_xlsx_a_csv.py")
    if tiene_csv:
        from ccus_columna import datos as d
        from ccus_columna.constantes import TASA_REF
        n = len(d.celdas_disponibles("malla_original", TASA_REF))
        todo_ok &= check(f"loader ccus_columna.datos (26 celdas: {n})", n == 26)

    # La referencia de la fase 0D (referencia_lapaz_0d) no se distribuye en
    # este repositorio publico; su nucleo esta portado en ccus_columna.mapa_saturacion.

    print("\n" + ("Entorno COMPLETO." if todo_ok else "Entorno INCOMPLETO (ver arriba)."))
    return 0 if todo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

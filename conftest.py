"""Configuracion pytest: pone src/ en el path para importar ccus_columna."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

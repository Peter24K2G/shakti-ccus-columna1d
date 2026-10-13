"""Rutas de figuras: una carpeta por metodo / malla / tasa / tipo de corrida."""

from __future__ import annotations

import re
from pathlib import Path

from ccus_columna.constantes import RAIZ

# malla_original_q<Q>_completo_anual  ->  (malla_original, <Q>, completo_anual)
_RE_CORRIDA = re.compile(
    r"^(malla_original|malla_refinada)_q(\d+)_(.+)$"
)


def parse_id_corrida(nombre: str) -> tuple[str, int, str] | None:
    m = _RE_CORRIDA.match(Path(nombre).name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def malla_desde_run(run_dir: Path) -> str | None:
    """malla_original / malla_refinada desde params.txt o el nombre de carpeta."""
    run_dir = Path(run_dir)
    params = run_dir / "params.txt"
    if params.exists():
        for ln in params.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.startswith("malla="):
                val = ln.split("=", 1)[1].strip()
                if val in ("malla_original", "malla_refinada"):
                    return val
    parsed = parse_id_corrida(run_dir.name)
    if parsed is not None:
        return parsed[0]
    return None


def dir_figuras_corrida(metodo: str, run_dir: Path) -> Path:
    """figuras/{met1|met2}/{malla}/q{tasa}/{variante}/"""
    parsed = parse_id_corrida(Path(run_dir).name)
    if parsed is None:
        return RAIZ / "figuras" / metodo / Path(run_dir).name
    malla, tasa, variante = parsed
    return RAIZ / "figuras" / metodo / malla / f"q{tasa}" / variante


def dir_figuras_comparacion(malla: str, tasa: int) -> Path:
    """figuras/comparacion/{malla}/q{tasa}/"""
    return RAIZ / "figuras" / "comparacion" / malla / f"q{tasa}"


def dir_figuras_cirrus() -> Path:
    return RAIZ / "figuras" / "cirrus"

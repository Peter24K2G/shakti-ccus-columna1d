"""Ejecucion de PHREEQC por consola (identico al runner de la fase 0D).

Invoca phreeqc.exe (version batch, no Interactive) via subprocess:

    phreeqc.exe entrada.pqi salida.pqo base_de_datos.dat [screen.log]

Uso desde consola:
    python -m ccus_columna.runner ruta/al/modelo.pqi
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Rutas por defecto = instalacion estandar de PHREEQC 3.8.6 (USGS) en Windows.
# Se pueden cambiar con las variables de entorno PHREEQC_EXE y PHREEQC_DB.
PHREEQC_EXE = Path(os.environ.get(
    "PHREEQC_EXE", r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\bin\Release\phreeqc.exe"))
DATABASE = Path(os.environ.get(
    "PHREEQC_DB", r"C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\database\phreeqc.dat"))
# Base de datos alternativa (instalacion de PHREEQC Interactive)
DATABASE_ALT = Path(r"C:\Program Files (x86)\USGS\Phreeqc Interactive 3.8.6-17100\database\phreeqc.dat")

TIMEOUT_S = 7200


@dataclass
class RunResult:
    """Resultado de una corrida de PHREEQC."""

    ok: bool
    pqi: Path
    pqo: Path
    selected_outputs: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def summary(self) -> str:
        estado = "OK" if self.ok else "ERROR"
        lineas = [
            f"[{estado}] {self.pqi.name}  ({self.elapsed_s:.1f} s)",
            f"  .pqo: {self.pqo}",
        ]
        for so in self.selected_outputs:
            lineas.append(f"  selected_output: {so}")
        for e in self.errors:
            lineas.append(f"  ERROR: {e}")
        if self.warnings:
            lineas.append(f"  ({len(self.warnings)} advertencias; ver .pqo)")
        return "\n".join(lineas)


def _database_path() -> Path:
    if DATABASE.exists():
        return DATABASE
    if DATABASE_ALT.exists():
        return DATABASE_ALT
    raise FileNotFoundError("No se encontro phreeqc.dat en las rutas conocidas.")


def _parse_pqo(pqo: Path) -> tuple[list[str], list[str]]:
    """Extrae lineas de ERROR y WARNING del archivo .pqo."""
    errors: list[str] = []
    warnings: list[str] = []
    if not pqo.exists():
        return ["No se genero el archivo .pqo"], warnings
    with pqo.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            ls = line.strip()
            if ls.startswith("ERROR"):
                errors.append(ls)
            elif ls.startswith("WARNING"):
                warnings.append(ls)
    return errors, warnings


def _selected_output_files(pqi: Path) -> list[Path]:
    """Rutas declaradas en lineas '-file' de SELECTED_OUTPUT (relativas al .pqi)."""
    files: list[Path] = []
    for line in pqi.read_text(encoding="utf-8", errors="replace").splitlines():
        ls = line.split("#")[0].strip()
        if ls.lower().startswith("-file"):
            parts = ls.split(None, 1)
            if len(parts) == 2:
                ruta = (pqi.parent / parts[1].strip()).resolve()
                if ruta not in files:
                    files.append(ruta)
    return files


def run_phreeqc(pqi_path: str | Path, pqo_path: str | Path | None = None,
                database: str | Path | None = None,
                timeout_s: int | None = None) -> RunResult:
    """Ejecuta PHREEQC en modo batch sobre un archivo .pqi.

    El proceso corre con cwd = carpeta del .pqi, de modo que las rutas
    relativas de SELECTED_OUTPUT (p. ej. ../resultados/x.txt) funcionan
    igual que en PHREEQC Interactive.
    """
    pqi = Path(pqi_path).resolve()
    if not pqi.exists():
        raise FileNotFoundError(pqi)
    if not PHREEQC_EXE.exists():
        raise FileNotFoundError(f"phreeqc.exe no encontrado: {PHREEQC_EXE}")

    pqo = Path(pqo_path).resolve() if pqo_path else pqi.with_suffix(".pqo")
    db = Path(database).resolve() if database else _database_path()

    t0 = datetime.now()
    proc = subprocess.run(
        [str(PHREEQC_EXE), str(pqi), str(pqo), str(db)],
        cwd=str(pqi.parent),
        capture_output=True,
        text=True,
        timeout=timeout_s if timeout_s is not None else TIMEOUT_S,
    )
    elapsed = (datetime.now() - t0).total_seconds()

    errors, warnings = _parse_pqo(pqo)
    if proc.returncode != 0 and not errors:
        errors.append(f"exit code {proc.returncode}: {proc.stderr.strip()[:500]}")

    selected = [p for p in _selected_output_files(pqi) if p.exists()]
    return RunResult(
        ok=proc.returncode == 0 and not errors,
        pqi=pqi,
        pqo=pqo,
        selected_outputs=selected,
        errors=errors,
        warnings=warnings,
        elapsed_s=elapsed,
    )


def run_in_dir(pqi_text: str, run_dir: str | Path, name: str = "modelo",
               timeout_s: int | None = None) -> RunResult:
    """Escribe un .pqi generado en run_dir y lo ejecuta (corrida autocontenida)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    pqi = run_dir / f"{name}.pqi"
    pqi.write_text(pqi_text, encoding="utf-8")
    return run_phreeqc(pqi, timeout_s=timeout_s)


def new_run_dir(base: str | Path) -> Path:
    """Crea <base>/runs/<timestamp>/ y devuelve la ruta."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> int:
    if len(sys.argv) < 2:
        print("Uso: python -m ccus_columna.runner <modelo.pqi> [salida.pqo]")
        return 2
    res = run_phreeqc(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(res.summary())
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

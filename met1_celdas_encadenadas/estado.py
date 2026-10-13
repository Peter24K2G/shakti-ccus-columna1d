"""Persistencia de estado PHREEQC entre pasos (DUMP / INCLUDE$).

Convencion interna por llamada (una celda / un paso):
    SOLUTION 1, EQUILIBRIUM_PHASES 1, KINETICS 1, GAS_PHASE 1
    SOLUTION 99 = efluente exportado hacia la celda aguas abajo
"""

from __future__ import annotations

from pathlib import Path

from ccus_columna import quimica
from ccus_columna.constantes import M_WATER_REF_KG

SOL_LOCAL = 1
SOL_EFLUENTE = 99


def ruta_dump(run_dir: Path, k: int, paso: int) -> Path:
    return run_dir / "estado" / f"estado_k{k:02d}_p{paso:04d}.dmp"


def ruta_efluente(run_dir: Path, k: int, paso: int) -> Path:
    return run_dir / "estado" / f"efluente_k{k:02d}_p{paso:04d}.dmp"


def _rel(run_dir: Path, path: Path) -> str:
    return path.resolve().relative_to(run_dir.resolve()).as_posix()


def bloque_dump(run_dir: Path, dump_path: Path) -> str:
    """DUMP del estado completo de la celda (solucion + eq + kinetics)."""
    rel = _rel(run_dir, dump_path)
    return f"""DUMP
    -file {rel}
    -solution {SOL_LOCAL}
    -equilibrium_phases {SOL_LOCAL}
    -kinetics {SOL_LOCAL}
END
"""


def bloque_dump_efluente(run_dir: Path, efluente_path: Path) -> str:
    """Copia solucion local a 99 y la DUMP como efluente (simulacion aparte)."""
    rel = _rel(run_dir, efluente_path)
    return f"""USE solution {SOL_LOCAL}
COPY solution {SOL_LOCAL} {SOL_EFLUENTE}
DUMP
    -file {rel}
    -solution {SOL_EFLUENTE}
END
"""


def bloque_restaurar(run_dir: Path, dump_path: Path) -> str:
    """INCLUDE$ del DUMP previo. GAS_PHASE se recrea en cada paso (como fase 0D)."""
    rel = _rel(run_dir, dump_path)
    return f"""INCLUDE$ {rel}
USE solution {SOL_LOCAL}
USE equilibrium_phases {SOL_LOCAL}
USE kinetics {SOL_LOCAL}
"""


def bloque_cargar_efluente(run_dir: Path, efluente_path: Path) -> str:
    rel = _rel(run_dir, efluente_path)
    return f"INCLUDE$ {rel}\n"


def bloque_inicial(agua_kg: float = M_WATER_REF_KG,
                   presion_atm: float = 100.0,
                   k: int | None = None,
                   malla: str | None = None) -> str:
    """Estado inicial de una celda (agua nativa + rocas + kinetics)."""
    return f"""{quimica.solucion_nativa(SOL_LOCAL, "Agua_Nativa", agua_kg, presion_atm)}
{quimica.equilibrium_rocas(SOL_LOCAL, k, malla=malla)}
{quimica.bloque_kinetics_inicial(SOL_LOCAL, k, malla=malla)}
SAVE solution {SOL_LOCAL}
END
USE solution {SOL_LOCAL}
USE equilibrium_phases {SOL_LOCAL}
USE kinetics {SOL_LOCAL}
SAVE solution {SOL_LOCAL}
END
"""

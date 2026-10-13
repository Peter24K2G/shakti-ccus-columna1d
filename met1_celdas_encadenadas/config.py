"""Configuracion de una corrida de celdas encadenadas (met1)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ccus_columna.constantes import TASA_REF  # parametros del proyecto
from ccus_columna.constantes import SG_UMBRAL_GAS


@dataclass
class CorridaConfig:
    """Parametros de una corrida met1 (ver DC-004/DC-005 en docs/decisiones.md)."""

    malla: str = "malla_original"
    tasa_m3d: int = TASA_REF
    # None => orden segun dominio (DC-004 / DC-019)
    celdas: list[int] | None = None
    # "rama_sello" k15->k10 | "columna_pozo" fondo perforacion->k10 (DC-017/DC-019)
    dominio: str = "rama_sello"
    # paso de tiempo global en anos; None => usar los 184 nodos del CSV
    dt_anos: float | None = None
    # regla de fraccion transferida: "tau_residencia" | "delta_sg" | "constante"
    regla_f: str = "tau_residencia"
    f_constante: float = 0.25          # solo si regla_f == "constante"
    sg_umbral: float = SG_UMBRAL_GAS   # activacion por Sg (legacy)
    # "sg" solo Sg>0 | "sg_o_dp" Sg>0 o gradiente P profundo->sello (DC-021)
    activacion_f: str = "sg"
    dp_umbral_bar: float = 0.5         # minimo deltaP origen-destino para MIX
    n_kinetics_base: int = 8           # sub-pasos KINETICS por paso
    usar_p_real: bool = True           # DC-003: P_CO2 con P(t) de celda
    # Limites opcionales (smoke / depuracion)
    max_pasos: int | None = None
    t_max_anos: float | None = None
    etiqueta: str = ""                 # sufijo opcional del id de corrida
    transferir: bool = True            # False = celdas aisladas (validacion 0D)

    def id_corrida(self) -> str:
        if self.etiqueta:
            variante = self.etiqueta
        elif self.celdas is None:
            variante = self.regla_f
        else:
            variante = f"{self.regla_f}_custom"
        return f"{self.malla}_q{self.tasa_m3d}_{variante}"

    def guardar(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        p = run_dir / "params.json"
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p


@dataclass
class EstadoCelda:
    """Referencias al estado persistido de una celda entre pasos."""

    k: int
    dump_path: Path | None = None      # DUMP del ultimo paso reaccionado
    efluente_path: Path | None = None  # solution reaccionada disponible aguas abajo
    historico: list[dict] = field(default_factory=list)  # hitos quimicos por paso

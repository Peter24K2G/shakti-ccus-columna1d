"""Calcula volumen y caudal de CO2 por celda desde Sg Cirrus (DC-023).

Uso:
  python scripts/calcular_volumen_inyectado.py
  python scripts/calcular_volumen_inyectado.py --malla malla_original --tasa <Q>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from ccus_columna.cli import tasas_disponibles, validar_malla_tasa
from ccus_columna.constantes import MALLAS
from ccus_columna.inyeccion_cirrus import (
    exportar_todos,
    resumen_columna_inyeccion,
    resumen_escenario,
    tabla_volumen_escenario,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Volumen/caudal CO2 por celda desde Sg Cirrus (DC-023)",
    )
    ap.add_argument("--malla", default=None, help="malla_original | malla_refinada")
    ap.add_argument("--tasa", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="carpeta salida CSV")
    args = ap.parse_args()

    if args.malla is None and args.tasa is None:
        paths = exportar_todos(args.out)
        print(f"Exportados {len(paths)} archivos en {paths[0].parents[2]}")
        for p in paths:
            print(f"  {p}")
        return 0

    mallas = [args.malla] if args.malla else list(MALLAS)
    for malla in mallas:
        tasas = [args.tasa] if args.tasa else list(tasas_disponibles(malla))
        for tasa in tasas:
            validar_malla_tasa(malla, tasa)
            tab = tabla_volumen_escenario(malla, tasa)
            out = args.out or (_RAIZ / "datos" / "derivados" / "volumen_gas_cirrus" / malla / f"q{tasa}")
            out.mkdir(parents=True, exist_ok=True)
            tab.to_csv(out / "volumen_gas_todas_celdas.csv", index=False, float_format="%.6g")
            resumen_escenario(malla, tasa, tab).to_csv(
                out / "resumen_por_celda.csv", index=False, float_format="%.6g",
            )
            rc = resumen_columna_inyeccion(malla, tasa, tab)
            print(f"\n{malla} q={tasa}:")
            print(f"  tasa nominal pozo: {rc['tasa_nominal_pozo_m3d']} m3/d")
            print(f"  suma Q desde Sg k15-30 (media iny.): {rc['Q_desde_Sg_suma_media_m3d']:.3f} m3/d")
            print(f"  ratio suma/tasa pozo: {rc['ratio_suma_vs_tasa_pozo']:.5f}")
            print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

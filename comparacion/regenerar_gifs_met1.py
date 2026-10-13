"""Regenera GIFs met1 (SI, porosidad, permeabilidad, carbono/pH, caudal Cirrus).

Salida: figuras/met1/{malla}/q{tasa}/{variante}/

Uso:
  python comparacion/regenerar_gifs_met1.py --malla malla_original --tasa <Q>
  python comparacion/regenerar_gifs_met1.py --todas
  python comparacion/regenerar_gifs_met1.py --run-dir met1_celdas_encadenadas/corridas/malla_original_q<Q>_completo
"""

from __future__ import annotations

import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ / "comparacion"))

from ccus_columna.constantes import TASA_REF  # parametros del proyecto
from ccus_columna.cli import tasas_disponibles, validar_malla_tasa
from ccus_columna.rutas import dir_figuras_corrida, parse_id_corrida

from gifs_met1 import MET1, generar_gifs_met1
from sg_carbono import gif_sg_carbono_met1
from gif_caudal_inyeccion import gif_caudal_inyeccion
from gif_delta_permeabilidad_met1 import gif_delta_permeabilidad_met1


def regenerar(
    malla: str,
    tasa: int,
    run_dir: Path | None = None,
    variante: str = "completo",
) -> list[Path]:
    run_dir = Path(run_dir) if run_dir else MET1 / f"{malla}_q{tasa}_{variante}"
    if not (run_dir / "perfil_quimica_met1.csv").exists():
        raise FileNotFoundError(f"Sin corrida: {run_dir / 'perfil_quimica_met1.csv'}")

    out = dir_figuras_corrida("met1", run_dir)
    paths: list[Path] = []
    paths.extend(generar_gifs_met1(malla, tasa, run_dir, out_dir=out))
    paths.append(gif_sg_carbono_met1(malla, tasa, run_dir, out_dir=out))
    parsed = parse_id_corrida(run_dir.name)
    var = parsed[2] if parsed else variante
    paths.append(gif_caudal_inyeccion(malla, tasa, variante=var,
                                      out_path=out / "gif_caudal_inyeccion_cirrus.gif"))
    paths.append(gif_delta_permeabilidad_met1(malla, tasa, run_dir, out_dir=out))
    return paths


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Regenerar GIFs met1")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--variante", default="completo",
                    help="sufijo de corrida: completo, completo_dp, ...")
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="ruta o nombre de carpeta de corrida")
    ap.add_argument("--todas", action="store_true",
                    help="todas las mallas y tasas con corrida completo")
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    if args.todas:
        ok, fail = 0, 0
        for malla in ("malla_original", "malla_refinada"):
            for tasa in tasas_disponibles(malla):
                run = MET1 / f"{malla}_q{tasa}_completo"
                if not (run / "perfil_quimica_met1.csv").exists():
                    print(f"SKIP {malla} q{tasa}")
                    fail += 1
                    continue
                try:
                    for p in regenerar(malla, tasa, run):
                        print(p)
                    print(f"OK {malla} q{tasa}")
                    ok += 1
                except Exception as e:
                    print(f"ERROR {malla} q{tasa}: {e}")
                    fail += 1
        print(f"\nListo: {ok} ok, {fail} skip/fail")
        return 0

    if args.run_dir:
        run_dir = args.run_dir if args.run_dir.is_absolute() else MET1 / args.run_dir
        parsed = parse_id_corrida(run_dir.name)
        if parsed:
            malla, tasa, _ = parsed
        else:
            malla, tasa = args.malla, args.tasa
    else:
        validar_malla_tasa(args.malla, args.tasa)
        malla, tasa = args.malla, args.tasa
        run_dir = MET1 / f"{malla}_q{tasa}_{args.variante}"

    for p in regenerar(malla, tasa, run_dir):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

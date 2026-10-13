"""Pipeline completo: met1 + met2 + figuras + comparacion.

Uso:
  python scripts/correr_columna_completa.py --rapido     # demo ~minutos
  python scripts/correr_columna_completa.py --completo   # 70 a (horas met1)
  python scripts/correr_columna_completa.py --solo-figuras --met1 ... --met2 ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccus_columna.constantes import TASA_REF, TASAS_M3D  # parametros del proyecto
MET1 = RAIZ / "met1_celdas_encadenadas" / "motor.py"
MET2 = RAIZ / "met2_transporte_phreeqc" / "generador.py"
COMP = RAIZ / "comparacion" / "comparar.py"


def _run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(RAIZ), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Correr met1+met2 y generar figuras")
    ap.add_argument("--malla", default="malla_original",
                    help="malla_original o malla_refinada")
    ap.add_argument("--tasa", type=int, default=TASA_REF,
                    help="tasas disponibles: " + ", ".join(str(q) for q in TASAS_M3D))
    ap.add_argument("--rapido", action="store_true",
                    help="rama corta met1 + reactivo_corto met2")
    ap.add_argument("--completo", action="store_true",
                    help="met1 completo anual + met2 reactivo 70a")
    ap.add_argument("--completo-fino", action="store_true",
                    help="met1 todos los nodos CSV (~183 pasos; muy largo)")
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--solo-figuras", action="store_true")
    ap.add_argument("--met1", type=Path, default=None, help="dir corrida met1")
    ap.add_argument("--met2", type=Path, default=None, help="dir corrida met2")
    ap.add_argument("--skip-met1", action="store_true")
    ap.add_argument("--skip-met2", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(RAIZ / "src"))
    from ccus_columna.cli import validar_malla_tasa
    validar_malla_tasa(args.malla, args.tasa)

    py = sys.executable

    if args.solo_figuras:
        if not args.met1 or not args.met2:
            raise SystemExit("--solo-figuras requiere --met1 y --met2")
        _run([py, str(MET1), "--solo-exportar", str(args.met1), "--figuras"])
        _run([py, str(MET2), "--solo-exportar", str(args.met2), "--figuras",
              "--modo", "reactivo"])
        _run([py, str(COMP), "--met1", str(args.met1), "--met2", str(args.met2),
              "--malla", args.malla, "--tasa", str(args.tasa)])
        return 0

    met1_dir = args.met1
    met2_dir = args.met2

    if not args.skip_met1:
        cmd = [py, str(MET1), "--malla", args.malla, "--tasa", str(args.tasa), "--figuras"]
        if args.reanudar:
            cmd.append("--reanudar")
        if args.rapido:
            cmd.append("--rama-corta")
        elif args.completo_fino:
            cmd.append("--completo")
        elif args.completo:
            cmd.append("--completo-anual")
        else:
            cmd.extend(["--rama-corta"])  # default sensato
        _run(cmd)
        if args.rapido:
            met1_dir = RAIZ / "met1_celdas_encadenadas" / "corridas" / f"{args.malla}_q{args.tasa}_rama_t15"
        elif args.completo_fino:
            met1_dir = RAIZ / "met1_celdas_encadenadas" / "corridas" / f"{args.malla}_q{args.tasa}_completo"
        elif args.completo:
            met1_dir = RAIZ / "met1_celdas_encadenadas" / "corridas" / f"{args.malla}_q{args.tasa}_completo_anual"
        else:
            met1_dir = RAIZ / "met1_celdas_encadenadas" / "corridas" / f"{args.malla}_q{args.tasa}_rama_t15"

    if not args.skip_met2:
        cmd = [py, str(MET2), "--malla", args.malla, "--tasa", str(args.tasa), "--figuras"]
        if args.reanudar:
            cmd.append("--reanudar")
        if args.rapido:
            cmd.extend(["--modo", "reactivo_corto"])
            met2_dir = RAIZ / "met2_transporte_phreeqc" / "corridas" / f"{args.malla}_q{args.tasa}_reactivo_corto"
        elif args.completo or args.completo_fino:
            cmd.extend(["--modo", "reactivo"])
            met2_dir = RAIZ / "met2_transporte_phreeqc" / "corridas" / f"{args.malla}_q{args.tasa}_reactivo"
        else:
            cmd.extend(["--modo", "reactivo_corto"])
            met2_dir = RAIZ / "met2_transporte_phreeqc" / "corridas" / f"{args.malla}_q{args.tasa}_reactivo_corto"
        _run(cmd)

    if met1_dir and met2_dir and met1_dir.exists() and met2_dir.exists():
        _run([py, str(RAIZ / "met1_celdas_encadenadas" / "generar_figuras.py"), str(met1_dir)])
        _run([py, str(RAIZ / "met2_transporte_phreeqc" / "generar_figuras.py"), str(met2_dir)])
        _run([py, str(COMP), "--met1", str(met1_dir), "--met2", str(met2_dir),
              "--malla", args.malla, "--tasa", str(args.tasa)])

    print("\nListo.")
    print(f"  met1: {met1_dir}")
    print(f"  met2: {met2_dir}")
    print(f"  figuras: {RAIZ / 'figuras'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

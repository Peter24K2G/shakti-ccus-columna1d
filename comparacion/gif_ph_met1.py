"""Genera el GIF combinado C + columna pH (ver sg_carbono.gif_sg_carbono_met1)."""

from sg_carbono import gif_sg_carbono_met1, MET1
from ccus_columna.constantes import TASA_REF  # parametros del proyecto

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="GIF met1: perfil carbono + columna pH (colores)",
    )
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    run = args.run_dir or MET1 / f"{args.malla}_q{args.tasa}_completo"
    print(gif_sg_carbono_met1(args.malla, args.tasa, run, args.n_frames, args.fps))

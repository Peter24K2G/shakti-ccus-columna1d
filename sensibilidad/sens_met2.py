"""Sensibilidad met2 a la dispersividad (copia local)."""
import sys, json, time
from pathlib import Path
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[0] / "src"))
fac = float(sys.argv[1]); salida = Path(sys.argv[2]); resto = sys.argv[3:]
import calibracion as cal
_d = cal.dispersividades_m
cal.dispersividades_m = lambda *a, **k: [x * fac for x in _d(*a, **k)]
import generador
generador.RAIZ_CORRIDAS = salida
salida.mkdir(parents=True, exist_ok=True)
(salida / "sens_params.json").write_text(json.dumps({"variable": "dispersividad", "factor": fac, "args": resto,
    "inicio": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
t0 = time.time(); rc = generador.main(resto)
(salida / "sens_tiempo.txt").write_text(f"{time.time()-t0:.1f} s\n", encoding="utf-8")
sys.exit(rc)

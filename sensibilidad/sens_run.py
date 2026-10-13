"""Envoltura de sensibilidad OAT para met1 (copia local; no modifica el paquete original)."""
import sys, os, json, time
from pathlib import Path
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[0] / "src"))
var, fac = sys.argv[1], float(sys.argv[2]); salida = Path(sys.argv[3]); resto = sys.argv[4:]
from ccus_columna import kinetics_pk2004 as pk, mineralogia as mi, mapa_saturacion as ms
if var == "rsa_silicatos":
    _o = pk.rsa_m2
    pk.rsa_m2 = lambda m, n: _o(m, n) * (1.0 if m == "Calcite" else fac)
elif var == "calcita_inicial":
    mi.INVENTARIO_EQ_RESERVORIO["Calcite"] *= fac; mi.MOLES_SELLO["Calcite"] *= fac
elif var == "pco2_escala":
    _p = ms.p_co2_fugacidad_proporcional
    ms.p_co2_fugacidad_proporcional = lambda sg, sgm, p_ref=80.0: _p(sg, sgm, p_ref) * fac
elif var == "xrd_sello":
    nuevo = mi.moles_desde_wt(mi.wt_promedio_xrd((mi.XRD_ESMERALDAS[int(fac)],)), mi.PHI_SELLO)
    mi.MOLES_SELLO.clear(); mi.MOLES_SELLO.update(nuevo)
elif var != "base":
    raise SystemExit("variable desconocida")
import motor
motor.RAIZ_CORRIDAS = salida
salida.mkdir(parents=True, exist_ok=True)
(salida / "sens_params.json").write_text(json.dumps({"variable": var, "factor": fac, "args": resto,
    "inicio": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2), encoding="utf-8")
t0 = time.time(); rc = motor.main(resto)
(salida / "sens_tiempo.txt").write_text(f"{time.time()-t0:.1f} s\n", encoding="utf-8")
sys.exit(rc)

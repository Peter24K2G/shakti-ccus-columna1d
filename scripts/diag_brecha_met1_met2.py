"""Diagnostico de brecha met1 vs met2 + consistencia k15 vs etapas."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccus_columna.constantes import TASA_REF, T_FIN_EXTRACCION, T_FIN_INYECCION, T_FIN  # parametros del proyecto
OUT = ROOT / "figuras" / "publicacion"
OUT.mkdir(parents=True, exist_ok=True)
DOC = ROOT / "docs" / "04_ROBUSTEZ_DIAGNOSTICO.md"

MET1 = ROOT / f"met1_celdas_encadenadas/corridas/malla_original_q{TASA_REF}_completo_anual/perfil_quimica_met1.csv"
MET2 = ROOT / f"met2_transporte_phreeqc/corridas/malla_original_q{TASA_REF}_reactivo_dc018/perfil_quimica_met2.csv"

# optional Cirrus Sg for k15/k10
SG_CSV = ROOT / f"datos/csv/malla_original/q{TASA_REF}/celda_k15.csv"
SG_K10 = ROOT / f"datos/csv/malla_original/q{TASA_REF}/celda_k10.csv"

def load(p):
    return pd.read_csv(p)

def series_at(df, k, cols):
    sub = df[df["k"] == k].sort_values("tiempo_anos")
    return sub[["tiempo_anos"] + [c for c in cols if c in sub.columns]]

m1 = load(MET1)
m2 = load(MET2)
cols = ["pH", "C_total_mol_kgw", "calcita_mol", "I_alm_mol", "delta_phi"]

# Align comparison at common times via interpolation
summary = {"casos": f"malla_original q{TASA_REF}", "met1": str(MET1.name), "met2": str(MET2.name), "celdas": {}, "hipotesis": []}

for k in (15, 10):
    s1 = series_at(m1, k, cols)
    s2 = series_at(m2, k, cols)
    t_common = np.linspace(0, 70, 71)
    row = {"k": k}
    for c in cols:
        y1 = np.interp(t_common, s1["tiempo_anos"], s1[c], left=np.nan, right=np.nan)
        y2 = np.interp(t_common, s2["tiempo_anos"], s2[c], left=np.nan, right=np.nan)
        d = y1 - y2
        row[c] = {
            "met1_t0": float(y1[0]),
            "met2_t0": float(y2[0]),
            "met1_t70": float(y1[-1]),
            "met2_t70": float(y2[-1]),
            "rmse": float(np.sqrt(np.nanmean(d**2))),
            "bias_mean": float(np.nanmean(d)),
            "max_abs": float(np.nanmax(np.abs(d))),
        }
    summary["celdas"][str(k)] = row

# Stage metrics
for stage, t0, t1 in [("extraccion", 0, T_FIN_EXTRACCION), ("inyeccion", T_FIN_EXTRACCION, T_FIN_INYECCION), ("observacion", T_FIN_INYECCION, T_FIN)]:
    block = {}
    for k in (15, 10):
        s1 = series_at(m1, k, ["pH", "C_total_mol_kgw"])
        s2 = series_at(m2, k, ["pH", "C_total_mol_kgw"])
        m1s = s1[(s1.tiempo_anos >= t0) & (s1.tiempo_anos <= t1)]
        m2s = s2[(s2.tiempo_anos >= t0) & (s2.tiempo_anos <= t1)]
        block[f"k{k}"] = {
            "met1_pH_mean": float(m1s.pH.mean()) if len(m1s) else None,
            "met2_pH_mean": float(m2s.pH.mean()) if len(m2s) else None,
            "met1_C_mean": float(m1s.C_total_mol_kgw.mean()) if len(m1s) else None,
            "met2_C_mean": float(m2s.C_total_mol_kgw.mean()) if len(m2s) else None,
        }
    summary.setdefault("por_etapa", {})[stage] = block

# Hypotheses based on numbers
h = []
h.append("Baseline DC-018: a t~0 ambos parten cerca (pH~6.5, C~0.002); la brecha crece con la inyeccion, no en extraccion.")
ph15 = summary["celdas"]["15"]["pH"]
c15 = summary["celdas"]["15"]["C_total_mol_kgw"]
cal15 = summary["celdas"]["15"]["calcita_mol"]
if ph15["met1_t70"] < 6.0 and ph15["met2_t70"] > 6.5:
    h.append("A 70 a, met1 permanece acido (forzante GAS_PHASE / P_CO2 alto ligado a Sg) mientras met2 queda casi neutro: sugiere buffer CO2(g) mas debil o lavado advectivo de acidez en met2.")
if c15["met1_t70"] > 3 * c15["met2_t70"]:
    h.append("C acuoso met1 >> met2 a 70 a: met2 no sostiene la misma fugacidad efectiva de CO2 en observacion, o TRANSPORT diluye C hacia el sello/pozo.")
if cal15["met2_t70"] < 0.5 * cal15["met1_t70"]:
    h.append("Calcita k15: met2 disuelve mucho mas que met1 — inconsistente con pH mas alto en met2; posible artefacto de inventario/volumen o de reporte, o historia transitoria distinta (disolucion temprana luego reprecipitacion parcial).")
h.append("met1: cada celda es batch con mapa S->(mw,Vgas,P_CO2) cada paso (GAS_PHASE fixed_pressure tipico de fase 0D).")
h.append("met2: CO2(g) como EQUILIBRIUM_PHASES con SI desde mapa de fugacidad; TRANSPORT mueve solucion; SI_CO2_SUAVE=-1.62 solo alinea baseline (DC-018), no garantiza misma trayectoria bajo Sg alto.")
h.append("DC-029 calibra llegadas/ancho del frente de transporte, no el equilibrio gas-solucion bajo Sg residual en shut-in.")
summary["hipotesis"] = h

# Figure diagnostic
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Rectangle

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, constrained_layout=True)
pairs = [
    (axes[0, 0], "pH", "pH"),
    (axes[0, 1], "C_total_mol_kgw", "C total (mol/kgw)"),
    (axes[1, 0], "calcita_mol", "Calcita (mol)"),
    (axes[1, 1], "I_alm_mol", "I_alm (mol)"),
]
for ax, col, ylab in pairs:
    for k, ls in ((15, "-"), (10, "--")):
        s1 = series_at(m1, k, [col])
        s2 = series_at(m2, k, [col])
        ax.plot(s1.tiempo_anos, s1[col], ls=ls, color="#0072B2", lw=1.6, label=f"met1 k={k}")
        ax.plot(s2.tiempo_anos, s2[col], ls=ls, color="#E69F00", lw=1.6, label=f"met2 k={k}")
    for t0, t1, c in ((0, T_FIN_EXTRACCION, "#56B4E9"), (T_FIN_EXTRACCION, T_FIN_INYECCION, "#E69F00"), (T_FIN_INYECCION, T_FIN, "#009E73")):
        ax.axvspan(t0, t1, color=c, alpha=0.08, zorder=0)
    ax.set_ylabel(ylab)
    ax.legend(fontsize=7, ncol=2, loc="best")
axes[1, 0].set_xlabel("t (años)")
axes[1, 1].set_xlabel("t (años)")
fig.suptitle("Diagnóstico brecha met1 vs met2 — q50k original (DC-018 runs)", fontsize=12)
fig.savefig(OUT / "FIG07_diagnostico_brecha_met1_met2.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "FIG07_diagnostico_brecha_met1_met2.pdf", bbox_inches="tight")
plt.close(fig)

# Markdown report
lines = [
    "# 04 — Robustez: diagnóstico brecha met1 ↔ met2",
    "",
    f"Caso: `malla_original` q={TASA_REF}. met1 `completo_anual` vs met2 `reactivo_dc018`.",
    "",
    "## Hallazgos cuantitativos (RMSE / extremos)",
    "",
]
for k, block in summary["celdas"].items():
    lines.append(f"### Celda k={k}")
    lines.append("")
    lines.append("| Variable | met1 t0 | met2 t0 | met1 t70 | met2 t70 | RMSE | bias (m1-m2) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in cols:
        b = block[c]
        lines.append(
            f"| {c} | {b['met1_t0']:.4g} | {b['met2_t0']:.4g} | {b['met1_t70']:.4g} | {b['met2_t70']:.4g} | {b['rmse']:.4g} | {b['bias_mean']:.4g} |"
        )
    lines.append("")

lines += [
    "## Por etapa (medias)",
    "",
]
for stage, block in summary["por_etapa"].items():
    lines.append(f"- **{stage}**: k15 met1 pH={block['k15']['met1_pH_mean']:.3f} / met2 pH={block['k15']['met2_pH_mean']:.3f}; "
                 f"C met1={block['k15']['met1_C_mean']:.4f} / met2={block['k15']['met2_C_mean']:.4f}")

lines += [
    "",
    "## Hipótesis (ordenadas)",
    "",
]
for i, item in enumerate(h, 1):
    lines.append(f"{i}. {item}")

lines += [
    "",
    "## Acciones recomendadas (para cerrar la brecha)",
    "",
    "1. Verificar en met2 que `si_co2_por_celda` bajo Sg alto reproduce `log10(P_CO2)` del mapa compartido (`mapa_saturacion.p_co2_fugacidad_proporcional`), no solo el encendido/apagado del buffer.",
    "2. En observación (shut-in): confirmar que EQUILIBRIUM_PHASES_MODIFY refresca SI residual (DC-017) y que no hay lavado advectivo espurio (v debería ser ~0 por signo Darcy).",
    "3. Homogeneizar reporte de calcita: mismo V_bulk / factor de moles de campo (DC-015) en ambos exportadores.",
    "4. Test de consistencia: celda k=15 met1 aislada (sin MIX) vs corrida 0D de referencia a la misma tasa.",
    "5. Sensibilidad: SI_CO2_SUAVE no debería dominar tras llegada de gas; barrer solo el mapa de fugacidad (P_ref y Sg/Sg_max).",
    "",
    "Figura: `figuras/publicacion/FIG07_diagnostico_brecha_met1_met2.png`.",
    "",
]

DOC.write_text("\n".join(lines), encoding="utf-8")
(OUT.parent.parent / "docs").mkdir(exist_ok=True)
# also dump json
(OUT / f"diagnostico_brecha_q{TASA_REF}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("WROTE", DOC)
print("FIG07 OK")
print(json.dumps({k: summary["celdas"][k]["pH"] for k in summary["celdas"]}, indent=2))

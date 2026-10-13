"""Diagnostico y figura: adveccion/dispersion met2 vs frente Cirrus (DC-029).

Calcula:
  - v por hop k_i→k_{i+1} = dz / Δt_primer_gas   (adveccion)
  - a_L desde ancho 0.1–0.9 del frente Sg         (dispersion)

Uso:
  python comparacion/diagnostico_met2_transporte.py
  python comparacion/diagnostico_met2_transporte.py --malla malla_original --tasa <Q>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RAIZ / "src"))
sys.path.insert(0, str(_RAIZ / "met2_transporte_phreeqc"))

import calibracion as cal
from ccus_columna.constantes import TASA_REF  # parametros del proyecto
from ccus_columna import datos
from ccus_columna.constantes import K_INYECCION, K_SELLO, T_FIN_EXTRACCION
from ccus_columna.geometria import dz_m
from ccus_columna.rutas import dir_figuras_comparacion


def fig_llegadas_y_v(malla: str, tasa: int, out: Path) -> Path:
    hops = cal.hops_hacia_sello(malla, tasa)
    r = datos.resumen_frente()
    r = r[(r["malla"] == malla) & (r["tasa_m3d"] == tasa)].set_index("k")
    ks = [h[0] for h in hops] + [hops[-1][1]] if hops else []
    t_cirrus = [float(r.loc[k, "t_primer_gas_anos"]) for k in ks]

    # Llegada PHREEQC con v por hops (acumulando travel)
    t_ph = [t_cirrus[0]]
    t_acc = t_cirrus[0]
    for _k0, k1, t0, t1, v in hops:
        t_acc = t_acc + dz_m(k1) / v  # = t1 by construction
        t_ph.append(t_acc)

    # Llegada con v unica (media hops) — peor ajuste
    v_med = cal.v_media_m_por_ano(malla, tasa)
    t_uni = [t_cirrus[0]]
    z = 0.0
    for _k0, k1, *_rest in hops:
        z += dz_m(k1)
        t_uni.append(t_cirrus[0] + z / v_med)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax.plot(t_cirrus, ks, "ko-", ms=7, lw=2, label="Cirrus $t_{primer\\,gas}$")
    ax.plot(t_ph, ks, "s--", color="#1565c0", ms=6, lw=1.8, label="met2 v por hop (DC-029)")
    ax.plot(t_uni, ks, "^:", color="#c62828", ms=6, lw=1.5, label=f"met2 v unica={v_med:.1f} m/a")
    ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylim(max(ks) + 0.5, min(ks) - 0.5)
    ax.set_xlabel("Tiempo de llegada (a)")
    ax.set_ylabel("Celda k (sello arriba)")
    ax.set_title("Adveccion: llegadas del frente")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    labels = [f"{a}→{b}" for a, b, *_ in hops]
    vs = [h[4] for h in hops]
    ax2.bar(range(len(vs)), vs, color="#1565c0", alpha=0.85)
    ax2.set_xticks(range(len(vs)))
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("v (m/a)")
    ax2.set_title(r"Adveccion por hop $v=\Delta z/\Delta t_{gas}$")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"met2 calibracion transporte — {malla} q={tasa}", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_dispersion(malla: str, tasa: int, out: Path) -> Path:
    info = cal.ancho_frente_sg_m(malla, tasa)
    a_L = cal.a_L_calibrada_m(malla, tasa)
    a_L_old = 0.1 * cal.dz_shift_m(cal.celdas_transport(malla, tasa))
    zs = np.asarray(info["z_m"])
    sgs = np.asarray(info["Sg"])
    # perfil teorico erf con a_L calibrada (solo ilustrativo)
    x_c = zs[np.argmax(np.gradient(sgs))] if len(sgs) > 2 else zs.mean()
    smax = info["Sg_max"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ax.plot(sgs, zs, "ko-", ms=5, lw=1.8, label=f"Sg Cirrus @ t={info['t_perfil']:.1f} a")
    ax.axhline(zs[0], color="#6a1b9a", ls=":", alpha=0.5)
    ax.set_xlabel("Sg")
    ax.set_ylabel("z desde entrada (m, hacia sello)")
    ax.set_title(f"Frente espacial (ancho 0.1–0.9 = {info['ancho_m']:.1f} m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2.bar(["a_L DC-014\n(0.1 dz)", "a_L DC-029\n(frente Cirrus)"],
            [a_L_old, a_L], color=["#90a4ae", "#1565c0"], alpha=0.9)
    ax2.set_ylabel(r"$a_L$ (m)")
    ax2.set_title(r"Dispersion: $a_L=(L_{0.1-0.9}/3.92)^2 / x$")
    ax2.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate([a_L_old, a_L]):
        ax2.text(i, v + 0.05, f"{v:.2f} m", ha="center", fontsize=9)

    fig.suptitle(f"met2 dispersion — {malla} q={tasa}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnostico adveccion/dispersion met2")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    args = ap.parse_args()
    malla, tasa = args.malla, args.tasa

    out = dir_figuras_comparacion(malla, tasa) / "met2_calibracion"
    out.mkdir(parents=True, exist_ok=True)

    celdas = cal.celdas_transport(malla, tasa)
    hops = cal.hops_hacia_sello(malla, tasa)
    info = cal.ancho_frente_sg_m(malla, tasa)
    a_L = cal.a_L_calibrada_m(malla, tasa)
    v_et = cal.v_por_etapa(malla, tasa)

    print("=== MET2 adveccion / dispersion", malla, f"q={tasa}", "===")
    print("celdas:", celdas)
    print(f"a_L calibrada: {a_L:.3f} m  (antes 0.1 dz ~ {0.1*cal.dz_shift_m(celdas):.2f} m)")
    print(f"ancho frente: {info['ancho_m']:.2f} m @ t={info['t_perfil']:.2f} a")
    print("hops (adveccion):")
    for k0, k1, t0, t1, v in hops:
        print(f"  {k0}->{k1}: dt={t1-t0:.3f} a  v={v:.3f} m/a  dz={dz_m(k1):.1f} m")
    print("v_por_etapa:")
    for k, v in v_et.items():
        print(f"  {k:20s} {v:8.3f} m/a")

    pd.DataFrame([
        {"k0": a, "k1": b, "t0": t0, "t1": t1, "v_m_a": v, "dz_m": dz_m(b)}
        for a, b, t0, t1, v in hops
    ]).to_csv(out / "adveccion_hops.csv", index=False, float_format="%.6g")

    pd.DataFrame([{
        "a_L_calibrada_m": a_L,
        "a_L_dc014_m": 0.1 * cal.dz_shift_m(celdas),
        "ancho_frente_m": info["ancho_m"],
        "x_travel_m": info["x_travel_m"],
        "t_perfil_a": info["t_perfil"],
        "v_media_hops_m_a": cal.v_media_m_por_ano(malla, tasa),
    }]).to_csv(out / "dispersion_calibrada.csv", index=False, float_format="%.6g")

    rows = []
    for ta, tb, nom in cal.ventanas_operativas(malla, tasa):
        v = v_et.get(nom, cal.velocidad_ventana_m_por_ano(malla, tasa, ta, tb))
        p = cal.parametros_transport(malla, tasa, ta, tb)
        # override v display for hop windows
        if nom.startswith("frente_"):
            p = {**p, "v_m_por_ano": v}
        rows.append({"ventana": nom, "t_a": ta, "t_b": tb, "v_m_a": v, **{
            k: p[k] for k in ("time_step_s", "shifts", "dt_shift_anos", "flow_direction")
            if k in p
        }})
    pd.DataFrame(rows).to_csv(out / "adveccion_ventanas.csv", index=False, float_format="%.6g")

    p1 = fig_llegadas_y_v(malla, tasa, out / "adveccion_llegadas.png")
    p2 = fig_dispersion(malla, tasa, out / "dispersion_aL.png")
    print(p1)
    print(p2)
    print("CSV ->", out)
    print()
    print("Para regenerar el .pqi met2 con esta calibracion:")
    print(f"  python met2_transporte_phreeqc/generador.py --modo reactivo "
          f"--malla {malla} --tasa {tasa}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

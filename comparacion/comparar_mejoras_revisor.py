"""Comparacion baseline vs mejoras revisor IA (DC-021).

Mejoras:
  1. SI Calcite (reservorio) / Quartz (sello) en lugar de mineral dominante
  2. delta_phi incluye silicatos cinéticos (postproceso)
  3. Transporte activacion sg_o_dp (corrida completo_dp)

Uso:
  python comparacion/comparar_mejoras_revisor.py --malla malla_original --tasa <Q>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from ccus_columna.constantes import TASA_REF, T_FIN_EXTRACCION, T_FIN_INYECCION  # parametros del proyecto
from ccus_columna import datos
from ccus_columna.constantes import K_INYECCION, K_SELLO
from ccus_columna.mineralogia import inventario_eq, inventario_kin, phi0
from ccus_columna.postproceso import _fila_final_paso, calcular_delta_phi, leer_selected_output
from ccus_columna.rutas import dir_figuras_comparacion

from sg_carbono import (
    LABEL_Q,
    MET1,
    _etapa,
    _perfil_en_t,
    ks_columna_met1,
    tabla_carbono_met1,
)

RUN_BASE = MET1 / f"malla_original_q{TASA_REF}_completo"
RUN_MEJOR = MET1 / f"malla_original_q{TASA_REF}_completo_dp"


def _tabla_si(row_fn, run_dir: Path, ks: list[int]) -> pd.DataFrame:
    pasos_map = {}
    if (run_dir / "pasos.json").exists():
        raw = json.loads((run_dir / "pasos.json").read_text(encoding="utf-8"))
        pasos_map = {int(p["paso"]): float(p["t_fin"]) for p in raw}
    filas = []
    pat = re.compile(r"k(\d+)_p(\d+)")
    for path in sorted((run_dir / "hitos").glob("k*_p*.txt")):
        m = pat.search(path.name)
        if not m:
            continue
        k, paso = int(m.group(1)), int(m.group(2))
        if k not in ks:
            continue
        t = pasos_map.get(paso, float("nan"))
        if not np.isfinite(t):
            continue
        row = _fila_final_paso(leer_selected_output(path))
        filas.append({"tiempo_anos": t, "k": k, "valor": row_fn(row, k)})
    return pd.DataFrame(filas).sort_values(["tiempo_anos", "k"])


_FASE_A_SI = {
    "Calcite": "si_calcite",
    "Quartz": "si_quartz",
    "Pyrite": "si_pyrite",
    "Dolomite": "si_dolomite",
    "Dawsonite": "si_dawsonite",
    "Ankerite": "si_ankerite",
    "Gypsum": "si_gypsum",
    "K-feldspar": "si_k-feldspar",
    "Albite": "si_albite",
    "Kaolinite": "si_kaolinite",
    "Illite": "si_illite",
    "Chlorite(14A)": "si_chlorite(14a)",
    "Ca-Montmorillonite": "si_ca-montmorillonite",
    "K-mica": "si_k-mica",
    "Siderite": "si_siderite",
}


def mineral_dominante(k: int, malla: str) -> tuple[str, str]:
    inv: dict[str, float] = {}
    for f, m in inventario_eq(k, malla).items():
        inv[f] = inv.get(f, 0.0) + float(m)
    for f, m in inventario_kin(k, malla).items():
        inv[f] = inv.get(f, 0.0) + float(m)
    fase = max(inv, key=inv.get)
    col = _FASE_A_SI.get(fase, "si_" + fase.lower().replace("(", "").replace(")", ""))
    return fase, col


def si_baseline(row: pd.Series, k: int, malla: str) -> float:
    """Mineral dominante por moles (legacy figuras)."""
    _, col = mineral_dominante(k, malla)
    return float(row.get(col, 0.0) or 0.0)


def si_mejorado(row: pd.Series, k: int) -> float:
    if k == K_SELLO:
        return float(row.get("si_quartz", 0.0) or 0.0)
    return float(row.get("si_calcite", 0.0) or 0.0)


def dphi_baseline(row: pd.Series, k: int, malla: str) -> float:
    return 100.0 * calcular_delta_phi(row, k=k, malla=malla, incluir_cinetica=False) / phi0(k)


def dphi_mejorado(row: pd.Series, k: int, malla: str) -> float:
    return 100.0 * calcular_delta_phi(row, k=k, malla=malla, incluir_cinetica=True) / phi0(k)


def gif_comparacion_doble(
    tab_c: pd.DataFrame,
    tab_base: pd.DataFrame,
    tab_mej: pd.DataFrame,
    ks: list[int],
    malla: str,
    tasa: int,
    k_fondo: int,
    out_path: Path,
    titulo: str,
    label_base: str,
    label_mej: str,
    cmap,
    norm_base,
    norm_mej,
    n_frames: int = 60,
    fps: int = 6,
) -> Path:
    """3 paneles: C/Sg | columna baseline | columna mejorado."""
    ks_arr = np.array(ks, dtype=float)
    t_anim = np.linspace(0.0, float(tab_c["tiempo_anos"].max()), n_frames)
    x_f = max(float(np.nanpercentile(tab_c["f_C"], 99)) * 1.08, 0.01)
    x_sg = max(float(tab_c["Sg_cirrus_pct"].max()) * 1.08, 5.0)
    y_lo, y_hi = ks_arr.min() - 0.5, ks_arr.max() + 0.5

    fig, (ax, ax_b, ax_m) = plt.subplots(
        1, 3, figsize=(13.5, 5.4),
        gridspec_kw={"width_ratios": [4.0, 0.65, 0.65], "wspace": 0.12},
    )
    axb = ax.twiny()
    line_f, = ax.plot([], [], "o-", color="#1565c0", ms=7, lw=2.0,
                      label=r"$\Delta n_C / n_{total}$")
    line_s, = axb.plot([], [], "s--", color="#e65100", ms=6, lw=1.6, label="Sg Cirrus")
    fill = {"p": None}
    ax.set_xlim(0, x_f)
    ax.set_ylim(y_hi, y_lo)
    for kk, c in ((K_INYECCION, "#1565c0"), (k_fondo, "#6a1b9a"), (K_SELLO, "#2e7d32")):
        ax.axhline(kk, color=c, ls=":", lw=0.8, alpha=0.8)
    ax.set_ylabel("Celda k (sello arriba)")
    ax.grid(True, alpha=0.3)
    axb.set_xlim(0, x_sg)
    axb.tick_params(axis="x", colors="#e65100", labelsize=8)
    ax.legend(fontsize=7, loc="lower right")

    for ax_r, lab, norm, cmap_r in (
        (ax_b, label_base, norm_base, cmap),
        (ax_m, label_mej, norm_mej, cmap),
    ):
        ax_r.set_xlim(0, 1)
        ax_r.set_ylim(y_hi, y_lo)
        ax_r.set_xticks([])
        ax_r.set_title(lab, fontsize=9)
        ax_r.tick_params(axis="y", labelleft=False)
        for kk, c in ((K_INYECCION, "#1565c0"), (k_fondo, "#6a1b9a"), (K_SELLO, "#2e7d32")):
            ax_r.axhline(kk, color=c, ls=":", lw=0.8, alpha=0.5)

    v0b = float(np.nanmedian(tab_base["valor"]))
    v0m = float(np.nanmedian(tab_mej["valor"]))
    im_b = ax_b.imshow(np.full((len(ks), 1), v0b), aspect="auto", cmap=cmap,
                       norm=norm_base, extent=[0, 1, y_lo, y_hi], origin="upper",
                       interpolation="nearest")
    im_m = ax_m.imshow(np.full((len(ks), 1), v0m), aspect="auto", cmap=cmap,
                       norm=norm_mej, extent=[0, 1, y_lo, y_hi], origin="upper",
                       interpolation="nearest")
    cax_b = make_axes_locatable(ax_b).append_axes("right", size="12%", pad=0.06)
    cax_m = make_axes_locatable(ax_m).append_axes("right", size="12%", pad=0.06)
    cb_b = fig.colorbar(im_b, cax=cax_b, label=label_base)
    cb_m = fig.colorbar(im_m, cax=cax_m, label=label_mej)
    cb_b.ax.tick_params(labelsize=7)
    cb_m.ax.tick_params(labelsize=7)

    tit = fig.suptitle("", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.82, bottom=0.12)

    def update(fr: int):
        tt = float(t_anim[fr])
        f = _perfil_en_t(tab_c, "f_C", tt, ks)
        sg = _perfil_en_t(tab_c, "Sg_cirrus_pct", tt, ks)
        vb = _perfil_en_t(tab_base, "valor", tt, ks)
        vm = _perfil_en_t(tab_mej, "valor", tt, ks)
        line_f.set_data(f, ks_arr)
        line_s.set_data(sg, ks_arr)
        if fill["p"] is not None:
            fill["p"].remove()
        fill["p"] = ax.fill_betweenx(ks_arr, 0, np.nan_to_num(f, nan=0.0),
                                     color="#90caf9", alpha=0.35)
        im_b.set_data(vb.reshape(-1, 1))
        im_m.set_data(vm.reshape(-1, 1))
        tit.set_text(
            f"{titulo}  |  {malla} q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t = {tt:.1f} a ({_etapa(tt)})  |  k{k_fondo}->k10"
        )
        return line_f, line_s, tit

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def fig_series_k15(
    tab_base: pd.DataFrame,
    tab_mej: pd.DataFrame,
    malla: str,
    tasa: int,
    out_path: Path,
    ylabel: str,
    titulo: str,
) -> Path:
    k = K_INYECCION
    b = tab_base[tab_base["k"] == k].sort_values("tiempo_anos")
    m = tab_mej[tab_mej["k"] == k].sort_values("tiempo_anos")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(b["tiempo_anos"], b["valor"], "k--", lw=1.5, label="baseline (revisor)")
    ax.plot(m["tiempo_anos"], m["valor"], color="#1565c0", lw=2, label="mejorado (DC-021)")
    ax.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#fff3e0", alpha=0.4, label="inyeccion")
    ax.set_xlabel("Tiempo (a)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{titulo} — k={k}, {malla} q={LABEL_Q.get(tasa, tasa)}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def diagnostic_transporte(malla: str, tasa: int, ks: list[int]) -> pd.DataFrame:
    """Pasos donde sg_o_dp activaria MIX pero sg no (sin re-correr PHREEQC)."""
    import transferencia as tr  # met1 local

    pasos = []
    for k_dest in ks[1:]:
        k_orig = ks[ks.index(k_dest) - 1] if k_dest in ks else None
        if k_orig is None:
            continue
        idx = ks.index(k_dest)
        k_orig = ks[idx - 1]
        df_o = datos.celda(malla, tasa, k_orig)
        for _, row in df_o.iterrows():
            t = float(row["tiempo_anos"])
            sg_act = tr.transferencia_activa(malla, tasa, k_orig, k_dest, t, "sg")
            dp_act = tr.transferencia_activa(malla, tasa, k_orig, k_dest, t, "sg_o_dp")
            if dp_act and not sg_act:
                pasos.append({
                    "t": t, "k_origen": k_orig, "k_destino": k_dest,
                    "P_orig": datos.interpolar_celda(malla, tasa, k_orig, t)["P_bar"],
                    "P_dest": datos.interpolar_celda(malla, tasa, k_dest, t)["P_bar"],
                })
    return pd.DataFrame(pasos)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--run-base", type=Path, default=RUN_BASE)
    ap.add_argument("--run-mejor", type=Path, default=RUN_MEJOR)
    args = ap.parse_args()

    run = Path(args.run_base)
    out = dir_figuras_comparacion(args.malla, args.tasa) / "comparacion_mejoras"
    out.mkdir(parents=True, exist_ok=True)
    ks = ks_columna_met1(args.malla, args.tasa, run)
    k_fondo = ks[0]
    tab_c = tabla_carbono_met1(run, args.malla, args.tasa)

    # --- SI ---
    tab_si_b = _tabla_si(lambda r, k: si_baseline(r, k, args.malla), run, ks)
    tab_si_m = _tabla_si(si_mejorado, run, ks)
    si_lo = min(tab_si_b["valor"].min(), tab_si_m["valor"].min()) - 0.2
    si_hi = max(tab_si_b["valor"].max(), tab_si_m["valor"].max()) + 0.2
    norm_si_b = BoundaryNorm(np.linspace(si_lo, si_hi, 24), plt.cm.RdYlBu_r.N, clip=True)
    norm_si_m = TwoSlopeNorm(vmin=si_lo, vcenter=0.0, vmax=si_hi) if si_lo < 0 < si_hi else norm_si_b
    p_si = gif_comparacion_doble(
        tab_c, tab_si_b, tab_si_m, ks, args.malla, args.tasa, k_fondo,
        out / "gif_cmp_si_met1.gif",
        "SI — baseline vs mejorado",
        "SI dominante\n(Quartz res)",
        "SI Calcite k>=11\nQuartz k=10",
        plt.cm.RdYlBu_r, norm_si_b, norm_si_m,
    )
    fig_series_k15(tab_si_b, tab_si_m, args.malla, args.tasa,
                   out / "series_si_k15.png", "SI", "Indice de saturacion")

    # --- Porosidad ---
    tab_phi_b = _tabla_si(lambda r, k: dphi_baseline(r, k, args.malla), run, ks)
    tab_phi_m = _tabla_si(lambda r, k: dphi_mejorado(r, k, args.malla), run, ks)
    phi_max = max(tab_phi_m["valor"].max(), tab_phi_b["valor"].max(), 0.01) * 1.15
    lv = np.linspace(0, phi_max, 24)
    norm_phi = BoundaryNorm(lv, plt.cm.YlOrRd.N, clip=True)
    p_phi = gif_comparacion_doble(
        tab_c, tab_phi_b, tab_phi_m, ks, args.malla, args.tasa, k_fondo,
        out / "gif_cmp_porosidad_met1.gif",
        "Porosidad — baseline vs mejorado",
        r"$\Delta\phi/\phi_0$\neq cinetica",
        r"$\Delta\phi/\phi_0$ + silicatos",
        plt.cm.YlOrRd, norm_phi, norm_phi,
    )
    fig_series_k15(tab_phi_b, tab_phi_m, args.malla, args.tasa,
                   out / "series_porosidad_k15.png", r"$\Delta\phi/\phi_0$ (%)",
                   "Cambio de porosidad")

    # --- Diagnostico transporte (sin corrida dp aun) ---
    diag = diagnostic_transporte(args.malla, args.tasa, ks)
    diag.to_csv(out / "transporte_dp_extra.csv", index=False)
    if not diag.empty:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.scatter(diag["t"], diag["k_destino"], c=diag["P_orig"] - diag["P_dest"],
                   cmap="viridis", s=12, alpha=0.7)
        ax.set_xlabel("Tiempo (a)")
        ax.set_ylabel("Celda destino k")
        ax.set_title("Pasos extra de MIX con sg_o_dp (dP>0.5 bar, sin Sg en destino)")
        fig.colorbar(ax.collections[0], ax=ax, label="dP (bar)")
        fig.tight_layout()
        fig.savefig(out / "mapa_transporte_dp_extra.png", dpi=120)
        plt.close(fig)

    run_mej = Path(args.run_mejor)
    if (run_mej / "perfil_quimica_met1.csv").exists():
        from ccus_columna.postproceso import perfil_desde_met1
        df_b = pd.read_csv(run / "perfil_quimica_met1.csv")
        df_m = pd.read_csv(run_mej / "perfil_quimica_met1.csv")
        sub_b = df_b[df_b["k"] == K_INYECCION].sort_values("tiempo_anos")
        sub_m = df_m[df_m["k"] == K_INYECCION].sort_values("tiempo_anos")
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for ax, col, ylab in zip(
            axes,
            ["pH", "C_total_mol_kgw"],
            ["pH", "C(4) (mol/kgw)"],
        ):
            ax.plot(sub_b["tiempo_anos"], sub_b[col], "k--", label="sg only")
            ax.plot(sub_m["tiempo_anos"], sub_m[col], color="#1565c0", label="sg_o_dp")
            ax.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#fff3e0", alpha=0.35)
            ax.set_ylabel(ylab)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Tiempo (a)")
        axes[0].set_title(f"Quimica k15 — baseline vs completo_dp ({args.malla} q={args.tasa})")
        fig.tight_layout()
        fig.savefig(out / "series_quimica_transporte_k15.png", dpi=120)
        plt.close(fig)
        print(f"  quimica transporte: {out / 'series_quimica_transporte_k15.png'}")
    else:
        print(f"  (pendiente corrida {run_mej.name} para comparar quimica transporte)")

    resumen = {
        "si_k15": {
            "baseline_min": float(tab_si_b.loc[tab_si_b["k"] == K_INYECCION, "valor"].min()),
            "baseline_max": float(tab_si_b.loc[tab_si_b["k"] == K_INYECCION, "valor"].max()),
            "mejorado_min": float(tab_si_m.loc[tab_si_m["k"] == K_INYECCION, "valor"].min()),
            "mejorado_max": float(tab_si_m.loc[tab_si_m["k"] == K_INYECCION, "valor"].max()),
        },
        "dphi_k15_pct": {
            "baseline_max": float(tab_phi_b.loc[tab_phi_b["k"] == K_INYECCION, "valor"].max()),
            "mejorado_max": float(tab_phi_m.loc[tab_phi_m["k"] == K_INYECCION, "valor"].max()),
            "delta_max_pp": float(
                tab_phi_m.loc[tab_phi_m["k"] == K_INYECCION, "valor"].max()
                - tab_phi_b.loc[tab_phi_b["k"] == K_INYECCION, "valor"].max()
            ),
        },
        "transporte_dp_extra_pasos": int(len(diag)),
    }
    (out / "resumen_mejoras.json").write_text(
        json.dumps(resumen, indent=2), encoding="utf-8",
    )

    print(p_si)
    print(p_phi)
    print(f"  diagnosticos: {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(_RAIZ / "met1_celdas_encadenadas"))
    main()

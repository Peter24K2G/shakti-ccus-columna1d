"""Saturacion desde inventario de carbono PHREEQC (sin Sg_max).

Cuenta C en:
  - disuelto: C(4) * m_water
  - minerales: Calcite, Dawsonite (1 C), Ankerite/Dolomite (2 C)
  - gas: n = P_CO2 * V_gas / (R T) del GAS_PHASE acoplado (met1)

Saturacion equivalente (unidades Cirrus, fraccion del poro, sin Sg_max):
    S_calc = Delta_n_C * R T / (P V_poro)
    Delta_n_C = n_C(t) - mediana extraccion por celda

Proxy adicional:
    f_C = Delta_n_C / (Delta_n_C + n_H2O_mol + n_gas)

Uso:
  python comparacion/sg_carbono.py --malla malla_original --tasa <Q>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from ccus_columna.constantes import TASA_REF, TASAS_M3D, etiqueta_tasa  # parametros del proyecto
from ccus_columna import datos
from ccus_columna.constantes import (
    BAR_A_ATM,
    K_INYECCION,
    K_SELLO,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.mapa_saturacion import estado_poro
from ccus_columna.mineralogia import m_water_ref, v_poro
from ccus_columna.postproceso import _fila_final_paso, leer_selected_output
from ccus_columna.rutas import dir_figuras_corrida

R_L_ATM = 0.08205746
T_K = 60.0 + 273.15
MW_H2O = 0.01801528  # kg/mol
C_EN_MINERAL = {
    "calcite": 1,
    "dawsonite": 1,
    "ankerite": 2,
    "dolomite": 2,
}
MET1 = _RAIZ / "met1_celdas_encadenadas" / "corridas"
LABEL_Q = {q: etiqueta_tasa(q) for q in TASAS_M3D}


def ks_columna_met1(malla: str, tasa: int, run_dir: Path | None = None) -> list[int]:
    """Celdas de la corrida met1 (k profundo -> k10)."""
    if run_dir is not None:
        run_dir = Path(run_dir)
        perfil = run_dir / "perfil_quimica_met1.csv"
        if perfil.exists():
            ks = sorted(pd.read_csv(perfil)["k"].unique(), reverse=True)
            if ks:
                return [int(k) for k in ks]
        pat = re.compile(r"k(\d+)_p")
        ks_h = sorted(
            {int(m.group(1)) for p in (run_dir / "hitos").glob("k*_p*.txt")
             if (m := pat.search(p.name))},
            reverse=True,
        )
        if ks_h:
            return ks_h
        params = run_dir / "params.json"
        if params.exists():
            cfg = json.loads(params.read_text(encoding="utf-8"))
            if cfg.get("celdas"):
                return list(cfg["celdas"])
            dominio = cfg.get("dominio", "rama_sello")
            if dominio == "columna_pozo":
                return datos.columna_met1_sello()
            if dominio == "rama_sello":
                return datos.rama_al_sello(malla, tasa)
    return datos.columna_met1_sello()


def _mol_mineral(row: pd.Series, nombre: str) -> float:
    for alias in (nombre, nombre.lower()):
        if alias in row.index:
            v = row[alias]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0.0
            return float(v)
    return 0.0


def _n_carbono_fila(row: pd.Series, k: int, malla: str) -> tuple[float, float, float, float]:
    """n_C total en receta PHREEQC: (n_dis, n_min, n_gas, n_C)."""
    mw = m_water_ref(k, malla)
    n_dis = float(row.get("c(4)", 0.0) or 0.0) * mw
    n_min = 0.0
    for mineral, n_c in C_EN_MINERAL.items():
        n_min += _mol_mineral(row, mineral) * n_c
    return n_dis, n_min, 0.0, n_dis + n_min  # n_gas se completa afuera


def tabla_carbono_met1(run_dir: Path, malla: str, tasa: int) -> pd.DataFrame:
    """Inventario de C por (t, k) desde hitos met1."""
    run_dir = Path(run_dir)
    ks = ks_columna_met1(malla, tasa, run_dir)
    k_min, k_max = min(ks), max(ks)
    pasos_map: dict[int, float] = {}
    pasos_file = run_dir / "pasos.json"
    if pasos_file.exists():
        raw = json.loads(pasos_file.read_text(encoding="utf-8"))
        pasos_map = {int(p["paso"]): float(p["t_fin"]) for p in raw}

    sg_max = {
        k: float(datos.celda(malla, tasa, k)["Sg"].max())
        for k in range(k_min, k_max + 1)
    }

    filas = []
    pat = re.compile(r"k(\d+)_p(\d+)")
    for path in sorted((run_dir / "hitos").glob("k*_p*.txt")):
        m = pat.search(path.name)
        if not m:
            continue
        k, paso = int(m.group(1)), int(m.group(2))
        if k < k_min or k > k_max:
            continue
        t = pasos_map.get(paso, float("nan"))
        if not np.isfinite(t):
            continue
        row = _fila_final_paso(leer_selected_output(path))
        n_dis, n_min, _, n_c_pre = _n_carbono_fila(row, k, malla)
        s = datos.interpolar_celda(malla, tasa, k, t)
        est = estado_poro(
            s["Sg"], s["Sw"], s["So"], s["P_bar"], sg_max[k],
            usar_p_real=True, k=k, malla=malla,
        )
        n_gas = float(est["p_co2_atm"] * est["v_gas"] / (R_L_ATM * T_K))
        n_c = n_dis + n_min + n_gas
        mw = m_water_ref(k, malla)
        n_h2o = mw / MW_H2O
        filas.append({
            "tiempo_anos": t,
            "k": k,
            "n_C_mol": n_c,
            "n_dis_mol": n_dis,
            "n_min_mol": n_min,
            "n_gas_mol": n_gas,
            "n_H2O_mol": n_h2o,
            "Sg_cirrus": s["Sg"],
            "Sg_cirrus_pct": 100.0 * s["Sg"],
            "P_bar": s["P_bar"],
        })

    df = pd.DataFrame(filas).sort_values(["tiempo_anos", "k"])
    if df.empty:
        return df

    ext = df[df["tiempo_anos"] <= T_FIN_EXTRACCION + 0.01]
    c0 = ext.groupby("k")["n_C_mol"].median()
    df = df.copy()
    df["n_C_delta"] = df.apply(
        lambda r: float(r["n_C_mol"] - c0.get(int(r["k"]), 0.0)), axis=1,
    )
    df["n_C_delta"] = df["n_C_delta"].clip(lower=0.0)

    for i, row in df.iterrows():
        k = int(row["k"])
        v_poro_l = v_poro(k, malla)
        p_atm = max(float(row["P_bar"]) * BAR_A_ATM, 1e-6)
        n_cap = p_atm * v_poro_l / (R_L_ATM * T_K)
        n_c = float(row["n_C_delta"])
        n_gas = float(row["n_gas_mol"])
        n_h2o = float(row["n_H2O_mol"])
        n_total = n_c + n_h2o + n_gas
        df.at[i, "S_calc"] = float(np.clip(n_c / max(n_cap, 1e-30), 0.0, 1.5))
        df.at[i, "S_calc_pct"] = 100.0 * df.at[i, "S_calc"]
        df.at[i, "n_total_mol"] = n_total
        df.at[i, "f_C"] = float(np.clip(n_c / max(n_total, 1e-30), 0.0, 1.0))

    return df


def _perfil_en_t(tab: pd.DataFrame, col: str, t_anos: float, ks: list[int]) -> np.ndarray:
    out = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        sub = tab[tab["k"].astype(int) == int(k)].sort_values("tiempo_anos")
        if sub.empty:
            continue
        out[i] = float(np.interp(
            t_anos,
            sub["tiempo_anos"].to_numpy(dtype=float),
            sub[col].to_numpy(dtype=float),
            left=float(sub[col].iloc[0]),
            right=float(sub[col].iloc[-1]),
        ))
    return out


def _etapa(t: float) -> str:
    if t < T_FIN_EXTRACCION:
        return "extraccion"
    if t < T_FIN_INYECCION:
        return "inyeccion"
    return "shut-in"


def _tabla_ph_met1(run_dir: Path, malla: str, tasa: int, ks: list[int]) -> pd.DataFrame:
    """pH y Delta pH desde perfil_quimica_met1.csv."""
    perfil = Path(run_dir) / "perfil_quimica_met1.csv"
    if not perfil.exists():
        raise FileNotFoundError(f"Falta {perfil}; correr --solo-exportar primero.")
    df = pd.read_csv(perfil)
    df = df[df["k"].isin(ks)].copy()
    ext = df[df["tiempo_anos"] <= T_FIN_EXTRACCION + 0.01]
    p0 = ext.groupby("k")["pH"].median()
    df["pH0"] = df["k"].map(p0)
    df["dpH"] = df["pH"] - df["pH0"]
    return df.sort_values(["tiempo_anos", "k"])


def gif_sg_carbono_met1(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    run_dir: Path | None = None,
    n_frames: int = 60,
    fps: int = 6,
    out_dir: Path | None = None,
) -> Path:
    """GIF met1: perfil carbono (izq) + columna pH por colores (der), sincronizados."""
    run_dir = Path(run_dir) if run_dir else MET1 / f"{malla}_q{tasa}_completo"
    tab = tabla_carbono_met1(run_dir, malla, tasa)
    tab_ph = _tabla_ph_met1(run_dir, malla, tasa, ks_columna_met1(malla, tasa, run_dir))
    ks = ks_columna_met1(malla, tasa, run_dir)
    k_fondo = ks[0]
    ks_arr = np.array(ks, dtype=float)
    t_end = float(tab["tiempo_anos"].max())
    t_anim = np.linspace(0.0, t_end, n_frames)

    x_f = max(float(np.nanpercentile(tab["f_C"], 99)) * 1.08, 0.01)
    x_sg = max(float(tab["Sg_cirrus_pct"].max()) * 1.08, 5.0)
    ph_all = tab_ph["pH"].to_numpy(dtype=float)
    ph_vmin = float(np.floor(np.nanmin(ph_all) * 100) / 100) - 0.02
    ph_vmax = float(np.ceil(np.nanmax(ph_all) * 100) / 100) + 0.02
    n_niveles = 32
    ph_levels = np.linspace(ph_vmin, ph_vmax, n_niveles)
    cmap_ph = plt.cm.RdYlBu_r
    norm_ph = BoundaryNorm(ph_levels, cmap_ph.N, clip=True)

    out_dir = Path(out_dir) if out_dir else dir_figuras_corrida("met1", run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax_ph) = plt.subplots(
        1, 2, figsize=(11.2, 5.4),
        gridspec_kw={"width_ratios": [4.2, 0.75], "wspace": 0.06},
    )
    axb = ax.twiny()
    line_f, = ax.plot([], [], "o-", color="#1565c0", ms=8, lw=2.2,
                      label=r"$\Delta n_C / n_{total}$ (proxy)")
    line_s, = axb.plot([], [], "s--", color="#e65100", ms=7, lw=1.8,
                       label="Sg Cirrus")
    fill = {"poly": None}

    y_lo, y_hi = ks_arr.min() - 0.5, ks_arr.max() + 0.5
    ax.set_xlim(0, x_f)
    ax.set_ylim(y_hi, y_lo)
    ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.8, alpha=0.8)
    ax.axhline(k_fondo, color="#6a1b9a", ls=":", lw=0.8, alpha=0.8)
    ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.8, alpha=0.8)
    ax.set_xlabel(r"Proxy $\Delta n_C / n_{total}$   (C extra vs extraccion)")
    ax.set_ylabel("Celda k (sello arriba)")
    ax.grid(True, alpha=0.3)

    axb.set_xlim(0, x_sg)
    axb.set_xlabel("Sg Cirrus [% poro]", color="#e65100")
    axb.tick_params(axis="x", colors="#e65100")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

    ax_ph.set_xlim(0, 1)
    ax_ph.set_ylim(y_hi, y_lo)
    ax_ph.set_xticks([])
    ax_ph.set_xlabel("pH", fontsize=9)
    ax_ph.set_ylabel("")
    ax_ph.tick_params(axis="y", labelleft=False)
    ax_ph.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.8, alpha=0.5)
    ax_ph.axhline(k_fondo, color="#6a1b9a", ls=":", lw=0.8, alpha=0.5)
    ax_ph.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.8, alpha=0.5)
    ph_init = np.full((len(ks), 1), ph_vmax)
    im_ph = ax_ph.imshow(
        ph_init,
        aspect="auto",
        cmap=cmap_ph,
        norm=norm_ph,
        extent=[0, 1, y_lo, y_hi],
        origin="upper",
        interpolation="nearest",
        animated=True,
    )
    divider = make_axes_locatable(ax_ph)
    cax = divider.append_axes("right", size="18%", pad=0.08)
    cbar = fig.colorbar(im_ph, cax=cax)
    cbar.set_label(f"pH (fija {ph_vmin:.2f}–{ph_vmax:.2f})", fontsize=8)
    cbar.set_ticks(ph_levels[::4])

    titulo = fig.suptitle("", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.82, bottom=0.14, wspace=0.18)

    def update(frame: int):
        tt = float(t_anim[frame])
        f = _perfil_en_t(tab, "f_C", tt, ks)
        sg = _perfil_en_t(tab, "Sg_cirrus_pct", tt, ks)
        ph_col = np.round(_perfil_en_t(tab_ph, "pH", tt, ks), 2)
        line_f.set_data(f, ks_arr)
        line_s.set_data(sg, ks_arr)
        if fill["poly"] is not None:
            fill["poly"].remove()
        fill["poly"] = ax.fill_betweenx(
            ks_arr, 0, np.nan_to_num(f, nan=0.0), color="#90caf9", alpha=0.35,
        )
        im_ph.set_data(ph_col.reshape(-1, 1))
        s_calc_k15 = _perfil_en_t(tab, "S_calc_pct", tt, [K_INYECCION])[0]
        titulo.set_text(
            f"met1 — C total + pH  |  {malla} q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t = {tt:.1f} a ({_etapa(tt)})  |  "
            rf"k{k_fondo}->k10  |  $S_{{calc}}$ k15={s_calc_k15:.1f}%"
        )
        return line_f, line_s, titulo

    path = out_dir / "gif_sg_carbono_met1.gif"
    from matplotlib import animation
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

    csv_path = out_dir / "sg_carbono_met1.csv"
    tab.to_csv(csv_path, index=False, float_format="%.6g")
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="GIF Sg desde carbono total (met1)")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    run = args.run_dir or MET1 / f"{args.malla}_q{args.tasa}_completo"
    p = gif_sg_carbono_met1(args.malla, args.tasa, run, args.n_frames, args.fps)
    print(p)

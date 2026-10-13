"""Figuras utiles: CO2 en volumen de PORO y % saturacion equivalente (k15→k10).

El almacenamiento no usa V_bulk (roca+poro): solo V_poro = phi * V_bulk.
n_sol usa C en exceso sobre el fondo nativo de extraccion (C0), no el C
absoluto (~0.002 mol/kgw de alcalinidad, presente aunque Sg=0 en Cirrus):

    n_sol    = max(0, C - C0) * m_water_ref * factor
    S_aq_eq  = n_sol * R * T / (P_atm * V_poro)
    S_co2_eq = (n_sol + n_gas) * R * T / (P_atm * V_poro)

Solo se grafica la rama al sello k=15..k=10 (aunque met2 simule perforacion).

Salida en figuras/comparacion/{malla}/q{tasa}/utiles/:

  CO2_saturacion_heatmap.png
  CO2_saturacion_series.png
  C4_heatmap.png          (ΔC = C - C0)
  calcita_pct.png
  porosidad_heatmap.png   (Δφ / φ0 %)
  porosidad_series.png
  co2_por_celda.csv
  gif_sco2_eq_met1_met2.gif   (S_CO2,calc desde C(4), DC-016; no n_gas Cirrus)
  gif_c4_met1_met2.gif          (perfil ΔC = C−C0 vs k, animado)

Uso:
  python comparacion/figuras_utiles.py --malla malla_original --tasa <Q>
  python comparacion/figuras_utiles.py --malla malla_original --tasa <Q> --gif
  python comparacion/figuras_utiles.py --todas
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

_RAIZ = Path(__file__).resolve().parents[1]
_COMP = Path(__file__).resolve().parent
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))
if str(_COMP) not in sys.path:
    sys.path.insert(0, str(_COMP))

from ccus_columna.constantes import TASA_REF, TASAS_M3D, etiqueta_tasa  # parametros del proyecto
from ccus_columna import datos
from ccus_columna.constantes import (
    BAR_A_ATM,
    K_INYECCION,
    K_SELLO,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.geometria import factor_moles_campo, v_bulk_m3
from ccus_columna.mapa_saturacion import estado_poro
from ccus_columna.mineralogia import m_water_ref, phi0
from ccus_columna.postproceso import anotar_delta_calcita
from ccus_columna.rutas import dir_figuras_comparacion
from sg_aparente import baseline_c0, calibrar_c_star, sg_calc_en_t, sg_max_celdas

MET1 = _RAIZ / "met1_celdas_encadenadas" / "corridas"
MET2 = _RAIZ / "met2_transporte_phreeqc" / "corridas"
LABEL_Q = {q: etiqueta_tasa(q) for q in TASAS_M3D}
R_L_ATM = 0.08205746          # L·atm/(mol·K)
R_M3_ATM = R_L_ATM * 1e-3     # m3·atm/(mol·K)
T_K = 60.0 + 273.15


def _cargar(malla: str, tasa: int, met: str) -> pd.DataFrame:
    if met == "met1":
        path = MET1 / f"{malla}_q{tasa}_completo" / "perfil_quimica_met1.csv"
    else:
        path = MET2 / f"{malla}_q{tasa}_reactivo" / "perfil_quimica_met2.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = anotar_delta_calcita(pd.read_csv(path), malla)
    # Solo rama al sello (graficas); el CSV de corrida puede tener k>15
    return df[(df["k"] >= K_SELLO) & (df["k"] <= K_INYECCION)].copy()


def _c0_extraccion(df: pd.DataFrame) -> float:
    """C nativo (alcalinidad) en extraccion; no es CO2 inyectado."""
    sub = df[df["tiempo_anos"] <= T_FIN_EXTRACCION]
    if sub.empty:
        sub = df.nsmallest(max(1, len(df) // 20), "tiempo_anos")
    return float(sub["C_total_mol_kgw"].median())


def tabla_por_celda(df: pd.DataFrame, malla: str, tasa: int, met: str) -> pd.DataFrame:
    """S_CO2,eq usa C en exceso sobre C0 (extraccion), no el C nativo."""
    ks = sorted(int(k) for k in df["k"].unique())
    sg_max = {k: float(datos.celda(malla, tasa, k)["Sg"].max()) for k in ks}
    c0 = _c0_extraccion(df)
    filas = []
    for _, row in df.iterrows():
        k = int(row["k"])
        t = float(row["tiempo_anos"])
        phi = phi0(k)
        v_bulk = v_bulk_m3(malla, k)
        v_poro = phi * v_bulk
        f = factor_moles_campo(malla, k)
        s = datos.interpolar_celda(malla, tasa, k, t)
        est = estado_poro(
            s["Sg"], s["Sw"], s["So"], s["P_bar"], sg_max[k],
            usar_p_real=True, k=k, malla=malla,
        )
        c = float(row["C_total_mol_kgw"])
        c_ex = max(0.0, c - c0)
        # Solo el carbono por encima del fondo nativo cuenta como "almacenado"
        n_sol = c_ex * m_water_ref(k, malla) * f
        n_gas = float(est["p_co2_atm"] * est["v_gas"] / (R_L_ATM * T_K)) * f
        dcal = float(row.get("delta_calcita_mol", 0.0) or 0.0)
        n_min = max(0.0, dcal) * f
        p_atm = max(s["P_bar"] * BAR_A_ATM, 1e-6)
        dphi = float(row.get("delta_phi", 0.0) or 0.0)
        # saturaciones equivalentes (fraccion del PORO, no del bulk)
        s_aq = (n_sol * R_M3_ATM * T_K) / (p_atm * max(v_poro, 1e-30))
        s_gas_eq = (n_gas * R_M3_ATM * T_K) / (p_atm * max(v_poro, 1e-30))
        s_co2 = (n_sol + n_gas) * R_M3_ATM * T_K / (p_atm * max(v_poro, 1e-30))
        filas.append({
            "met": met,
            "tiempo_anos": t,
            "k": k,
            "phi": phi,
            "V_bulk_m3": v_bulk,
            "V_poro_m3": v_poro,
            "Sg": s["Sg"],
            "Sw": s["Sw"],
            "So": s["So"],
            "P_bar": s["P_bar"],
            "C_mol_kgw": c,
            "C0_mol_kgw": c0,
            "C_exceso_mol_kgw": c_ex,
            "pct_calcita": float(row.get("pct_calcita", np.nan) or np.nan),
            "delta_phi": dphi,
            "delta_phi_pct": 100.0 * dphi / max(phi, 1e-12),
            "n_sol_mol": n_sol,
            "n_gas_mol": n_gas,
            "n_min_mol": n_min,
            "I_alm_mol": n_sol + n_gas + n_min,
            "S_aq_eq": s_aq,
            "S_gas_eq": s_gas_eq,
            "S_co2_eq": s_co2,
            "Sg_pct": 100.0 * s["Sg"],
            "S_aq_eq_pct": 100.0 * s_aq,
            "S_co2_eq_pct": 100.0 * s_co2,
        })
    return pd.DataFrame(filas).sort_values(["tiempo_anos", "k"])


def _heatmap(tab: pd.DataFrame, col: str, ax, titulo: str, cmap: str,
             vmin=None, vmax=None, cbar_label: str = ""):
    piv = tab.pivot_table(index="tiempo_anos", columns="k", values=col)
    t = piv.index.to_numpy()
    ks = np.array(sorted(piv.columns, reverse=True))
    Z = np.column_stack([piv[k].to_numpy() for k in ks])
    im = ax.pcolormesh(t, ks, Z.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.8)
    ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.8)
    ax.axvline(T_FIN_EXTRACCION, color="0.45", ls=":", lw=0.7)
    ax.axvline(T_FIN_INYECCION, color="0.45", ls=":", lw=0.7)
    ax.set_xlabel("tiempo [a]")
    ax.set_ylabel("k")
    ax.set_title(titulo, fontsize=10)
    return im, cbar_label


def generar(malla: str = "malla_original", tasa: int = TASA_REF) -> list[Path]:
    df1 = _cargar(malla, tasa, "met1")
    df2 = _cargar(malla, tasa, "met2")
    t1 = tabla_por_celda(df1, malla, tasa, "met1")
    t2 = tabla_por_celda(df2, malla, tasa, "met2")

    out = dir_figuras_comparacion(malla, tasa) / "utiles"
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    csv_path = out / "co2_por_celda.csv"
    pd.concat([t1, t2], ignore_index=True).to_csv(csv_path, index=False, float_format="%.6g")
    paths.append(csv_path)

    # --- 1. Saturacion equivalente CO2 (% del poro) ---
    vmax = max(
        float(np.nanpercentile(t1["S_co2_eq_pct"], 99)),
        float(np.nanpercentile(t2["S_co2_eq_pct"], 99)),
        5.0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, tab, lab in (
        (axes[0], t1, "met1"),
        (axes[1], t2, "met2"),
    ):
        im, _ = _heatmap(
            tab, "S_co2_eq_pct", ax,
            f"{lab} — $S_{{CO2,eq}}$ % del poro (k15→k10)",
            "YlOrRd", vmin=0, vmax=vmax,
        )
        fig.colorbar(im, ax=ax, fraction=0.046).set_label("% poro", fontsize=8)
    fig.suptitle(
        f"Almacenamiento CO$_2$ como % saturacion equiv. del PORO "
        f"(φ·V$_{{bulk}}$)  |  {malla} q={LABEL_Q.get(tasa, tasa)}  |  k15→k10\n"
        r"$S_{CO2,eq}=(n_{sol}^{exceso}+n_{gas})RT/(P\,V_{poro})$;  "
        r"$n_{sol}$ usa $C-C_0$ (fondo nativo en extraccion = 0)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = out / "CO2_saturacion_heatmap.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- 2. Series: S_co2_eq vs Sg en k15..k10 ---
    ks_plot = list(range(K_INYECCION, K_SELLO - 1, -1))  # 15..10
    fig, axes = plt.subplots(len(ks_plot), 1, figsize=(9, 2.0 * len(ks_plot)), sharex=True)
    if len(ks_plot) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks_plot):
        a = t1[t1["k"] == k].sort_values("tiempo_anos")
        b = t2[t2["k"] == k].sort_values("tiempo_anos")
        if not a.empty:
            ax.plot(a["tiempo_anos"], a["S_co2_eq_pct"], color="#c62828", lw=1.6,
                    label="met1 $S_{CO2,eq}$")
            ax.plot(a["tiempo_anos"], a["Sg_pct"], color="#e65100", ls="--", lw=1.2,
                    label="Sg Cirrus")
        if not b.empty:
            ax.plot(b["tiempo_anos"], b["S_co2_eq_pct"], color="#1565c0", lw=1.6,
                    label="met2 $S_{CO2,eq}$")
        ax.axvline(T_FIN_EXTRACCION, color="0.5", ls=":", lw=0.7)
        ax.axvline(T_FIN_INYECCION, color="0.5", ls=":", lw=0.7)
        vporo = phi0(k) * v_bulk_m3(malla, k)
        ax.set_ylabel("%")
        ax.set_title(f"k={k}  V_poro={vporo:.0f} m$^3$  (φ={phi0(k):.3f})", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", ncol=3)
    axes[-1].set_xlabel("tiempo [a]")
    fig.suptitle(
        f"$S_{{CO2,eq}}$ vs Sg  |  {malla} q={LABEL_Q.get(tasa, tasa)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out / "CO2_saturacion_series.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- 3. C(4) exceso (= C - C0): extraccion ~0; pluma en inyeccion ---
    c0_1 = float(t1["C0_mol_kgw"].iloc[0])
    c0_2 = float(t2["C0_mol_kgw"].iloc[0])
    cmax = max(
        float(t1["C_exceso_mol_kgw"].max()),
        float(t2["C_exceso_mol_kgw"].max()),
        0.002,
    ) * 1.05
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, tab, lab in (
        (axes[0], t1, "met1"),
        (axes[1], t2, "met2"),
    ):
        im, _ = _heatmap(
            tab, "C_exceso_mol_kgw", ax,
            f"{lab} — ΔC = C(4)−C$_0$ mol/kgw", "YlOrRd",
            vmin=0, vmax=cmax,
        )
        fig.colorbar(im, ax=ax, fraction=0.046).set_label("mol/kgw", fontsize=8)
    fig.suptitle(
        f"ΔC(4) (exceso sobre fondo nativo)  |  {malla} q={LABEL_Q.get(tasa, tasa)}  |  "
        f"C$_0$: met1={c0_1:.4f}  met2={c0_2:.4f} mol/kgw\n"
        "En extraccion (Sg=0) ΔC≈0; el C absoluto ~0.002 es alcalinidad nativa, no CO$_2$ inyectado",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = out / "C4_heatmap.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- 4. Calcita % ---
    fin = np.concatenate([
        t1["pct_calcita"].to_numpy(dtype=float),
        t2["pct_calcita"].to_numpy(dtype=float),
    ])
    fin = fin[np.isfinite(fin)]
    lim = max(float(np.nanmax(np.abs(fin))) if fin.size else 1e-3, 1e-3)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, tab, lab in (
        (axes[0], t1, "met1"),
        (axes[1], t2, "met2"),
    ):
        im, _ = _heatmap(
            tab, "pct_calcita", ax,
            f"{lab} — Δ calcita / m0 (%)", "RdBu", vmin=-lim, vmax=lim,
        )
        fig.colorbar(im, ax=ax, fraction=0.046).set_label("%", fontsize=8)
    fig.suptitle(
        f"Calcita (%)  |  {malla} q={LABEL_Q.get(tasa, tasa)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = out / "calcita_pct.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # --- 5. Porosidad: Δφ / φ0 (%) ---
    phi_all = np.concatenate([
        t1["delta_phi_pct"].to_numpy(dtype=float),
        t2["delta_phi_pct"].to_numpy(dtype=float),
    ])
    phi_all = phi_all[np.isfinite(phi_all)]
    lim_phi = max(float(np.nanmax(np.abs(phi_all))) if phi_all.size else 1e-3, 1e-3)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, tab, lab in (
        (axes[0], t1, "met1"),
        (axes[1], t2, "met2"),
    ):
        im, _ = _heatmap(
            tab, "delta_phi_pct", ax,
            f"{lab} — Δφ / φ$_0$ (%)", "RdBu_r",
            vmin=-lim_phi, vmax=lim_phi,
        )
        fig.colorbar(im, ax=ax, fraction=0.046).set_label("% de φ₀", fontsize=8)
    fig.suptitle(
        f"Cambio de porosidad  |  {malla} q={LABEL_Q.get(tasa, tasa)}  |  k15→k10\n"
        r"$\Delta\phi>0$ disolucion (abre poro);  $\Delta\phi<0$ precipitacion (cierra);  "
        r"$\phi_0$ por celda (sello $\neq$ reservorio)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    p = out / "porosidad_heatmap.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    ks_plot = list(range(K_INYECCION, K_SELLO - 1, -1))
    fig, axes = plt.subplots(len(ks_plot), 1, figsize=(9, 2.0 * len(ks_plot)), sharex=True)
    if len(ks_plot) == 1:
        axes = [axes]
    for ax, k in zip(axes, ks_plot):
        a = t1[t1["k"] == k].sort_values("tiempo_anos")
        b = t2[t2["k"] == k].sort_values("tiempo_anos")
        if not a.empty:
            ax.plot(a["tiempo_anos"], a["delta_phi_pct"], color="#c62828", lw=1.6,
                    label="met1")
        if not b.empty:
            ax.plot(b["tiempo_anos"], b["delta_phi_pct"], color="#1565c0", lw=1.6,
                    label="met2")
        ax.axhline(0.0, color="0.4", lw=0.7)
        ax.axvline(T_FIN_EXTRACCION, color="0.5", ls=":", lw=0.7)
        ax.axvline(T_FIN_INYECCION, color="0.5", ls=":", lw=0.7)
        ax.set_ylabel("%")
        ax.set_title(f"k={k}  φ$_0$={phi0(k):.4f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("tiempo [a]")
    fig.suptitle(
        f"Δφ / φ$_0$ (%) por celda  |  {malla} q={LABEL_Q.get(tasa, tasa)}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out / "porosidad_series.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


def _perfil_sco2_en_t(tab: pd.DataFrame, t_anos: float, ks: list[int]) -> np.ndarray:
    out = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        sub = tab[tab["k"].astype(int) == int(k)].sort_values("tiempo_anos")
        if sub.empty:
            continue
        out[i] = float(np.interp(
            t_anos,
            sub["tiempo_anos"].to_numpy(dtype=float),
            sub["S_co2_eq_pct"].to_numpy(dtype=float),
            left=float(sub["S_co2_eq_pct"].iloc[0]),
            right=float(sub["S_co2_eq_pct"].iloc[-1]),
        ))
    return out


def _etapa(t: float) -> str:
    if t < T_FIN_EXTRACCION:
        return "extraccion"
    if t < T_FIN_INYECCION:
        return "inyeccion"
    return "shut-in"


def gif_sco2_eq(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    n_frames: int = 60,
    fps: int = 6,
    out_dir: Path | None = None,
) -> Path:
    """GIF: S_CO2,calc desde C(4) (DC-016) — sin n_gas de Cirrus.

    S_CO2,calc(k,t) = 100 * Sg_max(k) * clip((C-C0)/C*, 0, 1)
    C* calibrado con met1 vs Sg (solo escala); el perfil de cada metodo
    sale de su propio C(4).
    """
    df1 = _cargar(malla, tasa, "met1")
    df2 = _cargar(malla, tasa, "met2")
    ks = list(range(K_INYECCION, K_SELLO - 1, -1))
    ks_arr = np.array(ks, dtype=float)
    sg_max = sg_max_celdas(malla, tasa, ks)
    c0_1 = baseline_c0(df1, ks)
    c0_2 = baseline_c0(df2, ks)
    c_star = calibrar_c_star(df1, malla, tasa, c0=c0_1, sg_max=sg_max)

    t_end = min(float(df1["tiempo_anos"].max()), float(df2["tiempo_anos"].max()))
    t_anim = np.linspace(0.0, t_end, n_frames)
    xlim = max(max(sg_max.values()) * 100.0 * 1.08, 5.0)

    # Sg Cirrus solo como referencia visual (no entra en S_calc)
    p_sg = datos.perfil(malla, tasa, "Sg")
    t_sg = p_sg["tiempo_anos"].to_numpy(dtype=float)
    Z_sg = p_sg[[f"k{int(k)}" for k in ks]].to_numpy(dtype=float)

    out_dir = Path(out_dir) if out_dir else dir_figuras_comparacion(malla, tasa) / "utiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.4), sharey=True)
    line1, = ax1.plot([], [], "o-", color="#c62828", ms=8, lw=2.2,
                      label=r"$S_{CO2,calc}$ met1")
    line2, = ax2.plot([], [], "o-", color="#1565c0", ms=8, lw=2.2,
                      label=r"$S_{CO2,calc}$ met2")
    line_ref1, = ax1.plot([], [], "s--", color="#e65100", ms=5, lw=1.2, alpha=0.7,
                          label="Sg Cirrus (ref)")
    line_ref2, = ax2.plot([], [], "s--", color="#e65100", ms=5, lw=1.2, alpha=0.7,
                          label="Sg Cirrus (ref)")
    fills: dict[str, object] = {"1": None, "2": None}

    for ax in (ax1, ax2):
        ax.set_xlim(0, xlim)
        ax.set_ylim(ks_arr.max() + 0.5, ks_arr.min() - 0.5)
        ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.9, alpha=0.8)
        ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.9, alpha=0.8)
        ax.set_xlabel(r"$S_{CO2,calc}$  [% poro]   (desde C(4), misma escala)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    ax1.set_ylabel("Celda k (sello arriba)")
    ax1.set_title("met1 — $S_{CO2,calc}$ desde C(4)")
    ax2.set_title("met2 — $S_{CO2,calc}$ desde C(4)")
    titulo = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    def update(frame: int):
        tt = float(t_anim[frame])
        s1 = 100.0 * sg_calc_en_t(df1, tt, ks, c0_1, c_star, sg_max)
        s2 = 100.0 * sg_calc_en_t(df2, tt, ks, c0_2, c_star, sg_max)
        sg = 100.0 * np.array([
            float(np.interp(tt, t_sg, Z_sg[:, j])) for j in range(len(ks))
        ])
        line1.set_data(s1, ks_arr)
        line2.set_data(s2, ks_arr)
        line_ref1.set_data(sg, ks_arr)
        line_ref2.set_data(sg, ks_arr)
        for key, ax, vals, color in (
            ("1", ax1, s1, "#ef9a9a"),
            ("2", ax2, s2, "#90caf9"),
        ):
            if fills[key] is not None:
                fills[key].remove()
            fills[key] = ax.fill_betweenx(
                ks_arr, 0, np.nan_to_num(vals, nan=0.0), color=color, alpha=0.35,
            )
        titulo.set_text(
            f"$S_{{CO2,calc}}$ desde C(4)  |  {malla}  q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t = {tt:.1f} a  ({_etapa(tt)})  |  C*={c_star:.4f}  |  k15→k10"
        )
        return line1, line2, line_ref1, line_ref2, titulo

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )
    path = out_dir / "gif_sco2_eq_met1_met2.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def _perfil_cex_en_t(tab: pd.DataFrame, t_anos: float, ks: list[int]) -> np.ndarray:
    out = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        sub = tab[tab["k"].astype(int) == int(k)].sort_values("tiempo_anos")
        if sub.empty:
            continue
        out[i] = float(np.interp(
            t_anos,
            sub["tiempo_anos"].to_numpy(dtype=float),
            sub["C_exceso_mol_kgw"].to_numpy(dtype=float),
            left=float(sub["C_exceso_mol_kgw"].iloc[0]),
            right=float(sub["C_exceso_mol_kgw"].iloc[-1]),
        ))
    return out


def gif_c4(
    malla: str = "malla_original",
    tasa: int = TASA_REF,
    n_frames: int = 60,
    fps: int = 6,
    out_dir: Path | None = None,
) -> Path:
    """GIF dos paneles: ΔC = C(4)−C0 met1 | met2 (misma escala, k15→k10)."""
    df1 = _cargar(malla, tasa, "met1")
    df2 = _cargar(malla, tasa, "met2")
    t1 = tabla_por_celda(df1, malla, tasa, "met1")
    t2 = tabla_por_celda(df2, malla, tasa, "met2")
    ks = list(range(K_INYECCION, K_SELLO - 1, -1))
    ks_arr = np.array(ks, dtype=float)
    t_end = min(float(t1["tiempo_anos"].max()), float(t2["tiempo_anos"].max()))
    t_anim = np.linspace(0.0, t_end, n_frames)
    c0_1 = float(t1["C0_mol_kgw"].iloc[0])
    c0_2 = float(t2["C0_mol_kgw"].iloc[0])
    xlim = max(
        float(t1["C_exceso_mol_kgw"].max()),
        float(t2["C_exceso_mol_kgw"].max()),
        0.002,
    ) * 1.08

    # Sg Cirrus de referencia (misma rama)
    p_sg = datos.perfil(malla, tasa, "Sg")
    t_sg = p_sg["tiempo_anos"].to_numpy(dtype=float)
    Z_sg = p_sg[[f"k{int(k)}" for k in ks]].to_numpy(dtype=float)
    sg_xlim = max(float(np.nanmax(Z_sg)) * 1.05, 0.15)

    out_dir = Path(out_dir) if out_dir else dir_figuras_comparacion(malla, tasa) / "utiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.4), sharey=True)
    line1, = ax1.plot([], [], "o-", color="#c62828", ms=8, lw=2.2, label="ΔC met1")
    line2, = ax2.plot([], [], "o-", color="#1565c0", ms=8, lw=2.2, label="ΔC met2")
    ax1b = ax1.twiny()
    ax2b = ax2.twiny()
    line_sg1, = ax1b.plot([], [], "s--", color="#e65100", ms=5, lw=1.3, alpha=0.75,
                          label="Sg Cirrus")
    line_sg2, = ax2b.plot([], [], "s--", color="#e65100", ms=5, lw=1.3, alpha=0.75,
                          label="Sg Cirrus")
    fills: dict[str, object] = {"1": None, "2": None}

    for ax, axb in ((ax1, ax1b), (ax2, ax2b)):
        ax.set_xlim(0, xlim)
        ax.set_ylim(ks_arr.max() + 0.5, ks_arr.min() - 0.5)
        ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.9, alpha=0.8)
        ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.9, alpha=0.8)
        ax.set_xlabel(r"$\Delta C = C(4)-C_0$  [mol/kgw]   ← misma escala")
        ax.grid(True, alpha=0.3)
        axb.set_xlim(0, sg_xlim)
        axb.set_xlabel(r"$S_g$ Cirrus (ref)", color="#e65100", fontsize=8)
        axb.tick_params(axis="x", colors="#e65100", labelsize=8)

    ax1.set_ylabel("Celda k (sello arriba)")
    ax1.set_title(f"met1 — saturaciones  (C$_0$={c0_1:.4f})")
    ax2.set_title(f"met2 — TRANSPORT  (C$_0$={c0_2:.4f})")
    for ax, axb in ((ax1, ax1b), (ax2, ax2b)):
        h, l = ax.get_legend_handles_labels()
        hb, lb = axb.get_legend_handles_labels()
        ax.legend(h + hb, l + lb, fontsize=7, loc="lower right")

    titulo = fig.suptitle("", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    def update(frame: int):
        tt = float(t_anim[frame])
        c1 = _perfil_cex_en_t(t1, tt, ks)
        c2 = _perfil_cex_en_t(t2, tt, ks)
        sg = np.array([float(np.interp(tt, t_sg, Z_sg[:, j])) for j in range(len(ks))])
        line1.set_data(c1, ks_arr)
        line2.set_data(c2, ks_arr)
        line_sg1.set_data(sg, ks_arr)
        line_sg2.set_data(sg, ks_arr)
        for key, ax, vals, color in (
            ("1", ax1, c1, "#ef9a9a"),
            ("2", ax2, c2, "#90caf9"),
        ):
            if fills[key] is not None:
                fills[key].remove()
            fills[key] = ax.fill_betweenx(
                ks_arr, 0, np.nan_to_num(vals, nan=0.0), color=color, alpha=0.35,
            )
        titulo.set_text(
            f"$\\Delta C(4)$ comparable  |  {malla}  q={LABEL_Q.get(tasa, tasa)}  "
            f"|  t = {tt:.1f} a  ({_etapa(tt)})  |  k15→k10"
        )
        return line1, line2, line_sg1, line_sg2, titulo

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False,
    )
    path = out_dir / "gif_c4_met1_met2.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def tasas_disponibles(malla: str) -> list[int]:
    out = []
    for p2 in sorted(MET2.glob(f"{malla}_q*_reactivo")):
        mid = p2.name[len(malla) + 1 :]
        q = mid.split("_")[0][1:]
        if not q.isdigit():
            continue
        tasa = int(q)
        if (MET1 / f"{malla}_q{tasa}_completo" / "perfil_quimica_met1.csv").exists():
            if (p2 / "perfil_quimica_met2.csv").exists():
                out.append(tasa)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Figuras utiles CO2/saturacion/C4/calcita")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--todas", action="store_true")
    ap.add_argument("--gif", action="store_true",
                    help="Generar GIFs S_CO2,eq y ΔC(4)")
    ap.add_argument("--solo-gif", action="store_true",
                    help="Solo los GIFs (sin PNG/CSV)")
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()
    tasas = tasas_disponibles(args.malla) if args.todas else [args.tasa]
    for q in tasas:
        print(f"=== {args.malla} q={q} ===")
        if not args.solo_gif:
            for p in generar(args.malla, q):
                print(f"  {p}")
        if args.gif or args.solo_gif:
            print(f"  {gif_sco2_eq(args.malla, q, args.n_frames, args.fps)}")
            print(f"  {gif_c4(args.malla, q, args.n_frames, args.fps)}")

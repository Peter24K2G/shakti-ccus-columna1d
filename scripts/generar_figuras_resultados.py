"""Genera figuras y GIFs de la columna Cirrus + quimica 0D (celda k15).

Salidas en figuras/cirrus/:
  C1_heatmap_sg_tasas.png       — mapas k-t de Sg (4 tasas)
  C2_heatmap_presion.png        — mapa k-t de P
  C3_llegada_frente.png         — primer gas por celda
  C4_quimica_0d_tasas.png       — pH, calcita, C, Sg (reactor 0D)
  C5_tablero_columna_quimica.png— columna + quimica acoplada (tasa de referencia)
  C6_almacenamiento.png         — I_alm por tasa (si hay CSV)
  gif_frente_sg_50k.gif         — animacion perfil Sg a lo largo de k
  gif_tablero_50k.gif           — animacion tablero (S + quimica)

Uso:
  python scripts/generar_figuras_resultados.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from ccus_columna.constantes import TASA_REF, TASAS_M3D, etiqueta_tasa  # parametros del proyecto
from ccus_columna import datos  # noqa: E402
from ccus_columna.constantes import (  # noqa: E402
    K_INYECCION,
    K_SELLO,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.rutas import dir_figuras_cirrus  # noqa: E402

OUT = dir_figuras_cirrus()
OUT.mkdir(parents=True, exist_ok=True)

# Quimica 0D (fase previa) — rutas locales del proyecto La Paz
LAPAZ = Path(r"d:\PedroRomero\Pedro_R\SHAKTI\ModelacionNumerica\Geoquimica_v2\Tarea_1\Fm_La_Paz\La Paz")
ESC_0D = LAPAZ / "tarea1b_reactor_70a" / "escenarios"
ALM_CSV = LAPAZ / "presentaciones" / "ccus_hidrogeologia" / "datos" / "almacenamiento_co2_resumen.csv"

TASAS = list(TASAS_M3D)
LABEL_Q = {q: etiqueta_tasa(q) for q in TASAS_M3D}

# Estilo sobrio (sin slop)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})


# ---------------------------------------------------------------------------
# Lectura quimica 0D (archivos grandes: filtrar al vuelo)
# ---------------------------------------------------------------------------

def load_quimica_0d(tasa: int) -> pd.DataFrame:
    """Carga FmSolution_70a y deja un punto por anos_acum (estado final)."""
    folder = etiqueta_tasa(tasa)
    path = ESC_0D / folder / "resultados" / "FmSolution_70a.txt"
    if not path.exists():
        raise FileNotFoundError(path)

    keep = {"time", "ph", "alk", "c(4)", "calcite", "anos_acum", "sg", "sw", "fase"}
    chunks = []
    for chunk in pd.read_csv(path, sep=r"\s+", engine="python", chunksize=80_000):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        cols = [c for c in chunk.columns if c in keep]
        chunk = chunk[cols]
        chunk = chunk[chunk["time"].astype(float) > 0]
        chunk = chunk[(chunk["ph"] > 2.5) & (chunk["ph"] < 13)]
        chunk = chunk[chunk["anos_acum"].astype(float) >= 0]
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = df.sort_values(["anos_acum", "time"]).groupby("anos_acum", as_index=False).last()
    df["anos"] = df["anos_acum"].astype(float)
    df["c_mol"] = df["c(4)"].astype(float)
    df["calcita"] = df["calcite"].astype(float)
    df["sg"] = df["sg"].astype(float)
    df["sw"] = df["sw"].astype(float)
    return df


def matriz_perfil(malla: str, tasa: int, var: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (t, k, Z) con Z shape (n_t, n_k), k descendente (sello arriba)."""
    p = datos.perfil(malla, tasa, var)
    t = p["tiempo_anos"].to_numpy()
    kcols = [c for c in p.columns if c.startswith("k")]
    ks = np.array([int(c[1:]) for c in kcols])
    # ordenar k descendente para plot: sello (10) arriba
    order = np.argsort(ks)[::-1]
    ks = ks[order]
    Z = p[[kcols[i] for i in order]].to_numpy(dtype=float)
    return t, ks, Z


def _bandas_etapa(ax, ymax=None):
    ax.axvspan(0, T_FIN_EXTRACCION, color="#bdbdbd", alpha=0.15, zorder=0)
    ax.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#90caf9", alpha=0.12, zorder=0)
    ax.axvspan(T_FIN_INYECCION, 70, color="#ffe082", alpha=0.12, zorder=0)
    if ymax is not None:
        ax.set_ylim(top=ymax)


# ---------------------------------------------------------------------------
# Figuras estaticas — columna
# ---------------------------------------------------------------------------

def fig_heatmap_sg():
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.2), sharey=True)
    for ax, q in zip(axes, TASAS):
        t, ks, Z = matriz_perfil("malla_original", q, "Sg")
        im = ax.pcolormesh(t, ks, Z.T, shading="auto", cmap="YlOrRd", vmin=0, vmax=0.8)
        ax.axhline(K_INYECCION, color="#1565c0", ls="--", lw=1, label="inyeccion k15")
        ax.axhline(K_SELLO, color="#2e7d32", ls="--", lw=1, label="sello k10")
        for tv, lab in [(T_FIN_EXTRACCION, "iny"), (T_FIN_INYECCION, "shut")]:
            ax.axvline(tv, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(f"q = {LABEL_Q[q]} m$^3$/d")
        ax.set_xlabel("Tiempo (años)")
        ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    axes[0].set_ylabel("Celda k (sello arriba)")
    axes[0].legend(loc="lower right", fontsize=7)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("$S_g$ (CO$_2$)")
    fig.suptitle("Frente de CO$_2$ en la columna — malla original Cirrus", y=1.02)
    fig.tight_layout()
    path = OUT / "C1_heatmap_sg_tasas.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(" ", path.name)


def fig_heatmap_presion():
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.2), sharey=True)
    for ax, q in zip(axes, TASAS):
        t, ks, Z = matriz_perfil("malla_original", q, "P")
        im = ax.pcolormesh(t, ks, Z.T, shading="auto", cmap="viridis")
        ax.axhline(K_INYECCION, color="w", ls="--", lw=1)
        ax.axhline(K_SELLO, color="w", ls="--", lw=1)
        ax.set_title(f"q = {LABEL_Q[q]}")
        ax.set_xlabel("Tiempo (años)")
        ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    axes[0].set_ylabel("Celda k")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("P (bar)")
    fig.suptitle("Presión de poro — columna Cirrus", y=1.02)
    fig.tight_layout()
    path = OUT / "C2_heatmap_presion.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(" ", path.name)


def fig_llegada_frente():
    r = datos.resumen_frente()
    r = r[r["malla"] == "malla_original"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for q in TASAS:
        g = r[r["tasa_m3d"] == q].dropna(subset=["t_primer_gas_anos"]).sort_values("k")
        ax.plot(g["t_primer_gas_anos"], g["k"], "o-", ms=4, label=LABEL_Q[q])
    ax.axhline(K_INYECCION, color="#1565c0", ls="--", lw=1)
    ax.axhline(K_SELLO, color="#2e7d32", ls="--", lw=1)
    ax.set_xlabel("Primer gas (años)")
    ax.set_ylabel("Celda k")
    ax.set_ylim(36, 9)
    ax.set_title("Llegada del frente de CO$_2$")
    ax.legend(title="tasa")

    ax = axes[1]
    for q in TASAS:
        g = r[r["tasa_m3d"] == q].sort_values("k")
        ax.plot(g["Sg_max"], g["k"], "s-", ms=4, label=LABEL_Q[q])
    ax.axhline(K_INYECCION, color="#1565c0", ls="--", lw=1)
    ax.axhline(K_SELLO, color="#2e7d32", ls="--", lw=1)
    ax.set_xlabel("$S_{g,max}$")
    ax.set_ylabel("Celda k")
    ax.set_ylim(36, 9)
    ax.set_title("Saturación máxima de gas")
    ax.legend(title="tasa")

    fig.suptitle("Diagnóstico del frente — base del transporte inducido (met1/met2)", y=1.02)
    fig.tight_layout()
    path = OUT / "C3_llegada_frente.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(" ", path.name)


# ---------------------------------------------------------------------------
# Figuras — quimica 0D
# ---------------------------------------------------------------------------

def fig_quimica_0d(quim: dict[int, pd.DataFrame]):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    ax_ph, ax_cal, ax_c, ax_sg = axes.ravel()

    for q, df in quim.items():
        lab = LABEL_Q[q]
        ax_ph.plot(df["anos"], df["ph"], lw=1.4, label=lab)
        ax_cal.plot(df["anos"], df["calcita"], lw=1.4, label=lab)
        ax_c.plot(df["anos"], df["c_mol"], lw=1.4, label=lab)
        ax_sg.plot(df["anos"], df["sg"], lw=1.4, label=lab)

    for ax in axes.ravel():
        _bandas_etapa(ax)
        ax.legend(loc="best", ncol=2)

    ax_ph.set_ylabel("pH")
    ax_ph.set_title("pH del reactor 0D (celda k15)")
    ax_cal.set_ylabel("Calcita (mol)")
    ax_cal.set_title("Escudo de calcita (EQUILIBRIUM)")
    ax_c.set_ylabel("C(4) (mol/kgw)")
    ax_c.set_title("Carbono disuelto")
    ax_sg.set_ylabel("$S_g$")
    ax_sg.set_title("Saturación CO$_2$ (dato Cirrus → PHREEQC)")
    ax_sg.set_xlabel("Tiempo (años)")
    ax_c.set_xlabel("Tiempo (años)")

    # anotacion etapas
    ax_ph.text(5, ax_ph.get_ylim()[0] + 0.15, "extracción", fontsize=7, color="#616161")
    ax_ph.text(25, ax_ph.get_ylim()[0] + 0.15, "inyección", fontsize=7, color="#1565c0")
    ax_ph.text(58, ax_ph.get_ylim()[0] + 0.15, "shut-in", fontsize=7, color="#f9a825")

    fig.suptitle(
        "Química 0D forzada por saturación — Fm. La Paz (referencia met1 mono-celda)",
        y=1.01,
    )
    fig.tight_layout()
    path = OUT / "C4_quimica_0d_tasas.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(" ", path.name)


def fig_tablero_acoplado(quim_50k: pd.DataFrame):
    """Columna Sg (heatmap) + series quimicas de k15 a la tasa de referencia."""
    t, ks, Z = matriz_perfil("malla_original", TASA_REF, "Sg")
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1], height_ratios=[1, 1, 1],
                          hspace=0.35, wspace=0.28)

    ax_h = fig.add_subplot(gs[:, 0])
    im = ax_h.pcolormesh(t, ks, Z.T, shading="auto", cmap="YlOrRd", vmin=0, vmax=0.55)
    ax_h.axhline(K_INYECCION, color="#1565c0", ls="--", lw=1.2)
    ax_h.axhline(K_SELLO, color="#2e7d32", ls="--", lw=1.2)
    ax_h.axvline(T_FIN_EXTRACCION, color="k", ls=":", lw=0.8)
    ax_h.axvline(T_FIN_INYECCION, color="k", ls=":", lw=0.8)
    ax_h.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    ax_h.set_xlabel("Tiempo (años)")
    ax_h.set_ylabel("Celda k (sello arriba)")
    ax_h.set_title(f"$S_g(k,t)$ — malla original, {etiqueta_tasa(TASA_REF)} m$^3$/d")
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    cbar.set_label("$S_g$")
    ax_h.annotate("inyección", xy=(0.02, 0.92), xycoords="axes fraction",
                  fontsize=8, color="#1565c0")
    ax_h.annotate("sello", xy=(0.02, 0.04), xycoords="axes fraction",
                  fontsize=8, color="#2e7d32")

    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)

    ax1.plot(quim_50k["anos"], quim_50k["ph"], color="#c62828", lw=1.5)
    ax1.set_ylabel("pH")
    ax1.set_title("Química 0D de la celda k15 (mismo forzante)")
    _bandas_etapa(ax1)

    ax2.plot(quim_50k["anos"], quim_50k["calcita"], color="#6a1b9a", lw=1.5)
    ax2.set_ylabel("Calcita (mol)")
    _bandas_etapa(ax2)

    ax3.plot(quim_50k["anos"], quim_50k["sg"], color="#e65100", lw=1.5, label="$S_g$")
    ax3.plot(quim_50k["anos"], quim_50k["c_mol"], color="#00838f", lw=1.5, label="C(4)")
    ax3.set_ylabel("$S_g$ / C(4)")
    ax3.set_xlabel("Tiempo (años)")
    ax3.legend(loc="upper right")
    _bandas_etapa(ax3)

    fig.suptitle(
        "Acople conceptual: saturación de columna (dato) → química del reactor 0D",
        fontsize=12, y=0.98,
    )
    path = OUT / "C5_tablero_columna_quimica.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(" ", path.name)


def fig_almacenamiento():
    if not ALM_CSV.exists():
        print("  (skip C6: no hay CSV de almacenamiento)")
        return
    df = pd.read_csv(ALM_CSV)
    # solo hito a 70 a
    fin = df[np.isclose(df["hito_a"], 70.0)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    cont = fin[fin["tipo"] == "continua"].sort_values("n_tot")
    # ordenar por tasa en el label
    order = [f"Continuo {etiqueta_tasa(q)}" for q in TASAS_M3D]
    cont = cont.set_index("label").reindex(order).reset_index()
    x = np.arange(len(cont))
    ax.bar(x - 0.2, cont["n_sol"], 0.2, label="$n_{sol}$", color="#00838f")
    ax.bar(x, cont["n_gas"], 0.2, label="$n_{gas}$", color="#e65100")
    ax.bar(x + 0.2, cont["n_tot"], 0.2, label="$I_{alm}$", color="#37474f")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("Continuo ", "") for s in cont["label"]])
    ax.set_ylabel("mol CO$_2$ / L reactor")
    ax.set_title("Continuo @ 70 a — por tasa")
    ax.legend()

    ax = axes[1]
    tipos = ["continua", "wag_gas", "wag_mix"]
    labels_t = {"continua": f"Continuo {etiqueta_tasa(TASA_REF)}", "wag_gas": "WAG gas", "wag_mix": "WAG mix"}
    colors = {"continua": "#d35400", "wag_gas": "#1a5276", "wag_mix": "#196f3d"}
    vals = []
    labs = []
    cols = []
    for t in tipos:
        g = fin[fin["tipo"] == t]
        if t == "continua":
            g = g[g["label"] == f"Continuo {etiqueta_tasa(TASA_REF)}"]
        if len(g) == 0:
            continue
        # promedio de ratios WAG si hay varios
        vals.append(float(g["n_tot"].mean()))
        labs.append(labels_t[t])
        cols.append(colors[t])
    ax.bar(labs, vals, color=cols)
    ax.set_ylabel("$I_{alm}$ (mol)")
    ax.set_title(f"Comparación de enfoques @ 70 a (ref. {etiqueta_tasa(TASA_REF)})")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle("Indicador de almacenamiento $I_{alm}=n_{sol}+n_{gas}+n_{min}$ — fase 0D", y=1.02)
    fig.tight_layout()
    path = OUT / "C6_almacenamiento.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(" ", path.name)


# ---------------------------------------------------------------------------
# GIFs
# ---------------------------------------------------------------------------

def gif_frente_sg(tasa: int = TASA_REF, n_frames: int = 70, fps: int = 8):
    """Animacion: perfil Sg(k) + Sw(k) + P(k) a lo largo del tiempo."""
    t, ks, Sg = matriz_perfil("malla_original", tasa, "Sg")
    _, _, Sw = matriz_perfil("malla_original", tasa, "Sw")
    _, _, P = matriz_perfil("malla_original", tasa, "P")

    # submuestrear tiempos
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    t_f = t[idx]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.8), sharey=True)
    ax_sg, ax_sw, ax_p = axes
    line_sg, = ax_sg.plot([], [], "o-", color="#e65100", ms=4, lw=1.5)
    line_sw, = ax_sw.plot([], [], "o-", color="#1565c0", ms=4, lw=1.5)
    line_p, = ax_p.plot([], [], "o-", color="#2e7d32", ms=4, lw=1.5)
    for ax, title, xlim in zip(
        axes,
        ["$S_g$ (CO$_2$)", "$S_w$", "P (bar)"],
        [(0, 0.85), (0, 1.05), (180, max(450, float(np.nanmax(P)) * 1.05))],
    ):
        ax.set_xlim(*xlim)
        ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
        ax.axhline(K_INYECCION, color="#1565c0", ls="--", lw=0.9, alpha=0.7)
        ax.axhline(K_SELLO, color="#2e7d32", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel(title.split()[0] if False else "")
    ax_sg.set_ylabel("Celda k (sello arriba)")
    ax_sg.set_xlabel("$S_g$")
    ax_sw.set_xlabel("$S_w$")
    ax_p.set_xlabel("P (bar)")
    titulo = fig.suptitle("", fontsize=12)
    fig.tight_layout()

    def init():
        line_sg.set_data([], [])
        line_sw.set_data([], [])
        line_p.set_data([], [])
        return line_sg, line_sw, line_p, titulo

    def update(frame):
        i = idx[frame]
        line_sg.set_data(Sg[i], ks)
        line_sw.set_data(Sw[i], ks)
        line_p.set_data(P[i], ks)
        etapa = "extracción" if t_f[frame] < T_FIN_EXTRACCION else (
            "inyección" if t_f[frame] < T_FIN_INYECCION else "shut-in"
        )
        titulo.set_text(
            f"Columna Cirrus — {LABEL_Q[tasa]} m³/d  |  t = {t_f[frame]:.1f} a  ({etapa})"
        )
        return line_sg, line_sw, line_p, titulo

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(idx), interval=1000 / fps, blit=False,
    )
    path = OUT / f"gif_frente_sg_{LABEL_Q[tasa]}.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(" ", path.name)


def gif_tablero_quimica(quim_50k: pd.DataFrame, n_frames: int = 60, fps: int = 7):
    """GIF: perfil Sg columna + puntero en series pH/calcita/C."""
    t, ks, Sg = matriz_perfil("malla_original", TASA_REF, "Sg")
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)

    fig = plt.figure(figsize=(11.5, 5.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], height_ratios=[1, 1],
                          hspace=0.35, wspace=0.3)
    ax_prof = fig.add_subplot(gs[:, 0])
    ax_ph = fig.add_subplot(gs[0, 1])
    ax_bot = fig.add_subplot(gs[1, 1])

    # perfiles base
    line_sg, = ax_prof.plot([], [], "o-", color="#e65100", ms=5, lw=1.8)
    ax_prof.set_xlim(0, 0.55)
    ax_prof.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    ax_prof.axhline(K_INYECCION, color="#1565c0", ls="--", lw=1)
    ax_prof.axhline(K_SELLO, color="#2e7d32", ls="--", lw=1)
    ax_prof.set_xlabel("$S_g$")
    ax_prof.set_ylabel("Celda k")
    ax_prof.set_title("Perfil de saturación CO$_2$")

    # series quimicas completas (fondo)
    ax_ph.plot(quim_50k["anos"], quim_50k["ph"], color="#c62828", lw=1.2, alpha=0.35)
    pt_ph, = ax_ph.plot([], [], "o", color="#c62828", ms=8)
    vline_ph = ax_ph.axvline(0, color="k", ls=":", lw=1)
    ax_ph.set_ylabel("pH")
    ax_ph.set_xlim(0, 70)
    ax_ph.set_title("Química 0D (celda k15)")
    _bandas_etapa(ax_ph)

    ax_bot.plot(quim_50k["anos"], quim_50k["calcita"], color="#6a1b9a", lw=1.2,
                alpha=0.35, label="calcita")
    ax_bot.plot(quim_50k["anos"], quim_50k["c_mol"], color="#00838f", lw=1.2,
                alpha=0.35, label="C(4)")
    pt_cal, = ax_bot.plot([], [], "o", color="#6a1b9a", ms=7)
    pt_c, = ax_bot.plot([], [], "o", color="#00838f", ms=7)
    vline_bot = ax_bot.axvline(0, color="k", ls=":", lw=1)
    ax_bot.set_xlabel("Tiempo (años)")
    ax_bot.set_ylabel("mol / mol·kgw$^{-1}$")
    ax_bot.legend(loc="upper right", fontsize=7)
    ax_bot.set_xlim(0, 70)
    _bandas_etapa(ax_bot)

    titulo = fig.suptitle("", fontsize=11)

    def interp_quim(tt):
        ph = float(np.interp(tt, quim_50k["anos"], quim_50k["ph"]))
        cal = float(np.interp(tt, quim_50k["anos"], quim_50k["calcita"]))
        cm = float(np.interp(tt, quim_50k["anos"], quim_50k["c_mol"]))
        return ph, cal, cm

    def update(frame):
        i = idx[frame]
        tt = float(t[i])
        line_sg.set_data(Sg[i], ks)
        ph, cal, cm = interp_quim(tt)
        pt_ph.set_data([tt], [ph])
        pt_cal.set_data([tt], [cal])
        pt_c.set_data([tt], [cm])
        vline_ph.set_xdata([tt, tt])
        vline_bot.set_xdata([tt, tt])
        etapa = "extracción" if tt < T_FIN_EXTRACCION else (
            "inyección" if tt < T_FIN_INYECCION else "shut-in"
        )
        titulo.set_text(
            f"t = {tt:.1f} a ({etapa})  |  pH={ph:.2f}  calcita={cal:.3f} mol  C={cm:.3f}"
        )
        return line_sg, pt_ph, pt_cal, pt_c, vline_ph, vline_bot, titulo

    anim = animation.FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps)
    path = OUT / "gif_tablero_50k.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(" ", path.name)


def gif_heatmap_sg(tasa: int = TASA_REF, n_frames: int = 50, fps: int = 6):
    """GIF: heatmap Sg que se va 'revelando' en el tiempo (mascara temporal)."""
    t, ks, Sg = matriz_perfil("malla_original", tasa, "Sg")
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))
    # fondo gris
    ax.pcolormesh(t, ks, np.zeros_like(Sg).T, shading="auto", cmap="Greys",
                  vmin=0, vmax=1, alpha=0.15)
    mesh = ax.pcolormesh(t, ks, np.full_like(Sg, np.nan).T, shading="auto",
                         cmap="YlOrRd", vmin=0, vmax=0.7)
    ax.axhline(K_INYECCION, color="#1565c0", ls="--", lw=1)
    ax.axhline(K_SELLO, color="#2e7d32", ls="--", lw=1)
    ax.set_ylim(ks.max() + 0.5, ks.min() - 0.5)
    ax.set_xlabel("Tiempo (años)")
    ax.set_ylabel("Celda k")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046)
    cbar.set_label("$S_g$")
    titulo = ax.set_title("")
    vline = ax.axvline(0, color="k", lw=1.5)
    fig.tight_layout()

    def update(frame):
        i = idx[frame]
        mask = np.full_like(Sg, np.nan)
        mask[: i + 1, :] = Sg[: i + 1, :]
        mesh.set_array(mask.T.ravel())
        vline.set_xdata([t[i], t[i]])
        titulo.set_text(f"Avance del frente CO$_2$ — {LABEL_Q[tasa]}  |  t = {t[i]:.1f} a")
        return mesh, vline, titulo

    anim = animation.FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps)
    path = OUT / f"gif_heatmap_sg_{LABEL_Q[tasa]}.gif"
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(" ", path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("== Figuras columna ==")
    fig_heatmap_sg()
    fig_heatmap_presion()
    fig_llegada_frente()

    print("== Cargando química 0D (puede tardar) ==")
    quim = {}
    for q in TASAS:
        print(f"  leyendo {LABEL_Q[q]}...")
        quim[q] = load_quimica_0d(q)
        print(f"    {len(quim[q])} puntos")

    print("== Figuras química ==")
    fig_quimica_0d(quim)
    fig_tablero_acoplado(quim[TASA_REF])
    fig_almacenamiento()

    print("== GIFs ==")
    gif_frente_sg(TASA_REF)
    gif_heatmap_sg(TASA_REF)
    gif_tablero_quimica(quim[TASA_REF])
    # un GIF extra a la tasa maxima para contraste de frente
    gif_frente_sg(TASAS_M3D[-1], n_frames=60)

    print("\nListo. Figuras en:", OUT)


if __name__ == "__main__":
    main()

"""Sintesis / cierre met1: figuras fijas + tabla multi-tasa (DC-028).

No re-corre PHREEQC: usa perfil_quimica_met1.csv de corridas *_completo.

Salida:
  figuras/comparacion/{malla}/q{tasa}/cierre/
  figuras/comparacion/sintesis_met1/
  docs/03_CIERRE_MET1.md  (plantilla generada si --escribir-doc)

Uso:
  python comparacion/sintesis_met1.py
  python comparacion/sintesis_met1.py --malla malla_original --tasa <Q>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))

from ccus_columna.constantes import TASA_REF, TASAS_M3D, etiqueta_tasa  # parametros del proyecto
from ccus_columna import datos
from ccus_columna.constantes import (
    K_INYECCION,
    K_SELLO,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.permeabilidad import ratio_kozeny_carman
from ccus_columna.rutas import dir_figuras_comparacion

MET1 = _RAIZ / "met1_celdas_encadenadas" / "corridas"
OUT_SINTESIS = _RAIZ / "figuras" / "comparacion" / "sintesis_met1"
LABEL_Q = {q: etiqueta_tasa(q) for q in TASAS_M3D}
SNAPSHOT_T = (10.0, 30.0, 70.0)
TASAS_ORIG = list(TASAS_M3D)


def _run_dir(malla: str, tasa: int) -> Path:
    return MET1 / f"{malla}_q{tasa}_completo"


def _cargar_perfil(malla: str, tasa: int) -> pd.DataFrame | None:
    p = _run_dir(malla, tasa) / "perfil_quimica_met1.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _anotar_k(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    phi0 = out["phi"] - out["delta_phi"]
    out["K_pct_potencia"] = [
        100.0 * ratio_kozeny_carman(float(phi), float(p0), completo=False)
        for phi, p0 in zip(out["phi"], phi0)
    ]
    out["K_pct_ck"] = [
        100.0 * ratio_kozeny_carman(float(phi), float(p0), completo=True)
        for phi, p0 in zip(out["phi"], phi0)
    ]
    return out


def _perfil_en_t(df: pd.DataFrame, col: str, t: float, ks: list[int]) -> np.ndarray:
    out = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        sub = df[df["k"].astype(int) == int(k)].sort_values("tiempo_anos")
        if sub.empty:
            continue
        out[i] = float(np.interp(
            t, sub["tiempo_anos"].to_numpy(float), sub[col].to_numpy(float),
            left=float(sub[col].iloc[0]), right=float(sub[col].iloc[-1]),
        ))
    return out


def _fila_t70(df: pd.DataFrame, k: int) -> pd.Series | None:
    sub = df[df["k"] == k].sort_values("tiempo_anos")
    if sub.empty:
        return None
    return sub.iloc[-1]


def tabla_multi_tasa(malla: str = "malla_original") -> pd.DataFrame:
    """Metricas comparables entre tasas (k10 sello, k15 inyeccion) @ t final."""
    filas = []
    for tasa in TASAS_ORIG:
        df = _cargar_perfil(malla, tasa)
        if df is None:
            continue
        df = _anotar_k(df)
        ks = sorted(int(k) for k in df["k"].unique())
        r10 = _fila_t70(df, K_SELLO)
        r15 = _fila_t70(df, K_INYECCION)
        # celda con max |delta_phi|
        tfin = float(df["tiempo_anos"].max())
        fin = df[np.isclose(df["tiempo_anos"], df["tiempo_anos"].unique()[
            np.argmin(np.abs(df["tiempo_anos"].unique() - tfin))
        ])]
        if fin.empty:
            fin = df.groupby("k").tail(1)
        i_max = fin["delta_phi"].abs().idxmax()
        rmax = fin.loc[i_max]
        # Sg max Cirrus
        sg15 = float(datos.celda(malla, tasa, K_INYECCION)["Sg"].max())
        sg10 = float(datos.celda(malla, tasa, K_SELLO)["Sg"].max())
        filas.append({
            "malla": malla,
            "tasa_m3d": tasa,
            "label_q": LABEL_Q.get(tasa, str(tasa)),
            "n_celdas": len(ks),
            "k_min": min(ks),
            "k_max": max(ks),
            "pH_k10": float(r10["pH"]) if r10 is not None else np.nan,
            "pH_k15": float(r15["pH"]) if r15 is not None else np.nan,
            "delta_phi_k10": float(r10["delta_phi"]) if r10 is not None else np.nan,
            "delta_phi_k15": float(r15["delta_phi"]) if r15 is not None else np.nan,
            "I_alm_k15": float(r15["I_alm_mol"]) if r15 is not None else np.nan,
            "calcita_k15": float(r15["calcita_mol"]) if r15 is not None else np.nan,
            "K_pct_pot_k10": float(r10["K_pct_potencia"]) if r10 is not None else np.nan,
            "K_pct_ck_k10": float(r10["K_pct_ck"]) if r10 is not None else np.nan,
            "k_max_dphi": int(rmax["k"]),
            "delta_phi_max": float(rmax["delta_phi"]),
            "K_pct_pot_max": float(rmax["K_pct_potencia"]),
            "Sg_max_k15": sg15,
            "Sg_max_k10": sg10,
        })
    return pd.DataFrame(filas)


def fig_perfiles_t70(df: pd.DataFrame, malla: str, tasa: int, out: Path) -> Path:
    df = _anotar_k(df)
    ks = sorted(df["k"].unique(), reverse=True)
    ks_arr = np.array(ks, dtype=float)
    t = float(df["tiempo_anos"].max())
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.5), sharey=True)
    specs = [
        ("pH", "pH", None),
        ("delta_phi", r"$\Delta\phi$", 1e4),  # mostrar en 1e-4
        ("K_pct_potencia", r"$K/K_0$ potencia (%)", None),
        ("K_pct_ck", r"$K/K_0$ Carman-Kozeny (%)", None),
    ]
    for ax, (col, xlab, scale) in zip(axes, specs):
        vals = _perfil_en_t(df, col, t, list(ks))
        plot_v = vals * scale if scale else vals
        ax.plot(plot_v, ks_arr, "o-", color="#1565c0", ms=5, lw=1.8)
        if col.startswith("K_"):
            ax.axvline(100.0, color="#616161", ls="--", lw=0.9, alpha=0.7)
        ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.8, alpha=0.7)
        ax.axhline(K_SELLO, color="#2e7d32", ls=":", lw=0.8, alpha=0.7)
        if 25 in ks:
            ax.axhline(25, color="#ef6c00", ls=":", lw=0.8, alpha=0.7)
        ax.set_xlabel(xlab + (r" ($\times 10^{-4}$)" if scale else ""))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Celda k (sello arriba)")
    axes[0].set_ylim(ks_arr.max() + 0.5, ks_arr.min() - 0.5)
    fig.suptitle(
        f"met1 cierre — perfiles t = {t:.0f} a  |  {malla} q={LABEL_Q.get(tasa, tasa)}",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_evolucion_clave(df: pd.DataFrame, malla: str, tasa: int, out: Path) -> Path:
    df = _anotar_k(df)
    ks_show = [k for k in (K_SELLO, K_INYECCION, 25) if k in set(df["k"].astype(int))]
    colors = {K_SELLO: "#2e7d32", K_INYECCION: "#1565c0", 25: "#ef6c00"}
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for k in ks_show:
        sub = df[df["k"] == k].sort_values("tiempo_anos")
        c = colors[k]
        axes[0].plot(sub["tiempo_anos"], sub["pH"], color=c, lw=1.8, label=f"k={k}")
        axes[1].plot(sub["tiempo_anos"], sub["delta_phi"] * 1e4, color=c, lw=1.8, label=f"k={k}")
        axes[2].plot(sub["tiempo_anos"], sub["K_pct_ck"], color=c, lw=1.8, label=f"k={k}")
    for ax in axes:
        ax.axvspan(T_FIN_EXTRACCION, T_FIN_INYECCION, color="#fff3e0", alpha=0.35)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("pH")
    axes[1].set_ylabel(r"$\Delta\phi$ ($\times 10^{-4}$)")
    axes[2].set_ylabel(r"$K/K_0$ Carman-Kozeny (%)")
    axes[2].axhline(100.0, color="#616161", ls="--", lw=0.9, alpha=0.7)
    axes[2].set_xlabel("Tiempo (a)")
    axes[0].set_title(
        f"Evolucion k10 / k15 / k25 — {malla} q={LABEL_Q.get(tasa, tasa)}",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_heatmap_dphi(df: pd.DataFrame, malla: str, tasa: int, out: Path) -> Path:
    sub = df.copy()
    ks = sorted(sub["k"].unique())
    ts = sorted(sub["tiempo_anos"].unique())
    mat = np.full((len(ks), len(ts)), np.nan)
    for i, k in enumerate(ks):
        sk = sub[sub["k"] == k].sort_values("tiempo_anos")
        mat[i, :] = np.interp(ts, sk["tiempo_anos"], sk["delta_phi"])
    fig, ax = plt.subplots(figsize=(10, 5))
    vmax = max(float(np.nanmax(np.abs(mat))), 1e-6)
    im = ax.imshow(
        mat * 1e4, aspect="auto", origin="upper",
        extent=[ts[0], ts[-1], ks[-1] + 0.5, ks[0] - 0.5],
        cmap="RdBu_r", vmin=-vmax * 1e4, vmax=vmax * 1e4,
    )
    ax.axvline(T_FIN_EXTRACCION, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(T_FIN_INYECCION, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Tiempo (a)")
    ax.set_ylabel("Celda k (sello arriba)")
    ax.set_title(f"$\\Delta\\phi$ ($\\times 10^{{-4}}$) — {malla} q={LABEL_Q.get(tasa, tasa)}")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(r"$\Delta\phi \times 10^{4}$")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_multi_tasa(tab: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(tab))
    labels = tab["label_q"].tolist()
    axes[0].bar(x - 0.18, tab["pH_k10"], 0.35, color="#2e7d32", label="k10")
    axes[0].bar(x + 0.18, tab["pH_k15"], 0.35, color="#1565c0", label="k15")
    axes[0].set_ylabel("pH @ 70 a")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, tab["delta_phi_k10"] * 1e4, color="#2e7d32", alpha=0.85)
    axes[1].set_ylabel(r"$\Delta\phi$ k10 ($\times 10^{-4}$) @ 70 a")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x - 0.18, tab["Sg_max_k15"], 0.35, color="#1565c0", label="k15")
    axes[2].bar(x + 0.18, tab["Sg_max_k10"], 0.35, color="#2e7d32", label="k10")
    axes[2].set_ylabel("Sg max Cirrus")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("met1 — comparacion multi-tasa (malla original, rama comparable k15/k10)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_porque_k25(df: pd.DataFrame, malla: str, tasa: int, out: Path) -> Path:
    """Figura explicativa: Sg/Sw Cirrus vs Delta calcita / Delta phi."""
    df = _anotar_k(df)
    t = float(df["tiempo_anos"].max())
    ks = sorted(df["k"].unique())
    dphi = _perfil_en_t(df, "delta_phi", t, ks)
    cal0 = {}
    for k in ks:
        sk = df[df["k"] == k].sort_values("tiempo_anos")
        cal0[k] = float(sk["calcita_mol"].iloc[0])
    dcal = np.array([
        float(df[df["k"] == k].sort_values("tiempo_anos")["calcita_mol"].iloc[-1]) - cal0[k]
        for k in ks
    ])
    sg = np.array([float(datos.celda(malla, tasa, k)["Sg"].max()) for k in ks])
    sw_at = []
    for k in ks:
        c = datos.celda(malla, tasa, k)
        i = int(c["Sg"].idxmax()) if c["Sg"].max() > 0 else 0
        sw_at.append(float(c.loc[i, "Sw"]) if c["Sg"].max() > 0 else float(c["Sw"].mean()))
    sw_at = np.array(sw_at)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    axb = ax.twiny()
    ax.plot(dphi * 1e4, ks, "o-", color="#c62828", lw=2, ms=6, label=r"$\Delta\phi\times10^4$")
    ax.plot(-dcal * 1e2, ks, "s--", color="#ef6c00", lw=1.5, ms=5, label=r"$-\Delta$calcita$\times10^2$")
    axb.plot(sg, ks, "^-", color="#1565c0", lw=1.5, ms=5, alpha=0.85, label="Sg max")
    axb.plot(sw_at, ks, "v-", color="#00838f", lw=1.5, ms=5, alpha=0.85, label="Sw @ Sg max")
    ax.axhline(K_INYECCION, color="#1565c0", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(25, color="#ef6c00", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylim(max(ks) + 0.5, min(ks) - 0.5)
    ax.set_xlabel(r"Alteracion mineral ($\Delta\phi$, $-\Delta$calcita)")
    axb.set_xlabel("Saturaciones Cirrus")
    ax.set_ylabel("Celda k (sello arriba)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Por que el maximo no esta en k15 — {malla} q={LABEL_Q.get(tasa, tasa)} @ 70 a",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def resumen_json(tab: pd.DataFrame, df50: pd.DataFrame | None) -> dict:
    out: dict = {"multi_tasa": tab.to_dict(orient="records")}
    if df50 is not None:
        df50 = _anotar_k(df50)
        fin = df50.groupby("k").tail(1)
        r25 = fin[fin["k"] == 25]
        r15 = fin[fin["k"] == 15]
        r10 = fin[fin["k"] == 10]
        out["q50k_columna"] = {
            "mensaje": (
                "El maximo de disolucion/DeltaK esta en k25 (borde inferior del "
                "penacho), no en k15: alli hay mas agua reactiva (Sw alta) aunque "
                "menos gas."
            ),
            "k10": r10.iloc[0][["pH", "delta_phi", "K_pct_ck", "I_alm_mol"]].to_dict() if len(r10) else {},
            "k15": r15.iloc[0][["pH", "delta_phi", "K_pct_ck", "I_alm_mol"]].to_dict() if len(r15) else {},
            "k25": r25.iloc[0][["pH", "delta_phi", "K_pct_ck", "I_alm_mol"]].to_dict() if len(r25) else {},
        }
    return out


def escribir_doc(tab: pd.DataFrame, resumen: dict, out_md: Path) -> Path:
    q50 = resumen.get("q50k_columna", {})
    lineas = [
        "# Cierre met1 — Columna 1D CCUS Fm. La Paz",
        "",
        "Documento de sintesis (DC-028). Generado junto a",
        "`python comparacion/sintesis_met1.py`. Figuras en",
        "`figuras/comparacion/sintesis_met1/` y",
        f"`figuras/comparacion/malla_original/q{TASA_REF}/cierre/`.",
        "",
        "## 1. Alcance cerrado",
        "",
        "- Metodologia met1 (celdas encadenadas, MIX hacia el sello k10).",
        f"- Corrida principal: **malla original q={etiqueta_tasa(TASA_REF)}, columna k30→k10**.",
        "- Otras tasas: corridas **rama k15→k10** (6 celdas);",
        "  comparables en sello/inyeccion, no en profundidad.",
        "- Permeabilidad de roca: potencia $(\\phi/\\phi_0)^3$ y Carman-Kozeny",
        "  completo; sin krw (Corey/Wyllie fuera del cierre).",
        "",
        "## 2. Mensajes clave",
        "",
        "1. **La saturacion dicta**: PHREEQC no inventa Sg; el cambio mineral",
        "   sigue al agua acidificada en el poro reactivo.",
        "2. **El sello (k10) se acidifica** (pH cae) pero $\\Delta\\phi$ y $\\Delta K$",
        "   son muy pequenos (~0.00x %). No hay evidencia de dano estructural",
        "   relevante en q50k.",
        "3. **El maximo de alteracion no esta en k15** (mas gas), sino en el",
        "   **borde inferior del penacho (k25)**: alli Sw es alta → mas",
        "   $m_{water}$ → mas disolucion de calcita → mayor $K/K_0$.",
        "4. Celdas sin gas (k27–k30 en q50k) no cambian.",
        "",
        "## 3. Numeros q50k @ 70 a (columna completa)",
        "",
    ]
    for key in ("k10", "k15", "k25"):
        d = q50.get(key, {})
        if not d:
            continue
        lineas.append(
            f"- **{key}**: pH={d.get('pH', float('nan')):.3f}, "
            f"Δφ={d.get('delta_phi', float('nan')):.2e}, "
            f"K/K0(CK)={d.get('K_pct_ck', float('nan')):.3f} %, "
            f"I_alm={d.get('I_alm_mol', float('nan')):.4f} mol"
        )
    lineas += [
        "",
        "## 4. Multi-tasa (malla original, k10/k15)",
        "",
        "| tasa | n celdas | pH k10 | pH k15 | Δφ k10 | Sg max k15 | Sg max k10 |",
        "|------|----------|--------|--------|--------|------------|------------|",
    ]
    for _, r in tab.iterrows():
        lineas.append(
            f"| {r['label_q']} | {int(r['n_celdas'])} | {r['pH_k10']:.3f} | "
            f"{r['pH_k15']:.3f} | {r['delta_phi_k10']:.2e} | "
            f"{r['Sg_max_k15']:.3f} | {r['Sg_max_k10']:.3f} |"
        )
    lineas += [
        "",
        "## 5. Figuras del paquete",
        "",
        "| Archivo | Contenido |",
        "|---------|-----------|",
        "| `perfiles_t70.png` | pH, Δφ, K potencia, K Carman-Kozeny @ 70 a |",
        "| `evolucion_k10_k15_k25.png` | Series temporales clave |",
        "| `heatmap_delta_phi.png` | Mapa k–t de Δφ |",
        "| `porque_k25.png` | Sg/Sw vs disolucion (explica el maximo) |",
        "| `multi_tasa_k10_k15.png` | Barras pH / Δφ / Sg entre tasas |",
        "| `tabla_multi_tasa.csv` | Metricas numericas |",
        "",
        "## 6. Que no se amplia en este cierre",
        "",
        "- Nuevos estimadores de krw (Corey/Wyllie).",
        "- Heterogeneidad φ/K por celda dentro del reservorio.",
        "- Comparacion formal met1 vs met2 (opcional, fuera de alcance).",
        "",
        "## 7. Conclusion",
        "",
        "La fase met1 esta **lista para presentar**: columna geoquimica forzada",
        "por Cirrus, sello con acidificacion leve y alteracion mineral minima,",
        "y el hallazgo interpretativo de que la mayor disolucion ocurre donde",
        "hay **agua + borde de gas**, no donde Sg es maxima.",
        "",
    ]
    out_md.write_text("\n".join(lineas), encoding="utf-8")
    return out_md


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Sintesis / cierre met1")
    ap.add_argument("--malla", default="malla_original")
    ap.add_argument("--tasa", type=int, default=TASA_REF)
    ap.add_argument("--escribir-doc", action="store_true", default=True)
    ap.add_argument("--sin-doc", action="store_true")
    args = ap.parse_args()

    out_cierre = dir_figuras_comparacion(args.malla, args.tasa) / "cierre"
    out_cierre.mkdir(parents=True, exist_ok=True)
    OUT_SINTESIS.mkdir(parents=True, exist_ok=True)

    df = _cargar_perfil(args.malla, args.tasa)
    if df is None:
        print(f"No hay perfil: {_run_dir(args.malla, args.tasa)}")
        return 1

    paths = []
    paths.append(fig_perfiles_t70(df, args.malla, args.tasa, out_cierre / "perfiles_t70.png"))
    paths.append(fig_evolucion_clave(df, args.malla, args.tasa, out_cierre / "evolucion_k10_k15_k25.png"))
    paths.append(fig_heatmap_dphi(df, args.malla, args.tasa, out_cierre / "heatmap_delta_phi.png"))
    if args.tasa == TASA_REF and 25 in set(df["k"].astype(int)):
        paths.append(fig_porque_k25(df, args.malla, args.tasa, out_cierre / "porque_k25.png"))

    tab = tabla_multi_tasa(args.malla)
    tab.to_csv(OUT_SINTESIS / "tabla_multi_tasa.csv", index=False, float_format="%.6g")
    tab.to_csv(out_cierre / "tabla_multi_tasa.csv", index=False, float_format="%.6g")
    if not tab.empty:
        paths.append(fig_multi_tasa(tab, OUT_SINTESIS / "multi_tasa_k10_k15.png"))
        paths.append(fig_multi_tasa(tab, out_cierre / "multi_tasa_k10_k15.png"))

    res = resumen_json(tab, df if args.tasa == TASA_REF else None)
    (out_cierre / "resumen_sintesis.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8",
    )
    (OUT_SINTESIS / "resumen_sintesis.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8",
    )

    if args.escribir_doc and not args.sin_doc:
        doc = escribir_doc(tab, res, _RAIZ / "docs" / "03_CIERRE_MET1.md")
        paths.append(doc)

    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

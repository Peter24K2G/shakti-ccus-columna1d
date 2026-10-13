"""Exporta los .xlsx de Cirrus (NAS) a CSV tidy por malla/tasa/celda.

Entrada (raiz del proyecto CCUS):
    PRESION_TIEMPO.xlsx / PRESION_TIEMPO - REFINADO.xlsx
        1 hoja por celda; bloque A-F: t, P(bar) x tasa (ver parametros: escenarios.tasas_m3d), Etapa
    SAT_TIEMPO-PEDRO.xlsx / SAT_TIEMPO-PEDRO_REFINADO.xlsx
        1 hoja por celda; bloque A-F: Sg (CO2) x tasa; bloque H-M: So (crudo) x tasa

Salida:
    datos/csv/{malla}/q{tasa}/celda_k{k:02d}.csv
        columnas: tiempo_anos, etapa, P_bar, Sg, So, Sw   (Sw = 1 - Sg - So)
    datos/csv/{malla}/q{tasa}/perfil_{Sg,So,Sw,P}.csv
        matrices t x k (columna por celda, k10..k35) para la modelacion 1D
    datos/csv/resumen_frente_sg.csv
        por malla/tasa/celda: primer t con Sg>1e-3, Sg_max, t de Sg_max

Notas de datos (ver docs/02_DICCIONARIO_DATOS.md):
    - Malla original: celdas (16,15,k). Malla refinada: (36,35,k). k=10 es el SELLO.
    - REFINADO no tiene todas las tasas (columna vacia) y un caso usa otro
      par K_res / K_sello (ver parametros: darcy.refinada_especial).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccus_columna.constantes import TASAS_M3D  # parametros del proyecto

BASE = Path(__file__).resolve().parents[1]  # raiz del proyecto CCUS en la NAS
OUT = BASE / "datos" / "csv"

TASAS = list(TASAS_M3D)
RE_CELDA = re.compile(r"CELDA \((\d+),\s*(\d+),\s*(\d+)\)")


def _hoja_a_bloque(rows: list[tuple], col_ini: int) -> pd.DataFrame:
    """Bloque de 6 columnas (t, 4 tasas, etapa) a partir de col_ini (0-based)."""
    data = []
    for r in rows[3:]:
        t = r[col_ini] if col_ini < len(r) else None
        if t is None or not isinstance(t, (int, float)):
            continue
        fila = {"tiempo_anos": float(t)}
        for j, tasa in enumerate(TASAS):
            v = r[col_ini + 1 + j] if col_ini + 1 + j < len(r) else None
            fila[tasa] = float(v) if isinstance(v, (int, float)) else np.nan
        etapa = r[col_ini + 5] if col_ini + 5 < len(r) else None
        fila["etapa"] = str(etapa) if etapa is not None else ""
        data.append(fila)
    return pd.DataFrame(data)


def leer_libro(path: Path) -> dict[int, pd.DataFrame]:
    """{k: DataFrame} con bloques de la hoja de cada celda."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    celdas: dict[int, dict] = {}
    es_sat = "SAT" in path.name.upper()
    for ws in wb.worksheets:
        m = RE_CELDA.search(ws.title)
        if not m:
            continue
        k = int(m.group(3))
        rows = list(ws.iter_rows(values_only=True))
        b1 = _hoja_a_bloque(rows, 0)
        if es_sat:
            b2 = _hoja_a_bloque(rows, 7)
            celdas[k] = {"Sg": b1, "So": b2}
        else:
            celdas[k] = {"P": b1}
    wb.close()
    return celdas


def exportar_malla(nombre_malla: str, f_pres: Path, f_sat: Path) -> pd.DataFrame:
    pres = leer_libro(f_pres)
    sat = leer_libro(f_sat)
    ks = sorted(set(pres) & set(sat))
    if not ks:
        raise RuntimeError(f"Sin celdas comunes entre {f_pres.name} y {f_sat.name}")
    resumen = []

    # tasas con datos (usar la celda mas profunda como referencia)
    k_ref = max(ks)
    tasas_ok = [q for q in TASAS if sat[k_ref]["Sg"][q].notna().sum() > 0]
    print(f"[{nombre_malla}] celdas k={min(ks)}..{max(ks)}  tasas={tasas_ok}")

    for q in tasas_ok:
        qdir = OUT / nombre_malla / f"q{q}"
        qdir.mkdir(parents=True, exist_ok=True)
        perfiles = {v: {} for v in ("Sg", "So", "Sw", "P")}
        t_ref = None
        for k in ks:
            sg = sat[k]["Sg"][["tiempo_anos", q, "etapa"]].rename(columns={q: "Sg"})
            so = sat[k]["So"][["tiempo_anos", q]].rename(columns={q: "So"})
            pp = pres[k]["P"][["tiempo_anos", q]].rename(columns={q: "P_bar"})
            df = sg.merge(so, on="tiempo_anos", how="outer").merge(pp, on="tiempo_anos", how="outer")
            df = df.sort_values("tiempo_anos").reset_index(drop=True)
            df["Sg"] = df["Sg"].clip(0, 1)
            df["So"] = df["So"].clip(0, 1)
            df["Sw"] = (1.0 - df["Sg"] - df["So"]).clip(0, 1)
            df = df[["tiempo_anos", "etapa", "P_bar", "Sg", "So", "Sw"]]
            df.to_csv(qdir / f"celda_k{k:02d}.csv", index=False, float_format="%.6f")

            if t_ref is None:
                t_ref = df["tiempo_anos"].to_numpy()
            for v in ("Sg", "So", "Sw"):
                perfiles[v][f"k{k:02d}"] = df[v].to_numpy()
            perfiles["P"][f"k{k:02d}"] = df["P_bar"].to_numpy()

            con_gas = df[df["Sg"] > 1e-3]
            resumen.append({
                "malla": nombre_malla, "tasa_m3d": q, "k": k,
                "t_primer_gas_anos": round(float(con_gas["tiempo_anos"].iloc[0]), 3) if len(con_gas) else np.nan,
                "Sg_max": round(float(df["Sg"].max()), 5),
                "t_Sg_max_anos": round(float(df.loc[df["Sg"].idxmax(), "tiempo_anos"]), 3),
                "P_max_bar": round(float(df["P_bar"].max()), 2),
            })

        for v, cols in perfiles.items():
            mat = pd.DataFrame({"tiempo_anos": t_ref, **cols})
            mat.to_csv(qdir / f"perfil_{v}.csv", index=False, float_format="%.6f")

    return pd.DataFrame(resumen)


def main() -> None:
    f_pres_orig = next(BASE.glob("PRESI*TIEMPO.xlsx"))
    f_pres_ref = next(BASE.glob("PRESI*REFINADO.xlsx"))
    f_sat_orig = next(p for p in BASE.glob("SAT_TIEMPO*.xlsx") if "REFINADO" not in p.name.upper())
    f_sat_ref = next(p for p in BASE.glob("SAT_TIEMPO*REFINADO*.xlsx"))

    res = pd.concat([
        exportar_malla("malla_original", f_pres_orig, f_sat_orig),
        exportar_malla("malla_refinada", f_pres_ref, f_sat_ref),
    ], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "resumen_frente_sg.csv", index=False)
    print("\nResumen frente de gas (primer t con Sg>1e-3):")
    for (malla, q), g in res.groupby(["malla", "tasa_m3d"]):
        con = g.dropna(subset=["t_primer_gas_anos"]).sort_values("t_primer_gas_anos")
        primeras = ", ".join(f"k{int(r.k)}@{r.t_primer_gas_anos:g}a" for r in con.head(6).itertuples())
        print(f"  {malla} q={q}: {len(con)}/{len(g)} celdas con gas | primeras: {primeras}")


if __name__ == "__main__":
    main()

"""Post-proceso de salidas PHREEQC -> perfil_quimica_{met}.csv (plan §6)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ccus_columna.constantes import (
    K_INYECCION,
    K_INYECCION_FONDO,
    K_SELLO,
    M_WATER_REF_KG,
    SEG_POR_ANO,
)
from ccus_columna.geometria import v_bulk_l
from ccus_columna.mineralogia import PROPS, inventario_eq, inventario_kin, m_water_ref, phi0
from ccus_columna.rutas import malla_desde_run

PERFIL_COLS = (
    "tiempo_anos", "k", "pH", "calcita_mol", "C_total_mol_kgw",
    "I_alm_mol", "delta_phi", "phi",
)

# Volumenes molares L/mol (D-004 / DC-010). m0 sale de mineralogia.inventario_eq(k).
VM_EQ: dict[str, float] = {
    "calcite": 0.0369,
    "pyrite": 0.0239,
    "gypsum": 0.074,
    "dawsonite": 0.053,
    "ankerite": 0.06606,
    "dolomite": 0.06434,
}
_ALIAS_MINERAL = {
    "calcite": ("calcite", "calcita_mol"),
    "pyrite": ("pyrite",),
    "gypsum": ("gypsum",),
    "dawsonite": ("dawsonite",),
    "ankerite": ("ankerite",),
    "dolomite": ("dolomite",),
}
_FASE_A_CLAVE = {
    "Calcite": "calcite",
    "Pyrite": "pyrite",
    "Gypsum": "gypsum",
    "Dawsonite": "dawsonite",
    "Ankerite": "ankerite",
    "Dolomite": "dolomite",
}
# Vm L/mol desde PROPS (cm3/mol / 1000). DC-021: silicatos cinéticos en delta_phi.
VM_KIN: dict[str, float] = {fase: vm_cm3 / 1000.0 for fase, (_, vm_cm3) in PROPS.items()}
_KIN_A_COL: dict[str, str] = {
    "Quartz": "k_quartz",
    "K-feldspar": "k_k-feldspar",
    "Albite": "k_albite",
    "Chlorite(14A)": "k_chlorite(14a)",
    "Illite": "k_illite",
    "Kaolinite": "k_kaolinite",
    "Ca-Montmorillonite": "k_ca-montmorillonite",
    "K-mica": "k_k-mica",
    "Siderite": "k_siderite",
}


def leer_selected_output(path: Path) -> pd.DataFrame:
    """Lee un SELECTED_OUTPUT de PHREEQC (whitespace-delimited)."""
    if not path.exists():
        raise FileNotFoundError(path)
    lineas = [
        ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if len(lineas) < 2:
        return pd.DataFrame()
    headers = lineas[0].split()
    filas = []
    n = len(headers)
    for ln in lineas[1:]:
        parts = ln.split()
        if len(parts) < n:
            continue
        if len(parts) > n:
            parts = parts[:n]
        filas.append(parts)
    df = pd.DataFrame(filas, columns=headers)
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except (ValueError, TypeError):
            pass
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _fila_final_paso(df: pd.DataFrame) -> pd.Series:
    """Ultima fila valida de un paso (time > 0, pH razonable)."""
    d = df.copy()
    if "time" in d.columns:
        d = d[pd.to_numeric(d["time"], errors="coerce") > 0]
    if "ph" in d.columns:
        ph = pd.to_numeric(d["ph"], errors="coerce")
        d = d[(ph > 2.5) & (ph < 13)]
    if d.empty:
        return df.iloc[-1]
    return d.iloc[-1]


def _mol_mineral(row: pd.Series, mineral: str) -> float | None:
    for alias in _ALIAS_MINERAL[mineral]:
        if alias in row.index:
            val = row[alias]
            if val is None or (isinstance(val, float) and val != val):
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _mol_kinetico(row: pd.Series, fase: str, m0: float) -> float | None:
    """Moles remanentes en KINETICS (DC-010 / DC-021).

    En met1 las columnas k_* suelen salir en 0 aunque el inventario no haya
    reaccionado; en ese caso se conserva m0. Si hay dk_* != 0, moles = m0 + dk.
    """
    col = _KIN_A_COL.get(fase)
    if col is None or col not in row.index:
        return None
    val = row[col]
    if val is None or (isinstance(val, float) and val != val):
        return None
    try:
        k_val = float(val)
    except (TypeError, ValueError):
        return None
    dk_col = "dk_" + col[2:]  # k_quartz -> dk_quartz
    dk_val = 0.0
    if dk_col in row.index:
        try:
            raw_dk = row[dk_col]
            if raw_dk is not None and not (isinstance(raw_dk, float) and raw_dk != raw_dk):
                dk_val = float(raw_dk)
        except (TypeError, ValueError):
            pass
    if abs(k_val) > 1e-12:
        return k_val
    if abs(dk_val) > 1e-12:
        return float(m0) + dk_val
    return float(m0)


def calcular_delta_phi(row: pd.Series, k: int | None = None,
                       v_bulk: float | None = None,
                       malla: str | None = None,
                       incluir_cinetica: bool = True) -> float:
    """Cambio de porosidad por disolucion/precipitacion (DC-010 / DC-021).

    Convencion (D-004, DC-010):
        delta_phi > 0  aumento de porosidad (disolucion neta)
        delta_phi < 0  disminucion (precipitacion neta)

    dphi = -(V_min - V_min_ref) / V_bulk, con V = sum(moles * V_molar).
    incluir_cinetica=True suma silicatos PK2004 (k_* en SELECTED_OUTPUT).
    """
    if v_bulk is None:
        v_bulk = v_bulk_l(malla, k)
    inv = inventario_eq(k, malla)
    v = 0.0
    v_ref = 0.0
    n_usados = 0
    for fase, m0 in inv.items():
        clave = _FASE_A_CLAVE.get(fase)
        if clave is None or clave not in VM_EQ:
            continue
        vm = VM_EQ[clave]
        moles = _mol_mineral(row, clave)
        if moles is None:
            continue
        v += moles * vm
        v_ref += float(m0) * vm
        n_usados += 1
    if incluir_cinetica:
        for fase, m0 in inventario_kin(k, malla).items():
            vm = VM_KIN.get(fase)
            if vm is None:
                continue
            moles = _mol_kinetico(row, fase, float(m0))
            if moles is None:
                continue
            v += moles * vm
            v_ref += float(m0) * vm
            n_usados += 1
    if n_usados == 0:
        return 0.0
    return -(v - v_ref) / v_bulk


def calcular_i_alm(row: pd.Series, m_water_kg: float = M_WATER_REF_KG) -> float:
    """Proxy I_alm = n_sol(C) + n_min (calcita + pirita) por celda batch 1 L."""
    c = float(row.get("c(4)", row.get("c_4", 0)) or 0)
    calcite = float(row.get("calcite", 0) or 0)
    pyrite = float(row.get("pyrite", 0) or 0)
    n_sol = c * m_water_kg
    return n_sol + calcite + pyrite


def perfil_desde_met1(run_dir: Path) -> pd.DataFrame:
    """Consolida hitos/k##_p####.txt -> perfil_quimica_met1.csv."""
    run_dir = Path(run_dir)
    malla = malla_desde_run(run_dir)
    pasos_map: dict[int, float] = {}
    pasos_file = run_dir / "pasos.json"
    if pasos_file.exists():
        raw = json.loads(pasos_file.read_text(encoding="utf-8"))
        pasos_map = {int(p["paso"]): float(p["t_fin"]) for p in raw}
    elif (run_dir / "params.json").exists():
        # inferir t_fin desde celda de inyeccion si falta pasos.json (corridas antiguas)
        import json as _json
        params = _json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
        from ccus_columna import datos as _datos
        malla, tasa = params["malla"], params["tasa_m3d"]
        k0 = _datos.orden_frente(malla, tasa)[0]
        tvec = _datos.celda(malla, tasa, k0)["tiempo_anos"].tolist()
        pasos_map = {i: tvec[i + 1] for i in range(len(tvec) - 1)}

    filas: list[dict] = []
    hitos = sorted((run_dir / "hitos").glob("k*_p*.txt"))
    pat = re.compile(r"k(\d+)_p(\d+)")
    for path in hitos:
        m = pat.search(path.name)
        if not m:
            continue
        k, paso = int(m.group(1)), int(m.group(2))
        df = leer_selected_output(path)
        row = _fila_final_paso(df)
        t = pasos_map.get(paso, float("nan"))
        ph = float(row.get("ph", np.nan))
        calcite = float(row.get("calcite", np.nan))
        c_tot = float(row.get("c(4)", np.nan))
        i_alm = calcular_i_alm(row, m_water_ref(k, malla))
        dphi = calcular_delta_phi(row, k=k, malla=malla)
        filas.append({
            "tiempo_anos": t,
            "k": k,
            "pH": ph,
            "calcita_mol": calcite,
            "C_total_mol_kgw": c_tot,
            "I_alm_mol": i_alm,
            "delta_phi": dphi,
            "phi": phi0(k) + dphi,
            "paso": paso,
        })

    out = pd.DataFrame(filas)
    if out.empty:
        return pd.DataFrame(columns=list(PERFIL_COLS))
    out = out.sort_values(["tiempo_anos", "k"]).drop(columns=["paso"], errors="ignore")
    return out[list(PERFIL_COLS)]


def perfil_desde_met2(run_dir: Path,
                      archivo: str = "transporte_reactivo.txt") -> pd.DataFrame:
    """Consolida salida TRANSPORT -> perfil_quimica_met2.csv."""
    run_dir = Path(run_dir)
    path = run_dir / archivo
    if not path.exists():
        # fallback trazador
        alt = run_dir / "transporte_trazador.txt"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"No hay {archivo} en {run_dir}")

    malla = malla_desde_run(run_dir)
    df = leer_selected_output(path)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["tiempo_anos"] = df["time"] / SEG_POR_ANO
    df["k"] = pd.to_numeric(df.get("k_cirrus", df.get("kmap", 0)), errors="coerce")
    df["ph"] = pd.to_numeric(df["ph"], errors="coerce")
    df["c(4)"] = pd.to_numeric(df.get("c(4)", 0), errors="coerce")

    calcite_col = "calcite" if "calcite" in df.columns else None
    if calcite_col:
        df["calcite"] = pd.to_numeric(df["calcite"], errors="coerce")
    else:
        df["calcite"] = np.nan

    filas = []
    for (t, k), g in df.groupby(["tiempo_anos", "k"], sort=True):
        row = g.loc[g["time"].idxmax()]
        ph = float(row["ph"])
        calcite = float(row["calcite"]) if calcite_col else np.nan
        c_tot = float(row["c(4)"])
        i_alm = calcular_i_alm(row, m_water_ref(int(k), malla)) if calcite_col else c_tot * m_water_ref(int(k), malla)
        dphi = calcular_delta_phi(row, k=int(k), malla=malla) if calcite_col else 0.0
        filas.append({
            "tiempo_anos": float(t),
            "k": int(k),
            "pH": ph,
            "calcita_mol": calcite,
            "C_total_mol_kgw": c_tot,
            "I_alm_mol": i_alm,
            "delta_phi": dphi,
            "phi": phi0(int(k)) + dphi,
        })

    out = pd.DataFrame(filas)
    if out.empty:
        return pd.DataFrame(columns=list(PERFIL_COLS))
    out = out.sort_values(["tiempo_anos", "k"])[list(PERFIL_COLS)]
    return _filtrar_columna(out)


def _filtrar_columna(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva k del sello al fondo de perforacion (DC-017). Descarta CELL_NO=0."""
    if df.empty or "k" not in df.columns:
        return df
    k = pd.to_numeric(df["k"], errors="coerce")
    return df[(k >= K_SELLO) & (k <= K_INYECCION_FONDO)].copy()


def guardar_perfil(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def anotar_delta_calcita(df: pd.DataFrame, malla: str | None = None) -> pd.DataFrame:
    """delta_calcita_mol y pct_calcita (= 100*delta/m0) por celda.

    Signo: < 0 disolucion, > 0 precipitacion. El sello k=10 tiene otro m0
    (XRD Esmeraldas); el reservorio usa el inventario de calcita (mol/L_receta, parametros) * V_bulk (DC-015).
    """
    out = df.copy()
    if "calcita_mol" not in out.columns:
        out["delta_calcita_mol"] = np.nan
        out["pct_calcita"] = np.nan
        return out
    m0 = out["k"].map(
        lambda k: inventario_eq(int(k), malla)["Calcite"] if pd.notna(k) else np.nan
    )
    out["delta_calcita_mol"] = out["calcita_mol"] - m0
    out["pct_calcita"] = np.where(
        np.asarray(m0, dtype=float) > 0,
        100.0 * out["delta_calcita_mol"] / m0,
        np.nan,
    )
    return out


def matriz_perfil_quimica(df: pd.DataFrame, variable: str,
                          ks: list[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (t, k, Z) para heatmaps; k descendente (sello arriba)."""
    if df.empty:
        raise ValueError("DataFrame vacio")
    ks = sorted(df["k"].unique() if ks is None else ks, reverse=True)
    t = np.sort(df["tiempo_anos"].unique())
    Z = np.full((len(t), len(ks)), np.nan)
    col = {"pH": "pH", "calcita": "calcita_mol", "delta_calcita": "delta_calcita_mol",
           "pct_calcita": "pct_calcita",
           "C": "C_total_mol_kgw",
           "I_alm": "I_alm_mol", "delta_phi": "delta_phi", "phi": "phi",
           }.get(variable, variable)
    for i, ti in enumerate(t):
        for j, kj in enumerate(ks):
            sub = df[(df["tiempo_anos"] == ti) & (df["k"] == kj)]
            if len(sub) and col in sub.columns:
                Z[i, j] = float(sub[col].iloc[0])
    return t, np.array(ks), Z

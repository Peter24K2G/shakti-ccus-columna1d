"""Motor del bucle celdas encadenadas (met1).

Contrato del bucle en README.md y docs/01_PLAN_IMPLEMENTACION.md §4.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[1]
if str(_RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(_RAIZ / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccus_columna.constantes import TASA_REF, TASAS_M3D  # parametros del proyecto
from ccus_columna import datos, quimica, runner
from ccus_columna.cli import listar_corridas, resolver_dir_corrida, validar_malla_tasa
from ccus_columna.constantes import (
    SEG_POR_ANO,
    T_FIN,
    T_FIN_EXTRACCION,
    T_FIN_INYECCION,
)
from ccus_columna.mapa_saturacion import estado_poro
from ccus_columna.postproceso import guardar_perfil, perfil_desde_met1

import estado as est
import transferencia as tr
from config import CorridaConfig, EstadoCelda

RAIZ_CORRIDAS = Path(__file__).resolve().parent / "corridas"


def fase_id(t_anos: float) -> int:
    if t_anos < T_FIN_EXTRACCION - 1e-9:
        return 1
    if t_anos < T_FIN_INYECCION - 1e-9:
        return 2
    return 3


def pasos_de_tiempo(cfg: CorridaConfig) -> list[tuple[float, float]]:
    """Lista de (t_ini, t_fin) en anos."""
    k0 = _celdas(cfg)[0]
    df = datos.celda(cfg.malla, cfg.tasa_m3d, k0)
    t = df["tiempo_anos"].tolist()
    if cfg.dt_anos:
        import numpy as np
        t = list(np.arange(0.0, t[-1] + cfg.dt_anos, cfg.dt_anos))
        if t[-1] < df["tiempo_anos"].iloc[-1] - 1e-9:
            t.append(float(df["tiempo_anos"].iloc[-1]))
    if cfg.t_max_anos is not None:
        t = [x for x in t if x <= cfg.t_max_anos + 1e-12]
        if len(t) < 2:
            raise ValueError("t_max_anos deja menos de 2 nodos temporales")
    pasos = list(zip(t[:-1], t[1:]))
    if cfg.max_pasos is not None:
        pasos = pasos[: cfg.max_pasos]
    return pasos


def sg_max_celda(cfg: CorridaConfig, k: int) -> float:
    r = datos.resumen_frente()
    fila = r[(r["malla"] == cfg.malla) & (r["tasa_m3d"] == cfg.tasa_m3d) & (r["k"] == k)]
    return float(fila["Sg_max"].iloc[0]) if len(fila) else 1.0


def mapa_saturacion_celda(cfg: CorridaConfig, k: int, t_mid: float) -> dict:
    """Mapa S->estado del poro para la celda k en el punto medio del paso."""
    s = datos.interpolar_celda(cfg.malla, cfg.tasa_m3d, k, t_mid)
    return estado_poro(
        s["Sg"], s["Sw"], s["So"], s["P_bar"], sg_max_celda(cfg, k),
        usar_p_real=cfg.usar_p_real, k=k, malla=cfg.malla,
    )


def _bloque_gas(fase: int, sg: float, v_gas: float, p_co2: float) -> str:
    if sg <= 1e-8 or p_co2 <= 0.0:
        return f"""DELETE
    -gas_phase {est.SOL_LOCAL}
"""
    if fase == 2:
        return f"""GAS_PHASE {est.SOL_LOCAL}
    -fixed_pressure
    -volume    {v_gas:.6f}
    CO2(g)     {p_co2:.6f}
USE gas_phase {est.SOL_LOCAL}
"""
    return f"""GAS_PHASE {est.SOL_LOCAL}
    -volume    {v_gas:.6f}
    CO2(g)     {p_co2:.6f}
USE gas_phase {est.SOL_LOCAL}
"""


def _user_punch(t_ini: float, t_fin: float, sg_ini: float, sg_fin: float,
                 sw_ini: float, sw_fin: float, fase: int, k: int) -> str:
    seg = max(1, int(round((t_fin - t_ini) * SEG_POR_ANO)))
    t_off = int(round(t_ini * SEG_POR_ANO))
    return f"""USER_PUNCH 1
    -headings   anos_acum sg sw fase k_celda
    -start
    10 frac = TIME / {seg}
    20 IF (frac > 1) THEN frac = 1
    30 sg_v = {sg_ini:.10f} + ({sg_fin:.10f} - {sg_ini:.10f}) * frac
    40 sw_v = {sw_ini:.10f} + ({sw_fin:.10f} - {sw_ini:.10f}) * frac
    50 anos_v = ({t_off} + TIME) / 86400.0 / 365.25
    60 PUNCH anos_v, sg_v, sw_v, {fase}, {k}
    -end
"""


def pqi_paso_celda(cfg: CorridaConfig, k: int, paso: int,
                   t_ini: float, t_fin: float,
                   run_dir: Path,
                   dump_previo: Path | None,
                   efluente_previo: Path | None,
                   f_transfer: float) -> str:
    """Texto .pqi de UN paso de UNA celda."""
    t_mid = 0.5 * (t_ini + t_fin)
    s_ini = datos.interpolar_celda(cfg.malla, cfg.tasa_m3d, k, t_ini)
    s_fin = datos.interpolar_celda(cfg.malla, cfg.tasa_m3d, k, t_fin)
    mapa = mapa_saturacion_celda(cfg, k, t_mid)
    fase = fase_id(t_mid)
    dt_s = max(1, int(round((t_fin - t_ini) * SEG_POR_ANO)))
    dsg = abs(s_fin["Sg"] - s_ini["Sg"])
    n_kin = max(
        cfg.n_kinetics_base,
        min(120, int(math.ceil(dsg / 0.40 * 120)) if dsg > 1e-12 else cfg.n_kinetics_base),
    )

    so_path = f"hitos/k{k:02d}_p{paso:04d}.txt"
    dump_path = est.ruta_dump(run_dir, k, paso)
    efluente_path = est.ruta_efluente(run_dir, k, paso)
    partes: list[str] = [
        f"# met1 k={k} paso={paso} t={t_ini:.4f}-{t_fin:.4f}a fase={fase}",
        f"# m_water={mapa['m_water']:.6f} v_gas={mapa['v_gas']:.6f} "
        f"P_CO2={mapa['p_co2_atm']:.4f} atm f_transfer={f_transfer:.4f}",
        quimica.prefijo_definiciones(),
        quimica.selected_output_batch(so_path),
    ]

    punch = _user_punch(
        t_ini, t_fin, s_ini["Sg"], s_fin["Sg"], s_ini["Sw"], s_fin["Sw"], fase, k,
    )

    if dump_previo is None:
        partes.append(est.bloque_inicial(
            mapa["m_water"], max(mapa["p_bar"] * 0.986923, 1.0), k=k,
            malla=cfg.malla,
        ))
    else:
        partes.append(est.bloque_restaurar(run_dir, dump_previo))

    if efluente_previo is not None and f_transfer > 1e-12:
        partes.append(est.bloque_cargar_efluente(run_dir, efluente_previo))
        partes.append(tr.bloque_mix(f_transfer))

    gas = _bloque_gas(fase, mapa["sg"], mapa["v_gas"], mapa["p_co2_atm"])
    partes.append(f"""{punch}
USE solution {est.SOL_LOCAL}
USE equilibrium_phases {est.SOL_LOCAL}
USE kinetics {est.SOL_LOCAL}
SOLUTION_MODIFY {est.SOL_LOCAL}
    -mass_water    {mapa['m_water']:.6f}
{gas}KINETICS {est.SOL_LOCAL}
    -steps {dt_s} in {n_kin}
SAVE equilibrium_phases {est.SOL_LOCAL}
SAVE solution {est.SOL_LOCAL}
END
{est.bloque_dump(run_dir, dump_path)}
{est.bloque_dump_efluente(run_dir, efluente_path)}
""")
    return "\n".join(partes)


def _celdas(cfg: CorridaConfig) -> list[int]:
    if cfg.celdas is not None:
        return list(cfg.celdas)
    return tr.orden_de_flujo(cfg.malla, cfg.tasa_m3d, dominio=cfg.dominio)


def _leer_ultima_fila_selected(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="replace") as fh:
        lineas = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if len(lineas) < 2:
        return None
    headers = lineas[0].split()
    vals = lineas[-1].split()
    if len(vals) < len(headers):
        return None
    return dict(zip(headers, vals))


def consolidar_hitos(run_dir: Path, celdas: list[int]) -> None:
    """Une selected_outputs por celda en hitos_k##.csv."""
    for k in celdas:
        filas = []
        for p in sorted((run_dir / "hitos").glob(f"k{k:02d}_p*.txt")):
            row = _leer_ultima_fila_selected(p)
            if row:
                row["archivo"] = p.name
                filas.append(row)
        if not filas:
            continue
        fields: list[str] = []
        for row in filas:
            for key in row:
                if key not in fields:
                    fields.append(key)
        out = run_dir / f"hitos_k{k:02d}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(filas)


def _guardar_pasos(run_dir: Path, pasos: list[tuple[float, float]]) -> None:
    payload = [{"paso": i, "t_ini": t0, "t_fin": t1} for i, (t0, t1) in enumerate(pasos)]
    (run_dir / "pasos.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cargar_estado_desde_disco(run_dir: Path, celdas: list[int], paso: int) -> dict[int, EstadoCelda]:
    """Restaura referencias DUMP/efluente desde archivos existentes."""
    estados = {k: EstadoCelda(k) for k in celdas}
    for k in celdas:
        dump = est.ruta_dump(run_dir, k, paso)
        eflu = est.ruta_efluente(run_dir, k, paso)
        if dump.exists():
            estados[k].dump_path = dump
        if eflu.exists():
            estados[k].efluente_path = eflu
    return estados


def _paso_completo(run_dir: Path, celdas: list[int], paso: int) -> bool:
    return all(est.ruta_dump(run_dir, k, paso).exists() for k in celdas)


def exportar_resultados(run_dir: Path) -> Path:
    """Genera perfil_quimica_met1.csv en el directorio de corrida."""
    df = perfil_desde_met1(run_dir)
    return guardar_perfil(df, run_dir / "perfil_quimica_met1.csv")


def _rmtree_robusto(path: Path) -> None:
    """Elimina un arbol en Windows/UNC (WinError 145 si rmtree directo falla)."""
    import os
    import shutil
    import stat
    import time

    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass

    if not path.exists():
        return
    for intento in range(3):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except OSError:
            if intento == 2:
                raise
            time.sleep(0.5)
    if path.exists():
        for item in sorted(path.rglob("*"), reverse=True):
            if item.is_file() or item.is_symlink():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                try:
                    item.rmdir()
                except OSError:
                    pass
        path.rmdir()


def correr(cfg: CorridaConfig, reanudar: bool = False) -> Path:
    """Bucle principal. Devuelve el directorio de la corrida."""
    import shutil

    run_dir = RAIZ_CORRIDAS / cfg.id_corrida()
    if run_dir.exists() and not reanudar:
        for sub in ("estado", "hitos"):
            d = run_dir / sub
            if d.exists():
                _rmtree_robusto(d)
        for stale in run_dir.glob("k*_p*.pq*"):
            stale.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "estado").mkdir(exist_ok=True)
    (run_dir / "hitos").mkdir(exist_ok=True)
    cfg.guardar(run_dir)

    celdas = _celdas(cfg)
    pasos = pasos_de_tiempo(cfg)
    _guardar_pasos(run_dir, pasos)
    errores: list[str] = []

    paso_ini = 0
    estados = {k: EstadoCelda(k) for k in celdas}
    if reanudar:
        for p in range(len(pasos) - 1, -1, -1):
            if _paso_completo(run_dir, celdas, p):
                paso_ini = p + 1
                if paso_ini > 0:
                    estados = _cargar_estado_desde_disco(run_dir, celdas, paso_ini - 1)
                break
        if paso_ini > 0:
            print(f"Reanudando desde paso {paso_ini}/{len(pasos)}")

    total = len(pasos) * len(celdas)
    hecho = paso_ini * len(celdas)

    for paso, (t0, t1) in enumerate(pasos):
        if paso < paso_ini:
            continue
        if _paso_completo(run_dir, celdas, paso):
            estados = _cargar_estado_desde_disco(run_dir, celdas, paso)
            hecho += len(celdas)
            continue

        if paso > 0:
            for k in celdas:
                prev = est.ruta_dump(run_dir, k, paso - 1)
                if not prev.exists():
                    raise RuntimeError(f"Falta estado previo {prev} para paso {paso}")
                estados[k].dump_path = prev
                estados[k].efluente_path = None

        for i, k in enumerate(celdas):
            k_prev = celdas[i - 1] if i > 0 else None
            f = 0.0
            efluente_prev = None
            if cfg.transferir and k_prev is not None:
                f = tr.fraccion_transferida(
                    cfg.regla_f, t1 - t0, cfg.malla, cfg.tasa_m3d,
                    k_prev, k, t0, cfg.f_constante,
                    activacion_f=cfg.activacion_f,
                    sg_umbral=cfg.sg_umbral,
                    dp_umbral_bar=cfg.dp_umbral_bar,
                )
                efluente_prev = estados[k_prev].efluente_path

            dump_prev = None if paso == 0 else est.ruta_dump(run_dir, k, paso - 1)

            pqi = pqi_paso_celda(
                cfg, k, paso, t0, t1, run_dir,
                dump_prev, efluente_prev, f,
            )
            name = f"k{k:02d}_p{paso:04d}"
            res = runner.run_in_dir(pqi, run_dir, name=name)
            hecho += 1
            if hecho % 10 == 0 or hecho == total:
                print(f"  progreso {hecho}/{total}  paso {paso+1}/{len(pasos)} k={k} t={t1:.2f}a")

            if not res.ok:
                msg = f"k={k} paso={paso} t={t0:.3f}-{t1:.3f}: " + "; ".join(res.errors[:3])
                errores.append(msg)
                (run_dir / "errores.txt").write_text("\n".join(errores), encoding="utf-8")
                raise RuntimeError(f"PHREEQC fallo: {msg}")

            estados[k].dump_path = est.ruta_dump(run_dir, k, paso)
            estados[k].efluente_path = est.ruta_efluente(run_dir, k, paso)
            so = run_dir / "hitos" / f"k{k:02d}_p{paso:04d}.txt"
            row = _leer_ultima_fila_selected(so)
            if row:
                row.update({"paso": str(paso), "t_ini": f"{t0:.6f}", "t_fin": f"{t1:.6f}", "f": f"{f:.6f}"})
                estados[k].historico.append(row)

    consolidar_hitos(run_dir, celdas)
    csv_path = exportar_resultados(run_dir)
    (run_dir / "OK.txt").write_text(
        f"pasos={len(pasos)} celdas={celdas}\nperfil={csv_path.name}\n", encoding="utf-8",
    )
    return run_dir


def config_smoke() -> CorridaConfig:
    """Corrida corta: celda k15 aislada, pocos pasos (validacion entorno)."""
    return CorridaConfig(
        malla="malla_original",
        tasa_m3d=TASA_REF,
        celdas=[15],
        max_pasos=5,
        transferir=False,
        etiqueta="smoke_k15",
    )


def config_rama_corta(malla: str = "malla_original", tasa: int = TASA_REF) -> CorridaConfig:
    """Rama al sello, limitada a ~15 a (llegada a k14) para prueba de transferencia."""
    return CorridaConfig(
        malla=malla,
        tasa_m3d=tasa,
        t_max_anos=15.0,
        etiqueta="rama_t15",
    )


def config_completo(malla: str = "malla_original", tasa: int = TASA_REF) -> CorridaConfig:
    """Columna k30->k10, transporte hacia sello, 70 a, nodos CSV (~183 pasos)."""
    return CorridaConfig(
        malla=malla,
        tasa_m3d=tasa,
        celdas=None,
        dominio="columna_pozo",
        transferir=True,
        etiqueta="completo",
    )


def config_rama_completo(malla: str = "malla_original", tasa: int = TASA_REF) -> CorridaConfig:
    """Rama al sello k15->k10 solamente (v1 legacy)."""
    return CorridaConfig(
        malla=malla,
        tasa_m3d=tasa,
        celdas=None,
        dominio="rama_sello",
        transferir=True,
        etiqueta="completo_rama",
    )


def config_completo_mejorado(malla: str = "malla_original", tasa: int = TASA_REF,
                             dp_umbral_bar: float = 0.5) -> CorridaConfig:
    """Columna k30->k10 con activacion Sg OR dP (DC-021, recomendacion revisor)."""
    etiqueta = "completo_dp" if abs(dp_umbral_bar - 0.5) < 1e-9 else f"completo_dp_{dp_umbral_bar:g}bar"
    return CorridaConfig(
        malla=malla,
        tasa_m3d=tasa,
        celdas=None,
        dominio="columna_pozo",
        transferir=True,
        activacion_f="sg_o_dp",
        dp_umbral_bar=dp_umbral_bar,
        etiqueta=etiqueta,
    )


def config_completo_anual(malla: str = "malla_original", tasa: int = TASA_REF) -> CorridaConfig:
    """Columna perforacion->sello con malla anual (71 pasos)."""
    return CorridaConfig(
        malla=malla,
        tasa_m3d=tasa,
        dt_anos=1.0,
        dominio="columna_pozo",
        transferir=True,
        etiqueta="completo_anual",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Correr met1 celdas encadenadas")
    ap.add_argument("--smoke", action="store_true", help="celda k15, 5 pasos")
    ap.add_argument("--rama-corta", action="store_true", help="rama sello hasta 15 a")
    ap.add_argument("--completo", action="store_true",
                    help="columna perforacion->sello, 70 a (nodos CSV, DC-019)")
    ap.add_argument("--completo-rama", action="store_true",
                    help="solo rama k15->k10, 70 a (v1 legacy)")
    ap.add_argument("--completo-anual", action="store_true",
                    help="columna perforacion->sello, paso 1 a")
    ap.add_argument("--completo-mejorado", action="store_true",
                    help="columna k30->k10, activacion Sg OR dP (DC-021)")
    ap.add_argument("--dp-umbral", type=float, default=0.5,
                    help="umbral dP (bar) para --completo-mejorado (DC-022 sensibilidad)")
    ap.add_argument("--reanudar", action="store_true", help="continuar corrida interrumpida")
    ap.add_argument("--figuras", dest="figuras", action="store_true", default=True,
                    help="generar figuras al terminar (por defecto: si)")
    ap.add_argument("--sin-figuras", dest="figuras", action="store_false",
                    help="no generar PNG al terminar")
    ap.add_argument("--listar", action="store_true", help="listar corridas existentes y salir")
    ap.add_argument("--malla", default="malla_original",
                    help="malla_original o malla_refinada")
    ap.add_argument("--tasa", type=int, default=TASA_REF,
                    help="tasas disponibles: " + ", ".join(str(q) for q in TASAS_M3D))
    ap.add_argument("--max-pasos", type=int, default=None)
    ap.add_argument("--t-max", type=float, default=None)
    ap.add_argument("--celda", type=int, action="append", default=None,
                    help="repetible; si se omite usa rama al sello")
    ap.add_argument("--sin-transferencia", action="store_true")
    ap.add_argument("--solo-exportar", type=str, default=None,
                    help="nombre o ruta de una corrida existente (NO use <id>)")
    args = ap.parse_args(argv)

    if args.listar:
        print(f"Corridas en {RAIZ_CORRIDAS}:")
        print(listar_corridas(RAIZ_CORRIDAS))
        return 0

    if args.solo_exportar:
        run_dir = resolver_dir_corrida(args.solo_exportar, RAIZ_CORRIDAS)
        exportar_resultados(run_dir)
        if args.figuras:
            from generar_figuras import figuras_met1
            for p in figuras_met1(run_dir):
                print(f"  figura: {p}")
        print(f"Exportado -> {run_dir / 'perfil_quimica_met1.csv'}")
        return 0

    validar_malla_tasa(args.malla, args.tasa)

    if args.smoke:
        cfg = config_smoke()
    elif args.rama_corta:
        cfg = config_rama_corta(args.malla, args.tasa)
    elif args.completo:
        cfg = config_completo(args.malla, args.tasa)
    elif args.completo_mejorado:
        cfg = config_completo_mejorado(args.malla, args.tasa, dp_umbral_bar=args.dp_umbral)
    elif args.completo_rama:
        cfg = config_rama_completo(args.malla, args.tasa)
    elif args.completo_anual:
        cfg = config_completo_anual(args.malla, args.tasa)
    else:
        cfg = CorridaConfig(
            malla=args.malla,
            tasa_m3d=args.tasa,
            celdas=args.celda,
            max_pasos=args.max_pasos,
            t_max_anos=args.t_max,
            transferir=not args.sin_transferencia,
            etiqueta="custom" if args.celda or args.max_pasos or args.t_max else "",
        )
    print(f"Corrida {cfg.id_corrida()} celdas={_celdas(cfg)} pasos={len(pasos_de_tiempo(cfg))}")
    run_dir = correr(cfg, reanudar=args.reanudar)
    if args.figuras:
        from generar_figuras import figuras_met1
        for p in figuras_met1(run_dir):
            print(f"  figura: {p}")
    print(f"OK -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

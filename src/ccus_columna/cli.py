"""Ayudas de consola: validar malla/tasa y resolver carpetas de corrida."""

from __future__ import annotations

from pathlib import Path

from ccus_columna.constantes import DATOS_CSV, MALLAS, TASA_REF, TASAS_POR_MALLA

# TASAS_POR_MALLA viene de los parametros del proyecto (ccus_columna.constantes).


def tasas_disponibles(malla: str) -> tuple[int, ...]:
    qdir = DATOS_CSV / malla
    if not qdir.is_dir():
        return TASAS_POR_MALLA.get(malla, ())
    encontradas = []
    for p in sorted(qdir.iterdir()):
        if p.is_dir() and p.name.startswith("q") and p.name[1:].isdigit():
            encontradas.append(int(p.name[1:]))
    return tuple(encontradas) if encontradas else TASAS_POR_MALLA.get(malla, ())


def validar_malla_tasa(malla: str, tasa: int) -> None:
    if malla not in MALLAS:
        raise SystemExit(
            f"Malla desconocida: {malla!r}\n"
            f"  Use --malla malla_original  o  --malla malla_refinada"
        )
    tasas = tasas_disponibles(malla)
    if tasa not in tasas:
        lista = ", ".join(str(t) for t in tasas) or "(ninguna)"
        extra = ""
        if tasa in TASAS_POR_MALLA.get("malla_original", ()) and tasa not in tasas:
            extra = f"\n  Nota: la malla {malla} no tiene tasa {tasa}."
        raise SystemExit(
            f"Tasa {tasa} no existe para {malla}.\n"
            f"  Tasas disponibles: {lista}{extra}"
        )


def listar_corridas(raiz: Path) -> str:
    if not raiz.is_dir():
        return f"  (no hay carpeta {raiz})"
    dirs = sorted(p.name for p in raiz.iterdir() if p.is_dir())
    if not dirs:
        return "  (ninguna corrida todavia)"
    return "\n".join(f"  {d}" for d in dirs)


def resolver_dir_corrida(arg: str, raiz_corridas: Path) -> Path:
    """Acepta ruta completa o solo el nombre de la carpeta. Rechaza placeholders."""
    texto = (arg or "").strip()
    if not texto or "<" in texto or ">" in texto:
        raise SystemExit(
            "No copies el placeholder de la documentacion.\n"
            "Use el nombre REAL de la carpeta de corrida, por ejemplo:\n"
            "  python met1_celdas_encadenadas/motor.py --solo-exportar "
            f"malla_original_q{TASA_REF}_completo_anual --figuras\n\n"
            f"Corridas en {raiz_corridas}:\n{listar_corridas(raiz_corridas)}"
        )

    candidatos = [
        Path(texto),
        raiz_corridas / texto,
        raiz_corridas / Path(texto).name,
    ]
    for c in candidatos:
        if c.is_dir():
            return c.resolve()

    raise SystemExit(
        f"No existe la corrida: {texto}\n"
        f"Corridas en {raiz_corridas}:\n{listar_corridas(raiz_corridas)}"
    )

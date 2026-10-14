# shakti-ccus-columna1d

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23353992.svg)](https://doi.org/10.5281/zenodo.23353992)

Columna 1D de transporte reactivo para el almacenamiento geológico de CO2 en la Fm. La Paz con sello Fm. Esmeraldas (proyecto SHAKTI, Universidad Nacional de Colombia, Facultad de Ingeniería – línea CCUS).

**English summary:** Python code for a 1D reactive-transport column for CO2 storage (La Paz Fm. reservoir, Esmeraldas Fm. seal). Gas saturation/pressure from the Cirrus flow simulator drives PHREEQC 3.8.6 geochemistry through two methods: *met1* (chained batch cells) and *met2* (PHREEQC `TRANSPORT`). **No site data, input files or results are distributed**: all project parameters are read from an external YAML file; the repository only ships `config/parametros_ejemplo.yaml` with clearly fictitious values that document the schema.

## Estructura
- `src/ccus_columna/` – paquete común (parámetros, constantes, geometría, mineralogía, cinética Palandri & Kharaka 2004, lectura de Cirrus, postproceso, ejecución de PHREEQC).
- `met1_celdas_encadenadas/` – método 1, celdas encadenadas.
- `met2_transporte_phreeqc/` – método 2, bloque TRANSPORT de PHREEQC (`plantillas/transporte_base.pqi` es una semilla sintáctica con valores ficticios).
- `comparacion/` – comparación met1 vs met2 y figuras.
- `scripts/` – utilidades (exportación xlsx→csv, volúmenes, figuras, verificación del entorno).
- `sensibilidad/` – envolturas de sensibilidad OAT (met1, met2).
- `config/parametros_ejemplo.yaml` – esquema de parámetros con **valores ficticios**.
- `tests/`, `met*/tests/` – pruebas (pytest).

## Instalación
1. Python 3.10+ y `pip install -r requirements.txt`.
2. **PHREEQC 3.8.6** (USGS, versión batch). Por defecto se usa la instalación estándar de Windows
   (`C:\Program Files\USGS\phreeqc-3.8.6-17100-x64\...`, ver `src/ccus_columna/runner.py`); otra ubicación se indica con
   las variables de entorno `PHREEQC_EXE` (ejecutable) y `PHREEQC_DB` (`phreeqc.dat`).

## Parámetros y datos de entrada (NO incluidos)
- **Parámetros del sitio** (porosidades, permeabilidades, etapas operativas, tasas de inyección, geometría de la columna,
  composición de la brina, XRD del sello, inventarios minerales y áreas reactivas): se leen de un YAML externo, en este orden:
  1. variable de entorno `SHAKTI_PARAMS=<ruta al yaml>`;
  2. `datos/parametros.yaml` (ignorado por git);
  3. `config/parametros_ejemplo.yaml` (**valores ficticios**, solo para probar la instalación; se emite una advertencia).
  Copie el archivo de ejemplo a `datos/parametros.yaml` y reemplace los valores con los del proyecto.
- **Resultados de flujo de Cirrus** (Sg, presión por celda y escenario) exportados a `datos/csv/` (`scripts/exportar_xlsx_a_csv.py` convierte los xlsx).

Los parámetros reales y los datos de Cirrus pertenecen al proyecto SHAKTI y se solicitan al grupo de investigación (UNAL, Facultad de Ingeniería). Las corridas se escriben en `*/corridas/` y las figuras en `figuras/` (ignorados por git).

## Ejecución
- `python -m pytest`: con solo los parámetros de ejemplo, las pruebas que requieren datos reales se omiten (*skip*).
- `python scripts/verificar_entorno.py`: revisa dependencias, PHREEQC y datos.
- met1: `python met1_celdas_encadenadas/motor.py --help` (p. ej. `--smoke`).
- met2: `python met2_transporte_phreeqc/generador.py --help`.
- Comparación y figuras: scripts en `comparacion/`.

## Cómo citar
Romero León, P. J., & Rendón, J. (2026). *SHAKTI CCUS – Columna 1D geoquímica (PHREEQC) acoplada a Cirrus* (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23353992

Metadatos en `CITATION.cff`.

## Licencia
MIT (ver `LICENSE`).

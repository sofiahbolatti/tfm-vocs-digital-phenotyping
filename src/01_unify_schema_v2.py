"""
Unificacion de esquema (version 2 - basada en el codigo real de los loaders)
=============================================================================
Este script parte de que la tabla `muestras` YA EXISTE con las filas de
Lazazzara2018 y Laupheimer2024 (comparten esquema via to_sql(if_exists="append")).

Las 4 tablas restantes tienen su propio esquema porque incluyen columnas que
`muestras` no tiene:
    - muestras_alinc2026   -> fungal_inoculant, herbivore_species, treatment_group, zenodo_doi
    - muestras_ayelo2026   -> cultivar_code, cultivar_resistance, herbivore_population,
                               treatment_group, voc_total_reported
    - muestras_moreira2024 -> herbivore_species, treatment_group, height_cm,
                               voc_total_reported, dryad_doi, raw_id
    - muestras_ghosh2022   -> virus, herbivore_species, treatment_group

Este script:
  1. Agrega a `muestras` dos columnas nuevas: eje_biologico, metadata_extra (JSON)
  2. Backfillea eje_biologico='estandar' en las filas existentes (Lazazzara/Laupheimer)
  3. Inserta las filas de las 4 tablas outlier en `muestras`, mapeando lo que
     coincide en nombre/tipo y empaquetando el resto en metadata_extra

No borra ninguna tabla original (muestras_alinc2026, etc. quedan de respaldo).

Uso:
    python src/01_unify_schema_v2.py db/tfm_vocs.db
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone

COLUMNAS_MUESTRAS_ACTUALES = {
    "sample_id", "species", "genotype", "genotype_notes", "microorganism",
    "strain", "physiological_state", "dataset_origin", "doi",
    "analytical_technique", "timepoint", "tissue", "experiment_replicate",
    "biological_replicate", "load_date", "validation_status", "intended_use",
}

TABLAS_OUTLIER = {
    "muestras_alinc2026": "herbivoria_biocontrol",
    "muestras_ayelo2026": "resistencia_cultivar_herbivoria",
    "muestras_moreira2024": "herbivoria",
    "muestras_ghosh2022": "infeccion_viral",
}


def columnas_de_tabla(cur, tabla):
    cur.execute(f"PRAGMA table_info({tabla});")
    return [row[1] for row in cur.fetchall()]


def asegurar_columnas_nuevas(cur):
    cols_actuales = columnas_de_tabla(cur, "muestras")
    if "eje_biologico" not in cols_actuales:
        cur.execute("ALTER TABLE muestras ADD COLUMN eje_biologico TEXT;")
        print("Columna 'eje_biologico' agregada a `muestras`.")
    if "metadata_extra" not in cols_actuales:
        cur.execute("ALTER TABLE muestras ADD COLUMN metadata_extra TEXT;")
        print("Columna 'metadata_extra' agregada a `muestras`.")


def backfill_estandar(cur):
    cur.execute(
        "UPDATE muestras SET eje_biologico = 'estandar' "
        "WHERE eje_biologico IS NULL;"
    )
    print(f"Backfill eje_biologico='estandar': {cur.rowcount} filas actualizadas.")


def fusionar_tabla_outlier(cur, tabla_origen, eje_biologico, fecha):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (tabla_origen,))
    if not cur.fetchone():
        print(f"[AVISO] Tabla '{tabla_origen}' no existe. Se omite.")
        return 0

    cols_origen = columnas_de_tabla(cur, tabla_origen)
    cur.execute(f"SELECT * FROM {tabla_origen};")
    filas = cur.fetchall()

    insertadas = 0
    for fila in filas:
        row_dict = dict(zip(cols_origen, fila))
        sample_id = row_dict.get("sample_id")
        if not sample_id:
            continue

        comunes = {}
        extra = {}
        for col, val in row_dict.items():
            if col in COLUMNAS_MUESTRAS_ACTUALES:
                comunes[col] = val
            else:
                extra[col] = val

        metadata_extra_json = json.dumps(extra, ensure_ascii=False, default=str)

        columnas_insert = list(comunes.keys()) + ["eje_biologico", "metadata_extra"]
        valores_insert = list(comunes.values()) + [eje_biologico, metadata_extra_json]
        placeholders = ", ".join(["?"] * len(columnas_insert))
        columnas_sql = ", ".join(columnas_insert)

        try:
            cur.execute(
                f"INSERT INTO muestras ({columnas_sql}) VALUES ({placeholders});",
                valores_insert,
            )
            insertadas += 1
        except sqlite3.Error as e:
            print(f"  [ERROR] sample_id={sample_id} en {tabla_origen}: {e}")

    print(f"{tabla_origen}: {insertadas}/{len(filas)} filas fusionadas en `muestras` (eje={eje_biologico}).")
    return insertadas


def migrar(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    fecha = datetime.now(timezone.utc).isoformat()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='muestras';")
    if not cur.fetchone():
        print("[ERROR] La tabla `muestras` no existe todavia.")
        conn.close()
        return

    asegurar_columnas_nuevas(cur)
    conn.commit()

    backfill_estandar(cur)
    conn.commit()

    total = 0
    for tabla, eje in TABLAS_OUTLIER.items():
        total += fusionar_tabla_outlier(cur, tabla, eje, fecha)
        conn.commit()

    cur.execute("SELECT COUNT(*) FROM muestras;")
    total_final = cur.fetchone()[0]
    cur.execute("SELECT eje_biologico, COUNT(*) FROM muestras GROUP BY eje_biologico;")
    por_eje = cur.fetchall()
    cur.execute("SELECT species, COUNT(*) FROM muestras GROUP BY species;")
    por_especie = cur.fetchall()

    print("\n=== RESUMEN ===")
    print(f"Filas fusionadas desde tablas outlier: {total}")
    print(f"Total filas en `muestras` (todo unificado): {total_final}")
    print("\nPor eje biologico:")
    for eje, n in por_eje:
        print(f"  {eje}: {n}")
    print("\nPor especie:")
    for especie, n in por_especie:
        print(f"  {especie}: {n}")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python src/01_unify_schema_v2.py db/tfm_vocs.db")
        sys.exit(1)
    migrar(sys.argv[1])

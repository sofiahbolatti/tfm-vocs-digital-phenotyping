"""
Loader: Ghosh et al. 2022 (DOI 10.3390/insects13090840)
Tomate (S. lycopersicum) + Pimiento (Capsicum annuum) x virus transmitidos pormosca blanca (Bemisia tabaci) 


Estructura de cada hoja (identica en concepto, distinta en nombres de columna):
- Fila 4: encabezados: Compound, Cas No, Class, [muestras x 3  grupos], f.value, p.value, -log10(p)/-log(10), FDR
- Filas 5 en adelante: un compuesto por fila
- Tomate: 76 compuestos, 15 muestras (5 control + 5 healthy whitefly + 5 virus/TYLCV)
- Pimiento: 93 compuestos, 14 muestras (5 control + 5 healthy whitefly + 4 virus/PeWBVYV)
- Abundancia: son valores normalizados contra un estandar interno. Se guardan tal cual, unidad "relative_to_IS".

Diseno biologico (3 grupos por especie):
- control = planta sin mosca blanca, sin virus
- Healthy  = planta infestada con mosca blanca SIN virus (vector sano)
- Virus = planta infestada con mosca blanca portadora del virus
"""
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

EXCEL_PATH = "data/raw/insects-1892196-supplementary.xlsx"

DB_PATH = "output/tfm_vocs.db"
PARQUET_DIR = "output/data/processed"
PARQUET_PATH = f"{PARQUET_DIR}/abundancias_ghosh2022.parquet"

DATASET_ORIGIN = "Ghosh2022"
DOI = "10.3390/insects13090840"
ANALYTICAL_TECHNIQUE = "HS-SPME-GC-MS"
TISSUE = "leaves"

HEADER_ROW = 3 


SPECIES_CONFIG = {
    "tomato": {
        "sheet_name": "3W tomato VOCs",
        "species": "Solanum lycopersicum",
        "virus": "TYLCV (Tomato yellow leaf curl virus)",
        "sample_columns": {
            "control": ["control 3W1", "control 3W2", "control 3W3", "control 3W4", "control 3W5"],
            "healthy_whitefly": ["Healthy3W1", "Healthy3W2", "Healthy3W3", "Healthy3W4", "Healthy3W5"],
            "virus": ["Virus3W1", "Virus3W2", "Virus3W3", "Virus3W4", "Virus3W5"],
        },
    },
    "pepper": {
        "sheet_name": "3W pepper VOCs",
        "species": "Capsicum annuum",
        "virus": "PeWBVYV (Pepper whitefly-borne vein yellows virus)",
        "sample_columns": {
            "control": ["Control 3w 1", "Control 3w 2", "Control 3w 3", "Control 3w 4", "Control 3w 5"],
            "healthy_whitefly": ["Healthy 3w 1", "Healthy 3w 2", "Healthy 3w 3", "Healthy 3w 4", "Healthy 3w 5"],
            "virus": ["Virus 3w 1", "Virus 3w 2", "Virus 3w 3", "Virus 3w 4"],  # solo 4 replicas, no 5
        },
    },
}

TREATMENT_TO_STATE = {
    "control": "control",
    "healthy_whitefly": "herbivory_only",
    "virus": "infected_viral",
}

HERBIVORE = "Bemisia tabaci (whitefly)"


# -----------------------------------------------------------------------------
# Paso 1: leer una hoja del Excel
# ---------------------------------------------------------------------------

def read_sheet(excel_path: str, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=HEADER_ROW)
    df = df.dropna(subset=[df.columns[0]]).reset_index(drop=True)  
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Paso 2: construir tabla de muestras
# ---------------------------------------------------------------------------

def build_species_tables(species_key: str, cfg: dict, df: pd.DataFrame):
    species = cfg["species"]
    virus = cfg["virus"]
    sample_columns = cfg["sample_columns"]


    muestra_rows = []
    col_to_sample_id = {}
    for treatment, cols in sample_columns.items():
        for i, col in enumerate(cols, start=1):
            sample_id = f"{DATASET_ORIGIN.lower()}_{species_key}_{treatment}_r{i:02d}"
            col_to_sample_id[col] = sample_id
            muestra_rows.append({
                "sample_id": sample_id,
                "species": species,
                "physiological_state": TREATMENT_TO_STATE[treatment],
                "virus": virus if treatment == "virus" else None,
                "herbivore_species": HERBIVORE if treatment in ("healthy_whitefly", "virus") else None,
                "treatment_group": treatment,
                "biological_replicate": i,
                "dataset_origin": DATASET_ORIGIN,
                "doi": DOI,
                "analytical_technique": ANALYTICAL_TECHNIQUE,
                "tissue": TISSUE,
                "load_date": date.today().isoformat(),
                "validation_status": "pendiente_validacion",
                "intended_use": "classifier",
            })
    muestras = pd.DataFrame(muestra_rows)

# ---------------------------------------------------------------------------
# Paso 3: construir un catalogo de compuestos 
# -----------------------------------------------------------------------------
    compuesto_rows = []
    for i, row in df.iterrows():
        compuesto_rows.append({
            "compuesto_id": f"{DATASET_ORIGIN.lower()}_{species_key}_cmp{i+1:03d}",
            "name_original": row["Compound"],
            "cas": row["Cas No"],
            "compound_class": row.get("Class"),
            "identified": True,
            "pubchem_cid": None, 
            "dataset_origin": DATASET_ORIGIN,
            "species": species,
        })
    compuestos = pd.DataFrame(compuesto_rows)

  # ----------------------------------------------------------------------------
# Paso 4: construir una matriz de abundancias
# ---------------------------------------------------------------------------
    all_sample_cols = [c for cols in sample_columns.values() for c in cols]
    long_rows = []
    for i, row in df.iterrows():
        compuesto_id = f"{DATASET_ORIGIN.lower()}_{species_key}_cmp{i+1:03d}"
        for col in all_sample_cols:
            long_rows.append({
                "sample_id": col_to_sample_id[col],
                "compuesto_id": compuesto_id,
                "abundancia": float(row[col]),
                "unidad": "relative_to_internal_standard",
            })
    abundancias = pd.DataFrame(long_rows)

    return muestras, compuestos, abundancias


# ---------------------------------------------------------------------------
# Paso 5: guardar en SQLite + Parquet
# ---------------------------------------------------------------------------

def save_to_sqlite(muestras: pd.DataFrame, compuestos: pd.DataFrame, db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    muestras.to_sql("muestras_ghosh2022", con, if_exists="replace", index=False)
    compuestos.to_sql("compuestos", con, if_exists="append", index=False)
    con.close()


def save_to_parquet(abundancias: pd.DataFrame, parquet_path: str) -> None:
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    abundancias.to_parquet(parquet_path, index=False)


# ---------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    all_muestras, all_compuestos, all_abundancias = [], [], []

    for species_key, cfg in SPECIES_CONFIG.items():
        df = read_sheet(EXCEL_PATH, cfg["sheet_name"])
        muestras, compuestos, abundancias = build_species_tables(species_key, cfg, df)
        all_muestras.append(muestras)
        all_compuestos.append(compuestos)
        all_abundancias.append(abundancias)
        print(f"[{species_key}] muestras: {len(muestras)}, compuestos: {len(compuestos)}, "
              f"filas de abundancia: {len(abundancias)}")

    muestras = pd.concat(all_muestras, ignore_index=True)
    compuestos = pd.concat(all_compuestos, ignore_index=True)
    abundancias = pd.concat(all_abundancias, ignore_index=True)

    save_to_sqlite(muestras, compuestos, DB_PATH)
    save_to_parquet(abundancias, PARQUET_PATH)

    print()
    print(f"TOTAL muestras: {len(muestras)}")
    print(f"TOTAL compuestos: {len(compuestos)} (0 con CID por ahora - pendiente)")
    print(f"TOTAL filas de abundancia: {len(abundancias)}")


if __name__ == "__main__":
    main()

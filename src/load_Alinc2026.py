"""
Loader: Alınc et al. 2026 DOI 10.1002/ps.70436)
Tomate (S. lycopersicum) x T. harzianum T22 (biocontrol) x herbivoria de Halyomorpha halys (feeding + oviposition) - VOCs por HS-GC-MS

Estructura del Excel:
- Fila 4: encabezados (N, RT, RI, compound, FO 1-5, T22-FO 1-5, CTRL 1-5)
- Filas 5-27: 23 compuestos (21 identificados + 2 "Unknown")
- N = ID unico de compuesto tal como viene en el archivo original. 

Diseño biologico (es biocontrol x herbivoria):
- CTRL = plantas no inoculadas, no infestadas (control)
- FO   = plantas no inoculadas + alimentacion/oviposicion de H. halys
- T22-FO = plantas inoculadas con T. harzianum T22 + alimentacion/oviposicion de H. halys
- "FO" en el nombre del grupo NO es Fusarium oxysporum, es "Feeding + Oviposition"  del insecto

Nota sobre abundancias: el archivo usa "0" explicito para picos no detectados. Se preservan como 0.0, no como None, para no perder esa
distincion frente a valores realmente faltantes.

Nota sobre integracion a la DB: este dataset introduce dos ejes biologicos que los loaders anteriores no tenian (especie de herbivoro, agente de biocontrol
por separado del "patogeno"). Se guarda en una tabla separada (muestras_alinc2026) en vez de la tabla compartida "muestras", 
hasta que se defina el esquema unificado en la etapa de limpieza.
"""

import re
import sqlite3
from datetime import date
from pathlib import Path

import openpyxl
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

EXCEL_PATH = "data/raw/Alinc2026_dataset.xlsx"
SHEET_NAME = "VOC analysis"
HEADER_ROW = 4
FIRST_DATA_ROW = 5
LAST_DATA_ROW = 27

DB_PATH = "output/tfm_vocs.db"
PARQUET_DIR = "output/data/processed"
PARQUET_PATH = f"{PARQUET_DIR}/abundancias_alinc2026.parquet"

DATASET_ORIGIN = "Alinc2026"
DOI = "10.1002/ps.70436"
ZENODO_DOI = "10.5281/zenodo.20308380"
ANALYTICAL_TECHNIQUE = "HS-GC-MS (Porapak Q, 24h, Agilent 6890/MS5973)"
TISSUE = "whole plant (headspace, cylindrical glass chamber)"
SPECIES = "Solanum lycopersicum"
HERBIVORE = "Halyomorpha halys (feeding + oviposition)"
BIOCONTROL_AGENT = "Trichoderma harzianum T22"

# Grupo de tratamiento 
TREATMENT_MAP = {
    "CTRL": (None, None, "control"),
    "FO": (None, HERBIVORE, "herbivory_only"),
    "T22-FO": (BIOCONTROL_AGENT, HERBIVORE, "biocontrol_herbivory"),
}

# Mapeo compuesto -> PubChem CID, indexado por N (columna "N" del Excel, ID
# unico de compuesto). Se usa N y no el nombre porque "alpha-terpinene" esta repetido con dos RT/RI distintos.
PUBCHEM_CID_BY_N = {
    1: 17868,     
    2: 6654,     
    3: None,      
    4: 14896,     
    5: 521268,  
    6: 31253,     
    7: 78249, 
    8: 7460,  
    9: 7462,  
    10: 7463,  
    11: 22311, 
    12: 11142,
    13: 7461,  
    14: 11463,  
    15: None,      
    16: None,      
    17: 6918391,   
    18: 5281515,  
    19: 6432312,   
    20: None,      
    21: 5281520,  
    22: 5317570,   
    23: 1742210,   
}


# ---------------------------------------------------------------------------
# Paso 1: leer el Excel
# ---------------------------------------------------------------------------

def read_excel_raw(path: str, sheet: str) -> tuple[list[str], list[list]]:
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    header = [ws.cell(row=HEADER_ROW, column=c).value for c in range(1, ws.max_column + 1)]
    header = [h for h in header if h is not None]
    rows = []
    for r in range(FIRST_DATA_ROW, LAST_DATA_ROW + 1):
        values = [ws.cell(row=r, column=c).value for c in range(1, len(header) + 1)]
        rows.append(values)
    return header, rows


def parse_sample_columns(header: list[str]) -> pd.DataFrame:
    meta_cols = {"N", "RT", "RI", "compound"}
    parsed = []
    for col in header:
        if col in meta_cols:
            continue
        m = re.match(r"^(FO|T22-FO|CTRL) (\d+)$", col)
        if not m:
            raise ValueError(f"No se pudo parsear la columna de muestra: {col!r}")
        treatment_group, replicate = m.groups()
        parsed.append({
            "raw_column": col,
            "treatment_group": treatment_group,
            "biological_replicate": int(replicate),
        })
    return pd.DataFrame(parsed)


# ---------------------------------------------------------------------------
# Paso 2: construir tabla de muestras
# ---------------------------------------------------------------------------

def build_muestras(sample_meta: pd.DataFrame) -> pd.DataFrame:
    df = sample_meta.copy()

    def _map_treatment(group):
        fungal, herbivore, state = TREATMENT_MAP[group]
        return pd.Series({
            "fungal_inoculant": fungal,
            "herbivore_species": herbivore,
            "physiological_state": state,
        })

    df = pd.concat([df, df["treatment_group"].apply(_map_treatment)], axis=1)

    df["sample_id"] = (
        DATASET_ORIGIN.lower()
        + "_" + df["treatment_group"].str.lower().str.replace("-", "", regex=False)
        + "_r" + df["biological_replicate"].astype(str).str.zfill(2)
    )
    df["species"] = SPECIES
    df["genotype"] = None
    df["dataset_origin"] = DATASET_ORIGIN
    df["doi"] = DOI
    df["zenodo_doi"] = ZENODO_DOI
    df["analytical_technique"] = ANALYTICAL_TECHNIQUE
    df["tissue"] = TISSUE
    df["timepoint"] = None
    df["load_date"] = date.today().isoformat()
    df["validation_status"] = "pendiente_validacion"
    df["intended_use"] = "classifier"

    cols = [
        "sample_id", "species", "genotype", "physiological_state",
        "fungal_inoculant", "herbivore_species", "treatment_group",
        "biological_replicate", "dataset_origin", "doi", "zenodo_doi",
        "analytical_technique", "tissue", "timepoint",
        "load_date", "validation_status", "intended_use", "raw_column",
    ]
    return df[cols]


# ---------------------------------------------------------------------------
# Paso 3: construir catalogo de compuestos
# ---------------------------------------------------------------------------

def build_compuestos(header: list[str], rows: list[list]) -> pd.DataFrame:
    idx_N = header.index("N")
    idx_RT = header.index("RT")
    idx_RI = header.index("RI")
    idx_name = header.index("compound")

    out = []
    for row in rows:
        n = row[idx_N]
        name = row[idx_name]
        out.append({
            "compuesto_id": f"{DATASET_ORIGIN.lower()}_cmp{n:03d}",
            "n_original": n,
            "name_original": name,
            "retention_time": row[idx_RT],
            "retention_index": row[idx_RI],
            "cas": None,
            "identified": not str(name).lower().startswith("unknown"),
            "pubchem_cid": PUBCHEM_CID_BY_N.get(n),
            "dataset_origin": DATASET_ORIGIN,
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Paso 4: construir matriz de abundancias
# ---------------------------------------------------------------------------

def build_abundancias(header: list[str], rows: list[list], sample_meta: pd.DataFrame,
                       muestras: pd.DataFrame) -> pd.DataFrame:
    idx_N = header.index("N")
    col_to_sample_id = dict(zip(muestras["raw_column"], muestras["sample_id"]))
    sample_cols = sample_meta["raw_column"].tolist()

    long_rows = []
    for row in rows:
        n = row[idx_N]
        compuesto_id = f"{DATASET_ORIGIN.lower()}_cmp{n:03d}"
        for col in sample_cols:
            col_idx = header.index(col)
            valor = row[col_idx]
            abundancia = float(valor) if valor is not None else 0.0
            long_rows.append({
                "sample_id": col_to_sample_id[col],
                "compuesto_id": compuesto_id,
                "abundancia": abundancia,
                "unidad": "area_de_pico",
            })
    return pd.DataFrame(long_rows)


# ---------------------------------------------------------------------------
# Paso 5: pguardar en SQLite + Parquet
# ---------------------------------------------------------------------------

def save_to_sqlite(muestras: pd.DataFrame, compuestos: pd.DataFrame, db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    muestras.drop(columns=["raw_column"]).to_sql(
        "muestras_alinc2026", con, if_exists="replace", index=False
    )
    compuestos.drop(columns=["n_original", "retention_time", "retention_index"]).to_sql(
        "compuestos", con, if_exists="append", index=False
    )
    con.close()


def save_to_parquet(abundancias: pd.DataFrame, parquet_path: str) -> None:
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    abundancias.to_parquet(parquet_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    header, rows = read_excel_raw(EXCEL_PATH, SHEET_NAME)
    sample_meta = parse_sample_columns(header)

    muestras = build_muestras(sample_meta)
    compuestos = build_compuestos(header, rows)
    abundancias = build_abundancias(header, rows, sample_meta, muestras)

    save_to_sqlite(muestras, compuestos, DB_PATH)
    save_to_parquet(abundancias, PARQUET_PATH)

    print(f"Muestras cargadas: {len(muestras)}")
    print(f"Compuestos cargados: {len(compuestos)} ({compuestos['identified'].sum()} identificados)")
    print(f"Compuestos con CID: {compuestos['pubchem_cid'].notna().sum()} / {len(compuestos)}")
    print(f"Filas de abundancia (formato largo): {len(abundancias)}")


if __name__ == "__main__":
    main()

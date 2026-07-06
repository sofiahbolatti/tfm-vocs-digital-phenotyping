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

EXCEL_PATH = "data/raw/Ghosh/rawdata_ghosh.xlsx"

DB_PATH = "db/tfm_vocs.db"
PARQUET_DIR = "db/data/processed"
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

HERBIVORE = "Bemisia tabaci (whitefly)"

# Mapeo CAS -> PubChem CID 

CAS_TO_PUBCHEM_CID = {
    "100-51-6": 244, "100-52-7": 240, "10030-74-7": 14258, "1004-29-1": 13861,
    "104-51-8": 7705, "104-53-0": 5054, "104-76-7": 7720, "106-70-7": 7824,
    "108-38-3": 7929, "110-42-9": 8050, "110-62-3": 8063, "110-93-0": 9862,
    "111-11-5": 8091, "111-27-3": 8103, "111-70-6": 8129, "111-82-0": 8139,
    "111-87-5": 957, "112-31-2": 8175, "112-39-0": 8181, "112-61-8": 8201,
    "112-63-0": 8203, "1120-25-8": 14258, "118-93-4": 8375, "119-36-8": 4133,
    "122-78-1": 998, "123-35-3": 31253, "124-06-1": 31283, "124-10-7": 31284,
    "124-18-5": 15600, "127-41-3": 24680, "13019-16-4": 25610, "13419-69-7": 14486,
    "13466-78-9": 26049, "13877-93-5": 26318, "142-29-0": 8882, "142-62-1": 8892,
    "142-83-6": 8901, "14309-57-0": 26630, "1438-59-1": 5984, "149-57-5": 379,
    "14901-07-6": 26955, "15356-74-8": 27209, "1576-87-0": 12993, "1576-95-0": 15306,
    "1599-47-9": 74137, "16635-54-4": 10460, "17066-67-0": 28237, "1731-84-6": 15606,
    "17427-21-3": 28536, "18368-95-1": 176983, "18409-17-1": 29060, "18829-55-5": 17167,
    "1931-63-1": 74732, "19549-87-2": 123385, "19888-34-7": 524129, "2000215-97-9": 19725,
    "20013-73-4": 88332, "20307-84-0": 89316, "2213-23-2": 16656, "23229-68-7": 580133,
    "23267-57-4": 67471, "2363-89-5": 16900, "2371-19-9": 25611, "2396-77-2": 61310,
    "247-43-4": 16997, "2471-84-3": 75581, "2492-43-5": 3014106, "25152-83-4": 16899,
    "26444-18-8": 81886, "2765-11-9": 17697, "2867-05-2": 17868, "30086-02-3": 181575,
    "301-00-8": 9316, "3208-16-0": 18554, "3391-86-4": 18827, "34246-54-3": 118623,
    "34246-57-6": 520680, "3777-69-3": 19602, "3796-70-1": 19633, "38049-04-6": 642875,
    "3884-92-2": 535040, "39029-41-9": 15094, "40716-66-3": 8888, "41624-92-4": 4586585,
    "4170-30-3": 20138, "4173-41-5": 181575, "4312-99-6": 61346, "4313-02-4": 11788274,
    "4313-03-5": 20307, "432-25-7": 9895, "470-82-6": 2758, "472-66-2": 61124,
    "48376-1": 10223, "488-10-8": 10261, "497-23-4": 10341, "500-02-7": 92780,
    "502-69-2": 10408, "515-13-9": 10583, "51911-82-1": 103555, "53448-07-0": 62447,
    "555-10-2": 11142, "557-48-2": 11196, "56134-03-3": 92017, "56554-30-4": 556196,
    "5689-23-6": 6430898, "57266-86-1": 17167, "5779-93-1": 34224, "586-62-9": 11463,
    "586-67-4": 521851, "589-43-5": 11511, "589-53-7": 11512, "60-12-8": 6054,
    "616-25-1": 12020, "67133-86-2": 565262, "6728-26-3": 10460, "6728-31-0": 71590,
    "673-84-7": 12658, "6753-98-6": 23204, "6789-80-6": 23234, "6813-21-4": 522296,
    "70786-44-6": 586292, "71-41-0": 6276, "7132-64-1": 23518, "75-98-9": 6417,
    "78-70-6": 6549, "79-20-9": 6584, "80-56-8": 6654, "816-19-3": 69872,
    "87-44-5": 6887, "90-05-1": 460, "91-16-7": 7043, "93-58-3": 7150,
    "95910-36-4": 530426, "97-53-0": 3314, "99-85-4": 7461, "99-86-5": 7462,
    "99-87-6": 7463, "112-05-0": 8158, "89-81-6": 6987, "108-95-2": 996,
    "4630-07-3": 160748,
}
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

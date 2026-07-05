"""
Loader: Ayelo et al. 2026 ( DOI 10.1002/ps.70789)
Algodon (Gossypium hirsutum) x consorcios PGPR (AU8, AU9a, TX1, TX2a, TX3) vs
control, cruzado con 2 cultivares (RC=resistente, SC=susceptible) - VOCs foliares por GC-MS

Estructura del CSV:
- Filas 1-3: leyenda (SC/RC) y linea en blanco
- Fila 4: nombres de compuestos (columnas VOC1-VOC13). 
- Fila 5: encabezados reales (Cultivar, Treatment, VOC1..VOC13, TOTAL)
- Filas 6-65: 60 muestras individuales (2 cultivos x 6 tratamientos x 5 replicas)

Diseno biologico:
- Cultivar: RC (resistente) / SC (susceptible)
- Treatment: Ctrl o uno de 5 consorcios PGPR (AU8, AU9a, TX1, TX2a, TX3)
"""

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

CSV_PATH = "data/raw/Ayelo/rawdata_ayelo.csv"

DB_PATH = "db/tfm_vocs.db"
PARQUET_DIR = "db/data/processed"
PARQUET_PATH = f"{PARQUET_DIR}/abundancias_ayelo2026.parquet"

DATASET_ORIGIN = "Ayelo2026"
DOI = "10.1002/ps.70789"
ANALYTICAL_TECHNIQUE = "GC-MS (foliar headspace)"
TISSUE = "leaves"
SPECIES = "Gossypium hirsutum"

HEADER_ROW = 4

CULTIVAR_MAP = {
    "RC": "resistant",
    "SC": "susceptible",
}

# Columna VOC  -> PubChem CID

VOC_INFO = {
    "VOC1": ("alpha-Pinene", 6654),
    "VOC2": ("Benzaldehyde", 240),
    "VOC3": ("beta-Pinene", 14896),
    "VOC4": ("beta-Myrcene", 31253),
    "VOC5": ("(Z)-3-Hexenyl acetate", 5363388),
    "VOC6": ("D-Limonene", 440917),
    "VOC7": ("beta-Ocimene", 18756),
    "VOC8": ("Nonanal", 31289),
    "VOC9": ("DMNT (4,8-dimethylnona-1,3,7-triene)", 6427110),
    "VOC10": ("Decanal", 8175),
    "VOC11": ("beta-Caryophyllene", 5281515),
    "VOC12": ("alpha-Humulene", 5281520),
    "VOC13": ("alpha-Farnesene", 5281516),
}
VOC_COLUMNS = list(VOC_INFO.keys())


# ---------------------------------------------------------------------------
# Paso 1: leer el csv
# ---------------------------------------------------------------------------

def read_csv_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=HEADER_ROW)
    missing = [c for c in VOC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas VOC esperadas y no encontradas: {missing}")
    df = df[df["Cultivar"].isin(CULTIVAR_MAP.keys())].reset_index(drop=True)
    return df


  
# ---------------------------------------------------------------------------
# Paso 2: construir tabla de muestras
# ---------------------------------------------------------------------------

def build_muestras(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    counters = {}  
    for _, r in df.iterrows():
        cultivar_code = r["Cultivar"]
        treatment = r["Treatment"]
        key = (cultivar_code, treatment)
        counters[key] = counters.get(key, 0) + 1
        replicate = counters[key]

        pgpr_consortium = None if treatment == "Ctrl" else treatment

        sample_id = (
            DATASET_ORIGIN.lower()
            + "_" + cultivar_code.lower()
            + "_" + treatment.lower()
            + "_r" + str(replicate).zfill(2)
        )
        rows.append({
            "sample_id": sample_id,
            "species": SPECIES,
            "cultivar_code": cultivar_code,
            "cultivar_resistance": CULTIVAR_MAP[cultivar_code],
            "pgpr_consortium": pgpr_consortium,
            "physiological_state": "control" if treatment == "Ctrl" else "pgpr_treated",
            "treatment_group": treatment,
            "biological_replicate": replicate,
            "voc_total_reported": r["TOTAL"],
            "dataset_origin": DATASET_ORIGIN,
            "doi": DOI,
            "analytical_technique": ANALYTICAL_TECHNIQUE,
            "tissue": TISSUE,
            "load_date": date.today().isoformat(),
            "validation_status": "pendiente_validacion",
            "intended_use": "classifier",
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Paso 3: construir catalogo de compuestos
# -----------------------------------------------------------------------------

def build_compuestos() -> pd.DataFrame:
    rows = []
    for i, voc_col in enumerate(VOC_COLUMNS):
        name, cid = VOC_INFO[voc_col]
        rows.append({
            "compuesto_id": f"{DATASET_ORIGIN.lower()}_cmp{i+1:03d}",
            "name_original": name,
            "voc_column": voc_col,
            "cas": None,
            "identified": True,
            "pubchem_cid": cid,
            "dataset_origin": DATASET_ORIGIN,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paso 4: construir matriz de abundancias 
# ---------------------------------------------------------------------------

def build_abundancias(df: pd.DataFrame, muestras: pd.DataFrame) -> pd.DataFrame:
    compound_to_id = {
        voc_col: f"{DATASET_ORIGIN.lower()}_cmp{i+1:03d}"
        for i, voc_col in enumerate(VOC_COLUMNS)
    }
    sample_ids = muestras["sample_id"].tolist()

    long_rows = []
    for idx, (_, r) in enumerate(df.iterrows()):
        sample_id = sample_ids[idx]
        for voc_col in VOC_COLUMNS:
            long_rows.append({
                "sample_id": sample_id,
                "compuesto_id": compound_to_id[voc_col],
                "abundancia": float(r[voc_col]),
                "unidad": "unidad_relativa_reportada_en_paper", 
            })
    return pd.DataFrame(long_rows)


# ------------------------------------------------------------------------------
# Paso 5: guardasr en SQLite + Parquet
# ---------------------------------------------------------------------------

def save_to_sqlite(muestras: pd.DataFrame, compuestos: pd.DataFrame, db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    muestras.to_sql("muestras_ayelo2026", con, if_exists="replace", index=False)
    compuestos.drop(columns=["voc_column"]).to_sql(
        "compuestos", con, if_exists="append", index=False
    )
    con.close()


def save_to_parquet(abundancias: pd.DataFrame, parquet_path: str) -> None:
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    abundancias.to_parquet(parquet_path, index=False)


# ---------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------

def main():
    df = read_csv_raw(CSV_PATH)

    muestras = build_muestras(df)
    compuestos = build_compuestos()
    abundancias = build_abundancias(df, muestras)

    save_to_sqlite(muestras, compuestos, DB_PATH)
    save_to_parquet(abundancias, PARQUET_PATH)

    print(f"Muestras cargadas: {len(muestras)}")
    print(f"Compuestos cargados: {len(compuestos)} ({compuestos['pubchem_cid'].notna().sum()} con CID)")
    print(f"Filas de abundancia (formato largo): {len(abundancias)}")


if __name__ == "__main__":
    main()

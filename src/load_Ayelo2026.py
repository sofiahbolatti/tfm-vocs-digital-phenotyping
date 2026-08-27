"""
Loader: Ayelo et al. 2026 ( DOI 10.1002/ps.70789)
Algodon (Gossypium hirsutum) x consorcios PGPR (AU8, AU9a, TX1, TX2a, TX3) vs control, cruzado con 2 cultivares (RC=resistente, SC=susceptible) - VOCs foliares por GC-MS

Estructura del CSV:
- Filas 1-3: leyenda (SC/RC) y linea en blanco
- Fila 4: nombres de compuestos  
- Fila 5: encabezados reales 
- Filas 6-65: 60 muestras individuales (2 cultivos x 6 tratamientos x 5 replicas)

Diseño biologico:
- Cultivar: RC (resistente) / SC (susceptible)
- Treatment: Ctrl o uno de 5 consorcios PGPR
"""

import sqlite3
from datetime import date
from pathlib import Path
import pandas as pd

# Configuracion

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


# Paso 1: leer el csv

def read_csv_raw(path: str) -> pd.DataFrame:
    """
    Lee el CSV o y se queda solo con las filas de cultivares validos
    Parameters:
    path : str
    Ruta al CSV
    Precondition:
    El CSV debe tener las columnas VOC1 a VOC13 ; si falta alguna, pone ValueError
    Returns:
    pandas.DataFrame
    Filas del CSV original, filtradas a las que tienen cultivar en CULTIVAR_MAP
    """
    df = pd.read_csv(path, header=HEADER_ROW)
    missing = [c for c in VOC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas VOC esperadas y no encontradas: {missing}")
    df = df[df["Cultivar"].isin(CULTIVAR_MAP.keys())].reset_index(drop=True)
    return df

# Paso 2: construir tabla de muestras

def build_muestras(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arma la tabla de metadatos de muestras a partir del CSV filtrado
    Arma el sample_id combinando cultivo y tratamiento, cuenta las replicas biologicas por combinacion, 
    y traduce el tratamiento a pgpr_consortium (None si es "Ctrl") y a physiological_state (control o pgpr_treated)
    Parameters:
    df : pandas.DataFrame
    Salida de read_csv_raw
    Returns:
    pandas.DataFrame
    Una fila por muestra (60 en total), con sample_id, cultivar, consorcio PGPR y metadata fija del dataset
    """
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

# Paso 3: construir catalogo de compuestos

def build_compuestos() -> pd.DataFrame:
    """
    Arma el catalogo de compuestos a partir de VOC_INFO
    Returns:
    pandas.DataFrame
    Una fila por compuesto (los 13 VOCs, identificados con PubChem CID), con compuesto_id, name_original, voc_column y pubchem_cid
    """
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

# Paso 4: construir matriz de abundancias 

def build_abundancias(df: pd.DataFrame, muestras: pd.DataFrame) -> pd.DataFrame:
    """
    Arma la matriz de abundancias en formato largo
    Parameters:
    df : pandas.DataFrame
    Salida de read_csv_raw
    muestras : pandas.DataFrame
    Salida de build_muestras (debe estar en el mismo orden que df, ya que se mapean por posicion)
    Returns:
    pandas.DataFrame
    Columnas sample_id, compuesto_id, abundancia y unidad;780 filas
    """
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


# Paso 5: guardasr en SQLite + Parquet

def save_to_sqlite(muestras: pd.DataFrame, compuestos: pd.DataFrame, db_path: str) -> None:
    """
    Guarda las tablas de muestras y compuestos en la base SQLite
    Las muestras de Ayelo2026 se guardan en su propia tabla muestras_ayelo2026 (no en muestras), porque tienen columnas propias (cultivar_resistance, pgpr_consortium, etc.) 
    que no encajan en el esquema comun. Despues hay que correr src/01_unify_schema.py para fusionarlas en muestras
    Parameters:
    muestras : pandas.DataFrame
    Salida de build_muestras
    compuestos : pandas.DataFrame
    Salida de build_compuestos
    db_path : str
    Ruta al archivo SQLite
    Returns:
    None
    Reemplaza (if_exists="replace") la tabla muestras_ayelo2026, y agrega (append) a la tabla compuestos comun
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    muestras.to_sql("muestras_ayelo2026", con, if_exists="replace", index=False)
    compuestos.drop(columns=["voc_column"]).to_sql(
        "compuestos", con, if_exists="append", index=False
    )
    con.close()


def save_to_parquet(abundancias: pd.DataFrame, parquet_path: str) -> None:
    """
    Guarda la matriz de abundancias en formato parquet
    Parameters:
    abundancias : pandas.DataFrame
    Salida de build_abundancias
    parquet_path : str
    Ruta de destino del archivo parquet
    Returns:
    None
    Crea la carpeta de destino si no existe
    """
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    abundancias.to_parquet(parquet_path, index=False)


# Main

def main():
    """
    Lee el CSV, arma las tres tablas y las persiste en SQLite (muestras_ayelo2026) y parquet
    Returns:
    None
   Imprime un resumen de cuantas muestras y compuestos se cargaron, y cuantas filas de abundancia
    """
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

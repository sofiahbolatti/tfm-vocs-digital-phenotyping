"""
Loader: Moreira et al. 2024 (DOI 10.1111/1365-2745.14242)
Colza (Brassica rapa) x Sclerotinia sclerotiorum / Xanthomonas campestris pv.campestris / Mamestra brassicae / Brevicoryne brassicae vs control - VOCsflorales por bomba de aire + GC-MS

Estructura del CSV:
- Columnas: ID, Genotype, Treatment, height, [15 VOCs], total
- 60 filas = 60 muestras individuales
- Unidad de abundancia: nanogramos por hora (ng/h)
- Sin valores faltantes en los VOCs 

Diseno biologico (5 clases):
- control      = sin dano foliar (n=12)
- sclerotinia  = hongo Sclerotinia sclerotiorum (n=11)
- xanthomonas  = bacteria Xanthomonas campestris pv. campestris (n=14)
- mamestra     = oruga Mamestra brassicae, herbivoria (n=12)
- brevicoryne  = pulgon Brevicoryne brassicae, herbivoria (n=11)

El dataset mezcla patogenos y herbivoros en un solo eje de tratamiento. Se separan en dos columnas (microorganism / herbivore_species) para no perder esa distincion biologica.
"""

import sqlite3
from datetime import date
from pathlib import Path
import pandas as pd


# Configuracion


CSV_PATH = "data/raw/Moreira/rawdata_moreira.csv"

DB_PATH = "db/tfm_vocs.db"
PARQUET_DIR = "db/data/processed"
PARQUET_PATH = f"{PARQUET_DIR}/abundancias_moreira2024.parquet"

DATASET_ORIGIN = "Moreira2024"
DOI = "10.1111/1365-2745.14242"
DRYAD_DOI = "10.5061/dryad.rbnzs7hjh"
ANALYTICAL_TECHNIQUE = "air entrainment (bomba de aire) + GC-MS"
TISSUE = "flowers (floral VOCs)"
SPECIES = "Brassica rapa"

# Tratamientosd
TREATMENT_MAP = {
    "control": (None, None, "control"),
    "sclerotinia": ("Sclerotinia sclerotiorum", None, "infected_fungal"),
    "xanthomonas": ("Xanthomonas campestris pv. campestris", None, "infected_bacterial"),
    "mamestra": (None, "Mamestra brassicae", "herbivory_caterpillar"),
    "brevicoryne": (None, "Brevicoryne brassicae", "herbivory_aphid"),
}

# Mapeo manual compuesto identificado: PubChem CID 
PUBCHEM_CID_BY_NAME = {
    "Benzaldehyde": 240,
    "5-Hepten-2-one, 6-methyl-": 9862,          
    "trans-a-bergamotene": 6429302,
    "2-Hexen-1-ol, acetate, (Z)-": 5363374,     
    "Limonene": 22311,
    "Linalool": 6549,
    "Nonanal": 31289,
    "(E)-b-farnesene": 10407,
    "2-butyl-1-octanol": 19800,
    "Tridecane": 12388,
    "Decanal": 8175,
    "Farnesan": 19773,                          
    "Tetradecane": 12389,                         
    "Pentadecane": 12391,
    "a-farnesene": 5281516,                      
}

VOC_COLUMNS = list(PUBCHEM_CID_BY_NAME.keys())


# Paso 1: leer el csv

def read_csv_raw(path: str) -> pd.DataFrame:
    """
    Lee el CSV crudo y valida que esten todas las columnas de VOC esperadas
    Parameters:
    path : str
    Ruta al CSV 
    Precondition:
    El CSV debe tener las 15 columnas VOC listadas en PUBCHEM_CID_BY_NAME; si falta alguna, ValueError
    Returns:
    pandas.DataFrame
    Las 60 filas del CSV
    """
    df = pd.read_csv(path, sep=";")
    missing = [c for c in VOC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas de VOC esperadas y no encontradas: {missing}")
    return df


# Paso 2: construir tabla de muestras

def build_muestras(df: pd.DataFrame) -> pd.DataFrame:
    """
    Arma la tabla de metadatos de muestras, separando el tratamiento en microorganismo/herbivoro/estado fisiologico via TREATMENT_MAP
    El dataset mezcla patogenos y herbivoros en una sola columna Treatment; esta funcion los separa en dos columnas (microorganism, herbivore_species) para no perder esa distincion biologica
    Parameters:
    df : pandas.DataFrame
    Salida de read_csv_raw
    Returns:
    pandas.DataFrame
    Una fila por muestra (60 en total), con sample_id, especie, tratamiento separado en microorganismo/herbivoro/estado, y metadata fija del dataset, mas raw_id (por trazabilidad)
    """
    rows = []
    for _, r in df.iterrows():
        treatment = r["Treatment"]
        microorganism, herbivore, physiological_state = TREATMENT_MAP[treatment]
        sample_id = (
            DATASET_ORIGIN.lower()
            + "_" + treatment
            + "_" + str(r["Genotype"])
            + "_id" + str(int(r["ID"])).zfill(2)
        )
        rows.append({
            "sample_id": sample_id,
            "species": SPECIES,
            "genotype": str(r["Genotype"]),
            "physiological_state": physiological_state,
            "microorganism": microorganism,
            "herbivore_species": herbivore,
            "treatment_group": treatment,
            "height_cm": r["height"],
            "voc_total_reported": r["total"],
            "dataset_origin": DATASET_ORIGIN,
            "doi": DOI,
            "dryad_doi": DRYAD_DOI,
            "analytical_technique": ANALYTICAL_TECHNIQUE,
            "tissue": TISSUE,
            "load_date": date.today().isoformat(),
            "validation_status": "pendiente_validacion",
            "intended_use": "classifier",
            "raw_id": int(r["ID"]),
        })
    return pd.DataFrame(rows)


  
# Paso 3: construir catalogo de compuestos

def build_compuestos() -> pd.DataFrame:
    """
    Arma el catalogo de compuestos a partir de PUBCHEM_CID_BY_NAME
    Returns:
    pandas.DataFrame
    Una fila por compuesto (15 en totall), con compuesto_id, name_original y pubchem_cid
    """
    rows = []
    for i, name in enumerate(VOC_COLUMNS):
        rows.append({
            "compuesto_id": f"{DATASET_ORIGIN.lower()}_cmp{i+1:03d}",
            "name_original": name,
            "cas": None,
            "identified": True,
            "pubchem_cid": PUBCHEM_CID_BY_NAME[name],
            "dataset_origin": DATASET_ORIGIN,
        })
    return pd.DataFrame(rows)


# Paso 4: construir matriz de abundancias 

def build_abundancias(df: pd.DataFrame, muestras: pd.DataFrame) -> pd.DataFrame:
    """
    Arma la matriz de abundancias en formato largo, mapeando cada fila del CSV a su sample_id via el ID original
    Parameters:
    df : pandas.DataFrame
    Salida de read_csv_raw
    muestras : pandas.DataFrame
    Salida de build_muestras
    Returns:
    pandas.DataFrame
    Columnas sample_id, compuesto_id, abundancia y unidad
    """
    id_to_sample_id = dict(zip(muestras["raw_id"], muestras["sample_id"]))
    compound_to_id = {
        name: f"{DATASET_ORIGIN.lower()}_cmp{i+1:03d}"
        for i, name in enumerate(VOC_COLUMNS)
    }

    long_rows = []
    for _, r in df.iterrows():
        sample_id = id_to_sample_id[int(r["ID"])]
        for name in VOC_COLUMNS:
            long_rows.append({
                "sample_id": sample_id,
                "compuesto_id": compound_to_id[name],
                "abundancia": float(r[name]),
                "unidad": "ng_per_hour",
            })
    return pd.DataFrame(long_rows)


# Paso 5: guardar en SQLite + Parquet

def save_to_sqlite(muestras: pd.DataFrame, compuestos: pd.DataFrame, db_path: str) -> None:
    """
    Guarda las tablas de muestras y compuestos en la base SQLite
    Se guarda en su propia tabla muestras_moreira2024 (no en muestras), porque mezcla columnas que no encajan en el esquema comun 
    (herbivore_species, treatment_group, height_cm, etc.). Despues hay que correr src/01_unify_schema.py para fusionarla en muestras
    Parameters:
    muestras : pandas.DataFrame
    Salida de build_muestras
    compuestos : pandas.DataFrame
    Salida de build_compuestos
    db_path : str
    Ruta al archivo SQLite
    Returns:
    None
    Descarta la columna raw_id antes de guardar; reemplaza (if_exists="replace") la tabla muestras_moreira2024, y agrega (append) a la tabla `compuestos` comun
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    muestras.drop(columns=["raw_id"]).to_sql(
        "muestras_moreira2024", con, if_exists="replace", index=False
    )
    compuestos.to_sql("compuestos", con, if_exists="append", index=False)
    con.close()


def save_to_parquet(abundancias: pd.DataFrame, parquet_path: str) -> None:
    """
    Guarda la matriz de abundancias en formato parquet
    Parameters:
    abundancias : pandas.DataFrame
    Salida de build_abundancias
    parquet_path : str
    Ruta de destino del archivoparquet
    Returns:
    None
    Crea la carpeta de destino si no existe
    """
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    abundancias.to_parquet(parquet_path, index=False)


# Main

def main():
    """
    Lee el CSV, arma las tres tablas y las persiste en SQLite (tabla propia muestras_moreira2024) y parquet
    Returns:
    None
    Imprime un resumen de cuantas muestras y compuestos se cargaron, y cuantas filas de abundancia se generaron
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

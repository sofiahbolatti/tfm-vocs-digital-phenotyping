"""
Loader: Lazazzara et al. 2018 (DOI 10.1038/s41598-018-19776-2)
Grapevine x Plasmopara viticola - VOCs por HS-SPME-GC-MS

Fuente original: Excel "Metabolites_for_Christoph" recibido de Michele Perazzolli / Valentina Lazazzara.

Estructura del Excel (hoja unica):
- Filas 2-53: compuestos (52 en total: 36 identificados con CAS + 16 picos sin identificar, etiquetados "No match")
- Columnas A-I: metadatos del compuesto (nombre, CAS, fuente, score, iones de cuantificacion, RI promedio, RT promedio, S/N promedio, hits)
- Columnas J en adelante (98 columnas): una por muestra individual, con abundancia (area de pico) para ese compuesto en esa muestra
- Nombre de columna de muestra: "{experimento}_{genotipo} {timepoint}_{replica}.cmp"

Mapeo biologico:
- 0dpi = control (antes de inoculacion), 6dpi = infectado con P. viticola (6 dias post-inoculacion)
- Genotipos: Pinot Noir (V. vinifera, susceptible), BC4 (hibrido M. rotundifolia x V. vinifera, resistente), Kober 5BB / K5BB y SO4 (hibridos V. berlandieri x
  V. riparia, resistentes), Solaris (cultivar moderno, resistente)
- experimento 1 y 2 = dos repeticiones del experimento en invernadero 
"""

import re
import sqlite3
from datetime import date
from pathlib import Path

import openpyxl
import pandas as pd


# Configuracion


EXCEL_PATH = "data/raw/Lazazzara/rawdata_lazazzara.xlsx"
SHEET_NAME = "metabolites both exp."

DB_PATH = "db/tfm_vocs.db"
PARQUET_DIR = "db/data/processed"
PARQUET_PATH = f"{PARQUET_DIR}/abundancias_lazazzara2018.parquet"

DATASET_ORIGIN = "Lazazzara2018"
DOI = "10.1038/s41598-018-19776-2"
ANALYTICAL_TECHNIQUE = "HS-SPME-GC-MS"
TISSUE = "leaves"
SPECIES_BASE = "Vitis vinifera / hibridos interespecificos"

GENOTYPE_NOTES = {
    "Pinot Noir": "Vitis vinifera cv. Pinot Noir - susceptible a P. viticola",
    "BC4": "Hibrido Muscadinia rotundifolia x Vitis vinifera - resistente",
    "K5BB": "Kober 5BB, hibrido Vitis berlandieri x Vitis riparia - resistente",
    "SO4": "Hibrido Vitis berlandieri x Vitis riparia - resistente",
    "Solaris": "Cultivar moderno de origen hibrido - resistente",
}

SAMPLE_COL_PATTERN = re.compile(r"^(\d+)_(.+) (\d+)dpi_(\d+)\.cmp$")
DPI_TO_STATE = {0: "control", 6: "infected"}

# Mapeo manual compuesto identificado: PubChem CID (buscado por CAS). Los compuestos "No match" (picos sin identificar) y los que carecen de CAS  quedan sin CID porque no hay nada que buscar
PUBCHEM_CID_BY_INDEX = {
    1: 637566, 2: 5364752, 3: 521334, 4: 92313, 5: 441005, 6: 442393,
    7: 31253, 8: 6549, 9: 638014, 10: 9895, 11: 5281515, 12: 11463,
    13: 12306047, 14: 92762, 15: 23204, 16: 31291, 17: 31289, 18: 92812,
    19: 6428573, 20: 6184, 21: 20861, 23: 6782, 24: 6432173, 25: 8175,
    26: 244, 27: 6054, 28: 998, 29: 549664, 30: 637564, 31: 5283321,
    32: 8723, 33: 19602, 34: 5364920, 35: 18554, 36: 957,
}


# Paso 1: leer el Excel

def read_excel_raw(path: str, sheet: str) -> pd.DataFrame:
    """
    Lee la hoja del excel. Salta las filas vacias o de metadata del instrumento que quedan al final del export
    Parameters:
    path : str
    Ruta al archivo
    sheet : str
    Nombre de la hoja
    Returns:
    pandas.DataFrame
    Una fila por compuesto, con las columnas de metadata y una columna por muestra
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    header = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
    rows = []
    for r in range(2, ws.max_row + 1):
        values = [ws.cell(row=r, column=c).value for c in range(1, ws.max_column + 1)]
        # Saltar filas vacias o de metadata de instrumento (footer del export)
        if values[0] is None:
            continue
        rows.append(values)
    df = pd.DataFrame(rows, columns=header)
    return df


def parse_sample_columns(columns: list[str]) -> pd.DataFrame:
    """
    Parsea los nombres de columna de muestra a sus componentes biologicos, con el formato "{experimento}_{genotipo} {dpi}dpi_{replica}.cmp" 
    Parameters:
    columns : list[str]
    Nombres de columna de muestra
    Precondition:
    Cada nombre debe matchear SAMPLE_COL_PATTERN, si no, ValueError
    Returns:
    pandas.DataFrame
    Una fila por muestra, con columnas raw_column, experiment_replicate, genotype, dpi y biological_replicate
    """
    parsed = []
    for col in columns:
        m = SAMPLE_COL_PATTERN.match(col)
        if not m:
            raise ValueError(f"No se pudo parsear el nombre de columna: {col!r}")
        experiment, genotype, dpi, replicate = m.groups()
        parsed.append(
            {
                "raw_column": col,
                "experiment_replicate": int(experiment),
                "genotype": genotype,
                "dpi": int(dpi),
                "biological_replicate": int(replicate),
            }
        )
    return pd.DataFrame(parsed)


# Paso 2: construir tabla de muestras (metadatos)

def build_muestras(sample_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Arma la tabla de metadatos de muestras a partir de los componentes ya parseados
    Traduce dpi a physiological_state via DPI_TO_STATE, arma el sample_id, y completa las columnas del dataset
    Parameters:
    sample_meta : pandas.DataFrame
    Salida de parse_sample_columns
    Returns:
    pandas.DataFrame
    Una fila por muestra, con las columnas del esquema estandar de muestras mas raw_column 
    """
    df = sample_meta.copy()
    df["physiological_state"] = df["dpi"].map(DPI_TO_STATE)
    df["timepoint"] = df["dpi"].astype(str) + "dpi"
    df["sample_id"] = (
        DATASET_ORIGIN.lower()
        + "_exp"
        + df["experiment_replicate"].astype(str)
        + "_"
        + df["genotype"].str.replace(" ", "", regex=False).str.lower()
        + "_"
        + df["timepoint"]
        + "_r"
        + df["biological_replicate"].astype(str).str.zfill(2)
    )
    df["species"] = SPECIES_BASE
    df["genotype_notes"] = df["genotype"].map(GENOTYPE_NOTES)
    df["microorganism"] = "Plasmopara viticola"
    df["strain"] = None
    df["dataset_origin"] = DATASET_ORIGIN
    df["doi"] = DOI
    df["analytical_technique"] = ANALYTICAL_TECHNIQUE
    df["tissue"] = TISSUE
    df["load_date"] = date.today().isoformat()
    df["validation_status"] = "pendiente_validacion"
    df["intended_use"] = "classifier"

    cols = [
        "sample_id",
        "species",
        "genotype",
        "genotype_notes",
        "microorganism",
        "strain",
        "physiological_state",
        "dataset_origin",
        "doi",
        "analytical_technique",
        "timepoint",
        "tissue",
        "experiment_replicate",
        "biological_replicate",
        "load_date",
        "validation_status",
        "intended_use",
        "raw_column",
    ]
    return df[cols]

# Paso 3: construir catalogo de compuestos

def build_compuestos(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Arma el catalogo de compuestos a partir de las columnas de metadat
    Marca como identificado todo compuesto en el que el nombre no empiece con "No match", y le asigna el PubChem CID  de PUBCHEM_CID_BY_INDEX cuando existe
    Parameters:
    raw_df : pandas.DataFrame
    Salida de read_excel_raw
    Returns:
    pandas.DataFrame
    Una fila por compuesto, con compuesto_id, name_original, cas, identified, pubchem_cid y el resto de metadata
    """
    meta_cols = [
        "Metabolite",
        "CAS",
        "Source",
        "Score",
        "Quantification Ions",
        "Avg. RI",
        "Avg. RT (Min)",
        "Avg.S/N",
        "Hits",
    ]
    df = raw_df[meta_cols].copy().reset_index(drop=True)
    df.insert(0, "compuesto_id", [f"{DATASET_ORIGIN.lower()}_cmp{i+1:03d}" for i in df.index])
    df["identified"] = df["Metabolite"].apply(lambda x: not str(x).startswith("No match"))
    df["pubchem_cid"] = [PUBCHEM_CID_BY_INDEX.get(i + 1) for i in df.index]
    df = df.rename(
        columns={
            "Metabolite": "name_original",
            "CAS": "cas",
            "Source": "source_reference",
            "Score": "id_score",
            "Quantification Ions": "quantification_ions",
            "Avg. RI": "avg_retention_index",
            "Avg. RT (Min)": "avg_retention_time_min",
            "Avg.S/N": "avg_signal_noise",
            "Hits": "hits",
        }
    )
    return df


# Paso 4: construir matriz de abundancias en formato largo

def build_abundancias(raw_df: pd.DataFrame, sample_meta: pd.DataFrame, compuestos: pd.DataFrame) -> pd.DataFrame:
    """
    Arma la matriz de abundancias en formato largo  y vuelve a llamar a build_muestras  para obtener el mapeo raw_column a sample_id
    Parameters:
    raw_df : pandas.DataFrame
    Salida de read_excel_raw
    sample_meta : pandas.DataFrame
    Salida de parse_sample_columns
    compuestos : pandas.DataFrame
    Salida de build_compuestos
    Returns:
    pandas.DataFrame
    Columnas sample_id, compuesto_id, abundancia y unidad; una fila por cada valor de abundancia
    """
    sample_cols = sample_meta["raw_column"].tolist()
    long_rows = []
    compuesto_ids = compuestos["compuesto_id"].tolist()
    col_to_sample_id = dict(zip(sample_meta["raw_column"], build_muestras(sample_meta)["sample_id"]))

    for row_idx, compuesto_id in enumerate(compuesto_ids):
        row_values = raw_df.iloc[row_idx][sample_cols]
        for col, abundancia in row_values.items():
            long_rows.append(
                {
                    "sample_id": col_to_sample_id[col],
                    "compuesto_id": compuesto_id,
                    "abundancia": float(abundancia) if abundancia is not None else None,
                    "unidad": "area_de_pico",
                }
            )
    return pd.DataFrame(long_rows)


# Paso 5: persistir en SQLite + Parquet

def save_to_sqlite(muestras: pd.DataFrame, compuestos: pd.DataFrame, db_path: str) -> None:
    """
    Guarda las tablas de muestras y compuestos en la base SQLite. Descarta la columna raw_column antes de guardar muestras 
    Parameters:
    muestras : pandas.DataFrame
    Salida de build_muestras
    compuestos : pandas.DataFrame
    Salida de build_compuestos
    db_path : str
    Ruta al archivo SQLite
    Returns:
    None
    Escribe (append) en las tablas `muestras` y `compuestos`; crea el archivo si no existen
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    muestras.drop(columns=["raw_column"]).to_sql("muestras", con, if_exists="append", index=False)
    compuestos.to_sql("compuestos", con, if_exists="append", index=False)
    con.close()


def save_to_parquet(abundancias: pd.DataFrame, parquet_path: str) -> None:
    """
    Guarda la matriz de abundancias en formato parquet
    Parameters:
    abundancias : pandas.DataFrame
    Salida de build_abundancias
    parquet_path : str
    Ruta de destino del archivo 
    Returns:
    None
    Crea la carpeta de destino si no existe
    """
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    abundancias.to_parquet(parquet_path, index=False)


# Main

def main():
    """
    Orquesta la carga completa del dataset: lee el Excel, separa las columnas de muestra de las de metadata, arma las tres tablas (muestras, compuestos, abundancias) y las persiste en SQLite y parquet
    Returns:
    None
    No devuelve nada, solo  imprime un resumen de cuantas muestras, compuestos y filas de abundancia se cargaron
    """
    raw_df = read_excel_raw(EXCEL_PATH, SHEET_NAME)

    sample_cols = [c for c in raw_df.columns if c not in (
        "Metabolite", "CAS", "Source", "Score", "Quantification Ions",
        "Avg. RI", "Avg. RT (Min)", "Avg.S/N", "Hits",
    )]
    sample_meta = parse_sample_columns(sample_cols)

    muestras = build_muestras(sample_meta)
    compuestos = build_compuestos(raw_df)
    abundancias = build_abundancias(raw_df, sample_meta, compuestos)

    save_to_sqlite(muestras, compuestos, DB_PATH)
    save_to_parquet(abundancias, PARQUET_PATH)

    print(f"Muestras cargadas: {len(muestras)}")
    print(f"Compuestos cargados: {len(compuestos)} ({compuestos['identified'].sum()} identificados)")
    print(f"Filas de abundancia (formato largo): {len(abundancias)}")
    print(f"SQLite: {DB_PATH}")
    print(f"Parquet: {PARQUET_PATH}")


if __name__ == "__main__":
    main()

"""
Loader: Alınc et al. 2026 DOI 10.1002/ps.70436)
Tomate (S. lycopersicum) x T. harzianum T22 (biocontrol) x herbivoria de Halyomorpha halys (feeding + oviposition) - VOCs por HS-GC-MS

Estructura del Excel:
- Fila 4: encabezados (N, RT, RI, compound, FO 1-5, T22-FO 1-5, CTRL 1-5)
- Filas 5-27: 23 compuestos (21 identificados + 2 "Unknown")
- N = ID unico de compuesto

Diseño biologico
- CTRL = plantas no inoculadas, no infestadas
- FO   = plantas no inoculadas + alimentacion/oviposicion de H. halys
- T22-FO = plantas inoculadas con T. harzianum T22 + alimentacion/oviposicion de H. halys
- "FO" es "Feeding + Oviposition"  del insecto

Abundancias: el archivo usa "0" explicito para picos no detectados. Se preservan como 0, no como none, para no perder esa distincion frente a valores realmente faltantes

Este dataset introduce dos ejes biologicos que los loaders anteriores no tenian (especie de herbivoro, agente de biocontrol por separado del "patogeno"). Se guarda en una 
tabla separada (muestras_alinc2026).
"""

import re
import sqlite3
from datetime import date
from pathlib import Path
import openpyxl
import pandas as pd


# Configuracion

EXCEL_PATH = "data/raw/Alinc/rawdata_alinc.xlsx"
SHEET_NAME = "VOC analysis"
HEADER_ROW = 4
FIRST_DATA_ROW = 5
LAST_DATA_ROW = 27

DB_PATH = "db/tfm_vocs.db"
PARQUET_DIR = "db/data/processed"
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

# Mapeo compuesto -> PubChem CID, indexado por N (columna "N" del excel, ID unico de compuesto). Se usa N y no el nombre porque "alpha-terpinene" esta repetido con dos RT/RI distintos.
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


# Paso 1: leer el Excel

def read_excel_raw(path: str, sheet: str) -> tuple[list[str], list[list]]:
    """
    Lee el excel crudo y separa el encabezado de las filas de datos
    Parameters:
    path : str
    Ruta al archivo excel
    sheet : str
    Nombre de la hoja a lee
    Returns:
    tuple[list[str], list[list]]
    header: nombres de columna no vacios de la fila de encabezado; rows: valores crudos de las filas FIRST_DATA_ROW a LAST_DATA_ROW (23 compuestos)
    """
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
    """
    Identifica las columnas de muestra dentro del encabezado y parsea su grupo de tratamiento y replica
    Parameters:
    header : list[str]
    Encabezado completo, salida de read_excel_raw
    Precondition:
    Cada columna de muestra debe matchear el patron "{CTRL|FO|T22-FO} {numero}"; si no,  ValueError
    Returns:
    pandas.DataFrame
    Una fila por columna de muestra, con raw_column, treatment_group y biological_replicate
    """
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


# Paso 2: construir tabla de muestras

def build_muestras(sample_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Arma la tabla de metadatos de muestras, traduciendo el grupo de tratamiento  via la funcion interna _map_treatment
    Parameters:
    sample_meta : pandas.DataFrame
    Salida de parse_sample_columns
    Returns:
    pandas.DataFrame
    Una fila por muestra, con sample_id, especie, los tres ejes de tratamiento y metadata fija del dataset, mas raw_column 
    """
    df = sample_meta.copy()

    def _map_treatment(group):
        """
        Traduce un grupo de tratamiento a sus 3 componentes biologicos, segun TREATMENT_MAP
        Parameters:
        group : str
        Grupo de tratamiento 
        Returns:
        pandas.Series
        Con fungal_inoculant, herbivore_species y physiological_state
        """
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


# Paso 3: construir catalogo de compuestos

def build_compuestos(header: list[str], rows: list[list]) -> pd.DataFrame:
    """
    Arma el catalogo de compuestos, indexando por N 
    Parameters:
    header : list[str]
    Encabezado completo, salida de read_excel_raw
    rows : list[list]
    Filas de datos, salida de read_excel_raw
    Returns:
    pandas.DataFrame
    Una fila por compuesto (23 en total), con compuesto_id, n_original, name_original, retention_time, retention_index, identified y pubchem_cid
    """
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



# Paso 4: construir matriz de abundancias

def build_abundancias(header: list[str], rows: list[list], sample_meta: pd.DataFrame, muestras: pd.DataFrame) -> pd.DataFrame:
    """
    Arma la matriz de abundancias en formato largo, preservando los picos no detectados como 0.00
    Parameters:
    header : list[str]
    Encabezado completo, salida de read_excel_raw
    rows : list[list]
    Filas de datos, salida de read_excel_raw
    sample_meta : pandas.DataFrame
    Salida de parse_sample_columns
    muestras : pandas.DataFrame
    Salida de build_muestras
    Returns:
    pandas.DataFrame
    Columnas sample_id, compuesto_id, abundancia y unidad; 345 filas
    """
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


# Paso 5: pguardar en SQLite + Parquet

def save_to_sqlite(muestras: pd.DataFrame, compuestos: pd.DataFrame, db_path: str) -> None:
    """
    Guarda las tablas de muestras y compuestos en la base SQLite
    introduce dos ejes biologicos que los loaders anteriores no tenian (herbivoro y agente de biocontrol por separado del patogeno), asi que se guarda en su propia tabla muestras_alinc2026. 
    Antes de agregar a la tabla `compuestos` comun, borra las filas existentes de este dataset para evitar duplicados
    Despues hay que correr src/01_unify_schema.py para fusionarla en `muestras`
    Parameters:
    muestras : pandas.DataFrame
    Salida de build_muestras
    compuestos : pandas.DataFrame
    Salida de build_compuestos
    db_path : str
    Ruta al archivo SQLite
    Returns:
    None
    Descarta raw_column de muestras y n_original/retention_time/retention_index de compuestos antes de guardar; reemplaza (if_exists="replace") la tabla muestras_alinc2026, y agrega (append) a la tabla `compuestos` comun
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("DELETE FROM compuestos WHERE dataset_origin = ?", (DATASET_ORIGIN,))
    muestras.drop(columns=["raw_column"]).to_sql(
        "muestras_alinc2026", con, if_exists="replace", index=False
    )
    compuestos.drop(columns=["n_original", "retention_time", "retention_index"]).to_sql(
        "compuestos", con, if_exists="append", index=False
    )
    con.commit()
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
    Lee el Excel, arma las tres tablas y las persiste en SQLite  y parquet
    Returns:
    None
    Imprime un resumen de cuantas muestras y compuestos se cargaron, y cuantas filas de abundancia se generaron
    """
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

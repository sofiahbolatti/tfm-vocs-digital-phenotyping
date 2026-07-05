"""
Loader: Laupheimer et al. 2024 (Physiologia Plantarum, DOI 10.1111/ppl.14646)
Cebada x Blumeria hordei - VOCs por TD-GC/MS

Fuente original: dataset publico en figshare (DOI 27320754)

Estructura del CSV (archivo MetaboVOC.csv):
- Fila 1: numeros de muestra (1, 2, 3... 32)
- Fila 2: etiquetas de muestra (ej: "MLO WT 1", "mlo5 Bh 3")
- Filas 3-27: un VOC por fila, abundancia por muestra
- Separador: punto y coma (;)
- Valores faltantes: celda vacia o espacio (" ")

Mapeo biologico (confirmado contra el paper original):
- Genotipos: WT (wild type, susceptible), mlo5 (mutante resistente por perdida de funcion del gen MLO)
- "Bh" en la etiqueta = infectado con Blumeria hordei; sin "Bh" = control
- Numero al final de la etiqueta = timepoint en DAI (days after inoculation: 1, 3 o 5
- Unidad de abundancia: concentracion en ng por gramo de peso fresco (ng/gFW)
"""

import sqlite3
from datetime import date
from pathlib import Path
import pandas as pd

#----------------------------------------------------------------------------------
# Configuración
#----------------------------------------------------------------------------------

CSV_PATH = "data/raw/Laupheimer/rawdata_laupheimer.csv"
DATASET_ORIGIN = "Laupheimer2024"
DOI = "10.1111/ppl.14646"
ANALYTICAL_TECHNIQUE = "TD-GC/MS"
TISSUE = "leaves"
SPECIES = "Hordeum vulgare"

# Mapeo compuesto identificado -> PubChem CID (buscado y verificado en PubChem
# por nombre/CAS). Los 25 VOCs de este dataset estan identificados .

PUBCHEM_CID_BY_NAME = {
    "(Z)-3-Hexenol": 5281167,
    "(Z)-3-Hexenyl acetate": 5363388,
    "2-Ethyl hexanol": 7720,
    "3-Heptanol": 11520,
    "3-Heptanone": 7802,
    "6-Methyl-5-heptenone": 9862,
    "Heptane 2.3-dimethyl": 26375,
    "Heptane 2.2.4.6.6-pentamethyl": 26058,
    "Nonane": 8141,
    "Nonanal": 31289,
    "n-Decanal": 8175,
    "Decane-4-methyl": 17835,
    "Dodecane": 8182,
    "n-Tetradecane": 12389,
    "a-Pinene": 6654,
    "para-Cymene": 7463,
    "Camphene": 6616,
    "Limonene": 22311,
    "1.8-Cineol": 2758,
    "Linalool": 6549,
    "Camphor": 2537,
    "(-)-Menthol": 16666,
    "Bornyl acetate": 6448,
    "b-Caryophyllene": 5281515,
    "Methyl salicylate": 4133,
}

#---------------------------------------------------------------------------
# Paso 1: Leer el csv
#----------------------------------------------------------------------------

def read_csv_raw(path):
    df = pd.read_csv(path, sep=";", header=None)
    sample_numbers = df.iloc[0, 1:].tolist()
    sample_labels = df.iloc[1, 1:].tolist()
    voc_data = df.iloc[2:, :].copy()
    voc_data.columns = ["voc_name"] + sample_numbers
    voc_data = voc_data.reset_index(drop=True)
    return voc_data, sample_labels

#---------------------------------------------------------------------------
# Paso 2: construir tabla de muestras
#----------------------------------------------------------------------------

def parse_sample_label(label):
    genotipo = "mlo5" if label.startswith("mlo") else "WT"
    infectado = "Bh" in label
    physiological_state = "infected" if infectado else "control"
    timepoint = int(label.strip().split(" ")[-1])
    return genotipo, physiological_state, timepoint


def build_muestras(sample_labels, sample_numbers):
    rows = []
    for num, label in zip(sample_numbers, sample_labels):
        genotipo, physiological_state, timepoint = parse_sample_label(label)
        sample_id = (
            DATASET_ORIGIN.lower()
            + "_" + genotipo.lower()
            + "_" + physiological_state
            + "_" + str(timepoint) + "dai"
            + "_r" + str(num).zfill(2)
        )
        rows.append({
            "sample_id": sample_id,
            "species": SPECIES,
            "genotype": genotipo,
            "physiological_state": physiological_state,
            "microorganism": "Blumeria hordei",
            "strain": None,
            "timepoint": str(timepoint) + "DAI",
            "tissue": TISSUE,
            "dataset_origin": DATASET_ORIGIN,
            "doi": DOI,
            "analytical_technique": ANALYTICAL_TECHNIQUE,
            "load_date": date.today().isoformat(),
            "validation_status": "pendiente_validacion",
            "intended_use": "classifier",
            "raw_label": label,
        })
    return pd.DataFrame(rows)

#---------------------------------------------------------------------------
# Paso 3: Construir catálogo de compuestos
#----------------------------------------------------------------------------

def build_compuestos(voc_data):
    rows = []
    for i, name in enumerate(voc_data["voc_name"].tolist()):
        rows.append({
            "compuesto_id": f"{DATASET_ORIGIN.lower()}_cmp{i+1:03d}",
            "name_original": name,
            "cas": None,
            "identified": True,
            "pubchem_cid": PUBCHEM_CID_BY_NAME.get(name),
            "dataset_origin": DATASET_ORIGIN,
        })
    return pd.DataFrame(rows)

#---------------------------------------------------------------------------
# Paso 4: contruir matriz de abundancias
#----------------------------------------------------------------------------

def build_abundancias(voc_data, muestras):
    col_to_sample_id = dict(zip(
        muestras["raw_label"].index,
        muestras["sample_id"]
    ))
    long_rows = []
    compuesto_ids = [
        f"{DATASET_ORIGIN.lower()}_cmp{i+1:03d}"
        for i in range(len(voc_data))
    ]
    for row_idx, compuesto_id in enumerate(compuesto_ids):
        for col_idx, sample_id in enumerate(muestras["sample_id"]):
            col_num = voc_data.columns[col_idx + 1]
            valor = voc_data.iloc[row_idx][col_num]
            if valor == " " or valor == "" or pd.isna(valor):
                abundancia = None
            else:
                abundancia = float(valor)
            long_rows.append({
                "sample_id": sample_id,
                "compuesto_id": compuesto_id,
                "abundancia": abundancia,
                "unidad": "concentracion_ng_per_gfw",
            })
    return pd.DataFrame(long_rows)


#---------------------------------------------------------------------------
# Paso 5: Guardar en SQLite + Parquet
#----------------------------------------------------------------------------

ef save_to_sqlite(muestras, compuestos, db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    muestras.drop(columns=["raw_label"]).to_sql(
        "muestras", con, if_exists="append", index=False
    )
    compuestos.to_sql(
        "compuestos", con, if_exists="append", index=False
    )
    con.close()

def save_to_parquet(abundancias, parquet_path):
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    abundancias.to_parquet(parquet_path, index=False)

#---------------------------------------------------------------------------
# Paso 6: Main
#----------------------------------------------------------------------------

def main():
   DB_PATH = "db/tfm_vocs.db"
   PARQUET_PATH = "db/data/processed/abundancias_laupheimer2024.parquet"

    voc_data, sample_labels = read_csv_raw(CSV_PATH)
    sample_numbers = list(range(1, len(sample_labels) + 1))

    muestras = build_muestras(sample_labels, sample_numbers)
    compuestos = build_compuestos(voc_data)
    abundancias = build_abundancias(voc_data, muestras)

    save_to_sqlite(muestras, compuestos, DB_PATH)
    save_to_parquet(abundancias, PARQUET_PATH)

    print(f"Muestras cargadas: {len(muestras)}")
    print(f"Compuestos cargados: {len(compuestos)}")
    print(f"Filas de abundancia: {len(abundancias)}")


if __name__ == "__main__":
    main()

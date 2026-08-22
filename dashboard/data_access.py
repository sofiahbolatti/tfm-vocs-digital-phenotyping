import sqlite3
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from scipy.stats import zscore as _zscore

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "tfm_vocs.db"
MODELS_DIR = BASE_DIR / "models"

DATASET_ORIGIN = {
    "lazazzara2018": "Lazazzara2018",
    "laupheimer2024": "Laupheimer2024",
    "moreira2024": "Moreira2024",
    "ayelo2026": "Ayelo2026",
    "ghosh2022": "Ghosh2022",
    "alinc2026": "Alinc2026",
}

IMPUTA_CON_CERO = {"laupheimer2024"}

# Info para el generador de hipotesis (CVAE): de donde sale el CSV de abundancia
# y si hay que filtrar por especie dentro de ese CSV (caso Ghosh2022, que comparte
# un solo archivo entre tomate y pimiento)
CVAE_DATASET_INFO = {
    "lazazzara2018": {"csv_slug": "lazazzara2018", "dataset_origin": "Lazazzara2018", "species_filter": None},
    "laupheimer2024": {"csv_slug": "laupheimer2024", "dataset_origin": "Laupheimer2024", "species_filter": None},
    "moreira2024": {"csv_slug": "moreira2024", "dataset_origin": "Moreira2024", "species_filter": None},
    "ayelo2026": {"csv_slug": "ayelo2026", "dataset_origin": "Ayelo2026", "species_filter": None},
    "ghosh2022_tomate": {"csv_slug": "ghosh2022", "dataset_origin": "Ghosh2022", "species_filter": "Solanum lycopersicum"},
    "ghosh2022_pimiento": {"csv_slug": "ghosh2022", "dataset_origin": "Ghosh2022", "species_filter": "Capsicum annuum"},
    "alinc2026": {"csv_slug": "alinc2026", "dataset_origin": "Alinc2026", "species_filter": None},
}

# Categorias de physiological_state en el mismo orden que uso cada notebook al armar
# pd.get_dummies(..., drop_first=True) para condicionar el CVAE. La primera categoria
# de cada lista es la referencia (no se codifica - "estar en 0 en todas las columnas").
CATEGORIAS_CONDICION = {
    "lazazzara2018": ["control", "infected"],
    "laupheimer2024": ["control", "infected"],
    "moreira2024": ["control", "herbivory_caterpillar", "herbivory_aphid", "infected_bacterial", "infected_fungal"],
    "ayelo2026": ["control", "herbivory_exposed"],
    "ghosh2022_tomate": ["control", "herbivory_only", "infected_viral"],
    "ghosh2022_pimiento": ["control", "herbivory_only", "infected_viral"],
    "alinc2026": ["control", "herbivory_only", "biocontrol_herbivory"],
}


def get_db_connection():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def load_classifier(slug):
    d = MODELS_DIR / slug
    return {
        "model": joblib.load(d / "classifier.joblib"),
        "feature_columns": joblib.load(d / "feature_columns.joblib"),
        "meta": joblib.load(d / "classifier_meta.joblib"),
        "label_encoder": joblib.load(d / "label_encoder.joblib") if (d / "label_encoder.joblib").exists() else None,
    }


def _sampling(args):
    import tensorflow as tf
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * eps


def load_cvae_artifacts(slug):
    from tensorflow import keras
    d = MODELS_DIR / slug
    custom = {"sampling": _sampling}
    artifacts = {
        "encoder": keras.models.load_model(d / "cvae_encoder.keras", custom_objects=custom),
        "decoder": keras.models.load_model(d / "cvae_decoder.keras", custom_objects=custom),
        "scaler": joblib.load(d / "scaler.pkl"),
        "compound_ids": joblib.load(d / "compound_ids.pkl"),
        "meta": joblib.load(d / "cvae_meta.pkl"),
    }
    if (d / "isolation_forest.pkl").exists():
        artifacts["isolation_forest"] = joblib.load(d / "isolation_forest.pkl")
        artifacts["autoencoder"] = keras.models.load_model(d / "autoencoder.keras", custom_objects=custom)
        artifacts["scores_loo"] = np.load(d / "scores_loo.npy")
        artifacts["errores_loo_ae"] = np.load(d / "errores_loo_ae.npy")
    return artifacts


def load_predictor_data(slug):
    dataset_origin = DATASET_ORIGIN[slug]
    con = get_db_connection()
    muestras = pd.read_sql(
        "SELECT sample_id, species, physiological_state FROM muestras WHERE dataset_origin = ?",
        con, params=(dataset_origin,)
    )
    compuestos = pd.read_sql(
        "SELECT compuesto_id, pubchem_cid FROM compuestos WHERE dataset_origin = ? AND pubchem_cid IS NOT NULL",
        con, params=(dataset_origin,)
    )
    con.close()

    abundancias = pd.read_csv(BASE_DIR / "db" / "data" / "processed" / f"abundancias_{slug}.csv")
    abundancias = abundancias.merge(compuestos, on="compuesto_id", how="inner")
    abundancias["pubchem_cid"] = abundancias["pubchem_cid"].astype(int).astype(str)

    matriz = abundancias.pivot_table(index="sample_id", columns="pubchem_cid", values="abundancia")
    if slug in IMPUTA_CON_CERO:
        matriz = matriz.fillna(0)
    matriz_z = matriz.apply(_zscore)

    artefactos = load_classifier(slug)
    columnas = artefactos["feature_columns"]

    faltantes = [c for c in columnas if c not in matriz_z.columns]
    for c in faltantes:
        if c.startswith("species_"):
            valor_especie = c[len("species_"):]
            matriz_z[c] = (muestras.set_index("sample_id")["species"] == valor_especie).astype(int)
        else:
            matriz_z[c] = 0

    matriz_z = matriz_z.reindex(columns=columnas)
    matriz_z = matriz_z.join(muestras.set_index("sample_id")[["physiological_state"]])

    return {"artefactos": artefactos, "matriz": matriz_z}


def load_cvae_sample_matrix(slug):
    info = CVAE_DATASET_INFO[slug]
    con = get_db_connection()
    query = "SELECT sample_id, physiological_state FROM muestras WHERE dataset_origin = ?"
    params = [info["dataset_origin"]]
    if info["species_filter"]:
        query += " AND species = ?"
        params.append(info["species_filter"])
    muestras = pd.read_sql(query, con, params=params)
    con.close()

    abundancias = pd.read_csv(BASE_DIR / "db" / "data" / "processed" / f"abundancias_{info['csv_slug']}.csv")
    abundancias = abundancias[abundancias["sample_id"].isin(muestras["sample_id"])]
    matriz = abundancias.pivot_table(index="sample_id", columns="compuesto_id", values="abundancia")
    return matriz, muestras.set_index("sample_id")["physiological_state"]


def percentil_vs_referencia(score, referencia):
    return float(np.mean(np.asarray(referencia) <= score) * 100)


def construir_vector_condicion(slug, estado):
    """Arma el vector de condicion tal como lo espera el encoder/decoder de esta especie,
    replicando pd.get_dummies(categorias, drop_first=True): todo en 0 si es la categoria
    de referencia (la primera de la lista), un 1 en la posicion correspondiente si no."""
    categorias = CATEGORIAS_CONDICION[slug]
    columnas = categorias[1:]
    vector = [1.0 if estado == c else 0.0 for c in columnas]
    return np.array([vector], dtype="float32")


def generar_hipotesis(slug, sample_id, condicion_objetivo):
    art = load_cvae_artifacts(slug)
    matriz, estados = load_cvae_sample_matrix(slug)

    compound_ids = list(art["compound_ids"])
    perfil = matriz.loc[sample_id].reindex(compound_ids).fillna(0).values.reshape(1, -1)

    perfil_log = np.log1p(perfil)
    perfil_scaled = art["scaler"].transform(perfil_log).astype("float32")

    condicion_actual_valor = estados[sample_id]
    c_actual = construir_vector_condicion(slug, condicion_actual_valor)
    c_objetivo = construir_vector_condicion(slug, condicion_objetivo)

    z_mean, z_log_var, z = art["encoder"].predict([perfil_scaled, c_actual], verbose=0)
    reconstruido = art["decoder"].predict([z_mean, c_objetivo], verbose=0)

    resultado = {"meta": art["meta"], "condicion_actual": condicion_actual_valor, "condicion_objetivo": condicion_objetivo}

    if "isolation_forest" in art:
        score_if = -art["isolation_forest"].score_samples(reconstruido)[0]
        perc_if = percentil_vs_referencia(score_if, art["scores_loo"])

        recon_ae = art["autoencoder"].predict(reconstruido, verbose=0)
        error_ae = float(np.mean((reconstruido - recon_ae) ** 2))
        perc_ae = percentil_vs_referencia(error_ae, art["errores_loo_ae"])

        if perc_if < 90 and perc_ae < 90:
            clasificacion = "plausible"
        elif perc_if > 97.5 and perc_ae > 97.5:
            clasificacion = "novedoso"
        else:
            clasificacion = "frontera"

        resultado.update({
            "clasificacion": clasificacion,
            "percentil_isolation_forest": round(perc_if, 1),
            "percentil_autoencoder": round(perc_ae, 1),
        })
    else:
        resultado["clasificacion"] = None
        resultado["nota_cualitativa"] = art["meta"].get("estado_etapa4", "sin evaluacion cuantitativa")

    return resultado

def compute_latent_space(slug):
    art = load_cvae_artifacts(slug)
    matriz, estados = load_cvae_sample_matrix(slug)

    compound_ids = list(art["compound_ids"])
    perfiles = matriz.reindex(columns=compound_ids).fillna(0).values
    perfiles_log = np.log1p(perfiles)
    perfiles_scaled = art["scaler"].transform(perfiles_log).astype("float32")

    condiciones = np.vstack([
        construir_vector_condicion(slug, estados.loc[sid]) for sid in matriz.index
    ])

    z_mean, z_log_var, z = art["encoder"].predict([perfiles_scaled, condiciones], verbose=0)

    return pd.DataFrame({
        "sample_id": matriz.index,
        "z1": z_mean[:, 0],
        "z2": z_mean[:, 1],
        "physiological_state": [estados.loc[sid] for sid in matriz.index],
    })

import tempfile
import uuid
import datetime
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, StratifiedKFold

UPLOAD_TMP_DIR = BASE_DIR / "dashboard" / "_uploads_tmp"
UPLOAD_TMP_DIR.mkdir(exist_ok=True)


def load_raw_feature_matrix(slug):
    dataset_origin = DATASET_ORIGIN[slug]
    con = get_db_connection()
    muestras = pd.read_sql(
        "SELECT sample_id, species, physiological_state FROM muestras WHERE dataset_origin = ?",
        con, params=(dataset_origin,)
    )
    compuestos = pd.read_sql(
        "SELECT compuesto_id, pubchem_cid FROM compuestos WHERE dataset_origin = ? AND pubchem_cid IS NOT NULL",
        con, params=(dataset_origin,)
    )
    con.close()
    abundancias = pd.read_csv(BASE_DIR / "db" / "data" / "processed" / f"abundancias_{slug}.csv")
    abundancias = abundancias.merge(compuestos, on="compuesto_id", how="inner")
    abundancias["pubchem_cid"] = abundancias["pubchem_cid"].astype(int).astype(str)
    matriz = abundancias.pivot_table(index="sample_id", columns="pubchem_cid", values="abundancia")
    return matriz, muestras.set_index("sample_id")


def entrenar_y_evaluar(slug, csv_nuevo_path=None):
    matriz_actual, meta_actual = load_raw_feature_matrix(slug)
    artefactos = load_classifier(slug)
    columnas = artefactos["feature_columns"]
    modelo_base = artefactos["model"]

    matriz_total = matriz_actual.copy()
    estados_total = meta_actual["physiological_state"].copy()
    n_nuevas = 0

    if csv_nuevo_path is not None:
        nuevo = pd.read_csv(csv_nuevo_path)
        nuevo["pubchem_cid"] = nuevo["pubchem_cid"].astype(int).astype(str)
        matriz_nueva = nuevo.pivot_table(index="sample_id", columns="pubchem_cid", values="abundancia")
        estados_nuevos = nuevo.drop_duplicates("sample_id").set_index("sample_id")["physiological_state"]
        matriz_total = pd.concat([matriz_total, matriz_nueva])
        estados_total = pd.concat([estados_total, estados_nuevos])
        n_nuevas = len(matriz_nueva)

    if slug in IMPUTA_CON_CERO:
        matriz_total = matriz_total.fillna(0)

    matriz_z = matriz_total.apply(_zscore).reindex(columns=columnas).fillna(0)

    X_total = matriz_z.values
    y_total = estados_total.reindex(matriz_z.index).values
    f1_nuevo = _cv_f1(modelo_base, X_total, y_total)

    X_viejo = matriz_z.loc[matriz_actual.index].values
    y_viejo = meta_actual["physiological_state"].reindex(matriz_actual.index).values
    f1_viejo = _cv_f1(modelo_base, X_viejo, y_viejo)

    modelo_nuevo = clone(modelo_base)
    modelo_nuevo.fit(X_total, y_total)

    return {
        "n_muestras_nuevas": n_nuevas,
        "n_muestras_total": len(matriz_z),
        "f1_actual": round(float(f1_viejo), 3),
        "f1_nuevo": round(float(f1_nuevo), 3),
        "mejora": f1_nuevo >= f1_viejo,
        "modelo_entrenado": modelo_nuevo,
        "feature_columns": columnas,
    }


def guardar_token_csv(file_storage):
    token = uuid.uuid4().hex
    destino = UPLOAD_TMP_DIR / f"{token}.csv"
    file_storage.save(destino)
    return token


def ruta_token_csv(token):
    return UPLOAD_TMP_DIR / f"{token}.csv"


def aplicar_actualizacion(slug, csv_nuevo_path):
    resultado = entrenar_y_evaluar(slug, csv_nuevo_path)
    if not resultado["mejora"]:
        return resultado
    d = MODELS_DIR / slug
    joblib.dump(resultado["modelo_entrenado"], d / "classifier.joblib")
    meta = joblib.load(d / "classifier_meta.joblib")
    meta["f1_macro_test"] = resultado["f1_nuevo"]
    meta["ultima_actualizacion"] = datetime.date.today().isoformat()
    meta["n_muestras_incorporadas"] = resultado["n_muestras_nuevas"]
    joblib.dump(meta, d / "classifier_meta.joblib")
    return resultado

def _cv_f1(modelo, X, y):
    y = np.asarray(y, dtype=object)
    n_splits = max(2, min(5, pd.Series(y).value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_score(clone(modelo), X, y, cv=cv, scoring="f1_macro").mean()

def predecir_muestra_nueva(slug, archivo_csv, especie_ghosh=None):
    """Predice el estado fisiologico de una o mas muestras NUEVAS (no guardadas en la base),
    subidas como CSV con columnas sample_id, pubchem_cid, abundancia.
    El z-score se calcula con la media y el desvio de las muestras YA existentes de la especie
    (la misma logica que "ajustar con el set de entrenamiento, transformar los datos nuevos"),
    nunca con la muestra nueva sola - un z-score de un solo dato no tiene sentido."""
    matriz_ref, _ = load_raw_feature_matrix(slug)
    if slug in IMPUTA_CON_CERO:
        matriz_ref = matriz_ref.fillna(0)
    medias = matriz_ref.mean()
    desvios = matriz_ref.std(ddof=0)

    nuevo = pd.read_csv(archivo_csv)
    nuevo["pubchem_cid"] = nuevo["pubchem_cid"].astype(int).astype(str)
    matriz_nueva = nuevo.pivot_table(index="sample_id", columns="pubchem_cid", values="abundancia")
    if slug in IMPUTA_CON_CERO:
        matriz_nueva = matriz_nueva.fillna(0)

    matriz_nueva_z = (matriz_nueva - medias) / desvios

    artefactos = load_classifier(slug)
    columnas = artefactos["feature_columns"]

    faltantes = [c for c in columnas if c not in matriz_nueva_z.columns]
    for c in faltantes:
        if c.startswith("species_"):
            valor_especie = c[len("species_"):]
            matriz_nueva_z[c] = 1 if especie_ghosh == valor_especie else 0
        else:
            matriz_nueva_z[c] = 0

    matriz_nueva_z = matriz_nueva_z.reindex(columns=columnas).fillna(0)

    modelo = artefactos["model"]
    le = artefactos["label_encoder"]

    resultados = []
    for sample_id in matriz_nueva_z.index:
        X = matriz_nueva_z.loc[[sample_id]].values
        pred_cruda = modelo.predict(X)[0]
        prediccion = le.inverse_transform([pred_cruda])[0] if le is not None else pred_cruda

        probas = None
        if hasattr(modelo, "predict_proba"):
            probas_raw = modelo.predict_proba(X)[0]
            clases = le.classes_ if le is not None else modelo.classes_
            probas = sorted(zip(clases, probas_raw), key=lambda x: -x[1])

        resultados.append({"sample_id": sample_id, "prediccion": prediccion, "probas": probas})

    return resultados

CLASSIFIER_SPECIES = [
    {"slug": "lazazzara2018", "especie": "Vitis vinifera (vid)", "dataset": "Lazazzara2018"},
    {"slug": "laupheimer2024", "especie": "Hordeum vulgare (cebada)", "dataset": "Laupheimer2024"},
    {"slug": "moreira2024", "especie": "Brassica rapa (colza)", "dataset": "Moreira2024"},
    {"slug": "ayelo2026", "especie": "Gossypium hirsutum (algodon)", "dataset": "Ayelo2026"},
    {"slug": "ghosh2022", "especie": "Solanum lycopersicum + Capsicum annuum (tomate y pimiento)", "dataset": "Ghosh2022"},
    {"slug": "alinc2026", "especie": "Solanum lycopersicum (tomate, con Trichoderma)", "dataset": "Alinc2026"},
]

ACTUALIZABLES = [s for s in CLASSIFIER_SPECIES if s["slug"] != "ghosh2022"]

CVAE_SPECIES = [
    {"slug": "lazazzara2018", "especie": "Vitis vinifera (vid)", "tiene_detector": True},
    {"slug": "laupheimer2024", "especie": "Hordeum vulgare (cebada)", "tiene_detector": True},
    {"slug": "moreira2024", "especie": "Brassica rapa (colza)", "tiene_detector": True},
    {"slug": "ayelo2026", "especie": "Gossypium hirsutum (algodon)", "tiene_detector": True},
    {"slug": "ghosh2022_tomate", "especie": "Solanum lycopersicum (tomate, Ghosh2022)", "tiene_detector": False},
    {"slug": "ghosh2022_pimiento", "especie": "Capsicum annuum (pimiento, Ghosh2022)", "tiene_detector": False},
    {"slug": "alinc2026", "especie": "Solanum lycopersicum (tomate, con Trichoderma)", "tiene_detector": False},
]

import re

def limpiar_nombre_compuesto(nombre):
    """Limpia anotaciones de laboratorio en el nombre del compuesto para mostrarlo en
    pantalla (ej. "Hexanal (vitis leave_berry)_24" -> "Hexanal"). Solo afecta el texto
    que se muestra, no toca pubchem_cid ni ningun otro dato usado por los modelos."""
    limpio = re.sub(r'\s*\([^)]*\)', '', nombre)
    limpio = re.sub(r'\?.*$', '', limpio)
    limpio = re.sub(r'[\s_]*(Std)?[\s_]*\d+$', '', limpio, flags=re.IGNORECASE)
    return limpio.strip()


def load_eda_data(slug):
    dataset_origin = DATASET_ORIGIN[slug]
    con = get_db_connection()
    muestras = pd.read_sql(
        "SELECT sample_id, species, physiological_state FROM muestras WHERE dataset_origin = ?",
        con, params=(dataset_origin,)
    )
    compuestos = pd.read_sql(
        "SELECT compuesto_id, name_original, pubchem_cid FROM compuestos WHERE dataset_origin = ?",
        con, params=(dataset_origin,)
    )
    con.close()

    compuestos["name_original"] = compuestos["name_original"].apply(limpiar_nombre_compuesto)

    abundancias = pd.read_csv(BASE_DIR / "db" / "data" / "processed" / f"abundancias_{slug}.csv")
    abundancias = abundancias.merge(compuestos, on="compuesto_id", how="inner")
    abundancias = abundancias.merge(muestras, on="sample_id", how="inner")

    resumen_clases = muestras["physiological_state"].value_counts().reset_index()
    resumen_clases.columns = ["physiological_state", "n_muestras"]

    return {
        "n_muestras": len(muestras),
        "resumen_clases": resumen_clases,
        "abundancias": abundancias,
    }

def compute_shap_summary(slug, top_n=10):
    datos = load_predictor_data(slug)
    matriz = datos["matriz"]
    X = matriz.drop(columns=["physiological_state"])
    modelo = datos["artefactos"]["model"]

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        importancia = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif np.ndim(shap_values) == 3:
        importancia = np.abs(shap_values).mean(axis=(0, 2))
    else:
        importancia = np.abs(shap_values).mean(axis=0)

    resumen = pd.Series(importancia, index=X.columns).sort_values(ascending=False).head(top_n)

    dataset_origin = DATASET_ORIGIN[slug]
    con = get_db_connection()
    nombres = pd.read_sql(
        "SELECT pubchem_cid, name_original FROM compuestos WHERE dataset_origin = ? AND pubchem_cid IS NOT NULL",
        con, params=(dataset_origin,)
    )
    con.close()
    nombres["pubchem_cid"] = nombres["pubchem_cid"].astype(int).astype(str)
    nombres["name_original"] = nombres["name_original"].apply(limpiar_nombre_compuesto)
    mapa_nombres = nombres.drop_duplicates("pubchem_cid").set_index("pubchem_cid")["name_original"]

    etiquetas = [mapa_nombres.get(cid, cid) for cid in resumen.index]
    return pd.Series(resumen.values, index=etiquetas)

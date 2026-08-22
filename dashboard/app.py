from flask import Flask, render_template, request
import plotly.express as px
import plotly.io as pio
from data_access import (
    CLASSIFIER_SPECIES, CVAE_SPECIES, CATEGORIAS_CONDICION, load_eda_data, load_predictor_data,
    compute_shap_summary, load_cvae_sample_matrix, generar_hipotesis, compute_latent_space,ACTUALIZABLES, entrenar_y_evaluar, guardar_token_csv, ruta_token_csv, aplicar_actualizacion, load_classifier,
    predecir_muestra_nueva
)

app = Flask(__name__)

MODULOS = [
    {"ruta": "eda", "nombre": "EDA"},
    {"ruta": "predictor", "nombre": "Predictor"},
    {"ruta": "shap", "nombre": "SHAP"},
    {"ruta": "hipotesis", "nombre": "Generador de hipotesis"},
    {"ruta": "latente", "nombre": "Espacio latente"},
    {"ruta": "actualizar", "nombre": "Actualizador de modelo"},
]

@app.route("/")
def index():
    return render_template("index.html", modulos=MODULOS)

@app.route("/eda")
def eda():
    slug = request.args.get("especie", CLASSIFIER_SPECIES[0]["slug"])
    datos = load_eda_data(slug)

    top5 = (
        datos["abundancias"].groupby("compuesto_id")["abundancia"].var()
        .sort_values(ascending=False).head(5).index
    )
    df_top5 = datos["abundancias"][datos["abundancias"]["compuesto_id"].isin(top5)]

    fig = px.box(
        df_top5, x="name_original", y="abundancia", color="physiological_state",
        title="Top 5 compuestos con mayor variacion entre estados"
    )
    grafico_html = pio.to_html(fig, full_html=False)

    return render_template(
        "eda.html",
        especies=CLASSIFIER_SPECIES,
        slug_actual=slug,
        n_muestras=datos["n_muestras"],
        resumen_clases=datos["resumen_clases"].to_dict("records"),
        grafico_html=grafico_html,
    )

@app.route("/predictor", methods=["GET", "POST"])
def predictor():
    slug = request.values.get("especie", CLASSIFIER_SPECIES[0]["slug"])
    datos = load_predictor_data(slug)
    matriz = datos["matriz"]

    muestra_id = request.args.get("muestra")
    if muestra_id not in matriz.index:
        muestra_id = matriz.index[0]

    fila = matriz.loc[muestra_id]
    real = fila["physiological_state"]
    X = fila.drop("physiological_state").values.reshape(1, -1)

    modelo = datos["artefactos"]["model"]
    le = datos["artefactos"]["label_encoder"]

    pred_cruda = modelo.predict(X)[0]
    prediccion = le.inverse_transform([pred_cruda])[0] if le is not None else pred_cruda

    probas = None
    if hasattr(modelo, "predict_proba"):
        probas_raw = modelo.predict_proba(X)[0]
        clases = le.classes_ if le is not None else modelo.classes_
        probas = sorted(zip(clases, probas_raw), key=lambda x: -x[1])

    resultado_nuevo = None
    error_nuevo = None
    if request.method == "POST" and "archivo" in request.files and request.files["archivo"].filename:
        especie_ghosh = request.form.get("especie_ghosh")
        try:
            resultado_nuevo = predecir_muestra_nueva(slug, request.files["archivo"], especie_ghosh)
        except Exception as e:
            error_nuevo = str(e)

    return render_template(
        "predictor.html",
        especies=CLASSIFIER_SPECIES,
        slug_actual=slug,
        muestras_ids=list(matriz.index),
        muestra_actual=muestra_id,
        prediccion=prediccion,
        real=real,
        probas=probas,
        meta=datos["artefactos"]["meta"],
        resultado_nuevo=resultado_nuevo,
        error_nuevo=error_nuevo,
    )

@app.route("/shap")
def shap_module():
    slug = request.args.get("especie", CLASSIFIER_SPECIES[0]["slug"])
    resumen = compute_shap_summary(slug)

    fig = px.bar(
        x=resumen.values, y=resumen.index, orientation="h",
        labels={"x": "Importancia SHAP promedio", "y": "Compuesto"},
        title="Compuestos mas importantes para la clasificacion"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    grafico_html = pio.to_html(fig, full_html=False)

    return render_template("shap.html", especies=CLASSIFIER_SPECIES, slug_actual=slug, grafico_html=grafico_html)

@app.route("/hipotesis")
def hipotesis():
    slug = request.args.get("especie", CVAE_SPECIES[0]["slug"])
    matriz, estados = load_cvae_sample_matrix(slug)
    muestras_ids = list(matriz.index)

    muestra_id = request.args.get("muestra")
    if muestra_id not in muestras_ids:
        muestra_id = muestras_ids[0]

    categorias = CATEGORIAS_CONDICION[slug]
    condicion_objetivo = request.args.get("condicion")
    if condicion_objetivo not in categorias:
        condicion_objetivo = categorias[-1]

    resultado = generar_hipotesis(slug, muestra_id, condicion_objetivo)

    return render_template(
        "hipotesis.html",
        especies=CVAE_SPECIES,
        slug_actual=slug,
        muestras_ids=muestras_ids,
        muestra_actual=muestra_id,
        categorias=categorias,
        resultado=resultado,
    )

@app.route("/latente")
def latente():
    slug = request.args.get("especie", CVAE_SPECIES[0]["slug"])
    df = compute_latent_space(slug)

    fig = px.scatter(
        df, x="z1", y="z2", color="physiological_state", hover_data=["sample_id"],
        title="Espacio latente del CVAE"
    )
    grafico_html = pio.to_html(fig, full_html=False)

    return render_template("latente.html", especies=CVAE_SPECIES, slug_actual=slug, grafico_html=grafico_html)

@app.route("/actualizar", methods=["GET", "POST"])
def actualizar():
    slug = request.values.get("especie", ACTUALIZABLES[0]["slug"])
    resultado = None
    token = None

    if request.method == "POST":
        accion = request.form.get("accion")
        if accion == "probar" and "archivo" in request.files and request.files["archivo"].filename:
            token = guardar_token_csv(request.files["archivo"])
            resultado = entrenar_y_evaluar(slug, ruta_token_csv(token))
        elif accion == "confirmar":
            token = request.form.get("token")
            resultado = aplicar_actualizacion(slug, ruta_token_csv(token))

    meta_actual = load_classifier(slug)["meta"]

    return render_template(
        "actualizar.html",
        especies=ACTUALIZABLES,
        slug_actual=slug,
        meta_actual=meta_actual,
        resultado=resultado,
        token=token,
    )
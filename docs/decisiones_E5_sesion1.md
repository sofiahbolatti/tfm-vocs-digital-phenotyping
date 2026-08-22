# Registro de decisiones — Etapa E5 (Dashboard), Sesión 1

Fecha: 11 de agosto de 2026. Continúa el formato de las Partes I-IV del registro (D1.x a D4.x). Pegar al final del documento, después de D4.13.

---

## PARTE V — ETAPA E5: Dashboard interactivo

### D5.1 — Corrección de framework: Flask + Plotly en vez de Plotly/Dash

**Decisión:** Se reemplaza "Plotly/Dash" por "Flask + Plotly" en toda la documentación del proyecto (Propuesta, README.md del repositorio, README.md del módulo dashboard).

**Motivo:** Revisión del material real del módulo de Visualización Avanzada (carpeta SRC completa del curso, no solo el temario) mostró que Dash nunca aparece como ejercicio propio — las únicas menciones son comentarios al pasar ("esto podría extenderse con Plotly Dash") y un link externo en un README. Lo que sí está desarrollado en profundidad es el módulo "Aplicaciones Web" (4.1 a 4.3), con progresión completa: fundamentos de Flask → formularios → templates → integración con Plotly → integración con MongoDB → ejercicio final "Dashboard Completo". Streamlit aparece una sola vez, como script suelto al final del módulo de Mapas, sin desarrollo propio. Se comparó código real de ambos ejemplos del curso antes de decidir.

### D5.2 — Elección Flask+Plotly sobre Streamlit: justificación

**Decisión:** Entre las dos alternativas con respaldo real en el curso (Streamlit y Flask+Plotly), se elige Flask+Plotly.

**Motivo:** (1) Es la opción más desarrollada pedagógicamente en el curso (módulo dedicado de 3 niveles vs. un script bonus). (2) Requiere el cambio más chico en la Propuesta — se reemplaza "Dash" por "Flask", "Plotly" queda intacto. (3) Las rutas de Flask mapean 1 a 1 con los 6 módulos del dashboard (una ruta por módulo). (4) Da control total de estado, necesario para el módulo 6 (actualizador de modelo con reentrenamiento gateado por confirmación humana) — algo más difícil de garantizar con el modelo de rerun automático de Streamlit.

### D5.3 — Primer paso obligatorio: exportar modelos a disco antes de tocar el dashboard

**Decisión:** Antes de escribir cualquier código de interfaz visual, se exportan a disco (carpeta `models/`, un subdirectorio por dataset) todos los artefactos entrenados en las Etapas 2, 3 y 4: clasificadores (joblib), encoder/decoder del CVAE (formato `.keras`), Isolation Forest y autoencoder de Etapa 4 (donde corresponde), y los arrays de referencia LOO (`scores_loo.npy`, `errores_loo_ae.npy`, formato numpy).

**Motivo:** Ninguno de los notebooks (Etapa 2: `02_EDA_baseline_por_especie.ipynb`; Etapa 3+4: los 6 `03_CVAE_<especie>.ipynb`) guardaba nada a disco — todo vivía en memoria mientras el notebook corría. Un dashboard no puede reentrenar 7 CVAEs y sus clasificadores cada vez que alguien lo abre.

### D5.4 — `models/` se versiona en git, a diferencia de `data/raw`

**Decisión:** La carpeta `models/` se comitea normalmente al repositorio, sin agregarla a `.gitignore`.

**Motivo:** A diferencia de `data/raw` (excluido por tamaño/licencia de los datasets originales) y de `db/data/processed/*.parquet` (excluido por ser regenerable desde los loaders), los artefactos de modelo son chicos (KB a pocos MB, dado el tamaño de muestra de 9 a 98 por especie) y son indispensables para que el dashboard funcione sin reentrenar — sin ellos, cualquiera que clone el repo (directores, evaluadores) no podría abrir la app.

### D5.5 — Consistencia del `.gitignore`: CSVs procesados también se excluyen

**Decisión:** Se agrega `db/data/processed/*.csv` al `.gitignore`, además del `*.parquet` ya excluido.

**Motivo:** Los CSV en `db/data/processed/` son el mismo contenido que los `.parquet` ya ignorados (exportación redundante), y por la misma lógica de "regenerable desde los loaders" no tiene sentido versionarlos mientras el parquet sí queda afuera.

### D5.6 — Hallazgo: exportación de Etapa 3+4 ya estaba completa en GitHub

**Decisión:** Se confirma, revisando directamente el repositorio en GitHub, que la exportación de CVAE + detectores para las 7 especies/casos ya había sido realizada y comiteada en una sesión anterior (commits `713ac95` a `7ad5f84`, 9 de agosto de 2026) — no fue necesario rehacer ese trabajo.

**Motivo:** La carpeta local sincronizada vía OneDrive no reflejaba el estado real del repositorio remoto (el `git log` local estaba desactualizado varios commits). Se resolvió con `git pull`. Se verificó archivo por archivo vía GitHub (extensión Chrome) que las 7 carpetas de especie tienen los artefactos esperados.

### D5.7 — Ausencia de detector en 3 especies/casos: comportamiento correcto, no un bug

**Decisión:** Se confirma que `alinc2026`, `ghosh2022_tomate` y `ghosh2022_pimiento` no tienen `isolation_forest.pkl`, `autoencoder.keras` ni arrays LOO en `models/` de forma intencional, no por un error de exportación.

**Motivo:** Documentado en el registro de decisiones ya existente (D4.8, D4.9): para estos 3 casos, ambos detectores (Isolation Forest y autoencoder) se probaron exhaustivamente — incluyendo ajustes de hiperparámetros y una variante lineal por PCA — sin alcanzar un umbral de confianza aceptable, dada la relación desfavorable de compuestos/muestras (14-15 muestras reales, 20-93 compuestos). Se tratan como "caso cualitativo" sin clasificación plausible/frontera/novedoso. Esto se refleja en `cvae_meta.pkl` de cada carpeta con la clave `estado_etapa4`.

### D5.8 — Entorno de ejecución: Python 3.12 en entorno virtual separado

**Decisión:** El dashboard corre en un entorno virtual dedicado (`C:\venvs\tfm-vocs`, Python 3.12), separado del Python global de la máquina (3.14).

**Motivo:** TensorFlow no tiene distribución disponible para Python 3.14 (demasiado nuevo). Python 3.12 sí es compatible. Para evitar la confusión recurrente de "¿está activado el entorno?", se adoptó invocar siempre el intérprete por ruta completa (`C:\venvs\tfm-vocs\Scripts\python.exe`) en vez de depender de la activación (`Activate.ps1` además está bloqueado por la política de ejecución de PowerShell por defecto en esta máquina).

### D5.9 — Estructura de la app: separación de datos y rutas

**Decisión:** El dashboard se organiza en `app.py` (rutas Flask, una por módulo) + `data_access.py` (toda la lógica de acceso a base de datos y carga de modelos) + `templates/` (un HTML por módulo). Se define `CLASSIFIER_SPECIES` (6 entradas, para EDA/predictor/SHAP) y `CVAE_SPECIES` (7 entradas, para hipótesis/espacio latente) como listas separadas.

**Motivo:** Ghosh2022 tiene una estructura distinta según el módulo: el clasificador de Etapa 2 es un único modelo combinado de 3 clases (tomate+pimiento), mientras que el CVAE de Etapa 3 se entrenó por separado para cada subespecie (`ghosh2022_tomate/`, `ghosh2022_pimiento/`, cada una con su propio encoder/decoder). Usar una sola lista de especies para todos los módulos habría sido incorrecto.

### D5.10 — Módulo 1 (EDA): diseño

**Decisión:** Selector de especie + tabla de conteo por estado fisiológico + boxplot (Plotly Express) de los 5 compuestos con mayor varianza entre estados, usando `muestras` + `compuestos` (unidos por `pubchem_cid`) + los CSV de abundancia.

**Motivo:** Vista exploratoria estándar, rápida de construir y de interpretar como punto de partida.

### D5.11 — Módulo 2 (Predictor): reproducción exacta del preprocesamiento de Etapa 2

**Decisión:** El predictor NO usa abundancia cruda como input al clasificador — reproduce exactamente el z-score (`scipy.stats.zscore`, calculado sobre la matriz completa de la especie) que se usó al entrenar en `02_EDA_baseline_por_especie.ipynb`. Para Laupheimer2024 (cebada), además, los valores faltantes se imputan con 0 antes del z-score, replicando la decisión tomada en ese notebook (tratados como "no detectado"). Para Ghosh2022, se agrega una columna `species_Solanum lycopersicum` (one-hot) porque el clasificador combinado la usa como feature.

**Motivo:** El clasificador exportado espera datos en el mismo espacio en que fue entrenado. Alimentarlo con abundancia cruda habría dado predicciones sistemáticamente incorrectas sin ningún error visible que lo delate.

### D5.12 — Módulo 2: predicción sobre muestra existente, no carga manual

**Decisión:** El usuario elige una muestra ya existente en la base (no carga valores de compuestos a mano) y la app predice sobre ese perfil real, mostrando la predicción contra la etiqueta verdadera.

**Motivo:** Algunas especies tienen hasta 138 compuestos — un formulario de carga manual sería inmanejable. Queda documentado como decisión reversible: se puede agregar carga manual más adelante sin afectar lo ya construido.

### D5.13 — Módulo 3 (SHAP): simplificación respecto al notebook original

**Decisión:** Se calcula `shap.TreeExplainer` sobre el clasificador exportado y se promedia la importancia absoluta entre todas las clases (no una clase de interés específica por especie, como hacía el notebook original en algunos casos). Los compuestos se etiquetan por nombre (`name_original`), no por CID, para legibilidad.

**Motivo:** Simplificación deliberada para tener una vista general aplicable a las 6 especies sin lógica ad-hoc por caso. Documentado como diferencia consciente respecto al análisis original, no un error.

### D5.14 — Módulo 4 (Generador de hipótesis): diseño y alcance de la sesión

**Decisión:** Se construye primero para una sola especie (Vitis vinifera / Lazazzara2018) para validar el patrón antes de extender a las otras 6. El flujo: codificar una muestra real con su condición verdadera (`encoder.predict`) para obtener `z_mean`, decodificar ese mismo punto pero con la condición objetivo (`decoder.predict`) para generar un perfil sintético, y evaluarlo con Isolation Forest + autoencoder cuando la especie tiene detector cuantitativo (aplicando la regla de D4.4: plausible si ambos percentiles <90, novedoso si ambos >97.5, frontera en cualquier otro caso), o mostrar el aviso cualitativo de `cvae_meta.pkl` cuando no lo tiene.

**Motivo:** Cada notebook de Etapa 3+4 condicionó el CVAE de forma ligeramente distinta (nombres de variable, cantidad de columnas de condición según el número de clases de cada especie) — abordar las 7 especies de una implicaba el mismo riesgo de confusión que ya se había visto en la exportación de modelos. Se prioriza dejar un caso simple (binario: control/infected) completamente funcional y verificado antes de generalizar.

**Pendiente para la próxima sesión:** extender el módulo 4 a Laupheimer2024, Moreira2024 y Ayelo2026 (con detector cuantitativo), y a Alınç2026, Ghosh2022-tomate y Ghosh2022-pimiento (caso cualitativo, sin clasificación por percentil).

### D5.15 — Problemas técnicos resueltos durante la construcción del módulo 4

**Decisión / hallazgo:** Se identificaron y resolvieron dos problemas de carga de los archivos `.keras`:

1. Aparente incompatibilidad de versión de Keras (el archivo pedía 3.15.1, el entorno de verificación inicial solo tenía 3.12.4 disponible) — se confirmó que era un índice de paquetes desactualizado en el entorno de verificación, no un problema real: el entorno real del proyecto (`C:\venvs\tfm-vocs`) ya tenía Keras 3.15.1 instalado junto con TensorFlow.
2. Error real: la capa `Lambda` del encoder (el "truco de reparametrización" `z = z_mean + exp(0.5*z_log_var) * eps`) usa una función Python (`sampling`) que nunca se registró formalmente ante Keras al guardar el modelo, por lo que `load_model()` no podía reconstruirla automáticamente. Se resolvió pasando la función explícitamente vía el parámetro `custom_objects` al cargar el encoder, el decoder y el autoencoder.

**Motivo:** Documentado para que la próxima sesión (o cualquiera que reabra el proyecto) no pierda tiempo re-diagnosticando el mismo error — la función `sampling` debe pasarse siempre como `custom_objects` al cargar cualquier encoder/autoencoder de este proyecto con `keras.models.load_model()`.

### D5.16 — Verificación de alineación con el contenido del máster (metodología de trabajo)

**Decisión:** Cada patrón de código nuevo introducido en el dashboard (rutas Flask, `request.args.get`, loops/condicionales de Jinja2, `render_template`, `px.box`, `pio.to_html`, `.groupby()` de pandas) se verificó contra el material real del módulo de Visualización Avanzada (no de memoria) antes de darlo por válido. Se identificaron dos elementos sin verificación posible con el material disponible: las consultas SQL parametrizadas (`pd.read_sql(..., params=...)`, del módulo de Bases de Datos/Python, no cargado en esta sesión) y una línea de JavaScript genérico (`onchange="this.form.submit()"`, no específica de ningún contenido del curso).

**Motivo:** Mantener la alineación con el contenido efectivamente cursado fue el motivo original que disparó toda esta sesión (la corrección de "Dash"). Se aplica el mismo estándar de rigor a cada pieza nueva de código, en vez de asumir que "funciona" equivale a "está dentro del programa".

### D5.17 — Módulo 4: extensión completa a las 7 especies/casos

**Decisión:** Se generaliza el Módulo 4, construido en D5.14 solo para Vitis vinifera, a las 7 especies/casos (`CVAE_SPECIES`). Se reemplazan las funciones `construir_vector_condicion` y `generar_hipotesis` (antes con lógica hardcodeada para el caso binario de Vitis) por versiones que usan un diccionario `CATEGORIAS_CONDICION` con las categorías reales de cada especie (de 2 a 4 clases según el caso) y reconstruyen el vector de condición one-hot (`pd.get_dummies(..., drop_first=True)`) respetando el orden de columnas con el que se entrenó cada CVAE. El template `hipotesis.html` pasa de un desplegable fijo (control/infected) a uno dinámico que itera sobre `categorias`.

**Motivo:** Cada notebook condicionó el CVAE con su propio número de clases y orden de columnas (verificado individualmente por especie en la sesión anterior, D5.14). Generalizar sin ese trabajo previo habría producido vectores de condición mal alineados sin ningún error visible.

### D5.18 — Módulo 5 (Navegador del espacio latente): diseño

**Decisión:** Selector de especie + gráfico de dispersión (Plotly Express, `px.scatter`) de las 2 dimensiones del espacio latente (`z1`, `z2`) de cada muestra, coloreado por estado fisiológico, calculado con `encoder.predict()` sobre la matriz completa de la especie.

**Motivo:** Todas las 7 especies/casos usan `latent_dim=2`, lo que permite una única implementación (`compute_latent_space`) sin casos especiales. Da una vista exploratoria de si las clases se separan en el espacio latente, complementaria al Módulo 4 (que trabaja punto a punto).

### D5.19 — Módulo 6 (Actualizador de modelo): acotación de alcance

**Decisión:** Se descarta la visión original (carga libre de nuevos datasets con incorporación automática al dataset de la especie y reentrenamiento silencioso) en favor de un flujo más simple: el usuario sube un CSV con formato fijo, la app reentrena un modelo candidato *en paralelo* (sin tocar el modelo en producción), compara el F1 macro por validación cruzada del candidato contra el modelo actual, y solo permite reemplazar el modelo guardado si el candidato mejora — con confirmación humana explícita en un segundo paso.

**Motivo:** La incorporación automática al dataset base y el reentrenamiento "silencioso" (sin comparación ni gateo) quedaban fuera de lo razonable para el alcance de un TFM y agregaban riesgo de degradar modelos ya validados en las Etapas 2-4 sin que quede evidencia de por qué cambiaron. El diseño elegido preserva la idea original (permitir que el dashboard incorpore datos nuevos) pero con una salvaguarda: nunca se sobrescribe un modelo sin que la comparación de métricas lo respalde y sin que una persona lo confirme.

### D5.20 — Módulo 6: formato fijo del CSV de carga

**Decisión:** El CSV debe tener exactamente las columnas `sample_id, pubchem_cid, abundancia, physiological_state` (formato largo, una fila por compuesto y muestra), igual al formato interno ya usado por la base de datos.

**Motivo:** Evita tener que escribir un parser flexible o un mapeo de columnas configurable, algo fuera de alcance para este módulo. Mantiene consistencia con el formato de datos que ya maneja el resto del dashboard.

### D5.21 — Módulo 6: Ghosh2022 excluido de la actualización

**Decisión:** `ACTUALIZABLES` (las especies que aparecen en el selector del Módulo 6) es `CLASSIFIER_SPECIES` menos Ghosh2022.

**Motivo:** El clasificador de Ghosh2022 usa una feature adicional (`species_Solanum lycopersicum`, one-hot) que no está presente en las otras especies y que complica el pipeline genérico de reentrenamiento (habría que decidir con qué valor completarla para muestras nuevas). Se excluye para no introducir un caso especial en un módulo que, por diseño, busca ser uniforme entre especies.

### D5.22 — Módulo 6: mecanismo de "probar" y "confirmar" con token

**Decisión:** El flujo tiene dos pasos HTTP: (1) `POST /actualizar` con `accion=probar` guarda el CSV subido en una carpeta temporal (`_uploads_tmp/`) bajo un nombre único (token UUID) y devuelve la comparación de F1 sin tocar el modelo guardado; (2) un segundo `POST /actualizar` con `accion=confirmar` y el mismo `token` (viaja en un campo oculto del formulario) reentrena sobre ese mismo CSV y sobrescribe `classifier.joblib`/`classifier_meta.joblib` recién en ese momento.

**Motivo:** Garantiza que nunca se sobrescribe un modelo sin un paso de confirmación humana explícito y separado del cálculo de métricas, cumpliendo el requisito original de "si mejora se suma, si no se descarta" pero con la persona (no el sistema) tomando la decisión final.

### D5.23 — Problema técnico resuelto: incompatibilidad `pyarrow`/`ArrowDtype` con `cross_val_score`

**Decisión / hallazgo:** Al probar el Módulo 6 apareció `TypeError: only integer scalar arrays can be converted to a scalar index` dentro de `cross_val_score`, con el traceback bajando hasta `pandas.core.arrays.arrow.array.ArrowExtensionArray.__getitem__` → `pyarrow.ChunkedArray.__getitem__`. Se corrigió forzando `y = np.asarray(y, dtype=object)` al inicio de `_cv_f1`, antes de pasarlo a `cross_val_score`.

**Motivo:** La columna de etiquetas (`physiological_state`) llegaba representada con un dtype respaldado por `pyarrow` (comportamiento de versiones recientes de pandas), y el indexado por arrays enteros que hace `StratifiedKFold`/`cross_val_score` internamente no es compatible con ese tipo de columna en esta combinación de versiones de pandas/scikit-learn. Convertir explícitamente a un array de NumPy de tipo `object` evita el problema sin cambiar el resultado del cálculo.

### D5.24 — Verificación end-to-end del Módulo 6

**Decisión:** Se prueba el flujo completo con un CSV sintético de 4 muestras para Ayelo2026 (`ejemplo_actualizacion_ayelo2026.csv`, generado para la prueba, no son datos reales). El candidato entrenado con esos datos da peor F1 que el modelo actual, y la app bloquea correctamente el botón de confirmación — comportamiento esperado, no un error.

**Motivo:** Confirma que la salvaguarda de D5.19 funciona en el sentido "rechazar si empeora". El camino "aceptar si mejora" queda validado por diseño/código (mismo `if` evalúa ambos casos) pero no se ejecutó con datos reales en esta sesión, ya que hacerlo requeriría un dataset nuevo genuino de alguna de las especies.

---

## Estado al cierre de la sesión

**Completado:**
- Corrección Dash → Flask+Plotly en README.md y dashboard/README.md (comiteado). Propuesta en Word pendiente de confirmación final de que los 5 cambios se aplicaron.
- Exportación completa de modelos: Etapa 2 (6 clasificadores) + Etapa 3/4 (7 CVAEs, 4 con detector cuantitativo completo) — todo comiteado en GitHub.
- Esqueleto de la app Flask funcionando (`app.py`, `data_access.py`, `templates/`).
- Módulo 1 (EDA): funcionando, las 6 especies.
- Módulo 2 (Predictor): funcionando, las 6 especies.
- Módulo 3 (SHAP): funcionando, las 6 especies.
- Módulo 4 (Generador de hipótesis): funcionando, las 7 especies/casos.
- Módulo 5 (Navegador del espacio latente): funcionando, las 7 especies/casos.
- Módulo 6 (Actualizador de modelo): funcionando y verificado (probar + rechazo cuando empeora). Camino de confirmación validado por diseño, sin ejecutar con datos reales todavía.

**Pendiente:**
- Confirmar corrección de la Propuesta en Word (5 reemplazos de "Plotly/Dash").
- Probar el camino de "confirmar" del Módulo 6 con un dataset real que mejore el F1, cuando haya uno disponible.
- Definir una estrategia de limpieza para `dashboard/_uploads_tmp/` (por ahora acumula los CSV subidos por token sin borrarlos).

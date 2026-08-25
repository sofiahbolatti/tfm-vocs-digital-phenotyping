# TFM: Fenotipado digital basado en VOCs para modelar la interacción planta-microorganismo y la resilencia vegetal mediante apredizaje automático.

Trabajo de Fin de Master — Master de Formacion Permanente en Big Data, Data Science e Inteligencia Artificial, Universidad Complutense de Madrid (UCM).

**Autora:** Sofia Bolatti
**Directores:** Dr. Carlos Ortega, Dr. Santiago Mota

## Objetivo

Desarrollar un pipeline de machine learning para el fenotipado digital de estados fisiologicos de plantas a partir de sus perfiles de compuestos organicos volatiles (VOCs), considerando multiples especies y diferentes tipos de estrés bióticos

## Enfoque metodologico

1. Clasificador base: Random Forest / XGBoost, un modelo independiente por especie (7 modelos) + interpretabilidad con SHAP
2. Modelo generativo central: Conditional Variational Autoencoder (CVAE), un modelo por especie, aprende un espacio latente de 2 dimensiones condicionado al estado fisiologico
3. Deteccion de anomalias: Isolation Forest + autoencoder de reconstruccion, evaluan la plausibilidad biologica de perfiles generados por el CVAE
4. Visualizacion del espacio latente: proyeccion directa en 2 dimensiones (el CVAE se diseño con latent_dim=2, sin necesidad de reduccion de dimensionalidad)
5. Dashboard interactivo (Flask + Plotly) con seis modulos: exploracion de datos (EDA), predictor, interpretabilidad SHAP, navegador del espacio latente, generador de hipotesis, y actualizador de modelo (reentrenamiento validado por humano)

La combinacion CVAE + Isolation Forest aplicada a perfiles VOC de plantas es una aproximacion no reportada previamente en la literatura de volatilomica.

## Estructura del repositorio

```
data/
  raw/              Datasets originales sin modificar
db/                  Base de datos SQLite (tfm_vocs.db) + matrices Parquet/CSV
  data/processed/    Datos limpios y estandarizados por dataset (abundancias_<dataset>.{csv,parquet})
notebooks/           Notebooks de exploracion y modelado
src/                 Codigo fuente (carga de datos, pipeline ML, etc.)
dashboard/           Aplicacion Flask + Plotly
docs/                Documentacion
```

## Datos

Los datos provienen de estudios publicados y de colaboraciones directas con sus autores. La base consolidada combina:
- **SQLite**: metadatos de cada muestra 
- **Parquet**: matrices de abundancia de VOCs en formato largo
- **Catalogo de compuestos**: estandarizacion de nombres de VOCs via PubChem CID

## Estado del proyecto

Finaloizado

## Licencia

Pública

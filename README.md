# TFM: Digital Phenotyping Based on VOCs to Model Plant-Microorganism Interaction and Plant Resilience Using Machine Learning

Trabajo de Fin de Master — Master de Formacion Permanente en Big Data, Data Science e Inteligencia Artificial (8a edicion), Universidad Complutense de Madrid (UCM), en colaboracion con nticmaster.

**Autora:** Sofia Bolatti
**Directores:** Dr. Carlos Ortega, Dr. Santiago Mota

## Objetivo

Desarrollar un pipeline de machine learning para el fenotipado digital de estados fisiologicos de plantas a partir de sus perfiles de compuestos organicos volatiles (VOCs), considerando multiples especies y patogenos:

- **Trigo** x F. culmorum, F. avenaceum, F. graminearum, Parastagonospora nodorum, oidio
- **Tomate** x T. virens, B. cinerea, control
- **Vid** x P. viticola, control

(El alcance final depende de los datos que resulten utilizables.)

## Enfoque metodologico

1. **Clasificador base**: Random Forest / XGBoost (multiclase, multiespecie, especie como covariable) + interpretabilidad con SHAP
2. **Modelo generativo central**: Conditional Variational Autoencoder (CVAE) - aprende un espacio latente condicionado al estado fisiologico
3. **Deteccion de anomalias**: Isolation Forest - evalua la plausibilidad biologica de perfiles generados por el CVAE
4. **Reduccion de dimensionalidad / visualizacion**: PCA, t-SNE, UMAP
5. **Dashboard interactivo** (Flask + Plotly) con seis modulos: exploracion de datos (EDA), predictor, interpretabilidad SHAP, navegador del espacio latente, generador de hipotesis, y actualizador de modelo (reentrenamiento validado por humano)

La combinacion CVAE + Isolation Forest aplicada a perfiles VOC de plantas es una aproximacion no reportada previamente en la literatura de volatilomica.

## Estructura del repositorio

```
data/
  raw/              Datasets originales sin modificar
  processed/        Datos limpios y estandarizados
db/                  Base de datos SQLite + matrices Parquet
notebooks/           Notebooks de exploracion y modelado
src/                 Codigo fuente (carga de datos, pipeline ML, etc.)
dashboard/           Aplicacion Flask + Plotly
docs/                Documentacion, propuesta de TFM, notas
```

## Datos

Los datos provienen de estudios publicados y de colaboraciones directas con sus autores. La base consolidada combina:
- **SQLite**: metadatos de cada muestra (especie, microorganismo, estado fisiologico, origen, DOI, tecnica analitica, etc.)
- **Parquet**: matrices de abundancia de VOCs en formato largo
- **Catalogo de compuestos**: estandarizacion de nombres de VOCs via PubChem CID

## Estado del proyecto

En desarrollo activo - TFM en curso.

## Licencia

Repositorio privado mientras el trabajo esta en curso. Pendiente de definir licencia para publicacion futura.

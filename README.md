# Grupo 4 - Entrega TP Final Descriptiva

> Análisis del mercado de departamentos en venta en la Ciudad Autónoma de Buenos Aires.

---

## 📌 Caso de Negocio

Este proyecto simula el trabajo de una consultora cuyo cliente es una startup ficticia que tiene el objetivo de detectar propiedades subvaluadas dentro del mercado, para destinarlas a distintos modelos de negocio:

- **Flipping inmobiliario:** adquisición de propiedades con potencial de mejora, refacción y posterior venta a un valor superior.
- **Generación de renta:** compra de inmuebles a precios por debajo del mercado para destinarlos a alquiler.
- **Venta de propiedades:** compra de propiedades subvaluadas y su posterior publicación a un precio mayor.
- **Inversión en zonas emergentes:** adquisición en áreas con potencial de valorización futura (nuevas obras de infraestructura, mejora de transporte, desarrollo comercial) para posicionarse antes de que los precios reflejen la mejora.

---

## 📁 Estructura del Repositorio

```
├── data/
│   ├── Libro1.csv                   # Dataset crudo post-scraping (input de limpieza_y_eda)
│   ├── pca.csv                      # Dataset para PCA
│   ├── train.csv                    # Dataset para XGBoost
│   ├── df_clus.csv                  # Dataset para Clustering
│   ├── predicciones_xgboost.csv     # Output del modelo: precios reales y predichos (input de hipotesis)
│   └── top20_oportunidades.csv      # Top 20 oportunidades de inversión
├── notebooks/
│   ├── limpieza_y_eda.ipynb         # Limpieza, feature engineering y EDA
│   ├── pca.ipynb                    # Análisis de componentes principales
│   ├── hipotesis.ipynb              # Validación estadística de hipótesis
│   ├── clustering.ipynb             # Segmentación K-Means
│   └── xgboost.ipynb                # Modelo predictivo de precios
├── dashboard/
│   └── dashboard.pbix
└── README.md
```

Etapas del proyecto

## 1. Recolección de Datos

Los datos fueron obtenidos mediante **web scraping sobre Zonaprop**, extrayendo publicaciones de departamentos en venta en CABA. Se partió de aproximadamente 18.000 publicaciones, representando cerca del 23% del total de departamentos publicados en el portal, abarcando **21 barrios seleccionados**. Luego de una limpieza robusta, el dataset quedó en alrededor de **10.000 registros**.

El dataset fue enriquecido con **dos fuentes externas**:

- **Subtes (CSV):** para cada propiedad se calculó si tenía una estación de subte cercana, definiendo un radio máximo de 8 cuadras usando latitud y longitud.
- **Delitos (CSV):** a partir de un dataset de delitos del año 2024, se calculó para cada propiedad cuántos delitos ocurrieron en un radio de 3 cuadras, ponderando por gravedad del delito y construyendo un ranking percentilado. Esto dio origen a dos variables: `Indice_Seguridad` (continuo, donde 1 es zona más segura y 0 más insegura) y `Categoria_Seguridad` (Muy Segura / Segura / Insegura / Muy Insegura, con cortes en los percentiles 10, 50 y 90).

---

## 2. Limpieza y EDA — `limpieza_y_eda.ipynb`

### Limpieza

Se realizó una limpieza robusta y estricta variable por variable. Los criterios generales fueron:

- **Precio:** se eliminaron registros con precio nulo, igual a cero o con valor "Consultar". Se decidió también eliminar propiedades con precio por debajo de USD 20.000 y por encima de USD 1.000.000, cotejando los outliers superiores con su distribución por barrio antes de descartar.
- **Expensas:** conversión desde formato texto y tratamiento de ceros (el 37% de los registros tenía expensas = 0, interpretado como un error de scraping; se los trató como nulos para no contaminar el análisis).
- **Variables numéricas** (ambientes, dormitorios, baños, antigüedad, superficies): conversión de tipos y tratamiento de nulos con criterio contextual.
- **Outliers:** detección mediante boxplots, con decisiones justificadas en cada caso en lugar de aplicar reglas automáticas.

### Feature Engineering

- `Precio_m2`: precio por metro cuadrado cubierto.
- `Proporcion_Cubierta`: ratio superficie cubierta / superficie total.
- `Percentil_Precio_Barrio`: posición relativa de cada propiedad dentro de su barrio.
- `Indice_Seguridad` y `Categoria_Seguridad`: construidos a partir del dataset de delitos (ver sección Recolección).
- `Subte_Cerca`: variable booleana calculada a partir del CSV de estaciones.

### EDA — Principales hallazgos

- **Precio** y **Expensas** presentan distribuciones fuertemente sesgadas a la derecha, lo esperable en real estate.
- **Dormitorios y Baños** concentrados en 1 y 2; **Ambientes** entre 2 y 4.
- **Antigüedad** con picos marcados alrededor de 40 y 60 años.
- **Precio_m2** concentrado entre USD 1.500 y USD 3.500 por m².
- El precio se correlaciona fuertemente con superficie total, expensas y baños, y negativamente con antigüedad: las propiedades más nuevas tienden a cotizar a precios por m² más altos.
- **Análisis geográfico:** Puerto Madero lidera por amplio margen el precio mediano por m² (USD 5.400), seguido por Núñez, Palermo y Belgrano. En el extremo opuesto, Constitución y La Boca registran los valores más bajos. El gradiente norte-sur de la ciudad es claro en el mapa interactivo.
- **Atributos que mueven el precio:** cochera, seguridad 24hs, amenities, balcón aterrazado y categoría de seguridad presentan los mayores diferenciales de precio m² promedio. Aire acondicionado, luminosidad y losa radiante central muestran un efecto positivo más moderado. Contraintuitivamente, la cercanía al subte no mostró un efecto positivo en el precio, por lo que se decidió no incluirla como variable predictiva.

---

## 3. PCA — `pca.ipynb`

Se realizó un **Análisis de Componentes Principales** sobre las variables numéricas del dataset (`Precio`, `Expensas`, `Dormitorios`, `Baños`, `Ambientes`, `Antigüedad`, `Sup_Cubierta_m2`, `Sup_Total_m2`, `Precio_m2`, `Proporcion_Cubierta`).

Aplicando el **criterio de Kaiser** (autovalores > 1), se retuvieron **3 componentes principales**, los cuales explican aproximadamente el **82% de la varianza total**.

### Decisión de no usar los componentes

A pesar de los resultados, se decidió conservar las variables originales por las siguientes razones:

1. **Dimensionalidad manejable:** el dataset no tiene tantas variables como para que reducir dimensionalidad impacte significativamente en el tiempo de entrenamiento.
2. **Interpretabilidad:** en el contexto de negocio, importa saber exactamente qué variable influye en el precio (superficie, antigüedad, baños), algo que se pierde al trabajar con componentes abstractos.
3. **Correlaciones esperables:** la covariación entre ambientes, dormitorios, baños y superficie es una consecuencia lógica del diseño arquitectónico, no un problema que justifique transformación.

---

## 4. Validación de Hipótesis — `hipotesis.ipynb`

Se plantearon y validaron formalmente **4 hipótesis** mediante tests estadísticos.

---

### Hipótesis 1 — Los barrios difieren significativamente entre sí

**H₀:** Es indiferente segmentar por barrio; los barrios son parecidos entre sí.  
**H₁:** Los barrios son distintos entre sí y las estrategias de inversión deben ser diferenciadas.

**Test aplicado:** ANOVA de una vía sobre `Precio`, `Expensas`, `Sup_Cubierta_m2` y `Antigüedad`, agrupando por barrio.

**Resultado:** En todas las variables el p-value fue < 0.001 (F-statistic de 200.69 para Precio, 120.82 para Expensas, 28.18 para Superficie y 100.29 para Antigüedad). Se rechaza H₀. Los barrios son estadísticamente distintos y requieren estrategias diferenciadas.

---

### Hipótesis 2 — Más del 50% de las propiedades están subvaluadas

**H₀:** La proporción de propiedades publicadas por debajo de su precio predicho es ≤ 50%.  
**H₁:** Más del 50% de las propiedades están subvaluadas (precio publicado < precio predicho por el modelo).

**Test aplicado:** Test binomial de proporciones (cola superior).

**Resultado:** Se observaron 4.596 propiedades subvaluadas sobre el total del dataset, superando el valor crítico de 4.334 con un p-value prácticamente nulo. Se rechaza H₀. El modelo detecta que la mayoría de las propiedades tienen un precio de publicación por debajo del que justificarían sus características.

---

### Hipótesis 3 — Más del 3% del mercado representa una oportunidad fuerte de compra

**H₀:** La proporción de propiedades que representan una verdadera oportunidad de compra es ≤ 3%.  
**H₁:** Dicha proporción es > 3%.

**Definición de oportunidad fuerte:** una propiedad cuyo precio publicado es menor al precio predicho menos 1 RMSE del modelo (umbral de ~USD 39.000), absorbiendo así el error del modelo para ser más conservadores.

**Test aplicado:** Test binomial de proporciones (cola superior).

**Resultado:** Se observaron 294 oportunidades fuertes, superando el valor crítico de 283. Se rechaza H₀. Aunque son pocas en términos relativos, existen oportunidades de compra con una subvaluación suficientemente robusta como para resistir el error del modelo. Puerto Madero, Recoleta, Retiro y Palermo concentran la mayor proporción de estas oportunidades.

---

### Hipótesis 4 — Las propiedades antiguas tienen mayor proporción de oportunidades fuertes que las nuevas

**H₀:** La proporción de oportunidades fuertes entre propiedades antiguas (> 20 años) es igual o menor que entre propiedades nuevas.  
**H₁:** La proporción es mayor entre propiedades antiguas.

**Test aplicado:** Test Z de diferencia de proporciones (cola superior), con simulación bootstrap para validación.

**Resultado:** La diferencia observada fue de -2.20%, alejada en la dirección contraria al valor crítico (0.77%). Se rechaza H₀. Las propiedades antiguas concentran una mayor proporción de oportunidades fuertes, lo que sugiere que el mercado tiende a subvaluar más los inmuebles de mayor antigüedad respecto a su potencial real de precio.

---

## 5. Modelo Predictivo — `xgboost.ipynb`

Se entrenó un modelo de **XGBoost Regressor** para predecir el precio de publicación de cada propiedad. El objetivo no es solo predecir, sino **identificar oportunidades de inversión**: si el modelo predice un precio significativamente mayor al publicado, la propiedad estaría subvaluada respecto a sus pares de mercado.

### Modelos evaluados

Se compararon tres algoritmos capaces de capturar relaciones no lineales:

| Modelo | Descripción |
|---|---|
| XGBoost | Gradient boosting sobre árboles de decisión |
| Random Forest | Ensamble de árboles con bagging |
| Extra Trees | Variante de Random Forest con splits aleatorios |

El mejor desempeño en validación fue obtenido por **XGBoost**.

### Búsqueda de hiperparámetros

Se realizó una búsqueda aleatoria (`ParameterSampler`) sobre 50 combinaciones de hiperparámetros, optimizando por RMSE en el conjunto de validación (80/20). Los parámetros explorados incluyeron `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha` y `reg_lambda`.

### Métricas del mejor modelo

| Métrica | Valor (validación) |
|---|---|
| RMSE | ~USD 39.000 |
| R² | ~0.87 |
| MAE | ~USD 22.000 |
| MAPE | ~15% |

### Output

El modelo genera `predicciones_xgboost.csv` con el precio real publicado y el precio predicho para cada propiedad. La diferencia entre ambos (`Prediccion_Precio - Precio`) es el indicador de oportunidad utilizado en las hipótesis y en el dashboard.

---

## 6. Clustering K-Means — `clustering.ipynb`

Se aplicó **K-Means** sobre el dataset con el objetivo de identificar segmentos de mercado naturales más allá de la clasificación administrativa por barrio.

### Preprocesamiento

- Eliminación de columnas no relevantes para la segmentación (coordenadas, link, precio total, superficie total).
- Imputación de `Indice_Seguridad` nulos con la mediana del barrio.
- Codificación de barrios mediante `get_dummies`.
- Escalado con `StandardScaler`.

### Selección del número de clusters

Se evaluaron dos criterios:
- **Método del codo** (inercia vs. k)
- **Silhouette Score** para k entre 2 y 24

Ambos métodos convergieron en **k = 20** como número óptimo.

### Resultados

Los 20 clusters presentan una segmentación predominantemente geográfica: la mayoría agrupa propiedades de un único barrio, con excepción del Cluster 4, que captura unidades familiares grandes dispersas en múltiples zonas. Esto confirma que **el barrio es el principal determinante del perfil de una propiedad en CABA**.

### Categorías comerciales

Para facilitar la interpretación estratégica, los 20 clusters fueron agrupados en **5 categorías comerciales**:

| Categoría | Descripción |
|---|---|
| **Premium** | Alto precio/m², barrios de alto estatus, amenities completos |
| **Medio-Alto** | Buen precio/m², zonas consolidadas, perfil familiar o profesional |
| **Medio** | Precio/m² cercano a la mediana, barrios intermedios |
| **Accesible** | Precio/m² bajo, superficies más pequeñas o antigüedad alta |
| **Descuento Estructural** | Características que explican un precio persistentemente bajo |

---

## 7. Dashboard — Power BI `dashboard.pbix`

El dashboard integra los resultados de todas las etapas en un tablero interactivo de 5 páginas:

**Página 1 — Mercado de departamentos en venta:** KPIs principales (publicaciones, precio m² mediano, precio mediano, superficie mediana), ranking de barrios por precio m² y distribución por rango de precio.

**Página 2 — La ciudad dibujada por los datos:** mapa interactivo donde cada punto es una propiedad, coloreada por rango de precio m². Permite explorar el gradiente norte-sur de la ciudad.

**Página 3 — Qué atributos mueven el precio:** impacto de cada atributo en el precio m² mediano, relación antigüedad-precio, efecto de la seguridad del entorno y relación superficie-precio total.

**Página 4 — Oportunidades de inversión:** propiedades subvaluadas dentro de su propio barrio, detectadas combinando el percentil de precio por barrio y el score del modelo XGBoost. Incluye mapa de ubicación y tabla con las **top 20 oportunidades** rankeadas por brecha entre precio predicho y precio publicado.

**Página 5 — Segmentación K-Means:** los 20 segmentos con su perfil promedio (precio m², superficie, ambientes, antigüedad) y distribución geográfica, agrupados en las 5 categorías comerciales.

---

## 🏆 Top 20 Oportunidades de Inversión

La entrega final incluye un CSV con las **20 mejores oportunidades de compra** identificadas en el dataset completo: propiedades donde la brecha entre el precio predicho por XGBoost y el precio publicado es mayor, indicando una subvaluación significativa respecto a sus características objetivas.

---

## 📊 Conclusiones

- El barrio es el principal determinante de valor, confirmado de forma convergente por el mapa geográfico, el ANOVA y el clustering.
- Existen oportunidades reales: más del 10% del mercado aparece subvaluado según el modelo.
- Las oportunidades no se reparten parejo: se concentran en el corredor norte (Puerto Madero, Recoleta, Retiro, Palermo).
- El precio absoluto no alcanza para identificar oportunidades: lo que importa es el precio relativo dentro del propio barrio.
- Tres métodos distintos (estadístico, predictivo, no supervisado) convergieron en la misma conclusión.

## 📋 Recomendaciones para el cliente

- Priorizar los barrios con mayor proporción de oportunidades fuertes.
- Evaluar cada propiedad contra su propio barrio, no contra el promedio de CABA.
- Usar el modelo XGBoost como filtro inicial de búsqueda antes de visitar propiedades.
- Diferenciar la estrategia de negocio según el segmento de clustering (flipping aplica distinto en Premium que en Accesible).
- Adoptar el dashboard como herramienta de monitoreo continuo del mercado.

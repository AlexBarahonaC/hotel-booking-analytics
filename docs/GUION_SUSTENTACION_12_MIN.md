# Guion de sustentación — 12 minutos

## 0:00–1:00 · Problema y objetivo

“Buenos días. Nuestro proyecto analiza el conjunto Hotel Booking Demand. El problema principal es anticipar cancelaciones de reservas hoteleras, porque una habitación cancelada puede afectar la ocupación, la planificación y la estrategia de ingresos. Complementamos este problema con la estimación del ADR y con una segmentación de reservas.”

Mencionar: 119.390 registros originales, dos hoteles y periodo 2015–2017.

## 1:00–2:20 · Fase 1 — EDA

Explicar que se realizó análisis de calidad, faltantes, duplicados, variables con riesgo de fuga y asociaciones con cancelación. Destacar que desde la Fase 1 se definieron tres tareas: clasificación, regresión y clustering.

## 2:20–3:40 · Fase 2 — Preparación

Explicar: 119.193 registros finales; partición 70/15/15 agrupada por duplicados; transformaciones ajustadas solamente con entrenamiento; variables derivadas; encoding y escalamiento sin data leakage; modelos baseline como referencia.

## 3:40–6:00 · Fase 3 — Modelado

Clasificación: comparar baseline, Random Forest, XGBoost y MLP. Modelo final XGBoost. Resultado de prueba: ROC-AUC 0,9409, F1 0,8088, Accuracy 0,8624.

Regresión: comparar modelos lineales, Random Forest y XGBoost. Modelo final XGBoost. Resultado: R² 0,9180, RMSE 13,33 €, MAE 7,75 €.

Clustering: K-Means y jerárquico; solución desplegada con K=6. Explicar que cada clúster tiene perfil de riesgo y valor.

## 6:00–10:20 · Demostración en vivo del dashboard

1. Abrir “Resumen ejecutivo”. Señalar las dos filas de KPI: rendimiento de la solución (fija, conjunto de prueba) e indicadores de negocio. Filtrar por Resort Hotel para evidenciar que los indicadores y gráficos se actualizan.
2. Abrir “Predicción en vivo”. Ingresar una reserva y ejecutar. Explicar el puntaje de riesgo, el umbral fijado en validación y la tasa real observada en su banda.
3. En la misma pantalla, mostrar el ADR estimado, el segmento asignado y los gráficos de factores que explican cada resultado.
4. Abrir “Segmentación”. Ubicar la reserva consultada en el mapa PCA, resaltar un clúster, cambiar a 3D y leer los perfiles promedio.
5. Abrir “Modelos”. Mostrar la comparación, el resultado por cada 1.000 reservas (291 detectadas, 62 no detectadas, 75 falsas alarmas) y los límites del modelo.

## 10:20–11:20 · Rigor y reproducibilidad

Resaltar que el dashboard usa los modelos entrenados en la Fase 3, exportados a formato nativo sin reentrenar; que la prueba de integridad verifica que reproducen las métricas oficiales; que `agent` se trata como variable nominal, y que el repositorio fija las versiones y contiene datos, cuadernos, código y modelos.

## 11:20–12:00 · Cierre

“Como conclusión, el proyecto no se quedó en métricas de notebook. Integramos la clasificación, la regresión y la segmentación en una aplicación funcional que permite explorar los datos y realizar inferencia con los mismos modelos evaluados en la Fase 3. Esto permite apoyar decisiones de Revenue Management manteniendo trazabilidad metodológica y reproducibilidad.”

# Preguntas probables del docente

**¿Por qué XGBoost?** Porque fue el modelo con mejor equilibrio de desempeño en validación/prueba y capturó relaciones no lineales e interacciones que los modelos lineales no representaban igual de bien.

**¿Por qué el umbral no es 0,5?** Porque se seleccionó en validación. La prueba se utilizó una sola vez con las decisiones ya congeladas.

**¿Por qué el puntaje no es una probabilidad?** Porque el entrenamiento pondera la clase minoritaria (scale_pos_weight) y eso eleva los valores intermedios: entre 60 y 70 puntos cancela el 55 % en validación. El puntaje ordena bien el riesgo y la app muestra la tasa real de cada banda.

**¿Cumplieron todas las metas SMART?** ROC-AUC (0,941 frente a 0,85) y reducción del RMSE (68 % frente a 20 %), sí. La silueta no: 0,22 frente a 0,25; los clústeres son zonas de un continuo, pero son estables en prueba y separan tasas de cancelación de 12 % a 53 %.

**¿Funcionaría hoy?** No sin reentrenar: al evaluar en 2017 con datos previos, el ROC-AUC baja a 0,881 y el R² a 0,756.

**¿Cómo evitaron data leakage?** Las transformaciones que aprenden parámetros se ajustaron solo con entrenamiento; se excluyeron variables de desenlace y variables operativas identificadas como fuga; los duplicados se mantuvieron agrupados en la partición.

**¿Qué significa ROC-AUC 0,94?** Que el modelo tiene alta capacidad para ordenar una reserva cancelada por encima de una no cancelada en términos de riesgo; no implica 94 % de exactitud.

**¿Qué representa R² 0,918?** Que aproximadamente el 91,8 % de la variabilidad del ADR en el conjunto de prueba es explicada por el modelo bajo esta configuración y estas variables.

**¿Para qué sirve clustering si ya hay clasificación?** La clasificación predice un resultado individual; clustering descubre perfiles de reservas y facilita estrategias diferenciadas por comportamiento, riesgo y valor.

**¿El valor de reservas canceladas equivale a pérdida real?** No necesariamente. Es valor bruto reservado asociado a cancelaciones; una habitación puede revenderse. Se usa como aproximación de ingreso en riesgo, no como pérdida contable definitiva.

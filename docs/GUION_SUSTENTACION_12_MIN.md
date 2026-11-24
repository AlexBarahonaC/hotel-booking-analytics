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

1. Abrir “Resumen ejecutivo”. Mostrar filtros de fecha, hotel y segmento. Cambiar un filtro para evidenciar que KPI y gráficos se actualizan.
2. Abrir “Predicción en vivo → Riesgo de cancelación”. Ingresar una reserva y ejecutar. Explicar que la barra usa el umbral aprendido en validación, no un 0,5 arbitrario.
3. Abrir “Estimación de ADR”. Ejecutar una predicción y mostrar ADR y valor estimado de estancia.
4. Abrir “Segmentación”. Cambiar ejes 2D/3D; mostrar perfiles promedio; ingresar una nueva reserva y asignarla a un clúster.
5. Abrir “Modelos”. Mostrar métricas y una gráfica SHAP/codo-silueta como evidencia de interpretabilidad y selección.

## 10:20–11:20 · Rigor y reproducibilidad

Resaltar que los pipelines contienen el preprocesamiento junto con el modelo; `agent` se conserva como variable nominal; los artefactos fueron persistidos; se fija `scikit-learn==1.6.1`; el repositorio contiene datos, notebooks, src, dashboard, modelos y requirements.

## 11:20–12:00 · Cierre

“Como conclusión, el proyecto no se quedó en métricas de notebook. Integramos la clasificación, la regresión y la segmentación en una aplicación funcional que permite explorar los datos y realizar inferencia con los mismos modelos evaluados en la Fase 3. Esto permite apoyar decisiones de Revenue Management manteniendo trazabilidad metodológica y reproducibilidad.”

# Preguntas probables del docente

**¿Por qué XGBoost?** Porque fue el modelo con mejor equilibrio de desempeño en validación/prueba y capturó relaciones no lineales e interacciones que los modelos lineales no representaban igual de bien.

**¿Por qué el umbral no es 0,5?** Porque se seleccionó en validación. La prueba se utilizó una sola vez con las decisiones ya congeladas.

**¿Cómo evitaron data leakage?** Las transformaciones que aprenden parámetros se ajustaron solo con entrenamiento; se excluyeron variables de desenlace y variables operativas identificadas como fuga; los duplicados se mantuvieron agrupados en la partición.

**¿Qué significa ROC-AUC 0,94?** Que el modelo tiene alta capacidad para ordenar una reserva cancelada por encima de una no cancelada en términos de riesgo; no implica 94 % de exactitud.

**¿Qué representa R² 0,918?** Que aproximadamente el 91,8 % de la variabilidad del ADR en el conjunto de prueba es explicada por el modelo bajo esta configuración y estas variables.

**¿Para qué sirve clustering si ya hay clasificación?** La clasificación predice un resultado individual; clustering descubre perfiles de reservas y facilita estrategias diferenciadas por comportamiento, riesgo y valor.

**¿El valor de reservas canceladas equivale a pérdida real?** No necesariamente. Es valor bruto reservado asociado a cancelaciones; una habitación puede revenderse. Se usa como aproximación de ingreso en riesgo, no como pérdida contable definitiva.

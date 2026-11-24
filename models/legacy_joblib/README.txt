Estos son los pipelines .joblib originales exportados por la Fase 3.
Se conservan como evidencia/reproducibilidad, pero el dashboard NO los carga.
XGBoost documenta que pickle/joblib es un snapshot de memoria que puede no ser
portable entre versiones de Python. La Fase 4 usa xgb_clasificacion.ubj y
xgb_regresion.ubj, y reconstruye los preprocesadores con el conjunto train.

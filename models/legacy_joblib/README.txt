Pipelines .joblib originales exportados por la Fase 3 (Colab, 23-nov-2026).

En Windows, los dos pipelines con XGBoost fallan al cargar ("input stream corrupted"), tanto en
Python 3.12 como en 3.13. El modelo, sin embargo, esta completo dentro del archivo:
src/exportar_modelos_nativos.py lo extrae SIN reentrenar, comprueba que reproduce las metricas
oficiales y lo guarda en ../xgb_*.ubj y ../preproc_*.joblib, que son los que usa el dashboard.

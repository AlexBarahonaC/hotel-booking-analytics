# Hotel Booking Analytics — Fase 4

Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial — Universidad de La Salle.

**Integrantes:** Miguel Angel Ricardo y Jose Alexander Barahona Calderon  
**Asignatura:** Ciencia de Datos — Maestría en Inteligencia Artificial  
**Docente:** Fabián Camilo Castro Riveros  
**Dataset:** Hotel Booking Demand

## Ejecución rápida en Windows

1. Descomprime el ZIP completo en una carpeta.
2. Ten instalado **Python 3.13 o 3.12 de 64 bits**.
3. Haz doble clic en **`EJECUTAR_DASHBOARD.bat`**.
4. En la primera ejecución se crea `.venv`, se instalan las versiones exactas, se validan los modelos y se abre Streamlit.
5. Si el navegador no se abre solo, entra a `http://localhost:8501`.

No muevas archivos de las carpetas `data`, `models`, `src`, `figures` o `dashboard`.

## Qué incluye el dashboard

- **Filtros dinámicos:** rango de fechas, hotel, segmento de mercado, ADR y lead time.
- **KPI ejecutivos:** reservas, tasa de cancelación, ADR promedio, estancia media y valor asociado a reservas canceladas.
- **Analítica interactiva:** evolución temporal, cancelación por hotel, distribución de ADR y dispersión lead time–ADR.
- **Predicción de cancelación:** pipeline XGBoost real de Fase 3 con probabilidad y umbral aprendido en validación.
- **Predicción de ADR:** pipeline XGBoost de regresión con cálculo del valor estimado de la estancia.
- **Clustering:** K-Means con K=6, asignación de nuevas reservas, gráfico 2D/3D y perfiles promedio.
- **Comparación de modelos:** métricas de prueba y evidencias gráficas exportadas desde la Fase 3.
- **Trazabilidad:** resumen de las fases, estructura del repositorio y materiales originales.

## Resultados integrados

- Clasificación XGBoost: ROC-AUC test **0.9409**, F1 **0.8088**, Accuracy **0.8624**.
- Regresión XGBoost: R² test **0.9180**, RMSE **€13.3328**, MAE **€7.7484**.
- Clustering K-Means: **6 clústeres** con perfiles de riesgo y valor.

## Estructura

```text
Hotel_Booking_Fase4/
├── data/                     # df_train_final, df_val_final, df_test_final
├── models/                   # modelos XGBoost nativos, métricas, metadatos, clusters y MLP
├── notebooks/                # Colab de Fases 1, 2 y 3
├── src/                      # transformadores, carga de datos y servicios de modelos
├── dashboard/app.py          # aplicación web principal
├── figures/                  # gráficas exportadas de Fase 3
├── docs/                     # guía, informes previos, informe final y guion
├── source_materials/         # ZIP original exportado desde Colab
├── .streamlit/config.toml    # tema oscuro
├── requirements.txt          # dependencias exactas del dashboard
├── runtime.txt               # Python 3.13 o 3.12 para despliegue
├── EJECUTAR_DASHBOARD.bat    # Windows
└── EJECUTAR_DASHBOARD.sh     # Linux/macOS
```

## Ejecución manual

### Windows

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src\smoke_test.py
python -m streamlit run dashboard\app.py
```

### Linux/macOS

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/smoke_test.py
python -m streamlit run dashboard/app.py
```

## Publicar en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio público de GitHub.
2. En Streamlit Community Cloud crea una aplicación desde ese repositorio.
3. Selecciona `dashboard/app.py` como archivo principal.
4. Mantén `requirements.txt` y `runtime.txt` en la raíz.
5. Despliega. Los modelos y datos deben permanecer versionados en el repositorio.

> Compatibilidad Python 3.13: los `.joblib` originales de XGBoost se conservan únicamente como evidencia en `models/legacy_joblib/`. El dashboard usa modelos XGBoost nativos `.ubj` y reconstruye los preprocesadores con `df_train_final.csv`, evitando el problema de portabilidad de `pickle/joblib` entre versiones de Python. `scikit-learn==1.6.1` sigue fijado para reproducir el preprocesamiento original.

## Materiales preservados

El paquete conserva los tres notebooks, los tres informes, la guía oficial, todos los CSV finales, modelos XGBoost `.ubj`, los `.joblib` originales preservados, MLP `.pt`, metadatos, métricas y figuras que fueron suministrados para la Fase 4.

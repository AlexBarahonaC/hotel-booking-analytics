# Guía breve de despliegue

## 1. Crear el repositorio

Crea un repositorio público en GitHub y sube el contenido de la carpeta `Hotel_Booking_Fase4` manteniendo su estructura.

## 2. Verificar archivos indispensables

En la raíz deben estar `requirements.txt` y `runtime.txt`. El archivo principal es `dashboard/app.py`. Las carpetas `data`, `models`, `src` y `figures` deben estar presentes.

## 3. Streamlit Community Cloud

En Streamlit Community Cloud selecciona el repositorio, la rama principal y `dashboard/app.py`. La aplicación instalará las dependencias indicadas en `requirements.txt` y utilizará Python 3.12.

## 4. Validación antes de publicar

En local ejecuta:

```bash
python src/smoke_test.py
python -m streamlit run dashboard/app.py
```

Si el smoke test imprime `OK - datos y modelos cargados`, la estructura y los artefactos están accesibles.

## 5. Nota sobre tamaño

Los pipelines y datos procesados se incluyen para que la aplicación sea reproducible y no requiera reentrenamiento. Si GitHub rechaza algún archivo por límite de tamaño, utiliza Git LFS para ese archivo sin cambiar las rutas del proyecto.

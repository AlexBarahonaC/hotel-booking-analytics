# Guía de publicación

## 1. Repositorio público en GitHub

Crea un repositorio **público y vacío** (sin README) y, desde la carpeta del proyecto:

```bash
git init -b main
git add .
git commit -m "Hotel Booking Analytics: entrega de la Fase 4"
git remote add origin https://github.com/AlexBarahonaC/hotel-booking-analytics.git
git push -u origin main
```

El archivo más grande pesa unos 21 MB, por debajo del límite de 100 MB de GitHub: no hace falta Git LFS. Usa la consola y no la carga web, que limita cada archivo a 25 MB y maneja mal las carpetas grandes.

## 2. Streamlit Community Cloud

1. Entra a https://share.streamlit.io con tu cuenta de GitHub y elige **Create app**.
2. Selecciona el repositorio, la rama `main` y el archivo `dashboard/app.py`.
3. En **Advanced settings**, elige **Python 3.12** (o 3.13). La versión se define ahí.
4. Despliega. La primera instalación tarda unos minutos.

`requirements.txt` y `.streamlit/config.toml` deben quedar en la raíz del repositorio.

## 3. Después de publicar

- Abre la URL en una ventana de incógnito: debe cargar sin iniciar sesión.
- Ejecuta una predicción y cambia un filtro.
- Pega las dos URL en `README.md` y en el informe.
- El día de la sustentación, abre la app unos minutos antes: si estuvo inactiva puede tardar en despertar. Ten lista la versión local como respaldo.

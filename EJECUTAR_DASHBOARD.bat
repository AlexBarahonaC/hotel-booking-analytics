@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Hotel Booking Analytics - Fase 4

echo =====================================================
echo   HOTEL BOOKING ANALYTICS - FASE 4
echo   Compatible con Python 3.13 / 3.12 de 64 bits
echo =====================================================
echo.

set "PY="

REM Preferir Python 3.13 si esta instalado mediante Python Launcher
where py >nul 2>nul
if not errorlevel 1 (
    py -3.13 -c "import sys; assert sys.maxsize > 2**32" >nul 2>nul
    if not errorlevel 1 set "PY=py -3.13"
)

REM Como alternativa, usar Python 3.12
if not defined PY (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>nul
        if not errorlevel 1 set "PY=py -3.12"
    )
)

REM Ultima alternativa: python del PATH, siempre que sea 3.12 o 3.13 y 64 bits
if not defined PY (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; assert sys.version_info[:2] in [(3,12),(3,13)] and sys.maxsize > 2**32" >nul 2>nul
        if not errorlevel 1 set "PY=python"
    )
)

if not defined PY goto :python_error

echo Python detectado:
%PY% --version
echo.

REM Si existe un entorno virtual roto o creado con otra version, lo recreamos.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:2] in [(3,12),(3,13)]" >nul 2>nul
    if errorlevel 1 (
        echo Eliminando entorno virtual incompatible...
        rmdir /s /q ".venv"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creando entorno virtual...
    %PY% -m venv .venv
    if errorlevel 1 goto :python_error
) else (
    echo [1/4] Entorno virtual encontrado.
)

call ".venv\Scripts\activate.bat"

echo [2/4] Instalando/verificando dependencias...
python -m pip install --upgrade pip
if errorlevel 1 goto :install_error
python -m pip install -r requirements.txt
if errorlevel 1 goto :install_error

echo [3/4] Validando modelos y archivos...
python src\smoke_test.py
if errorlevel 1 goto :model_error

echo [4/4] Iniciando dashboard...
echo.
echo El navegador debe abrir: http://localhost:8501
echo Para cerrar el dashboard, vuelva a esta ventana y presione Ctrl+C.
echo.
python -m streamlit run dashboard\app.py
exit /b 0

:python_error
echo.
echo ERROR: No se encontro Python 3.13 o 3.12 de 64 bits.
echo Si ya tiene Python 3.13 instalado, compruebe en CMD: python --version
echo o: py -3.13 --version
echo.
pause
exit /b 1

:install_error
echo.
echo ERROR: No se pudieron instalar las dependencias.
echo Revise el mensaje mostrado arriba y su conexion a Internet.
echo.
pause
exit /b 1

:model_error
echo.
echo ERROR: La prueba de integridad de los modelos fallo.
echo Copie o tome captura del error mostrado arriba para revisarlo.
echo No borre ni mueva las carpetas data, models o src.
echo.
pause
exit /b 1

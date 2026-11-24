"""Carga de los datos depurados (Fase 2) y utilidades para la interfaz."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DTYPES = {"agent": str, "grupo_dup": str}

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Columnas que usa el dashboard. Las demás (llaves de control, desenlace) no se cargan.
COLUMNAS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_day_of_month", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "agent", "customer_type", "adr",
    "total_of_special_requests", "total_huespedes", "total_noches", "prop_fin_semana",
]
CATEGORICAS = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "agent", "customer_type",
]


def load_split(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"df_{name}_final.csv", dtype=DTYPES, usecols=columns)


def load_all_data() -> pd.DataFrame:
    """Une las tres particiones (solo con fines descriptivos) y agrega la fecha de llegada."""
    frames = [load_split(s, COLUMNAS).assign(split=s) for s in ("train", "val", "test")]
    out = pd.concat(frames, ignore_index=True)
    out["arrival_date"] = pd.to_datetime(
        dict(
            year=out["arrival_date_year"],
            month=out["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MONTH_ORDER)}),
            day=out["arrival_date_day_of_month"],
        )
    )
    out["valor_estancia"] = out["adr"] * out["total_noches"]
    for col in CATEGORICAS + ["split"]:              # reduce la memoria en el servidor
        out[col] = out[col].astype("category")
    return out


def numero_semana(fecha: date) -> int:
    """Semana del año con la convención del dataset: empieza en domingo y la semana 1 contiene el
    1 de enero (WEEKNUM de Excel). NO es la semana ISO, que difiere en el 54 % de las reservas."""
    dia_semana_1_enero = (date(fecha.year, 1, 1).weekday() + 1) % 7      # domingo = 0
    return (fecha.timetuple().tm_yday + dia_semana_1_enero - 1) // 7 + 1

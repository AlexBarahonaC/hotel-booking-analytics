"""Carga de los modelos de la Fase 3 (Colab, 23-nov-2026) e inferencia para el dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(ROOT / "src") not in sys.path:               # los .joblib referencian el módulo transformadores
    sys.path.insert(0, str(ROOT / "src"))

CAT_VARS = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "agent", "customer_type",
]

NOMBRES_CLUSTER = {
    0: "Último minuto",
    1: "Escapadas de fin de semana",
    2: "Estancias largas vacacionales",
    3: "Anticipadas entre semana",
    4: "Clientes comprometidos",
    5: "Familias y grupos premium",
}


def load_json(nombre: str):
    return json.loads((MODELS_DIR / nombre).read_text(encoding="utf-8"))


def load_metadata() -> dict:
    return load_json("metadatos_fase3.json")


def load_artifacts() -> dict:
    clf, reg = XGBClassifier(), XGBRegressor()
    clf.load_model(MODELS_DIR / "xgb_clasificacion.ubj")
    reg.load_model(MODELS_DIR / "xgb_regresion.ubj")
    return {
        "clasificacion": {"pre": joblib.load(MODELS_DIR / "preproc_clasificacion.joblib"), "model": clf},
        "regresion": {"pre": joblib.load(MODELS_DIR / "preproc_regresion.joblib"), "model": reg},
        "clustering": joblib.load(MODELS_DIR / "pipeline_clustering.joblib"),
    }


def _variable_original(columna: str) -> str:
    """'cat__country_PRT' -> 'country'; 'other_num__adults' -> 'adults'."""
    nombre = columna.split("__", 1)[-1]
    for cat in sorted(CAT_VARS, key=len, reverse=True):
        if nombre == cat or nombre.startswith(cat + "_"):
            return cat
    return nombre


def contribuciones(x: pd.DataFrame, model) -> pd.Series:
    """Valores SHAP de XGBoost (pred_contribs) de una fila, sumados por variable original."""
    contrib = model.get_booster().predict(xgb.DMatrix(x), pred_contribs=True)[0]
    serie = pd.Series(contrib[:-1], index=[_variable_original(c) for c in x.columns])
    serie = serie.groupby(level=0).sum()
    return serie.reindex(serie.abs().sort_values(ascending=False).index)


def predict_cancellation(row: pd.DataFrame, artifacts: dict, metadata: dict) -> dict:
    bundle = artifacts["clasificacion"]
    x = bundle["pre"].transform(row[metadata["clasificacion"]["variables"]])
    puntaje = float(bundle["model"].predict_proba(x)[0, 1])
    umbral = float(metadata["clasificacion"]["umbral"])
    return {"puntaje": puntaje, "umbral": umbral, "alerta": puntaje >= umbral,
            "factores": contribuciones(x, bundle["model"])}


def predict_adr(row: pd.DataFrame, artifacts: dict, metadata: dict) -> dict:
    bundle = artifacts["regresion"]
    x = bundle["pre"].transform(row[metadata["regresion"]["variables"]])
    return {"adr": max(0.0, float(bundle["model"].predict(x)[0])),
            "factores": contribuciones(x, bundle["model"])}


def assign_clusters(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Clúster (numeración de la Fase 3) y coordenadas PCA de cada reserva."""
    b = artifacts["clustering"]
    z = b["preprocesador"].transform(df[b["variables"]])
    lut = np.array([b["mapa_etiquetas"][i] for i in range(len(b["mapa_etiquetas"]))])
    pca = b["pca"].transform(z)
    return pd.DataFrame({"cluster": lut[b["kmeans"].predict(z)], "pca1": pca[:, 0], "pca2": pca[:, 1]},
                        index=df.index)


def tasa_observada(puntaje: float, confiabilidad: list[dict]) -> dict:
    """Banda de validación a la que pertenece el puntaje, con su tasa real de cancelación."""
    for banda in confiabilidad:
        if puntaje <= banda["hasta"]:
            return banda
    return confiabilidad[-1]

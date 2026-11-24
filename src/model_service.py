from __future__ import annotations

from pathlib import Path
import json
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transformadores import crear_preprocesador, crear_preprocesador_clustering  # noqa: E402

CAT_VARS = [
    "hotel", "arrival_date_month", "meal", "country", "market_segment",
    "distribution_channel", "reserved_room_type", "deposit_type", "agent",
    "customer_type",
]
DTYPES = {"agent": str, "grupo_dup": str}


def load_metadata() -> dict:
    with open(MODELS_DIR / "metadatos_fase3.json", encoding="utf-8") as fh:
        return json.load(fh)


def _load_train() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "df_train_final.csv", dtype=DTYPES)


def _build_supervised_artifacts(train: pd.DataFrame, metadata: dict):
    """Reconstruye los preprocesadores y carga XGBoost desde formato nativo estable.

    Los .joblib originales de Fase 3 contienen snapshots pickle de XGBoost y pueden
    fallar al cambiar de versión de Python. Los modelos .ubj se cargan con la API
    nativa de XGBoost y los preprocesadores se ajustan de nuevo SOLO con train,
    exactamente con las reglas de las Fases 2 y 3.
    """
    vars_c = metadata["clasificacion"]["variables"]
    vars_r = metadata["regresion"]["variables"]

    num_c = [c for c in vars_c if c not in CAT_VARS]
    cat_c = [c for c in vars_c if c in CAT_VARS]
    num_r = [c for c in vars_r if c not in CAT_VARS]
    cat_r = [c for c in vars_r if c in CAT_VARS]

    pre_c = crear_preprocesador(num_c, cat_c, es_regresion=False).fit(train[vars_c])
    pre_r = crear_preprocesador(num_r, cat_r, es_regresion=True).fit(train[vars_r])

    clf = XGBClassifier()
    clf.load_model(MODELS_DIR / "xgb_clasificacion.ubj")
    reg = XGBRegressor()
    reg.load_model(MODELS_DIR / "xgb_regresion.ubj")

    return {"pre": pre_c, "model": clf}, {"pre": pre_r, "model": reg}


def _build_clustering_artifact(train: pd.DataFrame, metadata: dict) -> dict:
    vars_clust = metadata["clustering"]["variables"]
    k = int(metadata["clustering"]["k"])
    pre = crear_preprocesador_clustering(vars_clust).fit(train[vars_clust])
    z = pre.transform(train[vars_clust])

    # Misma configuración utilizada en el Colab de la Fase 3.
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z)

    # Reordenamiento original: etiqueta 0 = clúster con menor ADR medio.
    adr_medio = pd.Series(train["adr"].to_numpy()).groupby(kmeans.labels_).mean().sort_values()
    mapping = {int(old): int(new) for new, old in enumerate(adr_medio.index)}

    pca = PCA(n_components=2, random_state=42).fit(z)
    return {
        "preprocesador": pre,
        "kmeans": kmeans,
        "pca": pca,
        "mapa_etiquetas": mapping,
        "variables": vars_clust,
    }


def load_artifacts() -> dict:
    metadata = load_metadata()
    train = _load_train()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf, reg = _build_supervised_artifacts(train, metadata)
        clu = _build_clustering_artifact(train, metadata)
    return {"clasificacion": clf, "regresion": reg, "clustering": clu}


def _coerce_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical = CAT_VARS
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].fillna("UNK").astype(str)
    if "agent" in df.columns:
        df["agent"] = df["agent"].replace({"<NA>": "0", "nan": "0"})
    return df


def predict_cancellation(row: pd.DataFrame, artifacts: dict, metadata: dict) -> dict:
    row = _coerce_categoricals(row)
    variables = metadata["clasificacion"]["variables"]
    bundle = artifacts["clasificacion"]
    x = bundle["pre"].transform(row[variables])
    proba = float(bundle["model"].predict_proba(x)[:, 1][0])
    threshold = float(metadata["clasificacion"]["umbral"])
    return {"probability": proba, "threshold": threshold, "prediction": int(proba >= threshold)}


def predict_adr(row: pd.DataFrame, artifacts: dict, metadata: dict) -> float:
    row = _coerce_categoricals(row)
    variables = metadata["regresion"]["variables"]
    bundle = artifacts["regresion"]
    x = bundle["pre"].transform(row[variables])
    return float(bundle["model"].predict(x)[0])


def predict_cluster(row: pd.DataFrame, artifacts: dict) -> dict:
    bundle = artifacts["clustering"]
    variables = bundle["variables"]
    z = bundle["preprocesador"].transform(row[variables])
    raw_cluster = int(bundle["kmeans"].predict(z)[0])
    mapping = bundle.get("mapa_etiquetas", {})
    mapped_cluster = mapping.get(raw_cluster, mapping.get(str(raw_cluster), raw_cluster))
    pca_coords = bundle["pca"].transform(z)[0]
    return {
        "cluster": int(mapped_cluster),
        "raw_cluster": raw_cluster,
        "pca1": float(pca_coords[0]),
        "pca2": float(pca_coords[1]),
    }

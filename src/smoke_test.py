"""Prueba de integridad: se ejecuta antes de abrir el dashboard.

Comprueba que los modelos cargados son los oficiales (reproducen sus métricas de prueba), que los
clústeres coinciden con los de la Fase 3 y que la semana calculada coincide con la del dataset.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, roc_auc_score

from data_service import load_split, numero_semana
from model_service import (MODELS_DIR, assign_clusters, load_artifacts, load_metadata, predict_adr,
                           predict_cancellation)


def verificar(nombre: str, obtenido: float, esperado: float) -> None:
    if abs(obtenido - esperado) > 1e-4:
        raise RuntimeError(f"{nombre}: se obtuvo {obtenido:.6f} y se esperaba {esperado:.6f}")
    print(f"  OK  {nombre}: {obtenido:.4f}")


def main() -> None:
    # Categorías de test no vistas en train: el codificador las agrupa como infrecuentes (por diseño)
    warnings.filterwarnings("ignore", message="Found unknown categories")
    meta, art = load_metadata(), load_artifacts()
    test, train = load_split("test"), load_split("train")

    c, oficial = art["clasificacion"], meta["clasificacion"]["metricas_test"]
    p = c["model"].predict_proba(c["pre"].transform(test[meta["clasificacion"]["variables"]]))[:, 1]
    verificar("ROC-AUC en prueba", roc_auc_score(test["is_canceled"], p), oficial["ROC-AUC"])
    verificar("F1 en prueba", f1_score(test["is_canceled"], p >= meta["clasificacion"]["umbral"]), oficial["F1"])

    r, e = art["regresion"], test[test["apto_regresion"] == 1]
    q = r["model"].predict(r["pre"].transform(e[meta["regresion"]["variables"]]))
    verificar("MAE del ADR en prueba (EUR)", mean_absolute_error(e["adr"], q),
              meta["regresion"]["metricas_test"]["MAE (€)"])

    guardados = pd.read_csv(MODELS_DIR / "clusters_train.csv").set_index("id_original")["cluster"]
    calculados = assign_clusters(train, art)["cluster"].to_numpy()
    verificar("Coincidencia de clusteres",
              float(np.mean(guardados.loc[train["id_original"]].to_numpy() == calculados)), 1.0)

    fechas = pd.to_datetime(test["arrival_date_year"].astype(str) + "-" + test["arrival_date_month"] + "-"
                            + test["arrival_date_day_of_month"].astype(str), format="%Y-%B-%d")
    semanas = fechas.map(lambda d: numero_semana(d.date()))
    verificar("Coincidencia de la semana", float((semanas == test["arrival_date_week_number"]).mean()), 1.0)

    fila = test.iloc[[0]]
    if not 0 <= predict_cancellation(fila, art, meta)["puntaje"] <= 1:
        raise RuntimeError("Puntaje de cancelacion fuera de rango")
    if not predict_adr(fila, art, meta)["adr"] > 0:
        raise RuntimeError("ADR fuera de rango")
    print("OK - datos y modelos oficiales cargados")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:                         # el .bat detiene el arranque si el código no es 0
        print(f"FALLO - {type(exc).__name__}: {exc}")
        sys.exit(1)

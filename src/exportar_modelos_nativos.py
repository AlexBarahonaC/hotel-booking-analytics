"""Exporta los modelos ORIGINALES de la Fase 3 (Colab, 23-nov-2026) a formatos portables.

Se ejecuta una sola vez, al construir el repositorio; el dashboard solo usa sus resultados.

Los pipelines .joblib de Colab fallan al cargar en Windows ("input stream corrupted"), tanto en
Python 3.12 como en 3.13. El modelo XGBoost completo, sin embargo, está intacto dentro del archivo:
al serializar, XGBoost guarda un diccionario UBJSON con las claves "Config" y "Model". Este script
lee la sección "Model" y la carga con la API nativa, SIN reentrenar nada. La prueba de que es el
mismo modelo es que reproduce las métricas de prueba de metadatos_fase3.json.

Uso:  python src/exportar_modelos_nativos.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost.core as xgb_core
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score, recall_score,
                             roc_auc_score, silhouette_score)
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = ROOT / "models"
LEGACY = MODELS / "legacy_joblib"
DTYPES = {"agent": str, "grupo_dup": str}
TOL = 1e-6


def extraer(nombre: str, clase):
    """Devuelve (preprocesador, modelo XGBoost) a partir del .joblib original."""
    capturado: list[bytes] = []
    original_setstate = xgb_core.Booster.__setstate__
    original_del = xgb_core.Booster.__del__

    def setstate_sin_handle(self, state):
        state = dict(state)
        capturado.append(bytes(state.pop("handle")))
        self.__dict__.update(state)

    xgb_core.Booster.__setstate__ = setstate_sin_handle
    xgb_core.Booster.__del__ = lambda self: None   # el objeto vacío no tiene memoria nativa que liberar
    try:
        pipeline = joblib.load(LEGACY / f"pipeline_{nombre}.joblib")
        preprocesador = pipeline.named_steps["pre"]
        del pipeline
    finally:
        xgb_core.Booster.__setstate__ = original_setstate
        xgb_core.Booster.__del__ = original_del

    if len(capturado) != 1:
        raise RuntimeError(f"Se esperaba un único modelo en {nombre} y se hallaron {len(capturado)}")
    bruto = capturado[0]
    marca = b"\x05Model{"                          # clave UBJSON "Model" seguida del objeto
    inicio = bruto.find(marca)
    if inicio < 0 or not bruto.endswith(b"}}"):
        raise RuntimeError(f"Formato inesperado en {nombre}")
    modelo = clase()
    modelo.load_model(bytearray(bruto[inicio + len(marca) - 1:-1]))
    return preprocesador, modelo


def comprobar(nombre: str, calculado: dict, oficial: dict) -> None:
    for clave, valor in calculado.items():
        if abs(valor - oficial[clave]) > TOL:
            raise AssertionError(f"{nombre} - {clave}: {valor} != {oficial[clave]} (oficial)")
        print(f"  {nombre:<14} {clave:<12} {valor:.6f} = oficial")


def main() -> None:
    warnings.simplefilter("ignore")
    meta = json.loads((MODELS / "metadatos_fase3.json").read_text(encoding="utf-8"))
    train, val, test = (pd.read_csv(ROOT / "data" / f"df_{s}_final.csv", dtype=DTYPES)
                        for s in ("train", "val", "test"))

    # --- Clasificación ---
    pre_c, clf = extraer("clasificacion", XGBClassifier)
    v, u = meta["clasificacion"]["variables"], meta["clasificacion"]["umbral"]
    p_test = clf.predict_proba(pre_c.transform(test[v]))[:, 1]
    p_val = clf.predict_proba(pre_c.transform(val[v]))[:, 1]
    y, pred = test["is_canceled"], p_test >= u
    comprobar("clasificacion",
              {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred),
               "Recall": recall_score(y, pred), "F1": f1_score(y, pred),
               "ROC-AUC": roc_auc_score(y, p_test)},
              meta["clasificacion"]["metricas_test"])
    pr, rc, th = precision_recall_curve(val["is_canceled"], p_val)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    comprobar("validacion", {"umbral": float(th[f1[:-1].argmax()])}, {"umbral": u})

    # --- Regresión ---
    pre_r, reg = extraer("regresion", XGBRegressor)
    e = test[test["apto_regresion"] == 1]
    q = reg.predict(pre_r.transform(e[meta["regresion"]["variables"]]))
    comprobar("regresion",
              {"MAE (€)": mean_absolute_error(e["adr"], q),
               "RMSE (€)": float(np.sqrt(mean_squared_error(e["adr"], q))),
               "R2": r2_score(e["adr"], q)},
              meta["regresion"]["metricas_test"])

    # --- Clustering (su .joblib sí carga; aquí solo se verifica y se mide la silueta) ---
    clu = joblib.load(LEGACY / "pipeline_clustering.joblib")
    lut = np.array([clu["mapa_etiquetas"][i] for i in range(meta["clustering"]["k"])])
    z_train = clu["preprocesador"].transform(train[clu["variables"]])
    z_test = clu["preprocesador"].transform(test[clu["variables"]])
    et_train, et_test = lut[clu["kmeans"].predict(z_train)], lut[clu["kmeans"].predict(z_test)]
    guardadas = pd.read_csv(MODELS / "clusters_train.csv").set_index("id_original")["cluster"]
    if not (guardadas.loc[train["id_original"]].to_numpy() == et_train).all():
        raise AssertionError("Las etiquetas de K-Means no coinciden con clusters_train.csv")
    idx = np.random.default_rng(42).choice(len(z_train), size=15_000, replace=False)  # muestra de la Fase 3
    sil_train = float(silhouette_score(z_train[idx], et_train[idx]))
    sil_test = float(silhouette_score(z_test, et_test, sample_size=10_000, random_state=42))
    print(f"  clustering     silueta      train {sil_train:.3f} | test {sil_test:.3f}")

    # --- Confiabilidad en validación: qué tasa real de cancelación corresponde a cada puntaje ---
    bandas = pd.cut(p_val, np.linspace(0, 1, 11), include_lowest=True, labels=False)
    conf = (pd.DataFrame({"banda": bandas, "p": p_val.astype(float), "y": val["is_canceled"].to_numpy()})
            .groupby("banda").agg(n=("y", "size"), puntaje_medio=("p", "mean"), tasa_observada=("y", "mean")))
    confiabilidad = [{"desde": int(b) / 10, "hasta": (int(b) + 1) / 10, "n": int(f.n),
                      "puntaje_medio": float(f.puntaje_medio), "tasa_observada": float(f.tasa_observada)}
                     for b, f in conf.iterrows()]

    # --- Guardado ---
    clf.save_model(MODELS / "xgb_clasificacion.ubj")
    reg.save_model(MODELS / "xgb_regresion.ubj")
    joblib.dump(pre_c, MODELS / "preproc_clasificacion.joblib", compress=3)
    joblib.dump(pre_r, MODELS / "preproc_regresion.joblib", compress=3)
    joblib.dump(clu, MODELS / "pipeline_clustering.joblib", compress=3)
    (MODELS / "metricas_clustering.json").write_text(
        json.dumps({"silueta_train": sil_train, "silueta_test": sil_test}, indent=2), encoding="utf-8")
    (MODELS / "confiabilidad_validacion.json").write_text(
        json.dumps(confiabilidad, indent=2), encoding="utf-8")
    print("Artefactos portables guardados en models/")


if __name__ == "__main__":
    main()

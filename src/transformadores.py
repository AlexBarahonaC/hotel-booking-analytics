
# Transformadores compartidos por las Fases 2, 3 y 4 (código idéntico al de la Fase 2).
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


class Winsorizer(BaseEstimator, TransformerMixin):
    # Winsorización superior: recorta cada columna en su percentil q, aprendido SOLO en fit (train).
    def __init__(self, q=0.99):
        self.q = q

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.upper_bounds_ = X_df.quantile(self.q).to_dict()
        self.columnas_ = np.asarray(X_df.columns.astype(str), dtype=object)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, limite in self.upper_bounds_.items():
            X_df[col] = X_df[col].clip(upper=limite)
        return X_df.values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.columnas_
        return np.asarray(input_features, dtype=object)


def crear_preprocesador(num_vars, cat_vars, es_regresion=False):
    # ColumnTransformer de la Fase 2: log1p+escala en lead_time, winsorización p99+escala en
    # adr/total_noches, StandardScaler en el resto y One-Hot con agrupación de categorías raras.
    lead_time_t = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                            ("scaler", StandardScaler())])
    otras_num = [v for v in num_vars if v not in ["lead_time", "total_noches", "adr"]]
    trs = [("lead_time_t", lead_time_t, ["lead_time"])]
    if es_regresion:
        trs.append(("noches_t", Pipeline([("wins", Winsorizer(q=0.99)), ("sc", StandardScaler())]),
                    ["total_noches"]))
    else:
        trs.append(("adr_noches_t", Pipeline([("wins", Winsorizer(q=0.99)), ("sc", StandardScaler())]),
                    ["adr", "total_noches"]))
    trs.append(("other_num", StandardScaler(), otras_num))
    cat_ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                            sparse_output=False, drop="if_binary")
    ct = ColumnTransformer(transformers=[*trs, ("cat", cat_ohe, cat_vars)])
    ct.set_output(transform="pandas")
    return ct


def crear_preprocesador_clustering(vars_clust):
    # Preprocesador del clustering: log1p en lead_time, winsorización p99 en adr y total_noches
    # y estandarización final de todas las variables (ajustado solo en train).
    resto = [v for v in vars_clust if v not in ["lead_time", "adr", "total_noches"]]
    ct = ColumnTransformer([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ["lead_time"]),
        ("wins", Winsorizer(q=0.99), ["adr", "total_noches"]),
        ("resto", "passthrough", resto),
    ], verbose_feature_names_out=False)
    return Pipeline([("ct", ct), ("sc", StandardScaler())])

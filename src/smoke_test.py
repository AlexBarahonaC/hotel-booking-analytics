"""Prueba mínima de integridad de datos y modelos antes de lanzar el dashboard."""
from data_service import load_split
from model_service import load_artifacts, load_metadata, predict_cancellation, predict_adr, predict_cluster


def main():
    df = load_split("test")
    if df.empty:
        raise RuntimeError("df_test_final.csv está vacío")
    metadata = load_metadata()
    artifacts = load_artifacts()
    row = df.iloc[[0]].copy()

    c = predict_cancellation(row, artifacts, metadata)
    r = predict_adr(row, artifacts, metadata)
    cl = predict_cluster(row[metadata["clustering"]["variables"]], artifacts)

    if not 0 <= c["probability"] <= 1:
        raise RuntimeError("Probabilidad de clasificación fuera de rango")
    if not (r == r):
        raise RuntimeError("Predicción ADR inválida")
    if cl["cluster"] not in range(int(metadata["clustering"]["k"])):
        raise RuntimeError("Clúster fuera de rango")

    print("OK - datos y modelos cargados")
    print(f"Probabilidad cancelación: {c['probability']:.6f}")
    print(f"ADR estimado: {r:.4f}")
    print(f"Clúster: {cl['cluster']}")


if __name__ == "__main__":
    main()

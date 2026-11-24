from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"df_{name}_final.csv"
    return pd.read_csv(path, dtype={"agent": "string", "country": "string"})


def load_all_data() -> pd.DataFrame:
    frames = []
    for split in ("train", "val", "test"):
        df = load_split(split)
        df["split"] = split
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["agent"] = out["agent"].fillna("0").astype(str)
    out["country"] = out["country"].fillna("UNK").astype(str)
    month_map = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
    out["arrival_month_num"] = out["arrival_date_month"].map(month_map)
    out["arrival_date"] = pd.to_datetime(
        dict(
            year=out["arrival_date_year"],
            month=out["arrival_month_num"],
            day=out["arrival_date_day_of_month"],
        ),
        errors="coerce",
    )
    out["valor_estancia"] = out["adr"].clip(lower=0) * out["total_noches"].clip(lower=0)
    return out


def category_options(df: pd.DataFrame, column: str) -> list[str]:
    values = df[column].dropna().astype(str).unique().tolist()
    return sorted(values)


def mode_value(df: pd.DataFrame, column: str, fallback=None):
    mode = df[column].dropna().mode()
    if mode.empty:
        return fallback
    return mode.iloc[0]

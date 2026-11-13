from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", nrows=nrows)
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows)
    if suffix == ".parquet":
        if nrows is None:
            return pd.read_parquet(path)
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        batches = pf.iter_batches(batch_size=max(int(nrows), 1))
        try:
            batch = next(batches)
        except StopIteration:
            return pd.DataFrame()
        return batch.to_pandas().head(nrows)
    raise ValueError(f"Unsupported table: {path}")


def normalize_name(x: str) -> str:
    return "".join(ch.lower() for ch in str(x) if ch.isalnum())


def find_column(columns: Iterable[str], candidates: Iterable[str], required: bool = True) -> str | None:
    cols = list(columns)
    norm = {normalize_name(c): c for c in cols}
    for cand in candidates:
        n = normalize_name(cand)
        if n in norm:
            return norm[n]
    # tolerant suffix/contains matching, but only if unique
    for cand in candidates:
        n = normalize_name(cand)
        hits = [c for c in cols if n and (normalize_name(c).endswith(n) or n in normalize_name(c))]
        if len(hits) == 1:
            return hits[0]
    if required:
        raise KeyError(f"Could not match candidates {list(candidates)} among columns {cols}")
    return None


def detect_array_column(df: pd.DataFrame, exclude: set[str]) -> str | None:
    for c in df.columns:
        if c in exclude:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        v = s.iloc[0]
        if isinstance(v, (list, tuple, np.ndarray)):
            return c
    return None


def table_profile(path: str | Path, nrows: int = 5) -> dict:
    path = Path(path)
    try:
        df = read_table(path, nrows=nrows)
        preview = {}
        for c in df.columns:
            vals = []
            for v in df[c].head(3):
                if isinstance(v, np.ndarray):
                    vals.append({"array_shape": list(v.shape), "dtype": str(v.dtype)})
                elif isinstance(v, (list, tuple)):
                    arr = np.asarray(v)
                    vals.append({"array_shape": list(arr.shape), "dtype": str(arr.dtype)})
                else:
                    vals.append(str(v)[:120])
            preview[c] = vals
        return {
            "path": str(path),
            "exists": True,
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "preview": preview,
        }
    except Exception as e:
        return {"path": str(path), "exists": path.exists(), "error": repr(e)}


def profile_json_schema(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            keys = list(obj.keys())[:100]
        else:
            keys = []
        return {"path": str(path), "exists": True, "top_level_type": type(obj).__name__, "keys": keys}
    except Exception as e:
        return {"path": str(path), "exists": True, "error": repr(e)}

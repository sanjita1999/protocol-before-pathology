from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_seed(base: int, *parts: Any) -> int:
    payload = "|".join([str(base), *map(str, parts)]).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_sha256(df: pd.DataFrame) -> str:
    data = df.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def json_dump(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def runtime_manifest(config_path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    import importlib.metadata as md
    versions = {}
    for label, pkg in [("numpy","numpy"),("pandas","pandas"),("scikit_learn","scikit-learn"),("scipy","scipy"),("pyarrow","pyarrow"),("pyyaml","PyYAML")]:
        try:
            versions[label] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[label] = None
    m = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "git_commit": git_commit(Path(config_path).resolve().parent.parent),
        "versions": versions,
    }
    if extra:
        m.update(extra)
    return m


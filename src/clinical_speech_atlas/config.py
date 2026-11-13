from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    if path.resolve().parent.name == "config":
        project_root = path.resolve().parent.parent
    elif (Path.cwd() / "config" / "task_families.yaml").exists():
        project_root = Path.cwd().resolve()
    else:
        project_root = path.resolve().parent
    cfg["_project_root"] = str(project_root)
    # Environment variables in paths (e.g. ${B2AI_ROOT}); relative covariate files resolve from the project root.
    cfg["data"]["root"] = os.path.expandvars(cfg["data"]["root"])
    mf = (cfg.get("covariates") or {}).get("metadata_file")
    if mf:
        mf = Path(os.path.expandvars(mf))
        cfg["covariates"]["metadata_file"] = str(mf if mf.is_absolute() else project_root / mf)
    return cfg


def resolve_data_path(cfg: dict[str, Any], rel: str) -> Path:
    return Path(cfg["data"]["root"]).expanduser().resolve() / rel


def output_dir(cfg: dict[str, Any], kind: str) -> Path:
    root = Path(cfg["_project_root"]) / cfg["project"].get("output_root", "outputs")
    return root / kind


def task_map_path(cfg: dict[str, Any]) -> Path:
    return Path(cfg["_project_root"]) / "config" / "task_families.yaml"

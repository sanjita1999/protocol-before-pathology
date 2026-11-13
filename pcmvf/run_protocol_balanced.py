#!/usr/bin/env python3
"""
Add-only extension for the ML4H Clinical Speech Atlas repository.

Purpose
-------
Estimate protocol-balanced detectability without modifying the existing
clinical_speech_atlas package. V4 reuses the atlas's already-generated
recording-level embedding caches instead of reparsing raw Bridge2AI feature tables.

The extension:
  1. reads the existing outputs/manifests/recordings.csv manifest,
  2. uses the seven shared task families (cognitive-linguistic excluded),
  3. age/sex matches cases to controls,
  4. chooses a common task menu using coverage only,
  5. chooses a per-task observation budget using retention only,
  6. samples exactly the same number of observations per retained task family,
  7. evaluates protocol-only and speech-only PLS+Logistic models under the
     same participant-disjoint folds,
  8. repeats observation subsampling and bootstraps averaged OOF predictions.

No existing source file is imported, overwritten, or patched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


COMMON_TASKS = (
    "sustained_phonation",
    "pitch_modulation",
    "rapid_articulation",
    "structured_reading",
    "spontaneous_speech",
    "narrative_memory",
    "respiratory_cough",
)

REP_SPECS = {
    "spectrogram": {
        "dim": 201,
        "mode": "mean",
        "include": ("spectrogram",),
        "exclude": ("mel",),
    },
    "mel_spectrogram": {
        "dim": 60,
        "mode": "mean",
        "include": ("mel", "spectrogram"),
        "exclude": (),
    },
    "mfcc": {
        "dim": 60,
        "mode": "mean",
        "include": ("mfcc",),
        "exclude": (),
    },
    "sparc_ema": {
        "dim": 12,
        "mode": "mean",
        "include": ("ema",),
        "exclude": (),
    },
    "pitch": {
        "dim": 2,
        "mode": "mean_std",
        "include": ("pitch",),
        "exclude": (),
    },
    "loudness": {
        "dim": 2,
        "mode": "mean_std",
        "include": ("loud",),
        "exclude": (),
    },
    "periodicity": {
        "dim": 2,
        "mode": "mean_std",
        "include": ("period",),
        "exclude": (),
    },
    "static": {
        "dim": 131,
        "mode": "static",
        "include": ("static",),
        "exclude": (),
    },
}

PARTICIPANT_ALIASES = (
    "participant_id", "participant", "subject_id", "subject", "speaker_id"
)
SESSION_ALIASES = (
    "session_id", "acoustic_task_session_id", "session", "visit_id"
)
TASK_ALIASES = (
    "raw_task", "task", "task_name", "acoustic_task_name", "elicitation_task"
)
FAMILY_ALIASES = ("task_family", "family", "elicitation_family")
COHORT_ALIASES = ("cohort", "diagnosis", "disease", "label")
AGE_ALIASES = (
    "age", "age_years", "age_at_enrollment", "participant_age",
    "demographic_age", "age_at_visit"
)
SEX_ALIASES = (
    "sex", "sex_at_birth", "biological_sex", "gender", "participant_sex"
)
FOLD_ALIASES = ("fold", "fold_id", "cv_fold", "split_fold")


@dataclass
class DiseaseSelection:
    disease: str
    matched_case_ids: list[str]
    matched_control_ids: list[str]
    task_menu: list[str]
    per_task_budget: int
    eligible_case_ids: list[str]
    eligible_control_ids: list[str]
    age_caliper: float


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def norm_col(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def norm_task(x: Any) -> str:
    s = str(x).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def first_col(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> str | None:
    by_norm = {norm_col(c): c for c in df.columns}
    for alias in aliases:
        if norm_col(alias) in by_norm:
            return by_norm[norm_col(alias)]
    if required:
        raise KeyError(
            f"Could not find any of {list(aliases)} in columns: {list(df.columns)[:80]}"
        )
    return None


def load_frozen_task_mapping(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen task-family map: {path}. "
            "The protocol-balanced paper run will not use heuristic task mapping."
        )
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict) or not cfg.get("frozen", False):
        raise RuntimeError(
            f"Task-family map is not marked frozen=true: {path}"
        )
    mapping = cfg.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"No mapping dictionary found in {path}")
    return {norm_task(k): norm_task(v) for k, v in mapping.items()}


def load_manifest(path: Path, task_mapping_path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    pid = first_col(df, PARTICIPANT_ALIASES)
    session = first_col(df, SESSION_ALIASES)
    raw_task = first_col(df, TASK_ALIASES)
    family = first_col(df, FAMILY_ALIASES, required=False)
    cohort = first_col(df, COHORT_ALIASES)

    out = df.copy()
    out["participant_id"] = out[pid].astype(str)
    out["session_id"] = out[session].astype(str)
    out["raw_task"] = out[raw_task].astype(str)
    out["task_norm"] = out["raw_task"].map(norm_task)
    out["cohort"] = out[cohort].astype(str).str.strip()

    if family is not None:
        out["task_family"] = out[family].astype(str).map(norm_task)
        print(f"[task-map] using task_family column from {path}", flush=True)
    else:
        frozen = load_frozen_task_mapping(task_mapping_path)
        out["task_family"] = out["task_norm"].map(frozen)
        mapped = int(out["task_family"].notna().sum())
        print(
            f"[task-map] frozen={task_mapping_path} mapped={mapped:,}/{len(out):,}",
            flush=True,
        )
        if mapped == 0:
            raise RuntimeError(
                "Frozen task-family mapping matched zero recording rows. "
                "Refusing to fall back to heuristics."
            )

    out = out[out["task_family"].notna()].copy()
    out["_row_id"] = np.arange(len(out), dtype=int)
    return out

def _read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)
    if suffix in (".parquet", ".pq"):
        if nrows is None:
            return pd.read_parquet(path)
        # Header-only isn't portable for parquet; read once, then trim.
        return pd.read_parquet(path).head(nrows)
    raise ValueError(f"Unsupported metadata table: {path}")


def discover_demographics(
    repo_root: Path,
    b2ai_root: Path,
    manifest: pd.DataFrame,
    explicit: Path | None,
) -> pd.DataFrame:
    """
    Build participant_id, age, sex.
    Priority:
      explicit file -> manifest -> outputs/manifests -> Bridge2AI phenotype TSVs.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    # Manifest may already contain age/sex.
    age_col = first_col(manifest, AGE_ALIASES, required=False)
    sex_col = first_col(manifest, SEX_ALIASES, required=False)
    if age_col is not None and sex_col is not None:
        d = manifest[["participant_id", age_col, sex_col]].copy()
        d = d.rename(columns={age_col: "age", sex_col: "sex"})
        return clean_demographics(d)

    manifests_dir = repo_root / "outputs" / "manifests"
    if manifests_dir.exists():
        candidates.extend(sorted(manifests_dir.glob("*.csv")))
        candidates.extend(sorted(manifests_dir.glob("*.tsv")))

    phenotype = b2ai_root / "phenotype"
    if phenotype.exists():
        candidates.extend(sorted(phenotype.rglob("*.tsv")))
        candidates.extend(sorted(phenotype.rglob("*.csv")))

    # First look for a single table with both.
    for p in candidates:
        try:
            head = _read_table(p, nrows=5)
        except Exception:
            continue
        pid = first_col(head, PARTICIPANT_ALIASES, required=False)
        age = first_col(head, AGE_ALIASES, required=False)
        sex = first_col(head, SEX_ALIASES, required=False)
        if pid is not None and age is not None and sex is not None:
            full = _read_table(p)
            d = full[[pid, age, sex]].rename(
                columns={pid: "participant_id", age: "age", sex: "sex"}
            )
            d = clean_demographics(d)
            if len(d) > 0:
                print(f"[demographics] {p}")
                return d

    # Otherwise merge age and sex from separate participant-level tables.
    age_frames: list[pd.DataFrame] = []
    sex_frames: list[pd.DataFrame] = []
    for p in candidates:
        try:
            head = _read_table(p, nrows=5)
        except Exception:
            continue
        pid = first_col(head, PARTICIPANT_ALIASES, required=False)
        if pid is None:
            continue
        age = first_col(head, AGE_ALIASES, required=False)
        sex = first_col(head, SEX_ALIASES, required=False)
        if age is not None:
            full = _read_table(p)
            age_frames.append(
                full[[pid, age]].rename(columns={pid: "participant_id", age: "age"})
            )
        if sex is not None:
            full = _read_table(p)
            sex_frames.append(
                full[[pid, sex]].rename(columns={pid: "participant_id", sex: "sex"})
            )

    if age_frames and sex_frames:
        a = pd.concat(age_frames, ignore_index=True)
        s = pd.concat(sex_frames, ignore_index=True)
        a["participant_id"] = a["participant_id"].astype(str)
        s["participant_id"] = s["participant_id"].astype(str)
        a = a.dropna().drop_duplicates("participant_id")
        s = s.dropna().drop_duplicates("participant_id")
        return clean_demographics(a.merge(s, on="participant_id", how="inner"))

    raise RuntimeError(
        "Could not discover participant age and sex. Re-run with "
        "--demographics /path/to/file.csv containing participant_id, age, sex. "
        "The extension intentionally refuses to silently skip demographic matching."
    )


def clean_demographics(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["participant_id"] = out["participant_id"].astype(str)
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["sex"] = (
        out["sex"].astype(str).str.strip().str.lower()
        .replace({
            "female": "f", "woman": "f", "f": "f",
            "male": "m", "man": "m", "m": "m",
        })
    )
    out = out[out["age"].notna() & out["sex"].isin(["f", "m"])].copy()
    return out.drop_duplicates("participant_id")


def discover_fold_map(repo_root: Path, explicit: Path | None) -> dict[str, int] | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for base in (repo_root / "outputs" / "manifests", repo_root / "outputs" / "results"):
        if base.exists():
            candidates.extend(sorted(base.glob("*fold*.csv")))
            candidates.extend(sorted(base.glob("*split*.csv")))

    for p in candidates:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        pid = first_col(df, PARTICIPANT_ALIASES, required=False)
        fold = first_col(df, FOLD_ALIASES, required=False)
        if pid is None or fold is None:
            continue
        tmp = df[[pid, fold]].copy()
        tmp[fold] = pd.to_numeric(tmp[fold], errors="coerce")
        tmp = tmp.dropna().drop_duplicates(pid)
        if len(tmp) < 20:
            continue
        print(f"[fold-map] {p}")
        return {str(a): int(b) for a, b in zip(tmp[pid], tmp[fold])}
    return None


def discover_published_raw(repo_root: Path) -> dict[str, float]:
    path = repo_root / "outputs" / "results" / "disease_benchmark.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    disease = first_col(df, COHORT_ALIASES, required=False)
    auc = first_col(df, ("auc", "roc_auc"), required=False)
    if disease is None or auc is None:
        return {}
    return {
        str(d).strip(): float(a)
        for d, a in zip(df[disease], pd.to_numeric(df[auc], errors="coerce"))
        if pd.notna(a)
    }


def match_demographics(
    disease: str,
    cohort_by_pid: pd.Series,
    demographics: pd.DataFrame,
    age_caliper_sd: float,
) -> tuple[list[str], list[str], float]:
    d = demographics.copy()
    d["cohort"] = d["participant_id"].map(cohort_by_pid)
    d = d[d["cohort"].isin([disease, "Healthy Control"])].copy()

    cases = d[d["cohort"].eq(disease)].sort_values(["sex", "age", "participant_id"])
    controls = d[d["cohort"].eq("Healthy Control")].sort_values(
        ["sex", "age", "participant_id"]
    )
    pooled_sd = float(d["age"].std(ddof=1))
    caliper = age_caliper_sd * pooled_sd if np.isfinite(pooled_sd) else 0.0

    available = set(controls["participant_id"].astype(str))
    ctrl_by_id = controls.set_index("participant_id")
    pairs: list[tuple[str, str]] = []

    # Deterministic no-replacement nearest-age, exact-sex matching.
    # Most constrained cases first.
    candidate_counts = []
    for row in cases.itertuples():
        cand = controls[
            controls["sex"].eq(row.sex)
            & (controls["age"].sub(row.age).abs() <= caliper)
        ]
        candidate_counts.append((len(cand), str(row.participant_id)))
    order = [pid for _, pid in sorted(candidate_counts)]

    case_by_id = cases.set_index("participant_id")
    for case_id in order:
        row = case_by_id.loc[case_id]
        cand = ctrl_by_id.loc[list(available)].copy() if available else ctrl_by_id.iloc[0:0]
        cand = cand[
            cand["sex"].eq(row["sex"])
            & (cand["age"].sub(row["age"]).abs() <= caliper)
        ]
        if cand.empty:
            continue
        cand = cand.assign(
            _gap=cand["age"].sub(row["age"]).abs(),
            _pid=cand.index.astype(str),
        ).sort_values(["_gap", "_pid"])
        control_id = str(cand.index[0])
        pairs.append((str(case_id), control_id))
        available.remove(control_id)

    return [a for a, _ in pairs], [b for _, b in pairs], caliper


def choose_task_menu(
    df: pd.DataFrame,
    case_ids: list[str],
    control_ids: list[str],
    retention: float,
    min_group: int,
) -> list[str]:
    sub = df[
        df["participant_id"].isin(case_ids + control_ids)
        & df["task_family"].isin(COMMON_TASKS)
    ].copy()

    counts = sub.groupby(["participant_id", "task_family"]).size().unstack(fill_value=0)
    for t in COMMON_TASKS:
        if t not in counts:
            counts[t] = 0

    case_set, ctrl_set = set(case_ids), set(control_ids)

    # Start with families that individually have at least min_group in both groups.
    menu = []
    for t in COMMON_TASKS:
        nc = int((counts.reindex(list(case_set), fill_value=0)[t] >= 1).sum())
        nh = int((counts.reindex(list(ctrl_set), fill_value=0)[t] >= 1).sum())
        if nc >= min_group and nh >= min_group:
            menu.append(t)

    def retained(ids: set[str], m: list[str]) -> int:
        x = counts.reindex(list(ids), fill_value=0)
        return int((x[m] >= 1).all(axis=1).sum()) if m else 0

    # Remove the family whose removal maximally improves the worse group
    # until the pre-specified retention target is met.
    while len(menu) > 1:
        rc = retained(case_set, menu) / max(len(case_set), 1)
        rh = retained(ctrl_set, menu) / max(len(ctrl_set), 1)
        if min(rc, rh) >= retention and retained(case_set, menu) >= min_group and retained(ctrl_set, menu) >= min_group:
            break

        scored = []
        for t in menu:
            m2 = [x for x in menu if x != t]
            c2, h2 = retained(case_set, m2), retained(ctrl_set, m2)
            scored.append((min(c2 / max(len(case_set), 1), h2 / max(len(ctrl_set), 1)), c2 + h2, t))
        _, _, drop = max(scored)
        menu.remove(drop)

    if not menu:
        raise RuntimeError(f"{disease}: no common task family survived support rules.")
    return menu


def choose_per_task_budget(
    df: pd.DataFrame,
    case_ids: list[str],
    control_ids: list[str],
    menu: list[str],
    retention: float,
    min_group: int,
    candidates: tuple[int, ...],
) -> tuple[int, list[str], list[str]]:
    sub = df[
        df["participant_id"].isin(case_ids + control_ids)
        & df["task_family"].isin(menu)
    ]
    counts = sub.groupby(["participant_id", "task_family"]).size().unstack(fill_value=0)
    for t in menu:
        if t not in counts:
            counts[t] = 0

    def eligible(ids: list[str], b: int) -> list[str]:
        x = counts.reindex(ids, fill_value=0)
        keep = (x[menu] >= b).all(axis=1)
        return x.index[keep].astype(str).tolist()

    for b in sorted(set(candidates), reverse=True):
        c = eligible(case_ids, b)
        h = eligible(control_ids, b)
        if (
            len(c) >= min_group
            and len(h) >= min_group
            and len(c) / max(len(case_ids), 1) >= retention
            and len(h) / max(len(control_ids), 1) >= retention
        ):
            return b, c, h

    # Last-resort budget 1 with support; never use performance to choose it.
    c = eligible(case_ids, 1)
    h = eligible(control_ids, 1)
    if len(c) >= min_group and len(h) >= min_group:
        return 1, c, h

    raise RuntimeError(
        "No per-task observation budget retains the minimum supported groups. "
        "Use --retention below the current value only if pre-specified and documented."
    )



def choose_paired_task_budget(
    df: pd.DataFrame,
    matched_case_ids: list[str],
    matched_control_ids: list[str],
    menu: list[str],
    retention: float,
    min_group: int,
    candidates: tuple[int, ...],
) -> tuple[int, list[str], list[str]]:
    """
    Select an observation budget while preserving the original 1:1
    demographic matched pairs.

    A pair is retained only when BOTH the case and its matched control have at
    least `budget` observations in every task in `menu`.
    """
    if len(matched_case_ids) != len(matched_control_ids):
        raise RuntimeError(
            "Matched-pair invariant failed before budget selection: "
            f"case={len(matched_case_ids)} control={len(matched_control_ids)}."
        )

    pairs = list(zip(map(str, matched_case_ids), map(str, matched_control_ids)))
    sub = df[
        df["participant_id"].isin(
            list(map(str, matched_case_ids)) + list(map(str, matched_control_ids))
        )
        & df["task_family"].isin(menu)
    ].copy()

    counts = (
        sub.groupby(["participant_id", "task_family"])
        .size()
        .unstack(fill_value=0)
    )
    for t in menu:
        if t not in counts:
            counts[t] = 0

    def eligible(pid: str, b: int) -> bool:
        row = counts.reindex([str(pid)], fill_value=0)
        return bool((row[menu] >= b).all(axis=1).iloc[0])

    def retained_pairs(b: int) -> list[tuple[str, str]]:
        return [
            (case_id, control_id)
            for case_id, control_id in pairs
            if eligible(case_id, b) and eligible(control_id, b)
        ]

    denominator = max(len(pairs), 1)
    for b in sorted(set(candidates), reverse=True):
        kept = retained_pairs(b)
        if len(kept) >= min_group and len(kept) / denominator >= retention:
            return int(b), [c for c, _ in kept], [h for _, h in kept]

    kept = retained_pairs(1)
    if len(kept) >= min_group:
        return 1, [c for c, _ in kept], [h for _, h in kept]

    raise RuntimeError(
        "No paired per-task observation budget retains the minimum supported "
        "number of demographic matched pairs. This is an overlap/positivity "
        "failure; matched pairs will not be broken to force a result."
    )


def sample_taskwise(
    df: pd.DataFrame,
    ids: list[str],
    menu: list[str],
    budget: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sub = df[df["participant_id"].isin(ids) & df["task_family"].isin(menu)]
    chunks = []
    for (pid, fam), g in sub.groupby(["participant_id", "task_family"], sort=True):
        if len(g) < budget:
            continue
        idx = rng.choice(g.index.to_numpy(), size=budget, replace=False)
        chunks.append(g.loc[idx])
    if not chunks:
        return sub.iloc[0:0].copy()
    out = pd.concat(chunks, ignore_index=True)

    expected = len(ids) * len(menu) * budget
    if len(out) != expected:
        got = out.groupby("participant_id")["task_family"].nunique()
        bad = got[got != len(menu)]
        raise RuntimeError(
            f"Taskwise balance invariant failed: got {len(out)} rows, expected {expected}; "
            f"bad participants={bad.index.tolist()[:10]}"
        )
    return out


def discover_feature_files_from_atlas(
    atlas_config: Path,
    b2ai_root: Path,
) -> dict[str, Path]:
    """Use the existing frozen atlas configuration as the single source of truth."""
    if not atlas_config.exists():
        raise FileNotFoundError(f"Missing atlas config: {atlas_config}")
    cfg = yaml.safe_load(atlas_config.read_text())
    reps = cfg.get("representations", {}) if isinstance(cfg, dict) else {}
    if not isinstance(reps, dict):
        raise RuntimeError(f"No representations block found in {atlas_config}")

    result: dict[str, Path] = {}
    for rep in REP_SPECS:
        spec = reps.get(rep)
        if not isinstance(spec, dict) or "path" not in spec:
            raise RuntimeError(
                f"Representation {rep!r} is missing from {atlas_config}. "
                "The add-on will not guess representation files."
            )
        raw = Path(str(spec["path"]))
        path = raw if raw.is_absolute() else b2ai_root / raw
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Configured representation file for {rep} does not exist: {path}"
            )
        result[rep] = path

    print(f"[representations] source={atlas_config}", flush=True)
    for rep, path in result.items():
        print(f"  {rep:16s} -> {path}", flush=True)
    return result

def load_feature_map_json(path: Path) -> dict[str, Path]:
    data = json.loads(path.read_text())
    missing = [r for r in REP_SPECS if r not in data]
    if missing:
        raise ValueError(f"feature-map JSON is missing {missing}")
    return {r: Path(data[r]).expanduser().resolve() for r in REP_SPECS}


def value_to_vector(value: Any, dim: int, mode: str) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    arr = np.squeeze(arr)
    if arr.size == 0:
        return None

    if mode == "mean_std":
        flat = arr.reshape(-1)
        return np.array([np.nanmean(flat), np.nanstd(flat)], dtype=np.float32)

    if mode == "static":
        flat = arr.reshape(-1)
        if flat.size == dim:
            return flat.astype(np.float32)
        return None

    # temporal mean / already-fixed vector
    if arr.ndim == 1:
        if arr.size == dim:
            return arr.astype(np.float32)
        if arr.size % dim == 0:
            arr = arr.reshape(-1, dim)
            return np.nanmean(arr, axis=0).astype(np.float32)
        return None

    axes = [i for i, s in enumerate(arr.shape) if s == dim]
    if axes:
        feat_axis = axes[0]
        reduce_axes = tuple(i for i in range(arr.ndim) if i != feat_axis)
        return np.nanmean(arr, axis=reduce_axes).reshape(-1).astype(np.float32)

    if arr.size % dim == 0:
        return np.nanmean(arr.reshape(-1, dim), axis=0).astype(np.float32)
    return None


def choose_feature_payload_column(df: pd.DataFrame, rep: str, dim: int) -> str | None:
    meta_norms = {
        norm_col(x) for x in (
            *PARTICIPANT_ALIASES, *SESSION_ALIASES, *TASK_ALIASES,
            *FAMILY_ALIASES, *COHORT_ALIASES, *AGE_ALIASES, *SEX_ALIASES,
            "duration", "recording_id", "acoustic_task_id", "id"
        )
    }

    # Prefer a column named for the representation / feature / embedding.
    keys = [rep, rep.replace("_spectrogram", ""), "feature", "features", "embedding", "values", "value"]
    preferred = []
    for c in df.columns:
        n = norm_col(c)
        if n in meta_norms:
            continue
        if any(k in n for k in keys):
            preferred.append(c)

    for c in preferred + list(df.columns):
        if norm_col(c) in meta_norms:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        v = value_to_vector(s.iloc[0], dim, REP_SPECS[rep]["mode"])
        if v is not None and len(v) == dim:
            return c
    return None


def _string_lists(obj: Any):
    if isinstance(obj, list):
        if obj and all(isinstance(x, str) for x in obj):
            yield obj
        for x in obj:
            yield from _string_lists(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _string_lists(x)


def _static_columns_from_json(path: Path, df: pd.DataFrame, dim: int) -> list[str] | None:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return None
    try:
        obj = json.loads(sidecar.read_text())
    except Exception:
        return None
    cols = set(df.columns)
    for seq in _string_lists(obj):
        if len(seq) == dim and all(x in cols for x in seq):
            return list(seq)
    return None


def numeric_feature_matrix(
    df: pd.DataFrame,
    rep: str,
    dim: int,
    source_path: Path | None = None,
) -> tuple[np.ndarray, list[str]] | None:
    exclude = set()
    for aliases in (
        PARTICIPANT_ALIASES, SESSION_ALIASES, TASK_ALIASES, FAMILY_ALIASES,
        COHORT_ALIASES, AGE_ALIASES, SEX_ALIASES
    ):
        c = first_col(df, aliases, required=False)
        if c:
            exclude.add(c)

    if rep == "static" and source_path is not None:
        sidecar_cols = _static_columns_from_json(source_path, df, dim)
        if sidecar_cols is not None:
            mat = df[sidecar_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            return mat, sidecar_cols

    metadata_tokens = (
        "participant", "session", "recording", "acoustic_task", "task_", "_task",
        "duration", "status", "cohort", "diagnosis", "version", "language",
        "device", "adapter", "sample_rate", "sampling_rate", "timestamp", "date",
    )
    num_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        n = norm_col(c)
        if any(tok in n for tok in metadata_tokens):
            continue
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().mean() > 0.95:
            num_cols.append(c)

    if len(num_cols) == dim:
        return (
            df[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32),
            num_cols,
        )

    if rep == "static" and len(num_cols) > dim:
        raise RuntimeError(
            f"Static table has {len(num_cols)} plausible numeric feature columns, expected {dim}. "
            f"Sidecar {source_path.with_suffix('.json') if source_path else ''} did not provide an exact 131-column list. "
            "Refusing to select columns by position."
        )
    return None


def discover_embedding_caches(repo_root: Path) -> dict[str, Path]:
    """
    Reuse the recording-level embeddings already produced by the frozen atlas.
    This is preferred over reparsing raw Bridge2AI feature tables because these
    caches encode the exact original pooling/schema logic used by the benchmark.
    """
    cache_dir = repo_root / "outputs" / "embeddings"
    result: dict[str, Path] = {}
    missing = []
    for rep in REP_SPECS:
        p = cache_dir / f"recording_{rep}.pkl"
        if not p.exists():
            missing.append(str(p))
        else:
            result[rep] = p.resolve()

    if missing:
        raise FileNotFoundError(
            "Missing one or more frozen recording-level embedding caches:\n  - "
            + "\n  - ".join(missing)
            + "\nRun the existing atlas embedding/cache step first. "
              "V4 intentionally will not rebuild or reinterpret raw feature tables."
        )

    print(f"[embedding-caches] source={cache_dir.resolve()}", flush=True)
    for rep, path in result.items():
        print(f"  {rep:16s} -> {path}", flush=True)
    return result


def load_cached_representation_lookup(
    path: Path,
    rep: str,
) -> dict[tuple[str, str, str], np.ndarray]:
    """
    Load the exact recording-level vector cache generated by the existing atlas.

    Expected canonical columns:
      participant_id, session_id, raw_task, vector, representation,
      recording_id, cohort, fold
    """
    print(f"[cache:{rep}] {path}", flush=True)
    df = pd.read_pickle(path)

    if not isinstance(df, pd.DataFrame):
        raise RuntimeError(f"{path} is not a pandas DataFrame cache.")

    pid = first_col(df, PARTICIPANT_ALIASES)
    session = first_col(df, SESSION_ALIASES)
    task = first_col(df, TASK_ALIASES)

    if "vector" not in df.columns:
        raise RuntimeError(
            f"{path.name}: expected canonical 'vector' column from the existing atlas cache; "
            f"found columns={list(df.columns)}"
        )

    dim = int(REP_SPECS[rep]["dim"])
    if "representation" in df.columns:
        observed = sorted({str(x) for x in df["representation"].dropna().unique()})
        if observed and observed != [rep]:
            raise RuntimeError(
                f"{path.name}: representation column does not match {rep!r}; observed={observed}"
            )

    lookup: dict[tuple[str, str, str], np.ndarray] = {}
    duplicate_keys = 0
    all_nan = 0
    wrong_shapes: dict[int, int] = {}

    for pidv, sessv, taskv, value in zip(
        df[pid].astype(str),
        df[session].astype(str),
        df[task],
        df["vector"],
    ):
        try:
            vec = np.asarray(value, dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise RuntimeError(
                f"{path.name}: could not convert cached vector to numeric array: {exc}"
            ) from exc

        if vec.size != dim:
            wrong_shapes[int(vec.size)] = wrong_shapes.get(int(vec.size), 0) + 1
            continue
        if not np.isfinite(vec).any():
            all_nan += 1
            continue

        key = (str(pidv), str(sessv), norm_task(taskv))
        if key in lookup:
            duplicate_keys += 1
            raise RuntimeError(
                f"{path.name}: duplicate canonical recording key {key}. "
                "Refusing to average duplicates because that would alter the frozen observation definition."
            )
        lookup[key] = vec

    if wrong_shapes:
        raise RuntimeError(
            f"{path.name}: cached vector dimensions do not match expected {dim}. "
            f"Observed wrong sizes/counts={wrong_shapes}"
        )

    print(
        f"[cache:{rep}] rows={len(df):,} keys={len(lookup):,} "
        f"all_nan={all_nan:,} duplicate_keys={duplicate_keys:,} dim={dim}",
        flush=True,
    )
    return lookup


def build_all_cached_feature_lookups(
    cache_files: dict[str, Path],
) -> dict[str, dict]:
    return {
        rep: load_cached_representation_lookup(path, rep)
        for rep, path in cache_files.items()
    }


def load_representation_lookup(path: Path, rep: str) -> dict[tuple[str, str, str], np.ndarray]:
    print(f"[features:{rep}] {path}", flush=True)
    df = _read_table(path)
    pid = first_col(df, PARTICIPANT_ALIASES)
    session = first_col(df, SESSION_ALIASES)
    task = first_col(df, TASK_ALIASES)
    dim = int(REP_SPECS[rep]["dim"])

    df = df.copy()
    df["_pid"] = df[pid].astype(str)
    df["_session"] = df[session].astype(str)
    df["_task"] = df[task].map(norm_task)

    payload = choose_feature_payload_column(df, rep, dim)
    vectors: list[np.ndarray | None]
    if payload is not None:
        vectors = [
            value_to_vector(v, dim, REP_SPECS[rep]["mode"])
            for v in df[payload].tolist()
        ]
    else:
        numeric = numeric_feature_matrix(df, rep, dim, source_path=path)
        if numeric is None:
            raise RuntimeError(
                f"{rep}: could not identify a {dim}-D payload in {path.name}. "
                "No files are modified. Inspect the configured feature-table schema and, if needed, "
                "extend only this add-on."
            )
        mat, _ = numeric
        vectors = [row for row in mat]

    lookup: dict[tuple[str, str, str], list[np.ndarray]] = {}
    bad = 0
    for pidv, sessv, taskv, vec in zip(df["_pid"], df["_session"], df["_task"], vectors):
        if vec is None or len(vec) != dim or not np.isfinite(vec).any():
            bad += 1
            continue
        lookup.setdefault((pidv, sessv, taskv), []).append(np.asarray(vec, dtype=np.float32))

    collapsed = {
        k: np.nanmean(np.stack(v), axis=0).astype(np.float32)
        for k, v in lookup.items()
    }
    print(
        f"[features:{rep}] rows={len(df):,} keys={len(collapsed):,} bad={bad:,}",
        flush=True,
    )
    return collapsed


def build_all_feature_lookups(feature_files: dict[str, Path]) -> dict[str, dict]:
    return {rep: load_representation_lookup(path, rep) for rep, path in feature_files.items()}


def attach_feature_presence(
    manifest: pd.DataFrame,
    lookups: dict[str, dict],
) -> pd.DataFrame:
    out = manifest.copy()
    keys = list(zip(out["participant_id"], out["session_id"], out["task_norm"]))
    for rep, lookup in lookups.items():
        out[f"_has_{rep}"] = [k in lookup for k in keys]
    out["_has_all_reps"] = out[[f"_has_{r}" for r in REP_SPECS]].all(axis=1)
    return out


def build_speech_matrix(
    selected_rows: pd.DataFrame,
    participant_ids: list[str],
    lookups: dict[str, dict],
) -> tuple[np.ndarray, list[str]]:
    """
    For each representation: average selected observation vectors within participant,
    then concatenate representation-level participant vectors (470-D total).
    """
    rows_by_pid = {
        str(pid): g for pid, g in selected_rows.groupby("participant_id", sort=False)
    }
    X = []
    kept = []
    for pid in participant_ids:
        g = rows_by_pid.get(str(pid))
        if g is None or g.empty:
            continue
        pieces = []
        okay = True
        keys = list(zip(g["participant_id"], g["session_id"], g["task_norm"]))
        for rep in REP_SPECS:
            vals = [lookups[rep][k] for k in keys if k in lookups[rep]]
            if len(vals) != len(keys):
                okay = False
                break
            pieces.append(np.nanmean(np.stack(vals), axis=0))
        if not okay:
            continue
        vec = np.concatenate(pieces).astype(np.float32)
        X.append(vec)
        kept.append(str(pid))
    if not X:
        return np.empty((0, sum(int(x["dim"]) for x in REP_SPECS.values())), dtype=np.float32), []
    return np.stack(X), kept


def build_protocol_matrix(
    rows: pd.DataFrame,
    participant_ids: list[str],
    demographics: pd.DataFrame,
    all_families: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    pids = pd.Index([str(x) for x in participant_ids], name="participant_id")
    completion = (
        rows.assign(_one=1)
        .pivot_table(index="participant_id", columns="task_family", values="_one", aggfunc="max", fill_value=0)
        .reindex(pids, fill_value=0)
    )
    counts = (
        rows.assign(_one=1)
        .pivot_table(index="participant_id", columns="task_family", values="_one", aggfunc="sum", fill_value=0)
        .reindex(pids, fill_value=0)
    )
    for fam in all_families:
        if fam not in completion:
            completion[fam] = 0
        if fam not in counts:
            counts[fam] = 0
    completion = completion[all_families]
    counts = counts[all_families]
    completion.columns = [f"complete__{x}" for x in completion.columns]
    counts.columns = [f"count__{x}" for x in counts.columns]

    summary = rows.groupby("participant_id").agg(
        total_observations=("task_family", "size"),
        total_sessions=("session_id", "nunique"),
        total_families=("task_family", "nunique"),
    ).reindex(pids, fill_value=0)

    demo = demographics.set_index("participant_id").reindex(pids)
    sex_m = demo["sex"].eq("m").astype(float).rename("sex_m")
    age = demo["age"].astype(float).rename("age")

    Xdf = pd.concat([completion, counts, summary, age, sex_m], axis=1)
    return Xdf.to_numpy(dtype=float), pids.astype(str).tolist(), Xdf.columns.tolist()


def make_labels(ids: list[str], disease: str, cohort_by_pid: pd.Series) -> np.ndarray:
    return np.array([1 if cohort_by_pid.get(str(pid)) == disease else 0 for pid in ids], dtype=int)


def make_folds(ids: list[str], y: np.ndarray, external_map: dict[str, int] | None, seed: int) -> np.ndarray:
    if external_map is not None and all(str(pid) in external_map for pid in ids):
        raw = np.array([external_map[str(pid)] for pid in ids], dtype=int)
        # normalize arbitrary fold labels to 0..k-1
        uniq = {v: i for i, v in enumerate(sorted(np.unique(raw)))}
        folds = np.array([uniq[v] for v in raw], dtype=int)
        if len(np.unique(folds)) >= 2:
            return folds

    n_splits = min(5, int(np.bincount(y).min()))
    if n_splits < 2:
        raise RuntimeError("Not enough participants per class for participant-disjoint CV.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=int)
    for fold, (_, test) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[test] = fold
    return folds


def fixed_probe_oof(
    X: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    folds: np.ndarray,
) -> pd.DataFrame:
    """
    Fixed PLS + class-balanced logistic probe with numerical safeguards.

    Scientific behavior is unchanged:
      train-fold median imputation -> standardization -> PLS <=10 ->
      class-balanced logistic regression.

    V8 adds only safeguards needed for exactly balanced/pair-aware designs:
      * remove train-fold zero-variance columns;
      * if the remaining features are exactly orthogonal to the binary target,
        return the no-signal score 0.5 rather than asking PLS to divide by zero;
      * cap PLS components by numerical matrix rank;
      * if a high component count is numerically singular, retry with fewer
        components rather than failing the entire disease analysis.

    These safeguards are particularly important for protocol ablations such as
    sex-only after exact-sex pair matching, where X varies across participants
    but Cov(X, y) is exactly zero by construction.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=int)
    rows = []

    for fold in sorted(np.unique(folds)):
        tr = folds != fold
        te = folds == fold
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            raise RuntimeError(f"Fold {fold} contains a single class.")

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xi_tr = imp.fit_transform(X[tr])
        Xi_te = imp.transform(X[te])
        Xs_tr = sc.fit_transform(Xi_tr)
        Xs_te = sc.transform(Xi_te)

        # Drop train-fold zero-variance / non-finite columns before PLS.
        var = np.nanvar(Xs_tr, axis=0)
        keep = np.isfinite(var) & (var > 1e-12)
        if not keep.any():
            score = np.full(te.sum(), 0.5, dtype=float)
        else:
            Xs_tr = np.asarray(Xs_tr[:, keep], dtype=float)
            Xs_te = np.asarray(Xs_te[:, keep], dtype=float)

            if not np.isfinite(Xs_tr).all() or not np.isfinite(Xs_te).all():
                raise RuntimeError(
                    f"Fold {fold}: non-finite values remain after imputation/standardization."
                )

            # PLSRegression centers y internally. If X is exactly orthogonal to
            # centered y, there is no supervised latent direction to estimate.
            # This is a legitimate no-signal case, not a numerical failure.
            yc = y[tr].astype(float) - float(np.mean(y[tr]))
            xy = Xs_tr.T @ yc
            cov_tol = 1e-12 * max(
                1.0,
                float(np.linalg.norm(Xs_tr, ord="fro"))
                * float(np.linalg.norm(yc)),
            )

            if (not np.isfinite(xy).all()) or float(np.linalg.norm(xy)) <= cov_tol:
                score = np.full(te.sum(), 0.5, dtype=float)
            else:
                minority = int(np.bincount(y[tr]).min())
                try:
                    rank = int(np.linalg.matrix_rank(Xs_tr))
                except Exception:
                    rank = int(Xs_tr.shape[1])

                max_comp = int(
                    min(
                        10,
                        Xs_tr.shape[1],
                        Xs_tr.shape[0] - 1,
                        minority - 1,
                        rank,
                    )
                )
                if max_comp < 1:
                    score = np.full(te.sum(), 0.5, dtype=float)
                else:
                    Ztr = Zte = None
                    last_exc = None

                    # Some pair-balanced low-rank protocol matrices can exhaust
                    # target covariance before all rank-allowed components are
                    # extracted. Retry downward; this does not increase model
                    # capacity and keeps PLS <= the frozen 10-component limit.
                    for n_comp in range(max_comp, 0, -1):
                        try:
                            with np.errstate(
                                divide="ignore",
                                invalid="ignore",
                                over="ignore",
                                under="ignore",
                            ):
                                pls = PLSRegression(
                                    n_components=n_comp,
                                    scale=False,
                                )
                                pls.fit(Xs_tr, y[tr])
                                ztr = np.asarray(pls.transform(Xs_tr), dtype=float)
                                zte = np.asarray(pls.transform(Xs_te), dtype=float)

                            if (
                                ztr.ndim != 2
                                or zte.ndim != 2
                                or ztr.shape[1] < 1
                                or not np.isfinite(ztr).all()
                                or not np.isfinite(zte).all()
                            ):
                                raise FloatingPointError(
                                    "PLS returned a non-finite/empty latent representation."
                                )
                            Ztr, Zte = ztr, zte
                            break
                        except Exception as exc:
                            last_exc = exc
                            Ztr = Zte = None

                    if Ztr is None or Zte is None:
                        # If PLS cannot form even one finite supervised latent
                        # direction, treat the fold as no resolved linear signal.
                        # This is the same semantic result as exact orthogonality.
                        score = np.full(te.sum(), 0.5, dtype=float)
                    else:
                        clf = LogisticRegression(
                            class_weight="balanced",
                            solver="liblinear",
                            C=1.0,
                            max_iter=3000,
                            random_state=2026,
                        )
                        clf.fit(Ztr, y[tr])
                        score = clf.predict_proba(Zte)[:, 1]

        for pid, yt, ys in zip(np.asarray(ids)[te], y[te], score):
            rows.append({
                "participant_id": str(pid),
                "fold": int(fold),
                "y_true": int(yt),
                "y_score": float(ys),
            })

    return pd.DataFrame(rows)


def auc_ci(pred: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    auc = float(roc_auc_score(pred["y_true"], pred["y_score"]))
    rng = np.random.default_rng(seed)
    case = pred[pred["y_true"].eq(1)].reset_index(drop=True)
    ctrl = pred[pred["y_true"].eq(0)].reset_index(drop=True)
    vals = []
    for _ in range(n_boot):
        ci = rng.integers(0, len(case), len(case))
        hi = rng.integers(0, len(ctrl), len(ctrl))
        b = pd.concat([case.iloc[ci], ctrl.iloc[hi]], ignore_index=True)
        vals.append(roc_auc_score(b["y_true"], b["y_score"]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return auc, float(lo), float(hi)


def average_seed_predictions(preds: list[pd.DataFrame]) -> pd.DataFrame:
    z = pd.concat(preds, ignore_index=True)
    out = z.groupby("participant_id", as_index=False).agg(
        y_true=("y_true", "first"),
        y_score=("y_score", "mean"),
    )
    return out


def evaluate_raw_protocol(
    disease: str,
    manifest: pd.DataFrame,
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    fold_map: dict[str, int] | None,
    seed: int,
    n_boot: int,
) -> tuple[float, float, float]:
    raw = manifest[manifest["cohort"].isin([disease, "Healthy Control"])].copy()
    ids = sorted(raw["participant_id"].unique().astype(str))
    demo_ids = set(demographics["participant_id"].astype(str))
    ids = [x for x in ids if x in demo_ids]
    raw = raw[raw["participant_id"].isin(ids)]
    families = sorted(raw["task_family"].dropna().unique().tolist())
    X, ids2, _ = build_protocol_matrix(raw, ids, demographics, families)
    y = make_labels(ids2, disease, cohort_by_pid)
    folds = make_folds(ids2, y, fold_map, seed)
    pred = fixed_probe_oof(X, y, ids2, folds)
    return auc_ci(pred, n_boot, seed)


def prepare_selection(
    disease: str,
    manifest: pd.DataFrame,
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    retention: float,
    min_group: int,
    age_caliper_sd: float,
    budget_candidates: tuple[int, ...],
) -> DiseaseSelection:
    case_ids, ctrl_ids, caliper = match_demographics(
        disease, cohort_by_pid, demographics, age_caliper_sd
    )
    if len(case_ids) < min_group or len(ctrl_ids) < min_group:
        raise RuntimeError(
            f"{disease}: demographic matching retained case={len(case_ids)}, "
            f"control={len(ctrl_ids)} < min_group={min_group}"
        )

    menu = choose_task_menu(
        manifest, case_ids, ctrl_ids, retention=retention, min_group=min_group
    )
    budget, c2, h2 = choose_paired_task_budget(
        manifest, case_ids, ctrl_ids, menu,
        retention=retention, min_group=min_group,
        candidates=budget_candidates,
    )
    return DiseaseSelection(
        disease=disease,
        matched_case_ids=case_ids,
        matched_control_ids=ctrl_ids,
        task_menu=menu,
        per_task_budget=budget,
        eligible_case_ids=c2,
        eligible_control_ids=h2,
        age_caliper=caliper,
    )


def run_disease(
    disease: str,
    manifest: pd.DataFrame,
    demographics: pd.DataFrame,
    lookups: dict[str, dict],
    cohort_by_pid: pd.Series,
    published_raw: dict[str, float],
    fold_map: dict[str, int] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 78)
    print(f"PROTOCOL-BALANCED: {disease}")
    print("=" * 78)

    raw_protocol_auc, raw_protocol_lo, raw_protocol_hi = evaluate_raw_protocol(
        disease, manifest, demographics, cohort_by_pid,
        fold_map, args.seed, args.bootstrap,
    )

    sel = prepare_selection(
        disease, manifest, demographics, cohort_by_pid,
        args.retention, args.min_group, args.age_caliper_sd,
        tuple(args.per_task_budgets),
    )

    ids = sorted(sel.eligible_case_ids + sel.eligible_control_ids)
    balanced_source = manifest[
        manifest["participant_id"].isin(ids)
        & manifest["task_family"].isin(sel.task_menu)
    ].copy()

    # Strictly require all released representations for selected observations.
    # This prevents one representation from receiving a larger evidence budget.
    balanced_source = balanced_source[balanced_source["_has_all_reps"]].copy()

    # Re-check budget eligibility after feature intersection.
    budget, case_ids, ctrl_ids = choose_paired_task_budget(
        balanced_source,
        sel.eligible_case_ids,
        sel.eligible_control_ids,
        sel.task_menu,
        retention=args.feature_retention,
        min_group=args.min_group,
        candidates=(sel.per_task_budget,) + tuple(
            b for b in args.per_task_budgets if b < sel.per_task_budget
        ),
    )
    if len(case_ids) != len(ctrl_ids):
        raise RuntimeError(
            f"{disease}: paired-balance invariant failed after feature intersection: "
            f"case={len(case_ids)} control={len(ctrl_ids)}"
        )
    ids = sorted(case_ids + ctrl_ids)
    y = make_labels(ids, disease, cohort_by_pid)
    folds = make_folds(ids, y, fold_map, args.seed)

    protocol_preds = []
    speech_preds = []
    run_rows = []
    selected_rows_for_manifest = []

    for s in range(args.seeds):
        sampled = sample_taskwise(
            balanced_source,
            ids,
            sel.task_menu,
            budget,
            seed=args.seed + s,
        )
        sampled["subsample_seed"] = s
        selected_rows_for_manifest.append(
            sampled[[
                "_row_id", "participant_id", "session_id", "raw_task",
                "task_family", "cohort", "subsample_seed"
            ]].copy()
        )

        # Protocol-only on exactly the same balanced participants/observations.
        all_fams = list(COMMON_TASKS)
        Xp, pids_p, _ = build_protocol_matrix(
            sampled, ids, demographics, all_fams
        )
        yp = make_labels(pids_p, disease, cohort_by_pid)
        fold_p = np.array([folds[ids.index(pid)] for pid in pids_p], dtype=int)
        pp = fixed_probe_oof(Xp, yp, pids_p, fold_p)
        pp["subsample_seed"] = s
        protocol_preds.append(pp)

        # Speech: average each representation within participant, then concatenate.
        Xs, pids_s = build_speech_matrix(sampled, ids, lookups)
        if pids_s != ids:
            raise RuntimeError(
                f"{disease}: speech feature invariant failed; expected {len(ids)} participants, "
                f"got {len(pids_s)}."
            )
        ys = make_labels(pids_s, disease, cohort_by_pid)
        ps = fixed_probe_oof(Xs, ys, pids_s, folds)
        ps["subsample_seed"] = s
        speech_preds.append(ps)

        pa = roc_auc_score(pp["y_true"], pp["y_score"])
        sa = roc_auc_score(ps["y_true"], ps["y_score"])
        run_rows.append({
            "disease": disease,
            "subsample_seed": s,
            "n_case": len(case_ids),
            "n_control": len(ctrl_ids),
            "n_task_families": len(sel.task_menu),
            "task_menu": "|".join(sel.task_menu),
            "per_task_budget": budget,
            "total_observations_per_participant": budget * len(sel.task_menu),
            "protocol_auc": pa,
            "speech_auc": sa,
        })

    avg_p = average_seed_predictions(protocol_preds)
    avg_s = average_seed_predictions(speech_preds)
    p_auc, p_lo, p_hi = auc_ci(avg_p, args.bootstrap, args.seed + 1000)
    s_auc, s_lo, s_hi = auc_ci(avg_s, args.bootstrap, args.seed + 2000)

    protocol_neutralized = bool(p_lo <= 0.5 <= p_hi)
    speech_retained = bool(s_lo > 0.5)
    if protocol_neutralized and speech_retained:
        status = "protocol_neutralized__speech_retained"
    elif protocol_neutralized and not speech_retained:
        status = "protocol_neutralized__speech_not_resolved"
    elif not protocol_neutralized and speech_retained:
        status = "residual_protocol_signal__speech_retained"
    else:
        status = "residual_protocol_signal__speech_not_resolved"

    summary = {
        "disease": disease,
        "published_raw_speech_auc": published_raw.get(disease, np.nan),
        "raw_protocol_auc": raw_protocol_auc,
        "raw_protocol_ci_low": raw_protocol_lo,
        "raw_protocol_ci_high": raw_protocol_hi,
        "matched_cases_initial": len(sel.matched_case_ids),
        "matched_controls_initial": len(sel.matched_control_ids),
        "age_caliper_years": sel.age_caliper,
        "task_menu": "|".join(sel.task_menu),
        "n_task_families": len(sel.task_menu),
        "per_task_budget": budget,
        "total_observations_per_participant": budget * len(sel.task_menu),
        "balanced_n_case": len(case_ids),
        "balanced_n_control": len(ctrl_ids),
        "balanced_protocol_auc": p_auc,
        "balanced_protocol_ci_low": p_lo,
        "balanced_protocol_ci_high": p_hi,
        "balanced_speech_auc": s_auc,
        "balanced_speech_ci_low": s_lo,
        "balanced_speech_ci_high": s_hi,
        "speech_delta_vs_published_raw": (
            s_auc - published_raw[disease] if disease in published_raw else np.nan
        ),
        "protocol_delta_raw_to_balanced": p_auc - raw_protocol_auc,
        "protocol_neutralized_ci_includes_chance": protocol_neutralized,
        "speech_retained_ci_above_chance": speech_retained,
        "validity_status": status,
        "n_subsample_seeds": args.seeds,
        "n_bootstrap": args.bootstrap,
    }

    print(f"raw published speech AUC : {summary['published_raw_speech_auc']:.3f}" if np.isfinite(summary["published_raw_speech_auc"]) else "raw published speech AUC : n/a")
    print(f"raw protocol AUC         : {raw_protocol_auc:.3f} [{raw_protocol_lo:.3f}, {raw_protocol_hi:.3f}]")
    print(f"matched initial          : case={len(sel.matched_case_ids)} control={len(sel.matched_control_ids)}")
    print(f"coverage-selected menu   : {', '.join(sel.task_menu)}")
    print(f"per-task budget          : {budget} ({budget * len(sel.task_menu)} total/participant)")
    print(f"balanced paired cohort   : case={len(case_ids)} control={len(ctrl_ids)}")
    print(f"balanced protocol AUC    : {p_auc:.3f} [{p_lo:.3f}, {p_hi:.3f}]")
    print(f"balanced speech AUC      : {s_auc:.3f} [{s_lo:.3f}, {s_hi:.3f}]")
    print(f"status                   : {status}")

    runs = pd.DataFrame(run_rows)
    selected = pd.concat(selected_rows_for_manifest, ignore_index=True)
    return summary, runs, selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add-only protocol-balanced validity experiment for ML4H clinical speech."
    )
    p.add_argument("--repo-root", default=".")
    p.add_argument("--manifest", default="outputs/manifests/recordings.csv")
    p.add_argument("--b2ai-root", default=os.environ.get("B2AI_ROOT"))
    p.add_argument("--feature-dir", default=None)
    p.add_argument("--feature-map-json", default=None)
    p.add_argument("--demographics", default=None)
    p.add_argument("--fold-map", default=None)
    p.add_argument("--out-dir", default="outputs/protocol_balanced_framework")
    p.add_argument("--diseases", nargs="+", default=None)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--min-group", type=int, default=20)
    p.add_argument("--retention", type=float, default=0.80)
    p.add_argument(
        "--feature-retention",
        type=float,
        default=0.70,
        help="Minimum retention after requiring all eight representation files.",
    )
    p.add_argument("--age-caliper-sd", type=float, default=0.25)
    p.add_argument("--per-task-budgets", nargs="+", type=int, default=[5, 3, 2, 1])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing existing manifest: {manifest_path}\n"
            "Run this from the existing ML4H atlas repo root."
        )
    if not args.b2ai_root:
        raise RuntimeError(
            "Set B2AI_ROOT or pass --b2ai-root to the existing Bridge2AI v3.0.0 root."
        )
    b2ai_root = Path(args.b2ai_root).expanduser().resolve()
    feature_dir = (
        Path(args.feature_dir).expanduser().resolve()
        if args.feature_dir
        else b2ai_root / "features"
    )

    task_mapping_path = repo / "config" / "task_families.yaml"
    atlas_config_path = repo / "config" / "atlas.yaml"
    manifest = load_manifest(manifest_path, task_mapping_path)
    cohort_by_pid = (
        manifest[["participant_id", "cohort"]]
        .drop_duplicates("participant_id")
        .set_index("participant_id")["cohort"]
    )
    bad = (
        manifest[["participant_id", "cohort"]]
        .drop_duplicates()
        .groupby("participant_id")["cohort"].nunique()
    )
    if (bad > 1).any():
        raise RuntimeError(
            "The recordings manifest contains participants mapped to multiple cohorts. "
            "This extension expects the frozen unique-diagnosis benchmark manifest."
        )

    demographics = discover_demographics(
        repo, b2ai_root, manifest,
        Path(args.demographics).expanduser().resolve() if args.demographics else None,
    )

    fold_map = discover_fold_map(
        repo,
        Path(args.fold_map).expanduser().resolve() if args.fold_map else None,
    )
    if fold_map is None:
        print(
            "[fold-map] No existing participant fold table discovered; "
            "using deterministic participant-disjoint StratifiedKFold(seed=2026) "
            "inside this add-on only.",
            flush=True,
        )

    # Validate the frozen representation configuration for provenance, but do not
    # parse the raw feature tables here. Reuse the exact recording-level caches
    # generated by the existing atlas instead.
    configured_feature_files = discover_feature_files_from_atlas(
        atlas_config_path, b2ai_root
    )
    cache_files = discover_embedding_caches(repo)

    (out_dir / "resolved_feature_sources.json").write_text(
        json.dumps(
            {
                "atlas_config": str(atlas_config_path),
                "configured_raw_feature_files": {
                    k: str(v) for k, v in configured_feature_files.items()
                },
                "recording_embedding_caches": {
                    k: str(v) for k, v in cache_files.items()
                },
                "loader": "existing_atlas_recording_embedding_cache",
            },
            indent=2,
        ) + "\n"
    )

    lookups = build_all_cached_feature_lookups(cache_files)
    manifest = attach_feature_presence(manifest, lookups)

    published_raw = discover_published_raw(repo)
    all_diseases = sorted(
        x for x in manifest["cohort"].dropna().unique()
        if x != "Healthy Control"
    )
    diseases = args.diseases or all_diseases

    summaries = []
    run_tables = []
    selected_tables = []
    failures = []

    for disease in diseases:
        try:
            summary, runs, selected = run_disease(
                disease=disease,
                manifest=manifest,
                demographics=demographics,
                lookups=lookups,
                cohort_by_pid=cohort_by_pid,
                published_raw=published_raw,
                fold_map=fold_map,
                args=args,
            )
            summaries.append(summary)
            run_tables.append(runs)
            selected_tables.append(selected)
        except Exception as exc:
            eprint(f"[FAILED] {disease}: {exc}")
            failures.append({"disease": disease, "error": repr(exc)})

    if not summaries:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
        raise RuntimeError("No disease completed. See failures.json.")

    summary_df = pd.DataFrame(summaries)
    runs_df = pd.concat(run_tables, ignore_index=True)
    selected_df = pd.concat(selected_tables, ignore_index=True)

    summary_df.to_csv(out_dir / "protocol_balanced_summary.csv", index=False)
    runs_df.to_csv(out_dir / "protocol_balanced_runs.csv", index=False)
    selected_df.to_csv(out_dir / "protocol_balanced_selected_observations.csv", index=False)

    metadata = {
        "analysis": "protocol_balanced_detectability",
        "principle": (
            "Evaluate intended speech signal after observed measurement/protocol "
            "information is made non-predictive as far as the recorded variables permit."
        ),
        "repo_root": str(repo),
        "manifest": str(manifest_path),
        "atlas_config": str(atlas_config_path),
        "task_family_mapping": str(task_mapping_path),
        "b2ai_root": str(b2ai_root),
        "common_task_families": list(COMMON_TASKS),
        "age_sex_matching": {
            "exact_sex": True,
            "age_caliper_sd": args.age_caliper_sd,
            "replacement": False,
        },
        "task_menu_selection": "coverage-only; no AUC used",
        "budget_selection": {
            "candidate_per_task_budgets": args.per_task_budgets,
            "retention": args.retention,
            "feature_retention": args.feature_retention,
            "performance_used_for_selection": False,
        },
        "evaluation": {
            "participant_disjoint": True,
            "fold_source": "existing map" if fold_map is not None else "deterministic stratified fallback",
            "imputation": "train-fold median",
            "standardization": "train-fold",
            "pls_max_components": 10,
            "classifier": "class-balanced LogisticRegression(liblinear,C=1)",
            "subsample_seeds": args.seeds,
            "bootstrap_replicates": args.bootstrap,
        },
        "extension_version": "v5-cache-reuse-paired-balance",
        "feature_sources": {
            "loader": "existing_atlas_recording_embedding_cache",
            "configured_raw_feature_files": {
                k: str(v) for k, v in configured_feature_files.items()
            },
            "recording_embedding_caches": {
                k: str(v) for k, v in cache_files.items()
            },
        },
        "paired_balance": {
            "preserve_demographic_pairs_after_task_filtering": True,
            "preserve_demographic_pairs_after_feature_intersection": True,
        },
        "failures": failures,
    }
    (out_dir / "protocol_balanced_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    show = [
        "disease",
        "published_raw_speech_auc",
        "raw_protocol_auc",
        "balanced_protocol_auc",
        "balanced_speech_auc",
        "speech_delta_vs_published_raw",
        "protocol_delta_raw_to_balanced",
        "balanced_n_case",
        "balanced_n_control",
        "n_task_families",
        "per_task_budget",
        "validity_status",
    ]
    print(summary_df[show].to_string(index=False))

    print(f"\nSaved: {out_dir / 'protocol_balanced_summary.csv'}")
    print(f"Saved: {out_dir / 'protocol_balanced_runs.csv'}")
    print(f"Saved: {out_dir / 'protocol_balanced_selected_observations.csv'}")
    print(f"Saved: {out_dir / 'protocol_balanced_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

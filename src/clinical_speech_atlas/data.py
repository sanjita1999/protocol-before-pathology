from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import resolve_data_path, task_map_path
from .schema import detect_array_column, find_column, read_table
from .utils import ensure_dir, stable_seed

TRUE_STRINGS = {"1", "true", "yes", "y", "positive", "present", "case", "diagnosed", "affected"}
FALSE_STRINGS = {"0", "false", "no", "n", "negative", "absent", "control", "healthy", "none"}


def normalize_identifier(value: Any) -> str:
    """Normalize identifier values without relying on zero padding.

    Bridge2AI TSV files may be read by pandas as numeric IDs (e.g. 001573 ->
    1573), while parquet feature files retain the zero-padded string.  The
    clinical identity is the same, so numeric-looking identifiers are
    canonicalized by removing leading zeros / a trailing ``.0``.  UUID-like
    identifiers are left unchanged.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip()
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", s):
        try:
            return str(int(float(s)))
        except Exception:
            pass
    return s


def normalize_id_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_identifier)


def feature_observation_id(participant_id: Any, session_id: Any, raw_task: Any) -> str:
    """Stable ID for one derived Bridge2AI feature observation.

    In Bridge2AI v3.0 the dense feature files are uniquely keyed by
    (participant_id, session_id, task_name).  ``recording.tsv`` is a lower-level
    audio-recording table and one acoustic task can map to several recordings,
    so it must not be used to force a one-to-one feature-to-recording join.
    """
    pid = normalize_identifier(participant_id)
    sid = str(session_id).strip()
    task = str(raw_task).strip()
    return f"{pid}::{sid}::{task}"


def canonical_disease_name(filename: str) -> str:
    stem = Path(filename).stem
    names = {
        "airway_stenosis": "Airway Stenosis",
        "parkinsons_disease": "Parkinson's Disease",
        "laryngeal_dystonia": "Laryngeal Dystonia",
        "unilateral_vocal_fold_paralysis": "Unilateral Vocal Fold Paralysis",
        "cognitive_impairment": "Cognitive Impairment",
        "muscle_tension_dysphonia": "Muscle Tension Dysphonia",
        "benign_lesions": "Benign Lesions",
        "unexplained_chronic_cough": "Unexplained Chronic Cough",
        "precancerous_lesions": "Precancerous Lesions",
        "control": "Healthy Control",
    }
    return names.get(stem, stem.replace("_", " ").title())


def _positive_mask(series: pd.Series) -> pd.Series | None:
    s = series.dropna()
    if s.empty:
        return None
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        uniq = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
        if uniq and uniq.issubset({0, 1, 0.0, 1.0}):
            return pd.to_numeric(series, errors="coerce").fillna(0).astype(float).eq(1)
    norm = series.astype(str).str.strip().str.lower()
    vals = set(norm.dropna().unique())
    if vals & TRUE_STRINGS and vals.issubset(TRUE_STRINGS | FALSE_STRINGS | {"nan", ""}):
        return norm.isin(TRUE_STRINGS)
    return None


def extract_diagnosis_membership(path: str | Path, cfg: dict[str, Any]) -> tuple[set[str], dict]:
    """Reproduce the diagnosis-membership rule used by the AIAS benchmark.

    Bridge2AI diagnosis TSVs are phenotype forms rather than simple one-column
    binary label files.  The original AIAS analysis treated a participant as
    belonging to a diagnosis table when the row contains at least one observed
    (non-null) phenotype value besides the participant identifier.  It did *not*
    search the form for an arbitrary binary field such as treatment, etiology,
    or a diagnostic-criterion checkbox.

    This matters for files such as Parkinson's disease and cognitive impairment,
    which contain many binary clinical fields.  Choosing one of those fields as
    the disease indicator can reduce a 100+ row disease table to only a handful
    of subjects and changes the benchmark cohort definition.
    """
    df = read_table(path)
    pid = find_column(df.columns, cfg["schema"]["participant_id_candidates"])
    non_id_cols = [c for c in df.columns if c != pid]

    if non_id_cols:
        mask = df[non_id_cols].notna().any(axis=1)
        rule = "any_non_null_non_id_field"
    else:
        # A degenerate one-column membership file is still a valid positive list.
        mask = pd.Series(True, index=df.index)
        rule = "participant_row_present"

    ids = set(normalize_id_series(df.loc[mask, pid].dropna()))
    ids.discard("")
    return ids, {
        "participant_column": pid,
        "positive_rule": rule,
        "n_positive": len(ids),
        "n_rows": len(df),
    }


def build_unique_diagnosis_cohort(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the AIAS-compatible unique-diagnosis cohort.

    Multi-diagnosis status must be determined against *all* Bridge2AI diagnosis
    TSVs, not only the nine diseases retained for the final benchmark.  After
    unique-diagnosis membership is established, the configured nine diseases
    plus the unique control cohort are selected for ML4H evaluation.
    """
    diag_dir = resolve_data_path(cfg, cfg["data"]["diagnosis_dir"])
    target_files = set(cfg["cohort"]["disease_files"])
    control_filename = cfg["cohort"]["control_filename"]
    target_names = {canonical_disease_name(f) for f in target_files}
    control_name = canonical_disease_name(control_filename)

    memberships: dict[str, set[str]] = {}
    audit_rows = []

    # This intentionally scans every diagnosis TSV.  The prior AIAS benchmark
    # used the full diagnosis directory to identify multi-diagnosis subjects.
    diagnosis_paths = sorted(diag_dir.glob("*.tsv"))
    if not diagnosis_paths:
        raise FileNotFoundError(f"No diagnosis TSV files found in {diag_dir}")

    for path in diagnosis_paths:
        ids, meta = extract_diagnosis_membership(path, cfg)
        disease = canonical_disease_name(path.name)
        memberships[disease] = ids
        audit_rows.append({
            "file": path.name,
            "disease": disease,
            "selected_for_benchmark": path.name in target_files or path.name == control_filename,
            **meta,
        })

    all_ids = sorted(set().union(*memberships.values())) if memberships else []
    rows = []
    for pid in all_ids:
        diagnoses = sorted([d for d, ids in memberships.items() if pid in ids])
        n_dx = len(diagnoses)
        if n_dx == 1:
            unique_dx = diagnoses[0]
            selected = unique_dx == control_name or unique_dx in target_names
            is_case = 0 if unique_dx == control_name else (1 if unique_dx in target_names else np.nan)
            cohort_name = unique_dx
            exclusion_reason = "" if selected else "unique_non_target_diagnosis"
        else:
            unique_dx = "EXCLUDED_MULTI_DX"
            selected = False
            is_case = np.nan
            cohort_name = "EXCLUDED_MULTI_DX"
            exclusion_reason = "multiple_diagnosis_tables"

        rows.append({
            "participant_id": pid,
            "cohort": cohort_name,
            "is_case": is_case,
            "n_diagnoses": n_dx,
            "diagnosis_list": ";".join(diagnoses),
            "unique_diagnosis": n_dx == 1,
            "included": selected,
            "exclusion_reason": exclusion_reason,
        })

    cohort = pd.DataFrame(rows)

    # Apply the benchmark minimum only after the full-diagnosis uniqueness test.
    min_n = int(cfg["cohort"].get("min_disease_n", 10))
    if min_n > 1 and not cohort.empty:
        counts = cohort.loc[cohort["included"] & cohort["is_case"].eq(1), "cohort"].value_counts()
        small = set(counts[counts < min_n].index)
        if small:
            m = cohort["cohort"].isin(small) & cohort["included"]
            cohort.loc[m, "included"] = False
            cohort.loc[m, "exclusion_reason"] = "below_min_disease_n"

    return cohort, pd.DataFrame(audit_rows)


def _merge_covariate_tables(cfg: dict[str, Any]) -> pd.DataFrame:
    rels = [
        cfg["data"].get("demographics_table"),
        cfg["data"].get("participant_table"),
        cfg["data"].get("enrollment_table"),
        cfg["data"].get("confounders_table"),
        cfg["data"].get("session_table"),
    ]
    tables = []
    for rel in rels:
        if not rel:
            continue
        path = resolve_data_path(cfg, rel)
        if not path.exists():
            continue
        df = read_table(path)
        try:
            pid = find_column(df.columns, cfg["schema"]["participant_id_candidates"])
        except KeyError:
            continue
        tmp = df.copy()
        tmp[pid] = normalize_id_series(tmp[pid])
        # Repeated session rows are collapsed by first non-null value per participant.
        tmp = tmp.groupby(pid, as_index=False).first()
        tmp = tmp.rename(columns={pid: "participant_id"})
        tables.append(tmp)
    if not tables:
        return pd.DataFrame(columns=["participant_id"])
    out = tables[0]
    for i, t in enumerate(tables[1:], 1):
        overlapping = [c for c in t.columns if c in out.columns and c != "participant_id"]
        t = t.rename(columns={c: f"{c}__src{i}" for c in overlapping})
        out = out.merge(t, on="participant_id", how="outer")
    return out


def _find_exact_column(columns, candidates) -> str | None:
    """Exact normalized-name matching only; no substring/suffix guessing."""
    def norm(x: Any) -> str:
        return re.sub(r"[^a-z0-9]", "", str(x).lower())
    lookup = {norm(c): c for c in columns}
    for cand in candidates:
        hit = lookup.get(norm(cand))
        if hit is not None:
            return hit
    return None


def extract_covariates(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Extract age/sex/site without silently inventing a site variable.

    The previous generic fuzzy matcher could map the candidate ``center`` to a
    questionnaire item such as ``voice_activity_v2___call_center_worker``.
    Site is therefore accepted only from an explicit override or an exact
    normalized column-name match.  A prepared AIAS participant metadata CSV can
    also be supplied via ``covariates.metadata_file``.
    """
    cov_cfg = cfg.get("covariates", {})
    metadata_file = cov_cfg.get("metadata_file")

    if metadata_file:
        path = Path(metadata_file).expanduser()
        if not path.is_absolute():
            # First try relative to the analysis project, then relative to data root.
            if path.exists():
                pass
            else:
                candidate = resolve_data_path(cfg, metadata_file)
                if candidate.exists():
                    path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Configured covariates.metadata_file not found: {path}")
        meta = pd.read_csv(path, dtype={"participant_id": str})
        pid = _find_exact_column(meta.columns, cfg["schema"]["participant_id_candidates"])
        if pid is None:
            raise KeyError(f"No participant ID found in covariate metadata file {path}")
        result = pd.DataFrame({"participant_id": normalize_id_series(meta[pid])})

        mapping = {
            "age": cov_cfg.get("age_column") or "age_numeric",
            "sex": cov_cfg.get("sex_column") or "sex_clean",
            "site": cov_cfg.get("site_column") or "site_clean",
        }
        detected = {}
        for key, requested in mapping.items():
            c = _find_exact_column(meta.columns, [requested]) if requested else None
            detected[key] = c
            result[key] = meta[c] if c else np.nan
        result["age"] = pd.to_numeric(result["age"], errors="coerce")
        return result.drop_duplicates("participant_id"), detected

    raw = _merge_covariate_tables(cfg)
    detected = {}
    result = pd.DataFrame({"participant_id": normalize_id_series(raw.get("participant_id", pd.Series(dtype=str)))})

    # Age and sex are safe when exact names exist in the raw phenotype tables.
    for key in ["age", "sex"]:
        override = cov_cfg.get(f"{key}_column")
        candidates = [override] if override else cfg["schema"][f"{key}_candidates"]
        c = _find_exact_column(raw.columns, candidates)
        detected[key] = c
        result[key] = raw[c] if c else np.nan

    # Never use fuzzy matching for site.
    site_override = cov_cfg.get("site_column")
    site_candidates = [site_override] if site_override else cfg["schema"]["site_candidates"]
    c = _find_exact_column(raw.columns, [x for x in site_candidates if x])
    detected["site"] = c
    result["site"] = raw[c] if c else np.nan

    result["age"] = pd.to_numeric(result["age"], errors="coerce")
    return result, detected


def load_recording_metadata(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    path = resolve_data_path(cfg, cfg["data"]["recording_table"])
    df = read_table(path)
    pid = find_column(df.columns, cfg["schema"]["participant_id_candidates"])
    rid = find_column(df.columns, cfg["schema"]["recording_id_candidates"])
    task = find_column(df.columns, cfg["schema"]["task_candidates"], required=False)
    session = find_column(df.columns, cfg["schema"]["session_id_candidates"], required=False)
    out = pd.DataFrame({
        "participant_id": normalize_id_series(df[pid]),
        "recording_id": df[rid].astype(str),
        "raw_task": df[task].astype(str) if task else "UNKNOWN_TASK",
    })
    if session:
        out["session_id"] = df[session].astype(str)
    else:
        out["session_id"] = np.nan
    return out.drop_duplicates("recording_id"), {"participant": pid, "recording": rid, "task": task, "session": session}


def load_task_mapping(cfg: dict[str, Any], require_frozen: bool | None = None) -> dict[str, Any]:
    path = task_map_path(cfg)
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if require_frozen is None:
        require_frozen = cfg.get("runtime", {}).get("fail_on_unfrozen_task_map", True)
    if require_frozen and not obj.get("frozen", False):
        raise RuntimeError(
            f"Task map is not frozen: {path}. Review the raw-task template, fill mapping, and set frozen: true before task analyses."
        )
    return obj


def apply_task_mapping(recordings: pd.DataFrame, cfg: dict[str, Any], require_frozen: bool = True) -> pd.DataFrame:
    obj = load_task_mapping(cfg, require_frozen=require_frozen)
    mapping = {str(k): str(v) for k, v in (obj.get("mapping") or {}).items()}
    out = recordings.copy()
    out["task_family"] = out["raw_task"].map(mapping)
    return out


def _as_python(value: Any) -> Any:
    """Convert Arrow/Pandas scalar containers to ordinary Python objects."""
    if hasattr(value, "as_py"):
        try:
            return value.as_py()
        except Exception:
            pass
    return value


def _flatten_numeric(value: Any) -> np.ndarray:
    """Recursively flatten a possibly ragged nested payload into numeric values.

    Bridge2AI parquet cells may arrive from Arrow/Pandas as object arrays whose
    elements are themselves ndarrays/lists.  ``np.asarray(..., dtype=float)``
    fails on these ragged containers even though every leaf is numeric.
    """
    value = _as_python(value)
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, np.ndarray) and value.dtype != object:
        return value.astype(float, copy=False).reshape(-1)
    if np.isscalar(value):
        try:
            return np.asarray([float(value)], dtype=float)
        except (TypeError, ValueError):
            return np.asarray([], dtype=float)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        parts = [_flatten_numeric(v) for v in value]
        parts = [x for x in parts if x.size]
        return np.concatenate(parts) if parts else np.asarray([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.asarray([], dtype=float)


def vectorize_payload(value: Any, expected_dim: int | None, pooling: str) -> np.ndarray:
    """Convert one Bridge2AI feature/embedding cell into a fixed-length vector.

    Handles both ordinary rectangular numeric arrays and Arrow/Pandas ragged
    object containers.  For time-varying representations with a known feature
    dimension, either feature-major ``[D][T]`` or time-major ``[T][D]`` input
    is accepted.
    """
    value = _as_python(value)

    # Fast path for regular rectangular numeric arrays.
    try:
        arr = np.asarray(value, dtype=float)
        regular = arr.dtype != object
    except (TypeError, ValueError):
        arr = None
        regular = False

    if pooling == "mean_std":
        flat = arr.reshape(-1) if regular else _flatten_numeric(value)
        if flat.size == 0:
            return np.asarray([np.nan, np.nan], dtype=float)
        return np.asarray([np.nanmean(flat), np.nanstd(flat)], dtype=float)

    if pooling != "mean_time":
        shape = getattr(arr, "shape", None)
        raise ValueError(f"Unsupported payload for pooling={pooling}, shape={shape}")

    if regular:
        if arr.ndim == 0:
            return np.asarray([float(arr)], dtype=float)
        if arr.ndim == 1:
            # Already a recording-level embedding/vector.
            if expected_dim is None or arr.size == expected_dim:
                return arr.astype(float)
            # A one-dimensional trajectory: average over time.
            return np.asarray([np.nanmean(arr)], dtype=float)
        # Move expected feature dimension to axis 0 and mean over time/other axes.
        axis = None
        if expected_dim:
            for i, size in enumerate(arr.shape):
                if size == expected_dim:
                    axis = i
                    break
        if axis is None:
            axis = int(np.argmin(arr.shape))
        arr = np.moveaxis(arr, axis, 0)
        return np.nanmean(arr.reshape(arr.shape[0], -1), axis=1)

    # Ragged/object path.  This is common for Arrow list columns.
    if isinstance(value, np.ndarray):
        seq = value.tolist()
    elif isinstance(value, (list, tuple)):
        seq = list(value)
    else:
        flat = _flatten_numeric(value)
        if expected_dim and flat.size == expected_dim:
            return flat
        if flat.size:
            return np.asarray([np.nanmean(flat)], dtype=float)
        raise ValueError(f"Could not vectorize payload of type {type(value).__name__}")

    # Feature-major ragged layout: exactly D rows, each containing time samples.
    if expected_dim and len(seq) == expected_dim:
        pooled = []
        for child in seq:
            vals = _flatten_numeric(child)
            pooled.append(np.nanmean(vals) if vals.size else np.nan)
        return np.asarray(pooled, dtype=float)

    # Time-major layout: every row is a D-dimensional embedding/frame.
    if expected_dim and len(seq) > 0:
        rows = [_flatten_numeric(child) for child in seq]
        if rows and all(r.size == expected_dim for r in rows):
            return np.nanmean(np.vstack(rows), axis=0)

    # Some parquet writers add a single extra list wrapper.
    if len(seq) == 1:
        return vectorize_payload(seq[0], expected_dim, pooling)

    child_shapes = [tuple(_flatten_numeric(x).shape) for x in seq[:8]]
    raise ValueError(
        "Could not infer time/feature axes for a ragged Bridge2AI payload. "
        f"expected_dim={expected_dim}, top_level_len={len(seq)}, "
        f"first_child_flat_shapes={child_shapes}. "
        "Inspect the parquet cell and set the correct expected_dim/pooling in config/atlas.yaml."
    )


def _feature_key_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, str | None]:
    """Detect how a feature table identifies a recording.

    Bridge2AI v3.x dense feature parquet files do not expose a standalone
    recording_id.  They identify each feature row by participant_id,
    session_id, and task_name.  Some tabular feature files may expose a
    recording identifier, so we support both layouts.
    """
    return {
        "recording_id": find_column(df.columns, cfg["schema"]["recording_id_candidates"], required=False),
        "participant_id": find_column(df.columns, cfg["schema"]["participant_id_candidates"], required=False),
        "session_id": find_column(df.columns, cfg["schema"]["session_id_candidates"], required=False),
        "raw_task": find_column(df.columns, cfg["schema"]["task_candidates"], required=False),
    }


def _standardize_feature_keys(df: pd.DataFrame, keys: dict[str, str | None]) -> pd.DataFrame:
    """Return standardized key columns for a feature chunk."""
    if keys["recording_id"]:
        return pd.DataFrame({"recording_id": df[keys["recording_id"]].astype(str)})

    missing = [k for k in ("participant_id", "session_id", "raw_task") if not keys[k]]
    if missing:
        raise KeyError(
            "Feature table has no recording_id and is missing natural recording keys "
            f"{missing}. Available columns={list(df.columns)}"
        )
    return pd.DataFrame({
        "participant_id": normalize_id_series(df[keys["participant_id"]]),
        "session_id": df[keys["session_id"]].astype(str),
        "raw_task": df[keys["raw_task"]].astype(str),
    })


def load_recording_representation(name: str, spec: dict[str, Any], cfg: dict[str, Any]) -> pd.DataFrame:
    """Load one Bridge2AI representation and pool each feature row over time.

    Supports two feature schemas:
      1) tables containing an explicit recording_id; or
      2) Bridge2AI v3.x dense feature tables keyed by
         (participant_id, session_id, task_name).

    The returned dataframe retains whichever key scheme is present.  For
    Bridge2AI v3.0 dense features, the natural key itself is the analysis-level
    observation key; it is intentionally *not* forced onto ``recording.tsv``
    because one acoustic task can contain several lower-level audio recordings.
    """
    path = resolve_data_path(cfg, spec["path"])
    pooling = spec["pooling"]
    expected_dim = spec.get("expected_dim")

    if path.suffix.lower() != ".parquet":
        df = read_table(path)
        keys = _feature_key_columns(df, cfg)
        key_source_cols = {v for v in keys.values() if v}
        meta_exclude = set(key_source_cols)
        # n_frames is metadata in Bridge2AI dense files, not a feature.
        meta_exclude.update(c for c in df.columns if re.sub(r"[^a-z0-9]", "", str(c).lower()) == "nframes")

        if pooling == "static_numeric":
            numeric = [c for c in df.columns if c not in meta_exclude and pd.api.types.is_numeric_dtype(df[c])]
            if not numeric:
                raise ValueError(f"No numeric static-feature columns found in {path}")
            vectors = [row for row in df[numeric].to_numpy(dtype=float)]
        else:
            payload_col = spec.get("value_column") or detect_array_column(df, exclude=meta_exclude)
            if not payload_col:
                non_id = [c for c in df.columns if c not in meta_exclude]
                if len(non_id) == 1:
                    payload_col = non_id[0]
                else:
                    raise KeyError(
                        f"Could not infer feature payload column for {name}. Columns={list(df.columns)}. "
                        f"Set representations.{name}.value_column in config/atlas.yaml."
                    )
            vectors = [vectorize_payload(v, expected_dim, pooling) for v in df[payload_col]]

        out = _standardize_feature_keys(df, keys)
        out["vector"] = vectors
        out["representation"] = name
        dedup = ["recording_id"] if "recording_id" in out.columns else ["participant_id", "session_id", "raw_task"]
        return out.drop_duplicates(dedup)

    # Stream parquet batches rather than materializing the full bundle at once.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    first = next(pf.iter_batches(batch_size=8)).to_pandas()
    keys = _feature_key_columns(first, cfg)
    key_source_cols = {v for v in keys.values() if v}
    meta_exclude = set(key_source_cols)
    meta_exclude.update(c for c in first.columns if re.sub(r"[^a-z0-9]", "", str(c).lower()) == "nframes")

    payload_col = spec.get("value_column") or detect_array_column(first, exclude=meta_exclude)
    if not payload_col:
        non_id = [c for c in first.columns if c not in meta_exclude]
        if len(non_id) == 1:
            payload_col = non_id[0]
        else:
            raise KeyError(
                f"Could not infer feature payload column for {name}. Columns={list(first.columns)}. "
                f"Set representations.{name}.value_column in config/atlas.yaml."
            )

    read_cols = [c for c in key_source_cols if c is not None] + [payload_col]
    # Preserve file-column order and remove accidental duplicates.
    read_cols = list(dict.fromkeys(read_cols))
    chunks = []
    for batch in pf.iter_batches(columns=read_cols, batch_size=2048):
        b = batch.to_pandas()
        vecs = [vectorize_payload(v, expected_dim, pooling) for v in b[payload_col]]
        key_df = _standardize_feature_keys(b, keys)
        key_df["vector"] = vecs
        chunks.append(key_df)

    if chunks:
        out = pd.concat(chunks, ignore_index=True)
    else:
        if keys["recording_id"]:
            out = pd.DataFrame(columns=["recording_id", "vector"])
        else:
            out = pd.DataFrame(columns=["participant_id", "session_id", "raw_task", "vector"])
    out["representation"] = name
    dedup = ["recording_id"] if "recording_id" in out.columns else ["participant_id", "session_id", "raw_task"]
    return out.drop_duplicates(dedup)

def aggregate_representation(
    recordings: pd.DataFrame,
    feature_df: pd.DataFrame,
    group_cols: list[str],
    selected_recordings: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    merged = recordings.merge(feature_df[["recording_id", "vector"]], on="recording_id", how="inner")
    if selected_recordings is not None:
        keep = merged.apply(lambda r: r["recording_id"] in selected_recordings.get(str(r["participant_id"]), set()), axis=1)
        merged = merged.loc[keep]
    rows = []
    for keys, g in merged.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        arrs = [np.asarray(v, dtype=float) for v in g["vector"]]
        dims = {a.shape for a in arrs}
        if len(dims) != 1:
            raise ValueError(f"Inconsistent vector dimensions in group {keys}: {dims}")
        vec = np.nanmean(np.stack(arrs), axis=0)
        row = dict(zip(group_cols, keys))
        row.update({"vector": vec, "n_recordings": len(g)})
        rows.append(row)
    return pd.DataFrame(rows)


def combine_participant_representations(rep_tables: dict[str, pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    merged = None
    for name, df in rep_tables.items():
        tmp = df[keys + ["vector", "n_recordings"]].copy()
        tmp = tmp.rename(columns={"vector": f"vector__{name}", "n_recordings": f"n_recordings__{name}"})
        merged = tmp if merged is None else merged.merge(tmp, on=keys, how="inner")
    if merged is None or merged.empty:
        return pd.DataFrame(columns=keys + ["vector", "n_recordings"])
    vec_cols = [c for c in merged.columns if c.startswith("vector__")]
    rec_cols = [c for c in merged.columns if c.startswith("n_recordings__")]
    merged["vector"] = merged[vec_cols].apply(lambda r: np.concatenate([np.asarray(v, dtype=float) for v in r]), axis=1)
    merged["n_recordings"] = merged[rec_cols].min(axis=1)
    return merged[keys + ["vector", "n_recordings"]]


def sample_recording_ids(recordings: pd.DataFrame, m: int, seed: int) -> dict[str, set[str]]:
    rng = np.random.default_rng(seed)
    selected: dict[str, set[str]] = {}
    for pid, g in recordings.groupby("participant_id"):
        ids = g["recording_id"].astype(str).unique()
        if len(ids) >= m:
            chosen = rng.choice(ids, size=m, replace=False)
            selected[str(pid)] = set(map(str, chosen))
    return selected

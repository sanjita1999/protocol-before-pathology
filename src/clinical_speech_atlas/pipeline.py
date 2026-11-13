from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .config import output_dir, resolve_data_path, task_map_path
from .data import (
    aggregate_representation,
    apply_task_mapping,
    build_unique_diagnosis_cohort,
    combine_participant_representations,
    extract_covariates,
    feature_observation_id,
    load_recording_metadata,
    load_recording_representation,
    normalize_id_series,
    load_task_mapping,
    sample_recording_ids,
)
from .model import (
    run_covariate_only, run_grouped_probe, run_speech_plus_covariates,
    run_tabular_probe, run_mlp_probe, run_gated_attention_mil,
)
from .schema import profile_json_schema, table_profile
from .stats import (
    shared_control_rank_bootstrap, stratified_auc_bootstrap, ranking_kendall,
    condition_rank_bootstrap, paired_auc_delta_bootstrap, benjamini_hochberg,
    crossed_gstudy_three_facets, sensitivity_at_specificity,
    bootstrap_sensitivity_at_specificity, calibration_summary,
)
from .utils import dataframe_sha256, ensure_dir, json_dump, runtime_manifest, stable_seed


def _save_manifest(cfg: dict[str, Any], extra: dict[str, Any] | None = None):
    out = output_dir(cfg, "logs") / "run_manifest.json"
    json_dump(runtime_manifest(cfg["_config_path"], extra=extra), out)


def audit_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["data"]["root"]).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    files = []
    rels = [
        cfg["data"]["recording_table"], cfg["data"].get("session_table"), cfg["data"].get("demographics_table"),
        cfg["data"].get("participant_table"), cfg["data"].get("enrollment_table"), cfg["data"].get("confounders_table")
    ]
    for rel in rels:
        if rel:
            files.append(table_profile(root / rel))
    diag_dir = root / cfg["data"]["diagnosis_dir"]
    for fname in [cfg["cohort"]["control_filename"], *cfg["cohort"]["disease_files"]]:
        files.append(table_profile(diag_dir / fname))
    for name, spec in cfg["representations"].items():
        files.append(table_profile(root / spec["path"]))
        sidecar = (root / spec["path"]).with_suffix(".json")
        if sidecar.exists():
            files.append(profile_json_schema(sidecar))
    report = {"dataset_root": str(root), "profiles": files}
    out = output_dir(cfg, "audit")
    ensure_dir(out)
    json_dump(report, out / "schema_audit.json")
    _save_manifest(cfg, {"stage": "audit"})
    return report


def _freeze_folds(cohort: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    included = cohort[cohort.included].copy()
    n_folds = int(cfg["project"]["n_folds"])
    base = int(cfg["project"]["seed"])
    rows = []
    for label, g in included.groupby("cohort"):
        ids = g.participant_id.astype(str).to_numpy()
        rng = np.random.default_rng(stable_seed(base, "folds", label))
        ids = ids[rng.permutation(len(ids))]
        for i, pid in enumerate(ids):
            rows.append({"participant_id": str(pid), "cohort": label, "fold": i % n_folds})
    return pd.DataFrame(rows)


def prepare_analysis(cfg: dict[str, Any]) -> dict[str, Any]:
    """Prepare the frozen participant cohort and derived-feature observations.

    Bridge2AI v3.0 dense feature files are keyed by
    ``(participant_id, session_id, task_name)``.  That key is unique in the
    feature tables and is the correct analysis observation.  We therefore do
    not force those rows onto ``phenotype/task/recording.tsv``: a single
    acoustic-task UUID in that table can legitimately contain several lower-
    level recordings (e.g. maximum-phonation-time-1/2/3).
    """
    out = output_dir(cfg, "manifests")
    emb_out = output_dir(cfg, "embeddings")
    ensure_dir(out); ensure_dir(emb_out)

    cohort, diag_audit = build_unique_diagnosis_cohort(cfg)
    cov, cov_cols = extract_covariates(cfg)
    folds = _freeze_folds(cohort, cfg)
    participants = (
        cohort[cohort.included]
        .merge(folds[["participant_id", "fold"]], on="participant_id", how="left")
        .merge(cov, on="participant_id", how="left")
    )
    participants["participant_id"] = normalize_id_series(participants["participant_id"])

    cohort.to_csv(out / "cohort_all.csv", index=False)
    participants.to_csv(out / "participants.csv", index=False)
    folds.to_csv(out / "folds.csv", index=False)
    diag_audit.to_csv(out / "diagnosis_loader_audit.csv", index=False)

    # Cache each representation and build one union observation manifest from
    # the feature tables themselves.  The union is appropriate for task
    # coverage; representation-specific analyses still use each representation's
    # own available observations.
    feature_summary = []
    observation_frames = []
    canonical_recording_metadata = None

    for name, spec in cfg["representations"].items():
        f = load_recording_representation(name, spec, cfg)

        if {"participant_id", "session_id", "raw_task"}.issubset(f.columns):
            f = f.copy()
            f["participant_id"] = normalize_id_series(f["participant_id"])
            f["session_id"] = f["session_id"].astype(str).str.strip()
            f["raw_task"] = f["raw_task"].astype(str).str.strip()
            natural = ["participant_id", "session_id", "raw_task"]

            dup = f.duplicated(natural, keep=False)
            if dup.any():
                examples = f.loc[dup, natural].drop_duplicates().head(10).to_dict("records")
                raise ValueError(
                    f"Duplicate feature observations for representation {name} under "
                    f"(participant_id, session_id, task_name). Examples={examples}."
                )

            f["recording_id"] = [
                feature_observation_id(pid, sid, task)
                for pid, sid, task in zip(f.participant_id, f.session_id, f.raw_task)
            ]
            key_mode = "feature_observation:participant_id+session_id+task_name"

        elif "recording_id" in f.columns:
            # Compatibility fallback for a representation that truly exposes a
            # recording_id.  Only this branch uses phenotype/task/recording.tsv.
            if canonical_recording_metadata is None:
                canonical_recording_metadata, _ = load_recording_metadata(cfg)
                canonical_recording_metadata["participant_id"] = normalize_id_series(
                    canonical_recording_metadata["participant_id"]
                )
            f = canonical_recording_metadata[[
                "recording_id", "participant_id", "session_id", "raw_task"
            ]].merge(
                f[["recording_id", "vector"]],
                on="recording_id", how="inner", validate="one_to_one"
            )
            key_mode = "explicit_recording_id"
        else:
            raise KeyError(
                f"Representation {name} exposes neither an explicit recording_id nor "
                "the Bridge2AI natural feature key "
                "(participant_id, session_id, task_name)."
            )

        # Restrict caches to participants in the frozen unique-diagnosis cohort.
        f = f.merge(
            participants[["participant_id", "cohort", "fold"]],
            on="participant_id", how="inner", validate="many_to_one"
        )

        # numpy arrays serialize safely as object columns in pickle.
        f.to_pickle(emb_out / f"recording_{name}.pkl")
        observation_frames.append(
            f[["recording_id", "participant_id", "session_id", "raw_task", "cohort", "fold"]]
        )
        feature_summary.append({
            "representation": name,
            "key_mode": key_mode,
            "n_recordings": len(f),
            "n_participants": f.participant_id.nunique(),
            "n_tasks": f.raw_task.nunique(),
            "vector_dim": int(len(f.vector.iloc[0])) if len(f) else 0,
        })

    if not observation_frames:
        raise RuntimeError("No representation observations were prepared")

    rec = pd.concat(observation_frames, ignore_index=True).drop_duplicates("recording_id")
    conflict = rec.groupby("recording_id")[["participant_id", "session_id", "raw_task"]].nunique().max(axis=1)
    if (conflict > 1).any():
        raise AssertionError("Synthetic feature observation ID maps to conflicting metadata")

    rec = rec.sort_values(["participant_id", "session_id", "raw_task"]).reset_index(drop=True)
    rec.to_csv(out / "recordings.csv", index=False)
    pd.DataFrame(feature_summary).to_csv(out / "feature_summary.csv", index=False)

    # Critical leakage guard.
    if folds.groupby("participant_id").fold.nunique().max() != 1:
        raise AssertionError("Participant assigned to multiple folds")

    summary = {
        "n_unique_diagnosis_all": int(cohort.get("unique_diagnosis", pd.Series(dtype=bool)).sum()),
        "n_excluded_multi_diagnosis": int((cohort.get("n_diagnoses", pd.Series(dtype=int)) > 1).sum()),
        "n_included_participants": int(participants.participant_id.nunique()),
        "n_feature_observations": int(rec.recording_id.nunique()),
        "n_raw_tasks": int(rec.raw_task.nunique()),
        "cohort_counts": participants.cohort.value_counts().to_dict(),
        "covariate_columns": cov_cols,
        "recording_columns": {
            "participant": "participant_id",
            "recording": "synthetic feature-observation id",
            "task": "task_name",
            "session": "session_id",
        },
        "feature_observation_key": ["participant_id", "session_id", "task_name"],
        "cohort_sha256": dataframe_sha256(participants[["participant_id", "cohort", "fold"]]),
    }
    json_dump(summary, out / "prepare_summary.json")
    _save_manifest(cfg, {"stage": "prepare", **summary})
    return summary


def make_task_template(cfg: dict[str, Any]) -> Path:
    rec_path = output_dir(cfg, "manifests") / "recordings.csv"
    if not rec_path.exists():
        raise FileNotFoundError("Run `speech-atlas prepare` first")
    rec = pd.read_csv(rec_path, dtype={"participant_id": str, "recording_id": str})
    raw_tasks = sorted(rec.raw_task.dropna().astype(str).unique())
    path = task_map_path(cfg)
    current = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            current = yaml.safe_load(f) or {}
    mapping = current.get("mapping") or {}
    for t in raw_tasks:
        mapping.setdefault(t, "UNMAPPED")
    obj = {
        "frozen": False,
        "mapping": mapping,
        "allowed_families": current.get("allowed_families", [
            "sustained_phonation", "rapid_articulation", "structured_reading", "structured_sentence",
            "picture_description_narrative", "spontaneous_speech", "respiratory_phonatory", "other_structured_vocal"
        ]),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
    coverage = rec.groupby("raw_task").agg(n_recordings=("recording_id", "nunique"), n_participants=("participant_id", "nunique")).reset_index()
    coverage.to_csv(output_dir(cfg, "manifests") / "raw_task_coverage.csv", index=False)
    return path


def _load_prepared(cfg: dict[str, Any], require_task_map=False):
    mdir = output_dir(cfg, "manifests")
    participants = pd.read_csv(mdir / "participants.csv", dtype={"participant_id": str})
    rec = pd.read_csv(mdir / "recordings.csv", dtype={"participant_id": str, "recording_id": str})
    if require_task_map:
        rec = apply_task_mapping(rec, cfg, require_frozen=True)
        if rec.task_family.isna().any() or rec.task_family.eq("UNMAPPED").any():
            missing = sorted(rec.loc[rec.task_family.isna() | rec.task_family.eq("UNMAPPED"), "raw_task"].unique())
            raise RuntimeError(f"Task map has unmapped raw tasks: {missing[:20]}")
    return participants, rec


def _load_recording_features(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    emb = output_dir(cfg, "embeddings")
    out = {}
    for name in cfg["representations"]:
        path = emb / f"recording_{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached embeddings {path}; run `speech-atlas prepare`")
        out[name] = pd.read_pickle(path)[["recording_id", "vector"]]
    return out


def _participant_matrix(cfg: dict[str, Any], rec: pd.DataFrame, participants: pd.DataFrame, group_cols: list[str], selected_recordings=None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    features = _load_recording_features(cfg)
    reps = {}
    for name, f in features.items():
        reps[name] = aggregate_representation(rec, f, group_cols, selected_recordings=selected_recordings)
    combined = combine_participant_representations(reps, group_cols)
    meta_cols = ["participant_id", "cohort", "fold", "age", "sex", "site"]
    if "participant_id" not in group_cols:
        raise ValueError("participant_id must be in group_cols")
    combined = combined.merge(participants[meta_cols], on="participant_id", how="inner")
    for name in reps:
        reps[name] = reps[name].merge(participants[meta_cols], on="participant_id", how="inner")
    return combined, reps


def _diseases(participants: pd.DataFrame) -> list[str]:
    return sorted(participants.loc[participants.cohort.ne("Healthy Control"), "cohort"].unique())


def _attach_ci(pred: pd.DataFrame, metrics: dict, cfg: dict[str, Any], label: str) -> dict:
    b = stratified_auc_bootstrap(pred, int(cfg["project"]["bootstrap_replicates"]), stable_seed(int(cfg["project"]["seed"]), "bootstrap", label))
    metrics = dict(metrics)
    metrics["ci_low"] = b["ci_low"]
    metrics["ci_high"] = b["ci_high"]
    return metrics


def run_disease_benchmark(cfg: dict[str, Any]) -> pd.DataFrame:
    participants, rec = _load_prepared(cfg, require_task_map=False)
    combined, _ = _participant_matrix(cfg, rec, participants, ["participant_id"])
    result_rows, pred_rows = [], []
    for d in _diseases(participants):
        pred, m = run_grouped_probe(combined, d, cfg, context="raw")
        m = _attach_ci(pred, m, cfg, f"disease|{d}")
        result_rows.append(m); pred_rows.append(pred)
    results = pd.DataFrame(result_rows).sort_values("auc", ascending=False)
    preds = pd.concat(pred_rows, ignore_index=True)
    out = output_dir(cfg, "results"); ensure_dir(out)
    results.to_csv(out / "disease_benchmark.csv", index=False)
    preds.to_csv(out / "pred_disease_benchmark.csv", index=False)
    return results


def task_support_table(cfg: dict[str, Any]) -> pd.DataFrame:
    participants, rec = _load_prepared(cfg, require_task_map=True)
    rows = []
    diseases = _diseases(participants)
    for t, gt in rec.groupby("task_family"):
        ctrl = gt[gt.cohort.eq("Healthy Control")]
        for d in diseases:
            case = gt[gt.cohort.eq(d)]
            rows.append({
                "disease": d, "task_family": t,
                "n_case": int(case.participant_id.nunique()),
                "n_control": int(ctrl.participant_id.nunique()),
                "n_case_recordings": int(case.recording_id.nunique()),
                "n_control_recordings": int(ctrl.recording_id.nunique()),
            })
    support = pd.DataFrame(rows)
    support["eligible"] = (support.n_case >= int(cfg["analysis"]["task_min_cases"])) & (support.n_control >= int(cfg["analysis"]["task_min_controls"]))
    support.to_csv(output_dir(cfg, "manifests") / "task_support.csv", index=False)
    return support


def run_disease_task(cfg: dict[str, Any]) -> pd.DataFrame:
    participants, rec = _load_prepared(cfg, require_task_map=True)
    support = task_support_table(cfg)
    combined, _ = _participant_matrix(cfg, rec, participants, ["participant_id", "task_family"])
    results, preds = [], []
    for row in support[support.eligible].itertuples(index=False):
        sub = combined[combined.task_family.eq(row.task_family)]
        pred, m = run_grouped_probe(sub, row.disease, cfg, context="raw")
        m.update({"task_family": row.task_family})
        m = _attach_ci(pred, m, cfg, f"task|{row.disease}|{row.task_family}")
        results.append(m)
        p = pred.copy(); p["task_family"] = row.task_family; preds.append(p)
    res = pd.DataFrame(results)
    out = output_dir(cfg, "results"); ensure_dir(out)
    res.to_csv(out / "disease_task.csv", index=False)
    pred_all = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    pred_all.to_csv(out / "pred_disease_task.csv", index=False)
    if not res.empty:
        variability = res.groupby("disease").agg(
            task_auc_mean=("auc","mean"),
            task_auc_sd=("auc","std"),
            task_auc_min=("auc","min"),
            task_auc_max=("auc","max"),
            n_eligible_tasks=("task_family","nunique"),
        ).reset_index()
        variability["task_auc_range"] = variability.task_auc_max - variability.task_auc_min
        variability.to_csv(out / "task_variability.csv", index=False)
    return res


def run_disease_representation(cfg: dict[str, Any]) -> pd.DataFrame:
    participants, rec = _load_prepared(cfg, require_task_map=False)
    _, reps = _participant_matrix(cfg, rec, participants, ["participant_id"])
    results, preds = [], []
    for rname, df in reps.items():
        for d in _diseases(participants):
            pred, m = run_grouped_probe(df, d, cfg, context="raw")
            m["representation"] = rname
            m = _attach_ci(pred, m, cfg, f"rep|{d}|{rname}")
            results.append(m)
            p = pred.copy(); p["representation"] = rname; preds.append(p)
    res = pd.DataFrame(results)
    out = output_dir(cfg, "results"); ensure_dir(out)
    res.to_csv(out / "disease_representation.csv", index=False)
    pred_all = pd.concat(preds, ignore_index=True)
    pred_all.to_csv(out / "pred_disease_representation.csv", index=False)
    rank_rows=[]
    for d, g in pred_all.groupby("disease"):
        rs = condition_rank_bootstrap(
            g, "representation", int(cfg["project"]["bootstrap_replicates"]),
            stable_seed(int(cfg["project"]["seed"]), "rep_rank", d)
        )
        if not rs.empty:
            rs["disease"] = d
            rank_rows.append(rs)
    if rank_rows:
        pd.concat(rank_rows, ignore_index=True).to_csv(out / "representation_rank_stability.csv", index=False)
    return res


def select_common_tasks(support: pd.DataFrame, cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    min_case = int(cfg["analysis"]["common_task_min_cases"])
    min_ctrl = int(cfg["analysis"]["common_task_min_controls"])
    min_diseases = int(cfg["analysis"]["common_min_diseases"])
    eligible = support[(support.n_case >= min_case) & (support.n_control >= min_ctrl)]
    task_sets = {t: set(g.disease) for t, g in eligible.groupby("task_family")}
    ordered = sorted(task_sets, key=lambda t: (-len(task_sets[t]), t))
    selected = []
    disease_intersection: set[str] | None = None
    for t in ordered:
        proposed = task_sets[t] if disease_intersection is None else disease_intersection & task_sets[t]
        if len(proposed) >= min_diseases:
            selected.append(t)
            disease_intersection = proposed
    if not selected or disease_intersection is None:
        raise RuntimeError("No coverage-only common task set satisfies the configured minimum disease support")
    return selected, sorted(disease_intersection)


def run_common_task(cfg: dict[str, Any]) -> pd.DataFrame:
    participants, rec = _load_prepared(cfg, require_task_map=True)
    support = task_support_table(cfg)
    tasks, diseases = select_common_tasks(support, cfg)
    rec_common = rec[rec.task_family.isin(tasks)].copy()
    combined, _ = _participant_matrix(cfg, rec_common, participants, ["participant_id"])
    rows, preds = [], []
    for d in diseases:
        pred, m = run_grouped_probe(combined, d, cfg, context="raw")
        m.update({"common_tasks": "|".join(tasks), "n_common_tasks": len(tasks)})
        m = _attach_ci(pred, m, cfg, f"common|{d}|{'|'.join(tasks)}")
        rows.append(m); preds.append(pred)
    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    baseline_path = output_dir(cfg, "results") / "disease_benchmark.csv"
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        merged = res[["disease", "auc"]].merge(base[["disease", "auc"]], on="disease", suffixes=("_common", "_pooled"))
        res = res.merge(merged.assign(
            delta_auc=merged.auc_common - merged.auc_pooled,
            rank_common=merged.auc_common.rank(ascending=False, method="min"),
            rank_pooled=merged.auc_pooled.rank(ascending=False, method="min"),
        )[["disease", "delta_auc", "rank_common", "rank_pooled"]], on="disease", how="left")
        res["delta_rank"] = res.rank_common - res.rank_pooled
        tau = ranking_kendall(
            base[base.disease.isin(diseases)][["disease", "auc"]],
            res[["disease", "auc"]],
        )
        res["kendall_tau_vs_pooled"] = tau
    out = output_dir(cfg, "results")
    res.to_csv(out / "common_task.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out / "pred_common_task.csv", index=False)
    json_dump({"selected_tasks": tasks, "selected_diseases": diseases}, output_dir(cfg, "manifests") / "common_task_selection.json")
    return res



def augment_common_task_paired_delta_ci(cfg: dict[str, Any]) -> pd.DataFrame:
    """Add paired participant bootstrap uncertainty to pooled-vs-shared-task deltas.

    This intentionally does not rerun either classifier. It reuses the frozen
    out-of-fold predictions from ``disease_benchmark`` and ``common_task`` and
    evaluates both scores on the exact same participants for each disease.

    ``delta_auc`` in ``common_task.csv`` remains the original marginal
    shared-task minus pooled AUC difference. The new ``paired_delta_auc`` and
    confidence interval are the inferential comparison and use only participants
    present in both prediction tables.
    """
    out = output_dir(cfg, "results")
    common_path = out / "common_task.csv"
    common_pred_path = out / "pred_common_task.csv"
    pooled_pred_path = out / "pred_disease_benchmark.csv"
    for p in [common_path, common_pred_path, pooled_pred_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required frozen result: {p}")

    common = pd.read_csv(common_path)
    pred_common = pd.read_csv(common_pred_path, dtype={"participant_id": str})
    pred_pooled = pd.read_csv(pooled_pred_path, dtype={"participant_id": str})
    B = int(cfg["project"]["bootstrap_replicates"])
    base_seed = int(cfg["project"]["seed"])

    rows = []
    for row in common.to_dict("records"):
        d = row["disease"]
        a = pred_pooled[pred_pooled.disease.eq(d)].copy()
        b = pred_common[pred_common.disease.eq(d)].copy()
        x = a[["participant_id", "y_true", "y_score"]].merge(
            b[["participant_id", "y_true", "y_score"]],
            on=["participant_id", "y_true"], suffixes=("_pooled", "_common"),
            validate="one_to_one",
        )
        if x.y_true.nunique() < 2:
            raise ValueError(f"Paired common-task comparison for {d} lacks both classes")
        paired_observed = float(
            roc_auc_score(x.y_true, x.y_score_common)
            - roc_auc_score(x.y_true, x.y_score_pooled)
        )
        delta = paired_auc_delta_bootstrap(
            a, b, B,
            stable_seed(base_seed, "common_paired_delta", d),
        )
        row.update({
            "paired_delta_auc": paired_observed,
            "paired_delta_ci_low": float(delta["delta_ci_low"]),
            "paired_delta_ci_high": float(delta["delta_ci_high"]),
            "paired_delta_boot_mean": float(delta["delta_auc"]),
            "n_paired": int(len(x)),
            "n_paired_case": int(x.y_true.sum()),
            "n_paired_control": int((1 - x.y_true).sum()),
        })
        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(common_path, index=False)
    return res


def _site_overlap_diagnostics(participants: pd.DataFrame, disease: str) -> tuple[pd.DataFrame, bool]:
    sub = participants[participants.cohort.isin(["Healthy Control", disease])].copy()
    tab = sub.groupby(["site","cohort"], dropna=False).participant_id.nunique().unstack(fill_value=0)
    rows=[]
    overlap_sites=0
    for site, r in tab.iterrows():
        nc=int(r.get(disease,0)); nh=int(r.get("Healthy Control",0))
        if nc >= 5 and nh >= 5:
            overlap_sites += 1
        rows.append({"disease":disease,"site":site,"n_case":nc,"n_control":nh})
    identifiable = overlap_sites >= 2
    return pd.DataFrame(rows), identifiable


def run_adjustments(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants, rec = _load_prepared(cfg, require_task_map=False)
    combined, _ = _participant_matrix(cfg, rec, participants, ["participant_id"])
    rows, preds, cov_rows, cov_preds, site_rows = [], [], [], [], []
    cov_cfg = cfg.get("covariates", {})
    use_age = bool(cov_cfg.get("age", True)) and participants["age"].notna().any()
    use_sex = bool(cov_cfg.get("sex", True)) and participants["sex"].notna().any()
    use_site = bool(cov_cfg.get("site", False)) and participants["site"].notna().any()

    contexts = ["raw"]
    cov_only_contexts = []
    if use_age:
        contexts.append("age"); cov_only_contexts.append("age")
    if use_sex:
        contexts.append("sex"); cov_only_contexts.append("sex")
    if use_age and use_sex:
        contexts.append("age_sex"); cov_only_contexts.append("age_sex")
    if use_site:
        contexts.append("site"); cov_only_contexts.append("site")
        if use_age and use_sex:
            contexts.append("joint"); cov_only_contexts.append("joint")

    for d in _diseases(participants):
        site_identifiable = False
        if use_site:
            site_tab, site_identifiable = _site_overlap_diagnostics(participants, d)
            site_rows.append(site_tab)
        for context in contexts:
            pred, m = run_grouped_probe(combined, d, cfg, context=context)
            m = _attach_ci(pred, m, cfg, f"adjust|{d}|{context}")
            m["site_overlap_identifiable"] = bool(site_identifiable) if context in {"site","joint"} else np.nan
            rows.append(m); p = pred.copy(); p["context"] = context; preds.append(p)
        for context in cov_only_contexts:
            p, m = run_covariate_only(combined, d, context, cfg)
            m = _attach_ci(p, m, cfg, f"covonly|{d}|{context}")
            cov_rows.append(m); pp=p.copy(); pp["context"] = context; cov_preds.append(pp)
    res = pd.DataFrame(rows)
    cov = pd.DataFrame(cov_rows)
    raw_auc = res[res.context.eq("raw")][["disease", "auc"]].rename(columns={"auc": "auc_raw"})
    res = res.merge(raw_auc, on="disease", how="left")
    res["delta_auc_vs_raw"] = res.auc - res.auc_raw
    out = output_dir(cfg, "results")
    res.to_csv(out / "adjustments.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out / "pred_adjustments.csv", index=False)
    cov.to_csv(out / "covariate_only.csv", index=False)
    pd.concat(cov_preds, ignore_index=True).to_csv(out / "pred_covariate_only.csv", index=False)
    if site_rows:
        pd.concat(site_rows, ignore_index=True).to_csv(output_dir(cfg, "manifests") / "site_overlap.csv", index=False)
    return res, cov


def _common_observation_manifest(cfg: dict[str, Any], rec: pd.DataFrame) -> pd.DataFrame:
    """Observations available in every configured representation.

    Equal-recording experiments must sample the same observation IDs for all
    representation families; otherwise ``m=10`` could mean a different number
    of usable observations in different representations.
    """
    features = _load_recording_features(cfg)
    common_ids = None
    for f in features.values():
        ids = set(f["recording_id"].astype(str))
        common_ids = ids if common_ids is None else common_ids & ids
    common_ids = common_ids or set()
    return rec[rec.recording_id.astype(str).isin(common_ids)].copy()


def _choose_primary_budget(rec: pd.DataFrame, participants: pd.DataFrame, cfg: dict[str, Any]) -> int:
    candidates = sorted(map(int, cfg["analysis"]["budget_candidates"]))
    threshold = float(cfg["analysis"]["budget_min_retention"])
    min_n = int(cfg["analysis"]["budget_select_disease_min_n"])
    disease_counts = participants[participants.cohort.ne("Healthy Control")].cohort.value_counts()
    selected_diseases = list(disease_counts[disease_counts >= min_n].index)
    counts = rec.groupby("participant_id").recording_id.nunique()
    for m in reversed(candidates):
        ok = True
        for cohort in ["Healthy Control", *selected_diseases]:
            ids = participants.loc[participants.cohort.eq(cohort), "participant_id"].astype(str)
            retention = (counts.reindex(ids).fillna(0) >= m).mean()
            if retention < threshold:
                ok = False; break
        if ok:
            return m
    return candidates[0]


def run_equal_budget(cfg: dict[str, Any]) -> pd.DataFrame:
    participants, rec = _load_prepared(cfg, require_task_map=False)
    rec = _common_observation_manifest(cfg, rec)
    primary_m = _choose_primary_budget(rec, participants, cfg)
    rows, all_preds = [], []
    base_seed = int(cfg["project"]["seed"])
    repeats = int(cfg["analysis"]["budget_repeats"])
    B = int(cfg["project"]["bootstrap_replicates"])
    diseases = _diseases(participants)
    pred_by_dm: dict[tuple[str,int], list[pd.DataFrame]] = {(d,m): [] for d in diseases for m in map(int,cfg["analysis"]["budget_candidates"])}

    for m in map(int, cfg["analysis"]["budget_candidates"]):
        auc_collect: dict[str, list[float]] = {d: [] for d in diseases}
        n_collect: dict[str, list[int]] = {d: [] for d in diseases}
        for sidx in range(repeats):
            selected = sample_recording_ids(rec, m, stable_seed(base_seed, "budget", m, sidx))
            eligible_ids = set(selected)
            rec_s = rec[rec.participant_id.astype(str).isin(eligible_ids)].copy()
            combined, _ = _participant_matrix(cfg, rec_s, participants, ["participant_id"], selected_recordings=selected)
            for d in diseases:
                sub = combined[combined.cohort.isin(["Healthy Control", d])]
                if sub[sub.cohort.eq(d)].participant_id.nunique() < 5 or sub[sub.cohort.eq("Healthy Control")].participant_id.nunique() < 5:
                    continue
                try:
                    pred, met = run_grouped_probe(combined, d, cfg, context="raw")
                except ValueError:
                    continue
                auc_collect[d].append(met["auc"])
                n_collect[d].append(met["n_case"])
                pred_by_dm[(d,m)].append(pred.copy())
                p = pred.copy(); p["budget"] = m; p["repeat"] = sidx; all_preds.append(p)

        for d, vals in auc_collect.items():
            if not vals:
                continue
            total_d = int((participants.cohort == d).sum())
            # Integrate recording-subsample and participant-sampling uncertainty:
            # each bootstrap draw first selects one completed recording subsample,
            # then resamples cases and controls within that out-of-fold prediction set.
            plist = pred_by_dm[(d,m)]
            rng = np.random.default_rng(stable_seed(base_seed, "budget_boot", d, m))
            boot_vals=[]
            for _ in range(B):
                pr = plist[int(rng.integers(0, len(plist)))]
                cases = pr[pr.y_true.eq(1)].reset_index(drop=True)
                ctrls = pr[pr.y_true.eq(0)].reset_index(drop=True)
                ci = rng.integers(0, len(cases), len(cases))
                hi = rng.integers(0, len(ctrls), len(ctrls))
                bs = pd.concat([cases.iloc[ci], ctrls.iloc[hi]], ignore_index=True)
                boot_vals.append(float(roc_auc_score(bs.y_true, bs.y_score)))
            lo, hi = np.quantile(boot_vals, [0.025,0.975])
            rows.append({
                "disease": d, "budget": m, "is_primary_budget": m == primary_m,
                "auc_mean": float(np.mean(vals)), "auc_sd_subsample": float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0,
                "ci_low": float(lo), "ci_high": float(hi),
                "n_repeats": len(vals), "mean_case_n": float(np.mean(n_collect[d])),
                "case_retention": float(np.mean(n_collect[d]) / total_d) if total_d else np.nan,
            })
    res = pd.DataFrame(rows)
    out = output_dir(cfg, "results")
    res.to_csv(out / "equal_budget.csv", index=False)
    (pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()).to_csv(out / "pred_equal_budget.csv", index=False)
    json_dump({"primary_budget": primary_m, "selection_rule": "largest candidate retaining configured minimum proportion in controls and disease cohorts with N>=budget_select_disease_min_n"}, output_dir(cfg, "manifests") / "budget_selection.json")
    return res


def run_ranking(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = output_dir(cfg, "results") / "pred_disease_benchmark.csv"
    pred = pd.read_csv(path, dtype={"participant_id": str})
    boot, summary, pair = shared_control_rank_bootstrap(pred, int(cfg["project"]["bootstrap_replicates"]), stable_seed(int(cfg["project"]["seed"]), "rank"), int(cfg["analysis"]["rank_top_k"]))
    out = output_dir(cfg, "results")
    boot.to_csv(out / "rank_bootstrap_long.csv", index=False)
    summary.to_csv(out / "rank_summary.csv", index=False)
    pair.to_csv(out / "pairwise_superiority.csv", index=False)
    return summary, pair



def _permute_cohort_within_folds(sub: pd.DataFrame, disease: str, rng: np.random.Generator) -> pd.Series:
    """Permute case/control labels within frozen folds while preserving fold balance."""
    out = sub["cohort"].copy()
    for _, idx in sub.groupby("fold", sort=True).groups.items():
        labels = sub.loc[idx, "cohort"].to_numpy().copy()
        out.loc[idx] = rng.permutation(labels)
    return out


def _run_permutation_for_disease(
    disease: str,
    sub: pd.DataFrame,
    cfg: dict[str, Any],
    repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Worker for one disease's fold-stratified permutation null."""
    rng = np.random.default_rng(
        stable_seed(int(cfg["project"]["seed"]), "perm_within_fold", disease, repeats)
    )
    rows: list[dict[str, Any]] = []
    success = 0
    attempts = 0
    max_attempts = max(repeats * 5, repeats + 25)
    last_error = None
    while success < repeats and attempts < max_attempts:
        attempts += 1
        tmp = sub.copy()
        tmp["cohort"] = _permute_cohort_within_folds(tmp, disease, rng)
        try:
            _, m = run_grouped_probe(tmp, disease, cfg)
        except Exception as e:
            last_error = repr(e)
            continue
        rows.append({
            "disease": disease,
            "repeat": success,
            "attempt": attempts,
            "auc": float(m["auc"]),
            "permutation_scheme": "within_fold",
        })
        success += 1
    summary = {
        "disease": disease,
        "requested": repeats,
        "successful": success,
        "attempts": attempts,
        "last_error": last_error,
    }
    return rows, summary


def run_permutation_ablation(cfg: dict[str, Any], repeats: int | None = None) -> pd.DataFrame:
    """Run an exact-count, fold-stratified participant-label permutation null.

    Exactly ``repeats`` successful null draws are required for every disease.
    Disease cohorts are parallelized using ``runtime.n_jobs``. Unlike the old
    global permutation with skip-on-error, within-fold shuffling preserves the
    case/control count in every frozen fold and therefore remains valid for the
    smallest cohort.
    """
    participants, rec = _load_prepared(cfg, require_task_map=False)
    combined, _ = _participant_matrix(cfg, rec, participants, ["participant_id"])
    out = output_dir(cfg, "results")
    R = int(repeats if repeats is not None else cfg["analysis"]["permutation_repeats"])
    if R < 1:
        raise ValueError("Permutation repeats must be >= 1")

    jobs = []
    for d in _diseases(participants):
        sub = combined[combined.cohort.isin(["Healthy Control", d])].copy().reset_index(drop=True)
        jobs.append((d, sub))

    n_jobs = int(cfg.get("runtime", {}).get("n_jobs", 1))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_permutation_for_disease)(d, sub, cfg, R)
        for d, sub in jobs
    )

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for disease_rows, disease_summary in results:
        rows.extend(disease_rows)
        summary.append(disease_summary)
        if int(disease_summary["successful"]) != R:
            raise RuntimeError(
                f"Permutation null incomplete for {disease_summary['disease']}: "
                f"requested={R}, successful={disease_summary['successful']}, "
                f"attempts={disease_summary['attempts']}, "
                f"last_error={disease_summary['last_error']}"
            )

    res = pd.DataFrame(rows).sort_values(["disease", "repeat"]).reset_index(drop=True)
    res.to_csv(out / "ablation_permutation.csv", index=False)
    json_dump({
        "permutation_scheme": "participant labels permuted within each frozen fold",
        "requested_per_disease": R,
        "diseases": summary,
    }, output_dir(cfg, "manifests") / "permutation_summary.json")
    return res


def run_final_fixes(cfg: dict[str, Any], permutation_repeats: int = 100) -> None:
    """Run only the two post-review analyses required before manuscript lock."""
    common = augment_common_task_paired_delta_ci(cfg)
    perm = run_permutation_ablation(cfg, repeats=permutation_repeats)
    _save_manifest(cfg, {
        "stage": "final_fixes_complete",
        "paired_common_task_diseases": int(common.disease.nunique()),
        "permutation_repeats_per_disease": int(permutation_repeats),
        "permutation_rows": int(len(perm)),
    })



def _repeated_fold_map(participants: pd.DataFrame, cfg: dict[str, Any], repeat: int) -> pd.DataFrame:
    """Create a deterministic participant-disjoint fold assignment for one repeat."""
    n_folds = int(cfg["project"]["n_folds"])
    base = int(cfg["project"]["seed"])
    rows = []
    for label, g in participants.groupby("cohort", sort=True):
        ids = g.participant_id.astype(str).to_numpy()
        rng = np.random.default_rng(stable_seed(base, "repeated_cv", repeat, label))
        ids = ids[rng.permutation(len(ids))]
        for i, pid in enumerate(ids):
            rows.append({"participant_id": str(pid), "fold_new": int(i % n_folds)})
    return pd.DataFrame(rows)


def run_repeated_cv_sensitivity(cfg: dict[str, Any], repeats: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify sensitivity of pooled AUC/rank to participant-fold assignment.

    This is deliberately separate from the participant bootstrap: each repeat
    refits the full frozen probe under a new participant-disjoint 5-fold split.
    It therefore measures CV-partition/model-refit sensitivity rather than
    sampling uncertainty conditional on one set of OOF fits.
    """
    participants, rec = _load_prepared(cfg, require_task_map=False)
    combined, _ = _participant_matrix(cfg, rec, participants, ["participant_id"])
    R = int(repeats if repeats is not None else cfg.get("acceptance_upgrade", {}).get("repeated_cv_repeats", 20))
    diseases = _diseases(participants)
    primary_path = output_dir(cfg, "results") / "disease_benchmark.csv"
    primary = pd.read_csv(primary_path) if primary_path.exists() else run_disease_benchmark(cfg)
    primary = primary[primary.disease.isin(diseases)][["disease", "auc"]].copy()
    primary["primary_rank"] = primary.auc.rank(ascending=False, method="average")

    rows = []
    for rep in range(R):
        fmap = _repeated_fold_map(participants, cfg, rep)
        tmp = combined.drop(columns=["fold"]).merge(fmap, on="participant_id", how="left", validate="many_to_one")
        tmp = tmp.rename(columns={"fold_new": "fold"})
        rep_rows = []
        for d in diseases:
            pred, m = run_grouped_probe(tmp, d, cfg, context="raw")
            rep_rows.append({"repeat": rep, "disease": d, "auc": float(m["auc"]),
                             "n_case": int(m["n_case"]), "n_control": int(m["n_control"])})
        rdf = pd.DataFrame(rep_rows)
        rdf["rank"] = rdf.auc.rank(ascending=False, method="average")
        merged = rdf.merge(primary, on="disease", how="inner")
        tau = float(merged[["rank", "primary_rank"]].corr(method="kendall").iloc[0, 1])
        rdf["kendall_tau_vs_primary"] = tau
        rows.extend(rdf.to_dict("records"))

    res = pd.DataFrame(rows)
    summary = res.groupby("disease").agg(
        auc_mean=("auc", "mean"),
        auc_sd=("auc", "std"),
        auc_min=("auc", "min"),
        auc_max=("auc", "max"),
        auc_q05=("auc", lambda x: float(np.quantile(x, 0.05))),
        auc_q95=("auc", lambda x: float(np.quantile(x, 0.95))),
        median_rank=("rank", "median"),
        rank_min=("rank", "min"),
        rank_max=("rank", "max"),
        p_top3=("rank", lambda x: float(np.mean(np.asarray(x) <= 3))),
    ).reset_index()
    summary["repeats"] = R
    out = output_dir(cfg, "results"); ensure_dir(out)
    res.to_csv(out / "repeated_cv.csv", index=False)
    summary.to_csv(out / "repeated_cv_summary.csv", index=False)
    json_dump({
        "repeats": R, "n_folds": int(cfg["project"]["n_folds"]),
        "mean_kendall_tau_vs_primary": float(res.groupby("repeat").kendall_tau_vs_primary.first().mean()),
        "interpretation": "CV partition sensitivity; not a replacement for participant bootstrap confidence intervals",
    }, output_dir(cfg, "manifests") / "repeated_cv_summary.json")
    return res, summary


def run_speech_demographics_incremental(cfg: dict[str, Any]) -> pd.DataFrame:
    """Test incremental value of speech beyond age+sex and vice versa."""
    participants, rec = _load_prepared(cfg, require_task_map=False)
    combined, _ = _participant_matrix(cfg, rec, participants, ["participant_id"])
    B = int(cfg["project"]["bootstrap_replicates"])
    base = int(cfg["project"]["seed"])
    rows, pred_rows = [], []

    pooled_path = output_dir(cfg, "results") / "pred_disease_benchmark.csv"
    pooled_all = pd.read_csv(pooled_path, dtype={"participant_id": str}) if pooled_path.exists() else None
    cov_path = output_dir(cfg, "results") / "pred_covariate_only.csv"
    cov_all = pd.read_csv(cov_path, dtype={"participant_id": str}) if cov_path.exists() else None

    for d in _diseases(participants):
        p_combo, m_combo = run_speech_plus_covariates(combined, d, cfg, context="age_sex")
        m_combo = _attach_ci(p_combo, m_combo, cfg, f"speech_demo|{d}")

        if pooled_all is not None:
            p_speech = pooled_all[pooled_all.disease.eq(d)].copy()
        else:
            p_speech, _ = run_grouped_probe(combined, d, cfg, context="raw")

        if cov_all is not None and "context" in cov_all.columns:
            p_demo = cov_all[(cov_all.disease.eq(d)) & (cov_all.context.eq("age_sex"))].copy()
        else:
            p_demo, _ = run_covariate_only(combined, d, "age_sex", cfg)

        if p_demo.empty:
            raise RuntimeError(f"No age+sex-only predictions available for {d}")

        delta_vs_demo = paired_auc_delta_bootstrap(
            p_demo, p_combo, B, stable_seed(base, "speech_demo_vs_demo", d)
        )
        delta_vs_speech = paired_auc_delta_bootstrap(
            p_speech, p_combo, B, stable_seed(base, "speech_demo_vs_speech", d)
        )
        auc_speech = float(roc_auc_score(p_speech.y_true, p_speech.y_score))
        auc_demo = float(roc_auc_score(p_demo.y_true, p_demo.y_score))
        rows.append({
            **m_combo,
            "auc_speech_only": auc_speech,
            "auc_age_sex_only": auc_demo,
            "delta_combo_minus_demo": float(m_combo["auc"] - auc_demo),
            "delta_combo_minus_demo_ci_low": delta_vs_demo["delta_ci_low"],
            "delta_combo_minus_demo_ci_high": delta_vs_demo["delta_ci_high"],
            "delta_combo_minus_demo_p": delta_vs_demo["p_bootstrap_two_sided"],
            "delta_combo_minus_speech": float(m_combo["auc"] - auc_speech),
            "delta_combo_minus_speech_ci_low": delta_vs_speech["delta_ci_low"],
            "delta_combo_minus_speech_ci_high": delta_vs_speech["delta_ci_high"],
            "delta_combo_minus_speech_p": delta_vs_speech["p_bootstrap_two_sided"],
        })
        pp = p_combo.copy(); pp["context"] = "speech_age_sex"; pred_rows.append(pp)

    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    out = output_dir(cfg, "results"); ensure_dir(out)
    res.to_csv(out / "speech_plus_demographics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(out / "pred_speech_plus_demographics.csv", index=False)
    return res


def run_matched_task_fdr(cfg: dict[str, Any]) -> pd.DataFrame:
    """Rerun matched task contrasts and control multiplicity with BH FDR."""
    res = run_matched_task_pairs(cfg)
    if res.empty or "p_bootstrap_two_sided" not in res.columns:
        raise RuntimeError("Matched-task bootstrap p-values were not generated")
    res = res.copy()
    res["q_bh_within_disease"] = np.nan
    for d, idx in res.groupby("disease").groups.items():
        res.loc[idx, "q_bh_within_disease"] = benjamini_hochberg(
            res.loc[idx, "p_bootstrap_two_sided"].astype(float).tolist()
        )
    res["q_bh_global"] = benjamini_hochberg(res["p_bootstrap_two_sided"].astype(float).tolist())
    res["significant_fdr_0_05"] = res["q_bh_within_disease"] < 0.05
    res.to_csv(output_dir(cfg, "results") / "matched_task_pairs.csv", index=False)
    summary = res.groupby("disease").agg(
        n_tested=("p_bootstrap_two_sided", "size"),
        n_fdr_0_05=("significant_fdr_0_05", "sum"),
        min_q=("q_bh_within_disease", "min"),
        max_abs_delta=("delta_auc", lambda x: float(np.max(np.abs(np.asarray(x, dtype=float))))),
    ).reset_index()
    summary.to_csv(output_dir(cfg, "results") / "matched_task_fdr_summary.csv", index=False)
    return res


def run_statistical_hardening(cfg: dict[str, Any], repeated_cv_repeats: int | None = None) -> None:
    run_repeated_cv_sensitivity(cfg, repeats=repeated_cv_repeats)
    run_speech_demographics_incremental(cfg)
    run_matched_task_fdr(cfg)
    _save_manifest(cfg, {"stage": "statistical_hardening_complete"})


def _participant_nonspeech_table(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Build participant-level protocol/completion features without acoustic values.

    Feature-observation duration is intentionally not included: Bridge2AI dense
    feature observations are keyed by participant/session/task and are not in a
    one-to-one relation with lower-level recording.tsv rows.
    """
    participants, rec = _load_prepared(cfg, require_task_map=True)
    fams = sorted(rec.task_family.dropna().astype(str).unique())
    base = participants[["participant_id", "cohort", "fold", "age", "sex", "site"]].copy()
    agg = rec.groupby("participant_id").agg(
        n_sessions=("session_id", "nunique"),
        n_observations=("recording_id", "nunique"),
        n_raw_tasks=("raw_task", "nunique"),
        n_task_families=("task_family", "nunique"),
    ).reset_index()
    counts = rec.pivot_table(index="participant_id", columns="task_family", values="recording_id",
                             aggfunc="nunique", fill_value=0)
    counts.columns = [f"count__{c}" for c in counts.columns]
    counts = counts.reset_index()
    out = base.merge(agg, on="participant_id", how="left").merge(counts, on="participant_id", how="left")
    count_cols = [f"count__{f}" for f in fams]
    for c in count_cols:
        if c not in out:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out[c.replace("count__", "done__")] = (out[c] > 0).astype(float)
    summary_cols = ["n_sessions", "n_observations", "n_raw_tasks", "n_task_families"]
    for c in summary_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    completion_cols = [c.replace("count__", "done__") for c in count_cols]
    return out, completion_cols, summary_cols + count_cols, fams


def run_nonspeech_baselines(cfg: dict[str, Any]) -> pd.DataFrame:
    """Validity audit: quantify disease signal available without acoustics."""
    table, completion_cols, protocol_cols, fams = _participant_nonspeech_table(cfg)
    B = int(cfg["project"]["bootstrap_replicates"]); base_seed = int(cfg["project"]["seed"])
    pred_speech_path = output_dir(cfg, "results") / "pred_disease_benchmark.csv"
    pred_speech_all = pd.read_csv(pred_speech_path, dtype={"participant_id": str}) if pred_speech_path.exists() else None
    rows, preds = [], []
    baseline_specs = {
        "completion_pattern": (completion_cols, []),
        "protocol_counts": (protocol_cols, []),
        "demographics": (["age"], ["sex"]),
        "combined_nonspeech": (["age", *completion_cols, *protocol_cols], ["sex"]),
    }
    participants = table[["participant_id", "cohort"]].drop_duplicates()
    for d in _diseases(participants):
        if pred_speech_all is not None:
            p_speech = pred_speech_all[pred_speech_all.disease.eq(d)].copy()
        else:
            _, rec = _load_prepared(cfg, require_task_map=False)
            prepared, _ = _participant_matrix(cfg, rec, table, ["participant_id"])
            p_speech, _ = run_grouped_probe(prepared, d, cfg)
        auc_speech = float(roc_auc_score(p_speech.y_true, p_speech.y_score))
        for name, (numeric, categorical) in baseline_specs.items():
            pred, met = run_tabular_probe(table, d, cfg, numeric, categorical, context=name)
            met = _attach_ci(pred, met, cfg, f"nonspeech|{d}|{name}")
            delta = paired_auc_delta_bootstrap(pred, p_speech, B, stable_seed(base_seed, "speech_minus_nonspeech", d, name))
            met.update({
                "auc_speech_only": auc_speech,
                "delta_speech_minus_baseline": float(auc_speech - met["auc"]),
                "delta_speech_minus_baseline_ci_low": delta["delta_ci_low"],
                "delta_speech_minus_baseline_ci_high": delta["delta_ci_high"],
                "delta_speech_minus_baseline_p": delta["p_bootstrap_two_sided"],
            })
            rows.append(met)
            pp = pred.copy(); pp["baseline"] = name; preds.append(pp)
    res = pd.DataFrame(rows)
    out = output_dir(cfg, "results"); ensure_dir(out)
    res.to_csv(out / "nonspeech_baselines.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out / "pred_nonspeech_baselines.csv", index=False)
    json_dump({
        "task_families": fams,
        "baselines": list(baseline_specs),
        "duration_included": False,
        "duration_reason": "feature observations are not one-to-one with lower-level recording.tsv rows",
    }, output_dir(cfg, "manifests") / "nonspeech_baseline_spec.json")
    return res


def _sample_cases_by_fold(cases: pd.DataFrame, n: int, seed: int) -> list[str]:
    folds = sorted(cases.fold.dropna().unique())
    base_n, rem = divmod(int(n), len(folds))
    rng = np.random.default_rng(seed)
    chosen = []
    for j, fold in enumerate(folds):
        g = cases[cases.fold.eq(fold)].participant_id.astype(str).to_numpy()
        take = base_n + (1 if j < rem else 0)
        if len(g) < take:
            raise ValueError(f"Fold {fold} has only {len(g)} cases; need {take}")
        chosen.extend(rng.choice(g, size=take, replace=False).tolist())
    return chosen


def run_equal_cohort_size(cfg: dict[str, Any], case_n: int | None = None, repeats: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold case-cohort size fixed while retaining the shared healthy-control pool."""
    participants, rec = _load_prepared(cfg, require_task_map=False)
    combined, _ = _participant_matrix(cfg, rec, participants, ["participant_id"])
    up = cfg.get("acceptance_upgrade", {})
    n = int(case_n if case_n is not None else up.get("equal_cohort_case_n", 30))
    R = int(repeats if repeats is not None else up.get("equal_cohort_repeats", 20))
    counts = participants[participants.cohort.ne("Healthy Control")].cohort.value_counts()
    diseases = sorted(counts[counts >= n].index)
    primary = pd.read_csv(output_dir(cfg, "results") / "disease_benchmark.csv")
    primary = primary[primary.disease.isin(diseases)][["disease", "auc"]].copy()
    primary["primary_rank"] = primary.auc.rank(ascending=False, method="average")
    control_ids = set(participants.loc[participants.cohort.eq("Healthy Control"), "participant_id"].astype(str))
    rows = []; base = int(cfg["project"]["seed"])
    for rep in range(R):
        rr = []
        for d in diseases:
            cases = combined[combined.cohort.eq(d)][["participant_id", "fold"]].drop_duplicates()
            chosen = set(_sample_cases_by_fold(cases, n, stable_seed(base, "equal_cohort", d, rep)))
            sub = combined[combined.participant_id.astype(str).isin(chosen | control_ids)].copy()
            _, met = run_grouped_probe(sub, d, cfg)
            rr.append({"repeat": rep, "disease": d, "auc": float(met["auc"]), "case_n": n, "n_control": int(met["n_control"])})
        rdf = pd.DataFrame(rr); rdf["rank"] = rdf.auc.rank(ascending=False, method="average")
        m = rdf.merge(primary, on="disease")
        tau = float(m[["rank", "primary_rank"]].corr(method="kendall").iloc[0,1])
        rdf["kendall_tau_vs_primary"] = tau
        rows.extend(rdf.to_dict("records"))
    res = pd.DataFrame(rows)
    summary = res.groupby("disease").agg(
        auc_mean=("auc","mean"), auc_sd=("auc","std"),
        auc_q025=("auc",lambda x:float(np.quantile(x,0.025))),
        auc_q975=("auc",lambda x:float(np.quantile(x,0.975))),
        median_rank=("rank","median"), rank_min=("rank","min"), rank_max=("rank","max"),
        p_top3=("rank",lambda x:float(np.mean(np.asarray(x)<=3))),
    ).reset_index()
    summary["case_n"] = n; summary["repeats"] = R
    out=output_dir(cfg,"results"); res.to_csv(out/"equal_cohort_size.csv",index=False); summary.to_csv(out/"equal_cohort_size_summary.csv",index=False)
    return res, summary


def _smd_numeric(case: pd.Series, ctrl: pd.Series) -> float:
    a=pd.to_numeric(case,errors="coerce").dropna().to_numpy(float); b=pd.to_numeric(ctrl,errors="coerce").dropna().to_numpy(float)
    if len(a)<2 or len(b)<2: return np.nan
    sd=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2.0)
    return float((np.mean(a)-np.mean(b))/sd) if sd>0 else 0.0


def _smd_binary(case: pd.Series, ctrl: pd.Series) -> float:
    cats=sorted(set(case.dropna().astype(str))|set(ctrl.dropna().astype(str)))
    if len(cats)<2: return 0.0
    target=cats[0]; p1=float((case.astype(str)==target).mean()); p0=float((ctrl.astype(str)==target).mean())
    denom=np.sqrt((p1*(1-p1)+p0*(1-p0))/2.0)
    return float((p1-p0)/denom) if denom>0 else 0.0


def _greedy_age_sex_match(participants: pd.DataFrame, disease: str, caliper_sd: float, seed: int) -> pd.DataFrame:
    sub=participants[participants.cohort.isin(["Healthy Control",disease])].copy()
    sub=sub[sub.age.notna() & sub.sex.notna()].copy(); sub["sex_key"]=sub.sex.astype(str).str.strip().str.lower()
    cases=sub[sub.cohort.eq(disease)].copy(); ctrls=sub[sub.cohort.eq("Healthy Control")].copy()
    age_sd=float(np.nanstd(sub.age.to_numpy(float),ddof=1)); caliper=float(caliper_sd*age_sd)
    rng=np.random.default_rng(seed); pairs=[]
    for sex, cg in cases.groupby("sex_key",sort=True):
        hg=ctrls[ctrls.sex_key.eq(sex)].copy(); available=set(hg.participant_id.astype(str))
        if not available: continue
        # Match hardest cases first; deterministic random jitter breaks exact ties.
        order=[]
        ages=hg.set_index(hg.participant_id.astype(str)).age.astype(float)
        for r in cg.itertuples(index=False):
            if len(ages):
                nearest=float(np.min(np.abs(ages.to_numpy()-float(r.age))))
                order.append((nearest,float(rng.random())*1e-9,str(r.participant_id),float(r.age)))
        for _,_,pid,age in sorted(order,reverse=True):
            candidates=[h for h in available if abs(float(ages.loc[h])-age)<=caliper]
            if not candidates: continue
            h=min(candidates,key=lambda z:(abs(float(ages.loc[z])-age),z)); available.remove(h)
            pairs.append({"disease":disease,"case_id":pid,"control_id":h,"case_age":age,"control_age":float(ages.loc[h]),"sex":sex,"abs_age_diff":abs(float(ages.loc[h])-age),"caliper_years":caliper})
    return pd.DataFrame(pairs)


def run_matched_controls(cfg: dict[str, Any]) -> pd.DataFrame:
    """Age/sex matched-control sensitivity analysis with frozen deterministic matches."""
    participants, rec=_load_prepared(cfg,require_task_map=False); combined,_=_participant_matrix(cfg,rec,participants,["participant_id"])
    caliper_sd=float(cfg.get("acceptance_upgrade",{}).get("matched_age_caliper_sd",0.25)); base=int(cfg["project"]["seed"]); rows=[]; pair_tabs=[]
    primary=pd.read_csv(output_dir(cfg,"results")/"disease_benchmark.csv").set_index("disease")
    min_n=int(cfg["analysis"].get("task_min_cases",20))
    for d in _diseases(participants):
        pairs=_greedy_age_sex_match(participants,d,caliper_sd,stable_seed(base,"age_sex_match",d))
        if pairs.empty or len(pairs)<min_n:
            rows.append({"disease":d,"status":"insufficient_matches","n_pairs":len(pairs),"auc":np.nan}); continue
        pair_tabs.append(pairs)
        ids=set(pairs.case_id.astype(str))|set(pairs.control_id.astype(str)); sub=combined[combined.participant_id.astype(str).isin(ids)].copy()
        try:
            pred,met=run_grouped_probe(sub,d,cfg); met=_attach_ci(pred,met,cfg,f"matched_controls|{d}")
        except ValueError as e:
            rows.append({"disease":d,"status":f"fold_failure:{e}","n_pairs":len(pairs),"auc":np.nan}); continue
        before=participants[participants.cohort.isin(["Healthy Control",d])]; after=participants[participants.participant_id.astype(str).isin(ids)]
        met.update({
            "status":"ok","n_pairs":len(pairs),"auc_pooled_full":float(primary.loc[d,"auc"]) if d in primary.index else np.nan,
            "age_smd_before":_smd_numeric(before[before.cohort.eq(d)].age,before[before.cohort.eq("Healthy Control")].age),
            "age_smd_after":_smd_numeric(after[after.cohort.eq(d)].age,after[after.cohort.eq("Healthy Control")].age),
            "sex_smd_before":_smd_binary(before[before.cohort.eq(d)].sex,before[before.cohort.eq("Healthy Control")].sex),
            "sex_smd_after":_smd_binary(after[after.cohort.eq(d)].sex,after[after.cohort.eq("Healthy Control")].sex),
            "caliper_sd":caliper_sd,"mean_abs_age_diff":float(pairs.abs_age_diff.mean()),
        }); rows.append(met)
    res=pd.DataFrame(rows); out=output_dir(cfg,"results"); res.to_csv(out/"matched_controls.csv",index=False)
    if pair_tabs: pd.concat(pair_tabs,ignore_index=True).to_csv(output_dir(cfg,"manifests")/"matched_control_pairs.csv",index=False)
    return res


def run_disease_vs_disease(cfg: dict[str, Any]) -> pd.DataFrame:
    """Pairwise differential-diagnosis atlas among adequately sized disease cohorts."""
    from itertools import combinations
    participants,rec=_load_prepared(cfg,require_task_map=False); combined,_=_participant_matrix(cfg,rec,participants,["participant_id"])
    min_n=int(cfg.get("acceptance_upgrade",{}).get("disease_pair_min_n",40)); counts=participants[participants.cohort.ne("Healthy Control")].cohort.value_counts(); diseases=sorted(counts[counts>=min_n].index)
    pooled=pd.read_csv(output_dir(cfg,"results")/"disease_benchmark.csv").set_index("disease"); rows=[]
    for a,b in combinations(diseases,2):
        sub=combined[combined.cohort.isin([a,b])].copy(); sub["cohort"]=sub.cohort.replace({b:"Healthy Control"})
        pred,met=run_grouped_probe(sub,a,cfg); met=_attach_ci(pred,met,cfg,f"disease_pair|{a}|{b}")
        ref=float(np.mean([pooled.loc[a,"auc"],pooled.loc[b,"auc"]]))
        rows.append({"disease_a":a,"disease_b":b,"n_a":met["n_case"],"n_b":met["n_control"],"auc":met["auc"],"ci_low":met["ci_low"],"ci_high":met["ci_high"],"mean_vs_healthy_auc":ref,"delta_vs_mean_healthy":float(met["auc"]-ref)})
    res=pd.DataFrame(rows).sort_values("auc"); res.to_csv(output_dir(cfg,"results")/"disease_vs_disease.csv",index=False); return res


def run_prompt_heterogeneity(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare within-family raw-prompt AUC heterogeneity with between-family heterogeneity."""
    participants,rec=_load_prepared(cfg,require_task_map=True); raw,_=_participant_matrix(cfg,rec,participants,["participant_id","raw_task"])
    family_map=rec[["raw_task","task_family"]].drop_duplicates().set_index("raw_task").task_family.to_dict(); raw["task_family"]=raw.raw_task.map(family_map)
    min_case=int(cfg["analysis"]["task_min_cases"]); min_ctrl=int(cfg["analysis"]["task_min_controls"]); rows=[]
    for d in _diseases(participants):
        dsub=raw[raw.cohort.isin(["Healthy Control",d])]
        for task,g in dsub.groupby("raw_task",sort=True):
            nc=g[g.cohort.eq(d)].participant_id.nunique(); nh=g[g.cohort.eq("Healthy Control")].participant_id.nunique()
            if nc<min_case or nh<min_ctrl: continue
            try: pred,met=run_grouped_probe(g,d,cfg)
            except ValueError: continue
            rows.append({"disease":d,"raw_task":task,"task_family":family_map.get(task),"n_case":nc,"n_control":nh,"auc":met["auc"]})
    raw_auc=pd.DataFrame(rows); out=output_dir(cfg,"results"); raw_auc.to_csv(out/"raw_task_auc.csv",index=False)
    fam_path=out/"disease_task.csv"; fam=pd.read_csv(fam_path) if fam_path.exists() else run_disease_task(cfg)
    summaries=[]
    for d in sorted(raw_auc.disease.unique()):
        x=raw_auc[raw_auc.disease.eq(d)]; parts=[]
        for family,g in x.groupby("task_family"):
            if len(g)>=2:
                parts.append({"task_family":family,"n":len(g),"var":float(g.auc.var(ddof=1)),"sd":float(g.auc.std(ddof=1))})
        f=pd.DataFrame(parts); dof=float(((f.n-1).sum()) if not f.empty else 0)
        within=float((((f.n-1)*f["var"]).sum()/dof) if dof>0 else np.nan)
        fg=fam[fam.disease.eq(d)].auc.astype(float); between=float(fg.var(ddof=1)) if len(fg)>=2 else np.nan
        summaries.append({"disease":d,"n_eligible_raw_tasks":len(x),"n_families_with_2plus_prompts":len(f),"pooled_within_family_var":within,"between_family_var":between,"within_between_var_ratio":float(within/between) if np.isfinite(within) and between>0 else np.nan,"within_sd":float(np.sqrt(within)) if np.isfinite(within) else np.nan,"between_sd":float(np.sqrt(between)) if np.isfinite(between) else np.nan})
    summary=pd.DataFrame(summaries); summary.to_csv(out/"prompt_heterogeneity_summary.csv",index=False); return raw_auc,summary


def run_validity_audit(cfg: dict[str, Any]) -> None:
    run_nonspeech_baselines(cfg)
    run_equal_cohort_size(cfg)
    run_matched_controls(cfg)
    run_disease_vs_disease(cfg)
    run_prompt_heterogeneity(cfg)
    _save_manifest(cfg,{"stage":"validity_audit_complete"})



def run_measurement_decomposition(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crossed-facet G-study decomposition plus empirical protocol intervals."""
    tensor_path = output_dir(cfg, "results") / "disease_task_representation_tensor.csv"
    tensor = pd.read_csv(tensor_path) if tensor_path.exists() else run_tensor(cfg)
    components = []
    summaries = {}
    for scale in ["logit", "raw"]:
        comp, summary = crossed_gstudy_three_facets(
            tensor, "disease", "task_family", "representation", "auc", transform=scale
        )
        components.append(comp)
        summaries[scale] = summary
    comps = pd.concat(components, ignore_index=True)
    out = output_dir(cfg, "results")
    comps.to_csv(out / "gtheory_variance_components.csv", index=False)

    pooled = pd.read_csv(out / "disease_benchmark.csv")[["disease", "auc", "ci_low", "ci_high"]].rename(
        columns={"auc": "pooled_auc", "ci_low": "sampling_ci_low", "ci_high": "sampling_ci_high"}
    )
    proto = tensor.groupby("disease").auc.agg(
        protocol_auc_mean="mean",
        protocol_auc_median="median",
        protocol_auc_sd="std",
        protocol_auc_min="min",
        protocol_auc_max="max",
        n_protocols="size",
    ).reset_index()
    qs = (
        tensor.groupby("disease").auc.quantile([0.025, 0.975]).unstack().reset_index()
        .rename(columns={0.025: "protocol_interval_low", 0.975: "protocol_interval_high"})
    )
    proto = proto.merge(qs, on="disease").merge(pooled, on="disease", how="left")
    proto["sampling_ci_width"] = proto.sampling_ci_high - proto.sampling_ci_low
    proto["protocol_interval_width"] = proto.protocol_interval_high - proto.protocol_interval_low
    proto["protocol_to_sampling_width_ratio"] = proto.protocol_interval_width / proto.sampling_ci_width
    proto.to_csv(out / "protocol_intervals.csv", index=False)
    json_dump({
        "primary_scale": "logit",
        "logit": summaries["logit"],
        "raw_auc_sensitivity": summaries["raw"],
        "protocol_interval_definition": (
            "empirical 2.5th-97.5th percentile across the complete 7-task x "
            "8-representation protocol grid; this is not a sampling CI"
        ),
        "residual_note": (
            "the disease x task x representation interaction is the residual and also "
            "contains cell-estimation noise because there is one AUC per cell"
        ),
    }, output_dir(cfg, "manifests") / "gtheory_summary.json")
    return comps, proto


def run_operating_points(cfg: dict[str, Any]) -> pd.DataFrame:
    """Operating-point and calibration view of frozen pooled OOF predictions."""
    pred = pd.read_csv(
        output_dir(cfg, "results") / "pred_disease_benchmark.csv",
        dtype={"participant_id": str},
    )
    benchmark = pd.read_csv(output_dir(cfg, "results") / "disease_benchmark.csv").set_index("disease")
    B = int(cfg["project"]["bootstrap_replicates"])
    base = int(cfg["project"]["seed"])
    bins = int(cfg.get("acceptance_upgrade", {}).get("calibration_bins", 10))
    targets = cfg.get("acceptance_upgrade", {}).get("specificity_targets", [0.90, 0.95])
    rows = []
    for disease, g in pred.groupby("disease"):
        row = {
            "disease": disease,
            "auc": float(benchmark.loc[disease, "auc"]),
            "n_case": int(g.y_true.sum()),
            "n_control": int((1 - g.y_true).sum()),
        }
        for target in targets:
            sens, spec, threshold = sensitivity_at_specificity(
                g.y_true.to_numpy(), g.y_score.to_numpy(), float(target)
            )
            boot = bootstrap_sensitivity_at_specificity(
                g, float(target), B, stable_seed(base, "sens_at_spec", disease, target)
            )
            tag = str(int(round(float(target) * 100)))
            row.update({
                f"sensitivity_at_spec_{tag}": sens,
                f"achieved_specificity_{tag}": spec,
                f"threshold_spec_{tag}": threshold,
                f"sensitivity_at_spec_{tag}_ci_low": boot["ci_low"],
                f"sensitivity_at_spec_{tag}_ci_high": boot["ci_high"],
            })
        row.update(calibration_summary(g.y_true.to_numpy(), g.y_score.to_numpy(), n_bins=bins))
        rows.append(row)
    res = pd.DataFrame(rows)
    res["auc_rank"] = res.auc.rank(ascending=False, method="average")
    if "sensitivity_at_spec_90" in res.columns:
        res["sens90_rank"] = res.sensitivity_at_spec_90.rank(ascending=False, method="average")
        tau = float(res[["auc_rank", "sens90_rank"]].corr(method="kendall").iloc[0, 1])
    else:
        tau = np.nan
    res.to_csv(output_dir(cfg, "results") / "operating_point_metrics.csv", index=False)
    json_dump({
        "kendall_tau_auc_vs_sensitivity_at_90_specificity": tau,
        "specificity_targets": targets,
        "decision_curve_included": False,
        "decision_curve_reason": (
            "No disease-specific deployment prevalence assumptions were frozen; "
            "net benefit would be assumption-driven in this case-control benchmark."
        ),
    }, output_dir(cfg, "manifests") / "operating_point_summary.json")
    return res


def validate_acceptance_upgrade(cfg: dict[str, Any]) -> list[str]:
    required = [
        "repeated_cv_summary.csv",
        "speech_plus_demographics.csv",
        "matched_task_fdr_summary.csv",
        "nonspeech_baselines.csv",
        "equal_cohort_size_summary.csv",
        "matched_controls.csv",
        "disease_vs_disease.csv",
        "prompt_heterogeneity_summary.csv",
        "gtheory_variance_components.csv",
        "protocol_intervals.csv",
        "operating_point_metrics.csv",
    ]
    missing = [f for f in required if not (output_dir(cfg, "results") / f).exists()]
    errors = [f"Missing acceptance-upgrade result: {f}" for f in missing]
    if not errors:
        mt = pd.read_csv(output_dir(cfg, "results") / "matched_task_pairs.csv")
        if "q_bh_within_disease" not in mt.columns:
            errors.append("Matched task results lack BH q-values")
        rc = pd.read_csv(output_dir(cfg, "results") / "repeated_cv_summary.csv")
        if rc.empty:
            errors.append("Repeated-CV summary is empty")
        gt = pd.read_csv(output_dir(cfg, "results") / "gtheory_variance_components.csv")
        if set(gt.scale) != {"logit", "raw"}:
            errors.append("G-study must include logit primary and raw-AUC sensitivity scales")
    return errors




def _recording_bag_table(
    cfg: dict[str, Any],
    rec: pd.DataFrame,
    participants: pd.DataFrame,
    selected_recordings: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Build participant bags of recording-level combined representation vectors.

    A recording is retained only if every configured representation is available,
    matching the evidence definition used by the equal-budget experiment.
    """
    features = _load_recording_features(cfg)
    base = rec[["recording_id", "participant_id"]].drop_duplicates("recording_id").copy()
    base["recording_id"] = base["recording_id"].astype(str)
    base["participant_id"] = base["participant_id"].astype(str)
    for name, f in features.items():
        tmp = f[["recording_id", "vector"]].copy()
        tmp["recording_id"] = tmp["recording_id"].astype(str)
        tmp = tmp.rename(columns={"vector": f"vector__{name}"})
        base = base.merge(tmp, on="recording_id", how="inner")

    if selected_recordings is not None:
        keep = []
        for r in base[["participant_id", "recording_id"]].itertuples(index=False):
            keep.append(str(r.recording_id) in selected_recordings.get(str(r.participant_id), set()))
        base = base.loc[np.asarray(keep, dtype=bool)].copy()

    vec_cols = [f"vector__{name}" for name in cfg["representations"]]
    base["combined_vector"] = base[vec_cols].apply(
        lambda row: np.concatenate([np.asarray(v, dtype=float) for v in row]), axis=1
    )
    rows = []
    for pid, g in base.groupby("participant_id", sort=False):
        arr = np.stack(g["combined_vector"].to_list()).astype(float)
        rows.append({"participant_id": str(pid), "bag": arr, "n_recordings": int(len(arr))})
    bags = pd.DataFrame(rows)
    meta = participants[["participant_id", "cohort", "fold", "age", "sex", "site"]].copy()
    meta["participant_id"] = meta["participant_id"].astype(str)
    return bags.merge(meta, on="participant_id", how="inner")


def _condition_model_stability(
    primary: pd.DataFrame,
    alternate: pd.DataFrame,
    condition_col: str,
    model_name: str,
) -> pd.DataFrame:
    """Compare per-disease condition rankings under an alternate model."""
    from scipy.stats import kendalltau

    rows = []
    for disease in sorted(set(primary.disease) & set(alternate.disease)):
        a = primary[primary.disease.eq(disease)][[condition_col, "auc"]].rename(columns={"auc": "auc_primary"})
        b = alternate[alternate.disease.eq(disease)][[condition_col, "auc"]].rename(columns={"auc": "auc_alternate"})
        m = a.merge(b, on=condition_col, how="inner")
        if m.empty:
            continue
        tau = np.nan
        if len(m) >= 2:
            ra = m.auc_primary.rank(ascending=False, method="average")
            rb = m.auc_alternate.rank(ascending=False, method="average")
            tau = float(kendalltau(ra, rb).statistic)
        best_a = str(m.loc[m.auc_primary.idxmax(), condition_col])
        best_b = str(m.loc[m.auc_alternate.idxmax(), condition_col])
        rows.append({
            "disease": disease,
            "alternate_model": model_name,
            "condition_type": condition_col,
            "n_conditions": int(len(m)),
            "kendall_tau_rank": tau,
            "primary_best": best_a,
            "alternate_best": best_b,
            "best_agreement": bool(best_a == best_b),
            "mean_abs_auc_delta": float(np.mean(np.abs(m.auc_alternate - m.auc_primary))),
            "max_abs_auc_delta": float(np.max(np.abs(m.auc_alternate - m.auc_primary))),
        })
    return pd.DataFrame(rows)


def run_model_robustness(
    cfg: dict[str, Any],
    scope: str = "pooled",
    skip_mil: bool = False,
) -> None:
    """Reviewer-facing model-capacity robustness experiment.

    The primary atlas remains PLS+balanced logistic regression.  This command
    asks whether conclusions change under (i) linear SVM, (ii) a nonlinear MLP,
    and, for the pooled equal-evidence comparison, (iii) gated-attention MIL.

    ``scope='pooled'`` runs the four-model comparison on the exact same fixed
    recording budget. ``scope='atlas'`` evaluates SVM/MLP task and representation
    rank stability. ``scope='all'`` runs both.
    """
    if scope not in {"pooled", "atlas", "all"}:
        raise ValueError("scope must be one of: pooled, atlas, all")
    participants, rec0 = _load_prepared(cfg, require_task_map=True)
    out = output_dir(cfg, "results")
    ensure_dir(out)
    rob = cfg.get("model_robustness", {})

    if scope in {"pooled", "all"}:
        rec = _common_observation_manifest(cfg, rec0)
        budget = int(rob.get("recording_budget", 20))
        selected = sample_recording_ids(
            rec, budget,
            stable_seed(int(cfg["project"]["seed"]), "model_robustness_budget", budget),
        )
        eligible_ids = set(selected)
        rec_s = rec[rec.participant_id.astype(str).isin(eligible_ids)].copy()
        combined, _ = _participant_matrix(
            cfg, rec_s, participants, ["participant_id"], selected_recordings=selected
        )
        bags = _recording_bag_table(cfg, rec_s, participants, selected_recordings=selected)
        shared_ids = set(combined.participant_id.astype(str)) & set(bags.participant_id.astype(str))
        combined = combined[combined.participant_id.astype(str).isin(shared_ids)].copy()
        bags = bags[bags.participant_id.astype(str).isin(shared_ids)].copy()
        if not bags.empty and not np.all(bags.n_recordings.eq(budget)):
            raise RuntimeError("MIL equal-budget bags do not all contain the requested recording count")

        rows = []
        pred_rows = []
        for d in _diseases(participants):
            sub = combined[combined.cohort.isin(["Healthy Control", d])]
            if sub[sub.cohort.eq(d)].participant_id.nunique() < 5:
                continue
            by_model: dict[str, pd.DataFrame] = {}
            for model_name, fn_kwargs in [
                ("pls_logistic", {"kind": "grouped", "probe": "logistic"}),
                ("pls_linear_svm", {"kind": "grouped", "probe": "linear_svm"}),
                ("mlp", {"kind": "mlp"}),
            ]:
                if fn_kwargs["kind"] == "grouped":
                    pred, met = run_grouped_probe(combined, d, cfg, probe=fn_kwargs["probe"])
                else:
                    pred, met = run_mlp_probe(combined, d, cfg)
                met = _attach_ci(pred, met, cfg, f"model_robust|{budget}|{d}|{model_name}")
                met.update({"model": model_name, "recording_budget": budget, "comparison_set": "equal_recording_budget"})
                rows.append(met)
                p = pred.copy(); p["model"] = model_name; p["recording_budget"] = budget; pred_rows.append(p)
                by_model[model_name] = pred

            if not skip_mil:
                try:
                    pred, met = run_gated_attention_mil(bags, d, cfg)
                    met = _attach_ci(pred, met, cfg, f"model_robust|{budget}|{d}|gated_attention_mil")
                    met.update({"model": "gated_attention_mil", "recording_budget": budget, "comparison_set": "equal_recording_budget"})
                    rows.append(met)
                    p = pred.copy(); p["model"] = "gated_attention_mil"; p["recording_budget"] = budget; pred_rows.append(p)
                    by_model["gated_attention_mil"] = pred
                except RuntimeError as e:
                    if "requires PyTorch" in str(e):
                        raise
                    raise

            # Paired delta against the frozen linear architecture on identical participants.
            primary_pred = by_model["pls_logistic"]
            for r in rows[-len(by_model):]:
                if r["disease"] != d:
                    continue
                model_name = r["model"]
                if model_name == "pls_logistic":
                    r.update({"delta_auc_vs_pls_logistic": 0.0, "delta_ci_low": 0.0, "delta_ci_high": 0.0, "delta_p_bootstrap": 1.0})
                    continue
                delta = paired_auc_delta_bootstrap(
                    primary_pred, by_model[model_name],
                    int(cfg["project"]["bootstrap_replicates"]),
                    stable_seed(int(cfg["project"]["seed"]), "model_delta", d, model_name, budget),
                )
                # paired_auc_delta_bootstrap(a,b) reports AUC(a)-AUC(b).
                r.update({
                    "delta_auc_vs_pls_logistic": float(delta["delta_auc"]),
                    "delta_ci_low": float(delta["delta_ci_low"]),
                    "delta_ci_high": float(delta["delta_ci_high"]),
                    "delta_p_bootstrap": float(delta["p_bootstrap_two_sided"]),
                })

        pooled = pd.DataFrame(rows)
        pooled.to_csv(out / "model_robustness_equal_budget.csv", index=False)
        (pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()).to_csv(
            out / "pred_model_robustness_equal_budget.csv", index=False
        )

        if not pooled.empty:
            summary = pooled.groupby("disease").agg(
                n_models=("model", "nunique"),
                auc_min=("auc", "min"),
                auc_max=("auc", "max"),
                auc_mean=("auc", "mean"),
            ).reset_index()
            summary["auc_range_across_models"] = summary.auc_max - summary.auc_min
            best = pooled.loc[pooled.groupby("disease").auc.idxmax(), ["disease", "model", "auc"]].rename(
                columns={"model": "best_model", "auc": "best_model_auc"}
            )
            summary = summary.merge(best, on="disease", how="left")
            summary.to_csv(out / "model_robustness_equal_budget_summary.csv", index=False)

    if scope in {"atlas", "all"}:
        support = task_support_table(cfg)
        combined_task, _ = _participant_matrix(cfg, rec0, participants, ["participant_id", "task_family"])
        task_alt_rows = []
        task_pred_rows = []
        for row in support[support.eligible].itertuples(index=False):
            sub = combined_task[combined_task.task_family.eq(row.task_family)]
            for model_name, mode in [("pls_linear_svm", "svm"), ("mlp", "mlp")]:
                if mode == "svm":
                    pred, met = run_grouped_probe(sub, row.disease, cfg, probe="linear_svm")
                else:
                    pred, met = run_mlp_probe(sub, row.disease, cfg)
                met.update({"task_family": row.task_family, "model": model_name})
                task_alt_rows.append(met)
                p = pred.copy(); p["task_family"] = row.task_family; p["model"] = model_name; task_pred_rows.append(p)
        task_alt = pd.DataFrame(task_alt_rows)
        task_alt.to_csv(out / "model_robustness_task_atlas.csv", index=False)
        (pd.concat(task_pred_rows, ignore_index=True) if task_pred_rows else pd.DataFrame()).to_csv(
            out / "pred_model_robustness_task_atlas.csv", index=False
        )

        primary_task_path = out / "disease_task.csv"
        if not primary_task_path.exists():
            raise FileNotFoundError("Run `speech-atlas task` before model-robustness --scope atlas")
        primary_task = pd.read_csv(primary_task_path)
        stability = []
        for model_name, g in task_alt.groupby("model"):
            stability.append(_condition_model_stability(primary_task, g, "task_family", model_name))
        task_stability = pd.concat(stability, ignore_index=True) if stability else pd.DataFrame()
        task_stability.to_csv(out / "model_robustness_task_stability.csv", index=False)

        _, reps = _participant_matrix(cfg, rec0, participants, ["participant_id"])
        rep_alt_rows = []
        rep_pred_rows = []
        for rname, rdf in reps.items():
            for d in _diseases(participants):
                for model_name, mode in [("pls_linear_svm", "svm"), ("mlp", "mlp")]:
                    if mode == "svm":
                        pred, met = run_grouped_probe(rdf, d, cfg, probe="linear_svm")
                    else:
                        pred, met = run_mlp_probe(rdf, d, cfg)
                    met.update({"representation": rname, "model": model_name})
                    rep_alt_rows.append(met)
                    p = pred.copy(); p["representation"] = rname; p["model"] = model_name; rep_pred_rows.append(p)
        rep_alt = pd.DataFrame(rep_alt_rows)
        rep_alt.to_csv(out / "model_robustness_representation_atlas.csv", index=False)
        (pd.concat(rep_pred_rows, ignore_index=True) if rep_pred_rows else pd.DataFrame()).to_csv(
            out / "pred_model_robustness_representation_atlas.csv", index=False
        )

        primary_rep_path = out / "disease_representation.csv"
        if not primary_rep_path.exists():
            raise FileNotFoundError("Run `speech-atlas representation` before model-robustness --scope atlas")
        primary_rep = pd.read_csv(primary_rep_path)
        stability = []
        for model_name, g in rep_alt.groupby("model"):
            stability.append(_condition_model_stability(primary_rep, g, "representation", model_name))
        rep_stability = pd.concat(stability, ignore_index=True) if stability else pd.DataFrame()
        rep_stability.to_csv(out / "model_robustness_representation_stability.csv", index=False)

    json_dump({
        "stage": "model_robustness",
        "scope": scope,
        "primary_model": "PLS + class-balanced logistic regression",
        "alternate_models": ["PLS + linear SVM", "MLP on standardized participant vectors", "gated-attention MIL over recording embeddings"],
        "mil_skipped": bool(skip_mil),
        "equal_budget_for_four_model_comparison": int(rob.get("recording_budget", 20)),
        "interpretation": "robustness only; the frozen PLS+logistic atlas remains primary",
    }, output_dir(cfg, "manifests") / "model_robustness_spec.json")
    _save_manifest(cfg, {"stage": "model_robustness", "scope": scope})

def run_acceptance_upgrade(
    cfg: dict[str, Any], repeated_cv_repeats: int | None = None
) -> None:
    run_statistical_hardening(cfg, repeated_cv_repeats=repeated_cv_repeats)
    run_validity_audit(cfg)
    run_measurement_decomposition(cfg)
    run_operating_points(cfg)
    _save_manifest(cfg, {"stage": "acceptance_upgrade_complete"})

def run_ablations(cfg: dict[str, Any]) -> None:
    participants, rec = _load_prepared(cfg, require_task_map=False)
    combined, reps = _participant_matrix(cfg, rec, participants, ["participant_id"])
    out = output_dir(cfg, "results")
    # A1: alternate probe
    rows=[]
    for d in _diseases(participants):
        pred, m = run_grouped_probe(combined, d, cfg, probe="linear_svm")
        m = _attach_ci(pred, m, cfg, f"svm|{d}"); rows.append(m)
    pd.DataFrame(rows).to_csv(out / "ablation_probe.csv", index=False)

    # A2: fixed PLS components on representation atlas.
    rows=[]
    fixed_k = int(cfg["analysis"]["fixed_pls_components_ablation"])
    for rname, df in reps.items():
        for d in _diseases(participants):
            try:
                pred, m = run_grouped_probe(df, d, cfg, fixed_pls_components=fixed_k)
            except Exception:
                continue
            m["representation"] = rname; m["fixed_pls_components"] = fixed_k; rows.append(m)
    pd.DataFrame(rows).to_csv(out / "ablation_fixed_pls.csv", index=False)

    # A3: fold-stratified participant-label permutation null.
    # Permuting labels within each frozen fold preserves the original case/control
    # count in every held-out fold, avoiding invalid folds for small cohorts.
    run_permutation_ablation(cfg)


def validate_freeze(cfg: dict[str, Any]) -> list[str]:
    errors=[]
    required = [
        output_dir(cfg, "manifests") / "participants.csv",
        output_dir(cfg, "manifests") / "folds.csv",
        output_dir(cfg, "manifests") / "task_support.csv",
        output_dir(cfg, "results") / "disease_benchmark.csv",
        output_dir(cfg, "results") / "disease_task.csv",
        output_dir(cfg, "results") / "disease_representation.csv",
        output_dir(cfg, "results") / "common_task.csv",
        output_dir(cfg, "results") / "adjustments.csv",
        output_dir(cfg, "results") / "equal_budget.csv",
        output_dir(cfg, "results") / "rank_summary.csv",
        output_dir(cfg, "results") / "pairwise_superiority.csv",
    ]
    for p in required:
        if not p.exists() or p.stat().st_size == 0:
            errors.append(f"Missing/empty: {p}")
    try:
        tm = load_task_mapping(cfg, require_frozen=True)
        if any(v == "UNMAPPED" for v in (tm.get("mapping") or {}).values()):
            errors.append("Task map still contains UNMAPPED values")
    except Exception as e:
        errors.append(str(e))
    folds_path = output_dir(cfg, "manifests") / "folds.csv"
    if folds_path.exists():
        f = pd.read_csv(folds_path, dtype={"participant_id": str})
        if f.groupby("participant_id").fold.nunique().max() != 1:
            errors.append("Fold leakage: participant appears in multiple folds")

    # Final-review requirements: paired pooled-vs-shared-task uncertainty and
    # a complete permutation null with the requested number of successful draws.
    common_path = output_dir(cfg, "results") / "common_task.csv"
    if common_path.exists():
        c = pd.read_csv(common_path)
        needed = {"paired_delta_auc", "paired_delta_ci_low", "paired_delta_ci_high", "n_paired"}
        if not needed.issubset(c.columns):
            errors.append("common_task.csv lacks paired delta AUC confidence intervals; run `speech-atlas final-fixes`")
        elif c[list(needed)].isna().any().any():
            errors.append("common_task.csv contains missing paired delta AUC results")

    perm_summary = output_dir(cfg, "manifests") / "permutation_summary.json"
    perm_path = output_dir(cfg, "results") / "ablation_permutation.csv"
    if not perm_summary.exists() or not perm_path.exists():
        errors.append("Missing final permutation summary/results; run `speech-atlas final-fixes`")
    else:
        ps = json.loads(perm_summary.read_text())
        requested = int(ps.get("requested_per_disease", 0))
        perm = pd.read_csv(perm_path)
        counts = perm.groupby("disease").size()
        if requested < 1 or counts.empty or (counts != requested).any():
            errors.append(f"Incomplete permutation null: requested {requested} successful draws per disease, got {counts.to_dict()}")
    return errors


def run_tier1(cfg: dict[str, Any]):
    # Enforce frozen task definitions before any task-dependent results are created.
    load_task_mapping(cfg, require_frozen=True)
    run_disease_benchmark(cfg)
    run_disease_task(cfg)
    run_disease_representation(cfg)
    run_common_task(cfg)
    run_adjustments(cfg)
    run_equal_budget(cfg)
    run_ranking(cfg)
    run_ablations(cfg)
    _save_manifest(cfg, {"stage": "tier1_complete"})


def run_tensor(cfg: dict[str, Any]) -> pd.DataFrame:
    """Optional disease × task-family × representation tensor for the supplement."""
    participants, rec = _load_prepared(cfg, require_task_map=True)
    support = task_support_table(cfg)
    _, reps = _participant_matrix(cfg, rec, participants, ["participant_id", "task_family"])
    rows=[]
    for rname, rdf in reps.items():
        for sr in support[support.eligible].itertuples(index=False):
            sub = rdf[rdf.task_family.eq(sr.task_family)]
            ncase = sub[sub.cohort.eq(sr.disease)].participant_id.nunique()
            nctrl = sub[sub.cohort.eq("Healthy Control")].participant_id.nunique()
            if ncase < int(cfg["analysis"]["task_min_cases"]) or nctrl < int(cfg["analysis"]["task_min_controls"]):
                continue
            try:
                pred, m = run_grouped_probe(sub, sr.disease, cfg, context="raw")
            except ValueError:
                continue
            m.update({"task_family":sr.task_family,"representation":rname})
            m = _attach_ci(pred,m,cfg,f"tensor|{sr.disease}|{sr.task_family}|{rname}")
            rows.append(m)
    res=pd.DataFrame(rows)
    res.to_csv(output_dir(cfg,"results")/"disease_task_representation_tensor.csv",index=False)
    return res


def run_matched_task_pairs(cfg: dict[str, Any]) -> pd.DataFrame:
    """Optional within-participant task-pair comparison for high-support pairs."""
    from itertools import combinations
    from .stats import paired_auc_delta_bootstrap
    participants, rec = _load_prepared(cfg, require_task_map=True)
    combined, _ = _participant_matrix(cfg, rec, participants, ["participant_id","task_family"])
    rows=[]
    min_case=int(cfg["analysis"]["task_min_cases"]); min_ctrl=int(cfg["analysis"]["task_min_controls"])
    B=int(cfg["project"]["bootstrap_replicates"]); base=int(cfg["project"]["seed"])
    for d in _diseases(participants):
        dsub=combined[combined.cohort.isin(["Healthy Control",d])]
        tasks=sorted(dsub.task_family.dropna().unique())
        for ta,tb in combinations(tasks,2):
            a=dsub[dsub.task_family.eq(ta)]; b=dsub[dsub.task_family.eq(tb)]
            shared=set(a.participant_id.astype(str)) & set(b.participant_id.astype(str))
            matched=dsub[dsub.participant_id.astype(str).isin(shared)]
            na=matched[matched.cohort.eq(d)].participant_id.nunique(); nh=matched[matched.cohort.eq("Healthy Control")].participant_id.nunique()
            if na<min_case or nh<min_ctrl: continue
            pa,ma=run_grouped_probe(a[a.participant_id.astype(str).isin(shared)],d,cfg)
            pb,mb=run_grouped_probe(b[b.participant_id.astype(str).isin(shared)],d,cfg)
            delta=paired_auc_delta_bootstrap(pa,pb,B,stable_seed(base,"matched",d,ta,tb))
            rows.append({"disease":d,"task_a":ta,"task_b":tb,"n_case":na,"n_control":nh,"auc_a":ma["auc"],"auc_b":mb["auc"],**delta})
    res=pd.DataFrame(rows)
    res.to_csv(output_dir(cfg,"results")/"matched_task_pairs.csv",index=False)
    return res


def run_leave_one_site_out(cfg: dict[str, Any]) -> pd.DataFrame:
    """Optional site-generalization analysis, run only for identifiable held-out sites."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from .model import safe_pls_components, compute_metrics
    participants, rec = _load_prepared(cfg, require_task_map=False)
    combined,_ = _participant_matrix(cfg,rec,participants,["participant_id"])
    rows=[]; pred_rows=[]
    for d in _diseases(participants):
        sub=combined[combined.cohort.isin(["Healthy Control",d])].copy(); sub["y"]=sub.cohort.eq(d).astype(int)
        for site in sorted(sub.site.dropna().astype(str).unique()):
            te=sub.site.astype(str).eq(site); tr=~te
            test=sub[te]; train=sub[tr]
            if test.y.sum()<5 or (1-test.y).sum()<5 or train.y.nunique()<2: continue
            Xtr=np.stack(train.vector.map(np.asarray)); Xte=np.stack(test.vector.map(np.asarray)); ytr=train.y.to_numpy(int)
            imp=SimpleImputer(strategy="median"); Xtr=imp.fit_transform(Xtr); Xte=imp.transform(Xte)
            sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
            k=safe_pls_components(Xtr,ytr,int(cfg["analysis"]["pls_max_components"])); pls=PLSRegression(n_components=k,scale=False); pls.fit(Xtr,ytr)
            ztr=pls.transform(Xtr); zte=pls.transform(Xte)
            clf=LogisticRegression(class_weight="balanced",solver="liblinear",max_iter=3000,random_state=int(cfg["project"]["seed"])); clf.fit(ztr,ytr)
            score=clf.predict_proba(zte)[:,1]
            m=compute_metrics(test.y.to_numpy(int),score); m.update({"disease":d,"held_out_site":site,"n_case":int(test.y.sum()),"n_control":int((1-test.y).sum())}); rows.append(m)
            pred_rows.extend({"participant_id":pid,"disease":d,"held_out_site":site,"y_true":int(y),"y_score":float(s)} for pid,y,s in zip(test.participant_id,test.y,score))
    res=pd.DataFrame(rows); res.to_csv(output_dir(cfg,"results")/"leave_one_site_out.csv",index=False)
    pd.DataFrame(pred_rows).to_csv(output_dir(cfg,"results")/"pred_leave_one_site_out.csv",index=False)
    return res


def run_leakage_stress(cfg: dict[str, Any]) -> pd.DataFrame:
    """Diagnostic only: deliberately incorrect recording-level CV to quantify identity leakage."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from .model import safe_pls_components
    participants, rec = _load_prepared(cfg, require_task_map=False)
    features=_load_recording_features(cfg)
    merged=None
    for name,f in features.items():
        t=f.rename(columns={"vector":f"vector__{name}"})
        merged=t if merged is None else merged.merge(t,on="recording_id",how="inner")
    vec_cols=[c for c in merged.columns if c.startswith("vector__")]
    merged["vector"]=merged[vec_cols].apply(lambda r:np.concatenate([np.asarray(v,float) for v in r]),axis=1)
    merged=rec[["recording_id","participant_id","cohort"]].merge(merged[["recording_id","vector"]],on="recording_id",how="inner")
    rows=[]; base=int(cfg["project"]["seed"])
    for d in _diseases(participants):
        sub=merged[merged.cohort.isin(["Healthy Control",d])].copy(); y=sub.cohort.eq(d).astype(int).to_numpy(); X=np.stack(sub.vector.map(np.asarray))
        skf=StratifiedKFold(n_splits=int(cfg["project"]["n_folds"]),shuffle=True,random_state=stable_seed(base,"leak",d)); score=np.zeros(len(sub))
        for tr,te in skf.split(X,y):
            imp=SimpleImputer(strategy="median"); xtr=imp.fit_transform(X[tr]); xte=imp.transform(X[te]); sc=StandardScaler(); xtr=sc.fit_transform(xtr); xte=sc.transform(xte)
            k=safe_pls_components(xtr,y[tr],int(cfg["analysis"]["pls_max_components"])); pls=PLSRegression(n_components=k,scale=False); pls.fit(xtr,y[tr]); ztr=pls.transform(xtr); zte=pls.transform(xte)
            clf=LogisticRegression(class_weight="balanced",solver="liblinear",max_iter=3000,random_state=base); clf.fit(ztr,y[tr]); score[te]=clf.predict_proba(zte)[:,1]
        rows.append({"disease":d,"recording_level_leakage_auc":float(roc_auc_score(y,score)),"n_recordings":len(sub),"diagnostic_only":True})
    res=pd.DataFrame(rows); res.to_csv(output_dir(cfg,"results")/"leakage_stress_recording_split.csv",index=False); return res

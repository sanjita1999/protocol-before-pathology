#!/usr/bin/env python3
"""
Comprehensive add-only measurement-validity experiment suite.

This script imports the known-working V5 protocol-balanced runner in the same
folder and adds:

1. a sequential stage analysis:
   raw same-probe -> age/sex matched -> shared 7-task menu ->
   complete shared menu -> final equal-evidence/all-representation support;
2. final protocol-feature ablations;
3. balance diagnostics, including age and session exposure;
4. age-caliper sensitivity (0.10, 0.25, 0.50 SD by default);
5. a fresh raw speech estimate under the same cached representation construction
   used in the staged analysis, alongside the published raw AUC reference.

No existing atlas source/config/results are modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# The script is intentionally colocated with the tested V5 implementation.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_protocol_balanced as pb  # noqa: E402


STAGE_ORDER = {
    "raw_same_probe": 0,
    "age_sex_matched": 1,
    "shared_7_task_menu": 2,
    "complete_shared_7_tasks": 3,
    "equal_taskwise_allrep": 4,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all protocol-conditioned measurement-validity experiments together."
    )
    p.add_argument("--repo-root", default=".")
    p.add_argument("--manifest", default="outputs/manifests/recordings.csv")
    p.add_argument("--b2ai-root", default=os.environ.get("B2AI_ROOT"))
    p.add_argument("--demographics", default=None)
    p.add_argument("--fold-map", default=None)
    p.add_argument("--out-dir", default="outputs/protocol_balanced_framework/all_experiments")
    p.add_argument("--diseases", nargs="+", default=["Parkinson's Disease", "Cognitive Impairment"])
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--min-group", type=int, default=20)
    p.add_argument("--feature-retention", type=float, default=0.70)
    p.add_argument("--age-caliper-sd", type=float, default=0.25)
    p.add_argument("--per-task-budgets", nargs="+", type=int, default=[5, 3, 2, 1])
    p.add_argument(
        "--caliper-grid",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50],
        help="Age-caliper SD values for sensitivity analysis.",
    )
    return p.parse_args()


def pair_list(case_ids: list[str], control_ids: list[str]) -> list[tuple[str, str]]:
    if len(case_ids) != len(control_ids):
        raise RuntimeError(
            f"Pair invariant failed: case={len(case_ids)} control={len(control_ids)}"
        )
    return list(zip(map(str, case_ids), map(str, control_ids)))


def flatten_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    return [x for p in pairs for x in p]


def filter_pairs_by_eligible(
    pairs: list[tuple[str, str]],
    eligible_ids: set[str],
) -> list[tuple[str, str]]:
    return [
        (c, h) for c, h in pairs
        if str(c) in eligible_ids and str(h) in eligible_ids
    ]


def complete_menu_pairs(
    rows: pd.DataFrame,
    pairs: list[tuple[str, str]],
    menu: list[str],
    budget: int = 1,
) -> list[tuple[str, str]]:
    ids = flatten_pairs(pairs)
    sub = rows[
        rows["participant_id"].isin(ids)
        & rows["task_family"].isin(menu)
    ]
    counts = (
        sub.groupby(["participant_id", "task_family"])
        .size()
        .unstack(fill_value=0)
    )
    for fam in menu:
        if fam not in counts:
            counts[fam] = 0

    def okay(pid: str) -> bool:
        x = counts.reindex([str(pid)], fill_value=0)
        return bool((x[menu] >= budget).all(axis=1).iloc[0])

    return [(c, h) for c, h in pairs if okay(c) and okay(h)]


def build_speech_matrix_available(
    rows: pd.DataFrame,
    participant_ids: list[str],
    lookups: dict[str, dict],
) -> tuple[np.ndarray, list[str]]:
    """
    Build a combined participant vector by averaging each representation over
    all available observations for that representation, then concatenating the
    eight participant-level representation vectors.

    Unlike the final balanced stage, this does not require each recording to
    exist in all eight representations. It is used only for the sequential
    sensitivity path before the final common-support control.
    """
    rows_by_pid = {
        str(pid): g for pid, g in rows.groupby("participant_id", sort=False)
    }
    X: list[np.ndarray] = []
    kept: list[str] = []

    for pid in map(str, participant_ids):
        g = rows_by_pid.get(pid)
        if g is None or g.empty:
            continue
        keys = list(zip(g["participant_id"], g["session_id"], g["task_norm"]))
        pieces = []
        valid = True
        for rep in pb.REP_SPECS:
            vals = [lookups[rep][k] for k in keys if k in lookups[rep]]
            if not vals:
                valid = False
                break
            pieces.append(np.nanmean(np.stack(vals), axis=0))
        if not valid:
            continue
        X.append(np.concatenate(pieces).astype(np.float32))
        kept.append(pid)

    total_dim = sum(int(x["dim"]) for x in pb.REP_SPECS.values())
    if not X:
        return np.empty((0, total_dim), dtype=np.float32), []
    return np.stack(X), kept


def protocol_dataframe(
    rows: pd.DataFrame,
    participant_ids: list[str],
    demographics: pd.DataFrame,
    all_families: list[str],
) -> pd.DataFrame:
    pids = pd.Index([str(x) for x in participant_ids], name="participant_id")

    completion = (
        rows.assign(_one=1)
        .pivot_table(
            index="participant_id",
            columns="task_family",
            values="_one",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(pids, fill_value=0)
    )
    counts = (
        rows.assign(_one=1)
        .pivot_table(
            index="participant_id",
            columns="task_family",
            values="_one",
            aggfunc="sum",
            fill_value=0,
        )
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

    summary = (
        rows.groupby("participant_id")
        .agg(
            total_observations=("task_family", "size"),
            total_sessions=("session_id", "nunique"),
            total_families=("task_family", "nunique"),
        )
        .reindex(pids, fill_value=0)
    )

    demo = demographics.set_index("participant_id").reindex(pids)
    age = demo["age"].astype(float).rename("age")
    sex_m = demo["sex"].eq("m").astype(float).rename("sex_m")

    return pd.concat([completion, counts, summary, age, sex_m], axis=1)


def protocol_components(pdf: pd.DataFrame) -> dict[str, list[str]]:
    completion = [c for c in pdf if c.startswith("complete__")]
    counts = [c for c in pdf if c.startswith("count__")]
    summary = ["total_observations", "total_sessions", "total_families"]

    return {
        "completion_only": completion,
        "counts_only": counts,
        "session_only": ["total_sessions"],
        "age_only": ["age"],
        "sex_only": ["sex_m"],
        "age_sex": ["age", "sex_m"],
        "task_structure": completion + counts,
        "assessment_intensity": counts + summary,
        "structural_protocol": completion + counts + summary,
        "session_age_sex": ["total_sessions", "age", "sex_m"],
        "full_protocol": completion + counts + summary + ["age", "sex_m"],
    }


def evaluate_matrix(
    X: np.ndarray,
    ids: list[str],
    disease: str,
    cohort_by_pid: pd.Series,
    fold_map: dict[str, int] | None,
    seed: int,
    bootstrap: int,
    folds: np.ndarray | None = None,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    y = pb.make_labels(ids, disease, cohort_by_pid)
    if folds is None:
        folds = pb.make_folds(ids, y, fold_map, seed)
    pred = pb.fixed_probe_oof(X, y, ids, folds)
    auc, lo, hi = pb.auc_ci(pred, bootstrap, seed)
    return {"auc": auc, "ci_low": lo, "ci_high": hi}, pred, folds


def chance_relation(lo: float, hi: float) -> str:
    if lo > 0.5:
        return "above_chance"
    if hi < 0.5:
        return "below_chance"
    return "includes_chance"


def smd(case: np.ndarray, control: np.ndarray) -> float:
    case = np.asarray(case, dtype=float)
    control = np.asarray(control, dtype=float)
    case = case[np.isfinite(case)]
    control = control[np.isfinite(control)]
    if len(case) < 2 or len(control) < 2:
        return float("nan")
    v = ((len(case) - 1) * np.var(case, ddof=1) + (len(control) - 1) * np.var(control, ddof=1))
    denom = len(case) + len(control) - 2
    pooled = math.sqrt(v / denom) if denom > 0 and v >= 0 else float("nan")
    diff = float(np.mean(case) - np.mean(control))
    if not np.isfinite(pooled) or pooled < 1e-12:
        return 0.0 if abs(diff) < 1e-12 else float("inf")
    return diff / pooled


def balance_diagnostics(
    rows: pd.DataFrame,
    ids: list[str],
    disease: str,
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    stage: str,
    seed_label: int,
) -> list[dict[str, Any]]:
    y = pb.make_labels(ids, disease, cohort_by_pid)
    idx = pd.Index(ids, name="participant_id")

    demo = demographics.set_index("participant_id").reindex(idx)
    rec = (
        rows.groupby("participant_id")
        .agg(
            total_observations=("task_family", "size"),
            total_sessions=("session_id", "nunique"),
            total_families=("task_family", "nunique"),
        )
        .reindex(idx, fill_value=0)
    )

    data = {
        "age": demo["age"].to_numpy(dtype=float),
        "sex_m": demo["sex"].eq("m").astype(float).to_numpy(),
        "total_observations": rec["total_observations"].to_numpy(dtype=float),
        "total_sessions": rec["total_sessions"].to_numpy(dtype=float),
        "total_families": rec["total_families"].to_numpy(dtype=float),
    }

    out = []
    for variable, values in data.items():
        case = values[y == 1]
        ctrl = values[y == 0]
        out.append({
            "disease": disease,
            "stage": stage,
            "seed": seed_label,
            "variable": variable,
            "case_mean": float(np.nanmean(case)),
            "control_mean": float(np.nanmean(ctrl)),
            "mean_difference": float(np.nanmean(case) - np.nanmean(ctrl)),
            "smd": smd(case, ctrl),
            "n_case": int((y == 1).sum()),
            "n_control": int((y == 0).sum()),
        })
    return out


def feature_diagnostics(
    pdf: pd.DataFrame,
    ids: list[str],
    disease: str,
    cohort_by_pid: pd.Series,
    seed_label: int,
) -> list[dict[str, Any]]:
    y = pb.make_labels(ids, disease, cohort_by_pid)
    out = []
    for col in pdf.columns:
        values = pd.to_numeric(pdf[col], errors="coerce").to_numpy(dtype=float)
        case = values[y == 1]
        ctrl = values[y == 0]
        out.append({
            "disease": disease,
            "seed": seed_label,
            "feature": col,
            "case_mean": float(np.nanmean(case)),
            "control_mean": float(np.nanmean(ctrl)),
            "smd": smd(case, ctrl),
            "n_unique": int(pd.Series(values).nunique(dropna=True)),
            "global_std": float(np.nanstd(values)),
        })
    return out


def evaluate_deterministic_stage(
    *,
    disease: str,
    stage: str,
    rows: pd.DataFrame,
    candidate_ids: list[str],
    pairs: list[tuple[str, str]] | None,
    lookups: dict[str, dict],
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    fold_map: dict[str, int] | None,
    bootstrap: int,
    seed: int,
    all_families: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[str, str]] | None]:
    # Build speech first so speech and protocol are always evaluated on exactly
    # the same participants.
    X0, speech_ids = build_speech_matrix_available(rows, candidate_ids, lookups)
    eligible = set(speech_ids)

    if pairs is not None:
        pairs = filter_pairs_by_eligible(pairs, eligible)
        ids = flatten_pairs(pairs)
        Xs, speech_ids = build_speech_matrix_available(rows, ids, lookups)
        if speech_ids != ids:
            raise RuntimeError(f"{disease}/{stage}: paired speech support invariant failed.")
    else:
        ids = speech_ids
        Xs = X0

    y = pb.make_labels(ids, disease, cohort_by_pid)
    if int((y == 1).sum()) < 2 or int((y == 0).sum()) < 2:
        raise RuntimeError(f"{disease}/{stage}: insufficient participants after feature support.")

    folds = pb.make_folds(ids, y, fold_map, seed)

    speech_pred = pb.fixed_probe_oof(Xs, y, ids, folds)
    s_auc, s_lo, s_hi = pb.auc_ci(speech_pred, bootstrap, seed + 100)

    pdf = protocol_dataframe(rows[rows["participant_id"].isin(ids)], ids, demographics, all_families)
    protocol_pred = pb.fixed_probe_oof(pdf.to_numpy(dtype=float), y, ids, folds)
    p_auc, p_lo, p_hi = pb.auc_ci(protocol_pred, bootstrap, seed + 200)

    row = {
        "disease": disease,
        "stage": stage,
        "stage_order": STAGE_ORDER[stage],
        "n_case": int((y == 1).sum()),
        "n_control": int((y == 0).sum()),
        "n_participants": len(ids),
        "speech_auc": s_auc,
        "speech_ci_low": s_lo,
        "speech_ci_high": s_hi,
        "protocol_auc": p_auc,
        "protocol_ci_low": p_lo,
        "protocol_ci_high": p_hi,
        "protocol_chance_relation": chance_relation(p_lo, p_hi),
        "n_rows": int(rows[rows["participant_id"].isin(ids)].shape[0]),
        "subsample_seeds": 1,
    }
    diag = balance_diagnostics(
        rows[rows["participant_id"].isin(ids)],
        ids,
        disease,
        demographics,
        cohort_by_pid,
        stage,
        -1,
    )
    return row, diag, pairs


def final_balanced_core(
    *,
    disease: str,
    manifest: pd.DataFrame,
    matched_pairs: list[tuple[str, str]],
    lookups: dict[str, dict],
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    fold_map: dict[str, int] | None,
    bootstrap: int,
    seeds: int,
    seed: int,
    min_group: int,
    feature_retention: float,
    per_task_budgets: tuple[int, ...],
    collect_ablation: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[str, str]],
]:
    menu = list(pb.COMMON_TASKS)

    # First require every matched pair member to have all seven shared task
    # families at least once in the frozen manifest.
    complete_pairs = complete_menu_pairs(manifest, matched_pairs, menu, budget=1)
    if len(complete_pairs) < min_group:
        raise RuntimeError(
            f"{disease}: only {len(complete_pairs)} complete shared-menu matched pairs "
            f"(< min_group={min_group})."
        )

    complete_ids = flatten_pairs(complete_pairs)
    strict_source = manifest[
        manifest["participant_id"].isin(complete_ids)
        & manifest["task_family"].isin(menu)
        & manifest["_has_all_reps"]
    ].copy()

    case0 = [c for c, _ in complete_pairs]
    ctrl0 = [h for _, h in complete_pairs]
    budget, case_ids, ctrl_ids = pb.choose_paired_task_budget(
        strict_source,
        case0,
        ctrl0,
        menu,
        retention=feature_retention,
        min_group=min_group,
        candidates=per_task_budgets,
    )
    final_pairs = pair_list(case_ids, ctrl_ids)
    ids = flatten_pairs(final_pairs)
    y = pb.make_labels(ids, disease, cohort_by_pid)
    folds = pb.make_folds(ids, y, fold_map, seed)

    speech_preds = []
    protocol_preds = []
    ablation_preds: dict[str, list[pd.DataFrame]] = {}
    ablation_feature_counts: dict[str, int] = {}
    diagnostics: list[dict[str, Any]] = []
    feature_diag_rows: list[dict[str, Any]] = []

    for s in range(seeds):
        sampled = pb.sample_taskwise(
            strict_source,
            ids,
            menu,
            budget,
            seed=seed + s,
        )

        Xs, pids_s = pb.build_speech_matrix(sampled, ids, lookups)
        if pids_s != ids:
            raise RuntimeError(
                f"{disease}: final strict speech support invariant failed "
                f"({len(pids_s)} vs {len(ids)})."
            )
        sp = pb.fixed_probe_oof(Xs, y, ids, folds)
        sp["subsample_seed"] = s
        speech_preds.append(sp)

        pdf = protocol_dataframe(sampled, ids, demographics, menu)
        pp = pb.fixed_probe_oof(pdf.to_numpy(dtype=float), y, ids, folds)
        pp["subsample_seed"] = s
        protocol_preds.append(pp)

        diagnostics.extend(
            balance_diagnostics(
                sampled, ids, disease, demographics, cohort_by_pid,
                "equal_taskwise_allrep", s
            )
        )
        feature_diag_rows.extend(
            feature_diagnostics(pdf, ids, disease, cohort_by_pid, s)
        )

        if collect_ablation:
            comps = protocol_components(pdf)
            for name, cols in comps.items():
                X = pdf[cols].to_numpy(dtype=float)
                pred = pb.fixed_probe_oof(X, y, ids, folds)
                pred["subsample_seed"] = s
                ablation_preds.setdefault(name, []).append(pred)
                ablation_feature_counts[name] = len(cols)

    avg_s = pb.average_seed_predictions(speech_preds)
    avg_p = pb.average_seed_predictions(protocol_preds)
    s_auc, s_lo, s_hi = pb.auc_ci(avg_s, bootstrap, seed + 2000)
    p_auc, p_lo, p_hi = pb.auc_ci(avg_p, bootstrap, seed + 3000)

    stage_row = {
        "disease": disease,
        "stage": "equal_taskwise_allrep",
        "stage_order": STAGE_ORDER["equal_taskwise_allrep"],
        "n_case": len(case_ids),
        "n_control": len(ctrl_ids),
        "n_participants": len(ids),
        "speech_auc": s_auc,
        "speech_ci_low": s_lo,
        "speech_ci_high": s_hi,
        "protocol_auc": p_auc,
        "protocol_ci_low": p_lo,
        "protocol_ci_high": p_hi,
        "protocol_chance_relation": chance_relation(p_lo, p_hi),
        "n_rows": len(ids) * len(menu) * budget,
        "n_task_families": len(menu),
        "per_task_budget": budget,
        "observations_per_participant": len(menu) * budget,
        "subsample_seeds": seeds,
    }

    ablation_rows: list[dict[str, Any]] = []
    if collect_ablation:
        for name, preds in ablation_preds.items():
            avg = pb.average_seed_predictions(preds)
            auc, lo, hi = pb.auc_ci(avg, bootstrap, seed + 4000)
            ablation_rows.append({
                "disease": disease,
                "protocol_model": name,
                "auc": auc,
                "ci_low": lo,
                "ci_high": hi,
                "chance_relation": chance_relation(lo, hi),
                "n_features": ablation_feature_counts[name],
                "n_case": len(case_ids),
                "n_control": len(ctrl_ids),
                "n_subsample_seeds": seeds,
            })

    return stage_row, diagnostics, feature_diag_rows, ablation_rows, final_pairs


def run_caliper_sensitivity(
    *,
    disease: str,
    caliper_sd: float,
    manifest: pd.DataFrame,
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    lookups: dict[str, dict],
    fold_map: dict[str, int] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_ids, ctrl_ids, caliper_years = pb.match_demographics(
        disease, cohort_by_pid, demographics, caliper_sd
    )
    pairs = pair_list(case_ids, ctrl_ids)
    if len(pairs) < args.min_group:
        raise RuntimeError(
            f"demographic matching retained {len(pairs)} pairs (< {args.min_group})"
        )

    row, _, _, _, final_pairs = final_balanced_core(
        disease=disease,
        manifest=manifest,
        matched_pairs=pairs,
        lookups=lookups,
        demographics=demographics,
        cohort_by_pid=cohort_by_pid,
        fold_map=fold_map,
        bootstrap=args.bootstrap,
        seeds=args.seeds,
        seed=args.seed,
        min_group=args.min_group,
        feature_retention=args.feature_retention,
        per_task_budgets=tuple(args.per_task_budgets),
        collect_ablation=False,
    )
    return {
        "disease": disease,
        "age_caliper_sd": caliper_sd,
        "age_caliper_years": caliper_years,
        "initial_matched_pairs": len(pairs),
        "final_balanced_pairs": len(final_pairs),
        "per_task_budget": row.get("per_task_budget"),
        "speech_auc": row["speech_auc"],
        "speech_ci_low": row["speech_ci_low"],
        "speech_ci_high": row["speech_ci_high"],
        "protocol_auc": row["protocol_auc"],
        "protocol_ci_low": row["protocol_ci_low"],
        "protocol_ci_high": row["protocol_ci_high"],
        "protocol_chance_relation": row["protocol_chance_relation"],
    }


def summarize_feature_diagnostics(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    z = raw.copy()
    z["abs_smd"] = z["smd"].abs()
    return (
        z.groupby(["disease", "feature"], as_index=False)
        .agg(
            case_mean=("case_mean", "mean"),
            control_mean=("control_mean", "mean"),
            mean_smd=("smd", "mean"),
            max_abs_smd=("abs_smd", "max"),
            mean_global_std=("global_std", "mean"),
            mean_n_unique=("n_unique", "mean"),
        )
        .sort_values(["disease", "max_abs_smd"], ascending=[True, False])
    )


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root).expanduser().resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not args.b2ai_root:
        raise RuntimeError("Set B2AI_ROOT or pass --b2ai-root.")

    b2ai_root = Path(args.b2ai_root).expanduser().resolve()
    task_map = repo / "config" / "task_families.yaml"
    atlas_config = repo / "config" / "atlas.yaml"

    manifest = pb.load_manifest(manifest_path, task_map)
    cohort_by_pid = (
        manifest[["participant_id", "cohort"]]
        .drop_duplicates("participant_id")
        .set_index("participant_id")["cohort"]
    )

    demographics = pb.discover_demographics(
        repo,
        b2ai_root,
        manifest,
        Path(args.demographics).expanduser().resolve() if args.demographics else None,
    )
    fold_map = pb.discover_fold_map(
        repo,
        Path(args.fold_map).expanduser().resolve() if args.fold_map else None,
    )

    configured_feature_files = pb.discover_feature_files_from_atlas(
        atlas_config, b2ai_root
    )
    cache_files = pb.discover_embedding_caches(repo)
    lookups = pb.build_all_cached_feature_lookups(cache_files)
    manifest = pb.attach_feature_presence(manifest, lookups)
    published_raw = pb.discover_published_raw(repo)

    stage_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    feature_diag_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    caliper_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for disease in args.diseases:
        print("\n" + "=" * 88)
        print(f"ALL EXPERIMENTS: {disease}")
        print("=" * 88)

        try:
            disease_rows = manifest[
                manifest["cohort"].isin([disease, "Healthy Control"])
            ].copy()
            demo_ids = set(demographics["participant_id"].astype(str))
            raw_ids = sorted(
                pid for pid in disease_rows["participant_id"].unique().astype(str)
                if pid in demo_ids
            )
            raw_families = sorted(
                disease_rows["task_family"].dropna().unique().tolist()
            )

            # S0: fresh raw speech + protocol on exactly the same speech-eligible
            # participants, using existing cached representation vectors.
            raw_row, raw_diag, _ = evaluate_deterministic_stage(
                disease=disease,
                stage="raw_same_probe",
                rows=disease_rows,
                candidate_ids=raw_ids,
                pairs=None,
                lookups=lookups,
                demographics=demographics,
                cohort_by_pid=cohort_by_pid,
                fold_map=fold_map,
                bootstrap=args.bootstrap,
                seed=args.seed,
                all_families=raw_families,
            )
            raw_row["published_raw_speech_auc_reference"] = published_raw.get(
                disease, np.nan
            )
            raw_row["fresh_minus_published_raw"] = (
                raw_row["speech_auc"] - published_raw[disease]
                if disease in published_raw else np.nan
            )
            stage_rows.append(raw_row)
            balance_rows.extend(raw_diag)

            # Frozen demographic matching.
            case_ids, ctrl_ids, caliper_years = pb.match_demographics(
                disease, cohort_by_pid, demographics, args.age_caliper_sd
            )
            matched_pairs = pair_list(case_ids, ctrl_ids)
            if len(matched_pairs) < args.min_group:
                raise RuntimeError(
                    f"{disease}: only {len(matched_pairs)} demographic matched pairs."
                )

            # S1: age/sex matched, all observed tasks/counts.
            matched_ids = flatten_pairs(matched_pairs)
            matched_rows = disease_rows[
                disease_rows["participant_id"].isin(matched_ids)
            ].copy()
            s1, d1, matched_supported_pairs = evaluate_deterministic_stage(
                disease=disease,
                stage="age_sex_matched",
                rows=matched_rows,
                candidate_ids=matched_ids,
                pairs=matched_pairs,
                lookups=lookups,
                demographics=demographics,
                cohort_by_pid=cohort_by_pid,
                fold_map=fold_map,
                bootstrap=args.bootstrap,
                seed=args.seed,
                all_families=raw_families,
            )
            s1["age_caliper_years"] = caliper_years
            stage_rows.append(s1)
            balance_rows.extend(d1)

            # S2: restrict observations to the fixed seven shared task families;
            # participants need not complete every family yet.
            shared_rows = matched_rows[
                matched_rows["task_family"].isin(pb.COMMON_TASKS)
            ].copy()
            s2, d2, shared_supported_pairs = evaluate_deterministic_stage(
                disease=disease,
                stage="shared_7_task_menu",
                rows=shared_rows,
                candidate_ids=flatten_pairs(matched_supported_pairs),
                pairs=matched_supported_pairs,
                lookups=lookups,
                demographics=demographics,
                cohort_by_pid=cohort_by_pid,
                fold_map=fold_map,
                bootstrap=args.bootstrap,
                seed=args.seed,
                all_families=list(pb.COMMON_TASKS),
            )
            stage_rows.append(s2)
            balance_rows.extend(d2)

            # S3: require both members of each preserved matched pair to have
            # at least one recording in every shared task family.
            complete_pairs = complete_menu_pairs(
                shared_rows, shared_supported_pairs, list(pb.COMMON_TASKS), 1
            )
            if len(complete_pairs) < args.min_group:
                raise RuntimeError(
                    f"{disease}: complete shared-menu support leaves only "
                    f"{len(complete_pairs)} pairs."
                )
            complete_ids = flatten_pairs(complete_pairs)
            complete_rows = shared_rows[
                shared_rows["participant_id"].isin(complete_ids)
            ].copy()
            s3, d3, complete_supported_pairs = evaluate_deterministic_stage(
                disease=disease,
                stage="complete_shared_7_tasks",
                rows=complete_rows,
                candidate_ids=complete_ids,
                pairs=complete_pairs,
                lookups=lookups,
                demographics=demographics,
                cohort_by_pid=cohort_by_pid,
                fold_map=fold_map,
                bootstrap=args.bootstrap,
                seed=args.seed,
                all_families=list(pb.COMMON_TASKS),
            )
            stage_rows.append(s3)
            balance_rows.extend(d3)

            # S4: final strict common-support/equal-taskwise evidence. Also run
            # all protocol ablations on these exact participants/observations.
            final_row, final_diag, fdiag, abl, final_pairs = final_balanced_core(
                disease=disease,
                manifest=manifest,
                matched_pairs=matched_pairs,
                lookups=lookups,
                demographics=demographics,
                cohort_by_pid=cohort_by_pid,
                fold_map=fold_map,
                bootstrap=args.bootstrap,
                seeds=args.seeds,
                seed=args.seed,
                min_group=args.min_group,
                feature_retention=args.feature_retention,
                per_task_budgets=tuple(args.per_task_budgets),
                collect_ablation=True,
            )
            final_row["published_raw_speech_auc_reference"] = published_raw.get(
                disease, np.nan
            )
            final_row["speech_delta_vs_published_raw_reference"] = (
                final_row["speech_auc"] - published_raw[disease]
                if disease in published_raw else np.nan
            )
            stage_rows.append(final_row)
            balance_rows.extend(final_diag)
            feature_diag_rows.extend(fdiag)
            ablation_rows.extend(abl)

            # Caliper sensitivity. The primary 0.25-SD point is still rerun
            # through the same function so the table is self-contained.
            for caliper in args.caliper_grid:
                try:
                    c = run_caliper_sensitivity(
                        disease=disease,
                        caliper_sd=float(caliper),
                        manifest=manifest,
                        demographics=demographics,
                        cohort_by_pid=cohort_by_pid,
                        lookups=lookups,
                        fold_map=fold_map,
                        args=args,
                    )
                    caliper_rows.append(c)
                except Exception as exc:
                    caliper_rows.append({
                        "disease": disease,
                        "age_caliper_sd": float(caliper),
                        "error": repr(exc),
                    })

            print(
                f"[done] {disease}: raw speech={raw_row['speech_auc']:.3f}, "
                f"final speech={final_row['speech_auc']:.3f}, "
                f"final protocol={final_row['protocol_auc']:.3f}, "
                f"final pairs={len(final_pairs)}"
            )

        except Exception as exc:
            print(f"[FAILED] {disease}: {exc}", file=sys.stderr, flush=True)
            failures.append({"disease": disease, "error": repr(exc)})

    if not stage_rows:
        (out_dir / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )
        raise RuntimeError("No disease completed.")

    stages = pd.DataFrame(stage_rows).sort_values(
        ["disease", "stage_order"]
    ).reset_index(drop=True)
    stages["speech_delta_vs_previous_stage"] = (
        stages.groupby("disease")["speech_auc"].diff()
    )
    stages["protocol_delta_vs_previous_stage"] = (
        stages.groupby("disease")["protocol_auc"].diff()
    )

    balance = pd.DataFrame(balance_rows)
    fdiag_raw = pd.DataFrame(feature_diag_rows)
    fdiag_summary = summarize_feature_diagnostics(fdiag_raw)
    ablations = pd.DataFrame(ablation_rows)
    calipers = pd.DataFrame(caliper_rows)

    stages.to_csv(out_dir / "stagewise_validity.csv", index=False)
    balance.to_csv(out_dir / "balance_diagnostics.csv", index=False)
    fdiag_raw.to_csv(out_dir / "final_protocol_feature_diagnostics_by_seed.csv", index=False)
    fdiag_summary.to_csv(out_dir / "final_protocol_feature_diagnostics.csv", index=False)
    ablations.to_csv(out_dir / "protocol_ablation.csv", index=False)
    calipers.to_csv(out_dir / "age_caliper_sensitivity.csv", index=False)

    metadata = {
        "analysis": "all_protocol_conditioned_measurement_validity_experiments",
        "extension_version": "v6-all-experiments",
        "repo_root": str(repo),
        "manifest": str(manifest_path),
        "atlas_config": str(atlas_config),
        "task_family_mapping": str(task_map),
        "b2ai_root": str(b2ai_root),
        "diseases": args.diseases,
        "stages": [
            "raw_same_probe",
            "age_sex_matched",
            "shared_7_task_menu",
            "complete_shared_7_tasks",
            "equal_taskwise_allrep",
        ],
        "shared_task_families": list(pb.COMMON_TASKS),
        "primary_age_caliper_sd": args.age_caliper_sd,
        "caliper_grid_sd": args.caliper_grid,
        "subsample_seeds": args.seeds,
        "bootstrap_replicates": args.bootstrap,
        "feature_sources": {
            "configured_raw_feature_files": {
                k: str(v) for k, v in configured_feature_files.items()
            },
            "recording_embedding_caches": {
                k: str(v) for k, v in cache_files.items()
            },
            "loader": "existing_atlas_recording_embedding_cache",
        },
        "interpretation_guardrails": [
            "Stage-to-stage AUC changes are descriptive sensitivity estimates, not causal confounding fractions.",
            "A protocol CI including 0.5 is reported as includes_chance, not proof of equivalence or complete neutralization.",
            "Balanced speech above chance supports observed-protocol-standardized detectability, not pathology-specific causality.",
            "External or unmeasured protocol variables may remain.",
        ],
        "failures": failures,
    }
    (out_dir / "all_experiments_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    if failures:
        (out_dir / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )

    print("\n" + "=" * 88)
    print("STAGEWISE SUMMARY")
    print("=" * 88)
    cols = [
        "disease", "stage", "n_case", "n_control",
        "speech_auc", "speech_ci_low", "speech_ci_high",
        "protocol_auc", "protocol_ci_low", "protocol_ci_high",
        "protocol_chance_relation",
    ]
    print(stages[cols].to_string(index=False))

    if not ablations.empty:
        print("\n" + "=" * 88)
        print("FINAL PROTOCOL ABLATION")
        print("=" * 88)
        print(
            ablations[
                ["disease", "protocol_model", "auc", "ci_low", "ci_high", "chance_relation"]
            ].to_string(index=False)
        )

    print("\nSaved:")
    for name in [
        "stagewise_validity.csv",
        "protocol_ablation.csv",
        "age_caliper_sensitivity.csv",
        "balance_diagnostics.csv",
        "final_protocol_feature_diagnostics.csv",
        "final_protocol_feature_diagnostics_by_seed.csv",
        "all_experiments_metadata.json",
    ]:
        print(f"  {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

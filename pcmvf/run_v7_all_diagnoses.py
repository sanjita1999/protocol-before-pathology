#!/usr/bin/env python3
"""
V7 final validation suite for the clinical-speech measurement-validity paper.

Add-only: this script never edits the existing atlas source/config/results.

V7 adds to V6:
- all nine frozen disease cohorts by default;
- pair-aware CV for every matched/balanced analysis;
- same-final-cohort decomposition:
    original evidence -> shared-menu observed counts -> equal taskwise evidence;
- final protocol ablations under pair-aware CV;
- single-session balancing sensitivity to remove residual session exposure;
- age-caliper sensitivity under pair-aware CV;
- explicit primary/exploratory support tiers instead of silently dropping
  small cohorts;
- publication-ready summary tables and support/failure logs.
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_protocol_balanced as pb  # noqa: E402
import run_all_experiments as v6  # noqa: E402


DEFAULT_DISEASES = [
    "Airway Stenosis",
    "Parkinson's Disease",
    "Laryngeal Dystonia",
    "Unilateral Vocal Fold Paralysis",
    "Cognitive Impairment",
    "Muscle Tension Dysphonia",
    "Benign Lesions",
    "Unexplained Chronic Cough",
    "Precancerous Lesions",
]

STAGE_ORDER = {
    "raw_same_probe_original_cv": 0,
    "age_sex_matched_paircv": 1,
    "shared_7_task_menu_paircv": 2,
    "complete_shared_7_tasks_paircv": 3,
    "final_cohort_original_evidence_paircv": 4,
    "final_cohort_shared_menu_observed_counts_paircv": 5,
    "equal_taskwise_allrep_paircv": 6,
    "single_session_equal_taskwise_allrep_paircv": 7,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run V7 final validation across all frozen diagnosis cohorts."
    )
    p.add_argument("--repo-root", default=".")
    p.add_argument("--manifest", default="outputs/manifests/recordings.csv")
    p.add_argument("--b2ai-root", default=os.environ.get("B2AI_ROOT"))
    p.add_argument("--demographics", default=None)
    p.add_argument("--fold-map", default=None)
    p.add_argument(
        "--out-dir",
        default="outputs/protocol_balanced_framework/v7_all_diagnoses",
    )
    p.add_argument("--diseases", nargs="+", default=DEFAULT_DISEASES)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--primary-min-pairs",
        type=int,
        default=20,
        help="At or above this many matched pairs, treat the analysis as primary-support.",
    )
    p.add_argument(
        "--exploratory-min-pairs",
        type=int,
        default=8,
        help="Hard lower bound for exploratory balanced analyses.",
    )
    p.add_argument("--feature-retention", type=float, default=0.70)
    p.add_argument("--age-caliper-sd", type=float, default=0.25)
    p.add_argument("--per-task-budgets", nargs="+", type=int, default=[5, 3, 2, 1])
    p.add_argument(
        "--caliper-grid",
        nargs="+",
        type=float,
        default=[0.10, 0.25, 0.50],
    )
    return p.parse_args()


def support_tier(n_pairs: int, args: argparse.Namespace) -> str:
    if n_pairs >= args.primary_min_pairs:
        return "primary_supported"
    if n_pairs >= args.exploratory_min_pairs:
        return "exploratory_small_n"
    return "insufficient_support"


def pair_aware_folds(
    pairs: list[tuple[str, str]],
    ids: list[str],
    seed: int,
    max_splits: int = 5,
) -> np.ndarray:
    """
    Assign each matched case-control pair to the same test fold.

    Because every pair contains one case and one control, every resulting fold
    remains class-balanced at the pair level. This avoids breaking the
    demographic matching structure across train/test folds.
    """
    pairs = [(str(a), str(b)) for a, b in pairs]
    if len(pairs) < 2:
        raise RuntimeError("At least two matched pairs are required for pair-aware CV.")

    n_splits = min(max_splits, len(pairs))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    pair_fold = np.empty(len(pairs), dtype=int)
    for rank, pair_idx in enumerate(order):
        pair_fold[pair_idx] = rank % n_splits

    mapping: dict[str, int] = {}
    for (case_id, control_id), fold in zip(pairs, pair_fold):
        if case_id in mapping or control_id in mapping:
            raise RuntimeError("Participant appears in more than one matched pair.")
        mapping[case_id] = int(fold)
        mapping[control_id] = int(fold)

    missing = [str(pid) for pid in ids if str(pid) not in mapping]
    if missing:
        raise RuntimeError(
            f"Pair-aware fold map is missing {len(missing)} participants; "
            f"examples={missing[:5]}"
        )

    folds = np.array([mapping[str(pid)] for pid in ids], dtype=int)
    for fold in np.unique(folds):
        members = np.asarray(ids)[folds == fold]
        # pair construction itself guarantees two members per test pair.
        if len(members) < 2:
            raise RuntimeError(f"Pair-aware fold {fold} is unexpectedly too small.")
    return folds


def choose_budget_with_support(
    rows: pd.DataFrame,
    pairs: list[tuple[str, str]],
    menu: list[str],
    args: argparse.Namespace,
) -> tuple[int, list[tuple[str, str]], str]:
    case_ids = [c for c, _ in pairs]
    control_ids = [h for _, h in pairs]

    try:
        budget, c, h = pb.choose_paired_task_budget(
            rows,
            case_ids,
            control_ids,
            menu,
            retention=args.feature_retention,
            min_group=args.primary_min_pairs,
            candidates=tuple(args.per_task_budgets),
        )
        kept = v6.pair_list(c, h)
        return budget, kept, "primary_supported"
    except Exception as primary_exc:
        try:
            budget, c, h = pb.choose_paired_task_budget(
                rows,
                case_ids,
                control_ids,
                menu,
                retention=args.feature_retention,
                min_group=args.exploratory_min_pairs,
                candidates=tuple(args.per_task_budgets),
            )
            kept = v6.pair_list(c, h)
            return budget, kept, "exploratory_small_n"
        except Exception as exploratory_exc:
            raise RuntimeError(
                "No equal-taskwise paired design met even the exploratory support "
                f"threshold ({args.exploratory_min_pairs} pairs). "
                f"Primary failure={primary_exc!r}; exploratory failure={exploratory_exc!r}"
            )


def evaluate_pair_stage(
    *,
    disease: str,
    stage: str,
    rows: pd.DataFrame,
    pairs: list[tuple[str, str]],
    lookups: dict[str, dict],
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    bootstrap: int,
    seed: int,
    all_families: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[str, str]]]:
    ids0 = v6.flatten_pairs(pairs)
    X0, speech_ids = v6.build_speech_matrix_available(rows, ids0, lookups)
    eligible = set(speech_ids)
    kept_pairs = v6.filter_pairs_by_eligible(pairs, eligible)
    ids = v6.flatten_pairs(kept_pairs)

    if len(kept_pairs) < 2:
        raise RuntimeError(f"{disease}/{stage}: fewer than two feature-supported pairs.")

    X, speech_ids2 = v6.build_speech_matrix_available(rows, ids, lookups)
    if speech_ids2 != ids:
        raise RuntimeError(f"{disease}/{stage}: pair-preserving speech support failed.")

    y = pb.make_labels(ids, disease, cohort_by_pid)
    folds = pair_aware_folds(kept_pairs, ids, seed)

    sp = pb.fixed_probe_oof(X, y, ids, folds)
    s_auc, s_lo, s_hi = pb.auc_ci(sp, bootstrap, seed + 101)

    pdf = v6.protocol_dataframe(
        rows[rows["participant_id"].isin(ids)],
        ids,
        demographics,
        all_families,
    )
    pp = pb.fixed_probe_oof(pdf.to_numpy(dtype=float), y, ids, folds)
    p_auc, p_lo, p_hi = pb.auc_ci(pp, bootstrap, seed + 202)

    row = {
        "disease": disease,
        "stage": stage,
        "stage_order": STAGE_ORDER[stage],
        "cv_design": "pair_aware",
        "n_case": len(kept_pairs),
        "n_control": len(kept_pairs),
        "n_pairs": len(kept_pairs),
        "speech_auc": s_auc,
        "speech_ci_low": s_lo,
        "speech_ci_high": s_hi,
        "protocol_auc": p_auc,
        "protocol_ci_low": p_lo,
        "protocol_ci_high": p_hi,
        "protocol_chance_relation": v6.chance_relation(p_lo, p_hi),
        "n_rows": int(rows[rows["participant_id"].isin(ids)].shape[0]),
        "subsample_seeds": 1,
    }

    diag = v6.balance_diagnostics(
        rows[rows["participant_id"].isin(ids)],
        ids,
        disease,
        demographics,
        cohort_by_pid,
        stage,
        -1,
    )
    return row, diag, kept_pairs


def evaluate_raw_stage(
    *,
    disease: str,
    disease_rows: pd.DataFrame,
    candidate_ids: list[str],
    lookups: dict[str, dict],
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    fold_map: dict[str, int] | None,
    bootstrap: int,
    seed: int,
    all_families: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    X, ids = v6.build_speech_matrix_available(disease_rows, candidate_ids, lookups)
    y = pb.make_labels(ids, disease, cohort_by_pid)
    if int((y == 1).sum()) < 2 or int((y == 0).sum()) < 2:
        raise RuntimeError(f"{disease}: insufficient raw feature-supported participants.")

    folds = pb.make_folds(ids, y, fold_map, seed)
    sp = pb.fixed_probe_oof(X, y, ids, folds)
    s_auc, s_lo, s_hi = pb.auc_ci(sp, bootstrap, seed + 303)

    pdf = v6.protocol_dataframe(
        disease_rows[disease_rows["participant_id"].isin(ids)],
        ids,
        demographics,
        all_families,
    )
    pp = pb.fixed_probe_oof(pdf.to_numpy(dtype=float), y, ids, folds)
    p_auc, p_lo, p_hi = pb.auc_ci(pp, bootstrap, seed + 404)

    row = {
        "disease": disease,
        "stage": "raw_same_probe_original_cv",
        "stage_order": STAGE_ORDER["raw_same_probe_original_cv"],
        "cv_design": "frozen_original_or_stratified",
        "n_case": int((y == 1).sum()),
        "n_control": int((y == 0).sum()),
        "n_pairs": np.nan,
        "speech_auc": s_auc,
        "speech_ci_low": s_lo,
        "speech_ci_high": s_hi,
        "protocol_auc": p_auc,
        "protocol_ci_low": p_lo,
        "protocol_ci_high": p_hi,
        "protocol_chance_relation": v6.chance_relation(p_lo, p_hi),
        "n_rows": int(disease_rows[disease_rows["participant_id"].isin(ids)].shape[0]),
        "subsample_seeds": 1,
    }
    diag = v6.balance_diagnostics(
        disease_rows[disease_rows["participant_id"].isin(ids)],
        ids,
        disease,
        demographics,
        cohort_by_pid,
        "raw_same_probe_original_cv",
        -1,
    )
    return row, diag


def run_final_equal_evidence(
    *,
    disease: str,
    manifest: pd.DataFrame,
    matched_pairs: list[tuple[str, str]],
    lookups: dict[str, dict],
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    args: argparse.Namespace,
    collect_ablation: bool = True,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[str, str]],
    pd.DataFrame,
]:
    menu = list(pb.COMMON_TASKS)
    complete_pairs = v6.complete_menu_pairs(manifest, matched_pairs, menu, budget=1)
    if len(complete_pairs) < args.exploratory_min_pairs:
        raise RuntimeError(
            f"Complete shared-menu support leaves {len(complete_pairs)} pairs, "
            f"below exploratory minimum {args.exploratory_min_pairs}."
        )

    complete_ids = v6.flatten_pairs(complete_pairs)
    strict_source = manifest[
        manifest["participant_id"].isin(complete_ids)
        & manifest["task_family"].isin(menu)
        & manifest["_has_all_reps"]
    ].copy()

    budget, final_pairs, tier = choose_budget_with_support(
        strict_source, complete_pairs, menu, args
    )
    ids = v6.flatten_pairs(final_pairs)
    y = pb.make_labels(ids, disease, cohort_by_pid)
    folds = pair_aware_folds(final_pairs, ids, args.seed)

    speech_preds: list[pd.DataFrame] = []
    protocol_preds: list[pd.DataFrame] = []
    ablation_preds: dict[str, list[pd.DataFrame]] = {}
    ablation_n_features: dict[str, int] = {}
    diagnostics: list[dict[str, Any]] = []
    feature_diag: list[dict[str, Any]] = []
    selected_rows: list[pd.DataFrame] = []

    for s in range(args.seeds):
        sampled = pb.sample_taskwise(
            strict_source,
            ids,
            menu,
            budget,
            seed=args.seed + s,
        )
        tmp = sampled.copy()
        tmp["subsample_seed"] = s
        tmp["disease"] = disease
        selected_rows.append(tmp)

        X, pids = pb.build_speech_matrix(sampled, ids, lookups)
        if pids != ids:
            raise RuntimeError(f"{disease}: final all-rep speech matrix lost participants.")

        sp = pb.fixed_probe_oof(X, y, ids, folds)
        sp["subsample_seed"] = s
        speech_preds.append(sp)

        pdf = v6.protocol_dataframe(sampled, ids, demographics, menu)
        pp = pb.fixed_probe_oof(pdf.to_numpy(dtype=float), y, ids, folds)
        pp["subsample_seed"] = s
        protocol_preds.append(pp)

        diagnostics.extend(
            v6.balance_diagnostics(
                sampled,
                ids,
                disease,
                demographics,
                cohort_by_pid,
                "equal_taskwise_allrep_paircv",
                s,
            )
        )
        feature_diag.extend(
            v6.feature_diagnostics(pdf, ids, disease, cohort_by_pid, s)
        )

        if collect_ablation:
            for name, cols in v6.protocol_components(pdf).items():
                pred = pb.fixed_probe_oof(
                    pdf[cols].to_numpy(dtype=float), y, ids, folds
                )
                pred["subsample_seed"] = s
                ablation_preds.setdefault(name, []).append(pred)
                ablation_n_features[name] = len(cols)

    avg_s = pb.average_seed_predictions(speech_preds)
    avg_p = pb.average_seed_predictions(protocol_preds)

    from paired_bootstrap_helper import paired_match_auc_ci

    boot = paired_match_auc_ci(
        avg_s,
        avg_p,
        final_pairs,
        n_boot=args.bootstrap,
        seed=args.seed + 5001,
    )
    s_auc = boot["speech_auc"]
    s_lo = boot["speech_ci_low"]
    s_hi = boot["speech_ci_high"]
    p_auc = boot["protocol_auc"]
    p_lo = boot["protocol_ci_low"]
    p_hi = boot["protocol_ci_high"]

    print(
        f"[paired bootstrap] {disease}: pairs={boot['n_pairs']} "
        f"speech={s_auc:.3f} [{s_lo:.3f}, {s_hi:.3f}] "
        f"protocol={p_auc:.3f} [{p_lo:.3f}, {p_hi:.3f}]"
    )

    row = {
        "disease": disease,
        "stage": "equal_taskwise_allrep_paircv",
        "stage_order": STAGE_ORDER["equal_taskwise_allrep_paircv"],
        "cv_design": "pair_aware",
        "support_tier": tier,
        "n_case": len(final_pairs),
        "n_control": len(final_pairs),
        "n_pairs": len(final_pairs),
        "speech_auc": s_auc,
        "speech_ci_low": s_lo,
        "speech_ci_high": s_hi,
        "protocol_auc": p_auc,
        "protocol_ci_low": p_lo,
        "protocol_ci_high": p_hi,
        "protocol_chance_relation": v6.chance_relation(p_lo, p_hi),
        "n_task_families": len(menu),
        "per_task_budget": budget,
        "observations_per_participant": len(menu) * budget,
        "sessions_forced": False,
        "subsample_seeds": args.seeds,
    }

    ablation_rows: list[dict[str, Any]] = []
    if collect_ablation:
        for name, preds in ablation_preds.items():
            avg = pb.average_seed_predictions(preds)
            auc, lo, hi = pb.auc_ci(avg, args.bootstrap, args.seed + 5003)
            ablation_rows.append({
                "disease": disease,
                "stage": "equal_taskwise_allrep_paircv",
                "cv_design": "pair_aware",
                "support_tier": tier,
                "protocol_model": name,
                "auc": auc,
                "ci_low": lo,
                "ci_high": hi,
                "chance_relation": v6.chance_relation(lo, hi),
                "n_features": ablation_n_features[name],
                "n_pairs": len(final_pairs),
                "n_subsample_seeds": args.seeds,
            })

    selected = (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows
        else strict_source.iloc[0:0].copy()
    )
    return row, diagnostics, feature_diag, ablation_rows, final_pairs, selected


def qualifying_sessions(
    rows: pd.DataFrame,
    participant_ids: list[str],
    menu: list[str],
    budget: int = 1,
) -> dict[str, list[str]]:
    sub = rows[
        rows["participant_id"].isin(participant_ids)
        & rows["task_family"].isin(menu)
    ]
    result: dict[str, list[str]] = {}
    for (pid, session), g in sub.groupby(["participant_id", "session_id"], sort=True):
        counts = g.groupby("task_family").size()
        if all(int(counts.get(fam, 0)) >= budget for fam in menu):
            result.setdefault(str(pid), []).append(str(session))
    return result


def sample_single_session_taskwise(
    rows: pd.DataFrame,
    ids: list[str],
    menu: list[str],
    session_map: dict[str, list[str]],
    seed: int,
    budget: int = 1,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chunks = []
    for pid in ids:
        sessions = session_map.get(str(pid), [])
        if not sessions:
            raise RuntimeError(f"{pid} has no qualifying single session.")
        session = str(rng.choice(np.asarray(sessions, dtype=object)))
        g = rows[
            rows["participant_id"].eq(str(pid))
            & rows["session_id"].eq(session)
            & rows["task_family"].isin(menu)
        ]
        for fam in menu:
            gf = g[g["task_family"].eq(fam)]
            if len(gf) < budget:
                raise RuntimeError(
                    f"{pid}/{session}/{fam}: insufficient within-session observations."
                )
            idx = rng.choice(gf.index.to_numpy(), size=budget, replace=False)
            chunks.append(gf.loc[idx])

    out = pd.concat(chunks, ignore_index=True)
    expected = len(ids) * len(menu) * budget
    if len(out) != expected:
        raise RuntimeError(
            f"Single-session sampling invariant failed: {len(out)} vs {expected}."
        )
    sessions_per_pid = out.groupby("participant_id")["session_id"].nunique()
    if not sessions_per_pid.eq(1).all():
        raise RuntimeError("Single-session balance invariant failed.")
    return out


def run_single_session_sensitivity(
    *,
    disease: str,
    manifest: pd.DataFrame,
    final_pairs: list[tuple[str, str]],
    lookups: dict[str, dict],
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[str, str]],
]:
    menu = list(pb.COMMON_TASKS)
    ids0 = v6.flatten_pairs(final_pairs)
    strict = manifest[
        manifest["participant_id"].isin(ids0)
        & manifest["task_family"].isin(menu)
        & manifest["_has_all_reps"]
    ].copy()

    session_map = qualifying_sessions(strict, ids0, menu, budget=1)
    eligible = set(session_map)
    kept_pairs = v6.filter_pairs_by_eligible(final_pairs, eligible)
    if len(kept_pairs) < args.exploratory_min_pairs:
        raise RuntimeError(
            f"Single-session support leaves {len(kept_pairs)} pairs, below "
            f"exploratory minimum {args.exploratory_min_pairs}."
        )
    tier = support_tier(len(kept_pairs), args)
    ids = v6.flatten_pairs(kept_pairs)
    y = pb.make_labels(ids, disease, cohort_by_pid)
    folds = pair_aware_folds(kept_pairs, ids, args.seed)

    speech_preds = []
    protocol_preds = []
    ablation_preds: dict[str, list[pd.DataFrame]] = {}
    ablation_n_features: dict[str, int] = {}
    diagnostics = []
    selected_rows = []

    for s in range(args.seeds):
        sampled = sample_single_session_taskwise(
            strict,
            ids,
            menu,
            session_map,
            seed=args.seed + 10000 + s,
            budget=1,
        )
        tmp = sampled.copy()
        tmp["subsample_seed"] = s
        tmp["disease"] = disease
        selected_rows.append(tmp)

        X, pids = pb.build_speech_matrix(sampled, ids, lookups)
        if pids != ids:
            raise RuntimeError(f"{disease}: single-session speech matrix lost participants.")
        sp = pb.fixed_probe_oof(X, y, ids, folds)
        sp["subsample_seed"] = s
        speech_preds.append(sp)

        pdf = v6.protocol_dataframe(sampled, ids, demographics, menu)
        pp = pb.fixed_probe_oof(pdf.to_numpy(dtype=float), y, ids, folds)
        pp["subsample_seed"] = s
        protocol_preds.append(pp)

        diagnostics.extend(
            v6.balance_diagnostics(
                sampled,
                ids,
                disease,
                demographics,
                cohort_by_pid,
                "single_session_equal_taskwise_allrep_paircv",
                s,
            )
        )

        for name, cols in v6.protocol_components(pdf).items():
            pred = pb.fixed_probe_oof(
                pdf[cols].to_numpy(dtype=float), y, ids, folds
            )
            pred["subsample_seed"] = s
            ablation_preds.setdefault(name, []).append(pred)
            ablation_n_features[name] = len(cols)

    avg_s = pb.average_seed_predictions(speech_preds)
    avg_p = pb.average_seed_predictions(protocol_preds)

    from paired_bootstrap_helper import paired_match_auc_ci

    boot = paired_match_auc_ci(
        avg_s,
        avg_p,
        kept_pairs,
        n_boot=args.bootstrap,
        seed=args.seed + 11001,
    )
    s_auc = boot["speech_auc"]
    s_lo = boot["speech_ci_low"]
    s_hi = boot["speech_ci_high"]
    p_auc = boot["protocol_auc"]
    p_lo = boot["protocol_ci_low"]
    p_hi = boot["protocol_ci_high"]

    print(
        f"[paired bootstrap single-session] {disease}: pairs={boot['n_pairs']} "
        f"speech={s_auc:.3f} [{s_lo:.3f}, {s_hi:.3f}] "
        f"protocol={p_auc:.3f} [{p_lo:.3f}, {p_hi:.3f}]"
    )

    row = {
        "disease": disease,
        "stage": "single_session_equal_taskwise_allrep_paircv",
        "stage_order": STAGE_ORDER["single_session_equal_taskwise_allrep_paircv"],
        "cv_design": "pair_aware",
        "support_tier": tier,
        "n_case": len(kept_pairs),
        "n_control": len(kept_pairs),
        "n_pairs": len(kept_pairs),
        "speech_auc": s_auc,
        "speech_ci_low": s_lo,
        "speech_ci_high": s_hi,
        "protocol_auc": p_auc,
        "protocol_ci_low": p_lo,
        "protocol_ci_high": p_hi,
        "protocol_chance_relation": v6.chance_relation(p_lo, p_hi),
        "n_task_families": len(menu),
        "per_task_budget": 1,
        "observations_per_participant": len(menu),
        "sessions_forced": True,
        "sessions_per_participant": 1,
        "subsample_seeds": args.seeds,
    }

    ablations = []
    for name, preds in ablation_preds.items():
        avg = pb.average_seed_predictions(preds)
        auc, lo, hi = pb.auc_ci(avg, args.bootstrap, args.seed + 11003)
        ablations.append({
            "disease": disease,
            "stage": "single_session_equal_taskwise_allrep_paircv",
            "cv_design": "pair_aware",
            "support_tier": tier,
            "protocol_model": name,
            "auc": auc,
            "ci_low": lo,
            "ci_high": hi,
            "chance_relation": v6.chance_relation(lo, hi),
            "n_features": ablation_n_features[name],
            "n_pairs": len(kept_pairs),
            "n_subsample_seeds": args.seeds,
        })

    selected = pd.concat(selected_rows, ignore_index=True)
    return row, diagnostics, ablations, [selected], kept_pairs


def run_caliper_paircv(
    *,
    disease: str,
    caliper_sd: float,
    manifest: pd.DataFrame,
    demographics: pd.DataFrame,
    cohort_by_pid: pd.Series,
    lookups: dict[str, dict],
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_ids, control_ids, caliper_years = pb.match_demographics(
        disease, cohort_by_pid, demographics, caliper_sd
    )
    pairs = v6.pair_list(case_ids, control_ids)
    tier0 = support_tier(len(pairs), args)
    if tier0 == "insufficient_support":
        raise RuntimeError(
            f"Demographic matching retained only {len(pairs)} pairs."
        )

    row, _, _, _, final_pairs, _ = run_final_equal_evidence(
        disease=disease,
        manifest=manifest,
        matched_pairs=pairs,
        lookups=lookups,
        demographics=demographics,
        cohort_by_pid=cohort_by_pid,
        args=args,
        collect_ablation=False,
    )
    return {
        "disease": disease,
        "age_caliper_sd": caliper_sd,
        "age_caliper_years": caliper_years,
        "initial_matched_pairs": len(pairs),
        "initial_support_tier": tier0,
        "final_balanced_pairs": len(final_pairs),
        "final_support_tier": row.get("support_tier"),
        "per_task_budget": row.get("per_task_budget"),
        "speech_auc": row["speech_auc"],
        "speech_ci_low": row["speech_ci_low"],
        "speech_ci_high": row["speech_ci_high"],
        "protocol_auc": row["protocol_auc"],
        "protocol_ci_low": row["protocol_ci_low"],
        "protocol_ci_high": row["protocol_ci_high"],
        "protocol_chance_relation": row["protocol_chance_relation"],
        "cv_design": "pair_aware",
    }


def final_status(row: pd.Series | None) -> str:
    if row is None:
        return "not_estimable"
    tier = str(row.get("support_tier", ""))
    if tier == "exploratory_small_n":
        return "exploratory_small_n"
    p_lo = float(row["protocol_ci_low"])
    p_hi = float(row["protocol_ci_high"])
    s_lo = float(row["speech_ci_low"])
    if p_lo > 0.5:
        return "protocol_sensitive"
    if p_lo <= 0.5 <= p_hi and s_lo > 0.5:
        return "observed_protocol_standardized_detectability_supported"
    if s_lo <= 0.5:
        return "speech_not_resolved_after_standardization"
    return "qualified_with_residual_uncertainty"


def make_publication_summary(
    stages: pd.DataFrame,
    published_raw: dict[str, float],
    diseases: list[str],
) -> pd.DataFrame:
    rows = []
    for disease in diseases:
        sub = stages[stages["disease"].eq(disease)].copy()
        by_stage = {r["stage"]: r for _, r in sub.iterrows()}

        raw = by_stage.get("raw_same_probe_original_cv")
        original = by_stage.get("final_cohort_original_evidence_paircv")
        menu = by_stage.get("final_cohort_shared_menu_observed_counts_paircv")
        final = by_stage.get("equal_taskwise_allrep_paircv")
        session = by_stage.get("single_session_equal_taskwise_allrep_paircv")

        out = {
            "disease": disease,
            "published_raw_speech_auc_reference": published_raw.get(disease, np.nan),
            "raw_same_probe_speech_auc": raw["speech_auc"] if raw is not None else np.nan,
            "raw_protocol_auc": raw["protocol_auc"] if raw is not None else np.nan,
            "final_pairs": final["n_pairs"] if final is not None else np.nan,
            "support_tier": final.get("support_tier", np.nan) if final is not None else np.nan,
            "final_cohort_original_speech_auc": (
                original["speech_auc"] if original is not None else np.nan
            ),
            "final_cohort_original_protocol_auc": (
                original["protocol_auc"] if original is not None else np.nan
            ),
            "final_cohort_shared_menu_speech_auc": (
                menu["speech_auc"] if menu is not None else np.nan
            ),
            "final_cohort_shared_menu_protocol_auc": (
                menu["protocol_auc"] if menu is not None else np.nan
            ),
            "balanced_speech_auc": final["speech_auc"] if final is not None else np.nan,
            "balanced_speech_ci_low": (
                final["speech_ci_low"] if final is not None else np.nan
            ),
            "balanced_speech_ci_high": (
                final["speech_ci_high"] if final is not None else np.nan
            ),
            "balanced_protocol_auc": final["protocol_auc"] if final is not None else np.nan,
            "balanced_protocol_ci_low": (
                final["protocol_ci_low"] if final is not None else np.nan
            ),
            "balanced_protocol_ci_high": (
                final["protocol_ci_high"] if final is not None else np.nan
            ),
            "single_session_pairs": session["n_pairs"] if session is not None else np.nan,
            "single_session_speech_auc": (
                session["speech_auc"] if session is not None else np.nan
            ),
            "single_session_protocol_auc": (
                session["protocol_auc"] if session is not None else np.nan
            ),
            "validity_status": final_status(final),
        }

        if original is not None and menu is not None and final is not None:
            out["same_cohort_menu_effect_speech"] = (
                menu["speech_auc"] - original["speech_auc"]
            )
            out["same_cohort_evidence_standardization_effect_speech"] = (
                final["speech_auc"] - menu["speech_auc"]
            )
            out["same_cohort_total_standardization_effect_speech"] = (
                final["speech_auc"] - original["speech_auc"]
            )
            out["same_cohort_menu_effect_protocol"] = (
                menu["protocol_auc"] - original["protocol_auc"]
            )
            out["same_cohort_evidence_standardization_effect_protocol"] = (
                final["protocol_auc"] - menu["protocol_auc"]
            )
            out["same_cohort_total_standardization_effect_protocol"] = (
                final["protocol_auc"] - original["protocol_auc"]
            )
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.exploratory_min_pairs < 2:
        raise ValueError("--exploratory-min-pairs must be >=2.")
    if args.primary_min_pairs < args.exploratory_min_pairs:
        raise ValueError("--primary-min-pairs must be >= --exploratory-min-pairs.")

    repo = Path(args.repo_root).expanduser().resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo / manifest_path
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

    observed_diseases = set(manifest["cohort"].dropna().astype(str))
    requested = list(dict.fromkeys(args.diseases))
    for disease in requested:
        if disease not in observed_diseases:
            print(f"[WARNING] requested cohort not found in frozen manifest: {disease}")

    stage_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    feature_diag_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    session_ablation_rows: list[dict[str, Any]] = []
    caliper_rows: list[dict[str, Any]] = []
    selected_final_rows: list[pd.DataFrame] = []
    selected_session_rows: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []

    for disease in requested:
        print("\n" + "=" * 96)
        print(f"V7 FINAL VALIDATION: {disease}")
        print("=" * 96)

        if disease not in observed_diseases:
            failures.append({
                "disease": disease,
                "stage": "setup",
                "error": "disease_not_found_in_manifest",
            })
            continue

        disease_rows = manifest[
            manifest["cohort"].isin([disease, "Healthy Control"])
        ].copy()
        raw_families = sorted(
            disease_rows["task_family"].dropna().unique().tolist()
        )
        demo_ids = set(demographics["participant_id"].astype(str))
        raw_ids = sorted(
            pid for pid in disease_rows["participant_id"].unique().astype(str)
            if pid in demo_ids
        )

        # RAW: always attempt, even for tiny cohorts.
        try:
            raw_row, raw_diag = evaluate_raw_stage(
                disease=disease,
                disease_rows=disease_rows,
                candidate_ids=raw_ids,
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
            stage_rows.append(raw_row)
            balance_rows.extend(raw_diag)
            print(
                f"[raw] speech={raw_row['speech_auc']:.3f} "
                f"protocol={raw_row['protocol_auc']:.3f} "
                f"n={raw_row['n_case']}+{raw_row['n_control']}"
            )
        except Exception as exc:
            failures.append({
                "disease": disease,
                "stage": "raw_same_probe_original_cv",
                "error": repr(exc),
            })
            print(f"[raw FAILED] {exc}", file=sys.stderr)

        # MATCHING.
        try:
            case_ids, control_ids, caliper_years = pb.match_demographics(
                disease, cohort_by_pid, demographics, args.age_caliper_sd
            )
            matched_pairs = v6.pair_list(case_ids, control_ids)
            tier0 = support_tier(len(matched_pairs), args)
            print(
                f"[matching] pairs={len(matched_pairs)} "
                f"tier={tier0} caliper_years={caliper_years:.3f}"
            )
            if tier0 == "insufficient_support":
                raise RuntimeError(
                    f"Only {len(matched_pairs)} matched pairs; exploratory minimum is "
                    f"{args.exploratory_min_pairs}."
                )
        except Exception as exc:
            failures.append({
                "disease": disease,
                "stage": "demographic_matching",
                "error": repr(exc),
            })
            print(f"[matching FAILED] {exc}", file=sys.stderr)
            continue

        # S1 matched pair-aware.
        try:
            matched_ids = v6.flatten_pairs(matched_pairs)
            matched_rows = disease_rows[
                disease_rows["participant_id"].isin(matched_ids)
            ].copy()
            s1, d1, matched_supported_pairs = evaluate_pair_stage(
                disease=disease,
                stage="age_sex_matched_paircv",
                rows=matched_rows,
                pairs=matched_pairs,
                lookups=lookups,
                demographics=demographics,
                cohort_by_pid=cohort_by_pid,
                bootstrap=args.bootstrap,
                seed=args.seed,
                all_families=raw_families,
            )
            s1["support_tier"] = support_tier(len(matched_supported_pairs), args)
            s1["age_caliper_years"] = caliper_years
            stage_rows.append(s1)
            balance_rows.extend(d1)
        except Exception as exc:
            failures.append({
                "disease": disease,
                "stage": "age_sex_matched_paircv",
                "error": repr(exc),
            })
            print(f"[matched FAILED] {exc}", file=sys.stderr)
            continue

        # S2 shared menu, counts still observed.
        try:
            shared_rows = matched_rows[
                matched_rows["task_family"].isin(pb.COMMON_TASKS)
            ].copy()
            s2, d2, shared_supported_pairs = evaluate_pair_stage(
                disease=disease,
                stage="shared_7_task_menu_paircv",
                rows=shared_rows,
                pairs=matched_supported_pairs,
                lookups=lookups,
                demographics=demographics,
                cohort_by_pid=cohort_by_pid,
                bootstrap=args.bootstrap,
                seed=args.seed,
                all_families=list(pb.COMMON_TASKS),
            )
            s2["support_tier"] = support_tier(len(shared_supported_pairs), args)
            stage_rows.append(s2)
            balance_rows.extend(d2)
        except Exception as exc:
            failures.append({
                "disease": disease,
                "stage": "shared_7_task_menu_paircv",
                "error": repr(exc),
            })
            print(f"[shared menu FAILED] {exc}", file=sys.stderr)
            continue

        # S3 complete all seven tasks, counts still observed.
        try:
            complete_pairs = v6.complete_menu_pairs(
                shared_rows,
                shared_supported_pairs,
                list(pb.COMMON_TASKS),
                budget=1,
            )
            if len(complete_pairs) < args.exploratory_min_pairs:
                raise RuntimeError(
                    f"Only {len(complete_pairs)} complete-menu pairs."
                )
            complete_ids = v6.flatten_pairs(complete_pairs)
            complete_rows = shared_rows[
                shared_rows["participant_id"].isin(complete_ids)
            ].copy()
            s3, d3, complete_supported_pairs = evaluate_pair_stage(
                disease=disease,
                stage="complete_shared_7_tasks_paircv",
                rows=complete_rows,
                pairs=complete_pairs,
                lookups=lookups,
                demographics=demographics,
                cohort_by_pid=cohort_by_pid,
                bootstrap=args.bootstrap,
                seed=args.seed,
                all_families=list(pb.COMMON_TASKS),
            )
            s3["support_tier"] = support_tier(len(complete_supported_pairs), args)
            stage_rows.append(s3)
            balance_rows.extend(d3)
        except Exception as exc:
            failures.append({
                "disease": disease,
                "stage": "complete_shared_7_tasks_paircv",
                "error": repr(exc),
            })
            print(f"[complete menu FAILED] {exc}", file=sys.stderr)
            continue

        # Final pair-aware equal evidence + all-representation support.
        try:
            final_row, final_diag, fdiag, abl, final_pairs, selected = (
                run_final_equal_evidence(
                    disease=disease,
                    manifest=manifest,
                    matched_pairs=matched_pairs,
                    lookups=lookups,
                    demographics=demographics,
                    cohort_by_pid=cohort_by_pid,
                    args=args,
                    collect_ablation=True,
                )
            )
            final_row["published_raw_speech_auc_reference"] = published_raw.get(
                disease, np.nan
            )
            stage_rows.append(final_row)
            balance_rows.extend(final_diag)
            feature_diag_rows.extend(fdiag)
            ablation_rows.extend(abl)
            selected_final_rows.append(selected)

            print(
                f"[final] pairs={len(final_pairs)} tier={final_row['support_tier']} "
                f"speech={final_row['speech_auc']:.3f} "
                f"protocol={final_row['protocol_auc']:.3f}"
            )
        except Exception as exc:
            failures.append({
                "disease": disease,
                "stage": "equal_taskwise_allrep_paircv",
                "error": repr(exc),
            })
            print(f"[final FAILED] {exc}", file=sys.stderr)
            continue

        # SAME FINAL COHORT decomposition.
        try:
            final_ids = v6.flatten_pairs(final_pairs)
            final_original_rows = disease_rows[
                disease_rows["participant_id"].isin(final_ids)
            ].copy()
            d0, dd0, kept0 = evaluate_pair_stage(
                disease=disease,
                stage="final_cohort_original_evidence_paircv",
                rows=final_original_rows,
                pairs=final_pairs,
                lookups=lookups,
                demographics=demographics,
                cohort_by_pid=cohort_by_pid,
                bootstrap=args.bootstrap,
                seed=args.seed,
                all_families=raw_families,
            )
            d0["support_tier"] = final_row["support_tier"]
            stage_rows.append(d0)
            balance_rows.extend(dd0)

            final_menu_rows = final_original_rows[
                final_original_rows["task_family"].isin(pb.COMMON_TASKS)
            ].copy()
            dmenu, ddmenu, keptmenu = evaluate_pair_stage(
                disease=disease,
                stage="final_cohort_shared_menu_observed_counts_paircv",
                rows=final_menu_rows,
                pairs=final_pairs,
                lookups=lookups,
                demographics=demographics,
                cohort_by_pid=cohort_by_pid,
                bootstrap=args.bootstrap,
                seed=args.seed,
                all_families=list(pb.COMMON_TASKS),
            )
            dmenu["support_tier"] = final_row["support_tier"]
            stage_rows.append(dmenu)
            balance_rows.extend(ddmenu)
        except Exception as exc:
            failures.append({
                "disease": disease,
                "stage": "same_final_cohort_decomposition",
                "error": repr(exc),
            })
            print(f"[decomposition FAILED] {exc}", file=sys.stderr)

        # SINGLE-SESSION sensitivity.
        try:
            ss_row, ss_diag, ss_abl, ss_selected_chunks, ss_pairs = (
                run_single_session_sensitivity(
                    disease=disease,
                    manifest=manifest,
                    final_pairs=final_pairs,
                    lookups=lookups,
                    demographics=demographics,
                    cohort_by_pid=cohort_by_pid,
                    args=args,
                )
            )
            if ss_row is not None:
                stage_rows.append(ss_row)
                balance_rows.extend(ss_diag)
                session_ablation_rows.extend(ss_abl)
                for x in ss_selected_chunks:
                    selected_session_rows.append(x)
                print(
                    f"[single-session] pairs={len(ss_pairs)} "
                    f"speech={ss_row['speech_auc']:.3f} "
                    f"protocol={ss_row['protocol_auc']:.3f}"
                )
        except Exception as exc:
            failures.append({
                "disease": disease,
                "stage": "single_session_equal_taskwise_allrep_paircv",
                "error": repr(exc),
            })
            print(f"[single-session unavailable] {exc}")

        # CALIPER sensitivity.
        for caliper in args.caliper_grid:
            try:
                caliper_rows.append(
                    run_caliper_paircv(
                        disease=disease,
                        caliper_sd=float(caliper),
                        manifest=manifest,
                        demographics=demographics,
                        cohort_by_pid=cohort_by_pid,
                        lookups=lookups,
                        args=args,
                    )
                )
            except Exception as exc:
                caliper_rows.append({
                    "disease": disease,
                    "age_caliper_sd": float(caliper),
                    "error": repr(exc),
                    "cv_design": "pair_aware",
                })

    # ---------- SAVE ----------
    stages = pd.DataFrame(stage_rows)
    if not stages.empty:
        stages = stages.sort_values(["disease", "stage_order"]).reset_index(drop=True)

    balance = pd.DataFrame(balance_rows)
    fdiag_raw = pd.DataFrame(feature_diag_rows)
    fdiag = v6.summarize_feature_diagnostics(fdiag_raw)
    ablations = pd.DataFrame(ablation_rows)
    session_ablation = pd.DataFrame(session_ablation_rows)
    calipers = pd.DataFrame(caliper_rows)
    failure_df = pd.DataFrame(failures)
    publication = make_publication_summary(stages, published_raw, requested)

    stages.to_csv(out_dir / "stagewise_validity_paircv.csv", index=False)
    balance.to_csv(out_dir / "balance_diagnostics_paircv.csv", index=False)
    fdiag_raw.to_csv(
        out_dir / "final_protocol_feature_diagnostics_by_seed_paircv.csv",
        index=False,
    )
    fdiag.to_csv(
        out_dir / "final_protocol_feature_diagnostics_paircv.csv",
        index=False,
    )
    ablations.to_csv(out_dir / "protocol_ablation_paircv.csv", index=False)
    session_ablation.to_csv(
        out_dir / "single_session_protocol_ablation_paircv.csv",
        index=False,
    )
    calipers.to_csv(
        out_dir / "age_caliper_sensitivity_paircv.csv",
        index=False,
    )
    failure_df.to_csv(out_dir / "support_and_failures.csv", index=False)
    publication.to_csv(out_dir / "publication_summary.csv", index=False)

    if selected_final_rows:
        pd.concat(selected_final_rows, ignore_index=True).to_csv(
            out_dir / "final_equal_evidence_selected_observations.csv",
            index=False,
        )
    if selected_session_rows:
        pd.concat(selected_session_rows, ignore_index=True).to_csv(
            out_dir / "single_session_selected_observations.csv",
            index=False,
        )

    metadata = {
        "analysis": "v8_final_validation_all_diagnoses",
        "extension_version": "v8-pair-aware-all-diagnoses-numerically-safe-pls",
        "default_diseases": DEFAULT_DISEASES,
        "requested_diseases": requested,
        "shared_task_families": list(pb.COMMON_TASKS),
        "cv": {
            "raw": "frozen original fold map when available",
            "matched_and_balanced": "matched case-control pair assigned to same fold",
        },
        "support": {
            "primary_min_pairs": args.primary_min_pairs,
            "exploratory_min_pairs": args.exploratory_min_pairs,
            "exploratory analyses are not promoted to primary claims": True,
        },
        "primary_age_caliper_sd": args.age_caliper_sd,
        "caliper_grid_sd": args.caliper_grid,
        "seeds": args.seeds,
        "bootstrap": args.bootstrap,
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
            "Stage and same-cohort AUC changes are descriptive measurement-sensitivity estimates, not causal confounding fractions.",
            "A protocol CI containing 0.5 is not an equivalence test.",
            "Balanced speech detectability is conditional on observed protocol controls and does not establish pathology-specific causality.",
            "Small cohorts below the primary pair threshold are explicitly labeled exploratory.",
            "Single-session analysis is a support-constrained sensitivity analysis and may be non-estimable for some cohorts.",
        ],
        "failures": failures,
    }
    (out_dir / "v7_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("\n" + "=" * 96)
    print("V7 PUBLICATION SUMMARY")
    print("=" * 96)
    show_cols = [
        "disease",
        "published_raw_speech_auc_reference",
        "raw_protocol_auc",
        "final_pairs",
        "support_tier",
        "balanced_speech_auc",
        "balanced_protocol_auc",
        "single_session_protocol_auc",
        "validity_status",
    ]
    print(publication[show_cols].to_string(index=False))

    print("\nSaved under:")
    print(out_dir)
    print("\nPrimary files:")
    for name in [
        "publication_summary.csv",
        "stagewise_validity_paircv.csv",
        "protocol_ablation_paircv.csv",
        "single_session_protocol_ablation_paircv.csv",
        "age_caliper_sensitivity_paircv.csv",
        "balance_diagnostics_paircv.csv",
        "final_protocol_feature_diagnostics_paircv.csv",
        "support_and_failures.csv",
        "v7_metadata.json",
    ]:
        print(f"  {out_dir / name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

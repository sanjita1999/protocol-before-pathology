"""
Shared utilities for the PCMVF analyses.

This module imports the frozen PCMVF code in `pcmvf/` and never modifies it.
Every script in `analysis/experiments/` builds on the functions here so that the
primary PCMVF pipeline (matching, shared-menu restriction, equal task-wise
evidence, pair-aware folds, 20 subsampling seeds, PLS+logistic probe) is reused
exactly, and only the quantity being tested changes.

Key entry points
----------------
load_data()                    -> Data (manifest, demographics, feature lookups, metadata)
default_args(**overrides)      -> argparse.Namespace with the frozen v8 defaults
standardized_cohort(...)       -> final matched pairs / budget / folds for one disease
standardized_predictions(...)  -> seed-averaged OOF predictions for any feature sets / models
raw_predictions(...)           -> full-cohort (pre-standardization) OOF predictions
pair_bootstrap(...)            -> AUC CIs + pairwise deltas, resampling matched pairs
delong_*                       -> DeLong AUC variance / paired test
tost_auc(...)                  -> equivalence test of an AUC against 0.5
model_oof(name)                -> alternative classifiers with the same OOF interface
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

COMMON_DIR = Path(__file__).resolve().parent
REPO = COMMON_DIR.parents[1]                 # release root (config/, outputs/, pcmvf/, analysis/)
FRAMEWORK = REPO / "pcmvf"
RESULTS = Path(os.environ.get("PCMVF_RESULTS", REPO / "results"))
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

import run_protocol_balanced as pb  # noqa: E402
import run_all_experiments as v6  # noqa: E402
import run_v7_all_diagnoses as v7  # noqa: E402
from paired_bootstrap_helper import paired_match_auc_ci  # noqa: E402,F401

DEFAULT_B2AI_ROOT = os.environ.get("B2AI_ROOT")  # Bridge2AI Voice v3.0.0 root (phenotype/, features/)
DISEASES = list(v7.DEFAULT_DISEASES)
CONTROL = "Healthy Control"
MENU = list(pb.COMMON_TASKS)
# The 8 released representations (470-D). New representations (PPG, SSL, external
# features) are added to Data.lookups only and are NEVER registered in
# pb.REP_SPECS, so the frozen speech model cannot change silently.
ORIGINAL_REPS = list(pb.REP_SPECS)
V8_OUT = REPO / "outputs" / "protocol_balanced_framework" / "v8_all_diagnoses"

ProbeFn = Callable[[np.ndarray, np.ndarray, list, np.ndarray], pd.DataFrame]
FeatureFn = Callable[[pd.DataFrame, list], np.ndarray]


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def default_args(**overrides: Any) -> argparse.Namespace:
    """Frozen v8 defaults (run_v7_all_diagnoses.parse_args)."""
    args = argparse.Namespace(
        seeds=20,
        bootstrap=2000,
        seed=2026,
        primary_min_pairs=20,
        exploratory_min_pairs=8,
        feature_retention=0.70,
        age_caliper_sd=0.25,
        per_task_budgets=[5, 3, 2, 1],
        caliper_grid=[0.10, 0.25, 0.50],
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def add_common_cli(p: argparse.ArgumentParser, out_default: str) -> argparse.ArgumentParser:
    p.add_argument("--b2ai-root", default=DEFAULT_B2AI_ROOT)
    p.add_argument("--diseases", nargs="+", default=DISEASES)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--age-caliper-sd", type=float, default=0.25)
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 2 seeds, 200 bootstrap replicates.")
    p.add_argument("--out-dir", default=out_default)
    return p


def finalize_cli(ns: argparse.Namespace) -> argparse.Namespace:
    if ns.quick:
        ns.seeds, ns.bootstrap = 2, 200
    out = Path(ns.out_dir)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)
    ns.out_dir = out
    return ns


def args_from_cli(ns: argparse.Namespace, **extra: Any) -> argparse.Namespace:
    return default_args(
        seeds=ns.seeds,
        bootstrap=ns.bootstrap,
        seed=ns.seed,
        age_caliper_sd=ns.age_caliper_sd,
        **extra,
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Data:
    manifest: pd.DataFrame
    cohort_by_pid: pd.Series
    demographics: pd.DataFrame
    lookups: dict[str, dict]
    fold_map: dict[str, int] | None
    b2ai_root: Path
    participant_meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def with_cohorts(self, cohort_by_pid: pd.Series, manifest: pd.DataFrame | None = None) -> "Data":
        """Copy with relabelled cohorts (used by semi-synthetic experiments)."""
        m = (manifest if manifest is not None else self.manifest).copy()
        m["cohort"] = m["participant_id"].map(cohort_by_pid)
        m = m[m["cohort"].notna()].copy()
        return Data(
            manifest=m,
            cohort_by_pid=cohort_by_pid,
            demographics=self.demographics,
            lookups=self.lookups,
            fold_map=None,
            b2ai_root=self.b2ai_root,
            participant_meta=self.participant_meta,
        )


def _norm_name(x: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(x).lower()).strip("-")


def _strip_pid(x: Any) -> str:
    s = str(x).strip()
    return s.lstrip("0") or "0"


def load_data(b2ai_root: str | Path | None = None, attach_metadata: bool = True) -> Data:
    """Replicates run_v7_all_diagnoses.main() setup (lines 904-926)."""
    if not (b2ai_root or DEFAULT_B2AI_ROOT):
        raise RuntimeError("Set B2AI_ROOT (or pass --b2ai-root) to the Bridge2AI Voice v3.0.0 directory.")
    b2ai = Path(b2ai_root or DEFAULT_B2AI_ROOT).expanduser().resolve()
    manifest = pb.load_manifest(
        REPO / "outputs" / "manifests" / "recordings.csv",
        REPO / "config" / "task_families.yaml",
    )
    cohort_by_pid = (
        manifest[["participant_id", "cohort"]]
        .drop_duplicates("participant_id")
        .set_index("participant_id")["cohort"]
    )
    demographics = pb.discover_demographics(REPO, b2ai, manifest, None)
    fold_map = pb.discover_fold_map(REPO, None)
    lookups = pb.build_all_cached_feature_lookups(pb.discover_embedding_caches(REPO))
    manifest = pb.attach_feature_presence(manifest, lookups)
    data = Data(manifest, cohort_by_pid, demographics, lookups, fold_map, b2ai)
    if attach_metadata:
        attach_acquisition_metadata(data)
    return data


def attach_acquisition_metadata(data: Data) -> None:
    """
    Join Bridge2AI acquisition metadata that the submitted paper did not use.

    Recording level  (phenotype/task/recording.tsv):
        rec_microphone, rec_input_gain, rec_duration
        join key = participant (leading zeros stripped) + session + normalised
        recording_name; verified to match 100% of manifest rows.
    Task level       (phenotype/task/acoustic_task.tsv):
        task_module = acoustic_task_cohort (generic / voice / neurology /
        respiratory / mood): which protocol module assigned the task.
    Participant level (phenotype/demographics/demographics.tsv, session.tsv):
        country (USA / Canada; site proxy), n_sessions_total, mean session
        duration, primary microphone.
    """
    ph = data.b2ai_root / "phenotype"
    m = data.manifest
    m["_pid"] = m["participant_id"].map(_strip_pid)
    m["_key"] = m["_pid"] + "::" + m["session_id"].astype(str) + "::" + m["raw_task"].map(_norm_name)

    rec_path = ph / "task" / "recording.tsv"
    if rec_path.exists():
        r = pd.read_csv(rec_path, sep="\t", low_memory=False, dtype={"participant_id": str})
        r["_key"] = (
            r["participant_id"].map(_strip_pid) + "::" + r["recording_session_id"].astype(str)
            + "::" + r["recording_name"].map(_norm_name)
        )
        r = r.drop_duplicates("_key").set_index("_key")
        m["rec_microphone"] = m["_key"].map(r["recording_microphone"])
        m["rec_input_gain"] = pd.to_numeric(m["_key"].map(r["recording_input_gain"]), errors="coerce")
        m["rec_duration"] = pd.to_numeric(m["_key"].map(r["recording_duration"]), errors="coerce")
        print(f"[metadata] recording.tsv matched {m['rec_microphone'].notna().mean():.1%} of rows")

    at_path = ph / "task" / "acoustic_task.tsv"
    if at_path.exists():
        a = pd.read_csv(at_path, sep="\t", low_memory=False, dtype={"participant_id": str})
        a["_k2"] = (
            a["participant_id"].map(_strip_pid) + "::" + a["acoustic_task_session_id"].astype(str)
            + "::" + a["acoustic_task_name"].map(_norm_name)
        )
        mod = a.drop_duplicates("_k2").set_index("_k2")["acoustic_task_cohort"]
        base = m["raw_task"].map(_norm_name).str.replace(r"-\d+$", "", regex=True)
        k_full = m["_pid"] + "::" + m["session_id"].astype(str) + "::" + m["raw_task"].map(_norm_name)
        k_base = m["_pid"] + "::" + m["session_id"].astype(str) + "::" + base
        m["task_module"] = k_full.map(mod).fillna(k_base.map(mod))
        print(f"[metadata] acoustic_task.tsv module matched {m['task_module'].notna().mean():.1%} of rows")

    pids = pd.Index(m["participant_id"].unique(), name="participant_id")
    meta = pd.DataFrame(index=pids)
    meta["_pid"] = meta.index.map(_strip_pid)
    demo_path = ph / "demographics" / "demographics.tsv"
    if demo_path.exists():
        d = pd.read_csv(demo_path, sep="\t", low_memory=False, dtype={"participant_id": str})
        d["_pid"] = d["participant_id"].map(_strip_pid)
        country = d.groupby("_pid")["country"].agg(
            lambda s: "|".join(sorted(s.dropna().astype(str).unique())) or np.nan
        )
        meta["country"] = meta["_pid"].map(country)
        meta["country_canada"] = meta["country"].fillna("").str.contains("Canada").astype(float)
    ses_path = ph / "task" / "session.tsv"
    if ses_path.exists():
        s = pd.read_csv(ses_path, sep="\t", low_memory=False, dtype={"participant_id": str})
        s["_pid"] = s["participant_id"].map(_strip_pid)
        meta["session_duration_mean"] = meta["_pid"].map(s.groupby("_pid")["session_duration"].mean())
    if "rec_microphone" in m:
        mic = m.groupby("participant_id")["rec_microphone"].agg(
            lambda x: x.mode().iat[0] if x.notna().any() else np.nan
        )
        meta["primary_microphone"] = mic
        meta["share_ipad_mic"] = m.assign(_i=m["rec_microphone"].eq("iPad Microphone")).groupby(
            "participant_id")["_i"].mean()
        meta["mean_input_gain"] = m.groupby("participant_id")["rec_input_gain"].mean()
        meta["mean_rec_duration"] = m.groupby("participant_id")["rec_duration"].mean()
    meta["cohort"] = meta.index.map(data.cohort_by_pid)
    data.participant_meta = meta.drop(columns=["_pid"])


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def speech_matrix(rows: pd.DataFrame, ids: list, lookups: dict, reps: list[str] | None = None,
                  strict: bool = True) -> tuple[np.ndarray, list[str]]:
    """Participant-level concatenation of selected representations (default: all 8 = 470-D)."""
    reps = list(reps or ORIGINAL_REPS)
    by_pid = {str(p): g for p, g in rows.groupby("participant_id", sort=False)}
    X, kept = [], []
    for pid in map(str, ids):
        g = by_pid.get(pid)
        if g is None or g.empty:
            continue
        keys = list(zip(g["participant_id"], g["session_id"], g["task_norm"]))
        pieces, ok = [], True
        for rep in reps:
            vals = [lookups[rep][k] for k in keys if k in lookups[rep]]
            if (strict and len(vals) != len(keys)) or not vals:
                ok = False
                break
            pieces.append(np.nanmean(np.stack(vals), axis=0))
        if ok:
            X.append(np.concatenate(pieces).astype(np.float32))
            kept.append(pid)
    dim = sum(int(pb.REP_SPECS[r]["dim"]) if r in pb.REP_SPECS
              else len(next(iter(lookups[r].values()))) for r in reps)
    return (np.stack(X) if X else np.empty((0, dim))), kept


def speech_fn(data: Data, reps: list[str] | None = None) -> FeatureFn:
    def f(rows: pd.DataFrame, ids: list) -> np.ndarray:
        X, kept = speech_matrix(rows, ids, data.lookups, reps)
        if kept != [str(i) for i in ids]:
            raise RuntimeError("speech matrix lost participants")
        return X
    return f


def protocol_df(data: Data, rows: pd.DataFrame, ids: list, families: list[str]) -> pd.DataFrame:
    return v6.protocol_dataframe(rows, ids, data.demographics, families)


def protocol_fn(data: Data, families: list[str] | None = None,
                columns: list[str] | None = None, extra: Callable | None = None) -> FeatureFn:
    """Frozen PCMVF protocol representation; optional column subset / extra columns."""
    fams = families or MENU

    def f(rows: pd.DataFrame, ids: list) -> np.ndarray:
        pdf = protocol_df(data, rows, ids, fams)
        if extra is not None:
            pdf = pd.concat([pdf, extra(rows, ids)], axis=1)
        if columns is not None:
            pdf = pdf[[c for c in columns if c in pdf]]
        return pdf.to_numpy(dtype=float)
    return f


def acquisition_frame(data: Data, rows: pd.DataFrame, ids: list,
                      include: tuple[str, ...] = ("country", "device", "gain", "duration")) -> pd.DataFrame:
    """Acquisition variables not used in the submitted paper, computed on the selected rows."""
    pids = pd.Index([str(i) for i in ids], name="participant_id")
    out = pd.DataFrame(index=pids)
    meta = data.participant_meta.reindex(pids)
    if "country" in include and "country_canada" in meta:
        out["acq_country_canada"] = meta["country_canada"].to_numpy()
    g = rows.groupby("participant_id")
    if "device" in include and "rec_microphone" in rows:
        for mic in ["iPad Microphone", "Headset Microphone", "USB-C to 3.5mm Headphone Jack Adapter"]:
            out[f"acq_mic__{_norm_name(mic)}"] = g["rec_microphone"].apply(
                lambda s, mic=mic: float((s == mic).mean())).reindex(pids).to_numpy()
    if "gain" in include and "rec_input_gain" in rows:
        out["acq_input_gain_mean"] = g["rec_input_gain"].mean().reindex(pids).to_numpy()
    if "duration" in include and "rec_duration" in rows:
        out["acq_rec_duration_mean"] = g["rec_duration"].mean().reindex(pids).to_numpy()
        out["acq_rec_duration_sum"] = g["rec_duration"].sum().reindex(pids).to_numpy()
    return out


def concat_fns(*fns: FeatureFn) -> FeatureFn:
    def f(rows: pd.DataFrame, ids: list) -> np.ndarray:
        return np.concatenate([fn(rows, ids) for fn in fns], axis=1)
    return f


# ---------------------------------------------------------------------------
# Cohort construction (identical to run_final_equal_evidence)
# ---------------------------------------------------------------------------

@dataclass
class Cohort:
    disease: str
    matched_pairs: list
    final_pairs: list
    budget: int
    tier: str
    source: pd.DataFrame
    ids: list
    y: np.ndarray
    folds: np.ndarray
    menu: list
    caliper_years: float


def standardized_cohort(data: Data, disease: str, args: argparse.Namespace,
                        menu: list[str] | None = None,
                        manifest: pd.DataFrame | None = None,
                        matched_pairs: list | None = None) -> Cohort:
    menu = list(menu or MENU)
    man = data.manifest if manifest is None else manifest
    caliper_years = np.nan
    if matched_pairs is None:
        case_ids, control_ids, caliper_years = pb.match_demographics(
            disease, data.cohort_by_pid, data.demographics, args.age_caliper_sd)
        matched_pairs = v6.pair_list(case_ids, control_ids)
    complete = v6.complete_menu_pairs(man, matched_pairs, menu, budget=1)
    if len(complete) < args.exploratory_min_pairs:
        raise RuntimeError(f"{disease}: only {len(complete)} complete-menu pairs")
    cids = v6.flatten_pairs(complete)
    src = man[man["participant_id"].isin(cids) & man["task_family"].isin(menu) & man["_has_all_reps"]].copy()
    budget, final_pairs, tier = v7.choose_budget_with_support(src, complete, menu, args)
    ids = v6.flatten_pairs(final_pairs)
    y = pb.make_labels(ids, disease, data.cohort_by_pid)
    folds = v7.pair_aware_folds(final_pairs, ids, args.seed)
    return Cohort(disease, matched_pairs, final_pairs, budget, tier, src, ids, y, folds, menu, caliper_years)


def standardized_predictions(data: Data, cohort: Cohort, args: argparse.Namespace,
                             feature_fns: dict[str, FeatureFn],
                             models: dict[str, ProbeFn] | None = None,
                             sampler: Callable | None = None,
                             keep_seed_preds: bool = False) -> dict[str, pd.DataFrame]:
    """
    Seed-averaged OOF predictions for every (feature set x model).

    Keys are "<feature>" when a single model is used, else "<feature>|<model>".
    Same subsampled rows are shared across all feature sets within a seed.
    """
    models = models or {"pls_logistic": pb.fixed_probe_oof}
    single = len(models) == 1
    per: dict[str, list] = {}
    for s in range(args.seeds):
        if sampler is None:
            sampled = pb.sample_taskwise(cohort.source, cohort.ids, cohort.menu, cohort.budget, seed=args.seed + s)
        else:
            sampled = sampler(cohort, args.seed + s)
        for fname, fn in feature_fns.items():
            X = fn(sampled, cohort.ids)
            for mname, mfn in models.items():
                key = fname if single else f"{fname}|{mname}"
                pred = mfn(X, cohort.y, cohort.ids, cohort.folds)
                pred["subsample_seed"] = s
                per.setdefault(key, []).append(pred)
    out = {k: pb.average_seed_predictions(v) for k, v in per.items()}
    if keep_seed_preds:
        out["_seed_preds"] = per  # type: ignore[assignment]
    return out


def raw_cohort_rows(data: Data, disease: str) -> tuple[pd.DataFrame, list[str]]:
    rows = data.manifest[data.manifest["cohort"].isin([disease, CONTROL])].copy()
    demo_ids = set(data.demographics["participant_id"].astype(str))
    ids = sorted(p for p in rows["participant_id"].unique().astype(str) if p in demo_ids)
    return rows, ids


def raw_predictions(data: Data, disease: str, args: argparse.Namespace,
                    feature_fns: dict[str, Callable] | None = None,
                    models: dict[str, ProbeFn] | None = None,
                    reps: list[str] | None = None) -> tuple[dict[str, pd.DataFrame], list[str], np.ndarray]:
    """
    Full-cohort predictions (paper Section 4.1 / stage raw_same_probe_original_cv).
    feature_fns here receive (rows, ids, families) and must return a DataFrame/array
    aligned to ids; default = available-speech matrix and full protocol.
    """
    rows, cand = raw_cohort_rows(data, disease)
    if reps is None:  # frozen behaviour (all 8 released representations)
        _, ids = v6.build_speech_matrix_available(rows, cand, data.lookups)
    else:
        _, ids = speech_matrix(rows, cand, data.lookups, reps, strict=False)
    fams = sorted(rows["task_family"].dropna().unique().tolist())
    y = pb.make_labels(ids, disease, data.cohort_by_pid)
    folds = pb.make_folds(ids, y, data.fold_map, args.seed)
    if feature_fns is None:
        feature_fns = {
            "speech": (lambda r, i, f: v6.build_speech_matrix_available(r, i, data.lookups)[0]) if reps is None
            else (lambda r, i, f: speech_matrix(r, i, data.lookups, reps, strict=False)[0]),
            "protocol": lambda r, i, f: protocol_df(data, r[r["participant_id"].isin(i)], i, f).to_numpy(float),
        }
    models = models or {"pls_logistic": pb.fixed_probe_oof}
    single = len(models) == 1
    out = {}
    for fname, fn in feature_fns.items():
        X = np.asarray(fn(rows, ids, fams), dtype=float)
        for mname, mfn in models.items():
            out[fname if single else f"{fname}|{mname}"] = mfn(X, y, ids, folds)
    return out, ids, folds


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _merge_preds(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = None
    for k, p in preds.items():
        q = p[["participant_id", "y_true", "y_score"]].rename(columns={"y_score": k})
        q["participant_id"] = q["participant_id"].astype(str)
        df = q if df is None else df.merge(q.drop(columns="y_true"), on="participant_id", validate="one_to_one")
    return df


def pair_bootstrap(preds: dict[str, pd.DataFrame], pairs: list | None, n_boot: int, seed: int,
                   deltas: list[tuple[str, str]] | None = None,
                   return_draws: bool = False) -> dict[str, Any]:
    """
    Generalisation of paired_match_auc_ci to any number of models.
    Resamples matched pairs (pairs given) or cases/controls separately (pairs=None).
    """
    df = _merge_preds(preds)
    keys = list(preds)
    y = df["y_true"].to_numpy(int)
    S = df[keys].to_numpy(float)
    rng = np.random.default_rng(seed)
    if pairs is not None:
        pos = {p: i for i, p in enumerate(df["participant_id"])}
        pidx = np.array([[pos[str(a)], pos[str(b)]] for a, b in pairs])
        draw = lambda: pidx[rng.integers(0, len(pidx), len(pidx))].ravel()  # noqa: E731
    else:
        ci, hi = np.where(y == 1)[0], np.where(y == 0)[0]
        draw = lambda: np.concatenate([rng.choice(ci, len(ci)), rng.choice(hi, len(hi))])  # noqa: E731
    point = {k: float(roc_auc_score(y, S[:, j])) for j, k in enumerate(keys)}
    boots = np.empty((n_boot, len(keys)))
    for b in range(n_boot):
        idx = draw()
        boots[b] = [roc_auc_score(y[idx], S[idx, j]) for j in range(len(keys))]
    res: dict[str, Any] = {"n": int(len(df)), "n_pairs": len(pairs) if pairs else np.nan}
    for j, k in enumerate(keys):
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5])
        res[f"{k}_auc"], res[f"{k}_ci_low"], res[f"{k}_ci_high"] = point[k], float(lo), float(hi)
    for a, b in deltas or []:
        d = boots[:, keys.index(a)] - boots[:, keys.index(b)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        res[f"delta[{a}-{b}]"] = point[a] - point[b]
        res[f"delta[{a}-{b}]_ci_low"], res[f"delta[{a}-{b}]_ci_high"] = float(lo), float(hi)
        res[f"delta[{a}-{b}]_boot_p"] = float(min(1.0, p))
    if return_draws:
        res["_draws"] = pd.DataFrame(boots, columns=keys)
    return res


def _structural_components(y: np.ndarray, s: np.ndarray):
    pos, neg = s[y == 1], s[y == 0]
    cmp = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    return cmp.mean(1), cmp.mean(0), cmp.mean()


def delong_auc_ci(y, s, alpha: float = 0.05) -> tuple[float, float, float, float]:
    """DeLong (1988) AUC, SE and Wald CI."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    v10, v01, auc = _structural_components(y, s)
    se = float(np.sqrt(np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01)))
    z = sps.norm.ppf(1 - alpha / 2)
    return float(auc), se, float(auc - z * se), float(auc + z * se)


def delong_paired_test(y, s1, s2) -> dict[str, float]:
    """DeLong test for two correlated AUCs on the same participants."""
    y = np.asarray(y, int)
    a10, a01, A = _structural_components(y, np.asarray(s1, float))
    b10, b01, B = _structural_components(y, np.asarray(s2, float))
    s10 = np.cov(np.vstack([a10, b10]))
    s01 = np.cov(np.vstack([a01, b01]))
    cov = s10 / len(a10) + s01 / len(a01)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (A - B) / np.sqrt(var) if var > 0 else 0.0
    return {"auc1": A, "auc2": B, "delta": A - B, "se_delta": float(np.sqrt(max(var, 0))),
            "z": float(z), "p": float(2 * sps.norm.sf(abs(z)))}


def tost_auc(draws: np.ndarray, point: float, margin: float, alpha: float = 0.05) -> dict[str, Any]:
    """
    Bootstrap TOST for H0: |AUC - 0.5| >= margin.
    Equivalence is declared when the (1-2*alpha) percentile interval lies inside
    [0.5 - margin, 0.5 + margin].
    """
    lo, hi = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
    p_lower = float(np.mean(draws <= 0.5 - margin))
    p_upper = float(np.mean(draws >= 0.5 + margin))
    return {
        "margin": margin, "auc": point, "ci90_low": float(lo), "ci90_high": float(hi),
        "p_tost": max(p_lower, p_upper),
        "equivalent": bool(lo > 0.5 - margin and hi < 0.5 + margin),
    }


def smallest_equivalence_margin(draws: np.ndarray, alpha: float = 0.05) -> float:
    lo, hi = np.percentile(draws, [100 * alpha, 100 * (1 - alpha)])
    return float(max(0.5 - lo, hi - 0.5))


def pcmvf_label(speech_lo: float, protocol_lo: float, protocol_hi: float, tier: str) -> str:
    return v7.final_status(pd.Series({
        "support_tier": tier, "speech_ci_low": speech_lo,
        "protocol_ci_low": protocol_lo, "protocol_ci_high": protocol_hi,
    }))


def smd(a: np.ndarray, b: np.ndarray) -> float:
    return float(v6.smd(np.asarray(a, float), np.asarray(b, float)))


# ---------------------------------------------------------------------------
# Alternative models with the fixed_probe_oof interface
# ---------------------------------------------------------------------------

def _sk_oof(make, tune_grid: dict | None = None, preprocess: str = "impute_scale") -> ProbeFn:
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fn(X, y, ids, folds):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        rows = []
        for f in sorted(np.unique(folds)):
            tr, te = folds != f, folds == f
            keep = np.nanstd(X[tr], axis=0) > 1e-12
            if not keep.any():
                score = np.full(te.sum(), 0.5)
            else:
                steps = [SimpleImputer(strategy="median")]
                if preprocess == "impute_scale":
                    steps.append(StandardScaler())
                pipe = make_pipeline(*steps, make())
                if tune_grid:
                    inner = StratifiedKFold(n_splits=min(3, int(np.bincount(y[tr]).min())),
                                            shuffle=True, random_state=2026)
                    grid = {f"{pipe.steps[-1][0]}__{k}": v for k, v in tune_grid.items()}
                    pipe = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1)
                pipe.fit(X[tr][:, keep], y[tr])
                score = pipe.predict_proba(X[te][:, keep])[:, 1]
            for pid, yt, ys in zip(np.asarray(ids)[te], y[te], score):
                rows.append({"participant_id": str(pid), "fold": int(f), "y_true": int(yt), "y_score": float(ys)})
        return pd.DataFrame(rows)
    return fn


def model_oof(name: str) -> ProbeFn:
    """
    pls_logistic : frozen primary probe
    logistic     : L2 logistic without PLS, C tuned in inner CV
    rf           : random forest (500 trees, depth/leaf tuned)
    hgb          : histogram gradient boosting (tuned)
    xgb          : XGBoost if installed (tuned)
    mlp          : MLP (64, 32) as in Table S1
    svm_rbf      : RBF-SVM (C, gamma tuned)
    knn          : k-nearest neighbours (k tuned)
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    if name == "pls_logistic":
        return pb.fixed_probe_oof
    if name == "logistic":
        return _sk_oof(lambda: LogisticRegression(class_weight="balanced", max_iter=5000),
                       {"C": [0.01, 0.1, 1, 10]})
    if name == "rf":
        return _sk_oof(lambda: RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                      random_state=2026, n_jobs=4),
                       {"max_depth": [3, None], "min_samples_leaf": [1, 5]}, preprocess="impute")
    if name == "hgb":
        return _sk_oof(lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=2026),
                       {"max_depth": [2, 4], "learning_rate": [0.05, 0.2]}, preprocess="impute")
    if name == "xgb":
        from xgboost import XGBClassifier
        return _sk_oof(lambda: XGBClassifier(n_estimators=300, eval_metric="logloss", random_state=2026,
                                             n_jobs=4, verbosity=0),
                       {"max_depth": [2, 4], "learning_rate": [0.05, 0.2]}, preprocess="impute")
    if name == "mlp":
        return _sk_oof(lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3,
                                             max_iter=500, random_state=2026))
    if name == "svm_rbf":
        return _sk_oof(lambda: SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=2026),
                       {"C": [0.1, 1, 10], "gamma": ["scale", 0.01]})
    if name == "knn":
        return _sk_oof(lambda: KNeighborsClassifier(), {"n_neighbors": [3, 7, 15]})
    raise KeyError(name)


ALL_MODELS = ["pls_logistic", "logistic", "rf", "hgb", "xgb", "mlp", "svm_rbf", "knn"]


def available_models(names: list[str]) -> dict[str, ProbeFn]:
    out = {}
    for n in names:
        try:
            out[n] = model_oof(n)
        except ImportError as exc:
            print(f"[models] skipping {n}: {exc}")
    return out


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save(df: pd.DataFrame | list, out_dir: Path, name: str) -> Path:
    df = pd.DataFrame(df) if not isinstance(df, pd.DataFrame) else df
    path = Path(out_dir) / name
    df.to_csv(path, index=False)
    print(f"[saved] {path}  ({len(df)} rows)")
    return path


def save_preds(preds: dict[str, pd.DataFrame], out_dir: Path, name: str, **tags: Any) -> None:
    frames = []
    for k, p in preds.items():
        if k.startswith("_"):
            continue
        q = p.copy()
        q["model"] = k
        for t, v in tags.items():
            q[t] = v
        frames.append(q)
    if frames:
        path = Path(out_dir) / name
        pd.concat(frames).to_csv(path, index=False, mode="a", header=not path.exists())


def write_json(obj: Any, out_dir: Path, name: str) -> None:
    (Path(out_dir) / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def run_each(diseases: list[str], fn: Callable[[str], Any]) -> list:
    """Run fn per disease, logging and skipping failures instead of aborting."""
    out = []
    for d in diseases:
        print(f"\n{'=' * 80}\n{d}\n{'=' * 80}", flush=True)
        try:
            r = fn(d)
            if r is not None:
                out.extend(r if isinstance(r, list) else [r])
        except Exception as exc:  # noqa: BLE001
            print(f"[FAILED] {d}: {exc!r}", file=sys.stderr)
            out.append({"disease": d, "error": repr(exc)})
    return out

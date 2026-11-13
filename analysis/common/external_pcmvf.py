#!/usr/bin/env python3
"""
Generic PCMVF runner for ANY recording-level dataset (independent validation,
other modalities). Reuses the frozen probe, pair-aware folds, equal task-wise
evidence and pair bootstrap from pcmvf_common.

Input manifest CSV (one row per recording):
    participant_id, session_id, task, label   (required; label 1 = case, 0 = control)
    task_family                               (optional; default = task)
    age, sex                                  (optional; sex in {f,m}); without them,
                                              pairs are formed at random (within sex if given)
Input features: one or more parquet files written by audio_features.py (or any
parquet with participant_id, session_id, task_norm, vector). For non-audio
modalities (e.g. mPower tapping/walking), write the same parquet schema.

Usage
  python external_pcmvf.py --manifest m.csv --features f1.parquet [f2.parquet ...] \
      --out-dir results/external/<dataset>

Outputs: external_pcmvf_summary.csv (raw + standardized speech/protocol AUC, label),
         external_protocol_ablation.csv, external_menu.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcmvf_common as C  # noqa: E402
from audio_features import norm_task, to_lookup  # noqa: E402

CASE = "Case"


def build_external_data(manifest_csv: str, feature_files: list[str]) -> tuple[C.Data, list[str]]:
    m = pd.read_csv(manifest_csv, dtype=str)
    m["participant_id"] = m["participant_id"].astype(str)
    m["session_id"] = m["session_id"].astype(str)
    m["raw_task"] = m["task"]
    m["task_norm"] = m["task"].map(norm_task)
    m["task_family"] = (m["task_family"] if "task_family" in m else m["task"]).map(norm_task)
    m["cohort"] = np.where(m["label"].astype(str).isin(["1", "case", "Case", "PD", "pd"]), CASE, C.CONTROL)
    lookups, reps = {}, []
    for f in feature_files:
        name = Path(f).stem
        lookups[name] = to_lookup(pd.read_parquet(f))
        reps.append(name)
    keys = list(zip(m["participant_id"], m["session_id"], m["task_norm"]))
    m["_has_all_reps"] = [all(k in lookups[r] for r in reps) for k in keys]
    m["_row_id"] = np.arange(len(m))
    cohort = m.drop_duplicates("participant_id").set_index("participant_id")["cohort"]
    p = m.drop_duplicates("participant_id").set_index("participant_id")
    demo = pd.DataFrame({
        "participant_id": p.index.astype(str),
        "age": pd.to_numeric(p["age"], errors="coerce").to_numpy() if "age" in p else np.nan,
        "sex": p["sex"].str.lower().str[0].to_numpy() if "sex" in p else "u",
    })
    return C.Data(m, cohort, demo, lookups, None, Path(".")), reps


def make_pairs(data: C.Data, args, seed: int) -> list:
    demo = data.demographics
    if demo["age"].notna().mean() > 0.9 and demo["sex"].isin(["f", "m"]).mean() > 0.9:
        c, h, _ = C.pb.match_demographics(CASE, data.cohort_by_pid, demo, args.age_caliper_sd)
        return list(zip(c, h))
    # No demographics: random 1:1 pairing (within sex where known) only to define pair-aware folds.
    rng = np.random.default_rng(seed)
    pairs = []
    for _, g in demo.assign(cohort=demo["participant_id"].map(data.cohort_by_pid)).groupby("sex"):
        cs = rng.permutation(g.loc[g["cohort"] == CASE, "participant_id"].to_numpy())
        hs = rng.permutation(g.loc[g["cohort"] == C.CONTROL, "participant_id"].to_numpy())
        pairs += list(zip(cs[: len(hs)], hs[: len(cs)]))
    return pairs


def shared_menu(data: C.Data, pairs: list, coverage: float) -> list[str]:
    m = data.manifest[data.manifest["_has_all_reps"]]
    menu = []
    for fam, g in m.groupby("task_family"):
        have = set(g["participant_id"])
        if all(np.mean([x in have for x in grp]) >= coverage for grp in ([c for c, _ in pairs], [h for _, h in pairs])):
            menu.append(fam)
    return sorted(menu)


def run(data: C.Data, reps: list[str], args, coverage: float = 0.9) -> tuple[list, list, dict]:
    has_demo = data.demographics["age"].notna().mean() > 0.9
    # raw stage
    raw, ids, _ = C.raw_predictions(data, CASE, args, reps=reps)
    braw = C.pair_bootstrap(raw, None, args.bootstrap, args.seed + 303)
    summary = [{"raw_n": braw["n"], "raw_speech_auc": braw["speech_auc"], "raw_speech_ci_low": braw["speech_ci_low"],
                "raw_speech_ci_high": braw["speech_ci_high"], "raw_protocol_auc": braw["protocol_auc"],
                "raw_protocol_ci_low": braw["protocol_ci_low"], "raw_protocol_ci_high": braw["protocol_ci_high"]}]
    menu = []
    try:  # standardization can legitimately fail (e.g. no age overlap); Detect results are still saved
        pairs = make_pairs(data, args, args.seed)
        menu = shared_menu(data, pairs, coverage)
        co = C.standardized_cohort(data, CASE, args, menu=menu, matched_pairs=pairs)
        std = C.standardized_predictions(data, co, args, {"speech": C.speech_fn(data, reps),
                                                          "protocol": C.protocol_fn(data, families=menu)})
        bstd = C.pair_bootstrap(std, co.final_pairs, args.bootstrap, args.seed + 5001, deltas=[("speech", "protocol")])
        summary = [{
            "raw_n": braw["n"], "raw_speech_auc": braw["speech_auc"], "raw_speech_ci_low": braw["speech_ci_low"],
            "raw_speech_ci_high": braw["speech_ci_high"], "raw_protocol_auc": braw["protocol_auc"],
            "raw_protocol_ci_low": braw["protocol_ci_low"], "raw_protocol_ci_high": braw["protocol_ci_high"],
            "pairs": len(co.final_pairs), "budget": co.budget, "menu": ";".join(menu),
            "std_speech_auc": bstd["speech_auc"], "std_speech_ci_low": bstd["speech_ci_low"],
            "std_speech_ci_high": bstd["speech_ci_high"], "residual_protocol_auc": bstd["protocol_auc"],
            "residual_protocol_ci_low": bstd["protocol_ci_low"], "residual_protocol_ci_high": bstd["protocol_ci_high"],
            "demographics_matched": has_demo,
            "pcmvf_label": C.pcmvf_label(bstd["speech_ci_low"], bstd["protocol_ci_low"], bstd["protocol_ci_high"], co.tier),
        }]
    except Exception as exc:  # noqa: BLE001
        summary[0].update({"pairs": 0, "pcmvf_label": "not_standardizable", "error": repr(exc)})
    # raw protocol ablation (what drives Detect?)
    rows, cand = C.raw_cohort_rows(data, CASE)
    fams = sorted(rows["task_family"].unique())
    pdf = C.protocol_df(data, rows[rows["participant_id"].isin(ids)], ids, fams)
    # Skip components with no usable variation (e.g. age/sex absent in MDVR-KCL).
    comps = {k: v for k, v in C.v6.protocol_components(pdf).items()
             if v and (pdf[v].nunique(dropna=True) > 1).any()}
    abl_preds, _, _ = C.raw_predictions(
        data, CASE, args, {k: (lambda cols: lambda r, i, f: pdf.loc[i, cols].to_numpy(float))(v) for k, v in comps.items()},
        reps=reps)
    babl = C.pair_bootstrap(abl_preds, None, args.bootstrap, args.seed + 404)
    abl = [{"protocol_model": k, "auc": babl[f"{k}_auc"], "ci_low": babl[f"{k}_ci_low"],
            "ci_high": babl[f"{k}_ci_high"]} for k in comps]
    return summary, abl, {"menu": menu, "final_pairs": summary[0].get("pairs", 0)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--min-pairs", type=int, default=20)
    p.add_argument("--coverage", type=float, default=0.9)
    ns = p.parse_args()
    out = Path(ns.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    args = C.default_args(seeds=ns.seeds, bootstrap=ns.bootstrap, primary_min_pairs=ns.min_pairs,
                          exploratory_min_pairs=min(8, ns.min_pairs))
    data, reps = build_external_data(ns.manifest, ns.features)
    summary, abl, meta = run(data, reps, args, ns.coverage)
    C.save(summary, out, "external_pcmvf_summary.csv")
    C.save(abl, out, "external_protocol_ablation.csv")
    C.write_json(meta, out, "external_menu.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

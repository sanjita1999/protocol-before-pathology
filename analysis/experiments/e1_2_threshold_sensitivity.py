#!/usr/bin/env python3
"""
Question addressed: How are thresholds such as the 0.25-SD age caliper justified?

Sweeps every analyst-chosen threshold in PCMVF one at a time, holding the
others at their frozen values, and records pairs retained, residual imbalance
(|age SMD|, |session SMD|), standardized speech / residual protocol AUC and the
PCMVF label:

  age caliper (SD)       0.05 0.10 0.15 0.20 0.25* 0.35 0.50 0.75 1.00 inf(sex-only)
  per-task budget set    {5,3,2,1}*  {1}  {2,1}  {3,2,1}
  retention fraction     0.50 0.60 0.70* 0.80 0.90
  primary min pairs      10 15 20* 25 30   (label tier only; no refit)

Rationale: 0.2-0.25 SD of the logit/propensity is
the standard caliper recommended by Austin (2011, Pharm Stat 10:150-161) and
Rosenbaum & Rubin (1985); the sweep shows which conclusions depend on it.

Outputs (results/e1_2_threshold_sensitivity/):
    caliper_sweep.csv, budget_sweep.csv, retention_sweep.csv, min_pairs_tier.csv
    label_stability_summary.csv   share of settings giving the primary label
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
CALIPERS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00, 1e6]
BUDGETS = {"5-3-2-1 (primary)": [5, 3, 2, 1], "1": [1], "2-1": [2, 1], "3-2-1": [3, 2, 1]}
RETENTIONS = [0.50, 0.60, 0.70, 0.80, 0.90]


def evaluate(data, d, args, setting: dict) -> dict:
    co = C.standardized_cohort(data, d, args)
    preds = C.standardized_predictions(data, co, args, {"speech": C.speech_fn(data), "protocol": C.protocol_fn(data)})
    b = C.pair_bootstrap(preds, co.final_pairs, args.bootstrap, args.seed + 5001)
    ids = co.ids
    demo = data.demographics.set_index("participant_id").reindex(ids)
    sess = co.source.groupby("participant_id")["session_id"].nunique().reindex(ids)
    case = co.y == 1
    return {
        "disease": d, **setting,
        "caliper_years": co.caliper_years,
        "matched_pairs": len(co.matched_pairs), "final_pairs": len(co.final_pairs),
        "budget": co.budget, "tier": co.tier,
        "age_smd": abs(C.smd(demo["age"].to_numpy()[case], demo["age"].to_numpy()[~case])),
        "session_smd": abs(C.smd(sess.to_numpy()[case], sess.to_numpy()[~case])),
        "speech_auc": b["speech_auc"], "speech_ci_low": b["speech_ci_low"], "speech_ci_high": b["speech_ci_high"],
        "protocol_auc": b["protocol_auc"], "protocol_ci_low": b["protocol_ci_low"],
        "protocol_ci_high": b["protocol_ci_high"],
        "pcmvf_label": C.pcmvf_label(b["speech_ci_low"], b["protocol_ci_low"], b["protocol_ci_high"], co.tier),
    }


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e1_2_threshold_sensitivity"))
    p.add_argument("--sweeps", nargs="+", default=["caliper", "budget", "retention", "min_pairs"])
    ns = C.finalize_cli(p.parse_args())
    base = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root, attach_metadata=False)
    out = {}

    if "caliper" in ns.sweeps:
        out["caliper"] = C.run_each(ns.diseases, lambda d: [
            _safe(evaluate, data, d, _with(base, age_caliper_sd=c),
                  {"sweep": "age_caliper_sd", "value": "inf" if c > 100 else c})
            for c in CALIPERS])
    if "budget" in ns.sweeps:
        out["budget"] = C.run_each(ns.diseases, lambda d: [
            _safe(evaluate, data, d, _with(base, per_task_budgets=v), {"sweep": "per_task_budgets", "value": k})
            for k, v in BUDGETS.items()])
    if "retention" in ns.sweeps:
        out["retention"] = C.run_each(ns.diseases, lambda d: [
            _safe(evaluate, data, d, _with(base, feature_retention=r), {"sweep": "feature_retention", "value": r})
            for r in RETENTIONS])
    for k, rows in out.items():
        C.save(rows, ns.out_dir, f"{k}_sweep.csv")

    if "min_pairs" in ns.sweeps:
        ref = pd.read_csv(C.V8_OUT / "publication_summary.csv")
        rows = []
        for t in [10, 15, 20, 25, 30]:
            for _, r in ref.iterrows():
                tier = "primary" if r["final_pairs"] >= t else ("exploratory" if r["final_pairs"] >= 8 else "insufficient")
                rows.append({"disease": r["disease"], "min_pairs": t, "final_pairs": r["final_pairs"], "tier": tier})
        C.save(rows, ns.out_dir, "min_pairs_tier.csv")

    allrows = pd.DataFrame([r for v in out.values() for r in v if "pcmvf_label" in r])
    if not allrows.empty:
        prim = allrows[
            ((allrows["sweep"] == "age_caliper_sd") & (allrows["value"] == 0.25))
        ].set_index("disease")["pcmvf_label"]
        allrows["same_as_primary"] = allrows["pcmvf_label"].eq(allrows["disease"].map(prim))
        C.save(allrows.groupby(["disease", "sweep"]).agg(
            n_settings=("pcmvf_label", "size"),
            share_same_label=("same_as_primary", "mean"),
            labels_seen=("pcmvf_label", lambda s: "; ".join(sorted(set(s)))),
        ).reset_index(), ns.out_dir, "label_stability_summary.csv")
    return 0


def _with(args, **kw):
    a = C.default_args(**vars(args))
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as exc:  # noqa: BLE001
        return {"disease": a[1], **a[3], "error": repr(exc)}


if __name__ == "__main__":
    raise SystemExit(main())

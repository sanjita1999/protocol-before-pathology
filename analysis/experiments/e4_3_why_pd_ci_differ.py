#!/usr/bin/env python3
"""
Question addressed: Why might Parkinson's and Cognitive Impairment differ from the
other diseases? Are there features of these diseases that explain it, or do they
manifest uniquely in audio?

Cohort-level profile linking each disease's protocol signal to how it was
recruited and assessed:

 A. Profile table per cohort vs controls: share Canada (site proxy), neurology
    module exposure, mean sessions, mean observations, age, sex, device mix,
    plus |SMD| for each, and the paper's raw protocol AUC / standardized speech
    and protocol AUC (from v8 outputs).
 B. Raw protocol ablation for EVERY cohort (the paper only shows PD/CI):
    completion, counts, sessions, age+sex, country, full protocol.
 C. Across the 9 cohorts: Spearman rho between raw protocol AUC and each
    profile SMD (n = 9; descriptive).
 D. Is the remaining speech signal disease-typical? Standardized per-family
    speech AUC (one family at a time, same final pairs), so PD/CI can be compared
    with the expected clinical pattern (e.g. PD: rapid articulation / reading;
    CI: narrative-memory / spontaneous speech).

Outputs (results/e4_3_why_pd_ci_differ/):
    cohort_profile.csv, raw_protocol_ablation_all_cohorts.csv,
    profile_vs_protocol_auc_corr.csv, standardized_speech_by_family.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__), str(C.RESULTS / "e4_3_why_pd_ci_differ"))
    p.add_argument("--parts", nargs="+", default=["A", "B", "C", "D"])
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root)
    m, meta = data.manifest, data.participant_meta.copy()
    meta["n_sessions"] = m.groupby("participant_id")["session_id"].nunique()
    meta["n_observations"] = m.groupby("participant_id").size()
    meta["neurology_module"] = m.assign(_n=m["task_module"].eq("neurology")).groupby("participant_id")["_n"].max().astype(float)
    demo = data.demographics.set_index("participant_id")
    meta["age"] = demo["age"].reindex(meta.index)
    meta["sex_m"] = demo["sex"].reindex(meta.index).eq("m").astype(float)
    cols = ["country_canada", "neurology_module", "n_sessions", "n_observations", "age", "sex_m",
            "share_ipad_mic", "mean_input_gain", "mean_rec_duration"]
    ref = pd.read_csv(C.V8_OUT / "publication_summary.csv").set_index("disease")

    prof = []
    ctrl = meta[meta["cohort"] == C.CONTROL]
    for d in ns.diseases:
        case = meta[meta["cohort"] == d]
        r = {"disease": d, "n": len(case)}
        for c in cols:
            if c in meta:
                r[f"{c}_case"], r[f"{c}_control"] = case[c].mean(), ctrl[c].mean()
                r[f"{c}_abs_smd"] = abs(C.smd(case[c].to_numpy(float), ctrl[c].to_numpy(float)))
        for k in ["raw_protocol_auc", "raw_same_probe_speech_auc", "balanced_speech_auc", "balanced_protocol_auc",
                  "validity_status"]:
            r[k] = ref.loc[d, k] if d in ref.index else np.nan
        prof.append(r)
    prof = pd.DataFrame(prof)
    if "A" in ns.parts:
        C.save(prof, ns.out_dir, "cohort_profile.csv")

    if "B" in ns.parts:
        def abl(d):
            rows, _ = C.raw_cohort_rows(data, d)
            fams = sorted(rows["task_family"].dropna().unique())
            def fn(kind):
                def f(r, i, fam):
                    P = C.protocol_df(data, r[r["participant_id"].isin(i)], i, fams)
                    comps = C.v6.protocol_components(P)
                    if kind == "country":
                        return meta.reindex(i)[["country_canada"]].to_numpy(float)
                    if kind == "full_plus_country":
                        return np.hstack([P.to_numpy(float), meta.reindex(i)[["country_canada"]].to_numpy(float)])
                    return P[comps[kind]].to_numpy(float)
                return f
            kinds = ["completion_only", "counts_only", "session_only", "age_sex", "country", "full_protocol",
                     "full_plus_country"]
            preds, _, _ = C.raw_predictions(data, d, args, {k: fn(k) for k in kinds})
            b = C.pair_bootstrap(preds, None, args.bootstrap, args.seed + 404)
            return [{"disease": d, "protocol_model": k, "auc": b[f"{k}_auc"], "ci_low": b[f"{k}_ci_low"],
                     "ci_high": b[f"{k}_ci_high"]} for k in kinds]
        C.save(C.run_each(ns.diseases, abl), ns.out_dir, "raw_protocol_ablation_all_cohorts.csv")

    if "C" in ns.parts:
        corr = []
        for c in [x for x in prof.columns if x.endswith("_abs_smd")]:
            ok = prof[[c, "raw_protocol_auc"]].dropna()
            if len(ok) >= 5:
                r = spearmanr(ok[c], ok["raw_protocol_auc"])
                corr.append({"profile_smd": c, "n_cohorts": len(ok), "rho_with_raw_protocol_auc": r.correlation,
                             "p": r.pvalue})
        C.save(corr, ns.out_dir, "profile_vs_protocol_auc_corr.csv")

    if "D" in ns.parts:
        def fam(d):
            co = C.standardized_cohort(data, d, args)
            out = []
            for f in co.menu:
                sampler = (lambda f: lambda cohort, seed: C.pb.sample_taskwise(
                    cohort.source, cohort.ids, [f], cohort.budget, seed))(f)
                pr = C.standardized_predictions(data, co, args, {"speech": C.speech_fn(data)}, sampler=sampler)
                b = C.pair_bootstrap(pr, co.final_pairs, args.bootstrap, args.seed + 5001)
                out.append({"disease": d, "task_family": f, "pairs": len(co.final_pairs), "speech_auc": b["speech_auc"],
                            "ci_low": b["speech_ci_low"], "ci_high": b["speech_ci_high"]})
            return out
        C.save(C.run_each(ns.diseases, fam), ns.out_dir, "standardized_speech_by_family.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Question addressed: Site/country identifiers unavailable; Shukla et al. (2026) report
complete country-label confounding in Bridge2AI PD/dementia subsets.
"Was no collection metadata available besides age/sex? Equipment,
examiner ..."
Also R2/limitations: "equal recording counts do not guarantee equal duration".

IMPORTANT FINDING (pre-check, 2026-10-06): Bridge2AI v3.0.0 DOES release
participant-level `country` (phenotype/demographics/demographics.tsv) and
recording-level `recording_microphone`, `recording_input_gain`,
`recording_duration` (phenotype/task/recording.tsv). In the 629-participant
cohort all 113 controls are USA, while 53/57 Cognitive Impairment and 45/83
Parkinson's participants are Canada. The submitted paper states that country
was unavailable; this script quantifies what that means for every claim.

Analyses
  A. Balance table: country, device mix, input gain, recording duration, session
     duration per cohort vs controls (SMD), in the full cohort and in the final
     PCMVF pairs.
  B. Extended Detect / residual test: AUC of acquisition-only, frozen protocol, and
     protocol + acquisition features, raw and after PCMVF standardization.
     (Recording duration is partly clinical for maximum-phonation tasks; reported
     with and without duration.)
  C. Country-restricted PCMVF: rerun the full standardization using USA
     participants only (removes the country confound because all controls are USA).
     Feasible for cohorts with enough USA cases (PD: ~37); CI is reported as
     not estimable.
  D. Device-stratified matching: exact match on sex AND primary microphone, age
     caliper unchanged, then the full PCMVF pipeline.

Outputs (results/e2_2_site_country_device/):
    acquisition_balance.csv, extended_protocol_auc.csv,
    usa_only_pcmvf.csv, device_matched_pcmvf.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
META_COLS = ["country_canada", "share_ipad_mic", "mean_input_gain", "mean_rec_duration", "session_duration_mean"]


def balance(data, d, ids_case, ids_ctrl, stage):
    m = data.participant_meta
    out = {"disease": d, "stage": stage, "n_case": len(ids_case), "n_control": len(ids_ctrl)}
    for c in META_COLS:
        if c not in m:
            continue
        a, b = m.loc[ids_case, c].to_numpy(float), m.loc[ids_ctrl, c].to_numpy(float)
        out[f"{c}__case_mean"], out[f"{c}__control_mean"] = np.nanmean(a), np.nanmean(b)
        out[f"{c}__abs_smd"] = abs(C.smd(a, b))
    return out


def subset_data(data, keep_ids):
    keep = set(map(str, keep_ids))
    man = data.manifest[data.manifest["participant_id"].isin(keep)].copy()
    return C.Data(man, data.cohort_by_pid[data.cohort_by_pid.index.isin(keep)],
                  data.demographics[data.demographics["participant_id"].astype(str).isin(keep)].copy(),
                  data.lookups, None, data.b2ai_root, data.participant_meta.loc[
                      data.participant_meta.index.isin(keep)])


def stratified_pairs(data, d, args, strat: pd.Series):
    """Exact matching on sex + `strat` (e.g. primary microphone), age caliper as primary."""
    pairs = []
    full_demo = data.demographics
    pooled_sd = float(full_demo[full_demo["participant_id"].map(data.cohort_by_pid).isin([d, C.CONTROL])]["age"].std())
    for level in strat.dropna().unique():
        ids = set(strat.index[strat == level])
        sub = full_demo[full_demo["participant_id"].astype(str).isin(ids)]
        if sub.empty:
            continue
        # match_demographics scales caliper by the stratum SD; rescale to the pooled SD.
        sd_sub = float(sub[sub["participant_id"].map(data.cohort_by_pid).isin([d, C.CONTROL])]["age"].std())
        cal = args.age_caliper_sd * pooled_sd / sd_sub if sd_sub and np.isfinite(sd_sub) else args.age_caliper_sd
        try:
            c, h, _ = C.pb.match_demographics(d, data.cohort_by_pid, sub, cal)
            pairs += list(zip(c, h))
        except Exception:  # noqa: BLE001
            continue
    return pairs


def pcmvf_row(data, d, args, setting, matched_pairs=None):
    co = C.standardized_cohort(data, d, args, matched_pairs=matched_pairs)
    preds = C.standardized_predictions(data, co, args, {"speech": C.speech_fn(data), "protocol": C.protocol_fn(data)})
    b = C.pair_bootstrap(preds, co.final_pairs, args.bootstrap, args.seed + 5001)
    return {"disease": d, "setting": setting, "pairs": len(co.final_pairs), "tier": co.tier,
            "speech_auc": b["speech_auc"], "speech_ci_low": b["speech_ci_low"], "speech_ci_high": b["speech_ci_high"],
            "protocol_auc": b["protocol_auc"], "protocol_ci_low": b["protocol_ci_low"],
            "protocol_ci_high": b["protocol_ci_high"],
            "pcmvf_label": C.pcmvf_label(b["speech_ci_low"], b["protocol_ci_low"], b["protocol_ci_high"], co.tier)}


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e2_2_site_country_device"))
    p.add_argument("--parts", nargs="+", default=["A", "B", "C", "D"])
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root)
    meta = data.participant_meta
    bal, ext, usa, dev = [], [], [], []

    acq_all = lambda r, i: C.acquisition_frame(data, r, i)  # noqa: E731
    acq_nodur = lambda r, i: C.acquisition_frame(data, r, i, ("country", "device", "gain"))  # noqa: E731

    def one(d):
        if {"A", "B"} & set(ns.parts):
            rows, cand = C.raw_cohort_rows(data, d)
            fams = sorted(rows["task_family"].dropna().unique())
            co = C.standardized_cohort(data, d, args)
        if "A" in ns.parts:
            case = [i for i in cand if data.cohort_by_pid.get(i) == d]
            ctrl = [i for i in cand if data.cohort_by_pid.get(i) == C.CONTROL]
            bal.append(balance(data, d, case, ctrl, "full_cohort"))
            bal.append(balance(data, d, [c for c, _ in co.final_pairs], [h for _, h in co.final_pairs],
                               "pcmvf_final_pairs"))
        if "B" in ns.parts:
            def raw_fn(kind):
                def f(r, i, fam):
                    sub = r[r["participant_id"].isin(i)]
                    P = C.protocol_df(data, sub, i, fam)
                    A = acq_all(sub, i) if kind != "protocol_plus_acq_nodur" else acq_nodur(sub, i)
                    return {"acquisition_only": A, "protocol": P,
                            "protocol_plus_acq": pd.concat([P, A], axis=1),
                            "protocol_plus_acq_nodur": pd.concat([P, A], axis=1)}[kind].to_numpy(float)
                return f
            kinds = ["acquisition_only", "protocol", "protocol_plus_acq", "protocol_plus_acq_nodur"]
            raw, _, _ = C.raw_predictions(data, d, args, {k: raw_fn(k) for k in kinds})
            braw = C.pair_bootstrap(raw, None, args.bootstrap, args.seed + 404)
            std = C.standardized_predictions(data, co, args, {
                "acquisition_only": lambda r, i: acq_all(r, i).to_numpy(float),
                "protocol": C.protocol_fn(data),
                "protocol_plus_acq": C.protocol_fn(data, extra=acq_all),
                "protocol_plus_acq_nodur": C.protocol_fn(data, extra=acq_nodur),
            })
            bstd = C.pair_bootstrap(std, co.final_pairs, args.bootstrap, args.seed + 5001)
            for stage, bb in [("raw_full_cohort", braw), ("pcmvf_residual", bstd)]:
                for k in kinds:
                    ext.append({"disease": d, "stage": stage, "features": k, "auc": bb[f"{k}_auc"],
                                "ci_low": bb[f"{k}_ci_low"], "ci_high": bb[f"{k}_ci_high"]})
        if "C" in ns.parts:
            usa_ids = meta.index[meta["country"].fillna("").eq("USA")]
            n_case = int((meta.loc[usa_ids, "cohort"] == d).sum())
            try:
                r = pcmvf_row(subset_data(data, usa_ids), d, args, "usa_only")
            except Exception as exc:  # noqa: BLE001
                r = {"disease": d, "setting": "usa_only", "error": repr(exc)}
            r["n_usa_cases_available"] = n_case
            usa.append(r)
        if "D" in ns.parts and "primary_microphone" in meta:
            try:
                pairs = stratified_pairs(data, d, args, meta["primary_microphone"])
                dev.append(pcmvf_row(data, d, args, "sex_age_microphone_matched", matched_pairs=pairs))
            except Exception as exc:  # noqa: BLE001
                dev.append({"disease": d, "setting": "sex_age_microphone_matched", "error": repr(exc)})
        return None

    C.run_each(ns.diseases, one)
    for name, rows in [("acquisition_balance", bal), ("extended_protocol_auc", ext),
                       ("usa_only_pcmvf", usa), ("device_matched_pcmvf", dev)]:
        if rows:
            C.save(rows, ns.out_dir, f"{name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

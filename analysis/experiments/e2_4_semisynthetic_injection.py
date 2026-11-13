#!/usr/bin/env python3
"""
Question addressed: External evidence for PCMVF itself is limited. Demonstrate that
PCMVF detects KNOWN collection artifacts. (complements e2_5 on independent data)

Positive/negative controls with ground truth, built from real Bridge2AI
recordings of HEALTHY CONTROLS ONLY, so there is no true pathology signal:

  pseudo-cohort  Healthy controls are split at random (stratified by sex) into
                 "PseudoCase" and "Healthy Control". True speech AUC = 0.5.

Injected artifacts (strength k = 0, 1, 2, 4):
  task_mix      PseudoCases get k extra copies of every respiratory/cough
                recording -> different evidence mixture. Raw speech AUC should
                rise above 0.5 (shortcut), Detect should flag counts, and
                equal task-wise evidence should return speech AUC to ~0.5.
  extra_session PseudoCases get k duplicated sessions (new session ids) ->
                Detect should flag sessions; the speech vector is unchanged.
  device_shift  PseudoCase recordings get an additive spectral-tilt offset of
                k * 0.1 SD on spectrogram/mel/mfcc features (a microphone
                signature). PCMVF CANNOT remove this unless device is modeled:
                a documented negative result motivating acquisition variables.

For each (artifact, k, replicate split) records raw speech AUC, raw protocol AUC,
standardized speech AUC, residual protocol AUC and the PCMVF label; the
expected behaviour is written in column `expected`.

Outputs (results/e2_4_semisynthetic_injection/): injection_results.csv,
injection_summary.csv (mean over replicate splits)
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
PSEUDO = "PseudoCase"
SHIFT_REPS = ("spectrogram", "mel_spectrogram", "mfcc")
EXPECTED = {
    "none": "all AUCs ~0.5",
    "task_mix": "raw speech>0.5 and raw protocol>0.5; standardized speech ~0.5, residual protocol ~0.5",
    "extra_session": "raw protocol>0.5; raw speech ~0.5; residual protocol>0.5 unless single-session (sessions stay in protocol)",
    "device_shift": "raw & standardized speech>0.5; protocol ~0.5 -> PCMVF labels 'supported' (known blind spot)",
}


def pseudo_split(data, rng):
    ctrl = data.cohort_by_pid[data.cohort_by_pid.eq(C.CONTROL)].index
    demo = data.demographics.set_index("participant_id").reindex(ctrl)
    lab = pd.Series(C.CONTROL, index=ctrl, dtype=object)
    for _, g in demo.groupby("sex"):
        ids = g.index.to_numpy()
        pick = rng.choice(ids, len(ids) // 2, replace=False)
        lab.loc[pick] = PSEUDO
    return lab


def inject(data, labels, artifact, k, rng):
    d = data.with_cohorts(labels)
    if artifact == "none" or k == 0:
        return d
    m = d.manifest
    cases = set(labels.index[labels.eq(PSEUDO)])
    if artifact == "task_mix":
        extra = m[m["participant_id"].isin(cases) & m["task_family"].eq("respiratory_cough")]
        d.manifest = pd.concat([m] + [extra] * k, ignore_index=True)
    elif artifact == "extra_session":
        reps = []
        for j in range(k):
            e = m[m["participant_id"].isin(cases)].copy()
            new_sid = e["session_id"] + f"_dup{j}"
            for rep in d.lookups:  # register duplicated keys
                lut = d.lookups[rep]
                for (pid, sid, t), ns_ in zip(zip(e["participant_id"], e["session_id"], e["task_norm"]), new_sid):
                    if (pid, sid, t) in lut:
                        lut[(pid, ns_, t)] = lut[(pid, sid, t)]
            e["session_id"] = new_sid
            reps.append(e)
        d.manifest = pd.concat([m] + reps, ignore_index=True)
    elif artifact == "device_shift":
        for rep in SHIFT_REPS:
            lut = d.lookups[rep]
            allv = np.stack(list(lut.values()))
            sd = np.nanstd(allv, 0)
            tilt = np.linspace(-1, 1, allv.shape[1]) * sd * 0.1 * k
            d.lookups[rep] = {key: (v + tilt if key[0] in cases else v) for key, v in lut.items()}
    d.manifest["_row_id"] = np.arange(len(d.manifest))
    return d


def evaluate(d, args):
    raw, _, _ = C.raw_predictions(d, PSEUDO, args)
    braw = C.pair_bootstrap(raw, None, args.bootstrap, args.seed + 303)
    co = C.standardized_cohort(d, PSEUDO, args)
    std = C.standardized_predictions(d, co, args, {"speech": C.speech_fn(d), "protocol": C.protocol_fn(d)})
    bstd = C.pair_bootstrap(std, co.final_pairs, args.bootstrap, args.seed + 5001)
    return {"raw_speech_auc": braw["speech_auc"], "raw_protocol_auc": braw["protocol_auc"],
            "pairs": len(co.final_pairs), "std_speech_auc": bstd["speech_auc"],
            "std_speech_ci_low": bstd["speech_ci_low"], "residual_protocol_auc": bstd["protocol_auc"],
            "residual_protocol_ci_low": bstd["protocol_ci_low"],
            "residual_protocol_ci_high": bstd["protocol_ci_high"],
            "pcmvf_label": C.pcmvf_label(bstd["speech_ci_low"], bstd["protocol_ci_low"],
                                         bstd["protocol_ci_high"], co.tier)}


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e2_4_semisynthetic_injection"))
    p.add_argument("--artifacts", nargs="+", default=["none", "task_mix", "extra_session", "device_shift"])
    p.add_argument("--strengths", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--replicates", type=int, default=5)
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns, primary_min_pairs=15)  # ~56 pseudo-cases; sex-stratified split
    base = C.load_data(ns.b2ai_root, attach_metadata=False)
    rows = []
    for rep in range(ns.replicates):
        rng = np.random.default_rng(10_000 + rep)
        labels = pseudo_split(base, rng)
        for art in ns.artifacts:
            for k in ([0] if art == "none" else ns.strengths):
                data = copy.copy(base)
                data.lookups = {r: dict(v) for r, v in base.lookups.items()}
                try:
                    r = evaluate(inject(data, labels, art, k, rng), args)
                except Exception as exc:  # noqa: BLE001
                    r = {"error": repr(exc)}
                rows.append({"replicate": rep, "artifact": art, "strength": k, "expected": EXPECTED[art], **r})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    C.save(df, ns.out_dir, "injection_results.csv")
    num = [c for c in df.columns if c.endswith("_auc")]
    if num:
        summ = df.groupby(["artifact", "strength"])[num].agg(["mean", "std"])
        summ.columns = ["__".join(c) for c in summ.columns]
        summ = summ.reset_index()
        summ["label_mode"] = df.groupby(["artifact", "strength"])["pcmvf_label"].agg(
            lambda s: s.mode().iat[0] if s.notna().any() else None).to_numpy()
        C.save(summ, ns.out_dir, "injection_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

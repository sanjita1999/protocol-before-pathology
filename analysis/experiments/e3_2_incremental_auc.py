#!/usr/bin/env python3
"""
Question addressed: Show incremental AUC in both directions: speech added to protocol,
and protocol added to speech.

Fits, on the same participants and folds:
    S    speech only (470-D released representations)
    P    protocol only (paper's protocol representation)
    S+P  early fusion (concatenation, same frozen PLS-logistic probe)
    S+P_late  late fusion (mean of S and P out-of-fold logits)
and reports
    dAUC(S+P - P)  = what speech adds beyond the assessment record
    dAUC(S+P - S)  = what the assessment record adds beyond speech
with paired bootstrap CIs (pairs resampled after standardization) and DeLong
paired tests, at three stages:
    raw_full_cohort, same_final_cohort_original_evidence, pcmvf_standardized.

Expected reading: in raw PD / CI, S+P - S > 0 and S+P - P ~ 0 (speech adds little
beyond protocol); after standardization S+P - P > 0 where speech is "supported".

Outputs (results/e3_2_incremental_auc/): incremental_auc.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
DELTAS = [("S+P", "P"), ("S+P", "S"), ("S+P_late", "P"), ("S+P_late", "S"), ("S", "P")]


def add_late(preds):
    s = preds["S"].set_index("participant_id")
    p = preds["P"].set_index("participant_id").reindex(s.index)
    clip = lambda x: np.clip(x, 1e-6, 1 - 1e-6)  # noqa: E731
    late = s[["y_true"]].copy()
    late["y_score"] = (logit(clip(s["y_score"])) + logit(clip(p["y_score"]))) / 2
    preds["S+P_late"] = late.reset_index()
    return preds


def summarise(d, stage, preds, pairs, args, seed):
    preds = add_late(preds)
    b = C.pair_bootstrap(preds, pairs, args.bootstrap, seed, deltas=DELTAS)
    df = C._merge_preds(preds)
    row = {"disease": d, "stage": stage, "n": b["n"], "pairs": b["n_pairs"]}
    for k in preds:
        row[f"auc_{k}"] = b[f"{k}_auc"]
    for a, c in DELTAS:
        key = f"delta[{a}-{c}]"
        row[key], row[key + "_ci_low"], row[key + "_ci_high"] = b[key], b[key + "_ci_low"], b[key + "_ci_high"]
        row[key + "_boot_p"] = b[key + "_boot_p"]
        row[key + "_delong_p"] = C.delong_paired_test(df["y_true"], df[a], df[c])["p"]
    return row


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e3_2_incremental_auc"))
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root, attach_metadata=False)

    def one(d):
        out = []
        # raw full cohort
        sp = lambda r, i, f: C.v6.build_speech_matrix_available(r, i, data.lookups)[0]  # noqa: E731
        pr = lambda r, i, f: C.protocol_df(data, r[r["participant_id"].isin(i)], i, f).to_numpy(float)  # noqa: E731
        raw, _, _ = C.raw_predictions(data, d, args, {
            "S": sp, "P": pr, "S+P": lambda r, i, f: np.hstack([sp(r, i, f), pr(r, i, f)])})
        out.append(summarise(d, "raw_full_cohort", raw, None, args, args.seed + 303))

        co = C.standardized_cohort(data, d, args)
        # same final cohort, original evidence (Table 1 "Original speech" column), pair-aware folds
        rows, _ = C.raw_cohort_rows(data, d)
        fams = sorted(rows["task_family"].dropna().unique())
        orig = rows[rows["participant_id"].isin(co.ids)]
        Xs, kept = C.v6.build_speech_matrix_available(orig, co.ids, data.lookups)
        if kept == [str(i) for i in co.ids]:
            Xp = C.protocol_df(data, orig, co.ids, fams).to_numpy(float)
            po = {k: C.pb.fixed_probe_oof(X, co.y, co.ids, co.folds)
                  for k, X in {"S": Xs, "P": Xp, "S+P": np.hstack([Xs, Xp])}.items()}
            out.append(summarise(d, "final_cohort_original_evidence", po, co.final_pairs, args, args.seed + 5001))

        std = C.standardized_predictions(data, co, args, {
            "S": C.speech_fn(data), "P": C.protocol_fn(data),
            "S+P": C.concat_fns(C.speech_fn(data), C.protocol_fn(data))})
        out.append(summarise(d, "pcmvf_standardized", std, co.final_pairs, args, args.seed + 5001))
        return out

    C.save(C.run_each(ns.diseases, one), ns.out_dir, "incremental_auc.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

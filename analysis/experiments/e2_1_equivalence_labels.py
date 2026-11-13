#!/usr/bin/env python3
"""
Question addressed: 'Supported' relies on failing to reject protocol predictability.
Prefer an equivalence-style criterion / predefined tolerance around 0.5. The rule
is sensitive to cohort size and power.

For each cohort's final standardized predictions (exactly the Table 1 run):

 1. Bootstrap TOST for the residual protocol AUC against equivalence margins
    delta in {0.05, 0.075, 0.10, 0.15}: equivalent if the 90 % pair-bootstrap
    interval lies inside [0.5 - delta, 0.5 + delta].
 2. Smallest equivalence margin delta* that each cohort can claim
    (= max(0.5 - L90, U90 - 0.5)). Reviewer's example: Cognitive Impairment
    0.566 [0.419, 0.715] -> delta* ~ 0.2, i.e. AUC up to ~0.7 not excluded.
 3. Revised four-way rule (pre-specified margin delta0 = 0.10 by default):
      supported_equivalent   speech L95 > 0.5  and protocol equivalent at delta0
      supported_provisional  speech L95 > 0.5, protocol CI covers 0.5, NOT equivalent
      protocol_sensitive     protocol L95 > 0.5
      unresolved             speech L95 <= 0.5
    plus a speech-vs-protocol margin: Delta = speech - protocol with pair-bootstrap CI.
 4. Power / detectability: minimum residual protocol AUC that this cohort size
    would detect with 80 % power (Hanley-McNeil variance), so readers see which
    cohorts are too small for a "supported" claim to be informative.

Outputs (results/e2_1_equivalence_labels/):
    tost_by_margin.csv, revised_labels.csv, protocol_bootstrap_draws.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
MARGINS = [0.05, 0.075, 0.10, 0.15, 0.20]


def hanley_mcneil_se(auc, n1, n0):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))


def min_detectable_auc(n1, n0, power=0.8, alpha=0.05):
    for a in np.arange(0.501, 0.999, 0.001):
        se = hanley_mcneil_se(a, n1, n0)
        if norm.cdf((a - 0.5) / se - norm.ppf(1 - alpha / 2)) >= power:
            return float(a)
    return np.nan


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e2_1_equivalence_labels"))
    p.add_argument("--primary-margin", type=float, default=0.10)
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root, attach_metadata=False)
    tost_rows, labels, draws_all = [], [], []

    def one(d):
        co = C.standardized_cohort(data, d, args)
        preds = C.standardized_predictions(data, co, args, {"speech": C.speech_fn(data),
                                                            "protocol": C.protocol_fn(data)})
        b = C.pair_bootstrap(preds, co.final_pairs, args.bootstrap, args.seed + 5001,
                             deltas=[("speech", "protocol")], return_draws=True)
        dr = b.pop("_draws")
        draws_all.append(dr.assign(disease=d))
        for m in MARGINS:
            t = C.tost_auc(dr["protocol"].to_numpy(), b["protocol_auc"], m)
            tost_rows.append({"disease": d, "pairs": len(co.final_pairs), **t})
        eq = C.tost_auc(dr["protocol"].to_numpy(), b["protocol_auc"], ns.primary_margin)["equivalent"]
        if co.tier != "primary_supported":
            lab = "exploratory_small_n"
        elif b["protocol_ci_low"] > 0.5:
            lab = "protocol_sensitive"
        elif b["speech_ci_low"] <= 0.5:
            lab = "unresolved"
        else:
            lab = "supported_equivalent" if eq else "supported_provisional"
        n = len(co.final_pairs)
        labels.append({
            "disease": d, "pairs": n, "tier": co.tier,
            "speech_auc": b["speech_auc"], "speech_ci_low": b["speech_ci_low"], "speech_ci_high": b["speech_ci_high"],
            "protocol_auc": b["protocol_auc"], "protocol_ci_low": b["protocol_ci_low"],
            "protocol_ci_high": b["protocol_ci_high"],
            "smallest_equivalence_margin": C.smallest_equivalence_margin(dr["protocol"].to_numpy()),
            "speech_minus_protocol": b["delta[speech-protocol]"],
            "speech_minus_protocol_ci_low": b["delta[speech-protocol]_ci_low"],
            "speech_minus_protocol_ci_high": b["delta[speech-protocol]_ci_high"],
            "min_detectable_protocol_auc_80pct": min_detectable_auc(n, n),
            "original_label": C.pcmvf_label(b["speech_ci_low"], b["protocol_ci_low"], b["protocol_ci_high"], co.tier),
            f"revised_label_delta{ns.primary_margin}": lab,
        })
        return None

    C.run_each(ns.diseases, one)
    C.save(tost_rows, ns.out_dir, "tost_by_margin.csv")
    C.save(labels, ns.out_dir, "revised_labels.csv")
    if draws_all:
        C.save(pd.concat(draws_all), ns.out_dir, "protocol_bootstrap_draws.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

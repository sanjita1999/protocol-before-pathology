#!/usr/bin/env python3
"""
Question addressed: Why only confidence-interval overlap instead of DeLong's test?

On exactly the Table 1 predictions (and on the raw full cohort), reports side by
side:
  - AUC CI: DeLong (asymptotic) vs pair bootstrap (primary) vs stratified bootstrap
  - speech vs protocol comparison: DeLong paired test p, pair-bootstrap delta CI/p
  - label agreement: PCMVF label computed from DeLong CIs vs bootstrap CIs

Note for the analysis: PCMVF labels are NOT based on CI overlap between speech and
protocol; each AUC is compared with 0.5. The paired speech-protocol difference
(paper Sec. 4.1) is already a paired bootstrap; DeLong is added as confirmation.
DeLong assumes independent participants; after matching, pairs are dependent,
which is why the matched-pair bootstrap is primary.

Outputs (results/e4_4_delong_vs_bootstrap/): delong_vs_bootstrap.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent


def compare(d, stage, preds, pairs, tier, args, seed):
    df = C._merge_preds(preds)
    y = df["y_true"]
    row = {"disease": d, "stage": stage, "n": len(df)}
    bp = C.pair_bootstrap(preds, pairs, args.bootstrap, seed, deltas=[("speech", "protocol")]) if pairs else None
    bs = C.pair_bootstrap(preds, None, args.bootstrap, seed, deltas=[("speech", "protocol")])
    for k in ("speech", "protocol"):
        auc, se, lo, hi = C.delong_auc_ci(y, df[k])
        row.update({f"{k}_auc": auc, f"{k}_delong_ci_low": lo, f"{k}_delong_ci_high": hi,
                    f"{k}_strat_boot_ci_low": bs[f"{k}_ci_low"], f"{k}_strat_boot_ci_high": bs[f"{k}_ci_high"]})
        if bp:
            row.update({f"{k}_pair_boot_ci_low": bp[f"{k}_ci_low"], f"{k}_pair_boot_ci_high": bp[f"{k}_ci_high"]})
    t = C.delong_paired_test(y, df["speech"], df["protocol"])
    row.update({"delta_speech_minus_protocol": t["delta"], "delong_paired_p": t["p"],
                "strat_boot_delta_p": bs["delta[speech-protocol]_boot_p"]})
    if bp:
        row["pair_boot_delta_p"] = bp["delta[speech-protocol]_boot_p"]
        row["label_pair_bootstrap"] = C.pcmvf_label(row["speech_pair_boot_ci_low"], row["protocol_pair_boot_ci_low"],
                                                    row["protocol_pair_boot_ci_high"], tier)
        row["label_delong"] = C.pcmvf_label(row["speech_delong_ci_low"], row["protocol_delong_ci_low"],
                                            row["protocol_delong_ci_high"], tier)
        row["labels_agree"] = row["label_pair_bootstrap"] == row["label_delong"]
    return row


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__), str(C.RESULTS / "e4_4_delong_vs_bootstrap"))
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root, attach_metadata=False)

    def one(d):
        raw, _, _ = C.raw_predictions(data, d, args)
        out = [compare(d, "raw_full_cohort", raw, None, "primary_supported", args, args.seed + 303)]
        co = C.standardized_cohort(data, d, args)
        std = C.standardized_predictions(data, co, args, {"speech": C.speech_fn(data), "protocol": C.protocol_fn(data)})
        out.append(compare(d, "pcmvf_standardized", std, co.final_pairs, co.tier, args, args.seed + 5001))
        return out

    C.save(C.run_each(ns.diseases, one), ns.out_dir, "delong_vs_bootstrap.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

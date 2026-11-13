#!/usr/bin/env python3
"""
Gate check: the shared library must reproduce the frozen v8 Table 1 numbers
(final pairs, standardized speech AUC, residual protocol AUC) before any
analysis is trusted. Also writes the per-participant seed-averaged
predictions that v8 never saved (needed by DeLong / TOST / incremental AUC).

Output: results/verify/table1_reproduction.csv, table1_predictions.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcmvf_common as C  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "verify"))
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root)
    ref = pd.read_csv(C.V8_OUT / "publication_summary.csv").set_index("disease")
    pred_file = ns.out_dir / "table1_predictions.csv"
    pred_file.unlink(missing_ok=True)

    def one(d):
        co = C.standardized_cohort(data, d, args)
        preds = C.standardized_predictions(data, co, args, {
            "speech": C.speech_fn(data), "protocol": C.protocol_fn(data)})
        C.save_preds(preds, ns.out_dir, "table1_predictions.csv", disease=d)
        b = C.pair_bootstrap(preds, co.final_pairs, args.bootstrap, args.seed + 5001)
        r = ref.loc[d] if d in ref.index else {}
        row = {
            "disease": d, "pairs": len(co.final_pairs), "budget": co.budget, "tier": co.tier,
            "speech_auc": b["speech_auc"], "speech_ci": f"[{b['speech_ci_low']:.3f},{b['speech_ci_high']:.3f}]",
            "protocol_auc": b["protocol_auc"],
            "protocol_ci": f"[{b['protocol_ci_low']:.3f},{b['protocol_ci_high']:.3f}]",
            "v8_pairs": r.get("final_pairs"), "v8_speech_auc": r.get("balanced_speech_auc"),
            "v8_protocol_auc": r.get("balanced_protocol_auc"),
        }
        row["match"] = (abs(row["speech_auc"] - (row["v8_speech_auc"] or -1)) < 1e-3
                        and abs(row["protocol_auc"] - (row["v8_protocol_auc"] or -1)) < 1e-3)
        print(row)
        return row

    out = pd.DataFrame(C.run_each(ns.diseases, one))
    C.save(out, ns.out_dir, "table1_reproduction.csv")
    if ns.seeds == 20 and "match" in out and not out["match"].fillna(False).all():
        print("WARNING: not all cohorts reproduce v8 exactly.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Question addressed: I don't understand the residual protocol test. How does it
interact with the Standardize stage?

Produces a worked, numeric walkthrough for chosen cohorts (default: Parkinson's
Disease = protocol-sensitive, Airway Stenosis = supported) on the SAME final
matched participants at each stage:

  original evidence  -> shared 7-family menu -> equal task-wise evidence (1 seed)

For every protocol feature it reports case mean, control mean, |SMD| and
whether it is constant at that stage (constant features cannot carry signal), and
the protocol AUC of the frozen probe at that stage. The final column shows that
after equal evidence only total_sessions, age and sex can still vary; the
"residual protocol AUC" is the frozen probe fitted on exactly what the
standardization did NOT equalize.

Outputs (results/e3_3_residual_protocol_walkthrough/):
    walkthrough_<cohort>.csv, walkthrough_<cohort>.md  (table ready for the analysis)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent


def stage_table(data, rows, co, fams, stage):
    P = C.protocol_df(data, rows, co.ids, fams)
    pred = C.pb.fixed_probe_oof(P.to_numpy(float), co.y, co.ids, co.folds)
    auc = roc_auc_score(pred["y_true"], pred["y_score"])
    case = co.y == 1
    recs = []
    for c in P.columns:
        v = P[c].to_numpy(float)
        recs.append({"stage": stage, "feature": c, "case_mean": np.nanmean(v[case]),
                     "control_mean": np.nanmean(v[~case]), "abs_smd": abs(C.smd(v[case], v[~case])),
                     "constant": bool(np.nanstd(v) < 1e-12), "stage_protocol_auc": auc})
    return recs


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e3_3_residual_protocol_walkthrough"))
    p.set_defaults(diseases=["Parkinson's Disease", "Airway Stenosis"])
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root, attach_metadata=False)
    for d in ns.diseases:
        co = C.standardized_cohort(data, d, args)
        rows, _ = C.raw_cohort_rows(data, d)
        orig = rows[rows["participant_id"].isin(co.ids)]
        fams = sorted(orig["task_family"].dropna().unique())
        menu = orig[orig["task_family"].isin(C.MENU)]
        eq = C.pb.sample_taskwise(co.source, co.ids, co.menu, co.budget, seed=args.seed)
        recs = (stage_table(data, orig, co, fams, "1_original_evidence")
                + stage_table(data, menu, co, C.MENU, "2_shared_menu")
                + stage_table(data, eq, co, C.MENU, "3_equal_taskwise_evidence"))
        df = pd.DataFrame(recs)
        tag = d.lower().replace("'", "").replace(" ", "_")
        C.save(df, ns.out_dir, f"walkthrough_{tag}.csv")
        wide = df.pivot_table(index="feature", columns="stage", values="abs_smd", aggfunc="first")
        const = df.pivot_table(index="feature", columns="stage", values="constant", aggfunc="first")
        aucs = df.groupby("stage")["stage_protocol_auc"].first()
        lines = [f"### {d}: {len(co.final_pairs)} matched pairs (same participants at every stage)", "",
                 "| feature | " + " | ".join(wide.columns) + " |", "|---" * (len(wide.columns) + 1) + "|"]
        for f, r in wide.iterrows():
            cells = [("constant" if const.loc[f, s] else f"{r[s]:.2f}") if s in r and pd.notna(r[s]) else "-"
                     for s in wide.columns]
            lines.append(f"| {f} | " + " | ".join(cells) + " |")
        lines.append("| **protocol AUC (1 seed)** | " + " | ".join(f"**{aucs[s]:.3f}**" for s in wide.columns) + " |")
        lines += ["", "Cells are |SMD| between cases and matched controls; 'constant' = identical for every "
                  "participant, so the probe cannot use it."]
        (ns.out_dir / f"walkthrough_{tag}.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

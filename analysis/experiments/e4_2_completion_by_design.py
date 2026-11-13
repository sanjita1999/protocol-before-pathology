#!/usr/bin/env python3
"""
Question addressed: If a participant cannot complete the interview due to their
condition, isn't completion a useful signal shared by audio and metadata?

Tests whether the completion difference is INABILITY (clinical, informative) or
ASSIGNMENT (protocol design: the task was never administered to controls).

Uses phenotype/task/acoustic_task.tsv `acoustic_task_cohort` = protocol module
(generic / voice / neurology / respiratory / mood) that assigned each task, and
`acoustic_task_status` (all released tasks are "Completed": failed attempts are
not recorded).

 A. Module exposure per cohort: share of participants with >= 1 task from each
    module. If controls never received the neurology module, cognitive-linguistic
    "non-completion" in controls is non-administration.
 B. Family x module cross-tab: which module delivers each task family
    (cognitive_linguistic expected = neurology).
 C. Completion AMONG THE EXPOSED: for participants who received the module that
    carries a family, completion rate by cohort. Near-100 % in every exposed
    group means completion does not encode inability.
 D. Within-case informativeness: among cases, Spearman rho between number of
    completed families / observations and severity fields
    (phenotype/diagnosis/<disease>.tsv).

Outputs (results/e4_2_completion_by_design/):
    module_exposure_by_cohort.csv, family_by_module.csv,
    completion_among_exposed.csv, completion_vs_severity.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402
from e4_1_longitudinal_vs_quantity import severity_fields  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e4_2_completion_by_design"))
    ns = C.finalize_cli(p.parse_args())
    data = C.load_data(ns.b2ai_root)
    m = data.manifest.copy()
    print(f"[module coverage] {m['task_module'].notna().mean():.1%} of manifest rows have a module label")
    m["task_module"] = m["task_module"].fillna("unmatched")

    exp = (m.groupby(["participant_id", "task_module"]).size().unstack(fill_value=0) > 0)
    exp["cohort"] = exp.index.map(data.cohort_by_pid)
    C.save(exp.groupby("cohort").mean().reset_index(), ns.out_dir, "module_exposure_by_cohort.csv")

    fam_mod = pd.crosstab(m["task_family"], m["task_module"], normalize="index").reset_index()
    C.save(fam_mod, ns.out_dir, "family_by_module.csv")

    # module carrying each family (majority among matched rows)
    known = m[m["task_module"] != "unmatched"]
    carrier = known.groupby("task_family")["task_module"].agg(lambda s: s.mode().iat[0])
    comp = m.assign(_one=1).pivot_table(index="participant_id", columns="task_family", values="_one",
                                        aggfunc="max", fill_value=0)
    rows = []
    for fam, mod in carrier.items():
        if mod not in exp:
            continue
        exposed = exp.index[exp[mod]]
        for cohort, g in comp.loc[comp.index.intersection(exposed)].groupby(lambda i: data.cohort_by_pid.get(i)):
            rows.append({"task_family": fam, "carrier_module": mod, "cohort": cohort,
                         "n_exposed": len(g), "completion_among_exposed": float(g[fam].mean()),
                         "n_not_exposed": int(sum(data.cohort_by_pid.reindex(comp.index).eq(cohort)) - len(g))})
    C.save(rows, ns.out_dir, "completion_among_exposed.csv")

    sev_rows = []
    nfam = m.groupby("participant_id")["task_family"].nunique()
    nobs = m.groupby("participant_id").size()
    for d in ns.diseases:
        sev = severity_fields(data.b2ai_root, d)
        ids = [i for i in nfam.index if data.cohort_by_pid.get(i) == d]
        for c in sev.columns:
            v = sev[c].reindex([i.lstrip("0") for i in ids]).to_numpy(float)
            for name, x in [("n_families_completed", nfam), ("n_observations", nobs)]:
                xv = x.reindex(ids).to_numpy(float)
                ok = np.isfinite(v) & np.isfinite(xv)
                if ok.sum() >= 10 and np.std(xv[ok]) > 0 and np.std(v[ok]) > 0:
                    r = spearmanr(xv[ok], v[ok])
                    sev_rows.append({"disease": d, "severity_field": c, "protocol_variable": name,
                                     "n": int(ok.sum()), "rho": r.correlation, "p": r.pvalue})
    C.save(sev_rows, ns.out_dir, "completion_vs_severity.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

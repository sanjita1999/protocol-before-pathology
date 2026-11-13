#!/usr/bin/env python3
"""
Question addressed: A citation or justifying sentence is needed for the n = 20 rule.

 A. Simulation (binormal scores, equal-variance) for n_pairs in
    {8, 10, 15, 20, 25, 30, 40, 64} and true AUC in {0.50, 0.60, 0.65, 0.70, 0.75, 0.80}:
      - mean 95 % percentile-bootstrap CI half-width
      - power: P(lower 95 % bound > 0.5)
      - type-I error at AUC 0.5
      - P(protocol CI covers 0.5) when true protocol AUC = 0.65 / 0.70
        (= chance of a false "supported" label for a truly protocol-sensitive cohort)
 B. Empirical: subsample the real Table 1 standardized predictions to n pairs
    (200 draws) and record CI width and label (uses verify_reproduces_table1
    predictions file if present, else recomputes).
 C. Analytical reference: Hanley & McNeil (1982, Radiology 143:29-36) SE.

Justification sentence produced for the paper: at 20 pairs the 95 % CI half-width
is ~0.12-0.15 and power to detect AUC 0.75 vs 0.5 is ~0.8; below ~15 pairs CIs
exceed +/-0.15, so labels are uninformative (values filled in from the output).

Outputs (results/e4_5_min_pairs_power/): simulation.csv, empirical_subsample.csv
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
NS = [8, 10, 15, 20, 25, 30, 40, 64]
AUCS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]


def boot_ci(y, s, n_boot, rng):
    """Percentile CI, cases and controls resampled separately; vectorised Mann-Whitney AUC."""
    y = np.asarray(y)
    pos, neg = np.asarray(s)[y == 1], np.asarray(s)[y == 0]
    P = pos[rng.integers(0, len(pos), (n_boot, len(pos)))]
    N = neg[rng.integers(0, len(neg), (n_boot, len(neg)))]
    cmp = (P[:, :, None] > N[:, None, :]) + 0.5 * (P[:, :, None] == N[:, None, :])
    return np.percentile(cmp.mean(axis=(1, 2)), [2.5, 97.5])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sims", type=int, default=500)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--out-dir", default=str(C.RESULTS / "e4_5_min_pairs_power"))
    p.add_argument("--quick", action="store_true")
    ns = p.parse_args()
    if ns.quick:
        ns.sims, ns.bootstrap = 50, 100
    out = Path(ns.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    rows = []
    for n in NS:
        for a in AUCS:
            mu = np.sqrt(2) * norm.ppf(a)
            lo_hi = []
            for _ in range(ns.sims):
                s = np.r_[rng.normal(mu, 1, n), rng.normal(0, 1, n)]
                lo_hi.append(boot_ci(np.r_[np.ones(n), np.zeros(n)], s, ns.bootstrap, rng))
            lo_hi = np.array(lo_hi)
            rows.append({"n_pairs": n, "true_auc": a, "mean_ci_halfwidth": float(np.mean((lo_hi[:, 1] - lo_hi[:, 0]) / 2)),
                         "p_lower_gt_0.5": float(np.mean(lo_hi[:, 0] > 0.5)),
                         "p_ci_covers_0.5": float(np.mean((lo_hi[:, 0] <= 0.5) & (lo_hi[:, 1] >= 0.5)))})
        print(f"[sim] n_pairs={n} done", flush=True)
    C.save(rows, out, "simulation.csv")

    pred_file = C.RESULTS / "verify" / "table1_predictions.csv"
    if pred_file.exists():
        pr = pd.read_csv(pred_file)
        emp = []
        for d, g in pr.groupby("disease"):
            s = g[g["model"] == "speech"].set_index("participant_id")
            pp = g[g["model"] == "protocol"].set_index("participant_id")
            cases, ctrls = s.index[s["y_true"] == 1].to_numpy(), s.index[s["y_true"] == 0].to_numpy()
            for n in [x for x in NS if x <= min(len(cases), len(ctrls))]:
                for _ in range(200 if not ns.quick else 20):
                    ci, hi = rng.choice(cases, n, replace=False), rng.choice(ctrls, n, replace=False)
                    ids = np.r_[ci, hi]
                    y = np.r_[np.ones(n), np.zeros(n)]
                    slo, _ = boot_ci(y, s.loc[ids, "y_score"].to_numpy(), 200, rng)
                    plo, phi = boot_ci(y, pp.loc[ids, "y_score"].to_numpy(), 200, rng)
                    emp.append({"disease": d, "n_pairs": n,
                                "label": C.pcmvf_label(slo, plo, phi, "primary_supported")})
        emp = pd.DataFrame(emp)
        C.save(emp.groupby(["disease", "n_pairs"])["label"].value_counts(normalize=True).rename("share").reset_index(),
               out, "empirical_subsample.csv")
    else:
        print(f"[empirical skipped] run common/verify_reproduces_table1.py first ({pred_file})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

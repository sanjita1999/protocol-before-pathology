#!/usr/bin/env python3
"""
Follow-up to e1_1: does the residual protocol signal that flexible models find
(AUC 0.55-0.65 in voice cohorts) survive correction for model selection?

On each final PCMVF cohort (same pairs, pair-aware folds, equal task-wise
evidence) and the residual protocol features:

  * Nested CV: in every outer training fold, a pair-aware inner 3-fold CV picks
    one model from the pool {logistic, rf, hgb, mlp, svm_rbf, knn}; the winner is
    refit and scores the outer test fold. Repeated R times (outer folds and
    subsampling seed change per repeat). Statistic = mean selected-model AUC.
  * Permutation calibration: labels are swapped within matched pairs (preserves
    pairing and fold structure) and the WHOLE nested procedure, including model
    selection, is rerun K times. p = (1 + #null >= observed) / (K + 1).
  * Per-model AUCs with a max-T correction: each model's observed AUC is compared
    with the null distribution of the maximum AUC over all models.
  * Driver check: the same per-model AUCs on age-only and sessions-only features.

Outputs (results/e1_1b_nested_permutation/): nested_permutation.csv, null_draws.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
POOL = {
    "logistic": lambda: LogisticRegression(class_weight="balanced", max_iter=5000),
    "rf": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced",
                                         random_state=0, n_jobs=1),
    "hgb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, class_weight="balanced",
                                                  random_state=0),
    "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, random_state=0),
    "svm_rbf": lambda: SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=0),
    "knn": lambda: KNeighborsClassifier(n_neighbors=7),
}


def fit_predict(name, Xtr, ytr, Xte):
    keep = np.nanstd(Xtr, 0) > 1e-12
    if not keep.any():
        return np.full(len(Xte), 0.5)
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), POOL[name]())
    m.fit(Xtr[:, keep], ytr)
    return m.predict_proba(Xte[:, keep])[:, 1]


def pair_folds(pair_idx, n_splits, rng):
    """pair_idx: array of pair ids per row -> fold per row (pairs kept together)."""
    up = np.unique(pair_idx)
    f = dict(zip(rng.permutation(up), np.arange(len(up)) % n_splits))
    return np.array([f[p] for p in pair_idx])


def nested_once(X, y, pair_idx, rng):
    outer = pair_folds(pair_idx, 5, rng)
    sel = np.zeros(len(y))
    per = {m: np.zeros(len(y)) for m in POOL}
    chosen = []
    for f in range(5):
        tr, te = outer != f, outer == f
        inner = pair_folds(pair_idx[tr], 3, rng)
        inner_auc = {}
        for m in POOL:
            s = np.zeros(tr.sum())
            for g in range(3):
                itr, ite = inner != g, inner == g
                s[ite] = fit_predict(m, X[tr][itr], y[tr][itr], X[tr][ite])
            inner_auc[m] = roc_auc_score(y[tr], s)
        best = max(inner_auc, key=inner_auc.get)
        chosen.append(best)
        for m in POOL:
            per[m][te] = fit_predict(m, X[tr], y[tr], X[te])
        sel[te] = per[best][te]
    return roc_auc_score(y, sel), {m: roc_auc_score(y, s) for m, s in per.items()}, chosen


def statistic(Xs, y, pair_idx, seed):
    """Mean over repeats (one subsampled matrix per repeat)."""
    rng = np.random.default_rng(seed)
    sels, pers, chosen = [], [], []
    for X in Xs:
        a, p, c = nested_once(X, y, pair_idx, rng)
        sels.append(a)
        pers.append(p)
        chosen += c
    return float(np.mean(sels)), pd.DataFrame(pers).mean().to_dict(), chosen


def permuted(y, pair_idx, rng):
    yp = y.copy()
    for p in np.unique(pair_idx):
        if rng.random() < 0.5:
            idx = np.where(pair_idx == p)[0]
            yp[idx] = yp[idx[::-1]]
    return yp


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e1_1b_nested_permutation"))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=12)
    p.add_argument("--age-bin-years", type=float, default=None,
                   help="non-linear age handling: exact match on age bins of this width + the caliper")
    ns = C.finalize_cli(p.parse_args())
    if ns.quick:
        ns.repeats, ns.permutations = 2, 20
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root, attach_metadata=False)
    rows, nulls = [], []

    def one(d):
        if ns.age_bin_years:
            # Reuse e2_2's stratified matcher: exact on stratum (age bin) + sex, caliper on pooled SD.
            sys.path.insert(0, str(HERE))
            from e2_2_site_country_device import stratified_pairs
            demo = data.demographics.set_index("participant_id")["age"]
            bins = (demo // ns.age_bin_years).rename("age_bin")
            co = C.standardized_cohort(data, d, args, matched_pairs=stratified_pairs(data, d, args, bins))
        else:
            co = C.standardized_cohort(data, d, args)
        pid = {x: i for i, pr in enumerate(co.final_pairs) for x in pr}
        pair_idx = np.array([pid[i] for i in co.ids])
        sets = {}
        for r in range(ns.repeats):
            sampled = C.pb.sample_taskwise(co.source, co.ids, co.menu, co.budget, seed=args.seed + r)
            P = C.protocol_df(data, sampled, co.ids, co.menu)
            sets.setdefault("full_residual", []).append(P.to_numpy(float))
            sets.setdefault("age_only", []).append(P[["age"]].to_numpy(float))
            sets.setdefault("sessions_only", []).append(P[["total_sessions"]].to_numpy(float))
        out = []
        for fs, Xs in sets.items():
            obs_sel, obs_per, chosen = statistic(Xs, co.y, pair_idx, args.seed)
            row = {"disease": d, "features": fs, "pairs": len(co.final_pairs),
                   "nested_selected_auc": obs_sel,
                   "models_chosen": pd.Series(chosen).value_counts().to_dict()}
            row.update({f"auc_{m}": v for m, v in obs_per.items()})
            if fs == "full_residual":
                fr = C.standardized_predictions(data, co, args, {"speech": C.speech_fn(data),
                                                                 "protocol": C.protocol_fn(data)})
                b = C.pair_bootstrap(fr, co.final_pairs, args.bootstrap, args.seed + 5001)
                row.update({k: b[k] for k in ["speech_auc", "speech_ci_low", "speech_ci_high",
                                              "protocol_auc", "protocol_ci_low", "protocol_ci_high"]})
                rng = np.random.default_rng(args.seed + 999)
                perms = [permuted(co.y, pair_idx, rng) for _ in range(ns.permutations)]
                res = Parallel(n_jobs=ns.n_jobs)(
                    delayed(statistic)(Xs, yp, pair_idx, args.seed + 1 + k) for k, yp in enumerate(perms))
                null_sel = np.array([r[0] for r in res])
                null_max = np.array([max(r[1].values()) for r in res])
                row["p_nested_selected"] = (1 + (null_sel >= obs_sel).sum()) / (len(res) + 1)
                row["null_selected_mean"], row["null_selected_95"] = null_sel.mean(), np.percentile(null_sel, 95)
                row["null_maxT_95"] = np.percentile(null_max, 95)
                for m, v in obs_per.items():
                    row[f"p_maxT_{m}"] = (1 + (null_max >= v).sum()) / (len(res) + 1)
                nulls.append(pd.DataFrame({"disease": d, "null_selected_auc": null_sel, "null_max_auc": null_max}))
            out.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        return out

    rows = C.run_each(ns.diseases, one)
    C.save(rows, ns.out_dir, "nested_permutation.csv")
    if nulls:
        C.save(pd.concat(nulls), ns.out_dir, "null_draws.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Question addressed: Is the protocol AUC bottlenecked by logistic-regression capacity?

Re-runs the Detect stage (full cohort) and the Interpret stage (final
standardized cohort, 20 subsampling seeds, pair-aware folds) with a model zoo
whose hyper-parameters are tuned in an inner CV inside each training fold:

    pls_logistic (frozen), logistic (no PLS), rf, hgb, xgb*, mlp, svm_rbf, knn

for both the protocol representation and the speech representation.

Also reports, for the residual protocol model, the TRAINING AUC of each model.
If flexible models fit the training folds well (train AUC >> 0.5) but held-out
AUC stays near 0.5, the residual protocol signal is absent from the data, not
missed for lack of capacity.

Learning curve: raw protocol AUC with 25/50/75/100 % of each training fold. A
flat curve near the ceiling means more data/capacity would not change Detect.

Outputs (results/e1_1_model_capacity/):
    capacity_raw.csv           Detect: AUC [CI] per disease x feature x model
    capacity_standardized.csv  Interpret: AUC [pair-bootstrap CI] + PCMVF label per model
    capacity_label_stability.csv  does any model flip a cohort's PCMVF label?
    residual_protocol_train_vs_test.csv
    protocol_learning_curve.csv
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


def _fit_predict(fn, X_tr, y_tr, X_te, y_te):
    """Train on (X_tr, y_tr), score X_te, using any OOF-interface model.

    The OOF function is called with two folds: fold 1 = test rows (scored by a
    model trained on fold 0 = training rows). Predictions for fold 0 are
    discarded.
    """
    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    ids = [f"tr{i}" for i in range(len(y_tr))] + [f"te{i}" for i in range(len(y_te))]
    folds = np.concatenate([np.zeros(len(y_tr), int), np.ones(len(y_te), int)])
    pred = fn(X, y, ids, folds)
    te = pred[pred["participant_id"].str.startswith("te")].copy()
    te["_i"] = te["participant_id"].str[2:].astype(int)
    return te.sort_values("_i")["y_score"].to_numpy()


def train_auc(model_name, X, y, folds):
    """Mean training-fold AUC: model scored on the rows it was fitted on."""
    fn = C.model_oof(model_name)
    vals = []
    for f in np.unique(folds):
        tr = folds != f
        s = _fit_predict(fn, X[tr], y[tr], X[tr], y[tr])
        vals.append(roc_auc_score(y[tr], s))
    return float(np.mean(vals))


def learning_curve_auc(X, y, folds, frac, rng):
    """OOF AUC when each training fold is subsampled (stratified) to `frac`."""
    scores = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        keep = np.concatenate([
            rng.choice(tr[y[tr] == c], max(2, int(round(frac * (y[tr] == c).sum()))), replace=False)
            for c in (0, 1)])
        scores[te] = _fit_predict(C.pb.fixed_probe_oof, X[keep], y[keep], X[te], y[te])
    return float(roc_auc_score(y, scores))


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e1_1_model_capacity"))
    p.add_argument("--models", nargs="+", default=C.ALL_MODELS)
    p.add_argument("--zoo-seeds", type=int, default=5,
                   help="subsampling seeds for the non-frozen models (frozen probe keeps --seeds)")
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root)
    models = C.available_models(ns.models)

    raw_rows, std_rows, tt_rows, lc_rows = [], [], [], []

    def one(d):
        # ---------------- Detect (full cohort) ----------------
        # ponytail: zoo on protocol only; speech stays on the frozen probe. Add speech zoo if asked.
        proto_raw = {"protocol": lambda r, i, f: C.protocol_df(data, r[r["participant_id"].isin(i)], i, f).to_numpy(float)}
        preds, ids, folds = C.raw_predictions(data, d, args, proto_raw, models=models)
        b = C.pair_bootstrap(preds, None, args.bootstrap, args.seed + 303)
        for k in preds:
            feat, model = k.split("|")
            raw_rows.append({"disease": d, "stage": "raw_full_cohort", "features": feat, "model": model,
                             "auc": b[f"{k}_auc"], "ci_low": b[f"{k}_ci_low"], "ci_high": b[f"{k}_ci_high"],
                             "n": b["n"]})

        # Learning curve for raw protocol (frozen probe).
        rows, _ = C.raw_cohort_rows(data, d)
        fams = sorted(rows["task_family"].dropna().unique())
        Xp = C.protocol_df(data, rows[rows["participant_id"].isin(ids)], ids, fams).to_numpy(float)
        y = C.pb.make_labels(ids, d, data.cohort_by_pid)
        rng = np.random.default_rng(args.seed)
        for frac in [0.25, 0.5, 0.75, 1.0]:
            for rep in range(5 if frac < 1 else 1):
                lc_rows.append({"disease": d, "train_fraction": frac, "rep": rep,
                                "protocol_auc": learning_curve_auc(Xp, y, folds, frac, rng)})

        # ---------------- Interpret (standardized) ----------------
        co = C.standardized_cohort(data, d, args)
        # ponytail: zoo uses fewer seeds (runtime); frozen probe + speech keep the full --seeds.
        zoo_args = C.default_args(**{**vars(args), "seeds": min(ns.zoo_seeds, args.seeds)})
        zoo = {k: v for k, v in models.items() if k != "pls_logistic"}
        sp = C.standardized_predictions(data, co, zoo_args, {"protocol": C.protocol_fn(data)}, models=zoo)
        if "pls_logistic" in models:
            sp["protocol|pls_logistic"] = C.standardized_predictions(data, co, args, {"protocol": C.protocol_fn(data)})["protocol"]
        sp["speech|frozen"] = C.standardized_predictions(data, co, args, {"speech": C.speech_fn(data)})["speech"]
        b = C.pair_bootstrap(sp, co.final_pairs, args.bootstrap, args.seed + 5001)
        for m in models:
            s, pr_ = "speech|frozen", f"protocol|{m}"
            std_rows.append({
                "disease": d, "model": m, "pairs": len(co.final_pairs), "tier": co.tier,
                "speech_auc": b[f"{s}_auc"], "speech_ci_low": b[f"{s}_ci_low"], "speech_ci_high": b[f"{s}_ci_high"],
                "protocol_auc": b[f"{pr_}_auc"], "protocol_ci_low": b[f"{pr_}_ci_low"],
                "protocol_ci_high": b[f"{pr_}_ci_high"],
                "pcmvf_label": C.pcmvf_label(b[f"{s}_ci_low"], b[f"{pr_}_ci_low"], b[f"{pr_}_ci_high"], co.tier),
            })
        # Train-vs-test AUC of the residual protocol model on one subsample seed.
        sampled = C.pb.sample_taskwise(co.source, co.ids, co.menu, co.budget, seed=args.seed)
        Xr = C.protocol_fn(data)(sampled, co.ids)
        for m in models:
            tt_rows.append({"disease": d, "model": m,
                            "train_auc": train_auc(m, Xr, co.y, co.folds),
                            "test_auc_seed_avg": b[f"protocol|{m}_auc"]})
        return None

    C.run_each(ns.diseases, one)
    C.save(raw_rows, ns.out_dir, "capacity_raw.csv")
    std = pd.DataFrame(std_rows)
    C.save(std, ns.out_dir, "capacity_standardized.csv")
    C.save(tt_rows, ns.out_dir, "residual_protocol_train_vs_test.csv")
    C.save(lc_rows, ns.out_dir, "protocol_learning_curve.csv")
    if not std.empty:
        stab = std.groupby("disease").agg(
            primary_label=("pcmvf_label", lambda s: s[std.loc[s.index, "model"].eq("pls_logistic")].iat[0]
                           if (std.loc[s.index, "model"] == "pls_logistic").any() else None),
            n_distinct_labels=("pcmvf_label", "nunique"),
            labels=("pcmvf_label", lambda s: "; ".join(f"{m}:{l}" for m, l in zip(std.loc[s.index, "model"], s))),
            max_protocol_auc=("protocol_auc", "max"),
            argmax_protocol_model=("protocol_auc", lambda s: std.loc[s.idxmax(), "model"]),
        ).reset_index()
        C.save(stab, ns.out_dir, "capacity_label_stability.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

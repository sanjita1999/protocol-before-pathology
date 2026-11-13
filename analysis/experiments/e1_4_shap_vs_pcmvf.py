#!/usr/bin/env python3
"""
Question addressed: How does the method compare with explainability / feature
attribution methods such as SHAP?

Argument to test: attribution explains what a fitted model uses; PCMVF asks
whether the discrimination survives once the measurement process is equalized.
Three analyses make the difference concrete:

 1. Joint model (speech + protocol features, raw full cohort, gradient boosting).
    SHAP share attributed to protocol vs speech features per disease. Expectation:
    SHAP flags protocol features for PD / CI, but gives no estimate of what speech
    discrimination remains after removing them.

 2. Speech-only model (raw cohort). SHAP top-k speech features are correlated
    with protocol variables (counts, sessions, completion, country, device). If
    top-attributed "acoustic" features are protocol proxies, SHAP on a speech-only
    model looks clinically plausible yet cannot reveal the shortcut, because the
    protocol variables are not inputs.

 3. Same speech-only SHAP on the PCMVF-standardized cohort: proxy correlations
    should shrink, and the AUC change is what PCMVF quantifies.

Uses `shap.TreeExplainer` when installed; otherwise falls back to sklearn
permutation importance on held-out folds (reported in column `method`).

Outputs (results/e1_4_shap_vs_pcmvf/):
    joint_attribution_share.csv         per disease: share speech vs protocol (+ per group)
    speech_top_features_proxy_corr.csv  top speech features x protocol-variable |Spearman rho|
    speech_proxy_summary.csv            raw vs standardized: max/mean proxy correlation, AUC
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
TOPK = 20


def speech_names():
    return [f"{rep}__{i}" for rep in C.ORIGINAL_REPS for i in range(int(C.pb.REP_SPECS[rep]["dim"]))]


def oof_attribution(X, y, folds, names, seed):
    """Per-fold HGB; attribution on held-out rows. Returns (mean|attr| per feature, oof auc, method)."""
    try:
        import shap  # noqa: F401
        use_shap = True
    except ImportError:
        use_shap = False
    imp = np.zeros(X.shape[1])
    scores = np.zeros(len(y))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        im = SimpleImputer(strategy="median").fit(X[tr])
        Xtr, Xte = im.transform(X[tr]), im.transform(X[te])
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                           class_weight="balanced", random_state=seed).fit(Xtr, y[tr])
        scores[te] = m.predict_proba(Xte)[:, 1]
        if use_shap:
            import shap
            sv = shap.TreeExplainer(m).shap_values(Xte)
            sv = sv[1] if isinstance(sv, list) else sv
            imp += np.abs(sv).mean(0) * te.sum()
        else:
            pi = permutation_importance(m, Xte, y[te], scoring="roc_auc", n_repeats=5, random_state=seed)
            imp += np.clip(pi.importances_mean, 0, None) * te.sum()
    imp /= len(y)
    return pd.Series(imp, index=names), float(roc_auc_score(y, scores)), ("shap" if use_shap else "permutation")


def proxy_table(Xs, sp_names, top, P: pd.DataFrame):
    rows = []
    for feat in top:
        j = sp_names.index(feat)
        for col in P.columns:
            v = P[col].to_numpy(float)
            if np.nanstd(v) < 1e-12:
                continue
            ok = np.isfinite(Xs[:, j]) & np.isfinite(v)
            rho = spearmanr(Xs[ok, j], v[ok]).correlation
            rows.append({"speech_feature": feat, "protocol_variable": col, "abs_rho": abs(rho)})
    return pd.DataFrame(rows)


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e1_4_shap_vs_pcmvf"))
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root)
    sp_names = speech_names()
    share_rows, proxy_rows, summary = [], [], []

    def one(d):
        rows, cand = C.raw_cohort_rows(data, d)
        Xs, ids = C.v6.build_speech_matrix_available(rows, cand, data.lookups)
        fams = sorted(rows["task_family"].dropna().unique())
        sub = rows[rows["participant_id"].isin(ids)]
        P = C.protocol_df(data, sub, ids, fams)
        A = C.acquisition_frame(data, sub, ids)
        P_ext = pd.concat([P, A], axis=1)
        y = C.pb.make_labels(ids, d, data.cohort_by_pid)
        folds = C.pb.make_folds(ids, y, data.fold_map, args.seed)

        # 1. joint model attribution
        Xj = np.hstack([Xs, P.to_numpy(float)])
        imp, auc, method = oof_attribution(Xj, y, folds, sp_names + list(P.columns), args.seed)
        tot = imp.sum() or 1.0
        grp = imp.groupby(lambda n: n.split("__")[0] if n in sp_names else
                          ("protocol_" + n.split("__")[0] if "__" in n else "protocol_" + n)).sum() / tot
        share_rows.append({"disease": d, "method": method, "joint_oof_auc": auc,
                           "share_speech": float(imp[sp_names].sum() / tot),
                           "share_protocol": float(imp[list(P.columns)].sum() / tot),
                           **{f"share__{k}": float(v) for k, v in grp.items()}})

        # 2. speech-only attribution, raw cohort
        imp_s, auc_s, _ = oof_attribution(Xs, y, folds, sp_names, args.seed)
        top = imp_s.sort_values(ascending=False).index[:TOPK].tolist()
        pt = proxy_table(Xs, sp_names, top, P_ext)
        pt["disease"], pt["cohort_stage"] = d, "raw"
        proxy_rows.append(pt)
        summary.append({"disease": d, "stage": "raw", "speech_oof_auc_hgb": auc_s,
                        "max_abs_rho": pt["abs_rho"].max(), "mean_top_abs_rho":
                        pt.groupby("speech_feature")["abs_rho"].max().mean(),
                        "strongest_proxy": pt.loc[pt["abs_rho"].idxmax(), "protocol_variable"] if len(pt) else None})

        # 3. standardized cohort (first subsample seed)
        co = C.standardized_cohort(data, d, args)
        sampled = C.pb.sample_taskwise(co.source, co.ids, co.menu, co.budget, seed=args.seed)
        Xs2 = C.speech_fn(data)(sampled, co.ids)
        P2 = pd.concat([C.protocol_df(data, sampled, co.ids, co.menu),
                        C.acquisition_frame(data, sampled, co.ids)], axis=1)
        imp2, auc2, _ = oof_attribution(Xs2, co.y, co.folds, sp_names, args.seed)
        top2 = imp2.sort_values(ascending=False).index[:TOPK].tolist()
        pt2 = proxy_table(Xs2, sp_names, top2, P2)
        pt2["disease"], pt2["cohort_stage"] = d, "pcmvf_standardized"
        proxy_rows.append(pt2)
        summary.append({"disease": d, "stage": "pcmvf_standardized", "speech_oof_auc_hgb": auc2,
                        "max_abs_rho": pt2["abs_rho"].max() if len(pt2) else np.nan,
                        "mean_top_abs_rho": pt2.groupby("speech_feature")["abs_rho"].max().mean() if len(pt2) else np.nan,
                        "strongest_proxy": pt2.loc[pt2["abs_rho"].idxmax(), "protocol_variable"] if len(pt2) else None})
        return None

    C.run_each(ns.diseases, one)
    C.save(share_rows, ns.out_dir, "joint_attribution_share.csv")
    if proxy_rows:
        C.save(pd.concat(proxy_rows), ns.out_dir, "speech_top_features_proxy_corr.csv")
    C.save(summary, ns.out_dir, "speech_proxy_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

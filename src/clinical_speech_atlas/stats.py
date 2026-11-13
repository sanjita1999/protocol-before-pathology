from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

from .utils import stable_seed


def stratified_auc_bootstrap(pred: pd.DataFrame, B: int, seed: int) -> dict[str, float | list[float]]:
    cases = pred[pred.y_true.eq(1)].reset_index(drop=True)
    controls = pred[pred.y_true.eq(0)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        ci = rng.integers(0, len(cases), len(cases))
        hi = rng.integers(0, len(controls), len(controls))
        s = pd.concat([cases.iloc[ci], controls.iloc[hi]], ignore_index=True)
        vals.append(float(roc_auc_score(s.y_true, s.y_score)))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"ci_low": float(lo), "ci_high": float(hi), "bootstrap_auc": vals}


def paired_auc_delta_bootstrap(a: pd.DataFrame, b: pd.DataFrame, B: int, seed: int) -> dict[str, float]:
    x = a[["participant_id", "y_true", "y_score"]].merge(
        b[["participant_id", "y_true", "y_score"]], on=["participant_id", "y_true"], suffixes=("_a", "_b")
    )
    cases = x[x.y_true.eq(1)].reset_index(drop=True)
    controls = x[x.y_true.eq(0)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(B):
        ci = rng.integers(0, len(cases), len(cases))
        hi = rng.integers(0, len(controls), len(controls))
        s = pd.concat([cases.iloc[ci], controls.iloc[hi]], ignore_index=True)
        auc_a = roc_auc_score(s.y_true, s.y_score_a)
        auc_b = roc_auc_score(s.y_true, s.y_score_b)
        deltas.append(float(auc_b - auc_a))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    arr = np.asarray(deltas, dtype=float)
    # +1 correction prevents zero p-values under finite bootstrap Monte Carlo.
    p_le = (1.0 + float(np.sum(arr <= 0.0))) / (len(arr) + 1.0)
    p_ge = (1.0 + float(np.sum(arr >= 0.0))) / (len(arr) + 1.0)
    p_two = min(1.0, 2.0 * min(p_le, p_ge))
    return {
        "delta_auc": float(np.mean(arr)),
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "p_bootstrap_two_sided": float(p_two),
    }


def shared_control_rank_bootstrap(predictions: pd.DataFrame, B: int, seed: int, top_k: int = 3):
    diseases = sorted(predictions["disease"].unique())
    # Validate identical control IDs across diseases where possible.
    control_sets = {d: set(predictions[(predictions.disease.eq(d)) & predictions.y_true.eq(0)].participant_id.astype(str)) for d in diseases}
    shared_controls = set.intersection(*control_sets.values()) if control_sets else set()
    if not shared_controls:
        raise ValueError("No shared control participants across disease prediction tables")
    control_base = sorted(shared_controls)
    disease_cases = {
        d: predictions[(predictions.disease.eq(d)) & predictions.y_true.eq(1)].reset_index(drop=True)
        for d in diseases
    }
    # Scores for shared controls can differ slightly by disease because each classifier differs.
    control_by_disease = {
        d: predictions[(predictions.disease.eq(d)) & predictions.y_true.eq(0)].set_index("participant_id").loc[control_base]
        for d in diseases
    }
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(B):
        sampled_control_pos = rng.integers(0, len(control_base), len(control_base))
        sampled_control_ids = [control_base[i] for i in sampled_control_pos]
        aucs = {}
        for d in diseases:
            case = disease_cases[d]
            sampled_case = case.iloc[rng.integers(0, len(case), len(case))]
            ctrl = control_by_disease[d].loc[sampled_control_ids]
            y = np.r_[np.ones(len(sampled_case), dtype=int), np.zeros(len(ctrl), dtype=int)]
            score = np.r_[sampled_case.y_score.to_numpy(float), ctrl.y_score.to_numpy(float)]
            aucs[d] = float(roc_auc_score(y, score))
        ordered = sorted(aucs, key=lambda d: (-aucs[d], d))
        ranks = {d: i + 1 for i, d in enumerate(ordered)}
        for d in diseases:
            rows.append({"bootstrap": b, "disease": d, "auc": aucs[d], "rank": ranks[d]})
    boot = pd.DataFrame(rows)
    summary = boot.groupby("disease").agg(
        auc_mean=("auc", "mean"),
        auc_ci_low=("auc", lambda x: np.quantile(x, 0.025)),
        auc_ci_high=("auc", lambda x: np.quantile(x, 0.975)),
        median_rank=("rank", "median"),
        rank_ci_low=("rank", lambda x: np.quantile(x, 0.025)),
        rank_ci_high=("rank", lambda x: np.quantile(x, 0.975)),
        top_k_probability=("rank", lambda x: np.mean(np.asarray(x) <= top_k)),
    ).reset_index()
    pair_rows = []
    piv = boot.pivot(index="bootstrap", columns="disease", values="auc")
    for di in diseases:
        for dj in diseases:
            if di == dj:
                p = np.nan
            else:
                p = float((piv[di] > piv[dj]).mean())
            pair_rows.append({"disease_i": di, "disease_j": dj, "p_auc_i_gt_j": p})
    pair = pd.DataFrame(pair_rows)
    return boot, summary, pair


def ranking_kendall(a: pd.DataFrame, b: pd.DataFrame, disease_col="disease", score_col="auc") -> float:
    merged = a[[disease_col, score_col]].merge(b[[disease_col, score_col]], on=disease_col, suffixes=("_a", "_b"))
    if len(merged) < 2:
        return np.nan
    ra = merged[f"{score_col}_a"].rank(ascending=False, method="average")
    rb = merged[f"{score_col}_b"].rank(ascending=False, method="average")
    return float(kendalltau(ra, rb).statistic)


def condition_rank_bootstrap(
    predictions: pd.DataFrame,
    condition_col: str,
    B: int,
    seed: int,
) -> pd.DataFrame:
    """Rank multiple conditions (e.g., representations) using paired participant resampling.

    Only participants present in every condition are used, making the bootstrap paired across
    conditions. Cases and controls are resampled separately.
    """
    conditions = sorted(predictions[condition_col].dropna().astype(str).unique())
    if len(conditions) < 2:
        return pd.DataFrame()
    ids_by_cond = {c: set(predictions[predictions[condition_col].astype(str).eq(c)].participant_id.astype(str)) for c in conditions}
    shared = sorted(set.intersection(*ids_by_cond.values()))
    if not shared:
        return pd.DataFrame()
    tables = {
        c: predictions[predictions[condition_col].astype(str).eq(c)].set_index("participant_id").loc[shared]
        for c in conditions
    }
    base_y = tables[conditions[0]]["y_true"].astype(int)
    case_ids = list(base_y[base_y.eq(1)].index)
    ctrl_ids = list(base_y[base_y.eq(0)].index)
    if len(case_ids) < 2 or len(ctrl_ids) < 2:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows=[]
    for b in range(B):
        scase = [case_ids[i] for i in rng.integers(0, len(case_ids), len(case_ids))]
        sctrl = [ctrl_ids[i] for i in rng.integers(0, len(ctrl_ids), len(ctrl_ids))]
        aucs={}
        for c in conditions:
            t=tables[c]
            y=np.r_[np.ones(len(scase),dtype=int), np.zeros(len(sctrl),dtype=int)]
            score=np.r_[t.loc[scase,"y_score"].to_numpy(float), t.loc[sctrl,"y_score"].to_numpy(float)]
            aucs[c]=float(roc_auc_score(y,score))
        ordered=sorted(aucs,key=lambda c:(-aucs[c],c))
        ranks={c:i+1 for i,c in enumerate(ordered)}
        for c in conditions:
            rows.append({"bootstrap":b, condition_col:c, "auc":aucs[c], "rank":ranks[c]})
    boot=pd.DataFrame(rows)
    summary=boot.groupby(condition_col).agg(
        auc_boot_mean=("auc","mean"),
        median_rank=("rank","median"),
        rank_ci_low=("rank",lambda x:np.quantile(x,0.025)),
        rank_ci_high=("rank",lambda x:np.quantile(x,0.975)),
        p_rank_1=("rank",lambda x:np.mean(np.asarray(x)==1)),
    ).reset_index()
    summary["n_shared_participants"] = len(shared)
    return summary


def benjamini_hochberg(pvalues: list[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving input order."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    return out



def _auc_transform(values: np.ndarray, transform: str) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if transform == "raw":
        return x
    if transform == "logit":
        x = np.clip(x, 1e-4, 1 - 1e-4)
        return np.log(x / (1.0 - x))
    raise ValueError(transform)


def crossed_gstudy_three_facets(
    df: pd.DataFrame,
    object_col: str,
    facet_a_col: str,
    facet_b_col: str,
    score_col: str,
    transform: str = "logit",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Balanced three-facet crossed G-study variance decomposition.

    Disease is the object of measurement; task and representation are crossed
    measurement facets. With one score per disease x task x representation
    cell, the three-way interaction is the residual and also absorbs cell-level
    estimation noise. Raw method-of-moments estimates are retained; negative
    estimates are truncated to zero only for proportions and coefficients.
    """
    cols = [object_col, facet_a_col, facet_b_col, score_col]
    x = df[cols].dropna().copy()
    if x.duplicated([object_col, facet_a_col, facet_b_col]).any():
        raise ValueError("G-study requires one score per crossed cell")
    objects = sorted(x[object_col].astype(str).unique())
    alevels = sorted(x[facet_a_col].astype(str).unique())
    blevels = sorted(x[facet_b_col].astype(str).unique())
    I, J, K = len(objects), len(alevels), len(blevels)
    expected = I * J * K
    if len(x) != expected:
        raise ValueError(
            f"G-study requires a complete crossed design: got {len(x)} cells, expected {expected}"
        )
    oi = {v: i for i, v in enumerate(objects)}
    aj = {v: j for j, v in enumerate(alevels)}
    bk = {v: k for k, v in enumerate(blevels)}
    Y = np.empty((I, J, K), dtype=float)
    for _, row in x.iterrows():
        Y[oi[str(row[object_col])], aj[str(row[facet_a_col])], bk[str(row[facet_b_col])]] = float(row[score_col])
    Y = _auc_transform(Y, transform)
    grand = float(Y.mean())
    mi = Y.mean(axis=(1, 2))
    mj = Y.mean(axis=(0, 2))
    mk = Y.mean(axis=(0, 1))
    mij = Y.mean(axis=2)
    mik = Y.mean(axis=1)
    mjk = Y.mean(axis=0)

    ss_i = J * K * np.sum((mi - grand) ** 2)
    ss_j = I * K * np.sum((mj - grand) ** 2)
    ss_k = I * J * np.sum((mk - grand) ** 2)
    ss_ij = K * np.sum((mij - mi[:, None] - mj[None, :] + grand) ** 2)
    ss_ik = J * np.sum((mik - mi[:, None] - mk[None, :] + grand) ** 2)
    ss_jk = I * np.sum((mjk - mj[:, None] - mk[None, :] + grand) ** 2)
    resid = (
        Y - mij[:, :, None] - mik[:, None, :] - mjk[None, :, :]
        + mi[:, None, None] + mj[None, :, None] + mk[None, None, :] - grand
    )
    ss_ijk = float(np.sum(resid ** 2))
    dfs = {
        "disease": I - 1,
        "task": J - 1,
        "representation": K - 1,
        "disease_task": (I - 1) * (J - 1),
        "disease_representation": (I - 1) * (K - 1),
        "task_representation": (J - 1) * (K - 1),
        "residual_dtr": (I - 1) * (J - 1) * (K - 1),
    }
    ss = {
        "disease": ss_i,
        "task": ss_j,
        "representation": ss_k,
        "disease_task": ss_ij,
        "disease_representation": ss_ik,
        "task_representation": ss_jk,
        "residual_dtr": ss_ijk,
    }
    ms = {name: ss[name] / dfs[name] for name in ss}
    ve = ms["residual_dtr"]
    raw = {
        "residual_dtr": ve,
        "disease_task": (ms["disease_task"] - ve) / K,
        "disease_representation": (ms["disease_representation"] - ve) / J,
        "task_representation": (ms["task_representation"] - ve) / I,
        "disease": (ms["disease"] - ms["disease_task"] - ms["disease_representation"] + ve) / (J * K),
        "task": (ms["task"] - ms["disease_task"] - ms["task_representation"] + ve) / (I * K),
        "representation": (ms["representation"] - ms["disease_representation"] - ms["task_representation"] + ve) / (I * J),
    }
    nonneg = {name: max(0.0, float(value)) for name, value in raw.items()}
    total = float(sum(nonneg.values()))
    rows = []
    order = [
        "disease", "task", "representation", "disease_task",
        "disease_representation", "task_representation", "residual_dtr",
    ]
    for component in order:
        rows.append({
            "component": component,
            "variance_raw": float(raw[component]),
            "variance_nonnegative": nonneg[component],
            "proportion_total": nonneg[component] / total if total > 0 else np.nan,
            "mean_square": float(ms[component]),
            "df": int(dfs[component]),
            "scale": transform,
        })
    rel_err = (
        nonneg["disease_task"] / J
        + nonneg["disease_representation"] / K
        + nonneg["residual_dtr"] / (J * K)
    )
    abs_err = (
        rel_err
        + nonneg["task"] / J
        + nonneg["representation"] / K
        + nonneg["task_representation"] / (J * K)
    )
    vd = nonneg["disease"]
    summary = {
        "n_diseases": I,
        "n_tasks": J,
        "n_representations": K,
        "n_cells": expected,
        "g_coefficient_relative": float(vd / (vd + rel_err)) if vd + rel_err > 0 else np.nan,
        "phi_absolute": float(vd / (vd + abs_err)) if vd + abs_err > 0 else np.nan,
        "disease_variance_fraction": float(nonneg["disease"] / total) if total > 0 else np.nan,
        "task_plus_disease_task_fraction": float((nonneg["task"] + nonneg["disease_task"]) / total) if total > 0 else np.nan,
        "representation_plus_disease_representation_fraction": float((nonneg["representation"] + nonneg["disease_representation"]) / total) if total > 0 else np.nan,
        "scale": transform,
    }
    return pd.DataFrame(rows), summary


def sensitivity_at_specificity(
    y_true: np.ndarray, score: np.ndarray, target_specificity: float
) -> tuple[float, float, float]:
    y = np.asarray(y_true, int)
    s = np.asarray(score, float)
    fpr, tpr, thresholds = roc_curve(y, s)
    specificity = 1.0 - fpr
    eligible = np.where(specificity >= float(target_specificity) - 1e-12)[0]
    if len(eligible) == 0:
        return np.nan, np.nan, np.nan
    best = eligible[np.argmax(tpr[eligible])]
    return float(tpr[best]), float(specificity[best]), float(thresholds[best])


def bootstrap_sensitivity_at_specificity(
    pred: pd.DataFrame, target_specificity: float, B: int, seed: int
) -> dict[str, float]:
    cases = pred[pred.y_true.eq(1)].reset_index(drop=True)
    controls = pred[pred.y_true.eq(0)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        c = cases.iloc[rng.integers(0, len(cases), len(cases))]
        h = controls.iloc[rng.integers(0, len(controls), len(controls))]
        z = pd.concat([c, h], ignore_index=True)
        sens, _, _ = sensitivity_at_specificity(
            z.y_true.to_numpy(), z.y_score.to_numpy(), target_specificity
        )
        vals.append(sens)
    arr = np.asarray(vals, float)
    return {
        "mean": float(np.nanmean(arr)),
        "ci_low": float(np.nanquantile(arr, 0.025)),
        "ci_high": float(np.nanquantile(arr, 0.975)),
    }


def calibration_summary(
    y_true: np.ndarray, score: np.ndarray, n_bins: int = 10
) -> dict[str, float]:
    y = np.asarray(y_true, int)
    p = np.clip(np.asarray(score, float), 1e-6, 1 - 1e-6)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(logits, y)
    n_q = min(n_bins, len(np.unique(p)))
    bins = pd.qcut(pd.Series(p), q=n_q, duplicates="drop")
    tab = (
        pd.DataFrame({"y": y, "p": p, "bin": bins})
        .groupby("bin", observed=True)
        .agg(n=("y", "size"), obs=("y", "mean"), pred=("p", "mean"))
    )
    ece = float(np.sum((tab.n / len(y)) * np.abs(tab.obs - tab.pred)))
    return {
        "calibration_intercept": float(cal.intercept_[0]),
        "calibration_slope": float(cal.coef_[0, 0]),
        "ece": ece,
    }

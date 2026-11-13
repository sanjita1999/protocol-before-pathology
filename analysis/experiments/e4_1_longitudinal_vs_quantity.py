#!/usr/bin/env python3
"""
Question addressed: Couldn't the audio pick up deterioration between sessions? For a
progressive disease, longitudinal features should be expected and encouraged.

The paper's claim concerns the PROTOCOL model, which sees only the session COUNT,
not any acoustic change. This script separates "repeat sessions carry disease
progression" from "repeat sessions are a collection artifact":

 A. Who gets repeat sessions? n_sessions by cohort x country (site proxy) and
    controls. If repeat sessions occur almost only in one site/arm and almost
    never in controls, the count is an enrolment-design variable.
 B. Is session count related to disease severity within cases? Spearman rho of
    n_sessions with every numeric/ordinal severity field released in
    phenotype/diagnosis/<disease>.tsv (e.g. PD disease stage, UPDRS; CI CDR), and
    with the standardized single-session speech score. If the count tracked
    severity, it would be (weakly) clinical; if not, it is protocol.
 C. Does acoustic change between sessions exceed ordinary test-retest variation?
    For participants with >= 2 sessions, the within-participant cross-session
    distance of the participant-level speech vector (same equal-evidence menu)
    is compared across cohorts (PD/CI vs voice cohorts with repeat sessions) and
    against within-session repeat-recording distance (noise floor). Note: v3.0.0
    releases no session dates, so session ORDER and interval are unknown;
    progression cannot be modelled directly. This is stated as a limitation.
 D. Speech-only vs speech + n_sessions on the single-session standardized cohort:
    if n_sessions adds AUC after the acoustic content is fixed to one session,
    the gain comes from the count, not from audio.

Outputs (results/e4_1_longitudinal_vs_quantity/):
    sessions_by_cohort_country.csv, sessions_vs_severity.csv,
    cross_session_change.csv, single_session_plus_count.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
DIAG_FILES = {"Parkinson's Disease": "parkinsons_disease.tsv", "Cognitive Impairment": "cognitive_impairment.tsv",
              "Airway Stenosis": "airway_stenosis.tsv", "Laryngeal Dystonia": "laryngeal_dystonia.tsv",
              "Unilateral Vocal Fold Paralysis": "unilateral_vocal_fold_paralysis.tsv",
              "Muscle Tension Dysphonia": "muscle_tension_dysphonia.tsv", "Benign Lesions": "benign_lesions.tsv",
              "Unexplained Chronic Cough": "unexplained_chronic_cough.tsv"}


def severity_fields(b2ai_root, disease):
    f = b2ai_root / "phenotype" / "diagnosis" / DIAG_FILES.get(disease, "_")
    if not f.exists():
        return pd.DataFrame()
    t = pd.read_csv(f, sep="\t", low_memory=False, dtype={"participant_id": str})
    t["_pid"] = t["participant_id"].str.lstrip("0")
    num = {}
    for c in t.columns:
        if c in ("participant_id", "_pid") or "___" in c:
            continue
        v = pd.to_numeric(t[c], errors="coerce")
        if v.notna().mean() >= 0.3 and v.nunique() >= 3:
            num[c] = v
    if not num:
        return pd.DataFrame()
    return pd.DataFrame(num).assign(_pid=t["_pid"]).groupby("_pid").max()


def participant_vectors(data, rows):
    """Mean speech vector per (participant, session) using all 470-D reps."""
    out = {}
    for (pid, sid), g in rows.groupby(["participant_id", "session_id"]):
        X, kept = C.speech_matrix(g, [pid], data.lookups)
        if kept:
            out[(pid, sid)] = X[0]
    return out


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e4_1_longitudinal_vs_quantity"))
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root)
    m, meta = data.manifest, data.participant_meta
    nses = m.groupby("participant_id")["session_id"].nunique().rename("n_sessions")

    # A
    a = meta.join(nses)
    a["multi_session"] = a["n_sessions"] >= 2
    C.save(a.groupby(["cohort", "country"], dropna=False).agg(
        n=("n_sessions", "size"), mean_sessions=("n_sessions", "mean"),
        share_multi_session=("multi_session", "mean")).reset_index(), ns.out_dir, "sessions_by_cohort_country.csv")

    # B
    sev_rows = []
    for d in ns.diseases:
        sev = severity_fields(data.b2ai_root, d)
        ids = [i for i in meta.index if meta.loc[i, "cohort"] == d]
        for c in sev.columns:
            v = sev[c].reindex([i.lstrip("0") for i in ids]).to_numpy(float)
            n = nses.reindex(ids).to_numpy(float)
            ok = np.isfinite(v) & np.isfinite(n)
            if ok.sum() >= 10 and np.nanstd(n[ok]) > 0:
                r = spearmanr(n[ok], v[ok])
                sev_rows.append({"disease": d, "severity_field": c, "n": int(ok.sum()),
                                 "spearman_rho_sessions": r.correlation, "p": r.pvalue})
    C.save(sev_rows, ns.out_dir, "sessions_vs_severity.csv")

    # C
    rows = m[m["task_family"].isin(C.MENU) & m["_has_all_reps"]]
    multi = nses.index[nses >= 2]
    vec = participant_vectors(data, rows[rows["participant_id"].isin(multi)])
    allv = np.stack(list(vec.values())) if vec else np.empty((0, 470))
    mu, sd = np.nanmean(allv, 0), np.nanstd(allv, 0) + 1e-9
    ch = []
    for pid in multi:
        ks = [k for k in vec if k[0] == pid]
        if len(ks) < 2:
            continue
        Z = (np.stack([vec[k] for k in ks]) - mu) / sd
        dists = [np.linalg.norm(Z[i] - Z[j]) / np.sqrt(Z.shape[1]) for i in range(len(Z)) for j in range(i + 1, len(Z))]
        # within-session noise floor: split each session's recordings in two halves
        floor = []
        for k in ks:
            g = rows[(rows["participant_id"] == pid) & (rows["session_id"] == k[1])].sample(frac=1, random_state=0)
            h = len(g) // 2
            if h >= 3:
                X1, k1 = C.speech_matrix(g.iloc[:h], [pid], data.lookups)
                X2, k2 = C.speech_matrix(g.iloc[h:], [pid], data.lookups)
                if k1 and k2:
                    floor.append(np.linalg.norm((X1[0] - X2[0]) / sd) / np.sqrt(len(sd)))
        ch.append({"participant_id": pid, "cohort": data.cohort_by_pid.get(pid), "n_sessions": len(ks),
                   "country": meta.loc[pid, "country"] if pid in meta.index else None,
                   "cross_session_dist": float(np.mean(dists)),
                   "within_session_split_dist": float(np.mean(floor)) if floor else np.nan})
    ch = pd.DataFrame(ch)
    if not ch.empty:
        ch["excess_over_floor"] = ch["cross_session_dist"] - ch["within_session_split_dist"]
        C.save(ch, ns.out_dir, "cross_session_change.csv")
        tests = []
        ref = ch[~ch["cohort"].isin(["Parkinson's Disease", "Cognitive Impairment"])]["excess_over_floor"].dropna()
        for d in ["Parkinson's Disease", "Cognitive Impairment"]:
            x = ch[ch["cohort"] == d]["excess_over_floor"].dropna()
            if len(x) >= 3 and len(ref) >= 3:
                u = mannwhitneyu(x, ref)
                tests.append({"cohort": d, "n": len(x), "median_excess": x.median(), "ref_n": len(ref),
                              "ref_median_excess": ref.median(), "mannwhitney_p": u.pvalue})
        C.save(tests, ns.out_dir, "cross_session_change_tests.csv")

    # D: single-session standardized cohort, speech vs speech + n_sessions (true count)
    def one(d):
        co = C.standardized_cohort(data, d, args)
        sm = C.v7.qualifying_sessions(co.source, co.ids, co.menu, budget=1)  # as in v8 single-session
        ok_pairs = [(c, h) for c, h in co.final_pairs if sm.get(c) and sm.get(h)]
        if len(ok_pairs) < args.exploratory_min_pairs:
            return {"disease": d, "error": f"only {len(ok_pairs)} single-session pairs"}
        ids = C.v6.flatten_pairs(ok_pairs)
        co1 = C.Cohort(d, co.matched_pairs, ok_pairs, co.budget, co.tier, co.source, ids,
                       C.pb.make_labels(ids, d, data.cohort_by_pid), C.v7.pair_aware_folds(ok_pairs, ids, args.seed),
                       co.menu, co.caliper_years)
        sampler = lambda cohort, seed: C.v7.sample_single_session_taskwise(  # noqa: E731
            cohort.source, cohort.ids, cohort.menu, sm, seed + 10000, budget=1)
        cnt = lambda r, i: nses.reindex([str(x) for x in i]).to_numpy(float)[:, None]  # noqa: E731
        preds = C.standardized_predictions(data, co1, args, {
            "speech": C.speech_fn(data), "n_sessions_only": cnt,
            "speech_plus_n_sessions": C.concat_fns(C.speech_fn(data), cnt)}, sampler=sampler)
        b = C.pair_bootstrap(preds, ok_pairs, args.bootstrap, args.seed + 11001,
                             deltas=[("speech_plus_n_sessions", "speech")])
        return {"disease": d, "pairs": len(ok_pairs), **{k: v for k, v in b.items() if k not in ("n", "n_pairs")}}

    C.save(C.run_each(ns.diseases, one), ns.out_dir, "single_session_plus_count.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

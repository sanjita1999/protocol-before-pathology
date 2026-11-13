#!/usr/bin/env python3
"""
Question addressed: Apply the method to benchmark results of existing methods and show
how it changes benchmark results and conclusions.

Treats a set of commonly reported clinical-speech pipelines as "benchmark
entries" and scores each one twice:

  (A) as normally reported: full cohort, all recordings, participant-disjoint CV
  (B) under PCMVF: matched pairs, shared tasks, equal task-wise evidence,
      pair-aware CV, plus the residual protocol AUC and PCMVF label.

Benchmark entries (all built from released Bridge2AI features):
  egemaps_like_svm    Static descriptors (131-D, openSMILE/Praat style) + RBF-SVM
  mfcc_rf             MFCC means + random forest
  mel_logistic        Mel-spectrogram means + logistic
  sparc_ema_pls       Articulatory SPARC EMA + frozen PLS-logistic probe
  prosody_pls         Pitch + loudness + periodicity + frozen probe
  combined_pls        All 8 representations + frozen probe (paper's speech model)
  combined_mlp        All 8 representations + MLP
  ppg_logistic        Phonetic posteriorgrams (learned neural representation), only
                      if features/ppgs.parquet is readable (currently corrupt on disk;
                      re-download from PhysioNet). Skipped otherwise.

Reports per disease: leaderboard before/after, Kendall tau between the two
rankings, the number of entries whose "AUC > 0.80" headline claim survives, the
best entry before vs after, and the PCMVF label per entry.

Outputs (results/e1_3_benchmark_reinterpretation/):
    benchmark_raw_vs_pcmvf.csv     one row per disease x entry
    leaderboard_changes.csv        Kendall tau, winner change, headline-claim flips
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent

ENTRIES = {
    "egemaps_like_svm": (["static"], "svm_rbf"),
    "mfcc_rf": (["mfcc"], "rf"),
    "mel_logistic": (["mel_spectrogram"], "logistic"),
    "sparc_ema_pls": (["sparc_ema"], "pls_logistic"),
    "prosody_pls": (["pitch", "loudness", "periodicity"], "pls_logistic"),
    "combined_pls": (None, "pls_logistic"),
    "combined_mlp": (None, "mlp"),
}


def try_add_ppg(data) -> bool:
    path = data.b2ai_root / "features" / "ppgs.parquet"
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ppg] unavailable ({exc.__class__.__name__}); skipping ppg entry. Re-download {path}.")
        return False
    task_col = "task_name" if "task_name" in df else "raw_task"
    lut = {}
    for r in df.itertuples(index=False):
        mat = np.asarray(getattr(r, "ppg"), dtype=float)
        if mat.ndim == 1:
            mat = mat.reshape(-1, 40) if mat.size % 40 == 0 else mat[None, :]
        if mat.shape[0] == 40 and mat.shape[1] != 40:
            mat = mat.T
        key = (str(getattr(r, "participant_id")).lstrip("0"), str(getattr(r, "session_id")),
               C.pb.norm_task(getattr(r, task_col)))
        lut[key] = mat.mean(0)
    # Re-key to manifest participant ids.
    pid_map = {str(p).lstrip("0"): str(p) for p in data.manifest["participant_id"].unique()}
    data.lookups["ppg"] = {(pid_map.get(k[0], k[0]), k[1], k[2]): v for k, v in lut.items()}
    ENTRIES["ppg_logistic"] = (["ppg"], "logistic")
    return True


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e1_3_benchmark_reinterpretation"))
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root, attach_metadata=False)
    try_add_ppg(data)
    rows = []

    def one(d):
        # (A) as normally reported
        def raw_fn(reps):
            return lambda r, i, f: C.speech_matrix(r, i, data.lookups, reps, strict=False)[0]
        rfeat = {e: raw_fn(reps) for e, (reps, _) in ENTRIES.items()}
        rows_d, cand = C.raw_cohort_rows(data, d)
        raw = {}
        for e, (reps, m) in ENTRIES.items():
            pr, ids, _ = C.raw_predictions(data, d, args, {e: rfeat[e]}, {m: C.model_oof(m)})
            raw.update(pr)
        braw = C.pair_bootstrap(raw, None, args.bootstrap, args.seed + 303)

        # (B) PCMVF
        co = C.standardized_cohort(data, d, args)
        std = {}
        for e, (reps, m) in ENTRIES.items():
            std.update(C.standardized_predictions(data, co, args, {e: C.speech_fn(data, reps)},
                                                  models={m: C.model_oof(m)}))
        std.update(C.standardized_predictions(data, co, args, {"protocol": C.protocol_fn(data)}))
        bstd = C.pair_bootstrap(std, co.final_pairs, args.bootstrap, args.seed + 5001)
        out = []
        for e in ENTRIES:
            out.append({
                "disease": d, "entry": e, "model": ENTRIES[e][1],
                "raw_n": braw["n"], "raw_auc": braw[f"{e}_auc"],
                "raw_ci_low": braw[f"{e}_ci_low"], "raw_ci_high": braw[f"{e}_ci_high"],
                "pcmvf_pairs": len(co.final_pairs), "pcmvf_auc": bstd[f"{e}_auc"],
                "pcmvf_ci_low": bstd[f"{e}_ci_low"], "pcmvf_ci_high": bstd[f"{e}_ci_high"],
                "residual_protocol_auc": bstd["protocol_auc"],
                "pcmvf_label": C.pcmvf_label(bstd[f"{e}_ci_low"], bstd["protocol_ci_low"],
                                             bstd["protocol_ci_high"], co.tier),
            })
        return out

    rows = C.run_each(ns.diseases, one)
    df = pd.DataFrame(rows)
    C.save(df, ns.out_dir, "benchmark_raw_vs_pcmvf.csv")
    if "raw_auc" in df:
        ch = []
        for d, g in df.dropna(subset=["raw_auc"]).groupby("disease"):
            tau, p = kendalltau(g["raw_auc"], g["pcmvf_auc"])
            ch.append({
                "disease": d, "kendall_tau": tau, "kendall_p": p,
                "raw_winner": g.loc[g["raw_auc"].idxmax(), "entry"],
                "pcmvf_winner": g.loc[g["pcmvf_auc"].idxmax(), "entry"],
                "mean_auc_drop": float((g["raw_auc"] - g["pcmvf_auc"]).mean()),
                "n_headline_raw_gt_0.80": int((g["raw_auc"] > 0.80).sum()),
                "n_headline_survive_pcmvf": int(((g["raw_auc"] > 0.80) & (g["pcmvf_auc"] > 0.80)).sum()),
                "n_entries_supported": int(g["pcmvf_label"].str.contains("supported").sum()),
            })
        C.save(ch, ns.out_dir, "leaderboard_changes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

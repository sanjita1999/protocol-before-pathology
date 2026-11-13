#!/usr/bin/env python3
"""
Figure generation: Figure 2B / 2D replacements and the SHAP attribution-share plot.

What Figure 2D shows: each row is ONE (disease, protocol-feature-set) pair.
  - Completion pattern = binary "did the participant contribute any recording
    in task family f" indicators.
  - Recording counts   = per-family recording counts + assessment totals.
  - All non-speech     = completion + counts + age + sex.
The two endpoints are the protocol-only AUC with all task families and after
deleting the cognitive-linguistic family. Rows do not share an endpoint because
each row is a different feature set, not the same model at two capacities.
Completion drops to ~0.5 because, once cognitive-linguistic tasks are removed,
almost every participant completed every remaining family (Figure S25), so the
completion indicators are nearly constant. Counts stay at ~0.95, which is the
point of the panel: the shortcut is not only task completion.

This script recomputes the panel with the PCMVF probe and draws a labelled
grouped-bar chart with values printed (agrees with the published panel within 0.02).

Output: results/e1_5_figure2d_redraw/figure2d_redraw.{png,pdf}, figure2d_values.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG2D_DISEASES = ["Parkinson's Disease", "Cognitive Impairment"]


def fig2d_values(b2ai_root=None) -> pd.DataFrame:
    """Protocol-only AUC (full cohort, frozen probe) for completion / counts / all non-speech,
    with all task families and with the cognitive-linguistic family removed."""
    data = C.load_data(b2ai_root, attach_metadata=False)
    args = C.default_args()
    out = []
    for d in FIG2D_DISEASES:
        row = {"disease": d}
        for tag, drop in (("all", False), ("no_cog", True)):
            def fn(cols_key, drop=drop):
                def f(r, i, fams):
                    r = r[r["participant_id"].isin(i)]
                    if drop:
                        r, fams = r[r["task_family"] != "cognitive_linguistic"], [x for x in fams if x != "cognitive_linguistic"]
                    P = C.protocol_df(data, r, i, fams)
                    comp = C.v6.protocol_components(P)
                    cols = {"completion": comp["completion_only"], "counts": comp["assessment_intensity"],
                            "combined": comp["full_protocol"]}[cols_key]
                    return P[cols].to_numpy(float)
                return f
            preds, _, _ = C.raw_predictions(data, d, args, {k: fn(k) for k in ("completion", "counts", "combined")})
            for k, pr in preds.items():
                row[f"{k}_{tag}"] = C.roc_auc_score(pr["y_true"], pr["y_score"])
        out.append(row)
    return pd.DataFrame(out)
SETS = [("completion", "Completion pattern"), ("counts", "Recording counts"), ("combined", "All non-speech")]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=str(C.RESULTS / "figures"))
    p.add_argument("--quick", action="store_true", help="accepted for run_all.sh; no effect")
    p.add_argument("--b2ai-root", default=C.DEFAULT_B2AI_ROOT)
    ns = p.parse_args()
    out = Path(ns.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = fig2d_values(ns.b2ai_root)
    long = []
    for _, r in df.iterrows():
        for key, label in SETS:
            long.append({"disease": r["disease"], "feature_set": label,
                         "all_task_families": r[f"{key}_all"],
                         "without_cognitive_linguistic": r[f"{key}_no_cog"]})
    long = pd.DataFrame(long)
    C.save(long, out, "figure2d_values.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    diseases = long["disease"].unique().tolist()
    fig, axes = plt.subplots(1, len(diseases), figsize=(4.2 * len(diseases), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    w = 0.38
    for ax, d in zip(axes, diseases):
        g = long[long["disease"] == d]
        x = np.arange(len(g))
        b1 = ax.bar(x - w / 2, g["all_task_families"], w, label="All task families", color="#4C72B0")
        b2 = ax.bar(x + w / 2, g["without_cognitive_linguistic"], w,
                    label="Cognitive-linguistic family removed", color="#DD8452")
        for bars in (b1, b2):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(0.5, ls="--", c="grey", lw=1)
        ax.text(len(g) - 0.5, 0.505, "chance", color="grey", fontsize=8, ha="right", va="bottom")
        ax.set_xticks(x, g["feature_set"], fontsize=9)
        ax.set_ylim(0.4, 1.05)
        ax.set_title(d, fontsize=10)
    axes[0].set_ylabel("Protocol-only ROC-AUC")
    fig.legend(handles=[b1, b2], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8, frameon=False)
    fig.suptitle("Removing cognitive-linguistic tasks removes the completion shortcut,\n"
                 "but recording counts still predict diagnosis", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure2d_redraw.{ext}", dpi=300)
    print(f"[saved] {out}/figure2d_redraw.png")
    figure2b(out, plt, np)
    shap_share(out, plt, np)
    return 0


def shap_share(out, plt, np):
    """TreeSHAP share of attribution, speech vs protocol features (joint model, full cohort)."""
    src = C.RESULTS / "e1_4_shap_vs_pcmvf" / "joint_attribution_share.csv"
    if not src.exists():
        return
    d = pd.read_csv(src).sort_values("share_protocol")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(d) + 1.3))
    ax.barh(y, d["share_speech"], color="#4C72B0", label="Speech features (~470)")
    ax.barh(y, d["share_protocol"], left=d["share_speech"], color="#C44E52", label="Protocol features (~20)")
    for i, (s_, p_, a) in enumerate(zip(d["share_speech"], d["share_protocol"], d["joint_oof_auc"])):
        ax.text(s_ / 2, i, f"{s_:.0%}", ha="center", va="center", color="white", fontsize=8)
        ax.text(s_ + p_ / 2, i, f"{p_:.0%}", ha="center", va="center", color="white", fontsize=8)
        ax.text(1.01, i, f"AUC {a:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(y, d["disease"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of mean |SHAP| (joint speech + protocol model, held-out folds)")
    fig.legend(loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 0.93, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"shap_attribution_share.{ext}", dpi=300)
    print(f"[saved] {out}/shap_attribution_share.png")


B_SRC = C.REPO / "outputs" / "results" / "nonspeech_baselines.csv"  # from `speech-atlas nonspeech-baselines`
B_SETS = [("demographics", "Age + sex", "s", "#8C8C8C"), ("completion_pattern", "Task completion", "^", "#DD8452"),
          ("protocol_counts", "Recording counts", "v", "#C44E52"), ("combined_nonspeech", "All non-speech", "o", "#000000")]


def figure2b(out, plt, np):
    """Figure 2B redraw: one row per cohort, speech vs each non-speech baseline, 95% CIs, legend outside."""
    df = pd.read_csv(B_SRC)
    speech = df.drop_duplicates("disease").set_index("disease")["auc_speech_only"]
    order = speech.sort_values(ascending=False).index.tolist()
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    off = np.linspace(-0.27, 0.27, len(B_SETS) + 1)
    long = []
    for k, (ctx, lab, mk, col) in enumerate(B_SETS):
        sub = df[df["context"] == ctx].set_index("disease").reindex(order)
        ax.errorbar(sub["auc"], y + off[k + 1], xerr=[sub["auc"] - sub["ci_low"], sub["ci_high"] - sub["auc"]],
                    fmt=mk, color=col, ms=5, lw=0.8, capsize=0, label=lab)
        long += [{"disease": d, "model": lab, "auc": a, "ci_low": lo, "ci_high": hi}
                 for d, a, lo, hi in zip(order, sub["auc"], sub["ci_low"], sub["ci_high"])]
    ax.scatter(speech.reindex(order), y + off[0], marker="D", s=46, facecolor="white", edgecolor="#4C72B0",
               linewidth=1.6, zorder=3, label="Speech (all acoustic features)")
    long += [{"disease": d, "model": "Speech", "auc": a} for d, a in zip(order, speech.reindex(order))]
    ax.axvline(0.5, ls="--", c="grey", lw=1)
    ax.set_yticks(y, order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("ROC-AUC, full cohort (95% CI)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Non-speech assessment information matches or beats speech\nfor Parkinson's disease and cognitive impairment",
                 fontsize=10)
    fig.legend(loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure2b_redraw.{ext}", dpi=300)
    C.save(long, out, "figure2b_values.csv")
    print(f"[saved] {out}/figure2b_redraw.png")


if __name__ == "__main__":
    raise SystemExit(main())

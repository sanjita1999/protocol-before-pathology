from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import output_dir
from .utils import ensure_dir


def _save(fig, path: Path):
    ensure_dir(path.parent)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1_framework(cfg: dict[str, Any]):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    labels = [
        (1.2, 2.5, "Disease\ncohort"),
        (3.4, 2.5, "Elicitation\ntask"),
        (5.6, 2.5, "Speech\nrepresentation"),
        (7.8, 2.5, "Context\n(age / sex / site / budget)"),
        (10.3, 2.5, "Fixed leakage-\ncontrolled probe"),
    ]
    for x, y, text in labels:
        ax.text(x, y, text, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fill=False))
    for x1, x2 in [(2.0,2.6),(4.2,4.8),(6.4,7.0),(8.8,9.3)]:
        ax.annotate("", xy=(x2,2.5), xytext=(x1,2.5), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(11.4,2.5), xytext=(11.0,2.5), arrowprops=dict(arrowstyle="->"))
    ax.text(11.7, 2.5, "AUC +\nuncertainty", ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fill=False))
    ax.text(6.1, 0.8, "Stress tests: common task • separate covariate adjustment • equal observation budget • bootstrap rank stability", ha="center", va="center")
    _save(fig, output_dir(cfg, "figures") / "fig1_conditional_detectability_framework")


def fig2_context(cfg: dict[str, Any]):
    path = output_dir(cfg, "results") / "adjustments.csv"
    df = pd.read_csv(path)
    contexts = [c for c in ["raw","age","sex","age_sex","site","joint"] if c in set(df.context)]
    diseases = df[df.context.eq("raw")].sort_values("auc")["disease"].tolist()
    fig, ax = plt.subplots(figsize=(8.5, max(5, len(diseases)*0.55)))
    offsets = np.linspace(-0.22, 0.22, len(contexts))
    for j, c in enumerate(contexts):
        s = df[df.context.eq(c)].set_index("disease").reindex(diseases)
        y = np.arange(len(diseases)) + offsets[j]
        xerr = np.vstack([s.auc - s.ci_low, s.ci_high - s.auc])
        ax.errorbar(s.auc, y, xerr=xerr, fmt="o", capsize=2, label=c)
    ax.axvline(0.5, linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(diseases)), diseases)
    ax.set_xlabel("Participant-level ROC-AUC")
    ax.set_title("Detectability under demographic adjustment")
    ax.legend(frameon=False, ncol=min(len(contexts),3))
    _save(fig, output_dir(cfg, "figures") / "fig2_context_sensitivity")


def _heatmap(df: pd.DataFrame, row: str, col: str, value: str, title: str, path: Path):
    pivot = df.pivot(index=row, columns=col, values=value)
    fig, ax = plt.subplots(figsize=(max(8, 1.0*len(pivot.columns)), max(5, 0.6*len(pivot.index))))
    arr = pivot.to_numpy(float)
    im = ax.imshow(arr, aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i,j]):
                ax.text(j, i, f"{arr[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="ROC-AUC")
    _save(fig, path)


def fig3_task(cfg: dict[str, Any]):
    df = pd.read_csv(output_dir(cfg, "results") / "disease_task.csv")
    _heatmap(df, "disease", "task_family", "auc", "Disease × task-family detectability atlas", output_dir(cfg, "figures") / "fig3_disease_task")


def fig4_representation(cfg: dict[str, Any]):
    df = pd.read_csv(output_dir(cfg, "results") / "disease_representation.csv")
    _heatmap(df, "disease", "representation", "auc", "Disease × representation detectability atlas", output_dir(cfg, "figures") / "fig4_disease_representation")


def fig5_ranking(cfg: dict[str, Any]):
    rank = pd.read_csv(output_dir(cfg, "results") / "rank_summary.csv").sort_values("median_rank")
    common_path = output_dir(cfg, "results") / "common_task.csv"
    budget_path = output_dir(cfg, "results") / "equal_budget.csv"
    fig, ax = plt.subplots(figsize=(9, max(5, 0.6*len(rank))))
    y = np.arange(len(rank))
    xerr = np.vstack([rank.median_rank-rank.rank_ci_low, rank.rank_ci_high-rank.median_rank])
    ax.errorbar(rank.median_rank, y, xerr=xerr, fmt="o", capsize=3, label="bootstrap rank")
    if common_path.exists():
        common = pd.read_csv(common_path).set_index("disease")
        xs=[]; ys=[]
        for i,d in enumerate(rank.disease):
            if d in common.index and "rank_common" in common.columns:
                xs.append(float(common.loc[d,"rank_common"])); ys.append(i)
        if xs:
            ax.scatter(xs, ys, marker="x", label="common-task rank")
    if budget_path.exists():
        b = pd.read_csv(budget_path)
        bp = b[b.is_primary_budget.astype(str).str.lower().isin(["true","1"])] if "is_primary_budget" in b else pd.DataFrame()
        if not bp.empty:
            bp = bp.copy(); bp["rank_budget"] = bp.auc_mean.rank(ascending=False, method="min")
            bm = bp.set_index("disease")
            xs=[]; ys=[]
            for i,d in enumerate(rank.disease):
                if d in bm.index:
                    xs.append(float(bm.loc[d,"rank_budget"])); ys.append(i)
            if xs:
                ax.scatter(xs, ys, marker="+", label="equal-budget rank")
    ax.set_yticks(y, rank.disease)
    ax.invert_yaxis()
    ax.set_xlabel("Rank (1 = highest point estimate)")
    ax.set_title("Ranking stability under sampling and controlled measurement")
    ax.legend(frameon=False)
    _save(fig, output_dir(cfg, "figures") / "fig5_rank_stability")


def make_all_figures(cfg: dict[str, Any]):
    fig1_framework(cfg)
    if (output_dir(cfg,"results")/"adjustments.csv").exists(): fig2_context(cfg)
    if (output_dir(cfg,"results")/"disease_task.csv").exists(): fig3_task(cfg)
    if (output_dir(cfg,"results")/"disease_representation.csv").exists(): fig4_representation(cfg)
    if (output_dir(cfg,"results")/"rank_summary.csv").exists(): fig5_ranking(cfg)


def _latex_escape(s: str) -> str:
    return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def dataframe_to_latex(df: pd.DataFrame, path: Path, caption: str, label: str):
    ensure_dir(path.parent)
    # pandas handles most escaping; keep index false for publication tables.
    tex = df.to_latex(index=False, escape=True, caption=caption, label=label, float_format=lambda x: f"{x:.3f}")
    path.write_text(tex, encoding="utf-8")


def make_tables(cfg: dict[str, Any]):
    out = output_dir(cfg, "tables"); ensure_dir(out)
    r = output_dir(cfg, "results")
    mappings = [
        ("disease_benchmark.csv", "table_disease_benchmark.tex", "Pooled-task disease benchmark under the fixed subject-level probe.", "tab:disease_benchmark"),
        ("common_task.csv", "table_common_task.tex", "Common-task controlled benchmark.", "tab:common_task"),
        ("adjustments.csv", "table_adjustments.tex", "Raw and covariate-adjusted detectability.", "tab:adjustments"),
        ("equal_budget.csv", "table_equal_budget.tex", "Equal-recording sensitivity analysis.", "tab:equal_budget"),
        ("rank_summary.csv", "table_rank_summary.tex", "Bootstrap disease-rank stability.", "tab:rank_summary"),
    ]
    for csv_name, tex_name, caption, label in mappings:
        p = r/csv_name
        if p.exists():
            df = pd.read_csv(p)
            df.to_csv(out/csv_name, index=False)
            dataframe_to_latex(df, out/tex_name, caption, label)



# Acceptance-upgrade figures. These are separate from the legacy figures so
# manuscript revisions can compare both versions without overwriting artifacts.
def _pretty_disease(x: str) -> str:
    return {
        "Unilateral Vocal Fold Paralysis": "UVFP",
        "Unexplained Chronic Cough": "UCC",
        "Muscle Tension Dysphonia": "MTD",
    }.get(str(x), str(x))


def _primary_order(cfg: dict[str, Any], eligible_only: bool = False) -> list[str]:
    order = (
        pd.read_csv(output_dir(cfg, "results") / "disease_benchmark.csv")
        .sort_values("auc", ascending=False).disease.tolist()
    )
    if eligible_only:
        order = [x for x in order if x != "Precancerous Lesions"]
    return order


def figA1_hero(cfg: dict[str, Any]):
    bench = pd.read_csv(output_dir(cfg, "results") / "disease_benchmark.csv")
    task = pd.read_csv(output_dir(cfg, "results") / "disease_task.csv")
    common = pd.read_csv(output_dir(cfg, "results") / "common_task.csv")
    order = _primary_order(cfg)
    eligible = [d for d in order if d in set(common.disease)]
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], wspace=.42, hspace=.48)

    ax = fig.add_subplot(gs[0, 0])
    b = bench.set_index("disease").reindex(order)
    y = np.arange(len(order))
    xerr = np.vstack([b.auc - b.ci_low, b.ci_high - b.auc])
    ax.errorbar(b.auc, y, xerr=xerr, fmt="o", capsize=2.5, color="#1f4e79", ecolor="#8aa8bf")
    ax.axvline(.5, color="0.75", ls="--", lw=1)
    ax.set_yticks(y, [_pretty_disease(x) for x in order])
    ax.invert_yaxis(); ax.set_xlim(.5, 1.0); ax.set_xlabel("Participant-level ROC-AUC")
    ax.set_title("A. Pooled evaluation creates an apparent hierarchy", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    tv = task.groupby("disease").auc.agg(["min", "max"]).reindex(eligible)
    yy = np.arange(len(tv))
    ax.hlines(yy, tv["min"], tv["max"], color="#d4a72c", lw=4, alpha=.75)
    ax.scatter(tv["min"], yy, color="#7a5195", s=32, zorder=3)
    ax.scatter(tv["max"], yy, color="#ef5675", s=32, zorder=3)
    ax.set_yticks(yy, [_pretty_disease(x) for x in tv.index])
    ax.invert_yaxis(); ax.set_xlim(.5, 1.0); ax.set_xlabel("ROC-AUC across task families")
    ax.set_title("B. Task choice shifts AUC by up to 0.28", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[1, :])
    cm = common.set_index("disease").reindex(eligible)
    pooled = bench.set_index("disease").reindex(eligible).auc
    shared = cm.auc
    colors = plt.cm.tab10(np.linspace(0, 1, len(eligible)))
    for i, disease in enumerate(eligible):
        ax.plot([0, 1], [pooled.loc[disease], shared.loc[disease]], marker="o", lw=2,
                color=colors[i], label=_pretty_disease(disease))
        if "paired_delta_ci_low" in cm.columns and (
            cm.loc[disease, "paired_delta_ci_high"] < 0
            or cm.loc[disease, "paired_delta_ci_low"] > 0
        ):
            ax.text(1.03, shared.loc[disease], f"Δ={cm.loc[disease, 'paired_delta_auc']:+.03f}",
                    va="center", fontsize=8, color=colors[i])
    ax.set_xticks([0, 1], ["Pooled", "Shared task-family set"])
    ax.set_ylabel("Participant-level ROC-AUC"); ax.set_ylim(.55, 1.0); ax.set_xlim(-.08, 1.20)
    ax.set_title("C. Standardizing the task-family menu selectively attenuates top cohorts",
                 loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower left")
    fig.suptitle("Clinical-speech detectability changes with the measurement protocol",
                 fontsize=15, fontweight="bold", y=.99)
    _save(fig, output_dir(cfg, "figures") / "figA1_hero_conditionality")


def figA2_task_atlas(cfg: dict[str, Any]):
    df = pd.read_csv(output_dir(cfg, "results") / "disease_task.csv")
    order = [d for d in _primary_order(cfg) if d in set(df.disease)]
    task_order = [
        "narrative_memory", "pitch_modulation", "rapid_articulation", "respiratory_cough",
        "spontaneous_speech", "structured_reading", "sustained_phonation",
    ]
    piv = df.pivot(index="disease", columns="task_family", values="auc").reindex(
        index=order, columns=task_order
    )
    ranges = piv.max(axis=1) - piv.min(axis=1)
    fig = plt.figure(figsize=(13, 6.4)); gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.15], wspace=.12)
    ax = fig.add_subplot(gs[0, 0]); arr = piv.to_numpy(float)
    im = ax.imshow(arr, aspect="auto", vmin=.5, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(task_order)), [x.replace("_", " ").title() for x in task_order], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(order)), [_pretty_disease(x) for x in order])
    for i in range(len(order)):
        jmax = int(np.nanargmax(arr[i])); jmin = int(np.nanargmin(arr[i]))
        ax.scatter(jmax, i, s=70, facecolors="none", edgecolors="white", linewidths=1.5)
        ax.scatter(jmin, i, s=55, facecolors="none", edgecolors="black", linewidths=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=.032, pad=.02); cb.set_label("Participant-level ROC-AUC")
    ax.set_title("Task choice changes detectability within the same disease", loc="left", fontweight="bold")
    ax2 = fig.add_subplot(gs[0, 1]); yy = np.arange(len(order))
    ax2.barh(yy, ranges.reindex(order), color="#d4a72c"); ax2.set_yticks(yy, []); ax2.invert_yaxis()
    ax2.set_xlabel("AUC range"); ax2.set_xlim(0, max(.30, float(ranges.max()) * 1.08)); ax2.set_title("Task sensitivity", fontsize=10)
    for yv, value in zip(yy, ranges.reindex(order)):
        ax2.text(value + .006, yv, f"{value:.3f}", va="center", fontsize=8)
    _save(fig, output_dir(cfg, "figures") / "figA2_task_atlas_with_sensitivity")


def figA3_validity_audit(cfg: dict[str, Any]):
    nons = pd.read_csv(output_dir(cfg, "results") / "nonspeech_baselines.csv")
    combo = pd.read_csv(output_dir(cfg, "results") / "speech_plus_demographics.csv")
    bench = pd.read_csv(output_dir(cfg, "results") / "disease_benchmark.csv")
    order = _primary_order(cfg)
    ns = nons[nons.context.eq("combined_nonspeech")].set_index("disease")
    demo = nons[nons.context.eq("demographics")].set_index("disease")
    co = combo.set_index("disease"); b = bench.set_index("disease")
    fig, ax = plt.subplots(figsize=(9.5, 6.3)); y = np.arange(len(order))
    series = [
        ("Non-speech combined", ns, "#7a5195"),
        ("Age + sex only", demo, "#ffa600"),
        ("Speech only", b, "#1f77b4"),
        ("Speech + age + sex", co, "#2ca02c"),
    ]
    for offset, (label, table, color) in zip(np.linspace(-.24, .24, len(series)), series):
        ax.scatter(table.reindex(order).auc, y + offset, label=label, s=34, color=color)
    ax.axvline(.5, color="0.75", ls="--", lw=1); ax.set_yticks(y, [_pretty_disease(x) for x in order]); ax.invert_yaxis()
    ax.set_xlim(.45, 1.0); ax.set_xlabel("Participant-level ROC-AUC")
    ax.set_title("How much disease separability exists without acoustics?", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    _save(fig, output_dir(cfg, "figures") / "figA3_nonspeech_validity_audit")


def figA4_representation_stability(cfg: dict[str, Any]):
    df = pd.read_csv(output_dir(cfg, "results") / "disease_representation.csv")
    fixed = pd.read_csv(output_dir(cfg, "results") / "ablation_fixed_pls.csv")
    order = _primary_order(cfg)
    rep_order = ["sparc_ema", "static", "spectrogram", "mfcc", "mel_spectrogram", "periodicity", "loudness", "pitch"]
    piv = df.pivot(index="disease", columns="representation", values="auc").reindex(index=order, columns=rep_order)
    fig = plt.figure(figsize=(13, 6.6)); gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.8], wspace=.18)
    ax = fig.add_subplot(gs[0, 0]); arr = piv.to_numpy(); im = ax.imshow(arr, aspect="auto", vmin=.3, vmax=1, cmap="viridis")
    labels = [x.upper() if x == "mfcc" else x.replace("_", " ").title() for x in rep_order]
    ax.set_xticks(np.arange(len(rep_order)), labels, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(order)), [_pretty_disease(x) for x in order])
    fig.colorbar(im, ax=ax, fraction=.032, pad=.02, label="ROC-AUC")
    ax.set_title("No speech representation is uniformly optimal", loc="left", fontweight="bold")
    primary = df.loc[df.groupby("disease").auc.idxmax(), ["disease", "representation"]].set_index("disease")
    fwin = fixed.loc[fixed.groupby("disease").auc.idxmax(), ["disease", "representation"]].set_index("disease")
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_xlim(0, 1); ax2.set_ylim(-.7, len(order) - .3); ax2.invert_yaxis(); ax2.axis("off")
    ax2.text(.1, -.45, "Adaptive PLS", fontweight="bold"); ax2.text(.63, -.45, "Fixed PLS=2", fontweight="bold")
    for i, disease in enumerate(order):
        a = str(primary.loc[disease, "representation"]); b = str(fwin.loc[disease, "representation"])
        color = "#2ca02c" if a == b else "#d62728"
        ax2.plot([.28, .72], [i, i], color=color, alpha=.6, lw=1.5)
        ax2.text(.1, i, a.replace("_", " "), va="center", fontsize=8)
        ax2.text(.63, i, b.replace("_", " "), va="center", fontsize=8)
    ax2.set_title("Winner stability", fontsize=10)
    _save(fig, output_dir(cfg, "figures") / "figA4_representation_stability")


def figA5_protocol_uncertainty(cfg: dict[str, Any]):
    proto = pd.read_csv(output_dir(cfg, "results") / "protocol_intervals.csv")
    comp = pd.read_csv(output_dir(cfg, "results") / "gtheory_variance_components.csv")
    comp = comp[comp.scale.eq("logit")].copy(); order = [d for d in _primary_order(cfg) if d in set(proto.disease)]
    p = proto.set_index("disease").reindex(order)
    fig = plt.figure(figsize=(12.5, 6.8)); gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=.32)
    ax = fig.add_subplot(gs[0, 0]); y = np.arange(len(order))
    ax.hlines(y, p.protocol_interval_low, p.protocol_interval_high, color="#d4a72c", lw=5, label="95% empirical protocol interval")
    ax.hlines(y, p.sampling_ci_low, p.sampling_ci_high, color="#1f4e79", lw=1.5, label="95% participant-bootstrap CI")
    ax.scatter(p.pooled_auc, y, color="#1f4e79", s=28, zorder=3)
    ax.set_yticks(y, [_pretty_disease(x) for x in order]); ax.invert_yaxis(); ax.set_xlim(.25, 1.0); ax.set_xlabel("ROC-AUC")
    ax.set_title("Protocol uncertainty can exceed sampling uncertainty", loc="left", fontweight="bold"); ax.legend(frameon=False, fontsize=8)
    ax2 = fig.add_subplot(gs[0, 1]); c = comp.sort_values("proportion_total")
    ax2.barh(np.arange(len(c)), c.proportion_total, color="#5b8ff9")
    ax2.set_yticks(np.arange(len(c)), [x.replace("_", " × ") for x in c.component])
    ax2.set_xlabel("Fraction of total variance\n(logit-AUC scale)"); ax2.set_title("Crossed-facet variance decomposition", fontsize=10, fontweight="bold")
    for yv, value in enumerate(c.proportion_total):
        ax2.text(value + .005, yv, f"{100 * value:.1f}%", va="center", fontsize=8)
    _save(fig, output_dir(cfg, "figures") / "figA5_protocol_uncertainty_gstudy")


def figA6_rank_bump(cfg: dict[str, Any]):
    bench = pd.read_csv(output_dir(cfg, "results") / "disease_benchmark.csv")
    common = pd.read_csv(output_dir(cfg, "results") / "common_task.csv")
    budget = pd.read_csv(output_dir(cfg, "results") / "equal_budget.csv")
    adj = pd.read_csv(output_dir(cfg, "results") / "adjustments.csv")
    eligible = [d for d in _primary_order(cfg) if d != "Precancerous Lesions" and d in set(common.disease)]
    bp = budget[budget.is_primary_budget.astype(str).str.lower().isin(["true", "1"])]
    tables = {
        "Pooled": bench.set_index("disease").auc,
        "Shared task set": common.set_index("disease").auc,
        "20-observation budget": bp.set_index("disease").auc_mean,
        "Age+sex residualized": adj[adj.context.eq("age_sex")].set_index("disease").auc,
    }
    ranks = {name: series.reindex(eligible).rank(ascending=False, method="average") for name, series in tables.items()}
    fig, ax = plt.subplots(figsize=(10, 6.2)); xs = np.arange(len(ranks)); colors = plt.cm.tab10(np.linspace(0, 1, len(eligible)))
    for color, disease in zip(colors, eligible):
        ax.plot(xs, [ranks[name].loc[disease] for name in ranks], marker="o", lw=2, color=color, label=_pretty_disease(disease))
    ax.set_xticks(xs, list(ranks)); ax.set_ylabel("Rank (1 = highest)"); ax.set_yticks(range(1, len(eligible) + 1)); ax.invert_yaxis()
    ax.set_title("The apparent disease hierarchy moves when measurement conditions change", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    _save(fig, output_dir(cfg, "figures") / "figA6_rank_bump")


def make_acceptance_figures(cfg: dict[str, Any]):
    figA1_hero(cfg)
    figA2_task_atlas(cfg)
    figA3_validity_audit(cfg)
    figA4_representation_stability(cfg)
    figA5_protocol_uncertainty(cfg)
    figA6_rank_bump(cfg)

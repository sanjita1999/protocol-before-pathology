#!/usr/bin/env python3
"""
Question addressed: Task-family standardization is coarse: 628 raw labels -> 7
families; within-family prompt variability can exceed between-family
variability. A prompt-level sensitivity analysis would strengthen the claim.

 1. Heterogeneity: for each task family, ratio of within-family between-prompt
    variance to between-family variance of the participant-level speech vector
    (first PLS-free summary: mean of z-scored 470-D features), plus the share of
    each family's recordings coming from its most common prompt.
 2. Prompt-level PCMVF: identical pipeline, but the stratum used for the shared
    menu and the equal-evidence budget is the PROMPT instead of the family:
      --granularity prompt : raw label with the repetition suffix removed
                             (maximum-phonation-time-1/-2/-3 -> maximum-phonation-time)
      --granularity raw    : the released raw label itself
    The shared prompt menu = prompts that >= `--prompt-coverage` of the matched
    participants in BOTH groups contributed. Each participant then contributes
    the same number of recordings of each shared prompt.
 3. Compares speech / residual protocol AUC and the PCMVF label with the family
    analysis on the same disease.

Outputs (results/e2_3_prompt_level_standardization/):
    family_prompt_heterogeneity.csv, prompt_menus.csv, prompt_vs_family_pcmvf.csv
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import pcmvf_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent


def prompt_of(task_norm: str, granularity: str) -> str:
    if granularity == "raw":
        return task_norm
    return re.sub(r"[_-]\d+$", "", task_norm)


def heterogeneity(data):
    m = data.manifest[data.manifest["task_family"].isin(C.MENU) & data.manifest["_has_all_reps"]].copy()
    keys = list(zip(m["participant_id"], m["session_id"], m["task_norm"]))
    X = np.stack([np.concatenate([data.lookups[r][k] for r in C.ORIGINAL_REPS]) for k in keys])
    X = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9)
    m["score"] = np.nanmean(X, 1)
    m["prompt"] = m["task_norm"].map(lambda t: prompt_of(t, "prompt"))
    fam_mean = m.groupby("task_family")["score"].mean()
    between_family = float(fam_mean.var())
    rows = []
    for fam, g in m.groupby("task_family"):
        pm = g.groupby("prompt")["score"].mean()
        rows.append({
            "task_family": fam, "n_recordings": len(g), "n_prompts": g["prompt"].nunique(),
            "n_raw_labels": g["task_norm"].nunique(),
            "top_prompt": g["prompt"].value_counts().index[0],
            "top_prompt_share": float(g["prompt"].value_counts(normalize=True).iloc[0]),
            "within_family_prompt_var": float(pm.var()) if len(pm) > 1 else 0.0,
            "between_family_var": between_family,
            "ratio_within_to_between": float(pm.var() / between_family) if len(pm) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def prompt_manifest(data, granularity):
    m = data.manifest.copy()
    m = m[m["task_family"].isin(C.MENU)].copy()
    m["family_original"] = m["task_family"]
    m["task_family"] = m["task_norm"].map(lambda t: prompt_of(t, granularity))
    return m


def choose_prompt_menu(m, pairs, coverage):
    case = [c for c, _ in pairs]
    ctrl = [h for _, h in pairs]
    sub = m[m["_has_all_reps"]]
    menu = []
    for pr, g in sub.groupby("task_family"):
        have = set(g["participant_id"])
        if np.mean([c in have for c in case]) >= coverage and np.mean([h in have for h in ctrl]) >= coverage:
            menu.append(pr)
    return sorted(menu)


def family_prompt_matched(data, co, args):
    """
    Redesign (default): keep the 7 shared families, but inside every family each
    matched case and control get recordings of the SAME prompt(s), drawn from the
    prompts both members recorded. Pairs without a shared prompt in every family
    are dropped. Prompt composition is then identical within each pair.
    """
    src = co.source.assign(prompt=co.source["task_norm"].map(lambda t: prompt_of(t, "prompt")))
    g = {k: v for k, v in src.groupby(["participant_id", "task_family"])}
    shared = {}
    for c, h in co.final_pairs:
        sp = {}
        for fam in co.menu:
            a, b = g.get((c, fam)), g.get((h, fam))
            common = sorted(set(a["prompt"]) & set(b["prompt"])) if a is not None and b is not None else []
            if not common:
                break
            sp[fam] = common
        else:
            shared[(c, h)] = sp
    pairs = list(shared)
    if len(pairs) < args.exploratory_min_pairs:
        raise RuntimeError(f"only {len(pairs)} pairs share a prompt in every family")
    ids = C.v6.flatten_pairs(pairs)
    tier = v7_tier(len(pairs), args)
    cohort = C.Cohort(co.disease, co.matched_pairs, pairs, co.budget, tier, src, ids,
                      C.pb.make_labels(ids, co.disease, data.cohort_by_pid),
                      C.v7.pair_aware_folds(pairs, ids, args.seed), co.menu, co.caliper_years)

    def sampler(cohort, seed):
        rng = np.random.default_rng(seed)
        chunks = []
        for (c, h), sp in shared.items():
            for fam, prompts in sp.items():
                for pr in rng.choice(prompts, cohort.budget, replace=len(prompts) < cohort.budget):
                    for pid in (c, h):
                        x = g[(pid, fam)]
                        x = x[x["prompt"] == pr]
                        chunks.append(x.loc[[rng.choice(x.index.to_numpy())]])
        return pd.concat(chunks, ignore_index=True)

    n_shared = np.mean([len(v) for sp in shared.values() for v in sp.values()])
    return cohort, sampler, n_shared


def v7_tier(n, args):
    return C.v7.support_tier(n, args)


def row(data, co, args, d, setting, menu_size, sampler=None):
    preds = C.standardized_predictions(data, co, args, {"speech": C.speech_fn(data),
                                                        "protocol": C.protocol_fn(data, families=co.menu)},
                                       sampler=sampler)
    b = C.pair_bootstrap(preds, co.final_pairs, args.bootstrap, args.seed + 5001)
    return {"disease": d, "setting": setting, "menu_size": menu_size, "pairs": len(co.final_pairs),
            "budget": co.budget, "tier": co.tier, "speech_auc": b["speech_auc"],
            "speech_ci_low": b["speech_ci_low"], "speech_ci_high": b["speech_ci_high"],
            "protocol_auc": b["protocol_auc"], "protocol_ci_low": b["protocol_ci_low"],
            "protocol_ci_high": b["protocol_ci_high"],
            "pcmvf_label": C.pcmvf_label(b["speech_ci_low"], b["protocol_ci_low"], b["protocol_ci_high"], co.tier)}


def main() -> int:
    p = C.add_common_cli(argparse.ArgumentParser(description=__doc__),
                         str(C.RESULTS / "e2_3_prompt_level_standardization"))
    p.add_argument("--granularity", nargs="+", default=["family_matched"],
                   help="family_matched (default, redesign) and/or the old global menus: prompt raw")
    p.add_argument("--prompt-coverage", type=float, default=0.90)
    ns = C.finalize_cli(p.parse_args())
    args = C.args_from_cli(ns)
    data = C.load_data(ns.b2ai_root, attach_metadata=False)
    C.save(heterogeneity(data), ns.out_dir, "family_prompt_heterogeneity.csv")
    menus, results = [], []

    def one(d):
        co = C.standardized_cohort(data, d, args)
        results.append(row(data, co, args, d, "family (primary)", len(co.menu)))
        for gran in ns.granularity:
            if gran == "family_matched":
                try:
                    cop, sampler, n_shared = family_prompt_matched(data, co, args)
                    r = row(data, cop, args, d, "family + pair-matched prompts", len(co.menu), sampler)
                    r["mean_shared_prompts_per_family"] = n_shared
                    results.append(r)
                except Exception as exc:  # noqa: BLE001
                    results.append({"disease": d, "setting": "family + pair-matched prompts", "error": repr(exc)})
                continue
            man = prompt_manifest(data, gran)
            menu = choose_prompt_menu(man, co.matched_pairs, ns.prompt_coverage)
            fam_of = man.drop_duplicates("task_family").set_index("task_family")["family_original"]
            menus.append({"disease": d, "granularity": gran, "n_prompts": len(menu),
                          "families_covered": ";".join(sorted(set(fam_of.reindex(menu)))),
                          "prompts": ";".join(menu)})
            try:
                cop = C.standardized_cohort(data, d, args, menu=menu, manifest=man,
                                            matched_pairs=co.matched_pairs)
                results.append(row(data, cop, args, d, f"{gran}-level", len(menu)))
            except Exception as exc:  # noqa: BLE001
                results.append({"disease": d, "setting": f"{gran}-level", "menu_size": len(menu),
                                "error": repr(exc)})
        return None

    C.run_each(ns.diseases, one)
    if menus:
        C.save(menus, ns.out_dir, "prompt_menus.csv")
    name = "prompt_matched_vs_family_pcmvf.csv" if ns.granularity == ["family_matched"] else "prompt_vs_family_pcmvf.csv"
    C.save(results, ns.out_dir, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

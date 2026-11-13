from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import (
    audit_dataset, prepare_analysis, make_task_template, run_disease_benchmark,
    run_disease_task, run_disease_representation, run_common_task, run_adjustments,
    run_equal_budget, run_ranking, run_ablations, run_tier1, validate_freeze,
    run_tensor, run_matched_task_pairs, run_leave_one_site_out, run_leakage_stress,
    run_final_fixes, run_repeated_cv_sensitivity, run_speech_demographics_incremental,
    run_matched_task_fdr, run_statistical_hardening, run_nonspeech_baselines,
    run_equal_cohort_size, run_matched_controls, run_disease_vs_disease,
    run_prompt_heterogeneity, run_validity_audit, run_measurement_decomposition,
    run_operating_points, run_acceptance_upgrade, validate_acceptance_upgrade,
    run_model_robustness,
)
from .reporting import make_all_figures, make_tables, make_acceptance_figures


def parser():
    p = argparse.ArgumentParser(prog="speech-atlas", description="ML4H conditional clinical-speech detectability pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    commands = [
        "audit","prepare","make-task-template","disease","task","representation","common-task",
        "adjustments","equal-budget","ranking","ablations","tensor","matched-task","leave-one-site-out","leakage-stress",
        "figures","tables","run-tier1","final-fixes","repeated-cv","speech-demographics","matched-task-fdr","statistical-hardening","nonspeech-baselines","equal-cohort-size","matched-controls","disease-vs-disease","prompt-heterogeneity","validity-audit","measurement-decomposition","operating-points","acceptance-figures","acceptance-upgrade","model-robustness","validate-acceptance-upgrade","validate-freeze"
    ]
    for c in commands:
        q = sub.add_parser(c)
        q.add_argument("--config", default="config/atlas.yaml")
        if c == "final-fixes":
            q.add_argument("--permutations", type=int, default=100, help="Successful within-fold permutations per disease; use 1000 only if formal permutation p-values are needed")
        if c in {"repeated-cv", "statistical-hardening", "equal-cohort-size", "acceptance-upgrade"}:
            q.add_argument("--repeats", type=int, default=None, help="Repeated resamples/CV partitions (default from config)")
        if c == "equal-cohort-size":
            q.add_argument("--case-n", type=int, default=None, help="Cases per disease; default 30")
        if c == "model-robustness":
            q.add_argument("--scope", choices=["pooled", "atlas", "all"], default="pooled", help="pooled=four-model equal-budget comparison; atlas=SVM/MLP task+representation stability; all=both")
            q.add_argument("--skip-mil", action="store_true", help="Skip gated-attention MIL if PyTorch/GPU is unavailable")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = load_config(args.config)
    cmd = args.cmd
    if cmd == "audit": audit_dataset(cfg)
    elif cmd == "prepare": prepare_analysis(cfg)
    elif cmd == "make-task-template": print(make_task_template(cfg))
    elif cmd == "disease": run_disease_benchmark(cfg)
    elif cmd == "task": run_disease_task(cfg)
    elif cmd == "representation": run_disease_representation(cfg)
    elif cmd == "common-task": run_common_task(cfg)
    elif cmd == "adjustments": run_adjustments(cfg)
    elif cmd == "equal-budget": run_equal_budget(cfg)
    elif cmd == "ranking": run_ranking(cfg)
    elif cmd == "ablations": run_ablations(cfg)
    elif cmd == "tensor": run_tensor(cfg)
    elif cmd == "matched-task": run_matched_task_pairs(cfg)
    elif cmd == "leave-one-site-out": run_leave_one_site_out(cfg)
    elif cmd == "leakage-stress": run_leakage_stress(cfg)
    elif cmd == "figures": make_all_figures(cfg)
    elif cmd == "tables": make_tables(cfg)
    elif cmd == "run-tier1":
        run_tier1(cfg); make_all_figures(cfg); make_tables(cfg)
    elif cmd == "final-fixes":
        run_final_fixes(cfg, permutation_repeats=args.permutations)
    elif cmd == "repeated-cv":
        run_repeated_cv_sensitivity(cfg, repeats=args.repeats)
    elif cmd == "speech-demographics":
        run_speech_demographics_incremental(cfg)
    elif cmd == "matched-task-fdr":
        run_matched_task_fdr(cfg)
    elif cmd == "statistical-hardening":
        run_statistical_hardening(cfg, repeated_cv_repeats=args.repeats)
    elif cmd == "nonspeech-baselines":
        run_nonspeech_baselines(cfg)
    elif cmd == "equal-cohort-size":
        run_equal_cohort_size(cfg, case_n=args.case_n, repeats=args.repeats)
    elif cmd == "matched-controls":
        run_matched_controls(cfg)
    elif cmd == "disease-vs-disease":
        run_disease_vs_disease(cfg)
    elif cmd == "prompt-heterogeneity":
        run_prompt_heterogeneity(cfg)
    elif cmd == "validity-audit":
        run_validity_audit(cfg)
    elif cmd == "measurement-decomposition":
        run_measurement_decomposition(cfg)
    elif cmd == "operating-points":
        run_operating_points(cfg)
    elif cmd == "acceptance-figures":
        make_acceptance_figures(cfg)
    elif cmd == "acceptance-upgrade":
        run_acceptance_upgrade(cfg, repeated_cv_repeats=args.repeats)
    elif cmd == "model-robustness":
        run_model_robustness(cfg, scope=args.scope, skip_mil=args.skip_mil)
    elif cmd == "validate-acceptance-upgrade":
        errors = validate_acceptance_upgrade(cfg)
        if errors:
            print("ACCEPTANCE UPGRADE CHECK FAILED")
            for e in errors: print(" -", e)
            sys.exit(2)
        print("ACCEPTANCE UPGRADE CHECK PASSED")
    elif cmd == "validate-freeze":
        errors = validate_freeze(cfg)
        if errors:
            print("FREEZE CHECK FAILED")
            for e in errors: print(" -", e)
            sys.exit(2)
        print("FREEZE CHECK PASSED")

if __name__ == "__main__":
    main()

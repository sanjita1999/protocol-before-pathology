#!/usr/bin/env bash
# End-to-end reproduction from the raw Bridge2AI Voice v3.0.0 release.
#
#   export B2AI_ROOT=/path/to/b2ai-voice/3.0.0
#   bash scripts/run_all.sh            # full (20 seeds, 2000 bootstrap replicates)
#   bash scripts/run_all.sh --quick    # smoke test for the analyses (2 seeds, 200 replicates)
#
# Optional external datasets (analysis e2_5 / e3_1): set MDVR_ROOT, IPVS_ROOT, COSWARA_ROOT.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${B2AI_ROOT:?Set B2AI_ROOT to the Bridge2AI Voice v3.0.0 directory}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
PY="${PY:-python}"
QUICK=("$@")
mkdir -p results/logs

step() { echo "=== $1"; shift; if "$@" > "results/logs/$(echo "$1 $2" | tr ' /' '__').log" 2>&1; then echo "    ok"; else echo "    FAILED"; fi; }

# 1. Data products (manifest, participant-disjoint folds, recording embedding caches)
step covariates   $PY scripts/make_covariates.py --b2ai-root "$B2AI_ROOT"
step prepare      speech-atlas prepare --config config/atlas.yaml
step disease      speech-atlas disease --config config/atlas.yaml             # pooled speech benchmark
step baselines    speech-atlas nonspeech-baselines --config config/atlas.yaml # Figure 2B input

# 2. PCMVF (paper Table 1, stagewise analysis, single-session and caliper sensitivity)
step pcmvf $PY pcmvf/run_v7_all_diagnoses.py --repo-root . --manifest outputs/manifests/recordings.csv \
     --b2ai-root "$B2AI_ROOT" --out-dir outputs/protocol_balanced_framework/v8_all_diagnoses --seeds 20 --bootstrap 2000

# 3. Gate: analysis library reproduces Table 1
step verify $PY analysis/common/verify_reproduces_table1.py "${QUICK[@]}"

# 4. Analyses (Bridge2AI)
for s in e2_1_equivalence_labels e2_2_site_country_device e2_3_prompt_level_standardization \
         e2_4_semisynthetic_injection e2_6_structure_beyond_demographics \
         e1_1_model_capacity e1_1b_nested_permutation e1_2_threshold_sensitivity \
         e1_3_benchmark_reinterpretation e1_4_shap_vs_pcmvf \
         e3_2_incremental_auc e3_3_residual_protocol_walkthrough \
         e4_1_longitudinal_vs_quantity e4_2_completion_by_design e4_3_why_pd_ci_differ \
         e4_4_delong_vs_bootstrap e4_5_min_pairs_power; do
  step "$s" $PY "analysis/experiments/$s.py" "${QUICK[@]}"
done
# Non-linear age handling (5-year age bins) for the voice cohorts
step e1_1b_agebin5 $PY analysis/experiments/e1_1b_nested_permutation.py --age-bin-years 5 \
     --diseases "Airway Stenosis" "Laryngeal Dystonia" "Unilateral Vocal Fold Paralysis" \
     "Muscle Tension Dysphonia" "Benign Lesions" --out-dir results/e1_1b_nested_permutation_agebin5 "${QUICK[@]}"

# 5. Learned representations (needs requirements-audio.txt); external datasets if roots are set
step e3_1_ppg $PY analysis/experiments/e3_1_ssl_models.py --routes A "${QUICK[@]}"
[[ -n "${MDVR_ROOT:-}"    ]] && step e2_5_mdvr    $PY analysis/experiments/e2_5_external_dataset_pcmvf.py --dataset mdvr_kcl \
     --features egemaps ssl:facebook/wav2vec2-base ssl:facebook/hubert-base-ls960 ssl:microsoft/wavlm-base-plus
[[ -n "${IPVS_ROOT:-}"    ]] && step e2_5_ipvs_e  $PY analysis/experiments/e2_5_external_dataset_pcmvf.py --dataset ipvs_elderly --features egemaps --min-pairs 20
[[ -n "${IPVS_ROOT:-}"    ]] && step e2_5_ipvs_y  $PY analysis/experiments/e2_5_external_dataset_pcmvf.py --dataset ipvs_young --features egemaps --min-pairs 20
[[ -n "${COSWARA_ROOT:-}" ]] && step e2_5_coswara $PY analysis/experiments/e2_5_external_dataset_pcmvf.py --dataset coswara --features egemaps --min-pairs 20

# 6. Figures
step figures $PY analysis/experiments/make_figures.py
echo "Results in $ROOT/results/ ; logs in $ROOT/results/logs/"

# PCMVF: Protocol-Conditioned Measurement Validity Framework

This is the code for **"Protocol Before Pathology: A Measurement-Validity Framework for Clinical-Speech Benchmarks"** (ML4H 2026).

A high ROC-AUC for speech-based disease classification can partly reflect *how* the data were collected rather than the disease. The assessment record (which tasks were done, how many recordings, how many sessions, site, device) can itself predict the label. PCMVF runs in three stages:

1. **Detect:** measure how well the non-acoustic assessment record predicts the label.
2. **Standardize:** reduce observed collection differences:
   - age/sex matching;
   - shared task families;
   - equal recordings per family;
   - pair-aware participant-disjoint cross-validation.
3. **Interpret:** read speech AUC together with residual protocol AUC, uncertainty and matched-sample support.

## Repository layout

```
config/
  atlas.yaml               data paths (${B2AI_ROOT}), seeds, CV/bootstrap settings
  task_families.yaml       frozen mapping of 628 raw Bridge2AI task labels -> task families
src/clinical_speech_atlas/ data loading, cohort construction, participant-disjoint folds,
                           recording-level feature caches, pooled benchmark (CLI: speech-atlas)
pcmvf/                     the framework (paper Table 1, Figure 3, Tables S5-S9)
  run_v7_all_diagnoses.py    entry point: Detect -> Standardize -> Interpret, all cohorts
  run_protocol_balanced.py   matching, equal task-wise sampling, frozen PLS + logistic probe
  run_all_experiments.py     protocol features, ablations, balance diagnostics
  paired_bootstrap_helper.py matched-pair bootstrap
analysis/
  common/                  shared library (reuses pcmvf/ unchanged), feature extraction,
                           modality-agnostic PCMVF runner for external datasets, Table 1 check
  experiments/             sensitivity, robustness and validation analyses (table below)
scripts/
  make_covariates.py       age/sex covariates from the public demographics table
  run_all.sh               end-to-end reproduction
tests/test_stats.py        checks for the label rule, DeLong, bootstrap, TOST and pair-aware folds
```

`outputs/`, `results/` and `data/` are created at run time and are git-ignored.

## Data

| Dataset | Access | Used for |
|---|---|---|
| Bridge2AI Voice v3.0.0 (derived features + phenotype) | PhysioNet, credentialed access, DOI 10.13026/k81f-qr68 | Primary analyses. Set `B2AI_ROOT` to the directory containing `phenotype/` and `features/` |
| MDVR-KCL | Zenodo 10.5281/zenodo.2867216 | Independent negative control; SSL models (`MDVR_ROOT`) |
| Italian Parkinson's Voice and Speech | IEEE DataPort (free account) | Independent positive control (`IPVS_ROOT` = the unzipped folder) |
| Coswara | github.com/iiscleap/Coswara-Data | Independent dataset (`COSWARA_ROOT` containing `repo/combined_data.csv` and `extracted/<date>/<id>/*.wav`) |

No data are included in this repository.

## Installation

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # exact versions used for the reported results
pip install -e .                       # installs the `speech-atlas` CLI
pip install -r requirements-audio.txt  # only for learned representations / external datasets
```

## Reproducing the results

```bash
export B2AI_ROOT=/path/to/b2ai-voice/3.0.0
bash scripts/run_all.sh            # full run (20 subsampling seeds, 2,000 bootstrap replicates)
bash scripts/run_all.sh --quick    # smoke test of the analyses (2 seeds, 200 replicates)
pytest -q tests/                   # statistics checks, no data needed
```

`run_all.sh` does the following, in order:

1. `scripts/make_covariates.py`: builds `data/covariates.csv` (age, sex) from `phenotype/demographics/demographics.tsv`.
2. `speech-atlas prepare`: cohort, recording manifest, participant-disjoint folds and recording-level feature caches in `outputs/`.
3. `speech-atlas disease`, `speech-atlas nonspeech-baselines`: pooled benchmark and non-speech baselines.
4. `pcmvf/run_v7_all_diagnoses.py`: the framework. Writes `outputs/protocol_balanced_framework/v8_all_diagnoses/publication_summary.csv` (Table 1).
5. `analysis/common/verify_reproduces_table1.py`: checks that the analysis library reproduces Table 1 exactly and saves per-participant predictions.
6. `analysis/experiments/*`: writes to `results/<analysis>/`.
7. `analysis/experiments/make_figures.py`: writes to `results/figures/`.

Typical run time on a 16-core CPU:
- Table 1 alone: about 15 s per cohort.
- The full analysis suite: several hours. Most of it is `e1_1`, `e1_1b`, `e1_2` and `e2_4`.

## Analyses

| Script | Question |
|---|---|
| `e1_1_model_capacity` | Is the protocol AUC limited by the logistic probe? (7 tuned models, train vs test AUC, learning curve) |
| `e1_1b_nested_permutation` | Does residual protocol signal survive nested model selection + within-pair permutation? `--age-bin-years 5` adds exact age-bin matching |
| `e1_2_threshold_sensitivity` | Sensitivity to caliper, budget, retention and minimum pairs |
| `e1_3_benchmark_reinterpretation` | How common pipelines rank before vs after PCMVF |
| `e1_4_shap_vs_pcmvf` | TreeSHAP attribution (speech vs protocol) compared with PCMVF |
| `e2_1_equivalence_labels` | Equivalence (TOST) labels, smallest margin, power |
| `e2_2_site_country_device` | Country / device / gain / duration; USA-only and microphone-matched PCMVF |
| `e2_3_prompt_level_standardization` | Prompts matched within pairs inside each family |
| `e2_4_semisynthetic_injection` | Known artifacts injected into healthy controls |
| `e2_5_external_dataset_pcmvf` | MDVR-KCL, IPVS, Coswara |
| `e2_6_structure_beyond_demographics` | Assessment structure beyond age + sex + country |
| `e3_1_ssl_models` | Phonetic posteriorgrams (Bridge2AI); wav2vec2 / HuBERT / WavLM |
| `e3_2_incremental_auc` | Speech-over-protocol and protocol-over-speech incremental AUC |
| `e3_3_residual_protocol_walkthrough` | Stage-by-stage protocol balance |
| `e4_1_longitudinal_vs_quantity` | Repeat sessions: progression vs enrolment design |
| `e4_2_completion_by_design` | Task completion vs protocol-module assignment |
| `e4_3_why_pd_ci_differ` | Cohort profiles and protocol ablation for every cohort |
| `e4_4_delong_vs_bootstrap` | DeLong vs matched-pair bootstrap |
| `e4_5_min_pairs_power` | Justification of the 20-pair support threshold |

Every analysis script accepts `--quick`, `--diseases …` and `--out-dir`.

## Reproducibility notes

- **Fixed analysis specification:** median imputation, standardization, PLS (≤ 10 components) and class-balanced logistic regression (LIBLINEAR, C = 1), all fit on training folds only. Seeds are fixed (`--seed 2026`; 20 subsampling seeds). The original PCMVF confidence intervals use 2,000 stratified participant-bootstrap replicates. Matched-pair bootstrap is additionally used for the cognitive-impairment residual protocol analysis and the equivalence-testing experiments. All confidence intervals and testing procedures are described in the paper.
- **Table 1 check:** `verify_reproduces_table1.py` reproduces the framework's pairs and AUCs exactly.
- **Covariates:** `make_covariates.py` reproduces the covariates used in the paper exactly. Age "90 and above" is left missing.
- **Figure 2D:** `make_figures.py` recomputes it with the PCMVF probe and folds. Values agree with the published panel to within 0.02.
- **External audio:** eGeMAPS uses openSMILE eGeMAPSv02 functionals. SSL embeddings are mean-pooled hidden states (layers 6 and 12) from `facebook/wav2vec2-base`, `facebook/hubert-base-ls960` and `microsoft/wavlm-base-plus`.

## License

MIT (see `LICENSE`). Datasets are governed by their own licenses and data-use agreements.

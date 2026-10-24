# Supplementary material S1

This package accompanies “Diagnostic evaluation of vibration representations for bearing monitoring at equal retained dimensions”.

## Current manuscript evidence

Use `evidence/cwru_corrected_v2_index.json` and run:

```text
python verify_cwru_corrected_evidence.py
```

The portable checker verifies packaged SHA256 values and independently recalculates the current 40 clean, 80 noisy and 305 prefix evaluations. It requires Python and NumPy, uses package-relative paths and does not load a classifier or raw MAT data. PU results are retained from the validated original experiments. Current CWRU results come from the corrected benchmark 1784d2330d080a85432a807a and its locked noise and prefix evaluations. The corresponding implementation and independent validators are under `scripts/` and `src/data/cwru_corrected.py`.

Normal CWRU records explicitly select X097/X098/X099/X100 DE channels. The 1750 rpm record selects X099, avoiding a duplicated X098 variable. Normal baseline signals are resampled from 48 kHz to 12 kHz before segmentation. Selected raw channels and normalized windows are checked for exact cross-file duplicates. Current CWRU inputs contain 64 source files and 7524 windows. The correction reruns all five methods and five seeds with the original training settings. It includes 25 new training cells, 50 noisy inference cells and 200 prediction-only prefix evaluations. No PCA refitting or additional training occurs in the noisy evaluations.

Archived CWRU results in the older indexes and historical sections below are superseded for this manuscript. Their bytes are retained for traceability; verifying their stored metrics does not establish signal-content separation. The current evidence index identifies the corrected results. Earlier source inventories remain historical records; PACKAGE_SHA256.json and MST_CORRECTION_SHA256.json describe the distributed revision.

Raw MAT files, image caches and classifier PT weights are excluded. Their provenance hashes remain in the protocols. Repeating training requires provider data and a local configuration; repeating noise inference requires the corrected classifier checkpoints. Editing a local configuration and running a new prepare operation produces a new protocol signature. The included prediction-based verification works without those resources. Software identifiers based on original paths are retained inside archived manifests and mapped to portable package locations by the index.

## Historical package documentation

The following documentation describes retained earlier archives. For current CWRU manuscript values, use the corrected evidence index above.

# Supplementary material S1: code and experimental evidence

This package accompanies “Diagnostic performance and processing cost of low-dimensional vibration representations for bearing condition assessment”.

## Contents

- `src/`, `scripts/`, `configs/`, `requirements.txt`: experiment and analysis code.
- `evidence/artifact_index.json`: portable paths, original provenance paths, file sizes and SHA256 values.
- `evidence/cell_index.csv`: mapping from dataset, method, seed and retention setting to the corresponding files.
- `evidence/originals/`: unchanged experiment artifacts. Large `dataset_manifest.json` files are stored with lossless gzip compression; decompressing them recovers the bytes identified by `original_sha256`.
- `verify_evidence.py`: reads stored predictions and manifests to check window and source-file results without loading a classifier.

The evidence covers 64 distinct classification cells: 15 PU and 25 CWRU primary comparison cells, 18 additional retention-curve cells, and 6 random-window-split controls. The primary 40 cells include validation and test prediction NPZ files. The other 24 cells include their locked metrics and confusion matrices; no new predictions were generated for this package. The package also contains 17 PCA transformers, source-file analysis tables, timing results, experiment configurations and reconstruction-quality summaries.

## Checking the results

From the extracted package directory, with Python and NumPy installed:

```text
python verify_evidence.py --hashes --metrics
```

`--hashes` checks both packaged files and decompressed original manifests. `--metrics` independently calculates window Macro-F1, confusion matrices and source-file majority votes for the 40 primary cells. Votes tied between classes are resolved by summed logits over the source file; an exact remaining tie uses the lower class index. It checks these results against the archived summaries.

The absolute paths inside historical JSON files document the original runs. They are not paths on the reader's machine. Use `artifact_index.json` and `cell_index.csv` to locate the packaged copies. The verification script works from package-relative paths. Some original artifacts were written on a different drive; `originals/external_roots/` preserves their distinct provenance.

## Labels, model names and experiment history

The historical PU class named `ball` corresponds to the KB23/KB24/KB27 compound-damage records. The manuscript calls this class `compound`. This is a label-name correction for the same samples. CWRU `ball` denotes rolling-element faults. Keep the archived class order when reading predictions.

`resnet34_ema` is a historical configuration name. The implemented model is a ResNet34 backbone with a channel-weighting module after layer2; this name does not denote exponential moving average weights.

CWRU initially used seeds 42, 100 and 200. After observing substantial variation, the extension protocol fixed seeds 300 and 400 before the additional runs. All five methods and all five runs enter the reported statistics. Seeds describe repeated data splits and training runs, not independent physical bearing experiments.

The code includes the documented reproducibility fixes made after the archived results: PU legacy-label handling, RNG-state preservation during resumption, an independent CPU best-weight snapshot, and corrected report/validation entry points. Archived result files and protocol hashes were not rewritten. Old incomplete checkpoints without RNG states are not equivalent-resume inputs.

## Repeating model training

Obtain raw signals from the original providers:

- PU: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter
- CWRU: https://engineering.case.edu/bearingdatacenter

Install the dependencies in `requirements.txt` in a suitable environment. Set dataset paths in a new local configuration and use the experiment entry points under `scripts/`. Preserve the archived manifests when comparing original sample assignments; generated sample identifiers and hashes may depend on recorded paths. Re-running training is separate from the supplied prediction-based checks.

Raw MAT files, generated time-frequency image caches and classifier PT checkpoints are not included. The package supports inspection of the study design, recomputation of primary prediction metrics, and access to the training implementation. It does not provide a pretrained classifier inference bundle.

## Timing

Per-window timing uses 64 training windows with 5 repeats each. The timed PCA transform was fitted on 512 training windows. The separate formal-PCA fit benchmark uses the original 4096 selected training windows, five repeats, and excludes input loading and preprocessing. These are distinct measurements, as described in the manuscript.

## Package integrity

`PACKAGE_SHA256.json` lists the files supplied in S1. `SOURCE_SHA256.json` is the earlier source-code inventory. The original scientific artifact hashes are retained separately in `evidence/artifact_index.json`.

The local 2026-10-03 revision is recorded in `REVISION_20261003.md` and `SOURCE_REVISION_20261003_SHA256.json`. The earlier `SOURCE_SHA256.json` and all archived experiment files remain unchanged. The package inventory and supplementary-code inventory describe the revised distributed files, not the historical implementation signatures embedded in experiment protocols.

## Reproduction entry points after the 2026-10-03 revision

Prediction-based verification is portable and does not require raw data or model checkpoints. To run new training, first create a local YAML configuration with the provider dataset locations, keeping the signal and training rules fixed. The benchmark, CWRU extension, retention curve and efficiency runners now accept `--config path/to/local_config.yaml`. The PCA fit benchmark accepts `--formal-root path/to/the_formal_PCA_dataset_directory` and `--output path/to/a_new_timing_report.json`. Use `--help` to inspect the original protocol acknowledgement and prerequisite-stage arguments. Do not overwrite archived locks, metrics or predictions.

New training signatures record implementation hashes, installed package versions and worker settings. A metrics cache is reusable only when its selected checkpoint exists and its SHA256 matches. Archived metrics without these newly added provenance fields are scientific records, not compatible new-training resume inputs. A fresh run requires a separate workspace and the original dataset files. Historical sample identifiers incorporate original paths; relocating raw data produces a new run rather than silently reusing historical identifiers.

THOP reports multiply–accumulate operations. The archived field named `flops_gflops` stores 3.67844752 billion MACs and is interpreted as 3.68 GMACs; new timing output uses `macs_gmacs`. The archived timing values remain unchanged. Warm-up prepares cached recovery dictionaries and random-projection inverse operators; ILSOMP per-window measurement-matrix generation and measurement multiplication remain inside the transform timer. The efficiency loader materializes signals before selecting its training subset; only selected training windows are fitted or timed, and no test metrics are used.

## RIE supplementary experiments

The additional record-length analysis covers PU seeds 42/100/200 and CWRU seeds 42/100/200/300/400: 305 method-seed-length cells. It uses chronologically ordered prefixes of each original test source file and the original majority-vote rule. The added-noise analysis covers PU seed42 only: five methods at 20 and 10 dB, with unchanged classifiers and PCA transformers. The noise ratios are relative to the original observed signal, not a noise-free ground truth. These are supplementary analyses of previously evaluated tests, not new independent confirmations.

The four original experiment files are under `scripts/`:

- `rie_supplement_support.py`
- `run_rie_record_length.py`
- `run_rie_noise_sensitivity.py`
- `validate_rie_supplement.py`

The locked protocols and summaries are under `evidence/additional_experiments/rie_record_length/1ffb76f8a7eb311021123da7/` and `evidence/additional_experiments/rie_noise_sensitivity/4f52371b6d86d971763e57d4/`. The latter also contains the ten complete prediction NPZ files and cell records. `evidence/rie_supplement_index.json` maps original absolute paths to these package-relative copies. Historical protocols and summaries retain their original bytes and hashes.

From this package directory, recompute the supplementary metrics without training or model inference:

```powershell
python verify_rie_supplements.py --hashes --metrics
```

This portable entry point uses the independent validator's metric implementation, verifies the archived protocols and predictions, checks paired sample IDs/classes/targets and common noisy-input fingerprints, and reproduces all 315 cell metrics and their aggregates. It does not reload models or access UORED data. `RIE_SUPPLEMENT_SHA256.json` records the added scripts and index.

The original experiment runners preserve the paths recorded in the research workspace. Repeating noise inference requires the providers' raw data and the original classifier checkpoints, which are not distributed in S1. The portable prediction-based check above requires neither. Additive-noise generation and inference timings in the cell records are not the separate per-window software benchmark reported in the manuscript.

## NDE added-noise robustness extension

The extension covers all five methods at 20 and 10 dB on PU seeds 42/100/200 and CWRU seeds 42/100/200/300/400. It contains 80 noise-condition cells: 70 new fixed-checkpoint evaluations and 10 reused PU seed42 results. The 40 original-condition prediction cells remain the paired references. No classifier training, PCA refitting, checkpoint selection, or UORED access was performed for this extension.

Run the portable verification from the extracted package root:

```text
python verify_nde_noise_extension.py --hashes --metrics
```

The checker uses packaged paths and requires Python and NumPy, not classifier weights, raw MAT files, or a GPU. It recomputes window/class/source-file metrics from predictions and checks 120 condition rows, 96 paired method comparisons, and 64 changes in paired differences. It checks the supplied artifacts rather than claiming to recheck excluded raw signals or model weights.

The new index is `evidence/nde_noise_extension_index.json`; additional artifacts are under `evidence/additional_experiments/nde_noise_robustness_extension/`. JSON files may be gzip-compressed without changing their decompressed original bytes. `NDE_EXTENSION_SHA256.json` records the extension code and index fingerprints. Descriptive means, sample standard deviations, ranges, and runwise signs are reported separately for PU and CWRU. These runs reuse source/bearing identities and are not independent physical replications. Original means no additional experimental noise, not noise-free ground truth.

The original inference entry point is `scripts/run_nde_noise_robustness_extension.py`. Repeating inference requires the original benchmark checkpoints and datasets to be restored or reproduced; the portable checker above instead verifies the supplied prediction evidence.


# Vibration representations for bearing fault diagnosis

Corrected source code for **Diagnostic evaluation of vibration representations for bearing monitoring at equal retained dimensions**.

## Current corrected code — 24 October 2026

The source now explicitly selects the original CWRU DE channel identifiers. Normal records 097–100 use X097/X098/X099/X100_DE_time, respectively. The 1750 rpm file uses X099_DE_time rather than an earlier duplicated X098 variable. Normal baseline records are resampled from 48 kHz to 12 kHz before segmentation. Selected channels and normalized windows are checked for exact duplicates across source files.

The corrected benchmark retains the five methods, 50% retention, original architecture and training rules, and seeds 42/100/200/300/400. The corrected inputs comprise 64 source files and 7524 windows. All 25 main comparison cells, 50 added-noise inference cells and 200 prediction-only prefix evaluations passed independent validation.

### Main entry points

- `src/data/cwru_corrected.py`: explicit channel selection, resampling and content checks.
- `scripts/run_cwru_corrected_benchmark_v2.py`: prepare and run the corrected five-method benchmark.
- `scripts/validate_cwru_corrected_benchmark_v2.py`: independent input and primary-result validation.
- `scripts/run_cwru_corrected_supplements_v2.py`: locked observation-duration and 20/10 dB noise evaluations.
- `scripts/validate_cwru_corrected_supplements_v2.py`: independent metrics and raw noisy-input hash validation.
- `verify_cwru_corrected_evidence.py`: portable metric verification when used with the complete corrected S1 evidence package.

Install `requirements.txt` and configure the provider data paths in `configs/default_config.yaml` before preparing a new benchmark. Start with:

```powershell
python -B scripts/run_cwru_corrected_benchmark_v2.py --stage prepare --dry-run
```

The preview supplies the signature and lock path. Follow the runner's prepare confirmation and run stages. A changed local configuration produces a new protocol signature. Raw MAT files and classifier checkpoints are not included in this source repository.

## Evidence archive status

The corrected archive is named `Supplementary_S1_Code_and_Evidence_MST_Corrected_20261023.zip` (511971670 bytes).

SHA256: `f3c6e48f32ad8e73841af45a6ebde0d32e21983472770bb464368b2dcfe776db`.

The complete corrected evidence archive is available under [v1.1.0-mst-corrected](https://github.com/22150017/A-comparison-at-equal-retained-dimensions/releases/tag/v1.1.0-mst-corrected). Download `Supplementary_S1_Code_and_Evidence_MST_Corrected_20261023.zip`; its file size and GitHub SHA256 digest match the values above. The release tag points to corrected source commit `269bc23d82b53f561263b635d4426c3b6abe4432`.

`verify_cwru_corrected_evidence.py` requires this complete package, including `evidence/cwru_corrected_v2_index.json` and the indexed predictions.

Earlier pre-correction archives remain among the release assets for version traceability. Their CWRU results are superseded and must not be used as the corrected manuscript results. `README_S1.md` records the package version distinctions and verification scope.

`CODE_SHA256.json` records SHA256 checksums for the distributed text files. No raw signals, neural-network weights, credentials or unpublished manuscript files are uploaded here.

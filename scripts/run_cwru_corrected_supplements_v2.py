from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / 'scripts'):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.rie_supplement_support import artifact, arrays, align, ordered_records, prepare, read_json
from scripts.run_cwru_corrected_benchmark_v2 import load_lock as load_primary_lock

DEFAULT_LOCK = Path('F:/paper1/artifacts_v2/studies/cwru_corrected_benchmark_v2/1784d2330d080a85432a807a/protocol_lock.json')
DEFAULT_OUTPUT = Path('F:/paper1/artifacts_v2/studies/cwru_corrected_supplements_v2')


def build_protocol(base_lock: Path, experiment: str) -> dict:
    primary = load_primary_lock(base_lock)
    summary_path = base_lock.parent / 'corrected_benchmark_summary.json'
    summary = read_json(summary_path)
    if not summary['complete'] or summary['signature'] != primary['signature']:
        raise ValueError('Corrected primary benchmark is incomplete or mismatched')
    noise = experiment == 'noise_sensitivity'
    methods = primary['protocol']['methods']
    protocol = {'schema_version': 2, 'experiment': experiment, 'methods': methods, 'runs': [],
                'inputs': [artifact(base_lock), artifact(summary_path)], 'inference_cells': 50 if noise else 0,
                'training_performed': False, 'inference_performed': noise, 'uored_test_opened': False,
                'new_independent_test_opened': False, 'retained_values': 512, 'window_samples': 1024,
                'sampling_rate_hz': 12000, 'corrected_primary_summary': artifact(summary_path),
                'analysis_role': 'fixed supplementary re-evaluation using corrected CWRU models and inputs',
                'source_vote_rule': 'window majority; tied classes use float64 logit sums; remaining ties use lower class index'}
    if noise:
        protocol.update({'snr_db': [20.0, 10.0], 'noise_master_seed': 20260930,
                         'noise_rule': 'same zero-mean Gaussian realization per sample ID; exact power scaling before max_abs normalization',
                         'noise_power_reference': 'resampled observed window, not a noise-free ground truth',
                         'pca_refitted': False, 'checkpoint_selection_changed': False,
                         'batch_size': 32, 'chunk_size': 256,
                         'inference_precision': 'float32 without autocast; original export preprocessing'})
    code_paths = ['scripts/run_cwru_corrected_supplements_v2.py', 'scripts/validate_cwru_corrected_supplements_v2.py',
                  'scripts/rie_supplement_support.py', 'scripts/run_rie_record_length.py',
                  'scripts/run_rie_noise_sensitivity.py', 'scripts/validate_rie_supplement.py',
                  'scripts/run_clean_study.py', 'scripts/benchmark_followup_efficiency.py',
                  'src/data/cwru_corrected.py', 'src/data/cwru_loader.py', 'src/data/pu_loader.py',
                  'src/reconstruction/compression_baselines.py', 'src/reconstruction/sparse_recovery.py',
                  'src/training/train_classifier.py', 'src/models/attention_resnet.py']
    protocol['inputs'].extend(artifact(ROOT / name) for name in code_paths)
    for seed in primary['protocol']['seeds']:
        run = {'dataset': 'cwru', 'seed': seed, 'methods': {}, 'paper_label_names': {}}
        reference, reference_records = None, None
        for method in methods:
            cell = summary['runs'][str(seed)][method]
            original = cell['artifacts']
            entry = {name: artifact(original[key]['path'], original[key]['sha256'])
                     for name, key in [('prediction', 'test_prediction'), ('manifest', 'manifest'),
                                       ('prediction_metadata', 'prediction_metadata'), ('metrics', 'metrics')]}
            payload = arrays(entry['prediction']['path'])
            manifest = read_json(entry['manifest']['path'])
            metadata = read_json(entry['prediction_metadata']['path'])
            if metadata['files']['test']['sha256'] != entry['prediction']['sha256']:
                raise ValueError('Prediction export hash differs')
            if manifest['condition']['signal']['cwru_source_inventory'] != primary['protocol']['config']['signal']['cwru_source_inventory']:
                raise ValueError('Manifest does not use corrected preprocessing')
            records = ordered_records(manifest, payload)
            identities = [(r['source_file'], r['segment_id'], r['label']) for r in records]
            if reference is not None:
                align(reference, payload)
                if identities != reference_records:
                    raise ValueError('Method source windows differ')
            else:
                reference, reference_records = payload, identities
            if noise:
                entry['checkpoint'] = artifact(original['checkpoint']['path'], original['checkpoint']['sha256'])
                if entry['checkpoint']['sha256'] != metadata['export_config']['checkpoint_sha256']:
                    raise ValueError('Checkpoint does not match prediction export')
                if method == 'pca':
                    for name in ('pca', 'pca_metadata'):
                        entry[name] = artifact(original[name]['path'], original[name]['sha256'])
                indices = sorted({0, len(records) // 2, len(records) - 1})
                entry['clean_image_checks'] = [{'index': i, **artifact(records[i]['image_path'])} for i in indices]
                protocol['inputs'].extend({k: row[k] for k in ('path', 'sha256')} for row in entry['clean_image_checks'])
            protocol['inputs'].extend(value for value in entry.values() if isinstance(value, dict) and 'sha256' in value)
            run['methods'][method] = entry
        groups = defaultdict(list)
        for row in records:
            groups[row['source_file']].append(row)
        minimum = min(map(len, groups.values()))
        run.update({'test_windows': len(reference['y_true']), 'source_files': len(groups), 'minimum_source_windows': minimum,
                    'window_lengths': [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= minimum] + ['full'],
                    'skipped_lengths': [k for k in [1, 2, 4, 8, 16, 32, 64] if k > minimum]})
        if noise:
            declared = {str(Path(row['path']).resolve()): row['sha256'] for row in manifest['condition']['sources']}
            run['raw_sources'] = [artifact(path, declared[str(Path(path).resolve())]) for path in sorted(groups)]
            protocol['inputs'].extend(run['raw_sources'])
        protocol['runs'].append(run)
    unique = {}
    for row in protocol['inputs']:
        previous = unique.setdefault(row['path'], row['sha256'])
        if previous != row['sha256']:
            raise ValueError('Conflicting locked file hashes')
    protocol['inputs'] = [{'path': key, 'sha256': value} for key, value in sorted(unique.items())]
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'run'], required=True)
    parser.add_argument('--experiment', choices=['record_length', 'noise_sensitivity'], required=True)
    parser.add_argument('--base-lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--confirm-signature')
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(build_protocol(args.base_lock, args.experiment), args.dry_run, args.confirm_signature,
                args.output_root / args.experiment)
    else:
        if args.lock is None or args.dry_run:
            parser.error('run requires --lock and does not accept --dry-run')
        if read_json(args.lock)['protocol']['experiment'] != args.experiment:
            raise ValueError('Experiment differs from lock')
        if args.experiment == 'record_length':
            from scripts.run_rie_record_length import run
        else:
            from scripts.run_rie_noise_sensitivity import run
        run(args.lock)


if __name__ == '__main__':
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / 'scripts'):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.rie_supplement_support import arrays, artifact, load_lock, read_json, save_json
from scripts.validate_rie_supplement import identical, pair_check, independent_metrics, independent_aggregate


def expected_noise_hashes(run: dict, payload: dict, manifest: dict, snr: float, master_seed: int) -> list[str]:
    inventory = manifest['condition']['signal']['cwru_source_inventory']
    rows = {r['sample_id']: r for r in manifest['records'] if r['split'] == 'test'}
    selected = {r['source_file'] for r in rows.values()}
    vectors = {}
    for source in selected:
        row = next(r for r in inventory['sources'] if Path(source).as_posix().endswith(r['relative_path']))
        raw = np.asarray(loadmat(source)[row['signal_key']], dtype=np.float64).reshape(-1)
        if row['native_rate_hz'] == 48000:
            raw = resample_poly(raw, 1, 4, window=('kaiser', 5.0), padtype='constant')
        elif row['native_rate_hz'] != 12000:
            raise ValueError('Unexpected original sampling rate')
        vectors[source] = raw
    hashes = []
    for identity in payload['sample_ids'].tolist():
        row = rows[identity]
        start = int(row['segment_id']) * 1024
        signal = vectors[row['source_file']][start:start + 1024]
        seed = int.from_bytes(hashlib.sha256(f'{master_seed}:{identity}'.encode()).digest()[:8], 'little')
        noise = np.random.default_rng(seed).standard_normal(1024)
        noise -= noise.mean()
        energy = float(np.dot(signal, signal))
        noise *= np.sqrt(energy / (10 ** (snr / 10) * np.dot(noise, noise)))
        values = np.asarray(signal + noise, dtype='<f8')
        hashes.append(hashlib.sha256(values.tobytes()).hexdigest())
    return hashes


def validate(path: Path) -> dict:
    header = read_json(path)
    experiment = header['protocol']['experiment']
    if experiment not in {'record_length', 'noise_sensitivity'}:
        raise ValueError('Unsupported experiment')
    lock = load_lock(path, experiment)
    protocol, root = lock['protocol'], path.parent
    if protocol['training_performed'] or protocol['uored_test_opened'] or protocol['new_independent_test_opened']:
        raise ValueError('Scope flags differ')
    if [r['seed'] for r in protocol['runs']] != [42, 100, 200, 300, 400]:
        raise ValueError('Expected all five corrected seeds')
    primary = read_json(protocol['corrected_primary_summary']['path'])
    summary = read_json(root / f'{experiment}_summary.json')
    if not primary['complete'] or not summary['complete'] or summary['signature'] != lock['signature']:
        raise ValueError('Incomplete results or mismatched signature')
    identical(summary['protocol_lock'], artifact(path))
    indexed = {}
    dimension = 'length' if experiment == 'record_length' else 'snr_db'
    for cell in summary['cells']:
        key = (cell['seed'], cell['method'], cell[dimension])
        if key in indexed:
            raise ValueError('Duplicate cell')
        indexed[key] = cell
    recomputed, expected = [], set()
    for run in protocol['runs']:
        baseline = arrays(run['methods']['no_cs']['prediction']['path'])
        expected_hashes = {}
        for method, entry in run['methods'].items():
            clean = arrays(entry['prediction']['path'])
            pair_check(baseline, clean)
            manifest = read_json(entry['manifest']['path'])
            if manifest['condition']['signal']['cwru_source_inventory']['version'] != 'cwru_explicit_de_resampled_v2':
                raise ValueError('Legacy input in corrected supplement')
            reference = independent_metrics(clean, manifest, 'full')
            expected_primary = primary['runs'][str(run['seed'])][method]['metrics']
            identical(reference['window']['macro_f1'], expected_primary['window_macro_f1'])
            identical(reference['source']['macro_f1'], expected_primary['source_macro_f1'])
            counts = [r['window_count'] for r in reference['source_rows']]
            lengths = [k for k in [1, 2, 4, 8, 16, 32, 64] if k <= min(counts)] + ['full']
            identical(run['window_lengths'], lengths)
            settings = lengths if experiment == 'record_length' else protocol['snr_db']
            for setting in settings:
                key = (run['seed'], method, setting)
                expected.add(key)
                stored = indexed[key]
                if experiment == 'record_length':
                    metrics = independent_metrics(clean, manifest, setting)
                    identical(stored['observation_seconds'], None if setting == 'full' else setting * 1024 / 12000)
                else:
                    if stored['signature'] != lock['signature'] or stored['training_performed'] or stored['pca_refitted'] or stored['uored_test_opened']:
                        raise ValueError('Noise cell scope differs')
                    identical(stored['checkpoint'], entry['checkpoint'])
                    artifact(stored['prediction']['path'], stored['prediction']['sha256'])
                    data = arrays(stored['prediction']['path'])
                    pair_check(clean, data)
                    if not np.allclose(data['actual_added_snr_db'], setting, atol=1e-10, rtol=0):
                        raise ValueError('SNR differs')
                    if setting not in expected_hashes:
                        expected_hashes[setting] = expected_noise_hashes(run, clean, manifest, setting, protocol['noise_master_seed'])
                    identical(data['noisy_input_sha256'].tolist(), expected_hashes[setting])
                    metrics = independent_metrics(data, manifest, 'full')
                    identical(stored['clean_reference'], reference)
                    delta = {'window_macro_f1': metrics['window']['macro_f1'] - reference['window']['macro_f1'],
                             'minimum_class_recall': metrics['window']['minimum_class_recall'] - reference['window']['minimum_class_recall'],
                             'source_macro_f1': metrics['source']['macro_f1'] - reference['source']['macro_f1'],
                             'failed_source_count': metrics['failed_source_count'] - reference['failed_source_count'],
                             'per_class_recall': {k: v - reference['window']['per_class_recall'][k] for k, v in metrics['window']['per_class_recall'].items()}}
                    identical(stored['delta_vs_clean'], delta)
                    checks = [{'sample_id': str(clean['sample_ids'][r['index']]), 'pixels_equal': True} for r in entry['clean_image_checks']]
                    identical(stored['clean_frontend_checks'], checks)
                identical(stored['metrics'], metrics)
                recomputed.append({'dataset': 'cwru', 'seed': run['seed'], 'method': method, dimension: setting, 'metrics': metrics})
    if set(indexed) != expected:
        raise ValueError('Missing or extra experiment cells')
    identical(summary['aggregate'], independent_aggregate(recomputed, ['dataset', 'method', dimension]))
    if experiment == 'record_length':
        by_key = {(r['seed'], r['method'], r['length']): r for r in recomputed}
        paired = []
        for row in recomputed:
            if row['method'] == 'no_cs':
                continue
            reference = by_key[(row['seed'], 'no_cs', row['length'])]['metrics']
            paired.append({key: row[key] for key in ('dataset', 'seed', 'method', 'length')} | {
                'source_macro_f1_delta_vs_no_cs': row['metrics']['source']['macro_f1'] - reference['source']['macro_f1'],
                'failed_source_count_delta_vs_no_cs': row['metrics']['failed_source_count'] - reference['failed_source_count']})
        identical(summary['paired_comparisons'], paired)
    result = {'valid': True, 'signature': lock['signature'], 'experiment': experiment, 'verified_cells': len(recomputed),
              'metrics_independently_recomputed': True, 'raw_noisy_input_hashes_independently_recomputed': experiment == 'noise_sensitivity',
              'training_performed': False, 'validator_inference_performed': False, 'uored_signal_values_opened': False}
    save_json(root / 'validation_summary.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lock', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.lock.resolve()), indent=2))


if __name__ == '__main__':
    main()

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from rie_supplement_support import arrays, signature
from validate_rie_supplement import identical, independent_aggregate, independent_metrics, pair_check


def read(path):
    path = Path(path)
    payload = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return json.loads(payload)


def normalized(path):
    return str(path).replace('\\', '/').lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hashes', action='store_true')
    parser.add_argument('--metrics', action='store_true')
    args = parser.parse_args()
    if not (args.hashes or args.metrics):
        parser.error('Select --hashes, --metrics, or both')
    historical = read(ROOT / 'evidence/artifact_index.json')
    index = read(ROOT / 'evidence/rie_supplement_index.json')
    mapping = {}
    for artifact in historical['artifacts'] + index['artifacts']:
        mapping[normalized(artifact['original_path'])] = artifact
    checked = set()

    def packaged(artifact):
        path = (ROOT / 'evidence' / artifact['package_path']).resolve()
        if not path.is_relative_to((ROOT / 'evidence').resolve()):
            raise ValueError('Artifact path leaves evidence directory')
        if path not in checked:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != artifact['stored_sha256']:
                raise ValueError(f'Stored SHA256 mismatch: {path}')
            original = gzip.decompress(data) if artifact['gzip'] else data
            if hashlib.sha256(original).hexdigest() != artifact['original_sha256']:
                raise ValueError(f'Original SHA256 mismatch: {path}')
            checked.add(path)
        return path

    def resolve(item):
        entry = mapping[normalized(item['path'])]
        if entry['original_sha256'] != item['sha256']:
            raise ValueError('Protocol hash differs from packaged artifact')
        return packaged(entry)

    if args.hashes:
        for item in index['artifacts']:
            packaged(item)
        for relative, expected in read(ROOT / 'RIE_SUPPLEMENT_SHA256.json').items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Supplement code or index SHA256 mismatch: {relative}')
    total = 0
    for experiment in index['experiments']:
        lock_path = resolve(experiment['protocol_lock'])
        summary_path = resolve(experiment['summary'])
        lock, summary = read(lock_path), read(summary_path)
        protocol = lock['protocol']
        if signature(protocol) != lock['signature'] or summary['signature'] != lock['signature'] or not summary['complete']:
            raise ValueError('Protocol signature or summary status mismatch')
        identical(summary['protocol_lock'], experiment['protocol_lock'])
        dimension = 'length' if protocol['experiment'] == 'record_length' else 'snr_db'
        inference = dimension == 'snr_db'
        if inference and (summary['pca_refitted'] or len(summary['cells']) != protocol['inference_cells']):
            raise ValueError('Noise summary scope or cell count differs')
        for payload in (protocol, summary):
            if payload['training_performed'] or payload['uored_test_opened'] or payload['new_independent_test_opened'] or payload['inference_performed'] != inference:
                raise ValueError('Archived experiment scope differs')
        if args.metrics:
            recomputed, noises = [], {}
            indexed = {(c['dataset'], c['seed'], c['method'], c[dimension]): c for c in summary['cells']}
            if len(indexed) != len(summary['cells']):
                raise ValueError('Duplicate cells')
            for run in protocol['runs']:
                baseline = arrays(resolve(run['methods']['no_cs']['prediction']))
                for method in protocol['methods']:
                    entry = run['methods'][method]
                    clean = arrays(resolve(entry['prediction']))
                    manifest = read(resolve(entry['manifest']))
                    pair_check(baseline, clean)
                    reference = independent_metrics(clean, manifest, 'full')
                    settings = run['window_lengths'] if not inference else protocol['snr_db']
                    for setting in settings:
                        stored = indexed[(run['dataset'], run['seed'], method, setting)]
                        data = clean
                        if inference:
                            if stored['signature'] != lock['signature']:
                                raise ValueError('Noise cell signature differs')
                            identical(stored['checkpoint'], entry['checkpoint'])
                            checks = [{'sample_id': str(clean['sample_ids'][row['index']]), 'pixels_equal': True}
                                      for row in entry['clean_image_checks']]
                            identical(stored['clean_frontend_checks'], checks)
                            data = arrays(resolve(stored['prediction']))
                            pair_check(clean, data)
                            if stored['training_performed'] or stored['pca_refitted'] or stored['uored_test_opened']:
                                raise ValueError('Noise cell scope differs')
                            actual = data['actual_added_snr_db']
                            hashes = data['noisy_input_sha256']
                            if actual.shape != clean['y_true'].shape or not np.allclose(actual, setting, atol=1e-10, rtol=0):
                                raise ValueError('Added-noise SNR differs')
                            if hashes.shape != clean['y_true'].shape or any(len(str(h)) != 64 for h in hashes):
                                raise ValueError('Invalid input hashes')
                            key = (run['dataset'], run['seed'], setting)
                            if key in noises and not np.array_equal(noises[key], hashes):
                                raise ValueError('Methods received different noise inputs')
                            noises[key] = hashes
                            identical(stored['clean_reference'], reference)
                        metrics = independent_metrics(data, manifest, 'full' if inference else setting)
                        identical(stored['metrics'], metrics)
                        if inference:
                            delta = {key: metrics[level][field] - reference[level][field] for key, level, field in [('window_macro_f1', 'window', 'macro_f1'), ('minimum_class_recall', 'window', 'minimum_class_recall'), ('source_macro_f1', 'source', 'macro_f1')]}
                            delta['failed_source_count'] = metrics['failed_source_count'] - reference['failed_source_count']
                            delta['per_class_recall'] = {name: metrics['window']['per_class_recall'][name] - reference['window']['per_class_recall'][name] for name in clean['class_names'].tolist()}
                            identical(stored['delta_vs_clean'], delta)
                        else:
                            identical(stored['observation_seconds'], None if setting == 'full' else setting * 1024 / 12000)
                        recomputed.append({'dataset': run['dataset'], 'seed': run['seed'], 'method': method, dimension: setting, 'metrics': metrics})
            if len(recomputed) != len(indexed):
                raise ValueError('Missing or unexpected cells')
            identical(summary['aggregate'], independent_aggregate(recomputed, ['dataset', 'method', dimension]))
            if not inference:
                refs = {(c['dataset'], c['seed'], c['length']): c['metrics'] for c in recomputed if c['method'] == 'no_cs'}
                pairs = []
                for cell in recomputed:
                    if cell['method'] == 'no_cs':
                        continue
                    ref = refs[(cell['dataset'], cell['seed'], cell['length'])]
                    pairs.append({key: cell[key] for key in ['dataset', 'seed', 'method', 'length']} | {'source_macro_f1_delta_vs_no_cs': cell['metrics']['source']['macro_f1'] - ref['source']['macro_f1'], 'failed_source_count_delta_vs_no_cs': cell['metrics']['failed_source_count'] - ref['failed_source_count']})
                identical(summary['paired_comparisons'], pairs)
            total += len(recomputed)
    print(f'OK: {len(checked)} packaged evidence files checked; {total} supplement cells recomputed from stored predictions; no training or model inference')


if __name__ == '__main__':
    main()

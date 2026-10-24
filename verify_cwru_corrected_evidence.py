from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    root=args.package_root.resolve()
    sys.path.insert(0,str(root/'scripts'))
    from validate_rie_supplement import independent_metrics, independent_aggregate, identical, pair_check
    index=json.loads((root/'evidence/cwru_corrected_v2_index.json').read_text(encoding='utf-8'))
    lookup={str(r['original_path']).replace('\\','/').lower():r for r in index['artifacts']}
    def path_for(original):
        return root/'evidence'/lookup[str(original).replace('\\','/').lower()]['package_path']
    def read(original):
        p=path_for(original)
        data=p.read_bytes()
        if p.suffix=='.gz': data=gzip.decompress(data)
        return json.loads(data)
    def array(original):
        with np.load(path_for(original),allow_pickle=False) as z:
            return {k:z[k] for k in z.files}
    for record in index['artifacts']:
        data=(root/'evidence'/record['package_path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=record['stored_sha256']:
            raise ValueError('Packaged SHA mismatch')
        original=gzip.decompress(data) if record['gzip'] else data
        if hashlib.sha256(original).hexdigest()!=record['original_sha256']:
            raise ValueError('Original SHA mismatch')
    primary=read(index['primary_summary'])
    analysis=read(index['analysis'])
    row_index={(r['dataset'],r['seed'],r['method'],r['condition']):r for r in analysis['rows']}
    clean_data={}
    def check_row(dataset,seed,method,condition,metrics):
        row=row_index[(dataset,seed,method,condition)]
        for name,value in [('window_macro_f1',metrics['window']['macro_f1']),('minimum_class_recall',metrics['window']['minimum_class_recall']),
                           ('source_macro_f1',metrics['source']['macro_f1']),('failed_source_count',metrics['failed_source_count'])]:
            identical(row[name],value)
    clean_count=0
    for seed,methods in primary['runs'].items():
        for method,cell in methods.items():
            data=array(cell['artifacts']['test_prediction']['path'])
            manifest=read(cell['artifacts']['manifest']['path'])
            metrics=independent_metrics(data,manifest,'full')
            check_row('cwru',int(seed),method,'original',metrics)
            identical(cell['metrics']['window_macro_f1'],metrics['window']['macro_f1'])
            identical(cell['metrics']['source_macro_f1'],metrics['source']['macro_f1'])
            clean_data[('cwru',int(seed),method)]=(data,manifest)
            clean_count+=1
    for run in index['pu_runs']:
        for method,entry in run['methods'].items():
            data=array(entry['prediction']['path']);manifest=read(entry['manifest']['path'])
            metrics=independent_metrics(data,manifest,'full')
            check_row('pu',run['seed'],method,'original',metrics)
            clean_data[('pu',run['seed'],method)]=(data,manifest)
            clean_count+=1
    noise_count=0;seen_noise={}
    for source in index['noise_summaries']:
        summary=read(source)
        for cell in summary['cells']:
            if source==index['pu_noise_summary'] and cell['dataset']!='pu': continue
            key=(cell['dataset'],cell['seed'],cell['method'])
            clean,manifest=clean_data[key]
            data=array(cell['prediction']['path'])
            pair_check(clean,data)
            metrics=independent_metrics(data,manifest,'full')
            identical(cell['metrics'],metrics)
            check_row(*key,f"{cell['snr_db']:g}dB",metrics)
            if not np.allclose(data['actual_added_snr_db'],cell['snr_db'],atol=1e-10,rtol=0):
                raise ValueError('Noise ratios differ')
            paired=(cell['dataset'],cell['seed'],cell['snr_db'])
            values=data['noisy_input_sha256']
            if paired in seen_noise and not np.array_equal(seen_noise[paired],values):
                raise ValueError('Noisy inputs differ across methods')
            seen_noise[paired]=values
            noise_count+=1
    prefix_count=0
    for source in index['prefix_summaries']:
        summary=read(source);recomputed=[]
        for cell in summary['cells']:
            if source==index['pu_prefix_summary'] and cell['dataset']!='pu': continue
            key=(cell['dataset'],cell['seed'],cell['method'])
            data,manifest=clean_data[key]
            metrics=independent_metrics(data,manifest,cell['length'])
            identical(cell['metrics'],metrics)
            recomputed.append({**cell,'metrics':metrics})
            prefix_count+=1
        calculated=independent_aggregate(recomputed,['dataset','method','length'])
        stored=[r for r in summary['aggregate'] if source!=index['pu_prefix_summary'] or r['dataset']=='pu']
        identical(stored,calculated)
    for row in analysis['paired']:
        a=row_index[(row['dataset'],row['seed'],row['method'],row['condition'])]
        b=row_index[(row['dataset'],row['seed'],'no_cs',row['condition'])]
        for metric in ['window_macro_f1','source_macro_f1','minimum_class_recall','failed_source_count']:
            identical(row[metric+'_delta'],a[metric]-b[metric])
    if (clean_count,noise_count,prefix_count)!=(40,80,305):
        raise ValueError('Expected 40 clean, 80 noisy and 305 prefix evaluations')
    print(json.dumps({'valid':True,'verified_clean_cells':clean_count,'verified_noisy_cells':noise_count,'verified_prefix_cells':prefix_count,
                      'verified_condition_rows':len(row_index),'artifact_hashes_verified':len(index['artifacts']),
                      'training_performed':False,'inference_performed':False,'raw_signal_values_opened':False},indent=2))


if __name__=='__main__':
    main()

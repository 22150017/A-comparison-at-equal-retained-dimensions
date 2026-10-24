from pathlib import Path
import argparse
import gzip
import hashlib
import json
import csv
import numpy as np

ROOT=Path(__file__).resolve().parent/'evidence'

def read(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8') as f:
        return json.load(f)

def digest(path,compressed=False):
    h=hashlib.sha256()
    opener=gzip.open if compressed else open
    with opener(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()

def metrics(y,pred,k):
    cm=np.zeros((k,k),dtype=np.int64)
    np.add.at(cm,(y,pred),1)
    den=cm.sum(axis=0)+cm.sum(axis=1)
    f1=np.divide(2*np.diag(cm),den,out=np.zeros(k,dtype=float),where=den!=0)
    return float(f1.mean()),cm

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--hashes',action='store_true')
    cli.add_argument('--metrics',action='store_true')
    args=cli.parse_args()
    if not args.hashes and not args.metrics:
        cli.error('Choose --hashes, --metrics, or both.')
    index=read(ROOT/'artifact_index.json')
    if args.hashes:
        for item in index['artifacts']:
            path=ROOT/item['package_path']
            if digest(path)!=item['stored_sha256']:
                raise ValueError(f'Stored hash mismatch: {path}')
            if item['gzip'] and digest(path,True)!=item['original_sha256']:
                raise ValueError(f'Uncompressed hash mismatch: {path}')
        print(f"Verified {len(index['artifacts'])} artifact hashes",flush=True)
    checked=0
    if args.metrics:
        source_reports={}
        source_tables={}
        for dataset,filename in [('pu','formal-pu_fe02696922cd_source_file_summary.json'),('cwru','external-cwru_65b26947512b_source_file_summary.json')]:
            source_reports[dataset]=read(ROOT/'originals/artifacts_v2/reports/source_file_performance'/filename)
            csv_path=ROOT/'originals/artifacts_v2/reports/source_file_performance'/filename.replace('_summary.json','_rows.csv')
            with csv_path.open('r',encoding='utf-8-sig',newline='') as handle:
                rows=list(csv.DictReader(handle))
            lookup={(r['seed'],r['method'],r['source_file']):r for r in rows}
            if len(lookup)!=len(rows):
                raise ValueError('Duplicate source CSV rows.')
            source_tables[dataset]=lookup
        for cell in index['classification_cells']:
            if 'test' not in cell['predictions']:
                continue
            with np.load(ROOT/cell['predictions']['test'],allow_pickle=False) as z:
                ids=z['sample_ids'].astype(str)
                y=z['y_true'].astype(np.int64)
                logits=z['logits'].astype(np.float64)
                classes=z['class_names'].astype(str).tolist()
            pred=logits.argmax(axis=1)
            if logits.shape!=(len(y),len(classes)) or len(ids)!=len(y) or not np.all(np.isfinite(logits)) or np.any(y<0) or np.any(y>=len(classes)):
                raise ValueError('Invalid prediction dimensions or values.')
            score,cm=metrics(y,pred,len(classes))
            saved=read(ROOT/cell['metrics'])['paper_metrics']
            if abs(score-saved['macro_f1'])>1e-10 or not np.array_equal(cm,saved['confusion_matrix']):
                raise ValueError(f'Window metric mismatch: {cell["metrics"]}')
            manifest=read(ROOT/cell['manifest'])
            lookup={r['sample_id']:r for r in manifest['records'] if r['split']=='test'}
            if len(ids)!=len(set(ids)) or set(ids)!=set(lookup):
                raise ValueError('Test sample IDs are not aligned.')
            sources={}
            for i,sample in enumerate(ids):
                record=lookup[sample]
                if record['label']!=classes[y[i]]:
                    raise ValueError('Manifest label mismatch.')
                sources.setdefault(record['source_file'],[]).append(i)
            sy=[]
            sp=[]
            failed=[]
            for source,indices in sources.items():
                a=np.asarray(indices)
                if len(np.unique(y[a]))!=1:
                    raise ValueError('Mixed source labels.')
                votes=np.bincount(pred[a],minlength=len(classes))
                leaders=np.flatnonzero(votes==votes.max())
                chosen=leaders[np.argmax(logits[a][:,leaders].sum(axis=0))]
                sy.append(y[a[0]])
                sp.append(chosen)
                row=source_tables[cell['dataset']][(str(cell['seed']),cell['method'],source)]
                counts={name:int(votes[i]) for i,name in enumerate(classes)}
                correct=int((pred[a]==y[a]).sum())
                expected_row={'true_label':classes[y[a[0]]], 'majority_predicted_label':classes[chosen],
                              'window_count':len(a), 'correct_window_count':correct,
                              'window_accuracy':correct/len(a), 'majority_correct':bool(chosen==y[a[0]])}
                for key,value in expected_row.items():
                    if isinstance(value,bool):
                        observed=row[key]=='True'
                    elif isinstance(value,int):
                        observed=int(row[key])
                    elif isinstance(value,float):
                        observed=float(row[key])
                    else:
                        observed=row[key]
                    if isinstance(value,float):
                        match=np.isclose(observed,value,atol=1e-10,rtol=0)
                    else:
                        match=observed==value
                    if not match:
                        raise ValueError(f'Source CSV mismatch: {source}/{key}')
                if json.loads(row['prediction_counts'])!=counts:
                    raise ValueError('Source prediction counts mismatch.')
                source_id=hashlib.sha256(json.dumps(source,sort_keys=True,ensure_ascii=False,default=str).encode('utf-8')).hexdigest()[:16]
                label=f'{Path(source).name} [{source_id[:8]}]'
                if row['source_id']!=source_id or row['source_label']!=label:
                    raise ValueError('Source identifier mismatch.')
                if chosen!=y[a[0]]:
                    failed.append(label)
            sy=np.asarray(sy)
            sp=np.asarray(sp)
            source_f1,_=metrics(sy,sp,len(classes))
            expected=source_reports[cell['dataset']]['aggregate'][str(cell['seed'])][cell['method']]
            if abs(source_f1-expected['majority_vote_macro_f1'])>1e-10 or sorted(failed)!=sorted(expected['failed_source_files']):
                raise ValueError('Source metric mismatch.')
            checked+=1
        print(f'Verified {checked} paired window/source prediction cells; no training or model inference.')

if __name__=='__main__':
    main()

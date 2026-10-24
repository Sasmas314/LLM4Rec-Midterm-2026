"""Reproduce extraction, experiments, submission and analytical artifacts."""
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[key]='1'
os.environ.setdefault('POLARS_MAX_THREADS','4')
import argparse
import json
import time
from dataclasses import asdict
import numpy as np
import polars as pl
from src.data_loading import ROOT,read_template,save_json
from src.preprocessing import prepare,stamp
from src.interactions import extract
from src.validation import validate,period
from src.recommenders import ScoreEngine,ModelSpec,specifications
from src.submission import generate

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','preprocess','experiments','predict','report'],default='all')
    parser.add_argument('--data-dir')
    parser.add_argument('--force',action='store_true',help='Rebuild raw-data preprocessing and interval caches.')
    args=parser.parse_args(); start=time.perf_counter()
    cache,folder=prepare(args.data_dir,args.force)
    events=extract(cache,args.force)
    if args.stage=='preprocess': return
    users=pl.read_parquet(cache/'users.parquet')
    schedule=pl.read_parquet(cache/'schedule.parquet')
    items=np.sort(schedule['tv_show_id'].unique().to_numpy())
    template=read_template(folder)
    template_indices=template.join(users,on='user_id',how='left',maintain_order='left')['user_idx'].to_numpy()
    if args.stage in ['all','experiments']:
        spec=validate(events,schedule,items,users.height,template_indices)
        save_json(ROOT/'outputs/final_model.json',asdict(spec))
        if args.stage=='experiments': return
    if args.stage in ['all','predict']:
        spec=ModelSpec(**json.loads((ROOT/'outputs/final_model.json').read_text()))
        cutoff=stamp('2020-07-27'); end=stamp('2020-10-19')
        train=period(events,stamp('2020-03-09'),cutoff)
        hist=period(schedule,stamp('2020-03-09'),cutoff)
        future=period(schedule,cutoff,end)
        engine=ScoreEngine(train,hist,future,items,users.height,cutoff,template_indices)
        pred=generate(folder,users,engine,spec)
        final_predictions={s.name:engine.recommend(s) for s in specifications()}
        final_predictions['selected']=pred
        final_predictions['association']=final_predictions['association_s2_c0.03']
        np.savez_compressed(ROOT/'outputs/final_predictions.npz',user_idx=template_indices,**final_predictions)
    if args.stage in ['all','report']:
        from src.reporting import build_report
        build_report(cache,folder)
    save_json(ROOT/'outputs/run_summary.json',{'stage':args.stage,'runtime_seconds':time.perf_counter()-start,'seed':42,'status':'completed','timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()})
    print('Completed',args.stage,'in',round(time.perf_counter()-start,1),'seconds',flush=True)

if __name__=='__main__': main()

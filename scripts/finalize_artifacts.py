"""Prepare requested artifacts and independently audit recorded predictions."""
from pathlib import Path
import argparse,json,sys,shutil,hashlib,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import polars as pl
import yaml,psutil
from src.data_loading import ROOT,save_json,read_template
from src.preprocessing import stamp
from src.submission import check_submission

def ap(pred,truth):
    hits=0; score=0.; seen=set()
    for rank,item in enumerate(pred[:5],1):
        if item in truth and item not in seen: hits+=1;score+=hits/rank
        seen.add(item)
    return score/min(5,len(truth))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',required=True);args=parser.parse_args()
    out=ROOT/'outputs';res=ROOT/'results';res.mkdir(exist_ok=True);(ROOT/'configs').mkdir(exist_ok=True)
    cache=ROOT/'data/processed'; results=pl.read_csv(out/'validation_results.csv')
    specs=json.loads((out/'experiment_specs.json').read_text())
    results.with_row_index('experiment_id').write_csv(res/'experiments.csv')
    selection=json.loads((out/'selection.json').read_text()); model=selection['selected_model']
    config=json.loads((out/'final_model.json').read_text())
    (ROOT/'configs/best_model.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    shutil.copyfile(out/'submission.csv',ROOT/'submission.csv')
    users=pl.read_parquet(cache/'users.parquet');events=pl.read_parquet(cache/'interactions.parquet');schedule=pl.read_parquet(cache/'schedule.parquet')
    # Coverage is measured over unioned viewing intervals, not over broadcasts:
    # one interval may map to several shows, so those counts are not a percentage.
    from src.interactions import merge_intervals,intersect_intervals
    sessions=pl.read_parquet(cache/'sessions.parquet')
    historical=schedule.filter((pl.col('start_time')>=stamp('2020-03-09'))&(pl.col('end_time')<=stamp('2020-07-27')))
    total_intervals=0; mapped_intervals=0
    for (channel,),frame in sessions.partition_by('channel_id',as_dict=True).items():
        frame=frame.sort(['user_idx','start_time','end_time'])
        _,starts,ends=merge_intervals(frame['user_idx'].to_numpy().astype('int64'),frame['start_time'].to_numpy(),np.minimum(frame['end_time'].to_numpy(),stamp('2020-07-27')))
        total_intervals+=len(starts)
        b=historical.filter(pl.col('channel_id')==channel).sort('start_time')
        if b.height:
            ids,_,_=intersect_intervals(np.arange(len(starts),dtype=np.int64),starts,ends,b['start_time'].to_numpy(),b['end_time'].to_numpy(),b['broadcast_id'].to_numpy().astype('int64'))
            mapped_intervals+=len(np.unique(ids))
    save_json(res/'mapping_coverage.json',{'denominator':'All merged user-channel viewing intervals, clipped at historical end','merged_intervals':total_intervals,'intervals_overlapping_valid_annotated_broadcasts':mapped_intervals,'mapping_coverage':mapped_intervals/total_intervals,'note':'Any positive overlap before the 80% qualification rule. Zero-ID schedule rows are excluded.'})
    future=schedule.filter((pl.col('start_time')>=stamp('2020-07-27'))&(pl.col('end_time')<=stamp('2020-10-19')))
    template=read_template(args.data_dir);sub=pl.read_csv(ROOT/'submission.csv',schema_overrides={'user_id':pl.String,'tv_show_id':pl.String})
    summary=check_submission(template,sub,future['tv_show_id'].unique())
    checked=0
    for name,cutoff,finish in [('dev_4w','2020-04-06','2020-05-04'),('dev_2w','2020-04-20','2020-05-04'),('audit_12w','2020-05-04','2020-07-27')]:
        arr=np.load(out/f'predictions_{name}.npz')
        allowed=set(schedule.filter((pl.col('start_time')>=stamp(cutoff))&(pl.col('end_time')<=stamp(finish)))['tv_show_id'])
        for key in set(arr.files)-{'user_idx','truth'}:
            measured=float(np.mean([ap(pred,set(truth)) for pred,truth in zip(arr[key],arr['truth'])]))
            expected=results.filter((pl.col('split')==name)&(pl.col('model')==key))['map5'].item()
            assert abs(measured-expected)<1e-12,(key,name)
            assert all(len(set(row))==5 and all(int(i)>0 for i in row) for row in arr[key])
            if key!='historical_frequency_unfiltered': assert all(int(i) in allowed for row in arr[key] for i in row)
            checked+=1
    original=ROOT/'results/independent_audit.json'
    cases=pl.read_csv(out/'selected_users_analysis.csv',schema_overrides={'user_id':pl.String})
    assert cases['user_id'].n_unique()==2 and cases.height==60
    catalog=pl.read_csv(out/'program_catalog.csv');metadata={r['tv_show_id']:r for r in catalog.iter_rows(named=True)}
    for r in cases.iter_rows(named=True):
        assert r['tv_show_title']==metadata[r['tv_show_id']]['tv_show_title']
        assert r['category']==metadata[r['tv_show_id']]['tv_show_category']
    # Keep all tested variants for these same two cases, including per-row rationale.
    final=np.load(out/'final_predictions.npz');pos={int(u):i for i,u in enumerate(final['user_idx'])}
    summaries=json.loads((out/'selected_users_summary.json').read_text(encoding='utf-8')); all_cases=[]
    category_rows=[]
    for u in summaries:
        case_events=events.filter(pl.col('user_idx')==u['user_idx']).join(schedule.select('broadcast_id','tv_show_category').with_columns(pl.col('broadcast_id').cast(pl.Int64)),on='broadcast_id')
        for category,n in case_events.group_by('tv_show_category').len().sort('len',descending=True).iter_rows():
            category_rows.append({'case':u['case'],'user_id':u['user_id'],'category':category,'qualifying_events':n,'event_share':n/case_events.height})
        history=dict(events.filter(pl.col('user_idx')==u['user_idx']).group_by('tv_show_id').len().iter_rows())
        for spec in specs:
            for rank,item in enumerate(final[spec['name']][pos[u['user_idx']]],1):
                r=metadata[int(item)];n=history.get(int(item),0)
                all_cases.append({'case':u['case'],'user_id':u['user_id'],'model':spec['name'],'rank':rank,'tv_show_id':int(item),'title':r['tv_show_title'],'category':r['tv_show_category'],'historical_qualifying_views':n,'interpretation':f"{spec['family']} signal; {n} qualifying historical airings; {'repeat candidate' if n else 'discovery candidate'}"})
    pl.DataFrame(all_cases).write_csv(res/'two_users_all_configurations.csv')
    pl.DataFrame(category_rows).write_csv(res/'two_users_categories.csv')
    # Human-readable canonical interaction contract, distributed compressed locally.
    events.join(users.select('user_idx','user_id'),on='user_idx').select('user_id','tv_show_id',pl.lit(1.0).alias('interaction_weight'),pl.col('end_time').alias('timestamp'),'watched_seconds','screen_time_ratio').write_parquet(cache/'interaction_table.parquet')
    digest=hashlib.sha256((ROOT/'submission.csv').read_bytes()).hexdigest()
    save_json(original,{'status':'PASS','independent_map_rows':checked,'same_candidate_validation':True,'diagnostic_exception':'historical_frequency_unfiltered','two_user_rows':cases.height,'all_configuration_case_rows':len(all_cases),'metadata_verified':True,'submission':summary,'submission_sha256':digest})
    selected=results.filter(pl.col('model')==model).to_dicts()
    save_json(res/'final_model_metrics.json',{'selected_model':model,'configuration':config,'selection':selection,'validation':selected,'submission':summary,'submission_sha256':digest,'kaggle_score':None,'note':'Kaggle evidence is separate in external_evidence.json; offline MAP is not a Kaggle score.'})
    save_json(res/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'logical_cpus':psutil.cpu_count(),'ram_gib':psutil.virtual_memory().total/2**30,'free_disk_gib':shutil.disk_usage(ROOT).free/2**30})
    # Figures use only actual observations.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pop=pl.read_csv(out/'program_popularity.csv');cats=pl.read_csv(out/'popular_categorys.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    freq=np.sort(pop['qualifying_views'].to_numpy())[::-1]
    axes[0].loglog(np.arange(1,len(freq)+1),freq,color='#276a85');axes[0].set(xlabel='Program popularity rank',ylabel='Qualifying broadcast events',title='Long-tail frequency distribution')
    labels=cats['tv_show_category'].to_list();vals=cats['len'].to_list()
    axes[1].barh(labels[::-1],vals[::-1],color='#276a85');axes[1].set(xlabel='Qualifying broadcast events',title='Program category distribution')
    fig.tight_layout();fig.savefig(ROOT/'reports/figures/long_tail_categories.png',dpi=170);plt.close(fig)
    print('PASS',checked,'independent metric rows;',summary['rows'],'submission users;',len(all_cases),'case-study rows')

if __name__=='__main__':main()

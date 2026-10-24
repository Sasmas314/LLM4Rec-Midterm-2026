from dataclasses import asdict
import time
import json
import psutil
import numpy as np
import polars as pl
from .preprocessing import stamp
from .recommenders import ScoreEngine,specifications
from .evaluation import evaluate,per_user_metrics,paired_bootstrap
from .data_loading import ROOT,save_json

SPLITS=[('dev_4w','2020-04-06','2020-05-04','development'),('dev_2w','2020-04-20','2020-05-04','development'),('audit_12w','2020-05-04','2020-07-27','audit')]

def period(frame,start,end):
    return frame.filter((pl.col('start_time')>=start)&(pl.col('end_time')<=end))

def labels(events,users):
    counts=events.group_by(['user_idx','tv_show_id']).len().sort(['user_idx','len','tv_show_id'],descending=[False,True,False])
    groups=counts.group_by('user_idx',maintain_order=True).agg(pl.col('tv_show_id').head(5).alias('top5'),pl.len().alias('unique_programs'))
    # Primary cohort matches the competition's strictly five-label definition.
    eligible=groups.filter(pl.col('unique_programs')>=5).filter(pl.col('user_idx').is_in(users)).sort('user_idx')
    active=groups.filter(pl.col('user_idx').is_in(users)).sort('user_idx')
    return eligible,active

def validate(events,schedule,items,n_users,template_indices):
    records=[]; cohorts=[]; dev_scores={}; winner=None
    specs=specifications()
    save_json(ROOT/'outputs/experiment_specs.json',[asdict(s) for s in specs])
    for name,start,end,role in SPLITS:
        cutoff=stamp(start); finish=stamp(end)
        train=period(events,stamp('2020-03-09'),cutoff)
        target=period(events,cutoff,finish)
        hist=period(schedule,stamp('2020-03-09'),cutoff)
        future=period(schedule,cutoff,finish)
        eligible,active=labels(target,template_indices)
        idx=eligible['user_idx'].to_numpy(); truth=eligible['top5'].to_list()
        if not len(idx): raise ValueError('No users with five labels')
        eligible.write_parquet(ROOT/f'outputs/labels_{name}.parquet')
        cohorts.append({'split':name,'role':role,'cutoff':start,'end_exclusive':end,'horizon_days':(finish-cutoff)//86400,'train_events':train.height,'label_events':target.height,'template_users':len(template_indices),'eligible_five_label_users':len(idx),'active_users':active.height,'active_fewer_than_five_users':active.height-len(idx),'no_qualifying_validation_users':len(template_indices)-active.height,'boundary_broadcasts_excluded':schedule.filter((pl.col('start_time')<cutoff)&(pl.col('end_time')>cutoff)).height})
        # Score the same eligible users for every algorithm, plus secondary cohort.
        predict_users=active['user_idx'].to_numpy()
        lookup={int(u):j for j,u in enumerate(predict_users)}
        sel=np.array([lookup[int(u)] for u in idx])
        engine=ScoreEngine(train,hist,future,items,n_users,cutoff,predict_users)
        allowed=set(items[engine.candidates])
        candidate_label_coverage=float(np.mean([sum(x in allowed for x in row)/5 for row in truth]))
        predictions={}; metric_vectors={}
        print(f'{name}: {len(idx)} primary users, {active.height} active users',flush=True)
        for spec in specs:
            # Warm shared matrices belong to engine setup. Component fit and
            # scoring are timed separately; cached components are identified.
            cache_before=set(engine.cache)
            started=time.perf_counter()
            engine.score(spec)
            score_seconds=time.perf_counter()-started
            started=time.perf_counter()
            pred_all=engine.recommend(spec); pred=pred_all[sel]
            inference_seconds=time.perf_counter()-started
            metrics=evaluate(pred,truth)
            record={'split':name,'role':role,'model':spec.name,'family':spec.family,'users':len(idx),**metrics,
                    'parameters':json.dumps(asdict(spec),sort_keys=True),
                    'training_and_scoring_seconds':score_seconds,
                    'inference_seconds':inference_seconds,
                    'process_rss_mb':psutil.Process().memory_info().rss/2**20,
                    'new_component_cache_entries':len(set(engine.cache)-cache_before)}
            secondary=evaluate(pred_all,active['top5'].to_list())
            record['active_users_map5']=secondary['map5']
            record['candidate_label_coverage']=candidate_label_coverage
            records.append(record); predictions[spec.name]=pred
            metric_vectors[spec.name]=per_user_metrics(pred,truth)[:,0]
            print(f'  {spec.name}: MAP5={metrics["map5"]:.5f}',flush=True)
        np.savez_compressed(ROOT/f'outputs/predictions_{name}.npz',user_idx=idx,truth=np.asarray(truth),**predictions)
        if role=='development':
            for spec in specs: dev_scores.setdefault(spec.name,[]).append(next(r['map5'] for r in records if r['split']==name and r['model']==spec.name))
            if name==SPLITS[1][0]:
                ranking=sorted(((float(np.mean(v)),k) for k,v in dev_scores.items()),reverse=True)
                winner=ranking[0][1]
                save_json(ROOT/'outputs/selection.json',{'selection_metric':'Unweighted mean standard MAP@5 on the two development splits','selected_model':winner,'development_ranking':[{'model':k,'mean_map5':v} for v,k in ranking],'audit_used_for_selection':False})
                print('FROZEN SELECTION:',winner,flush=True)
        else:
            save_json(ROOT/'outputs/audit_uncertainty.json',paired_bootstrap(metric_vectors[winner],metric_vectors['frequency']))
            pl.DataFrame({'user_idx':idx,'selected_map5':metric_vectors[winner],'frequency_map5':metric_vectors['frequency']}).write_csv(ROOT/'outputs/audit_user_metrics.csv')
        del engine
    result=pl.DataFrame(records)
    result.write_csv(ROOT/'outputs/validation_results.csv')
    pl.DataFrame(cohorts).write_csv(ROOT/'outputs/validation_cohorts.csv')
    result.group_by(['model','family']).agg(pl.col('map5').mean().alias('mean_map5_all_splits'),pl.col('recall5').mean(),pl.col('ordered_agreement5').mean()).sort('mean_map5_all_splits',descending=True).write_csv(ROOT/'outputs/algorithm_comparison.csv')
    return next(s for s in specs if s.name==winner)

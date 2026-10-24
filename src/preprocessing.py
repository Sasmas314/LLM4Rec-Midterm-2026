"""Canonical timestamps are source-local, timezone-naive wall-clock seconds.

The source has no offsets. An explicit common offset can be supplied to
normalize two timezone-aware feeds before using the interval engine. No
unverifiable timezone or DST shift is guessed for this dataset.
"""
from datetime import datetime
import polars as pl
from .data_loading import ROOT, locate_data, provenance, save_json, read_template,inspect_inputs

def stamp(value):
    parsed=datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed=parsed.replace(tzinfo=__import__('datetime').timezone.utc)
    return int(parsed.timestamp())

def prepare(data_dir=None, force=False):
    folder, log_path, schedule_paths = locate_data(data_dir)
    cache=ROOT/'data/processed'; cache.mkdir(parents=True,exist_ok=True)
    for name in ['outputs','reports','notebooks']:
        (ROOT/name).mkdir(exist_ok=True)
    if not force and all((cache/f).exists() for f in ['sessions.parquet','schedule.parquet','users.parquet','quality.json']):
        import json
        old=json.loads((cache/'quality.json').read_text(encoding='utf-8'))
        if old['input_files']!=provenance([log_path,*schedule_paths,folder/'submission.csv']):
            raise ValueError('Input checksums changed. Use --force to rebuild preprocessing.')
        if not (ROOT/'outputs/data_inspection.json').exists(): inspect_inputs(folder)
        return cache,folder
    inspect_inputs(folder)
    logs=pl.read_csv(log_path,schema_overrides={'user_id':pl.String})
    users=logs.select('user_id').unique().sort('user_id').with_row_index('user_idx')
    users.write_parquet(cache/'users.parquet')
    logs=logs.rename({'vsetv_id':'channel_id','stop_time':'end_time','duraton':'duration'})
    for c in ['start_time','end_time']:
        logs=logs.with_columns(pl.col(c).str.to_datetime('%Y-%m-%d %H:%M:%S',strict=False).dt.epoch('s').alias(c))
    q={'log_rows_raw':logs.height,'log_duplicates':logs.height-logs.unique().height,
       'log_nulls':logs.null_count().row(0,named=True),
       'duration_timestamp_mismatches':logs.filter(pl.col('duration')!=pl.col('end_time')-pl.col('start_time')).height,
       'invalid_sessions':logs.filter((pl.col('end_time')<=pl.col('start_time'))|pl.col('start_time').is_null()|pl.col('end_time').is_null()).height,
       'timezone_assumption':'Common source-local wall time; no source offsets supplied. No guessed timezone conversion.'}
    logs=(logs.unique().filter((pl.col('end_time')>pl.col('start_time'))&pl.col('start_time').is_not_null()&pl.col('end_time').is_not_null())
          .join(users,on='user_id').select('user_idx','channel_id','start_time','end_time','duration'))
    q.update(users=users.height,channels_watched=logs['channel_id'].n_unique(),session_start_min=logs['start_time'].min(),session_start_max=logs['start_time'].max(),session_end_max=logs['end_time'].max())
    logs.write_parquet(cache/'sessions.parquet')
    del logs
    schedules=[]; q['schedule_files']=[]
    for path in schedule_paths:
        f=pl.read_csv(path,infer_schema_length=10000)
        info={'file':path.name,'rows':f.height,'missing':f.null_count().row(0,named=True),'duplicates':f.height-f.unique().height}
        f=f.with_columns(pl.col('start_time').str.to_datetime('%d.%m.%Y %H:%M:%S',strict=False).dt.epoch('s'))
        info.update(invalid_start=f['start_time'].null_count(),invalid_duration=f.filter(pl.col('duration')<=0).height,zero_id_rows=f.filter(pl.col('tv_show_id')==0).height)
        q['schedule_files'].append(info)
        schedules.append(f)
    schedule=pl.concat(schedules).with_columns((pl.col('start_time')+pl.col('duration')).alias('end_time'))
    q['schedule_rows_raw']=schedule.height
    schedule=schedule.filter((pl.col('duration')>0)&(pl.col('tv_show_id')>0)&pl.col('start_time').is_not_null())
    before=schedule.height
    schedule=schedule.unique(subset=['channel_id','start_time','end_time','tv_show_id'],maintain_order=True).sort(['channel_id','start_time','end_time','tv_show_id']).with_row_index('broadcast_id')
    q.update(schedule_semantic_duplicates=before-schedule.height,retained_nonzero_broadcasts=schedule.height,programs=schedule['tv_show_id'].n_unique(),schedule_channels=schedule['channel_id'].n_unique())
    # Keep conflicting distinct program annotations, but make this visible.
    q['conflicting_channel_start_slots']=schedule.group_by(['channel_id','start_time']).len().filter(pl.col('len')>1).height
    q['schedule_overlapping_previous_end']=schedule.with_columns(pl.col('end_time').cum_max().shift().over('channel_id').alias('previous_max_end')).filter(pl.col('start_time')<pl.col('previous_max_end')).height
    schedule.write_parquet(cache/'schedule.parquet')
    q['input_files']=provenance([log_path,*schedule_paths,folder/'submission.csv'])
    q['template_users']=read_template(folder).height
    save_json(cache/'quality.json',q)
    print('Preprocessing:',q['users'],'users;',q['retained_nonzero_broadcasts'],'nonzero broadcasts',flush=True)
    return cache,folder

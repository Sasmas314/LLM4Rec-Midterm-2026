"""Union viewing intervals, intersect with broadcasts, and apply >=80% rule.

Half-open [start,end) intervals. Numba + per-channel prefix-end index avoids
the all-session/all-broadcast Cartesian product and handles overlapping EPG.
"""
import numpy as np
import polars as pl
from numba import njit
from .data_loading import save_json
from .preprocessing import stamp

@njit(cache=True)
def merge_intervals(users, starts, ends):
    n=len(users)
    ou=np.empty(n,np.int64); os=np.empty(n,np.int64); oe=np.empty(n,np.int64)
    k=0
    for i in range(n):
        if ends[i]<=starts[i]: continue
        if k and users[i]==ou[k-1] and starts[i]<=oe[k-1]:
            oe[k-1]=max(oe[k-1],ends[i])
        else:
            ou[k]=users[i]; os[k]=starts[i]; oe[k]=ends[i]; k+=1
    return ou[:k],os[:k],oe[:k]

@njit(cache=True)
def intersect_intervals(users, starts, ends, bstarts, bends, bids):
    prefix=bends.copy()
    for j in range(1,len(prefix)):
        prefix[j]=max(prefix[j-1],prefix[j])
    out_u=[]; out_b=[]; out_s=[]
    for i in range(len(users)):
        lo=np.searchsorted(prefix,starts[i],side='right')
        hi=np.searchsorted(bstarts,ends[i],side='left')
        for j in range(lo,hi):
            overlap=min(ends[i],bends[j])-max(starts[i],bstarts[j])
            if overlap>0:
                out_u.append(users[i]); out_b.append(bids[j]); out_s.append(overlap)
    return np.array(out_u,np.int64),np.array(out_b,np.int64),np.array(out_s,np.int64)

def extract(cache,force=False):
    target=cache/'interactions.parquet'
    if target.exists() and not force: return pl.read_parquet(target)
    sessions=pl.read_parquet(cache/'sessions.parquet')
    # Training logs end at the historical cutoff, even for spillover sessions.
    sessions=sessions.with_columns(pl.col('end_time').clip(upper_bound=stamp('2020-07-27')).alias('end_time'))
    schedule=pl.read_parquet(cache/'schedule.parquet').filter((pl.col('start_time')>=stamp('2020-03-09'))&(pl.col('end_time')<=stamp('2020-07-27')))
    chunks=[]; merged_count=0; pieces=0
    for (channel,), frame in sessions.partition_by('channel_id',as_dict=True).items():
        broadcasts=schedule.filter(pl.col('channel_id')==channel).sort('start_time')
        if not broadcasts.height: continue
        frame=frame.sort(['user_idx','start_time','end_time'])
        u,s,e=merge_intervals(frame['user_idx'].to_numpy().astype('int64'),frame['start_time'].to_numpy(),frame['end_time'].to_numpy())
        merged_count+=len(u)
        a,b,c=intersect_intervals(u,s,e,broadcasts['start_time'].to_numpy(),broadcasts['end_time'].to_numpy(),broadcasts['broadcast_id'].to_numpy().astype('int64'))
        pieces+=len(a)
        if len(a): chunks.append(pl.DataFrame({'user_idx':a,'broadcast_id':b,'watched_seconds':c}))
    if not chunks: raise ValueError('No overlapping intervals; inspect timestamps/channel IDs.')
    overlaps=pl.concat(chunks).group_by(['user_idx','broadcast_id']).agg(pl.col('watched_seconds').sum())
    matched=overlaps.join(schedule.select('broadcast_id','tv_show_id','channel_id','start_time','end_time','duration').with_columns(pl.col('broadcast_id').cast(pl.Int64)),on='broadcast_id')
    if matched.filter(pl.col('watched_seconds')>pl.col('duration')).height: raise AssertionError('Double-counted coverage')
    matched=matched.with_columns((pl.col('watched_seconds')/pl.col('duration')).alias('screen_time_ratio'))
    qualified=matched.filter(pl.col('watched_seconds')*5>=pl.col('duration')*4).sort(['user_idx','start_time','broadcast_id'])
    if qualified.select(pl.struct('user_idx','broadcast_id').n_unique()).item()!=qualified.height: raise AssertionError('Duplicate events')
    qualified.write_parquet(target)
    save_json(cache/'interaction_quality.json',{'merged_intervals_on_matched_channels':merged_count,'overlap_pieces':pieces,'user_broadcast_overlaps':matched.height,'qualifying_events':qualified.height,'qualifying_users':qualified['user_idx'].n_unique(),'qualifying_programs':qualified['tv_show_id'].n_unique(),'min_ratio':qualified['screen_time_ratio'].min(),'max_ratio':qualified['screen_time_ratio'].max()})
    print('Qualifying events:',qualified.height,flush=True)
    return qualified

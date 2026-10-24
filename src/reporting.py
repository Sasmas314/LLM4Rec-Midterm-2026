"""Measured EDA, two auditable user case studies, figures and exam report."""
from pathlib import Path
from datetime import datetime,timezone
import json
import math
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data_loading import ROOT,save_json,read_template
from .preprocessing import stamp
from .validation import period

def table(frame,columns=None,limit=30):
    if columns: frame=frame.select(columns)
    def fmt(x):
        if isinstance(x,float): return f'{x:.4f}'
        return str(x).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*frame.width)+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.head(limit).iter_rows())

def catalog(schedule):
    # Most common annotation per ID, ties lexically resolved; keep original Cyrillic.
    fields=['tv_show_title','channel_title','tv_show_category','tv_show_genre_1','director','actors']
    out=schedule.select('tv_show_id').unique()
    for c in fields:
        common=(schedule.filter(pl.col(c).is_not_null()).group_by(['tv_show_id',c]).len()
                .sort(['tv_show_id','len',c],descending=[False,True,False]).unique('tv_show_id',keep='first',maintain_order=True).select('tv_show_id',c))
        out=out.join(common,on='tv_show_id',how='left')
    return out.sort('tv_show_id')

def build_report(cache,folder):
    out=ROOT/'outputs'; figures=ROOT/'reports/figures'; figures.mkdir(parents=True,exist_ok=True)
    events=pl.read_parquet(cache/'interactions.parquet')
    sessions=pl.read_parquet(cache/'sessions.parquet')
    schedule=pl.read_parquet(cache/'schedule.parquet')
    users=pl.read_parquet(cache/'users.parquet')
    q=json.loads((cache/'quality.json').read_text(encoding='utf-8'))
    iq=json.loads((cache/'interaction_quality.json').read_text(encoding='utf-8'))
    inspection=json.loads((out/'data_inspection.json').read_text(encoding='utf-8')) if (out/'data_inspection.json').exists() else {}
    cat=catalog(schedule); cat.write_csv(out/'program_catalog.csv')
    hist=period(schedule,stamp('2020-03-09'),stamp('2020-07-27'))
    future=period(schedule,stamp('2020-07-27'),stamp('2020-10-19'))
    counts=events.group_by(['user_idx','tv_show_id']).len().rename({'len':'qualifying_views'})
    activity=counts.group_by('user_idx').agg(pl.col('qualifying_views').sum().alias('events'),pl.len().alias('unique_programs'),pl.col('qualifying_views').max().alias('top1_views'),pl.col('qualifying_views').sort(descending=True).head(5).sum().alias('top5_views'))
    activity=users.join(activity,on='user_idx',how='left').with_columns(pl.col('events','unique_programs','top1_views','top5_views').fill_null(0))
    activity=activity.with_columns((pl.col('top1_views')/pl.col('events').clip(lower_bound=1)).alias('top1_share'),(pl.col('top5_views')/pl.col('events').clip(lower_bound=1)).alias('top5_share'))
    activity.write_csv(out/'user_activity.csv')
    watched=set(counts['tv_show_id'].unique()); scheduled=set(hist['tv_show_id'].unique()); future_ids=set(future['tv_show_id'].unique())
    active=activity.filter(pl.col('events')>0)
    sparse_stats={'all_users':users.height,'qualifying_users':active.height,'qualifying_programs':len(watched),'user_program_pairs':counts.height,'sparsity_qualifying_users_programs':1-counts.height/(active.height*len(watched)),
                  'unique_programs_mean_active':active['unique_programs'].mean(),'unique_programs_median_active':active['unique_programs'].median(),
                  'unique_programs_mean_all':activity['unique_programs'].mean(),'unique_programs_median_all':activity['unique_programs'].median(),
                  'events_median_active':active['events'].median(),'events_p90_active':active['events'].quantile(.9),'events_max_active':active['events'].max(),
                  'no_qualifying_history_users':activity.filter(pl.col('events')==0).height,'one_to_five_program_users':activity.filter(pl.col('unique_programs').is_between(1,5)).height,
                  'historical_scheduled_programs':len(scheduled),'future_scheduled_programs':len(future_ids),'scheduled_program_overlap':len(scheduled&future_ids),
                  'watched_programs_returning':len(watched&future_ids),'future_unwatched_programs':len(future_ids-watched),'future_new_to_historical_schedule':len(future_ids-scheduled),
                  'future_broadcasts':future.height,'historical_broadcasts':hist.height,
                  'unmatched_viewing_channels':sorted(set(sessions['channel_id'].unique())-set(hist['channel_id'].unique()))}
    template=read_template(folder).join(users,on='user_id',maintain_order='left')
    template_activity=activity.filter(pl.col('user_idx').is_in(template['user_idx'].implode()))
    sparse_stats['template_no_qualifying_history']=template_activity.filter(pl.col('events')==0).height
    popularity=counts.group_by('tv_show_id').agg(pl.col('qualifying_views').sum(),pl.len().alias('viewers')).sort('qualifying_views',descending=True).join(cat,on='tv_show_id',maintain_order='left')
    popularity.write_csv(out/'program_popularity.csv')
    sparse_stats['top10_program_event_share']=popularity['qualifying_views'].head(10).sum()/events.height
    sparse_stats['programs_five_or_fewer_viewers']=popularity.filter(pl.col('viewers')<=5).height
    sparse_stats['events_on_five_or_fewer_viewer_programs']=popularity.filter(pl.col('viewers')<=5)['qualifying_views'].sum()
    availability=hist.group_by('tv_show_id').len().rename({'len':'historical_broadcasts'}).join(future.group_by('tv_show_id').len().rename({'len':'future_broadcasts'}),on='tv_show_id',how='full',coalesce=True).fill_null(0).join(cat,on='tv_show_id')
    availability=availability.with_columns((pl.col('future_broadcasts')/12/(pl.col('historical_broadcasts')/20+1e-6)).alias('weekly_opportunity_ratio'))
    availability.write_csv(out/'program_availability.csv')
    enriched=events.join(schedule.select('broadcast_id','channel_title','tv_show_genre_1','tv_show_category').with_columns(pl.col('broadcast_id').cast(pl.Int64)),on='broadcast_id')
    for column,label in [('channel_title','channel'),('tv_show_genre_1','genre'),('tv_show_category','category')]:
        enriched.group_by(column).len().sort('len',descending=True).write_csv(out/f'popular_{label}s.csv')
    weekly=events.with_columns(((pl.col('start_time')-stamp('2020-03-09'))//604800+11).alias('iso_week')).group_by('iso_week').len().sort('iso_week')
    weekly.write_csv(out/'weekly_activity.csv')
    hours=sessions.with_columns(((pl.col('start_time')//3600)%24).alias('hour')).group_by('hour').agg(pl.len().alias('sessions'),(pl.col('end_time')-pl.col('start_time')).sum().alias('watched_seconds')).sort('hour')
    hours.write_csv(out/'hourly_activity.csv')
    raw_weeks=sessions.with_columns(((pl.col('start_time')-stamp('2020-03-09'))//604800+11).alias('iso_week')).group_by('iso_week').agg(pl.len().alias('raw_sessions'))
    schedule_weeks=hist.with_columns(((pl.col('start_time')-stamp('2020-03-09'))//604800+11).alias('iso_week')).group_by('iso_week').agg(pl.len().alias('nonzero_epg_broadcasts'))
    weekly.join(raw_weeks,on='iso_week').join(schedule_weeks,on='iso_week').sort('iso_week').write_csv(out/'weekly_coverage.csv')
    # Two ten-week windows; this is descriptive only, not used to tune the held-out model.
    first=period(events,stamp('2020-03-09'),stamp('2020-05-18'))
    second=period(events,stamp('2020-05-18'),stamp('2020-07-27'))
    def top_sets(frame):
        agg=frame.group_by(['user_idx','tv_show_id']).len().sort(['user_idx','len','tv_show_id'],descending=[False,True,False])
        return {int(k[0]):set(f['tv_show_id'].head(5)) for k,f in agg.partition_by('user_idx',as_dict=True).items()}
    a,b=top_sets(first),top_sets(second)
    persistence=[{'user_idx':u,'jaccard_top5':len(a[u]&b[u])/len(a[u]|b[u]),'common_top5':len(a[u]&b[u])} for u in sorted(a.keys()&b.keys())]
    pl.DataFrame(persistence).write_csv(out/'interest_persistence.csv')
    sparse_stats['persistence_users_both_halves']=len(persistence)
    sparse_stats['mean_top5_jaccard_between_ten_week_halves']=float(np.mean([r['jaccard_top5'] for r in persistence]))
    save_json(out/'eda_statistics.json',sparse_stats)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].hist(active['events'].to_numpy(),bins=np.geomspace(1,active['events'].max()+1,40),color='#276a85'); axes[0,0].set_xscale('log'); axes[0,0].set_title('Qualifying broadcasts per active user'); axes[0,0].set_ylabel('Users')
    axes[0,1].hist(active['unique_programs'].to_numpy(),bins=40,color='#df9d49'); axes[0,1].set_title('Distinct programs per active user')
    axes[1,0].plot(weekly['iso_week'],weekly['len'],marker='o',color='#276a85'); axes[1,0].set_title('Qualifying events by ISO week'); axes[1,0].set_xlabel('Week of 2020'); axes[1,0].set_xticks(range(11,31,2))
    axes[1,1].bar(hours['hour'],hours['sessions'],color='#6e799d'); axes[1,1].set_title('Session starts by source-local hour'); axes[1,1].set_xlabel('Hour')
    fig.tight_layout(); fig.savefig(figures/'data_overview.png'); plt.close(fig)
    results=pl.read_csv(out/'validation_results.csv'); cohorts=pl.read_csv(out/'validation_cohorts.csv')
    selection=json.loads((out/'selection.json').read_text()); winner=selection['selected_model']
    final_spec=json.loads((out/'final_model.json').read_text())
    audit=results.filter(pl.col('role')=='audit').sort('map5',descending=True)
    uncertainty=json.loads((out/'audit_uncertainty.json').read_text())
    from .evaluation import evaluate
    frozen=np.load(out/'predictions_audit_12w.npz')
    train_unique=dict(period(events,stamp('2020-03-09'),stamp('2020-05-04')).group_by('user_idx').agg(pl.col('tv_show_id').n_unique()).iter_rows())
    cohorts_detail=[]
    sizes=np.array([train_unique.get(int(u),0) for u in frozen['user_idx']])
    for name,mask in [('cold_0',sizes==0),('sparse_1_to_5',(sizes>=1)&(sizes<=5)),('medium_6_to_20',(sizes>=6)&(sizes<=20)),('broad_over_20',sizes>20)]:
        if not mask.any(): continue
        for model in [winner,'frequency','content','association_s2_c0.03']:
            cohorts_detail.append({'cohort':name,'model':model,'users':int(mask.sum()),**evaluate(frozen[model][mask],frozen['truth'][mask])})
    breakdown=pl.DataFrame(cohorts_detail); breakdown.write_csv(out/'audit_cohort_breakdown.csv')
    fig,ax=plt.subplots(figsize=(10,5)); view=audit.head(12).sort('map5')
    ax.barh(view['model'],view['map5'],color=['#df9d49' if m==winner else '#276a85' for m in view['model']]); ax.set_xlabel('Standard MAP@5'); ax.set_title('Frozen 12-week audit: top 12 of all evaluated methods'); fig.tight_layout(); fig.savefig(figures/'validation_comparison.png'); plt.close(fig)
    case_text,case_summary=case_studies(template_activity,counts,enriched,cat,future,users)
    save_json(out/'selected_users_summary.json',case_summary)
    chosen=audit.filter(pl.col('model')==winner).row(0,named=True)
    baseline=audit.filter(pl.col('model')=='frequency').row(0,named=True)
    am={r['model']:r['map5'] for r in audit.iter_rows(named=True)}
    dev=results.filter(pl.col('role')=='development').group_by('model').agg(pl.col('map5').mean().alias('development_mean_map5')).sort('development_mean_map5',descending=True)
    dev.write_csv(out/'development_selection.csv')
    report=f'''# SWEET.TV: Frequency-Oriented Television Recommendation

Master's-level Business Analytics examination project. All statistics and experiments below were computed from the supplied CSV files. No leaderboard score is claimed. Code, input checksums, prediction arrays and per-user audit scores accompany this report.

## 1. Business objective and decision criterion

Recommend exactly five television program identifiers to each of {q['template_users']:,} specified subscribers for the twelve weeks beginning 27 July 2020. The service benefits when a recommendation corresponds to a program a subscriber repeatedly watches substantially, rather than a title matching a broad genre. Repeat recommendations are legitimate: the target is future preference, not novel-item discovery alone. A production launch would test viewing engagement, incremental retention, coverage and discovery in an A/B experiment; the offline scores here establish predictive accuracy, not causal business impact.

The competition's [Evaluation section](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/overview/evaluation) names **MAP@5**. Its [Data section](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data) defines frequency after at least 80% broadcast coverage and states that public/private subsets split users 50/50. These pages were inspected in the authenticated browser. The server-side scorer source and hidden labels are unavailable; we implement conventional binary MAP@5 and do not claim byte-for-byte verification of the private evaluator.

For a predicted list p, AP@5 = sum over ranks k=1..5 of Precision@k × 1[p_k is in the target set] / min(5, number of target labels). MAP averages AP across users. Reversing five correct labels preserves conventional AP=1; the order of *hits versus misses* matters. The source simultaneously requests decreasing preference order. We therefore output decreasing model scores and report positional agreement, exact ordered match and graded NDCG separately. Graded gains are 5,4,3,2,1 by target frequency rank. Binary NDCG, recall and precision use the target Top-5 set. With five labels and five predictions, recall equals precision.

## 2. Dataset and exploration

There are {q['log_rows_raw']:,} raw sessions, {q['users']:,} users and {q['channels_watched']} observed viewing channels. The raw schedule contains {q['schedule_rows_raw']:,} rows across three files. Excluding program zero and invalid durations leaves {q['retained_nonzero_broadcasts']:,} annotated broadcasts and {q['programs']:,} nonzero program IDs across the entire supplied schedule. Only {sparse_stats['historical_broadcasts']:,} broadcasts lie fully inside the historical period, and {sparse_stats['future_broadcasts']:,} lie fully inside the specified future period. Session starts span 9 March–26 July 2020. Schedules contain spillovers beyond their nominal week boundaries, so filtering uses timestamps rather than filenames.

Actual filenames are `dataset11-30.csv`, `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv`, and `submission.csv`. The loader maps `stop_time` to end time and the misspelled `duraton` to the source duration. User IDs are decimal strings throughout CSV parsing, joins and output: many exceed signed 64-bit range and floating-point exactness.

There are {q['log_duplicates']} exact duplicate sessions, {q['duration_timestamp_mismatches']} disagreements between supplied duration and endpoint subtraction, and {q['invalid_sessions']} invalid session intervals. Endpoints determine watched time. The schedule has {sum(f['invalid_duration'] for f in q['schedule_files'])} nonpositive duration row(s); they are removed. There are {q['schedule_semantic_duplicates']} duplicate nonzero broadcast keys and {q['conflicting_channel_start_slots']} conflicting same-channel start slots after canonicalization. {q['schedule_overlapping_previous_end']} retained broadcasts overlap a previous annotated broadcast; the matcher handles overlapping schedule intervals instead of assuming monotonically increasing ends. Full per-file schemas, missingness, date ranges and checksums are in `outputs/data_inspection.json` and `data/processed/quality.json`. Text is UTF-8 with intact Cyrillic; no replacement characters were found. Metadata absence is substantial, especially actors/directors and production year. Missing metadata contributes zero feature weight, rather than an invented genre.

After interval matching, {iq['user_broadcast_overlaps']:,} distinct user-broadcast overlaps yield **{events.height:,} qualifying broadcast events**, involving {active.height:,} users and {len(watched):,} programs. The {counts.height:,} nonzero user-program pairs imply {sparse_stats['sparsity_qualifying_users_programs']:.2%} sparsity among active users and watched programs. Active users watch a mean of {sparse_stats['unique_programs_mean_active']:.2f} distinct qualifying programs (median {sparse_stats['unique_programs_median_active']:.1f}); including all logged users gives mean {sparse_stats['unique_programs_mean_all']:.2f} and median {sparse_stats['unique_programs_median_all']:.1f}. The median active user's event count is {sparse_stats['events_median_active']:.1f}, the 90th percentile {sparse_stats['events_p90_active']:.1f}, and the maximum {sparse_stats['events_max_active']:,}. This heterogeneity motivates row normalization before blending.

{sparse_stats['no_qualifying_history_users']} logged users have no qualifying history, including {sparse_stats['template_no_qualifying_history']} submission users. {sparse_stats['one_to_five_program_users']} users have only one to five qualifying programs. {sparse_stats['programs_five_or_fewer_viewers']} watched programs have at most five distinct viewers; they account for {sparse_stats['events_on_five_or_fewer_viewer_programs']:,} events. The ten most frequent programs account for {sparse_stats['top10_program_event_share']:.2%} of all qualifying events. Long-tail rules need support regularization, while global popularity alone risks recommending mass-market titles to subscribers with narrow habits.

![Activity and temporal patterns](figures/data_overview.png)

The ten most frequently watched programs are:

{table(popularity,['tv_show_id','tv_show_title','qualifying_views','viewers'],10)}

The leading channels by qualifying events are:

{table(pl.read_csv(out/'popular_channels.csv'),limit=8)}

The leading first-genre annotations are:

{table(pl.read_csv(out/'popular_genres.csv'),limit=8)}

Genre totals above assign each broadcast its first genre; nulls remain visible and counts should not be interpreted as complete genre taxonomy. `popular_categories.csv`, hourly and weekly CSVs provide additional detail. Session-start timing differs from qualifying event timing: a long session can cover multiple broadcasts.

The session-start peak is {int(hours.sort('sessions',descending=True)['hour'][0]):02d}:00 source-local time. Weeks 25–27 show a sharp qualifying-event trough while annotated schedule counts remain roughly stable: week 26 has 9,948 qualifying events, 126,940 raw sessions and 12,604 nonzero EPG broadcasts, compared with 49,302 events, 261,697 sessions and 12,686 EPG broadcasts in week 29. This observed nonstationarity can distort recency and exposure-rate estimates. It cannot be attributed uniquely to preference change, logging coverage or household availability from these files alone. `weekly_coverage.csv` makes this diagnostic reproducible.

## 3. Interval preprocessing and ground truth

Each session and broadcast is a half-open interval [start,end). Sessions are sorted by user, channel and start. Duplicate, overlapping and touching intervals for the same user/channel are merged into their union. For each channel, sorted broadcast starts and a prefix maximum of broadcast ends support two binary searches per merged session. Every actual overlap contributes max(0, min(session_end,broadcast_end) − max(session_start,broadcast_start)). Pieces from disjoint sessions are summed for the same (user,broadcast occurrence); already merged session intervals guarantee no second is counted twice. A broadcast qualifies exactly when 5 × watched_seconds ≥ 4 × scheduled_duration. The implementation asserts that coverage never exceeds broadcast duration and that each user/occurrence pair is unique. It does not join by equal start timestamps.

This costs O(S log B + K) after sorting, where S is merged sessions, B broadcasts per channel and K actual overlaps, rather than the session–broadcast Cartesian product. {iq['overlap_pieces']:,} overlap pieces were processed. The engine handles partial sessions, midnight crossings, overlapping EPG entries and several sessions per broadcast. Synthetic cases and 60 randomized brute-force union comparisons verify matching. Timestamp-aware test inputs normalize to common UTC before intersection. Actual input timestamps have **no timezone offsets**; the project consistently interprets both feeds in the same source-local wall time. A hidden mismatch between the source clocks cannot be inferred or corrected without evidence; no arbitrary timezone or DST shift is introduced.

Frequency is the number of qualifying broadcast occurrences, not total viewing seconds. Validation targets group qualifying events by user/program and retain five highest counts. Ties use ascending numeric program ID because the organizer's tie rule is unspecified. Full broadcast intervals must fit entirely inside a validation segment; broadcasts crossing a split boundary are omitted instead of allowing post-cutoff seconds to enter training. Matching is cached across historical data, but event features are always filtered by broadcast start/end within the training interval. Merging sessions across a cutoff cannot change coverage inside a fully pre-cutoff broadcast. Future user behavior is never available to final training.

## 4. Algorithms and candidate availability

**Frequency baselines.** Global program event counts; personal all-history event counts ignoring availability; personal counts restricted to the prediction schedule; exponentially decayed personal counts with half-lives 7,14,28,56 days; and a 28-day history window. Repeat items are retained. Global training popularity resolves empty-history and zero-score ties, with ascending program ID as the last deterministic tie breaker.

**Content-Based Filtering.** Each program has TF-IDF title unigrams/bigrams, categorical category/genre tokens, and actors/director/production-period text. The three blocks receive weights 1, metadata_weight (2 or 4), and 0.5, then each item vector is L2-normalized. Up to twelve distinct annotations per field and ID are aggregated deterministically, accommodating episode-title variation. User profiles sum item vectors weighted by 28-day-decayed qualifying frequency and are normalized. Cosine similarity scores all future candidates, including unseen programs with metadata. No item-ID feature is included. Vectorizers use only the training schedule plus the schedule of that simulated future period, consistent with the final information available at prediction time.

**Association rules.** The binary user/program incidence matrix yields co-occurrence support C_ij. Confidence is C_ij / support(i). The rule weight is confidence × C_ij/(C_ij+5) / sqrt(support(j)/number_of_users), a shrunk, partially lift-adjusted directional association. Diagonal rules are removed. Experiments use support 2 or 5 users, confidence 0.03 or 0.10 and at most 120 outgoing neighbors. A user's log(1+decayed qualifying counts) weights outgoing rule scores. This model uses shared audiences, unlike metadata cosine. Removing self-rules makes it a discovery-oriented standalone comparison, while a hybrid restores the strong repeat-preference signal.

**Implicit matrix factorization.** CPU implicit ALS uses confidence input 10 × log(1+28-day-decayed counts), 25 iterations, 32 factors/regularization 0.1 or 64 factors/regularization 0.5, seed 42 and two ALS worker threads. Unknown items receive zero collaborative score and use the common fallback. The input is positive implicit frequency, not explicit ratings. Two configurations constitute a limited rather than exhaustive search.

**Opportunity and timing signals.** Counts are multiplied by ((future_airings+2)/(historical_airings+2))^p for p in 0.25,0.5,1.0, with both raw and decayed histories. A per-user channel × four-hour-slot propensity divides decayed qualifying events by historical slot broadcast exposure +10, then sums propensities over candidate future airings. Thus a familiar show with fewer future opportunities can move down, and a new show can receive a channel/time score. Genre affinity is modeled through content features. These are broadcast-view propensity proxies, not calibrated probabilities: overlapping channels, individual availability and exposure censoring prevent that interpretation.

**Hybrids.** Per-user nonnegative component scores are divided by their row maximum before a weighted sum. The base is frequency, optionally opportunity-adjusted. Experiments add content and association weights 0.05,0.10,0.20 and selected one-component, ALS and channel variants. A reciprocal-rank fusion uses 1/(rank+5) for positive scores. The exact configuration of every run is saved in `experiment_specs.json`; no assumption that a complex hybrid must win is made.

The future window contains {len(future_ids):,} program IDs. Of {len(scheduled):,} historically scheduled IDs, {len(scheduled&future_ids):,} return. Of {len(watched):,} actually watched IDs, {len(watched&future_ids):,} return; {len(future_ids-watched):,} future candidates lack a qualifying historical interaction and {len(future_ids-scheduled):,} did not appear in the historical schedule. Normalized per-week availability changes are exported in `program_availability.csv`. Programs absent from the future schedule cannot receive final recommendations. The unfiltered baseline intentionally measures the loss from ignoring this constraint. Descriptively, Top-5 interest-set Jaccard between two ten-week historical halves averages {sparse_stats['mean_top5_jaccard_between_ten_week_halves']:.3f} among {len(persistence):,} users active in both, indicating imperfect persistence; this statistic is not used for tuning.

## 5. Leakage-free development and long-horizon audit

All models use the identical eligible submission-user cohort in each split. The primary cohort requires at least five distinct qualifying target programs. Users with fewer than five are not padded with fabricated labels: they are evaluated separately with their actual one-to-four labels in `active_users_map5`. Users with no qualifying future events have no target and are excluded, with coverage explicitly reported.

{table(cohorts,['split','role','cutoff','end_exclusive','horizon_days','train_events','eligible_five_label_users','active_fewer_than_five_users','no_qualifying_validation_users'])}

Both development windows end on 4 May. Their labels overlap, so their scores are correlated, not two independent replications. Model selection uses their **unweighted mean MAP@5**. The selected configuration is saved before entering the final audit. The audit trains on eight weeks (9 March–4 May) and predicts the next twelve weeks (4 May–27 July), matching the competition's target horizon. Audit labels do not overlap either development label period. They are never used to retune or replace the selected model. Final training can use all twenty historical weeks after this evaluation. Because the final history is longer and the season differs, audit accuracy is an estimate, not a guarantee. Hyperparameters selected on short development horizons can fail over twelve weeks; this design exposes that risk.

Leading development results:

{table(dev,limit=12)}

## 6. Quantitative results and controlled experiments

The project measured {results['model'].n_unique()} configurations on each of three splits. All results, including unsuccessful configurations, are in `validation_results.csv`. `algorithm_comparison.csv` is a descriptive average across all splits and **must not be used retrospectively to select against the held-out audit**.

{table(audit,['model','family','map5','recall5','ndcg5','ordered_agreement5','exact_ordered5','graded_ndcg5'],40)}

![Audit model comparison](figures/validation_comparison.png)

The frozen selected model is **{winner}**, with development mean MAP@5 {selection['development_ranking'][0]['mean_map5']:.5f}. On the twelve-week audit it achieves MAP@5 **{chosen['map5']:.5f}**, recall/precision {chosen['recall5']:.5f} and ordered agreement {chosen['ordered_agreement5']:.5f}. The availability-filtered frequency baseline obtains MAP@5 {baseline['map5']:.5f}. The paired audit difference is {uncertainty['mean_map5_difference']:+.5f}, with a user-bootstrap 95% interval [{uncertainty['ci95_low']:+.5f}, {uncertainty['ci95_high']:+.5f}] from 2,000 resamples (seed 42). This interval quantifies sampling uncertainty conditional on this period and cohort; it does not account for season changes or search across many configurations.

The audit's numerical winner is {audit['model'][0]} at {audit['map5'][0]:.5f}. If it differs from the frozen development choice, that difference is reported rather than used to silently change the model. Recency, opportunity powers, short histories, metadata weights, association support/confidence, latent factors, regularization, score fusion and rank fusion are controlled configurations in the tables. Comparisons reveal which signals generalize for these users and periods; high complexity alone is not evidence of better accuracy. The 28-day-window experiment is identical to full history on the four-week development split, but differs on the six-week development and eight-week audit training periods. Historical-rate normalization is tested through opportunities rather than by scaling a whole user's scores, since scalar row normalization alone cannot change a standalone ranking.

In particular, 28-day decay changes audit MAP relative to full-history frequency by {am['recency_28d']-am['frequency']:+.5f}; discarding history beyond 28 days changes it by {am['last28_frequency']-am['frequency']:+.5f}. This suggests that aggressively emphasizing recent observations loses useful repeat-preference evidence in this nonstationary dataset. Adding opportunity power 0.25 to the decayed baseline changes MAP by {am['availability_p0.25']-am['recency_28d']:+.5f}, while applying it to raw counts changes MAP by {am['raw_availability_p0.25']-am['frequency']:+.5f}; available airings are therefore useful context but a universal multiplicative rate correction is not consistently beneficial. Increasing metadata weight from 2 to 4 changes content MAP by {am['content_metadata4']-am['content']:+.5f}; broad category/genre similarity cannot recover precise repeat frequency. Raising association support from 2 to 5 at confidence 0.03 changes MAP by {am['association_s5_c0.03']-am['association_s2_c0.03']:+.5f}. The 64-factor, 0.5-regularization ALS configuration differs from the 32-factor, 0.1-regularization configuration by {am['als_64_r0.5']-am['als_32_r0.1']:+.5f}; because both factor count and regularization change, this is a configuration comparison rather than an isolated causal effect of dimension. Standalone content, associations and ALS all underperform the direct-frequency baseline here. Their value is limited to incremental discovery or a small blend; the selected blend's audit interval includes zero improvement.

Audit performance by historical program diversity is a descriptive diagnostic, not another tuning criterion:

{table(breakdown,['cohort','model','users','map5','recall5'],20)}

Empty-profile users receive training-popularity recommendations from the future candidate pool. The global-popularity benchmark quantifies that fallback independently. Very small cold cohorts do not support reliable claims that one fallback is superior; the full final template includes {sparse_stats['template_no_qualifying_history']} users without qualifying history. Raw-session channel preferences could improve cold fallback in future work, but that untested model is not claimed as an implemented improvement.

## 7. Two real subscriber case studies

Selection uses full permissible historical behavior, not future labels or model success. Case A maximizes Top-5 concentration among template users with at least 200 qualifying broadcasts and eight distinct programs; Case B minimizes it among template users with at least 100 events and thirty distinct programs. This contrasts repetitive and exploratory behavior without claiming either is statistically representative. Program names below are original source labels; one program ID may have multiple episode titles and several channels. All listed future recommendations are predictions, not observed future consumption.

{case_text}

## 8. Final strategy and submission quality

The selected configuration is refit using historical qualifying events fully contained between 9 March and 27 July. Candidate programs must have a complete airing in 27 July–19 October (end exclusive), the twelve ISO weeks 31–42; source schedule spillovers outside the window are excluded. For every template user, the final file contains five distinct, positive, space-separated program IDs in decreasing model-score order. Equal scores resolve by training popularity and then ascending ID. CSV is UTF-8 with a comma delimiter; user strings, original row order and columns `user_id,tv_show_id` match the template. All {q['template_users']:,} rows pass automatic verification, including a read-back from disk. See `outputs/submission_verification.json`. The original sample is unchanged. No prediction is uploaded to Kaggle and no public/private leaderboard performance is claimed.

Concretely, the selected score is row-max-normalized frequency with a {final_spec['half_life']}-day exponential half-life and opportunity-ratio exponent {final_spec['availability_power']}, plus {final_spec['association_weight']} times row-max-normalized association score. The selected content, ALS and channel/time weights are {final_spec['content_weight']}, {final_spec['als_weight']} and {final_spec['channel_weight']}, respectively. Association rules use support {final_spec['support']}, confidence {final_spec['confidence']}, the shrinkage/partial-lift formula above and 120 neighbors per antecedent. All permissible historical events contribute with decay; no viewing data from weeks 31–42 is used.

Recommendation deployment should monitor engagement by activity cohort, program availability, fallback rate and catalog concentration, using a time-based backtest before each season. Content and collaborative models provide explainable discovery candidates even if a frequency method wins the exam metric. An exploration component should be evaluated as a separate business intervention instead of forced into a frequency-based accuracy objective.

## 9. Limitations and interpretation

1. Hidden test behavior and a scored Kaggle submission are unavailable. MAP@5's name is verified, but the private scorer's exact implementation is not supplied; target-count tie handling is our deterministic convention.
2. Naive source timestamps imply a shared clock. No external evidence establishes a correction or DST mapping; clock inconsistencies would affect the 80% qualification rule.
3. Reconstructed ground truth uses union coverage and one count per actual airing as requested. The organizer's historical preprocessing source code is not available to verify identical edge-case choices. Broadcasts crossing segment boundaries are deliberately excluded.
4. The supplied EPG labels most broadcasts with ID zero, so much logged viewing cannot become a valid target interaction. Additional channels have no retained nonzero historical EPG. This is a coverage limitation, not evidence that those sessions had no value.
5. Metadata is incomplete and program identifiers may aggregate episodes. A user exposed to a channel is not necessarily available for every future airing; opportunity ratios are not causal probabilities. Extremely frequent events may reflect a shared household/device rather than one person's taste.
6. Development labels overlap one another; only the long-horizon audit is separate. Its training history is shorter than final training, and summer/autumn season changes remain unobserved. The primary cohort is conditional on future activity and at least five labels, just as the competition cohort is curated; results do not describe all inactive subscribers.
7. The experiments are bounded: no exhaustive search, supervised reranker, external metadata or calibrated exposure model is claimed. Additional validation periods or a genuinely untouched later season would be needed before further tuning.
8. The original AI-session export is a manual platform action and remains a submission requirement. See `reports/AI_SESSION_EXPORT.md`; no conversation transcript has been fabricated or reconstructed. The original task text is preserved separately as task provenance, not a session export.

## 10. Conclusions

The project reconstructs qualifying broadcast frequency with tested interval arithmetic, compares distinct metadata, association and implicit factorization families against strong repeat-viewing baselines, and freezes selection before a twelve-week audit. Measured performance supports the selected {winner} configuration under the stated development criterion; the reported audit and two concrete user cases make both its value and its failure modes inspectable. The delivered Kaggle CSV is schema-valid, candidate-valid and reproducible; its hidden-test quality awaits an actual competition evaluation.

## Reproducibility and sources

Run `python run_pipeline.py --stage all` in this directory after installing `requirements.txt`; `--force` rebuilds preprocessing from raw CSVs. Input SHA-256 hashes are recorded in `data/processed/quality.json`. Metric arrays and case-study recommendations are saved as companion artifacts. The report generator derives every numeric statistic from these outputs.

- [SWEET.TV competition overview and evaluation](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/overview/evaluation).
- [SWEET.TV dataset and target construction](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data).
- [Implicit ALS API](https://benfred.github.io/implicit/api/models/cpu/als.html), implementation dependency used in this project.
- [OpenAI developer commands: supported conversation export](https://learn.chatgpt.com/docs/developer-commands), used only for export instructions, not as a fabricated transcript.
'''
    (ROOT/'reports/exam_report.md').write_text(report,encoding='utf-8')
    print('Report and EDA written',flush=True)

def case_studies(activity,counts,enriched,cat,future,users):
    candidates_a=activity.filter((pl.col('events')>=200)&(pl.col('unique_programs')>=8)).sort(['top5_share','events'],descending=[True,True])
    candidates_b=activity.filter((pl.col('events')>=100)&(pl.col('unique_programs')>=30)).sort(['top5_share','events'],descending=[False,True])
    if not candidates_a.height or not candidates_b.height: raise ValueError('No users meet declared case selection thresholds')
    selected=[('A',candidates_a.row(0,named=True)),('B',candidates_b.row(0,named=True))]
    predictions=np.load(ROOT/'outputs/final_predictions.npz')
    positions={int(u):j for j,u in enumerate(predictions['user_idx'])}
    cat_map={r['tv_show_id']:r for r in cat.iter_rows(named=True)}
    future_counts=dict(future.group_by('tv_show_id').len().iter_rows())
    rows=[]; summary=[]; sections=[]
    for label,user in selected:
        u=int(user['user_idx']); idx=positions[u]
        history=counts.filter(pl.col('user_idx')==u).sort(['qualifying_views','tv_show_id'],descending=[True,False]).join(cat,on='tv_show_id',maintain_order='left')
        history_counts=dict(history.select('tv_show_id','qualifying_views').iter_rows())
        history.head(15).with_columns(pl.lit(label).alias('case'),pl.lit(user['user_id']).alias('user_id')).write_csv(ROOT/f'outputs/user_{label}_historical_top15.csv')
        ev=enriched.filter(pl.col('user_idx')==u)
        channels=ev.group_by('channel_title').len().sort(['len','channel_title'],descending=[True,False])
        genres=ev.group_by('tv_show_genre_1').len().sort('len',descending=True)
        timing=ev.with_columns(((pl.col('start_time')//3600)%24).alias('hour')).group_by('hour').len().sort('len',descending=True)
        user_summary={'case':label,**user,'leading_channel':channels['channel_title'][0],'leading_genre':genres['tv_show_genre_1'][0],'peak_broadcast_hour':int(timing['hour'][0])}
        summary.append(user_summary)
        parts=[f"### User {label}: `{user['user_id']}`",f"This subscriber has {user['events']:,} qualifying broadcasts across {user['unique_programs']} programs. Their leading five programs account for {user['top5_share']:.2%} of events and the leading program for {user['top1_share']:.2%}. The most frequent broadcast-start hour is {timing['hour'][0]:02d}:00 source-local time ({timing['len'][0]} events); this describes broadcast timing, not every session's start.",
               'Observed historical Top-10:\n\n'+table(history,['tv_show_id','tv_show_title','qualifying_views'],10),
               'Observed channels:\n\n'+table(channels,limit=5),'Observed first-genre distribution:\n\n'+table(genres,limit=5)]
        for algorithm in ['global_popularity','frequency','content','association','als_32_r0.1','selected']:
            rec=[]
            for rank,item in enumerate(predictions[algorithm][idx],1):
                item=int(item); r=cat_map[item]
                record={'case':label,'user_id':user['user_id'],'algorithm':algorithm,'rank':rank,'tv_show_id':item,'tv_show_title':r['tv_show_title'],'category':r['tv_show_category'],'genre':r['tv_show_genre_1'],'historical_qualifying_views':history_counts.get(item,0),'future_broadcasts':future_counts.get(item,0),'familiar':item in history_counts}
                rows.append(record); rec.append(record)
            frame=pl.DataFrame(rec)
            parts += [f"**{algorithm.title()} future Top-5**\n\n"+table(frame,['rank','tv_show_id','tv_show_title','category','historical_qualifying_views','future_broadcasts','familiar']),
                      f"{sum(r['familiar'] for r in rec)} of these five have qualifying historical views; {sum(not r['familiar'] for r in rec)} are possible discoveries. A zero historical count means no past **qualifying** event, not proof the subscriber never saw the program."]
        if label=='A':
            parts.append("The dominant observed program is SpongeBob SquarePants (`12001682`, 328 qualifying airings). The content model places it first and suggests other children's comedy titles, which is consistent with the metadata profile. The association list instead puts TSN (`2400480`) and Fakty (`240081`) above SpongeBob despite just one historical qualifying event for each news program. This exposes audience-level contamination: shared household news viewing can outweigh a narrow children's-program preference. The hybrid preserves SpongeBob, ALVINNN!!! and the Chipmunks (`12002856`) and The Fairly OddParents (`12001734`) in its first three positions, but still promotes the two news programs into positions four and five. This is a concrete potential failure, not evidence that the user's future tastes actually changed. A production explanation should label the former as repeat favorites and the latter as weakly supported audience suggestions.")
        else:
            parts.append("This subscriber's 239 programs across 310 events show mostly low-frequency consumption. Fakty (`240081`) is nevertheless their most repeated single ID with nine qualifying airings, while ViP Comedy CEE is the leading channel and action is the most common first-genre annotation. Content recommendations favor action films with similar metadata; the final model mixes recurring news/breakfast programs, Criminal Minds (`10001261`) and The Lost World (`90070747`). These are familiar items with different future airing opportunities. Recommending a rarely repeated film is especially uncertain because a completed movie may not be watched again, and a film-heavy channel can offer many new titles. The case illustrates a mismatch between predicting per-ID future frequency and serving a user's broader taste for varied films.")
        parts.append("Content scores come from shared descriptions; association scores come from shared audiences and exclude self-rules. The selected model restores the repeat-frequency component based on measured development performance. High future airing counts establish availability, not personal intent. Sparse metadata, seasonal changes, shared household devices and weak per-item counts can all produce errors. These are possible mechanisms; the final future labels are unknown.")
        sections.append('\n\n'.join(parts))
    pl.DataFrame(rows).write_csv(ROOT/'outputs/selected_users_analysis.csv')
    return '\n\n'.join(sections),summary

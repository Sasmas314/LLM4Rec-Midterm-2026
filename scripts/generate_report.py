"""Build an English PDF from measured artifacts. Never synthesize scores."""
from pathlib import Path
import json, sys, html
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from src.data_loading import ROOT

O=ROOT/'outputs'; R=ROOT/'reports'; RES=ROOT/'results'
def load(name): return json.loads((O/name).read_text(encoding='utf-8'))

def main():
    fonts=Path('C:/Windows/Fonts')
    if (fonts/'arial.ttf').exists():
        regular,bold=fonts/'arial.ttf',fonts/'arialbd.ttf'
    else:
        import matplotlib.font_manager as fm
        regular=Path(fm.findfont('DejaVu Sans')); bold=Path(fm.findfont(fm.FontProperties(family='DejaVu Sans',weight='bold')))
    pdfmetrics.registerFont(TTFont('Body',str(regular))); pdfmetrics.registerFont(TTFont('Bold',str(bold)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')
    styles=getSampleStyleSheet()
    for name in ['Normal','BodyText','Title','Heading1','Heading2','Heading3']:
        styles[name].fontName='Body' if name in ['Normal','BodyText'] else 'Bold'
    styles['BodyText'].fontSize=10; styles['BodyText'].leading=14.3; styles['BodyText'].spaceAfter=8
    styles['Heading1'].fontSize=20; styles['Heading1'].leading=25; styles['Heading1'].spaceAfter=16
    styles['Heading2'].fontSize=12; styles['Heading2'].leading=16
    styles['Title'].fontSize=30; styles['Title'].leading=36
    styles.add(ParagraphStyle(name='Cell',fontName='Body',fontSize=8,leading=10.5))
    styles.add(ParagraphStyle(name='SmallNote',fontName='Body',fontSize=8.5,leading=11.5,textColor=colors.HexColor('#425466'),spaceAfter=8))
    story=[]; table_no=0; figure_no=0
    def p(s,style='BodyText'): return Paragraph(s,styles[style])
    def add(s,style='BodyText'): story.append(p(s,style))
    def section(title):
        if story: story.append(PageBreak())
        add(title,'Heading1')
    def tab(headers,rows,widths=None,caption=None):
        nonlocal table_no
        table_no+=1
        if caption: add(f'Table {table_no}. {caption}','SmallNote')
        cells=[[p(html.escape(str(v)),'Cell') for v in row] for row in [headers,*rows]]
        t=Table(cells,colWidths=widths or [491/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#244d64')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#ccd5dc'))]))
        story.extend([t,Spacer(1,10)])
    def fig(name,caption,width=491):
        nonlocal figure_no
        from PIL import Image as PILImage
        path=R/'figures'/name
        with PILImage.open(path) as im: ratio=im.height/im.width
        figure_no+=1
        story.extend([Image(str(path),width=width,height=width*ratio),p(f'Figure {figure_no}. {caption}','SmallNote')])
    results=pd.read_csv(O/'validation_results.csv'); cohorts=pd.read_csv(O/'validation_cohorts.csv')
    selection=load('selection.json'); win=selection['selected_model']; config=load('final_model.json')
    audit=results[results.role=='audit'].set_index('model'); dev=results[results.role=='development'].groupby('model').map5.mean()
    eda=load('eda_statistics.json'); quality=json.loads((ROOT/'data/processed/quality.json').read_text(encoding='utf-8'))
    iq=json.loads((ROOT/'data/processed/interaction_quality.json').read_text(encoding='utf-8')); uncertainty=load('audit_uncertainty.json')
    evidence_path=RES/'external_evidence.json'
    evidence=json.loads(evidence_path.read_text(encoding='utf-8')) if evidence_path.exists() else {}
    repo=evidence.get('github_url'); kg=evidence.get('kaggle',{})
    models=['global_popularity','frequency','content','association_s2_c0.03','als_32_r0.1',win]
    names={'global_popularity':'Popularity','frequency':'Personal frequency','content':'Content-based','association_s2_c0.03':'Association rules','als_32_r0.1':'Implicit ALS',win:'Selected hybrid'}
    section('SWEET.TV\nProgram Recommendation')
    add('LLM4Rec Midterm Quiz','Title'); story.append(Spacer(1,26))
    add('Alexander Ivanov<br/>Александр Иванов<br/>Kaggle: Sasmas314','Heading2')
    add('Higher School of Economics<br/>Instructor: Seungmin Jin<br/>24 October 2026')
    story.append(Spacer(1,28))
    add('Temporal validation, reproducible experiments and an evidence-based recommendation strategy','Heading2')
    add(f'Final strategy: {html.escape(win)}. Development MAP@5: {dev[win]:.5f}. Frozen 12-week audit MAP@5: {audit.loc[win,"map5"]:.5f}.')
    add(f'Repository: <link href="{repo}">{repo}</link>' if repo else 'MISSING EVIDENCE: GitHub repository has not been published. The complete local repository accompanies this report.')
    add('Competition score: '+str(kg.get('private_score'))+' (private).' if kg.get('private_score') is not None else 'MISSING EVIDENCE: no verified private Kaggle score is available. Local metrics are not leaderboard scores.')
    section('1. Executive summary')
    add(f'The project predicts five program IDs for each of {quality["template_users"]:,} users. It reconstructs preferences from {quality["log_rows_raw"]:,} viewing records and the supplied broadcast schedule. A viewing event qualifies only when the union of watched intervals covers at least 80% of a broadcast. The resulting {iq["qualifying_events"]:,} events connect {eda["qualifying_users"]:,} active users to {eda["qualifying_programs"]:,} programs.')
    add('Three qualifying recommendation families are implemented and evaluated individually: content-based filtering, association rules and implicit matrix factorization. Global popularity supplies an additional reference. Personal repeat frequency, recency, availability and hybrid variants test whether more complex signals improve the competition objective.')
    add(f'The selection criterion is mean MAP@5 across two development periods. It chooses {html.escape(win)} at {dev[win]:.5f}. Its 12-week audit score is {audit.loc[win,"map5"]:.5f}, versus {audit.loc["frequency","map5"]:.5f} for personal frequency. The difference is {uncertainty["mean_map5_difference"]:+.5f}, with a paired 95% bootstrap interval [{uncertainty["ci95_low"]:+.5f}, {uncertainty["ci95_high"]:+.5f}]. This does not demonstrate superiority on the longer horizon. The frozen selection remains the designated final model, and the report preserves the negative audit evidence.')
    tab(['Method','Development MAP@5','Audit MAP@5'],[[names[m],f'{dev[m]:.5f}',f'{audit.loc[m,"map5"]:.5f}'] for m in models],[187,152,152],'Representative methods, all evaluated on identical cohorts and scheduled candidates')
    add('SWEET.TV should start with a repeat-preference strategy, use a modest association signal for complementary suggestions, and keep personal frequency as a deployment challenger. A live A/B test would be required to establish engagement, retention or revenue impact.')
    section('2. Task, sources and data')
    add('The supplied presentation is the primary assignment specification. It requires at least two qualifying methods, a popularity baseline, temporal evaluation, analysis of the same two users, a GitHub link, a screenshot of the returned private score and an original unmodified AI-session export. It asks the student to submit the PDF and session export. Code and the exact final CSV belong in the repository.')
    add('The competition defines future preferences over weeks 31-42 and splits the public/private test by users, 50/50. The Data and Evaluation pages were inspected in the authenticated browser. Both the course and competition request five ranked program IDs. No hidden labels or server-side scoring source are supplied.')
    inspection=load('data_inspection.json')
    tab(['Source file','Rows','Role'],[[k,f'{v["rows"]:,}','Viewing log' if k.startswith('dataset') else 'Test template' if k=='submission.csv' else 'Broadcast schedule'] for k,v in inspection.items()],[220,83,188],'Actual CSV files and measured row counts')
    add(f'Viewing logs contain {quality["users"]:,} users and {quality["channels_watched"]} channels. The complete schedule has {quality["programs"]:,} valid nonzero program IDs. Historical and target windows contain {eda["historical_broadcasts"]:,} and {eda["future_broadcasts"]:,} fully contained annotated broadcasts respectively.')
    add('User IDs remain decimal strings when reading, joining and writing CSV files. Several exceed the signed 64-bit range and cannot safely pass through floating-point values. The loader maps vsetv_id to channel_id, stop_time to end_time and the source spelling duraton to duration. UTF-8 source titles and categories remain unchanged.')
    section('3. Preprocessing and interval matching')
    add(f'The log has {quality["log_duplicates"]} exact duplicate rows, {quality["duration_timestamp_mismatches"]} disagreements between declared duration and endpoint subtraction, and {quality["invalid_sessions"]} invalid intervals. Endpoints govern watched time. The schedule audit records nulls, invalid durations and zero IDs per file; invalid IDs and intervals never become recommendation candidates.')
    add('All intervals use half-open boundaries [start,end). Both feeds have offset-free timestamps and share a source-local wall-clock assumption. The code does not invent a timezone or daylight-saving correction. Tests also cover equivalent timestamp-aware inputs.')
    add('For every user-channel pair, sort viewing intervals and union overlapping or adjacent intervals. This removes duplicated coverage across repeated or overlapping log rows. For each channel, sort broadcast starts and build a prefix maximum of broadcast end times. Two binary searches locate all broadcasts that can intersect each merged viewing interval. Every positive overlap contributes max(0, min(view_end, program_end) - max(view_start, program_start)) seconds.')
    add('Aggregate the overlap seconds per user-broadcast pair. Divide by full broadcast duration and retain ratios at least 0.8. Each qualifying broadcast contributes one frequency event. Multiple programs in a long session are handled separately. Partial watches accumulate across sessions. Gaps contribute no interaction. Conflicting broadcasts remain separately identified and are audited. Broadcasts crossing a train/validation cutoff are excluded from both segments, preventing coverage observed after the cutoff from entering training.')
    coverage=json.loads((RES/'mapping_coverage.json').read_text(encoding='utf-8'))
    add(f'Mapping coverage before the 80% qualification rule is {coverage["mapping_coverage"]:.2%}: {coverage["intervals_overlapping_valid_annotated_broadcasts"]:,} of {coverage["merged_intervals"]:,} merged viewing intervals overlap at least one valid nonzero annotated broadcast. The denominator includes channels without annotated schedule coverage. This is interval coverage, not the percentage of sessions that qualify as a complete program view.')
    tab(['Matching audit','Measured value'],[[k,f'{v:,}' if isinstance(v,int) else str(v)] for k,v in iq.items()],[340,151],'Interval reconstruction diagnostics')
    add('The matching cost is O(S log B + K) after sorting, for S merged intervals, B broadcasts per channel and K candidate overlaps examined. In pathological overlapping schedules K can be large. The implementation avoids a full session-broadcast Cartesian product.')
    add('The canonical event cache stores user_idx, broadcast_id, tv_show_id, watched_seconds, screen_time_ratio, start_time and end_time. A portable interaction table additionally exposes user_id, tv_show_id, interaction_weight and timestamp. Count weights match the competition target. Recency variants use 2^(-age_days / half_life), testing 7, 14, 28 and 56 days.')
    section('4. Exploratory analysis')
    add(f'There are {eda["user_program_pairs"]:,} user-program pairs. Sparsity among active users and watched programs is {eda["sparsity_qualifying_users_programs"]:.2%}. The median active user has {eda["events_median_active"]:.0f} qualifying events and {eda["unique_programs_median_active"]:.0f} distinct programs. {eda["no_qualifying_history_users"]} logged users have no qualifying event, including {eda["template_no_qualifying_history"]} test users.')
    fig('data_overview.png','User activity, weekly qualifying events and source-local viewing times')
    add(f'The ten most frequent programs account for {eda["top10_program_event_share"]:.2%} of qualifying events. {eda["programs_five_or_fewer_viewers"]} programs have at most five viewers and contribute {eda["events_on_five_or_fewer_viewer_programs"]:,} events. Heavy concentration favors popularity while sparse tail evidence makes unregularized association rules unstable.')
    if (R/'figures/long_tail_categories.png').exists(): fig('long_tail_categories.png','Program frequency and category distribution')
    section('5. Recommendation algorithms')
    add('Popularity and repeat frequency','Heading2')
    add('Global popularity ranks candidate programs by training qualifying-event count. Personal frequency ranks each user\'s own counts and breaks ties by training popularity, then ascending program ID. Previously watched programs remain eligible because the target measures repeated future preferences. Cold users receive the same training-popularity fallback restricted to eligible scheduled programs.')
    add('Content-based filtering','Heading2')
    add('TF-IDF uses title word unigrams/bigrams, category and genre tokens, plus actors, directors and production years when present. Metadata features receive weight 2 or 4, people/year features weight 0.5. Each item vector and the user\'s weighted aggregate profile receive L2 normalization. Cosine scores rank future candidates. Feature fitting uses the historical and target schedule, which is permitted catalog information. It never uses target user viewing behavior.')
    add('Association rules','Heading2')
    add('Binary user-program incidence yields co-occurrence C_ij. Rule confidence is C_ij/support(i). The final edge is confidence × C_ij/(C_ij+5) / sqrt(support(j)/U), with support 2 or 5 and confidence 0.03 or 0.10. Diagonal edges are removed and only 120 outgoing neighbors are retained. User log(1+decayed qualifying frequency) weights the outgoing association scores. This supports discovery but can pull narrow profiles toward globally common audience habits.')
    add('Implicit matrix factorization','Heading2')
    add('Alternating least squares minimizes a confidence-weighted implicit objective, with preference 1 for observed pairs and 0 for unobserved pairs. The implementation passes 10 log(1+decayed counts) for observed confidence. It tests 32 factors/regularization 0.1 and 64 factors/regularization 0.5, with 25 iterations, seed 42 and two worker threads. Counts are implicit positive signals, not explicit ratings. Unseen items lack learned preference evidence.')
    add('Hybrid and availability','Heading2')
    add('The frequency component may multiply scores by ((future_airings+2)/(historical_airings+2))^p, where p is 0.25, 0.5 or 1. Each nonnegative component is divided by its row maximum before blending. Tests also include rank fusion and a channel/time signal. The selected model combines decayed, availability-adjusted repeat frequency with a small association term. Complete settings are recorded in experiment_specs.json and configs/best_model.yaml.')
    section('6. Temporal validation and metrics')
    tab(['Split','Train end','Validation end','Days','Users with ≥5 labels'],[[r.split,r.cutoff,r.end_exclusive,int(r.horizon_days),int(r.eligible_five_label_users)] for r in cohorts.itertuples()],[90,98,98,50,155],'All training begins 9 March 2020, all end boundaries are exclusive')
    add('Two development periods share the end date 4 May and therefore overlap. Their unweighted mean is a tuning criterion, not two independent replications. The chosen configuration is frozen before the 4 May-27 July twelve-week audit. The audit matches the target horizon, though final training uses a longer twenty-week history and a different season. The previous local solution already contained these periods and results, so this rerun is a reproducibility audit, not a newly blinded study.')
    add('Every qualifying model uses the same candidate pool: nonzero IDs in broadcasts fully contained within the target period. No target ground-truth item filtering narrows the pool. Metadata from the target schedule is assumed known at recommendation time. An unfiltered-frequency diagnostic intentionally ignores availability and is excluded from the main equal-candidate comparison.')
    tab(['Split','Active with 1-4 labels','No qualifying labels','Candidate label coverage'],[[r.split,int(r.active_fewer_than_five_users),int(r.no_qualifying_validation_users),f'{results[results.split==r.split].candidate_label_coverage.iloc[0]:.3%}'] for r in cohorts.itertuples()],[90,130,130,141],'Excluded and secondary cohorts remain explicit')
    add('The primary cohort contains template users with at least five distinct qualifying target programs. Ground truth is their five most frequent IDs, with numeric-ID tie breaking. Active users with fewer labels are also evaluated using their actual labels. Zero-label users have undefined AP and are counted separately. Such cohort conditioning can make the primary score optimistic for all-user deployment.')
    add('AP@5 = Σ[r=1..5] Precision@r × relevant_hit_at_r / min(5, number_of_relevant_items). Duplicate predictions cannot create extra hits. MAP@5 averages AP over the declared cohort. Precision divides hits by five. Recall divides by the true relevant-set size. HitRate is 1 if any recommendation hits. With exactly five labels, precision and recall coincide. Binary MAP distinguishes hit positions, but not permutations of five correct items. Ordered agreement and graded NDCG are separate diagnostics because the competition text also emphasizes order. The hidden scoring implementation remains unavailable.')
    section('7. Measured results and model selection')
    tab(['Method','MAP@5','P@5','R@5','HitRate@5'],[[names[m],*[f'{audit.loc[m,c]:.5f}' for c in ['map5','precision5','recall5','hitrate5']]] for m in models],[167,81,81,81,81],'Frozen twelve-week audit results')
    fig('validation_comparison.png','Audit ranking, including diagnostic variants. Highlight identifies the frozen development winner')
    add(f'The selected model scores {dev[win]:.5f} on development. Personal frequency scores {dev["frequency"]:.5f}, a difference of {dev[win]-dev["frequency"]:+.5f}. On the audit, the selected strategy scores {audit.loc[win,"map5"]:.5f}; personal frequency scores {audit.loc["frequency","map5"]:.5f}. The audit\'s numerical winner is {html.escape(audit.map5.idxmax())} at {audit.map5.max():.5f}. The selected configuration is not replaced using this audit outcome.')
    add(f'The paired audit delta against frequency is {uncertainty["mean_map5_difference"]:+.5f}; the 95% interval [{uncertainty["ci95_low"]:+.5f}, {uncertainty["ci95_high"]:+.5f}] uses 2,000 user bootstrap resamples, seed 42. It reflects this cohort and period, not season changes, repeated model searches or uncertainty in hidden labels.')
    section('8. Optimization, cost and scalability')
    add(f'{results.model.nunique()} configurations are evaluated on all three periods. The search covers recency, restricted history, availability exponents, content weighting, association support/confidence, ALS factors/regularization and hybrid weights. Each standalone family is evaluated before hybrid selection. Full parameters and metrics are in results/experiments.csv. The strongest standalone qualifying family is identified by development mean rather than audit rank.')
    qualifying=dev[[m for m in dev.index if m.startswith(('association','content','als'))]].sort_values(ascending=False)
    add(f'The strongest qualifying configuration on development is {html.escape(qualifying.index[0])} ({qualifying.iloc[0]:.5f}). Association variants test meaningful support/confidence changes. No unbounded parameter search or Kaggle-driven search is performed.')
    tab(['Method','Fit + scoring (s)','Cached rank (s)','Process RSS (MiB)'],[[names[m],f'{audit.loc[m,"training_and_scoring_seconds"]:.3f}',f'{audit.loc[m,"inference_seconds"]:.3f}',f'{audit.loc[m,"process_rss_mb"]:.0f}'] for m in models],[185,102,102,102],'Measured audit-stage wall time and resident process memory')
    add('Times use a shared engine: fit + scoring includes a component build only when it is absent from the cache, and cached rank includes score copying, ranking and minor score operations. They are incremental workflow costs, not isolated cold-start benchmarks. RSS is process memory sampled after each method, not model-only or peak memory. ALS uses two workers, BLAS one thread and Polars four. Hardware: 20 logical CPUs, approximately 31.1 GiB RAM, AMD Radeon graphics; no GPU training is used.')
    add('Frequency training is linear in event count. Sparse association construction costs approximately the sum of squared user profile sizes before neighbor truncation; broad users can densify co-occurrence. The dense similarity intermediate therefore needs partitioning at larger catalog sizes. Content scoring can use sparse matrix multiplication or an approximate nearest-neighbor index. ALS stores O((U+I)f) factors and supports batched dot-product retrieval. The current implementation caches user-by-candidate score matrices and sorts the full candidate list. Production inference should batch users, retain sparse edges, select Top-K without full sorts and monitor memory.')
    section('9. Two-user study: historical preferences')
    cases=load('selected_users_summary.json'); recs=pd.read_csv(O/'selected_users_analysis.csv',dtype={'user_id':str})
    add('The same two users appear in every comparison. Selection uses permissible full historical behavior, without target outcomes. Case A maximizes Top-5 concentration among test users with at least 200 qualifying broadcasts and eight programs. Case B minimizes it among users with at least 100 events and thirty programs. These deliberately contrasting cases are explanatory, not a representative sample.')
    for u in cases:
        label=u['case']; add(f'User {label}: {u["user_id"]}','Heading2')
        add(f'{u["events"]:,} events, {u["unique_programs"]} distinct programs, Top-5 concentration {u["top5_share"]:.1%}, top-program share {u["top1_share"]:.1%}. Leading channel: {html.escape(u["leading_channel"])}. Leading first-genre: {html.escape(str(u["leading_genre"]))}.')
        history=pd.read_csv(O/f'user_{label}_historical_top15.csv')
        tab(['Program ID','Original program title','Category','Views'],[[int(r.tv_show_id),r.tv_show_title,r.tv_show_category,int(r.qualifying_views)] for r in history.head(5).itertuples()],[72,250,119,50],f'User {label}, observed historical Top-5')
        categories=pd.read_csv(RES/'two_users_categories.csv'); categories=categories[categories['case']==label].head(3)
        add('Category mix by qualifying events: '+', '.join(f'{html.escape(str(r.category))} {r.event_share:.1%}' for r in categories.itertuples())+'.','SmallNote')
    for u in cases:
        label=u['case']; section(f'10{label}. User {label}: model Top-5')
        add('Each row is a genuine future prediction. Titles and categories come from the supplied schedule. The interpretation indicates the model mechanism, not proof of user satisfaction.')
        for algorithm,labelname in [('global_popularity','Popularity'),('frequency','Personal frequency'),('content','Content-based'),('association','Association rules'),('als_32_r0.1','Implicit ALS'),('selected','Selected hybrid')]:
            block_start=len(story)
            rows=recs[(recs['case']==label)&(recs.algorithm==algorithm)].sort_values('rank')
            add(labelname,'Heading2')
            tab(['#','Program ID','Title','Category','Past views'],[[r.rank,int(r.tv_show_id),r.tv_show_title,r.category,r.historical_qualifying_views] for r in rows.itertuples()],[22,70,235,104,60])
            seen=int(rows.familiar.sum())
            reason={'global_popularity':'Training frequency across all users drives this list.','frequency':'The user\'s own qualifying-event counts drive this list.','content':'Cosine similarity to the user\'s metadata profile drives this list.','association':'Programs shared by overlapping audiences receive rule support.','als_32_r0.1':'Latent factors represent patterns of implicit viewing.','selected':'Repeat frequency, scheduled opportunities and association evidence combine.'}[algorithm]
            add(f'{reason} {seen}/5 recommendations have qualifying historical views.','SmallNote')
            story[block_start:]=[KeepTogether(story[block_start:])]
    section('11. User experience and observed failures')
    add('Case A is strongly concentrated in children\'s programming. SpongeBob SquarePants (12001682) is the most repeated historical ID. The association model can still promote TSN (2400480) and Fakty (240081), showing how shared audiences can inject news into a narrow children\'s profile. Household sharing is a plausible mechanism, not a measured demographic fact. The hybrid retains the repeat signal but can preserve some of that contamination.')
    add('Case B has a much broader catalog and low repeat concentration. Frequently repeated news IDs can dominate a frequency objective even when the broader channel/genre history suggests varied films. Content similarity supplies plausible discovery candidates, while their low per-ID repeat evidence makes them uncertain predictions of the five most frequent future programs. Completed films may not be watched again. These are hypotheses about recommendation utility; no future test-user outcomes are visible.')
    breakdown=pd.read_csv(O/'audit_cohort_breakdown.csv')
    view=breakdown[breakdown.model.isin([win,'frequency'])]
    tab(['Audit history cohort','Users','Method','MAP@5'],[[r.cohort,r.users,'Hybrid' if r.model==win else 'Frequency',f'{r.map5:.5f}'] for r in view.itertuples()],[160,50,160,121],'Observed cold and sparse-profile failure analysis')
    add(f'There are {eda["future_unwatched_programs"]:,} future scheduled IDs without qualifying historical interactions, including {eda["future_new_to_historical_schedule"]:,} IDs absent from the historical schedule. Pure collaborative factors cannot learn these programs. Content can describe them, but the offline metric supplies weak incentives for discovery. Candidate coverage, recommendation novelty and catalog concentration should accompany accuracy in deployment.')
    section('12. Business decision and limitations')
    add('For this submission, use the frozen development-selected hybrid. Its frequency backbone fits the explicitly defined repeat-viewing target. The modest association term allows complementary audience evidence, while availability removes impossible programs. The long-horizon audit does not establish an advantage over personal frequency, so a production decision should retain the simpler baseline as a serious challenger and evaluate the added component prospectively.')
    add('Global popularity is inexpensive and useful for empty profiles, but lacks individual preferences. Content makes metadata explanations and cold-item suggestions possible, yet broad category similarity does not reliably predict repeated per-ID preferences. Association rules are interpretable but inherit audience concentration and co-occurrence sparsity. ALS compresses the interaction matrix, but the tested configurations underperform frequency and require additional training and operational complexity.')
    add('A practical deployment would update histories incrementally, check schedule availability before serving, batch score retrieval, cache repeated requests and log fallback rates. Monitor accuracy by activity cohort, exposure concentration and long-tail coverage. Reserve a separate experimentally controlled discovery surface if the business objective extends beyond forecasting habitual viewing. No revenue, retention or satisfaction improvement is demonstrated by these offline results.')
    for s in [
        'No hidden test labels or server-side metric implementation are available. The assignment AP convention is implemented and hand-tested; target tie-breaking and organizer matching details remain assumptions.',
        'Overlapping development windows are correlated. The previous project exposed the audit results before this rerun, so it cannot serve as a newly blinded holdout. No retuning uses the rerun audit.',
        'Source clocks have no timezone offsets. Four annotated overlaps and extensive missing schedule IDs can distort coverage. Counting a broadcast with at least 80% coverage loses partial-interest information.',
        'Temporal cohorts exclude users with no qualifying labels and primarily require five targets. Small cold cohorts cannot support broad claims. Schedule availability is assumed known, while actual user availability and exposure are unobserved.',
        'Program IDs can aggregate episodes. Metadata is incomplete, and the modal source title is descriptive rather than a guaranteed future episode title. The model learns accounts/devices, which may represent households.',
        'Search is limited, especially for ALS. Timings use shared caches, memory is sampled rather than profiled at peak, and the implementation has not been load-tested at production scale.'
    ]: add(s)
    section('13. Reproducibility and deliverables')
    add(f'<link href="{repo}">{repo}</link>' if repo else 'MISSING EVIDENCE: publish the prepared local repository, then rerun this script with its verified URL in results/external_evidence.json.','Heading2')
    add('Use Python 3.12. Install requirements.txt, place the five original CSVs together, then run: python -m pytest tests -q; python run_pipeline.py --stage all --data-dir PATH --force; python scripts/finalize_artifacts.py --data-dir PATH; python scripts/generate_report.py. Commands are separate invocations in the README. Input SHA-256 hashes bind the processed caches to their sources. Raw data is excluded from Git.')
    add('Code reuses and extends a prior local solution found in Documents/ChatGPT/HSE/sweet_tv_exam. This run independently rebuilds raw preprocessing, interval matches and all experiment predictions. It adds HitRate, cost measurements, additional case-study methods, a PDF, export preservation and external submission evidence. It does not claim authorship of an entirely new pipeline or a newly blinded study.')
    add('The root submission.csv is an exact copy of outputs/submission.csv. The automated validator checks schema, row order, full user coverage, five positive unique IDs and membership in the final schedule. results/experiments.csv records every configuration. final_model_metrics.json distinguishes local validation from returned Kaggle scores.')
    add('The session-export directory preserves a byte-for-byte copy of the actual Codex JSONL record, with SHA-256 manifest. Because this chat is still running when it copies the file, the snapshot may omit the last tool calls and final response. Refresh after the chat finishes using scripts/refresh_session_export.ps1. The file is not translated, sanitized or reconstructed. The instructor has not confirmed whether native Codex JSONL satisfies any OpenCode-specific export requirement. The unrelated WARNING_BANNER session.json is not presented as evidence.')
    section('14. Kaggle submission evidence')
    if kg.get('private_score') is not None:
        add(f'Account: Sasmas314. Student: Александр Иванов. Private score: <b>{kg["private_score"]}</b>. Public score: {kg.get("public_score","not recorded")}. Submission ID: {kg.get("submission_id","see screenshot")}. File: submission.csv. This is the designated final submission, not a claim about leaderboard rank.')
        add(html.escape(kg.get('description','')))
        screenshot=R/'figures/kaggle_private_score.jpg'
        if screenshot.exists(): fig('kaggle_private_score.jpg','Genuine authenticated Kaggle submissions page with returned private score')
        else: add('MISSING EVIDENCE: returned score has no saved screenshot. Do not treat this section as complete.')
    else:
        add('MISSING EVIDENCE: no verified returned private score or score screenshot. The prepared CSV is valid locally; this does not prove Kaggle accepted or scored it. '+html.escape(kg.get('status','Submission not verified.')))
    add('Kaggle\'s page allows up to two final selections and may auto-select public-score candidates if fewer are selected. The course grades the returned private score of the designated submission rather than frozen leaderboard position. A genuine screenshot and matching CSV checksum are needed to connect the claimed score to the exact artifact.')
    section('15. References and experiment appendix')
    for ref in [
        'Jin, S. LLM4Rec. Midterm Quiz (1).pptx. Supplied course assignment, slides 2-9.',
        'ogurbych (2020). sweet.tv - TV program recommender. Kaggle. https://www.kaggle.com/competitions/sweettv-tv-program-recommender',
        'Kaggle competition Data and Evaluation pages, inspected 24 October 2026. https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data',
        'Implicit ALS implementation: implicit.cpu.als.AlternatingLeastSquares. Dependency version recorded in requirements.txt. Source: https://github.com/benfred/implicit',
        'Codex export documentation: https://learn.chatgpt.com/docs/developer-commands. The /export command creates a Markdown export after a turn completes. Native JSONL is preserved separately in this submission.'
    ]: add(ref,'SmallNote')
    section('16. All measured MAP@5 results')
    pivot=results.pivot(index='model',columns='split',values='map5').sort_index()
    tab(['Configuration','Dev 4w','Dev 2w','Audit 12w'],[[m,*[f'{pivot.loc[m,c]:.5f}' for c in ['dev_4w','dev_2w','audit_12w']]] for m in pivot.index],[251,80,80,80],'Complete experiment set. Unfiltered frequency is a separate candidate-constraint diagnostic')
    def footer(canvas,doc):
        canvas.setFont('Body',8); canvas.setFillColor(colors.HexColor('#546575'))
        canvas.drawString(52,28,'LLM4Rec Midterm | Alexander Ivanov | SWEET.TV')
        canvas.drawRightString(543,28,str(doc.page))
    path=R/'LLM4Rec_Midterm_Report.pdf'
    doc=SimpleDocTemplate(str(path),pagesize=(595.28,841.89),rightMargin=52,leftMargin=52,topMargin=49,bottomMargin=47,title='SWEET.TV Recommendation System - LLM4Rec Midterm',author='Alexander Ivanov')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(path)

if __name__=='__main__': main()

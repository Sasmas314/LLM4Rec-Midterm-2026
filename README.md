# SWEET.TV Recommendation System: LLM4Rec Midterm

Alexander Ivanov (Александр Иванов), Kaggle **Sasmas314**, HSE, 24 October 2026.

A reproducible Top-5 TV program recommender for 1,260 users. Implements **content-based filtering, association rules and implicit ALS**, plus global popularity, personal frequency, recency/availability variants and hybrids.

## Reproduce (Python 3.12)

Run separate commands from this repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe run_pipeline.py --stage all --data-dir 'C:\path\to\original-csvs' --force
.\.venv\Scripts\python.exe scripts\finalize_artifacts.py --data-dir 'C:\path\to\original-csvs'
.\.venv\Scripts\python.exe scripts\generate_report.py
```

On Linux/macOS create `.venv` with `python3.12 -m venv .venv` and use `.venv/bin/python`. PDF creation uses Arial on Windows and DejaVu Sans elsewhere. No GPU is required. Seeds are 42; ALS can have minor platform-dependent floating point differences.

## Inputs

Download the five original files from the [competition](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data), extract them into one directory, and pass it with `--data-dir`:

- `dataset11-30.csv`
- `export_arh_11-20-final.csv`
- `export_arh_21-30-final.csv`
- `export_arh_31-42-final.csv`
- `submission.csv` (unfilled official template)

Raw data and caches stay out of Git. SHA-256 provenance, schemas, missingness, duplicates and date ranges are recorded. Changed source checksums require `--force`. User IDs remain decimal strings to preserve values beyond signed 64-bit and floating-point exactness.

## Matching and interactions

Map `vsetv_id` to `channel_id`, `stop_time` to end time and `duraton` to declared duration. Endpoint subtraction governs watched time. Remove invalid intervals and zero/null/invalid show IDs. For every user/channel, union overlapping/adjacent viewing intervals, then enumerate all intersecting broadcasts using per-channel sorted starts and prefix-maximum ends. Sum `max(0,min(view_end,show_end)-max(view_start,show_start))` seconds per user/broadcast. Keep coverage at least 0.8 and count each broadcast once. This supports repeated fragments, sessions spanning shows, gaps and overlapping schedules. Cutoff-crossing broadcasts are excluded.

`data/processed/interaction_table.parquet` exposes `user_id,tv_show_id,interaction_weight,timestamp`, plus seconds and coverage. Base weight is one qualifying airing, matching the frequency target. Recency weights test 7/14/28/56-day half-lives. Both feeds use the common offset-free source-local clock. No guessed DST correction is applied.

## Validation

Training begins 9 March 2020. Development 4w: train before 6 April, validate until 4 May. Development 2w: train before 20 April, validate until 4 May. Audit 12w: train before 4 May, validate until 27 July. All end boundaries are exclusive. Freeze the model chosen by mean development MAP before auditing. Final training uses all twenty historical weeks and predicts weeks 31-42.

The development windows overlap and are correlated. The audit was already present in the prior local solution. This rerun verifies reproducibility, not a newly blinded holdout. Audit results do not retune or replace the chosen configuration.

Main methods use identical users, scheduled candidate IDs, labels and protocol. Known future schedule/metadata is permitted catalog information; future user behavior never trains a model. Previously watched programs remain eligible. Primary labels are the five most frequent qualifying IDs among users with at least five actual targets. Active users with fewer labels have separate metrics; zero-label users are counted explicitly. No synthetic labels are added. `historical_frequency_unfiltered` is an explicitly separate candidate-constraint diagnostic, excluded from the main equal-pool comparison.

AP@5 sums precision at every unique relevant prediction and divides by `min(5,number of ground-truth labels)`. MAP averages the defined cohort. Precision, recall, HitRate, binary/graded NDCG and ordered agreement are separate metrics. The hidden private scorer is unavailable.

## Measured results

31 configurations on three periods, 93 independently re-evaluated MAP rows. See `results/experiments.csv` for complete actual parameters, metrics, incremental fit/scoring time, cached ranking time and process RSS. Shared caches make these timings workflow costs, not isolated cold-start benchmarks.

| Method | Development mean MAP@5 | Twelve-week audit MAP@5 |
|---|---:|---:|
| global_popularity | 0.29339 | 0.32315 |
| frequency | 0.40773 | 0.39734 |
| content | 0.26570 | 0.26067 |
| association_s2_c0.03 | 0.29749 | 0.27878 |
| als_32_r0.1 | 0.28278 | 0.26352 |
| hybrid_c0_a0.15 | 0.41123 | 0.39552 |

Selected **hybrid_c0_a0.15** wins development at 0.41123, with 28-day decay, availability exponent .25 and association weight .15 after row-max normalization. Its audit MAP is 0.39552, versus 0.39734 for personal frequency. Paired difference -0.00182, bootstrap 95% CI [-0.00634,+0.00257]. The evidence does not establish an advantage on the longer horizon. The designated final model preserves the frozen development decision.

## Deliverables

- `submission.csv`: exact designated CSV, same bytes as `outputs/submission.csv`.
- `reports/LLM4Rec_Midterm_Report.pdf`: English academic report.
- `reports/figures/`: measured charts and genuine Kaggle screenshot if available.
- `results/final_model_metrics.json`: configuration, offline metrics and CSV SHA-256.
- `results/external_evidence.json`: verified GitHub/Kaggle evidence, if available.
- `results/independent_audit.json`: independent MAP, candidates, metadata and CSV verification.
- `results/two_users_all_configurations.csv`: 310 recommendation rows for exactly two users across all configurations, with real IDs/titles/categories and interpretations.
- `outputs/selected_users_analysis.csv`: six representative methods for the same two users.
- `outputs/predictions_*.npz`, `labels_*.parquet`: re-evaluable validation evidence.
- `configs/best_model.yaml`: final configuration.
- `session-export/`: local native session snapshot and integrity manifest; raw sessions are excluded from public Git.

Use `--stage preprocess`, `experiments`, `predict` or `report` for separate phases. The report phase creates Markdown/EDA, then `scripts/generate_report.py` creates the PDF. `scripts/finalize_artifacts.py` independently verifies every saved MAP, main candidate membership, real case metadata and the final CSV. Its validator requires exact template users/order/schema and five unique positive eligible IDs per row.

## Session and external submission

No Kaggle score is inferred from local validation. A screenshot and score metadata are saved only when the authenticated website confirms them. Late submission selection can be disabled by Kaggle; the course designation is recorded in the submission description and evidence.

The actual Codex JSONL is copied byte-for-byte with SHA-256 verification. During an active chat it is a snapshot. `scripts/refresh_session_export.ps1` refreshes the complete original after the final response, without translating, rewriting or fabricating messages. Instructor acceptance of native Codex JSONL instead of an OpenCode-specific `session.json` remains unconfirmed. The unrelated warning-only `session.json` is not a transcript and is not offered as evidence.

## Provenance and limitations

Reuses and extends the prior local solution `Documents/ChatGPT/HSE/sweet_tv_exam`. This run rebuilds source preprocessing, interval matches and all predictions. Added work includes HitRate, timing/memory, independent validation, expanded user examples, PDF authoring and external evidence. Metadata is incomplete, source times lack offsets, household accounts are not individuals, and cold cohorts are small. No production load test, revenue lift, retention lift or satisfaction improvement is claimed.

If publication is unavailable:

```powershell
gh auth login
gh repo create Sasmas314/LLM4Rec-Midterm-2026 --public --source . --remote origin --push
```

Do not commit raw datasets, session JSONL, credentials, private tokens or `.env` files.

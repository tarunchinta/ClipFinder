# Distill vs TwelveLabs search benchmark

Four files, run in order:

| File | Role |
|---|---|
| `create_twelvelabs_index.py` | Creates the TL index. Writes `twelvelabs_index_id` into the state file. |
| `sync_twelvelabs_index.py` | Uploads corpus videos (from `eval_corpus.json`) into that index. Records the `drive_file_id` → `tl_video_id` mapping. |
| `eval_queries.json` | Queries, relevance judgments, and arm configurations. |
| `run_eval.py` | Runs every arm against every query in parallel and scores both systems. |
| `eval_search.py` | Pre-existing Distill-only harness with per-leg attribution. Reads the same `eval_queries.json`. |

State lives in `eval_tl_sync_state.json`, which `run_eval.py` reads but never writes.

## Setup

```bash
cd backend
python create_twelvelabs_index.py            # once
python sync_twelvelabs_index.py --user you@example.com
python run_eval.py --user you@example.com --dry-run
```

`--dry-run` validates config, corpus and judgments without spending API calls.

## Arms

An arm is one system under one configuration. Arms sharing a query all run
concurrently, so a leg sweep costs roughly one query's wall clock rather than
one per arm.

```json
{ "name": "distill-visual-only", "system": "distill",
  "distill": { "legs": ["thumbnail", "frame"] } }

{ "name": "tl-visual", "system": "twelvelabs",
  "twelvelabs": { "search_options": ["visual"] } }
```

Config resolves in three layers, later winning: `defaults` → arm → per-query
override. A `null` never overrides.

**Distill legs** (`text`, `thumbnail`, `frame`, `caption`, `transcript`,
`color`) map to `IndexingService.hybrid_search_rrf`'s RRF legs. Omitting `legs`
runs all six. A disabled leg is not queried at all, and disabling every
embedding leg skips the query-embedding call entirely. `caption` and
`transcript` each fuse a lexical and a semantic sub-leg; they cannot currently
be isolated from each other.

**TwelveLabs options** pass through to `search.query`: `search_options`,
`group_by`, `operator`, `transcription_options`, `filter`. `search_options`
must be a subset of the index's `model_options`, which are fixed at index
creation — `run_eval.py` checks this up front and fails with the mismatch
rather than erroring once per query.

## Judgments

`eval_queries.json` is shared with `eval_search.py`, so `queries[]` follows
that script's schema and the judgments are authored once:

```json
{ "id": "q001", "query": "someone unboxing a package",
  "file_type": null, "expect_color_leg": false, "tags": ["visual"],
  "relevant": [
    { "drive_file_id": "instagram:DX5ktNIOwx7", "gain": 3 },
    { "drive_file_id": "instagram:C8xY1zQrLmN" }
  ] }
```

Every entry in `relevant` counts for recall/precision/MRR/MAP/hit-rate. `gain`
is optional, defaults to 1.0, and only affects nDCG — use graded gains (3 =
ideal answer, 1 = acceptable) to make nDCG say something. `run_eval.py` also
accepts a bare `"drive_file_id"` string as shorthand for gain 1.0.

`k_values`, `defaults`, `arms`, and each query's `id`/`tags` are `run_eval.py`
additions that `eval_search.py` ignores; `file_type` and `expect_color_leg` are
`eval_search.py` fields that `run_eval.py` ignores.

Ground truth is authored by hand. A query with no `relevant` entries produces
undefined metrics and is excluded from averages rather than scored as zero, and
the runner warns about it.

Both systems are normalized to ranked lists of `drive_file_id` before scoring,
so recall@k compares like with like: TL returns clips keyed by video, Distill
returns files. TL hits resolve via the `drive_file_id` written into
`user_metadata` at index time, falling back to the state file's id map.

## Running

```bash
python run_eval.py --user you@example.com
python run_eval.py --user you@example.com --arm distill-full --arm tl-visual
python run_eval.py --user you@example.com --query-id q001 --verbose
python run_eval.py --user you@example.com --tag speech --out results.json
```

`--out` writes per-query rankings, per-leg score breakdowns for Distill, and
clip timestamps for TL. `--concurrency` caps in-flight searches (default 8).
Exit code is 1 if any search failed.

## Caveats

- Metrics on a corpus of a handful of videos are noise. The runner warns below
  10. Grow `eval_corpus.json` and re-sync before drawing conclusions.
- A judged video that is not `status=ready` in the state file caps recall for
  TL arms, since TL cannot return what it has not indexed. The runner warns.
- `eval_search.py` still carries its own copy of the recall/precision/MRR/nDCG
  math. The two implementations agree numerically today, but converging it onto
  `eval_metrics.py` would remove the chance of them drifting apart.
- The current index was built with `model_options: ["visual", "audio"]`, so
  `transcription` search is unavailable on it. Benchmarking Distill's
  transcript leg against TL's transcription modality needs a new index and a
  full re-sync.

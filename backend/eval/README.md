# Distill vs TwelveLabs search benchmark

This package lives under `backend/eval/`. Run commands from `backend/`.

| File | Role |
|---|---|
| `create_twelvelabs_index.py` | Creates the TL index. Writes `twelvelabs_index_id` into the state file. |
| `sync_twelvelabs_index.py` | Uploads corpus videos (from `corpus.json`) into that index. Records the `drive_file_id` → `tl_video_id` mapping. |
| `queries.json` | Queries, relevance judgments, tags, and arm configurations. |
| `legs.py` | Maps `tags[]` / `expect_color_leg` onto Distill legs and TL `search_options`. |
| `run.py` | Runs every arm against every query in parallel and scores both systems. |
| `search.py` | Distill-only harness with per-leg attribution. Reads the same `queries.json`. |

State lives in `tl_sync_state.json`, which `eval.run` reads but never writes.

## Setup

```bash
cd backend
python -m eval.create_twelvelabs_index            # once
python -m eval.sync_twelvelabs_index --user you@example.com
python -m eval.run --user you@example.com --dry-run
```

`--dry-run` validates config, corpus and judgments without spending API calls.

Unit tests (from `backend/`):

```bash
python -m unittest discover -s tests
```

## Tags select modalities

Each query lists Distill legs in `tags[]`. Color is not a tag: set `expect_color_leg` true to enable Distill's color signature. The mapping (and comments) live in `legs.py`:

| Query field | Distill | TwelveLabs |
|---|---|---|
| `text` | filename trigram | (none) |
| `thumbnail` | poster embedding | (none — TL index addon `thumbnail`, not a search filter) |
| `frame` | video frame embeddings | `visual` |
| `caption` | caption lexical + semantic | `audio` |
| `transcript` | transcript lexical + semantic | `audio` |
| `expect_color_leg: true` | color signature | (none) |

`eval.run` and `eval.search` both call `eval.legs`, so a tagged query runs the same Distill legs in both harnesses. Unknown tags and `"color"` inside `tags[]` fail at load time.

`--tag thumbnail` still filters to queries that include that Distill-leg name; it does not change which legs fire.

Ablation is a query with a subset of tags (for example only `thumbnail` and `frame`), not a visual-only arm.

## Arms

An arm is one system to search. The shipped file has two: `distill` and `twelvelabs`. Arms sharing a query run concurrently.

```json
{ "name": "distill", "system": "distill" }
{ "name": "twelvelabs", "system": "twelvelabs" }
```

Config resolves in three layers, later winning: `defaults` → arm → **query tags / `expect_color_leg`** → per-query `"distill"` / `"twelvelabs"` override. A `null` never overrides.

Empty `tags` plus `expect_color_leg` other than true means "use the arm" (typically every Distill leg / the arm's `search_options`). Non-empty tags replace the arm's Distill `legs` and TL `search_options` for that query.

**Distill legs** (`text`, `thumbnail`, `frame`, `caption`, `transcript`,
`color`) map to `IndexingService.hybrid_search_rrf`'s RRF legs. Omitting `legs`
runs all six. A disabled leg is not queried at all, and disabling every
embedding leg skips the query-embedding call entirely. `caption` and
`transcript` each fuse a lexical and a semantic sub-leg; they cannot currently
be isolated from each other.

**TwelveLabs options** pass through to `search.query`: `search_options`,
`group_by`, `operator`, `transcription_options`, `filter`. `search_options`
must be a subset of the index's `model_options`, which are fixed at index
creation — `eval.run` checks the union of tag-derived options up front and
fails with the mismatch rather than erroring once per query. `text` and color
have no TL analog; a query that only sets those needs an explicit
`twelvelabs.search_options` override or it is rejected. TL **thumbnail** is
`addons: ["thumbnail"]` at index creation, not a `search_options` value;
`eval.run` checks the addon is present and records `thumbnail_url` on hits.

## Judgments

`queries.json` is shared with `eval.search`, so `queries[]` follows
that script's schema and the judgments are authored once:

```json
{ "id": "q001", "query": "someone unboxing a package, warm orange look",
  "file_type": null, "expect_color_leg": true,
  "tags": ["text", "thumbnail", "frame", "caption", "transcript"],
  "relevant": [
    { "drive_file_id": "instagram:DX5ktNIOwx7", "gain": 3 },
    { "drive_file_id": "instagram:C8xY1zQrLmN" }
  ] }
```

Every entry in `relevant` counts for recall/precision/MRR/MAP/hit-rate. `gain`
is optional, defaults to 1.0, and only affects nDCG — use graded gains (3 =
ideal answer, 1 = acceptable) to make nDCG say something. `eval.run` also
accepts a bare `"drive_file_id"` string as shorthand for gain 1.0.

`k_values`, `defaults`, `arms`, and each query's `id` are `eval.run`
additions that `eval.search` ignores. `tags` and `expect_color_leg` are
shared: they select Distill legs in both tools, and `eval.search` still
asserts the color-score contract when `expect_color_leg` is set. `file_type`
is an `eval.search` field that `eval.run` only honors via a per-query
`"distill": { "file_type": ... }` block.

Ground truth is authored by hand. A query with no `relevant` entries produces
undefined metrics and is excluded from averages rather than scored as zero, and
the runner warns about it.

Both systems are normalized to ranked lists of `drive_file_id` before scoring,
so recall@k compares like with like: TL returns clips keyed by video, Distill
returns files. TL hits resolve via the `drive_file_id` written into
`user_metadata` at index time, falling back to the state file's id map.

## Running

```bash
python -m eval.run --user you@example.com
python -m eval.run --user you@example.com --arm distill --arm twelvelabs
python -m eval.run --user you@example.com --query-id q001 --verbose
python -m eval.run --user you@example.com --tag transcript --out results.json
```

`--out` writes per-query rankings, per-leg score breakdowns for Distill, and
clip timestamps for TL. `--concurrency` caps in-flight searches (default 8).
Exit code is 1 if any search failed.

## Caveats

- Metrics on a corpus of a handful of videos are noise. The runner warns below
  10. Grow `corpus.json` and re-sync before drawing conclusions.
- A judged video that is not `status=ready` in the state file caps recall for
  TL arms, since TL cannot return what it has not indexed. The runner warns.
- `eval.search` still carries its own copy of the recall/precision/MRR/nDCG
  math. The two implementations agree numerically today, but converging it onto
  `eval.metrics` would remove the chance of them drifting apart.
- The current index was built with `model_options: ["visual", "audio"]`, so
  `transcription` search is unavailable on it. Benchmarking Distill's
  transcript leg against TL's transcription modality needs a new index and a
  full re-sync. Distill `transcript` currently maps to TL `audio`.

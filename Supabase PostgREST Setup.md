# Supabase setup for the PostgREST worker path

The Service Bus triggers no longer connect to Postgres. They read and write
through Supabase's PostgREST endpoint over one pooled HTTP connection per queue
message. This is what has to exist on the Supabase side for that to work.

## 1. Apply the migration

`supabase/migrations/20260903000000_distill_indexing_rpc.sql` creates seven
functions. Either:

```bash
supabase db push
```

or open the SQL editor (Dashboard → SQL Editor) and paste the file's contents.

| Function | Why it is a function and not a REST call |
|---|---|
| `distill_upsert_frame_embedding` | Upserts the frame **and** advances `frames_completed` **and** runs the completion check in one transaction |
| `distill_record_frame_failure` | Same, for `frames_failed`; skips frames that already stored an embedding |
| `distill_finalize_video_indexing` | Shared "is this video done" check |
| `distill_set_thumbnail_embedding` | Writes a `vector` column |
| `distill_set_vision_embedding` | Writes a `vector` column |
| `distill_set_color_signature` | Writes a `vector` column plus the Lab scalars |
| `distill_replace_transcript_segments` | Delete + insert in one transaction |

Two constraints drove that split:

* **Counters must be atomic.** Up to `FRAME_INDEX_PARALLELISM` frames finish at
  once. `frames_completed + 1` as a read-then-write over HTTP loses counts, and
  a video whose counters never reach `frames_total` is stuck in `processing`
  forever.
* **PostgREST cannot type a `vector`.** It has no way to know a JSON array is
  destined for a pgvector column, so embeddings travel as text (`[0.1,0.2,…]`)
  and the function casts them.

The functions take `FOR NO KEY UPDATE` on the `indexed_files` row *before*
inserting into `video_frame_embeddings`. That ordering matters: the child insert
takes a `FOR KEY SHARE` lock on the parent through the foreign key, and
incrementing the counter afterwards upgrades that lock — concurrent frames
upgrading at the same time deadlock. Locking the parent first makes them queue.

## 2. Reload the API schema cache

PostgREST caches the schema, so new functions 404 until it re-reads. The
migration ends with the notify, but if you applied it another way:

```sql
notify pgrst, 'reload schema';
```

Dashboard → Settings → API → "Reload schema cache" does the same thing.

## 3. Confirm the tables are exposed

Dashboard → Settings → API → **Exposed schemas** must include `public`. The
worker touches `indexed_files`, `video_frame_embeddings`,
`video_transcript_segments`, and `user`.

## 4. Grab the service_role key

Dashboard → Settings → API → Project API keys → `service_role` (**not** `anon`).

The worker needs it for two reasons: it bypasses RLS, and the RPCs are granted to
`service_role` only. Treat it as a secret — it is full database access, so it
belongs in the Function App's app settings, never in client code or the repo.

## 5. Set the environment variables

On the Function App (Configuration → Application settings) and in
`backend/.env` for local runs:

| Variable | Value | Notes |
|---|---|---|
| `SUPABASE_URL` | `https://<project-ref>.supabase.co` | Already set if frame images upload to Supabase Storage |
| `SUPABASE_SERVICE_ROLE_KEY` | the service_role key from step 4 | Falls back to `SUPABASE_KEY` if unset, but only works if that key is service_role |
| `POSTGREST_MAX_CONNECTIONS` | `1` | The "one connection per trigger" guarantee. Raise it only if a single trigger becomes HTTP-bound |
| `POSTGREST_TIMEOUT_SECONDS` | `30` | Optional |

`DATABASE_URL` still has to be set on the Function App even though the worker no
longer queries Postgres: `app/models/` imports `app/database.py`, which builds a
SQLAlchemy engine at import time. The engine is lazy and never opens a socket on
this path, but a missing or malformed URL fails the import. The web app still
uses it for real — hybrid search needs pgvector SQL that PostgREST cannot
express.

## 6. RLS

Nothing to change. The `service_role` key bypasses RLS and the functions are
`SECURITY DEFINER`, so existing policies keep applying to the web app and to
anon/authenticated clients exactly as before.

## Verifying it works

Index one reel and watch the video row. With `frames_total` set, every frame
should land:

```sql
select indexing_status, frames_total, frames_completed, frames_failed, indexed_at
  from indexed_files
 where id = '<video-id>';
```

`frames_completed + frames_failed` must reach `frames_total`, and
`indexing_status` must flip to `completed`. If frames stall short of the total,
check the Function App logs for `PostgrestError` — a 404 on `/rpc/...` means the
schema cache was not reloaded (step 2), and a 401/403 means the key is `anon`
rather than `service_role` (step 4).

## What this changed about connection use

| | Before | After |
|---|---|---|
| Postgres connections per trigger | up to ~12 (1 job + 8 frames + thumbnail + colour + transcript, each `async_session_maker()`) | 0 |
| Connections to Supabase per trigger | — | 1 pooled HTTPS connection |
| Frame counter updates | separate increment and completion check across 8 sessions | one atomic RPC |

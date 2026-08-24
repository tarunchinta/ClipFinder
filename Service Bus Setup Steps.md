# Azure Service Bus Setup — ClipFinder

This guide covers everything needed for indexing to work end-to-end after the Service Bus-ready code changes. Steps are ranked by priority: **P0** blocks all indexing, **P1** blocks job processing, **P2** is production hardening, **P3** is verification.

---

## How it works (quick reference)

```
User clicks Index (FastAPI)
  → save metadata to Postgres (no Gemini on HTTP path)
  → publish messages to Service Bus
       • image-indexing queue  →  Azure Function  →  thumbnail Gemini embed
       • frame-indexing queue  →  Azure Function  →  ffmpeg + frame Gemini embed

If Service Bus is not configured → API returns 503.
If publish fails transiently → in-process asyncio fallback in the API container.
```

**Message shapes**

| Queue | Default name | JSON body |
|-------|--------------|-----------|
| Video | `frame-indexing` | `{"video_id": "<uuid>", "trace_id": "...", "parent_span_id": "..."}` |
| Image | `image-indexing` | `{"file_id": "<uuid>", "trace_id": "...", "parent_span_id": "..."}` |

---

## P0 — Required before indexing works at all

Without these, `POST /api/drive/folders/{folder_id}/index` returns **503** or messages never leave the API.

### 1. Create an Azure Service Bus namespace

1. Azure Portal → **Create a resource** → **Service Bus**.
2. Choose a tier:
   - **Basic** is enough for MVP (queues only, no topics).
   - **Standard** if you want dead-letter inspection UI and longer message TTL controls.
3. Note the namespace name (e.g. `clipfinder-bus`).

### 2. Create two queues

In the namespace → **Queues** → **+ Queue**:

| Queue name | Purpose | Suggested settings (MVP) |
|------------|---------|--------------------------|
| `frame-indexing` | Video frame extraction + embedding | Max delivery count: `5` (tune later) |
| `image-indexing` | Image thumbnail vision embedding | Max delivery count: `5` |

Enable **dead-lettering on message expiration** on both queues (you can configure max delivery in P2).

Queue names must match your env vars unless you override defaults.

### 3. Create a Shared Access Policy and copy the connection string

1. Namespace → **Shared access policies** → **+ Add**.
2. Name it e.g. `SendListen` with permissions **Send** and **Listen** (Functions need listen; API needs send).
3. Open the policy → copy **Primary connection string**.

Format:

```
Endpoint=sb://<namespace>.servicebus.windows.net/;SharedAccessKeyName=SendListen;SharedAccessKey=...
```

### 4. Configure the FastAPI backend `.env`

In `backend/.env` (see `backend/env.template`):

```env
SERVICE_BUS_SAS_PRIMARY_CONNECTION_STRING=Endpoint=sb://...
VIDEO_INDEXING_QUEUE=frame-indexing
IMAGE_INDEXING_QUEUE=image-indexing
```

Restart the backend after saving.

### 5. Confirm the API can reach Service Bus

- Start the app (`python run.py` or your container).
- Trigger an index on a folder with at least one image or video.
- **Success:** HTTP 200, files show `indexing_status=pending` (videos) or `vision_indexing_status=pending` (images).
- **Failure:** HTTP 503 with `"Service Bus not configured; indexing unavailable"` → connection string missing or not loaded.

> At this point messages are **published** but **not processed** until P1 is done.

---

## P1 — Required before jobs are actually processed

The API only enqueues work. Azure Functions (or another consumer) must drain the queues.

### 6. Ensure worker dependencies are available

Functions call the same code as the API (`app.tasks._run_*_indexing_async`). Each consumer needs:

| Dependency | Used for | Env var(s) |
|------------|----------|------------|
| Postgres + pgvector | File rows, embeddings, tokens | `DATABASE_URL` |
| Gemini Embedding 2 | Image + frame embeddings | `GEMINI_API_KEY` |
| Google OAuth tokens | Download Drive thumbnails/videos | Stored in DB; `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` for refresh |
| Azure Blob or Supabase | Frame JPEG storage | `AZURE_BLOB_*` or `SUPABASE_*` |
| **ffmpeg** | Video frame extraction | Must be on `PATH` in the Functions image |

Optional but recommended: `LANGFUSE_*`, `FRAME_INDEX_PARALLELISM`.

### 7. Add ffmpeg to the Functions container (video indexing)

The stock `functions/Dockerfile` does **not** install ffmpeg. Video jobs will fail without it.

Add to `functions/Dockerfile` before `pip install`:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

Rebuild and redeploy after adding this.

### 8. Build and deploy Azure Functions

From the **repo root**:

```powershell
docker build -f functions/Dockerfile -t clipfinder-functions:latest .
```

Deploy to **Azure Functions** (Linux, custom container) or **Azure Container Apps** with the Functions host image. Minimum app settings / environment variables:

```env
SERVICE_BUS_SAS_PRIMARY_CONNECTION_STRING=Endpoint=sb://...
VIDEO_INDEXING_QUEUE=frame-indexing
IMAGE_INDEXING_QUEUE=image-indexing
DATABASE_URL=postgresql+asyncpg://...
GEMINI_API_KEY=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
# Blob or Supabase (at least one for video frames)
AZURE_BLOB_CONNECTION_STRING=...
AZURE_BLOB_STORAGE_CONTAINER_NAME=video-frames
# or
SUPABASE_URL=...
SUPABASE_KEY=...
SUPABASE_STORAGE_BUCKET=video-frames
```

`functions/function_app.py` registers two triggers:

- `video_frame_index_trigger` → `%VIDEO_INDEXING_QUEUE%`
- `image_vision_index_trigger` → `%IMAGE_INDEXING_QUEUE%`

Both use connection setting name `SERVICE_BUS_SAS_PRIMARY_CONNECTION_STRING` (Azure resolves this from app settings).

### 9. Scale Functions for workload

| Workload | Guidance |
|----------|----------|
| Image jobs | Lightweight; default consumption plan is usually fine |
| Video jobs | CPU-heavy (ffmpeg + many Gemini calls); consider Premium plan or Container Apps with higher CPU/memory |
| Concurrency | Service Bus triggers scale out per message; long videos can run many minutes — set function timeout accordingly (max 10 min on Consumption; longer on Premium/dedicated) |

### 10. Grant network access to Postgres

If the database is private (Supabase, Azure Postgres, VNet):

- Functions host must reach `DATABASE_URL` (allow-list outbound IP or use VNet integration).
- Same for blob storage and `generativelanguage.googleapis.com` (Gemini).

---

## P2 — Production hardening (do before real users)

### 11. Configure dead-letter queues and max delivery

Per queue in Azure Portal:

1. **Max delivery count** — e.g. `5` (after 5 failures message moves to dead-letter sub-queue).
2. **Dead lettering** — enable on message expiration and filter evaluation failures.
3. Monitor dead-letter depth; alert if &gt; 0.

Failed Function invocations **rethrow** exceptions so the runtime retries and eventually dead-letters.

### 12. Separate send vs listen credentials (recommended)

For least privilege:

| Component | SAS policy permissions |
|-----------|------------------------|
| FastAPI API | **Send** only on both queues |
| Azure Functions | **Listen** only on both queues |

Use different connection strings in API `.env` vs Functions app settings. Update `service_bus_publisher.py` to use a send-only policy string.

### 13. Set message TTL and lock duration

- **Lock duration** — default 60s may be too short for long video jobs; increase to 5 minutes on `frame-indexing` if you see duplicate processing.
- **Message TTL** — e.g. 7 days so stale jobs don't run after a user deletes a folder.

### 14. Application Insights / logging

- Enable Application Insights on the Function App.
- Correlate logs with `video_id` / `file_id` from trigger log lines.
- Optional: wire Langfuse in Functions using the same `LANGFUSE_*` keys as the API.

### 15. Keep API container ffmpeg-free (optional)

Only Functions need ffmpeg for video indexing. The API container publishes messages only — no ffmpeg required there.

---

## P3 — Verification and ongoing ops

### 16. End-to-end smoke test

1. Sign in with Google (Drive connected).
2. Index a folder with **one image** and **one short video**.
3. Check:

| Check | Where | Expected |
|-------|-------|----------|
| API response | Browser / Network tab | 200, fast response |
| Queue depth | Service Bus → Queues → Active messages | Brief spike, then → 0 |
| Image completion | DB `indexed_files` | `vision_indexing_status=completed`, `vision_embedding` populated |
| Video completion | DB `indexed_files` | `indexing_status=completed`, `frames_completed = frames_total` |
| Frame rows | `video_frame_embeddings` | One row per sampled frame |
| Search | `/search?q=...` | Image and video frame hits |

### 17. Confirm idempotency under redelivery

Service Bus is at-least-once. The codebase guards `frames_completed` so duplicate frame messages should not inflate counters. If you manually re-submit a message, counters should stay correct.

### 18. Monitor queue health

| Metric | Action if bad |
|--------|----------------|
| Active messages growing | Functions down, failing, or under-scaled |
| Dead-letter count &gt; 0 | Inspect dead-letter messages; fix root cause (token expiry, missing ffmpeg, Gemini quota) |
| Function failures | Check exception in App Insights |

### 19. Common failure modes

| Symptom | Likely cause |
|---------|----------------|
| 503 on index | `SERVICE_BUS_SAS_PRIMARY_CONNECTION_STRING` missing in API `.env` |
| Messages stuck in queue | Functions not deployed, wrong queue names, or listen SAS missing |
| Video jobs fail immediately | ffmpeg not in Functions image |
| Image jobs fail | `GEMINI_API_KEY` missing or invalid |
| `Could not get valid Google access token` | User needs to re-authenticate; refresh token expired |
| Frames stuck at `processing` | Function timeout, crash mid-job, or blob upload misconfigured |

---

## Checklist (copy/paste)

```
P0 — Blocking
[ ] Service Bus namespace created
[ ] Queues: frame-indexing, image-indexing
[ ] SAS connection string copied
[ ] backend/.env: SERVICE_BUS_SAS_PRIMARY_CONNECTION_STRING, VIDEO_INDEXING_QUEUE, IMAGE_INDEXING_QUEUE
[ ] API restarted; index returns 200 (not 503)

P1 — Processing
[ ] ffmpeg added to functions/Dockerfile
[ ] Functions image built and deployed
[ ] Functions env: Service Bus + DATABASE_URL + GEMINI_API_KEY + Google OAuth + blob storage
[ ] Functions can reach Postgres and external APIs
[ ] Test index: queues drain to 0

P2 — Hardening
[ ] Max delivery count + dead-letter configured
[ ] Send-only vs listen-only SAS split (optional)
[ ] Lock duration / timeout tuned for video jobs
[ ] Application Insights enabled

P3 — Verify
[ ] Image: vision_indexing_status=completed
[ ] Video: indexing_status=completed, frames in DB
[ ] Search returns results
[ ] Dead-letter queues empty after smoke test
```

---

## Local development options

| Approach | Pros | Cons |
|----------|------|------|
| **Point local API at real Azure Service Bus + deploy Functions** | Matches production | Needs Azure resources |
| **Publish failure fallback** | API falls back to in-process asyncio if send throws | Only triggers on exception, not when bus is unset |
| **Unset connection string** | — | Index always returns 503 (by design) |

There is no offline queue emulator in this repo. For local work, use a dev Service Bus namespace or temporarily rely on the publish-failure in-process fallback.

---

## Related files in this repo

| File | Role |
|------|------|
| `backend/app/services/service_bus_publisher.py` | Publishes jobs from API |
| `backend/app/routers/drive.py` | 503 if bus missing; enqueue after save |
| `backend/app/tasks.py` | `_run_frame_indexing_async`, `_run_image_indexing_async` |
| `functions/function_app.py` | Service Bus triggers |
| `functions/Dockerfile` | Functions container build |
| `backend/env.template` | Env var reference |

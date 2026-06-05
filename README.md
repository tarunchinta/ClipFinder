# ClipFinder

**Semantic search for video clips and images. Find a shot in seconds instead of scrubbing through hours of footage.**

[![Python](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Postgres + pgvector](https://img.shields.io/badge/Postgres-pgvector-336791.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/license-MIT-gray.svg)](LICENSE.txt)

---

## The Problem

Content creators accumulate thousands of clips and images, usually in Google Drive. Drive search can't see what's inside a video or image so finding "the shot of a red scooter pulling away at night" means scrubbing through folders by hand.

ClipFinder makes a media library searchable by content. Point it at a Drive folder; it indexes every image and video frame, and a plain-text query like `red car` returns the matching files and frame timestamps.

## Demo

<video src="https://github.com/tarunchinta/ClipFinder/raw/main/clipfinder-demo.mp4" controls width="100%"></video>

> If the player above doesn't load, [watch the demo here](https://github.com/tarunchinta/ClipFinder/raw/main/clipfinder-demo.mp4).

---

## How It Works

ClipFinder is a single **FastAPI** service. After a user signs in with Google (read-only Drive scope) and picks a folder, indexing and search run through one pipeline:

```
                          ┌──────────────────────────────────────────────┐
   Google OAuth           │                 FastAPI app                   │
   (drive.readonly)       │                                               │
        │                 │  Jinja UI ── /dashboard (picker) ── /search   │
        ▼                 │                     │                         │
   ┌──────────┐  list/    │   ┌─────────────────┴──────────────────┐      │
   │  Google  │  download │   │            Indexing                │      │
   │  Drive   │◀──────────┼──▶│  • upsert file metadata            │      │
   └──────────┘           │   │  • CLIP embed thumbnails (images)  │      │
        ▲                 │   │  • ffmpeg → frames → CLIP embed    │──┐   │
        │ frames stored   │   │  • text-embed filenames            │  │   │
        │                 │   └────────────────────────────────────┘  │   │
   ┌──────────┐           │   ┌────────────────────────────────────┐  │   │
   │  Blob     │◀─────────┼───│            Search                  │  │   │
   │  storage  │  frame    │   │  query ─┬─ pg_trgm (filenames)     │  │   │
   │ (Azure/   │  JPEGs    │   │         └─ CLIP text→image (visual)│  │   │
   │  Supabase)│           │   │     fuse → ranked frames + files   │  │   │
   └──────────┘           │   └────────────────────────────────────┘  │   │
                          │                     │                      │   │
                          └─────────────────────┼──────────────────────┼───┘
                                                 ▼                      ▼
                                       ┌───────────────────────────────────┐
                                       │   Postgres 16 + pgvector          │
                                       │   • indexed_files                 │
                                       │       filename_embedding  (1536)  │
                                       │       vision_embedding    (768)   │
                                       │   • video_frame_embeddings (768)  │
                                       │   HNSW (cosine) + pg_trgm GIN     │
                                       └───────────────────────────────────┘
```

### Indexing Pipeline

1. **List & validate** files in the selected folder via the Drive API (images + videos, ≤100 MB, videos ≤30 s).
2. **Images:** download the Drive thumbnail and embed it with CLIP → a 768-dim `vision_embedding`.
3. **Videos:** download the file, run **ffmpeg** server-side to sample frames (every 5th frame), embed each frame with CLIP, and store per-frame embeddings with their **timestamp** so search can deep-link into the exact moment. Frame JPEGs are persisted to blob storage (Azure Blob preferred, Supabase Storage as fallback).
4. **Filenames** are embedded with `text-embedding-3-small` and stored alongside metadata.
5. Heavy video work is offloaded to an **in-process `asyncio` background task** so the index request returns quickly.

### Retrieval

The shipped search path is **hybrid**: it fuses a lexical signal over filenames with a dense visual signal over image/frame embeddings.

- **Lexical leg** — Postgres `pg_trgm` trigram similarity on filenames.
- **Visual leg** — the query text is embedded with **CLIP's text encoder** and compared (cosine) against image and video-frame embeddings via **pgvector**. Because CLIP maps text and images into a *shared* vector space, a text query can be matched directly against pixels.
- **Fusion** — each leg's scores are min-max normalized and combined with a weighted sum, then re-ranked. Video matches surface the specific frame and its timestamp.

---

## Key Engineering Decisions & Tradeoffs

**CLIP for cross-modal retrieval.** CLIP embeds images and text into one shared space, so a text query can be compared directly to image embeddings via similarity search, with no captioning or object-detection step in between.

**Hybrid retrieval, without a semantic filename leg.** The pipeline can embed filenames semantically and an endpoint for it exists, but in testing filenames were already lexically close to their content and CLIP visual retrieval surfaced the same results. The shipped search therefore fuses lexical filename matching (`pg_trgm`) with CLIP visual search, keeping the cheap signal cheap and spending the expensive signal on pixels.

**Postgres + pgvector instead of a dedicated vector DB.** One datastore holds metadata, lexical indexes (`pg_trgm` GIN), and vector indexes (HNSW, cosine). This removes a moving part, keeps writes transactional, and stays portable to any managed Postgres (Supabase, Neon, RDS).

**Hosted embedding endpoints instead of local model weights.** Both encoders are called over HTTP — `text-embedding-3-small` via Azure OpenAI (1536-dim) and OpenAI's CLIP hosted on Azure ML (768-dim). This keeps the app container small and CPU-only and lets the model deployment scale independently. The tradeoff is per-call latency and a dependency on endpoint availability, which the indexing layer absorbs by running asynchronously.

**Frame sampling over full decode.** Sampling every 5th frame keeps embedding volume and storage proportional to motion while still capturing distinct moments. Each frame keeps its computed timestamp so results jump straight to the moment in Drive.

**In-process async over a message broker.** Video indexing runs as an `asyncio` background task, so the index request returns immediately with no Redis/Celery to operate. An Azure Service Bus producer path is in place, so the work can move to an external worker without reshaping the API.

---

## Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| API | FastAPI (Python 3.11) | Async, auto OpenAPI docs at `/docs` |
| UI | Server-rendered Jinja2 | Landing, dashboard (Drive picker), search |
| Auth | fastapi-users + Google OAuth2 | `drive.readonly` scope, JWT cookie session |
| Database | PostgreSQL 16 + pgvector | HNSW (cosine) vector indexes + `pg_trgm` GIN |
| Text embeddings | Azure OpenAI `text-embedding-3-small` | 1536-dim, over HTTP |
| Visual embeddings | OpenAI CLIP on Azure ML | 768-dim image & text encoders, shared space |
| Video | ffmpeg / ffprobe (subprocess) | Frame sampling + timestamps |
| Frame storage | Azure Blob / Supabase | SAS-signed frame image URLs |
| Async | `asyncio` background tasks | Service Bus producer path available |
| Observability | Langfuse | Traces embedding + retrieval calls |
| Migrations | Alembic | `vector` + `pg_trgm` extensions, table DDL |

---

## Scope and Limitations

- Search is lexical (filenames) + CLIP visual; transcript/audio search is not implemented.
- Video frame indexing requires `ffmpeg` on the host and configured Azure embedding/blob credentials.
- `indexing_status` reflects enqueue, not end-to-end per-file completion.
- Stripe and usage-limit fields exist in config and schema but are not enforced.

---

## Running Locally

### Prerequisites

- Python 3.11+
- PostgreSQL 16 with the `vector` and `pg_trgm` extensions (or use the included `docker-compose.yml`)
- `ffmpeg` / `ffprobe` on your `PATH` (required for video frame indexing)
- Google Cloud OAuth credentials (Drive API enabled)
- Azure OpenAI + Azure ML CLIP endpoints for embeddings

### Setup

```bash
git clone <your-repo-url>
cd ClipFinder-MVP/backend

python -m venv venv
# Windows: .\venv\Scripts\activate
source venv/bin/activate

pip install -r requirements.txt

cp env.template .env   # then fill in the values below
```

Minimum `.env` values:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/clipfinder
JWT_SECRET=run-`openssl rand -hex 32`

GOOGLE_CLIENT_ID=...apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=...
GOOGLE_API_KEY=...                       # for the Drive folder picker

AZURE_OPENAI_ENDPOINT_SAMPLE_FULL=...    # full text-embedding endpoint URL
AZURE_OPENAI_API_KEY=...
AZURE_AI_VISION_ENDPOINT=...             # Azure ML CLIP /score endpoint
AZURE_AI_VISION_KEY=...

# One of the following for frame image storage:
AZURE_BLOB_CONNECTION_STRING=...         # preferred
# SUPABASE_URL / SUPABASE_KEY            # fallback
```

### Database & Server

```bash
# Option A: bring up Postgres + the app together
docker compose up --build

# Option B: local Postgres + Alembic
alembic upgrade head
python run.py        # or: uvicorn app.main:app --reload --port 8000
```

Then open:

- App: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

> Note: the default `Dockerfile` does not install `ffmpeg`; add it to the image if you index videos in a container.

---

## Repository Layout

```
backend/
├── app/
│   ├── main.py                 # FastAPI app factory + routers
│   ├── config.py               # pydantic-settings configuration
│   ├── routers/                # auth, pages (Jinja), drive/search API
│   ├── services/
│   │   ├── embedding.py            # Azure OpenAI text embeddings (1536-d)
│   │   ├── vision_embedding.py     # Azure ML CLIP image/text embeddings (768-d)
│   │   ├── video_frame_indexing.py # ffmpeg frame extraction + embedding
│   │   ├── indexing.py             # search: trigram + vector + hybrid fusion
│   │   └── google_drive.py         # Drive listing / download
│   ├── templates/              # landing / dashboard / search UIs
│   └── observability/          # Langfuse tracing
├── alembic/                    # migrations (vector + pg_trgm)
└── requirements.txt
docker-compose.yml              # app + pgvector/pgvector:pg16
```

---

## License

MIT — see [LICENSE.txt](LICENSE.txt).

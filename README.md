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
   └──────────┘           │   │  • Gemini embed thumbnails (images)│      │
        ▲                 │   │  • ffmpeg → frames → Gemini embed  │──┐   │
        │ frames stored   │   │  • ffmpeg audio → WhisperX →       │  │   │
        │                 │   │    word-timed segments + embeddings│  │   │
   ┌──────────┐           │   └────────────────────────────────────┘  │   │
   │  Blob     │◀─────────┼───┌────────────────────────────────────┐  │   │
   │  storage  │  frame    │   │            Search                  │  │   │
   │ (Azure/   │  JPEGs    │   │  query ─┬─ pg_trgm (filenames)     │  │   │
   │  Supabase)│           │   │         ├─ Gemini text→image (visual)│  │   │
   └──────────┘           │   │         ├─ FTS (transcript text)   │  │   │
                          │   │         └─ Gemini text→transcript  │  │   │
                          │   │   RRF fuse → ranked files + times  │  │   │
                          │   └────────────────────────────────────┘  │   │
                          │                     │                      │   │
                          └─────────────────────┼──────────────────────┼───┘
                                                 ▼                      ▼
                                       ┌───────────────────────────────────┐
                                       │   Postgres 16 + pgvector          │
                                       │   • indexed_files                 │
                                       │       vision_embedding    (768)   │
                                       │   • video_frame_embeddings (768)  │
                                       │   • video_transcript_segments     │
                                       │       text + timestamps + (768)   │
                                       │   HNSW (cosine) + GIN (trgm, FTS) │
                                       └───────────────────────────────────┘
```

### Indexing Pipeline

1. **List & validate** files in the selected folder via the Drive API (images + videos, ≤100 MB, videos ≤30 s).
2. **Images:** download the Drive thumbnail and embed it with Gemini Embedding 2 → a 768-dim `vision_embedding`.
3. **Videos:** download the file, run **ffmpeg** server-side to sample frames (every 5th frame), embed each frame with Gemini Embedding 2, and store per-frame embeddings with their **timestamp** so search can deep-link into the exact moment. Frame JPEGs are persisted to blob storage (Azure Blob preferred, Supabase Storage as fallback).
4. **Transcription (runs alongside frame embedding):** extract the audio track with ffmpeg and transcribe it locally with **WhisperX** (batched Whisper ASR + wav2vec2 forced alignment, CPU/int8). Each speech segment is stored with its **text, segment timestamps, and word-level timestamps**, plus a Gemini text embedding, so spoken words are searchable both lexically and semantically — and results deep-link to the exact word.
5. Heavy video work is offloaded to an **in-process `asyncio` background task** so the index request returns quickly.

### Retrieval

The shipped search path is **hybrid**: four retrieval legs are fused with **Reciprocal Rank Fusion (RRF)**.

- **Filename leg** — Postgres `pg_trgm` trigram similarity on filenames.
- **Visual leg** — the query text is embedded with **Gemini Embedding 2** and compared (cosine) against image and video-frame embeddings via **pgvector**. Because Gemini maps text and images into a *shared* vector space, a text query can be matched directly against pixels.
- **Transcript lexical leg** — Postgres full-text search (`websearch_to_tsquery` + `ts_rank`, GIN-indexed) over transcript segments; if a clip *says* the query words, it matches here.
- **Transcript semantic leg** — the same query embedding compared (cosine) against per-segment transcript embeddings, so paraphrased speech still matches.
- **Fusion** — each leg contributes `1 / (k + rank)` per file (k = 60) and the sums are re-ranked, so a strong rank on any single leg (e.g. an exact spoken phrase) surfaces the clip near the top. Video matches carry the matched frame and/or transcript segment with its timestamp for deep-linking.

---

## Key Engineering Decisions & Tradeoffs

**Gemini Embedding 2 for cross-modal retrieval.** Gemini Embedding 2 maps images and text into one shared space, so a text query can be compared directly to image embeddings via similarity search, with no captioning or object-detection step in between.

**Hybrid retrieval with RRF.** The shipped search fuses lexical filename matching (`pg_trgm`), Gemini visual search, and lexical + semantic transcript search using Reciprocal Rank Fusion. RRF combines legs by rank rather than raw score, so heterogeneous signals (trigram similarity, `ts_rank`, cosine similarity) need no score calibration to be fused fairly.

**Local WhisperX for transcription.** Audio is transcribed with **WhisperX** (Whisper *tiny* ASR + wav2vec2 forced alignment, CPU/int8) during indexing — no extra API cost, and clips ≤30 s transcribe in seconds. Word-level timestamps let transcript hits deep-link to the exact moment a searched word is spoken, not just the segment it appears in.

**Postgres + pgvector instead of a dedicated vector DB.** One datastore holds metadata, lexical indexes (`pg_trgm` GIN), and vector indexes (HNSW, cosine). This removes a moving part, keeps writes transactional, and stays portable to any managed Postgres (Supabase, Neon, RDS).

**Hosted embedding endpoint instead of local model weights.** Gemini Embedding 2 is called over HTTP via Google AI Studio (768-dim). This keeps the app container small and CPU-only and lets the model deployment scale independently. The tradeoff is per-call latency and a dependency on endpoint availability, which the indexing layer absorbs by running asynchronously.

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
| Visual embeddings | Gemini Embedding 2 (Google AI Studio) | 768-dim multimodal, shared space |
| Transcription | WhisperX (Whisper tiny + wav2vec2 alignment) | Word-level timestamps, local CPU/int8 inference |
| Video | ffmpeg / ffprobe (subprocess) | Frame sampling, audio extraction + timestamps |
| Frame storage | Azure Blob / Supabase | SAS-signed frame image URLs |
| Async | `asyncio` background tasks | Service Bus producer path available |
| Observability | Langfuse | Traces embedding + retrieval calls |
| Migrations | Alembic | `vector` + `pg_trgm` extensions, table DDL |

---

## Scope and Limitations

- Transcription uses WhisperX with the Whisper *tiny* model — fast and free, but ASR accuracy is below larger Whisper sizes (set `WHISPER_MODEL_SIZE` to trade speed for accuracy). Word-level alignment requires a wav2vec2 model for the detected language; unsupported languages fall back to segment-level timestamps.
- Video frame indexing and transcription require `ffmpeg` on the host; frames also need configured embedding/blob credentials.
- `indexing_status` reflects enqueue, not end-to-end per-file completion.
- Stripe and usage-limit fields exist in config and schema but are not enforced.

---

## Running Locally

### Prerequisites

- Python 3.11+
- PostgreSQL 16 with the `vector` and `pg_trgm` extensions (or use the included `docker-compose.yml`)
- `ffmpeg` / `ffprobe` on your `PATH` (required for video frame indexing)
- Google Cloud OAuth credentials (Drive API enabled)
- Google AI Studio Gemini Embedding 2 API key

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

GEMINI_API_KEY=...                       # https://aistudio.google.com/apikey
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
GEMINI_EMBEDDING_DIMENSION=768

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
│   │   ├── vision_embedding.py     # Gemini Embedding 2 image/text embeddings (768-d)
│   │   ├── video_frame_indexing.py # ffmpeg frame extraction + embedding
│   │   ├── transcription.py        # WhisperX transcription (segments + word timestamps)
│   │   ├── indexing.py             # search: trigram + vector + transcript legs, RRF fusion
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

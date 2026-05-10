# ClipFinder-MVP

<p align="left">
  <strong>Built for creators who are tired of scrubbing through footage.</strong>
</p>

> **Semantic video clip search for creators** — find that 3-second shot of a "red scooter at night in the rain" without scrubbing through hundreds of files.


[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/your-org/clipfinder-mvp)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-gray.svg)](LICENSE)

---

## 🎯 Problem

YouTube, TikTok, and small-studio creators store tens of thousands of video clips in Google Drive. Google Drive search is **title-only**, so finding a specific shot means manually scrubbing through hundreds of files. ClipFinder indexes your clips with AI-powered visual and transcript search.

## ✨ MVP Features

| # | Feature | Est. Time |
|---|---------|-----------|
| 1 | Google OAuth read-only Drive scope | 1 h |
| 2 | Folder picker + file-list scan | 3 h |
| 3 | Hard validator: reject >30 s or >100 MB per file | 2 h |
| 4 | FFmpeg-wasm 1 fps key-frame extractor (browser/Lambda) | 1 d |
| 5 | CLIP ViT-B/32 512-dim embedding generator | 0.5 d |
| 6 | pgvector table + insert API | 0.5 d |
| 7 | Hybrid search endpoint: dense vector + BM25 filename | 1 d |
| 8 | Web search UI: input box + ranked grid of key-frames | 1 d |
| 9 | Click → 30 s scrub-able preview + "Open in Drive" link | 0.5 d |
| 10 | Whisper-tiny transcript → sentence embedding → text search tab | 1 d |
| 11 | Usage counter: 50 searches, then soft $5 pay-wall (email only) | 0.5 d |
| 12 | Progress bar + error badge for corrupt/too-long clips | 0.5 d |
| 13 | Basic analytics event stream (search, click, pay-wall view) | 0.5 d |
| 14 | Landing page + wait-list gate before OAuth | 0.5 d |

**Total estimated build time: ~3 weeks**

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Browser       │     │   FastAPI        │     │   Postgres 15   │
│                 │     │                  │     │   + pgvector    │
│  - Google OAuth │────▶│  - /api/embed    │────▶│                 │
│  - FFmpeg-wasm  │     │  - /api/search   │     │  - embeddings   │
│  - File picker  │◀────│  - /api/index    │◀────│  - metadata     │
│  - Search UI    │     │                  │     │  - BM25 index   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │   Celery + Redis │
                        │   (async jobs)   │
                        │                  │
                        │  - CLIP embed    │
                        │  - Whisper STT   │
                        └──────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| **Backend** | FastAPI (Python 3.11+) | Async, auto OpenAPI docs |
| **Database** | PostgreSQL 15 + pgvector | Portable to Supabase, Neon, RDS |
| **Auth** | fastapi-users + Google OAuth2 | JWT cookies + bearer tokens |
| **Queue** | Celery + Redis | Offload heavy GPU/CPU jobs |
| **ML - Vision** | CLIP ViT-B/32 (open-clip) | 512-dim embeddings |
| **ML - Audio** | Whisper-tiny | Transcription → sentence embeddings |
| **Video** | FFmpeg-wasm | Browser-side 1 fps frame extraction |
| **Analytics** | PostHog | Event tracking |
| **Payments** | Stripe (test mode) | Email capture only for MVP |

---

## 📦 Installation

### Prerequisites

- Python 3.11+
- PostgreSQL 15 with pgvector extension
- Redis (for Celery queue, optional for MVP)

### 1. Clone & Setup

```bash
git clone https://github.com/your-org/clipfinder-mvp.git
cd clipfinder-mvp

# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Start PostgreSQL and enable pgvector
psql -U postgres -c "CREATE DATABASE clipfinder;"
psql -U postgres -d clipfinder -c "CREATE EXTENSION vector;"
```

### 3. Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Create a new project or select existing
3. Enable the **Google Drive API**
4. Create **OAuth 2.0 Client ID** (Web application)
5. Add authorized redirect URI: `http://localhost:8000/auth/google/callback`
6. Copy the Client ID and Client Secret

### 4. Environment Variables

Copy the template and fill in your values:

```bash
cp env.template .env
```

Edit `.env`:

```env
# Database (use asyncpg driver)
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/clipfinder

# JWT Secret - generate with: openssl rand -hex 32
JWT_SECRET=your-random-256-bit-secret-here

# Google OAuth (from step 3)
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret

# App URLs
APP_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
```

### 5. Run Migrations

```bash
# Generate initial migration
alembic revision --autogenerate -m "initial"

# Apply migrations
alembic upgrade head
```

### 6. Start the Server

```bash
# Option 1: Using the run script
python run.py

# Option 2: Using uvicorn directly
uvicorn app.main:app --reload --port 8000
```

### 7. Access the App

- **Landing Page:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### Optional: ffmpeg (for video search by timestamp)

Video frame indexing (search by moment inside videos) uses **ffmpeg** to extract frames. If ffmpeg is not installed or not on your PATH, frame indexing will fail and the `video_frame_embeddings` table will stay empty.

- **Windows:** Install [ffmpeg](https://ffmpeg.org/download.html) (e.g. via [winget](https://winget.run/pkg/Gyan/FFmpeg) or [chocolatey](https://chocolatey.org/packages/ffmpeg)) and add the `bin` folder to your system PATH.
- **macOS:** `brew install ffmpeg`
- **Linux:** `apt install ffmpeg` or your distro’s package manager.

### Optional: Celery Worker (for ML jobs)

```bash
# In a separate terminal
celery -A app.worker worker --loglevel=info
```

---

## 🔌 API Reference

### `POST /api/embed`

Index a new video clip.

**Request Body:**
```json
{
  "fileId": "google-drive-file-id",
  "userId": "uuid",
  "frames": ["base64-jpeg-1", "base64-jpeg-2", "..."],
  "audioBase64": "base64-aac-audio",
  "filename": "red_scooter.mp4"
}
```

**Response:** `201 Created`
```json
{
  "clipId": "uuid"
}
```

### `GET /api/search`

Search indexed clips.

**Query Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ✅ | Search query |
| `mode` | enum | ✅ | `visual` or `transcript` |
| `userId` | uuid | ✅ | User identifier |

**Response:** `200 OK`
```json
{
  "results": [
    {
      "clipId": "uuid",
      "fileId": "google-drive-file-id",
      "driveUrl": "https://drive.google.com/file/d/...",
      "keyFrameUrl": "https://storage.example.com/frame.jpg",
      "timeOffset": 12.5,
      "score": 0.847
    }
  ]
}
```

### `DELETE /api/index`

Remove all embeddings for a user.

**Query Parameters:**
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| `userId` | uuid | ✅ | User identifier |

**Response:** `204 No Content`

---

## 🚀 User Flow

```
1. Landing Page → CTA "Start Indexing"
          ↓
2. Google OAuth (read-only Drive scope)
          ↓
3. Google Picker → Select ONE folder
          ↓
4. File validation (reject >30s or >100MB)
          ↓
5. Click "Index N clips" → Progress bar
          ↓
6. Processing per clip:
   ├─ FFmpeg → 1 fps frames → CLIP embedding
   ├─ Whisper → transcript → sentence embedding
   └─ Filename → BM25 index
          ↓
7. Auto-redirect to /search
          ↓
8. Search: Visual tab | Transcript tab
          ↓
9. Results: Grid of key-frames ranked by score
          ↓
10. Click → 30s preview modal + "Open in Drive"
          ↓
11. After 50 searches → $5/mo paywall (email capture)
```

---

## 📊 Limits

| Constraint | Limit |
|------------|-------|
| Clips per folder | 100 |
| Duration per clip | 30 seconds |
| File size per clip | 100 MB |
| Storage per user | 500 MB |
| Free searches | 50 |

---

## 🔒 Security & Privacy

- **OAuth Scope:** `drive.readonly` only — we never modify your files
- **No Raw Storage:** Frames discarded after embedding; audio discarded after transcript
- **Anonymized Data:** Embeddings are mathematical vectors with no personal data
- **Randomized IDs:** Clip IDs are UUID v4, no reversible mapping to filenames
- **GDPR Compliant:** No personal data stored in database
- **DMCA Safe:** We're a search index only; users retain full ownership

---

## 📈 Analytics Events

| Event | Description |
|-------|-------------|
| `index_start` | User begins indexing a folder |
| `index_complete` | Indexing finished successfully |
| `search_query` | User performs a search (includes mode) |
| `result_click` | User clicks a search result (includes position) |
| `paywall_view` | User sees the paywall |
| `paywall_click` | User clicks through paywall |
| `index_delete` | User deletes their index |

---

## ✅ Definition of Done (MVP)

- [ ] User completes full pipeline (OAuth → search results) in ≤2 min on 50-clip folder
- [ ] p95 search latency <1 s (cold) on 100-clip index
- [ ] ≥40% of activated users run ≥3 searches within 24 h
- [ ] ≥10% click fake $5 pay-wall
- [ ] No P1 bugs: auth loop, 0-byte preview, broken Drive deep-link

---

## 🚫 Out of Scope (Post-MVP)

These features are explicitly **not** part of the MVP:

- Multi-language UI
- Face recognition or OCR
- Duplicate detection
- Premiere / Final Cut Pro plug-in
- Mobile app
- Upload from local disk (Drive only)

---

## 🤝 Contributing

This is an MVP under active development. Please open an issue before submitting PRs.

---

## 📄 License

MIT © 2025 ClipFinder

---


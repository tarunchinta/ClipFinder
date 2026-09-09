"""
Sync Distill-indexed videos from eval_corpus.json into a TwelveLabs index.

Idempotent: videos already recorded as status=ready in eval_tl_sync_state.json
are skipped. Append drive_file_id entries to the corpus manifest and re-run to
index only the delta.

Prerequisites:
  1. Videos already indexed in Distill with blob_video_url set
  2. TWELVELABS_API_KEY set
  3. A TwelveLabs index id in eval_tl_sync_state.json
     (run create_twelvelabs_index.py first, or pass --create-index)

Usage:
    cd backend
    python sync_twelvelabs_index.py --user you@example.com
    python sync_twelvelabs_index.py --user you@example.com --create-index
    python sync_twelvelabs_index.py --user you@example.com --force instagram:DX5ktNIOwx7

Does not delete videos removed from the manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).parent))

from app.database import async_session_maker
from app.models.indexed_file import IndexedFile, IndexingStatus
from app.models.user import User
from app.services.video_frame_indexing import get_blob_url_with_sas
from create_twelvelabs_index import (
    create_index,
    get_api_key,
    load_state,
    save_state,
    utc_now_iso,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = BASE_DIR / "eval_corpus.json"
DEFAULT_STATE = BASE_DIR / "eval_tl_sync_state.json"
POLL_SECONDS = 5


# ---------------------------------------------------------------------------
# Manifest / Distill resolution
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    videos = raw.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError(f"{path} has no videos[] entries")
    for i, item in enumerate(videos):
        if not isinstance(item, dict) or not str(item.get("drive_file_id", "")).strip():
            raise ValueError(f"{path} videos[{i}] missing drive_file_id")
    return raw


async def resolve_user(session, identifier: str) -> User:
    try:
        stmt = select(User).where(User.id == UUID(identifier))
    except ValueError:
        stmt = select(User).where(User.email == identifier)
    user = (await session.execute(stmt)).unique().scalar_one_or_none()
    if user is None:
        raise SystemExit(f"No user matches {identifier!r}. Pass an email or a user UUID.")
    return user


async def resolve_distill_video(
    session,
    user_id: UUID,
    drive_file_id: str,
) -> tuple[Optional[IndexedFile], Optional[str]]:
    """
    Look up the Distill row. Returns (row, error_message).
    error_message is set when the row is missing or not uploadable to TL.
    """
    row = (
        await session.execute(
            select(IndexedFile).where(
                IndexedFile.user_id == user_id,
                IndexedFile.drive_file_id == drive_file_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        return None, "not found in Distill for this user"
    if row.file_type != "video":
        return row, f"file_type={row.file_type!r}, expected video"
    if row.indexing_status != IndexingStatus.COMPLETED.value:
        return row, f"indexing_status={row.indexing_status!r}, expected completed"
    if not row.blob_video_url:
        return row, "blob_video_url is empty (TwelveLabs needs a direct media URL)"
    return row, None


# ---------------------------------------------------------------------------
# TwelveLabs upload + index
# ---------------------------------------------------------------------------


def get_tl_client(api_key: str):
    try:
        from twelvelabs import TwelveLabs
    except ImportError as exc:
        raise SystemExit(
            "The twelvelabs package is required. Install with: pip install twelvelabs"
        ) from exc
    return TwelveLabs(api_key=api_key)


def wait_asset_ready(client, asset_id: str) -> Any:
    while True:
        asset = client.assets.retrieve(asset_id)
        status = getattr(asset, "status", None)
        print(f"    asset status={status}")
        if status == "ready":
            return asset
        if status == "failed":
            raise RuntimeError(f"TwelveLabs asset processing failed: id={asset_id}")
        time.sleep(POLL_SECONDS)


def wait_indexed_ready(client, index_id: str, indexed_asset_id: str) -> Any:
    while True:
        indexed = client.indexes.indexed_assets.retrieve(
            index_id=index_id,
            indexed_asset_id=indexed_asset_id,
        )
        status = getattr(indexed, "status", None)
        print(f"    indexed_asset status={status}")
        if status == "ready":
            return indexed
        if status == "failed":
            raise RuntimeError(
                f"TwelveLabs indexing failed: indexed_asset_id={indexed_asset_id}"
            )
        time.sleep(POLL_SECONDS)


def index_video_to_twelvelabs(
    client,
    index_id: str,
    media_url: str,
    *,
    drive_file_id: str,
    distill_filename: str,
) -> dict[str, Any]:
    """
    Upload via URL, wait for asset ready, index into the TL index, wait ready.
    Returns ids for the sync-state record.
    """
    print("    creating asset from SAS URL...")
    asset = client.assets.create(method="url", url=media_url)
    asset_id = asset.id
    print(f"    asset_id={asset_id}")
    wait_asset_ready(client, asset_id)

    print("    creating indexed_asset...")
    indexed = client.indexes.indexed_assets.create(
        index_id=index_id,
        asset_id=asset_id,
        enable_video_stream=True,  # HLS + thumbnail_urls on retrieve when addon enabled
        user_metadata={
            "drive_file_id": drive_file_id,
            "distill_filename": distill_filename,
        },
    )
    indexed_asset_id = indexed.id
    print(f"    indexed_asset_id={indexed_asset_id}")
    ready = wait_indexed_ready(client, index_id, indexed_asset_id)

    # Search results expose video_id; on the asset-based API this is the
    # indexed asset id. Fall back to .id if a dedicated field appears later.
    video_id = getattr(ready, "video_id", None) or ready.id

    return {
        "tl_asset_id": asset_id,
        "tl_indexed_asset_id": indexed_asset_id,
        "tl_video_id": video_id,
        "indexed_at": getattr(ready, "indexed_at", None) or utc_now_iso(),
    }


def mark_entry(
    state: dict,
    drive_file_id: str,
    *,
    status: str,
    row: Optional[IndexedFile] = None,
    tl: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    entry: dict[str, Any] = dict(state.get("videos", {}).get(drive_file_id) or {})
    entry["status"] = status
    entry["last_error"] = error
    if row is not None:
        entry["distill_file_id"] = str(row.id)
        entry["filename"] = row.filename
        entry["duration_seconds"] = row.duration_seconds
        entry["blob_video_url"] = row.blob_video_url
    if tl:
        entry.update(tl)
    if status == "ready":
        entry["last_error"] = None
    state.setdefault("videos", {})[drive_file_id] = entry


def should_skip(entry: Optional[dict], *, force: bool) -> bool:
    """Skip only ready entries unless --force. Failed/blocked/processing are retried."""
    if force or not entry:
        return False
    return entry.get("status") == "ready"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--user",
        required=True,
        help="Distill account that owns the indexed files: email or user UUID.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Corpus manifest JSON (default: {DEFAULT_MANIFEST.name}).",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE,
        help=f"Sync state JSON (default: {DEFAULT_STATE.name}).",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="If state has no twelvelabs_index_id, create one (full model_options + addons).",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help="Name used when --create-index runs. Defaults to distill-benchmark-YYYYMMDD.",
    )
    parser.add_argument(
        "--force",
        action="append",
        default=[],
        metavar="DRIVE_FILE_ID",
        help="Re-index this drive_file_id even if status=ready. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve Distill rows and print the plan without calling TwelveLabs.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest = load_manifest(args.manifest)
    state = load_state(args.state)
    force_ids = set(args.force)

    index_id = state.get("twelvelabs_index_id")
    if not index_id:
        if not args.create_index and not args.dry_run:
            print(
                "No twelvelabs_index_id in state. Run create_twelvelabs_index.py "
                "or pass --create-index.",
                file=sys.stderr,
            )
            return 2
        if args.create_index and not args.dry_run:
            api_key = get_api_key()
            index_name = (
                args.index_name
                or f"distill-benchmark-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
            )
            print(f"Creating TwelveLabs index {index_name!r}...")
            index = create_index(api_key, index_name)
            state["twelvelabs_index_id"] = index.id
            state["twelvelabs_index_name"] = index_name
            save_state(args.state, state)
            index_id = index.id
            print(f"Created index id={index_id}")

    client = None
    if not args.dry_run:
        client = get_tl_client(get_api_key())

    summary = {"ready_skipped": 0, "indexed": 0, "failed": 0, "blocked": 0}

    async with async_session_maker() as session:
        user = await resolve_user(session, args.user)
        print(
            f"Syncing {len(manifest['videos'])} corpus video(s) as {user.email} "
            f"-> TL index {index_id or '(dry-run / unset)'}"
        )
        if manifest.get("description"):
            print(f"Corpus: {manifest['description']}")

        for item in manifest["videos"]:
            drive_file_id = item["drive_file_id"].strip()
            notes = item.get("notes") or ""
            existing = state.get("videos", {}).get(drive_file_id)
            force = drive_file_id in force_ids

            print(f"\n[{drive_file_id}] {notes}".rstrip())

            if should_skip(existing, force=force):
                print(f"  skip - already status={existing.get('status')}")
                summary["ready_skipped"] += 1
                continue

            row, err = await resolve_distill_video(session, user.id, drive_file_id)
            if err:
                print(f"  blocked - {err}")
                mark_entry(state, drive_file_id, status="blocked", row=row, error=err)
                save_state(args.state, state)
                summary["blocked"] += 1
                continue

            assert row is not None
            media_url = get_blob_url_with_sas(row.blob_video_url)
            if not media_url:
                err = "failed to build SAS media URL"
                print(f"  blocked - {err}")
                mark_entry(state, drive_file_id, status="blocked", row=row, error=err)
                save_state(args.state, state)
                summary["blocked"] += 1
                continue

            if args.dry_run:
                print(f"  dry-run - would upload distill_file_id={row.id} filename={row.filename!r}")
                continue

            mark_entry(state, drive_file_id, status="processing", row=row)
            save_state(args.state, state)

            try:
                assert client is not None and index_id
                tl_ids = index_video_to_twelvelabs(
                    client,
                    index_id,
                    media_url,
                    drive_file_id=drive_file_id,
                    distill_filename=row.filename,
                )
                mark_entry(state, drive_file_id, status="ready", row=row, tl=tl_ids)
                save_state(args.state, state)
                print(f"  ready - tl_video_id={tl_ids['tl_video_id']}")
                summary["indexed"] += 1
            except Exception as exc:
                mark_entry(
                    state,
                    drive_file_id,
                    status="failed",
                    row=row,
                    error=str(exc),
                )
                save_state(args.state, state)
                print(f"  failed - {exc}")
                summary["failed"] += 1

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"State: {args.state}")
    return 1 if summary["failed"] or summary["blocked"] else 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())

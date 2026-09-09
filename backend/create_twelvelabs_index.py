"""
Create a TwelveLabs index for Distill benchmark comparisons.

Enables every Marengo model option and every supported addon:
  - model_options: visual, audio
  - addons: thumbnail

Writes the resulting index id into eval_tl_sync_state.json so
sync_twelvelabs_index.py can reuse it.

Usage:
    cd backend
    python create_twelvelabs_index.py
    python create_twelvelabs_index.py --name distill-benchmark-v1
    python create_twelvelabs_index.py --force   # replace index id in state

Requires TWELVELABS_API_KEY or TWELVE_LABS_API_KEY in the environment or backend/.env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATE = BASE_DIR / "eval_tl_sync_state.json"

# Marengo index config — full modality + addon surface currently documented.
MARENGO_MODEL_NAME = "marengo3.0"
MARENGO_MODEL_OPTIONS = ["visual", "audio"]
MARENGO_ADDONS = ["thumbnail"]

load_dotenv(BASE_DIR / ".env", override=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_state(path: Path) -> dict:
    if not path.is_file():
        return {
            "twelvelabs_index_id": None,
            "twelvelabs_index_name": None,
            "updated_at": None,
            "videos": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = utc_now_iso()
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def get_api_key() -> str:
    key = (
        os.environ.get("TWELVELABS_API_KEY", "").strip()
        or os.environ.get("TWELVE_LABS_API_KEY", "").strip()
    )
    if not key:
        raise SystemExit(
            "TWELVELABS_API_KEY (or TWELVE_LABS_API_KEY) is not set. "
            "Add it to the environment or backend/.env."
        )
    return key


def create_index(api_key: str, index_name: str):
    try:
        from twelvelabs import TwelveLabs
    except ImportError as exc:
        raise SystemExit(
            "The twelvelabs package is required. Install with: pip install twelvelabs"
        ) from exc

    client = TwelveLabs(api_key=api_key)
    index = client.indexes.create(
        index_name=index_name,
        models=[
            {
                "model_name": MARENGO_MODEL_NAME,
                "model_options": list(MARENGO_MODEL_OPTIONS),
            }
        ],
        addons=list(MARENGO_ADDONS),
    )
    if not getattr(index, "id", None):
        raise RuntimeError("TwelveLabs index create returned no id")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--name",
        default=None,
        help="Index name. Defaults to distill-benchmark-YYYYMMDD.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE,
        help=f"Sync state JSON to update (default: {DEFAULT_STATE.name}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create a new index even if state already has twelvelabs_index_id.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state = load_state(args.state)
    existing = state.get("twelvelabs_index_id")
    if existing and not args.force:
        print(
            f"State already has twelvelabs_index_id={existing}. "
            "Pass --force to create another index and overwrite the pointer.",
            file=sys.stderr,
        )
        return 2

    index_name = args.name or f"distill-benchmark-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    api_key = get_api_key()

    print(f"Creating TwelveLabs index {index_name!r}")
    print(f"  model={MARENGO_MODEL_NAME}")
    print(f"  model_options={MARENGO_MODEL_OPTIONS}")
    print(f"  addons={MARENGO_ADDONS}")

    index = create_index(api_key, index_name)
    state["twelvelabs_index_id"] = index.id
    state["twelvelabs_index_name"] = index_name
    state.setdefault("videos", {})
    save_state(args.state, state)

    print(f"Created index id={index.id}")
    print(f"Wrote {args.state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

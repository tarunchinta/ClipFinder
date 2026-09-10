"""Distill vs TwelveLabs search benchmark.

Runnable from backend/ as::

    python -m eval.run --user you@example.com --dry-run
    python -m eval.search --user you@example.com
    python -m eval.create_twelvelabs_index
    python -m eval.sync_twelvelabs_index --user you@example.com

JSON next to these modules is the query set, corpus, and TL sync state.
See eval/README.md for the field mapping and workflow.
"""

from pathlib import Path
import sys

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

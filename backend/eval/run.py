"""
Run search queries against Distill and TwelveLabs in parallel and score both.

Reads three files in this package:
  - tl_sync_state.json  which corpus videos are live in the TL index,
                        and the drive_file_id <-> tl_video_id mapping
  - corpus.json         the shared corpus manifest
  - queries.json        queries, relevance judgments, and arm configs

An "arm" is one system under one configuration. Every (arm, query) pair is an
independent task and they all run concurrently, so a Distill leg sweep and a
TwelveLabs search_options sweep cost about one query's worth of wall clock.

Results from both systems are normalized to ranked lists of drive_file_ids, so
recall@k is comparing like with like: TL returns clips keyed by video, Distill
returns files, and both collapse to "which corpus videos came back, in what
order".

Usage:
    cd backend
    python -m eval.run --user you@example.com
    python -m eval.run --user you@example.com --arm distill --arm twelvelabs
    python -m eval.run --user you@example.com --query-id q001 --verbose
    python -m eval.run --user you@example.com --out results.json
    python -m eval.run --user you@example.com --dry-run

Requires TWELVELABS_API_KEY (unless every selected arm is Distill-only) and the
same database configuration the app uses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import async_session_maker
from app.services.indexing import HYBRID_SEARCH_LEGS, IndexingService
from eval.create_twelvelabs_index import get_api_key, load_state
from eval.legs import distill_legs_for_query, tl_search_options_for_query, validate_query_tags
from eval.metrics import aggregate, evaluate_query, percentile
from eval.sync_twelvelabs_index import resolve_user

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_STATE = EVAL_DIR / "tl_sync_state.json"
DEFAULT_QUERIES = EVAL_DIR / "queries.json"
DEFAULT_CONCURRENCY = 8


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Arm:
    name: str
    system: str  # "distill" | "twelvelabs"
    limit: int
    distill: dict[str, Any] = field(default_factory=dict)
    twelvelabs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalQuery:
    id: str
    query: str
    relevant: list[str]
    graded: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    expect_color_leg: Optional[bool] = None
    distill: dict[str, Any] = field(default_factory=dict)
    twelvelabs: dict[str, Any] = field(default_factory=dict)


def _merge(*layers: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Shallow-merge config layers left to right; later non-None values win."""
    out: dict[str, Any] = {}
    for layer in layers:
        if layer:
            out.update({k: v for k, v in layer.items() if v is not None})
    return out


def _parse_judgments(
    raw: Any, qid: str
) -> tuple[list[str], dict[str, float]]:
    """
    Parse a query's relevant[] into (binary ids, graded gains).

    Accepts eval.search's schema, [{"drive_file_id": ..., "gain": 3}], which
    is the shared format — both tools read the same judgments file. A bare
    string is also accepted and means gain 1.0.
    """
    relevant: list[str] = []
    graded: dict[str, float] = {}
    for j in raw or []:
        if isinstance(j, str):
            drive_file_id, gain = j, 1.0
        elif isinstance(j, dict):
            drive_file_id = str(j.get("drive_file_id") or "").strip()
            gain = float(j.get("gain", 1.0))
        else:
            raise ValueError(f"query {qid}: relevant[] entries must be objects or strings")
        if not drive_file_id:
            raise ValueError(f"query {qid}: a relevant[] entry has no drive_file_id")
        relevant.append(drive_file_id)
        graded[drive_file_id] = gain
    return relevant, graded


def load_queries(path: Path) -> tuple[list[Arm], list[EvalQuery], list[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults") or {}
    default_k = int(raw.get("default_k") or 10)
    k_values = sorted(set(raw.get("k_values") or [1, 3, 5, default_k]))
    if any(k <= 0 for k in k_values):
        raise ValueError("k_values must all be positive")

    arms: list[Arm] = []
    for i, item in enumerate(raw.get("arms") or []):
        name = str(item.get("name") or "").strip()
        system = str(item.get("system") or "").strip()
        if not name:
            raise ValueError(f"arms[{i}] missing name")
        if system not in ("distill", "twelvelabs"):
            raise ValueError(f"arm {name!r}: system must be 'distill' or 'twelvelabs'")
        arms.append(
            Arm(
                name=name,
                system=system,
                limit=int(item.get("limit") or defaults.get("limit") or 10),
                distill=_merge(defaults.get("distill"), item.get("distill")),
                twelvelabs=_merge(defaults.get("twelvelabs"), item.get("twelvelabs")),
            )
        )
    if not arms:
        raise ValueError(f"{path} defines no arms[]")
    dupes = {a.name for a in arms if [x.name for x in arms].count(a.name) > 1}
    if dupes:
        raise ValueError(f"duplicate arm name(s): {sorted(dupes)}")

    queries: list[EvalQuery] = []
    for i, item in enumerate(raw.get("queries") or []):
        qid = str(item.get("id") or f"q{i:03d}")
        text = str(item.get("query") or "").strip()
        if not text:
            raise ValueError(f"queries[{i}] ({qid}) has an empty query string")
        relevant, graded = _parse_judgments(item.get("relevant"), qid)
        expect = item.get("expect_color_leg")
        if expect is not None:
            expect = bool(expect)
        tags = validate_query_tags(item.get("tags") or [], query_id=qid)
        queries.append(
            EvalQuery(
                id=qid,
                query=text,
                relevant=relevant,
                graded=graded,
                tags=tags,
                expect_color_leg=expect,
                distill=item.get("distill") or {},
                twelvelabs=item.get("twelvelabs") or {},
            )
        )
    if not queries:
        raise ValueError(f"{path} defines no queries[]")
    return arms, queries, k_values


# ---------------------------------------------------------------------------
# Corpus mapping
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    index_id: Optional[str]
    ready_drive_ids: set[str]
    tl_video_to_drive: dict[str, str]
    distill_file_to_drive: dict[str, str]


def build_corpus(state: dict) -> Corpus:
    """
    Map both systems' result ids back to drive_file_id, using only videos the
    sync marked ready — anything else is not searchable on the TL side.
    """
    ready: set[str] = set()
    tl_map: dict[str, str] = {}
    distill_map: dict[str, str] = {}
    for drive_file_id, entry in (state.get("videos") or {}).items():
        if (entry or {}).get("status") != "ready":
            continue
        ready.add(drive_file_id)
        for key in ("tl_video_id", "tl_indexed_asset_id"):
            value = entry.get(key)
            if value:
                tl_map[str(value)] = drive_file_id
        if entry.get("distill_file_id"):
            distill_map[str(entry["distill_file_id"])] = drive_file_id
    return Corpus(
        index_id=state.get("twelvelabs_index_id"),
        ready_drive_ids=ready,
        tl_video_to_drive=tl_map,
        distill_file_to_drive=distill_map,
    )


# ---------------------------------------------------------------------------
# Search execution
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    arm: str
    system: str
    query_id: str
    ranked: list[str]
    raw_count: int
    latency_ms: float
    error: Optional[str] = None
    detail: list[dict[str, Any]] = field(default_factory=list)


def _distill_cfg(arm: Arm, q: EvalQuery) -> dict[str, Any]:
    """Arm defaults, then tag-derived Distill legs, then per-query distill{}."""
    cfg = _merge(arm.distill)
    derived = distill_legs_for_query(q.tags, q.expect_color_leg, query_id=q.id)
    if derived is not None:
        cfg = _merge(cfg, {"legs": derived})
    return _merge(cfg, q.distill)


def _tl_cfg(arm: Arm, q: EvalQuery) -> dict[str, Any]:
    """Arm defaults, then tag-derived TL search_options, then per-query twelvelabs{}."""
    cfg = _merge(arm.twelvelabs)
    derived = tl_search_options_for_query(q.tags, q.expect_color_leg, query_id=q.id)
    if derived is not None:
        cfg = _merge(cfg, {"search_options": derived})
    return _merge(cfg, q.twelvelabs)


def _tl_search_options(cfg: dict[str, Any]) -> list[str]:
    """Preserve an explicit empty list; default to visual only when the key is absent."""
    options = cfg.get("search_options")
    if options is None:
        return ["visual"]
    return list(options)


async def run_distill(
    arm: Arm,
    q: EvalQuery,
    user_id: UUID,
    corpus: Corpus,
) -> RunResult:
    """One Distill hybrid search on its own session (sessions are not concurrency-safe)."""
    cfg = _distill_cfg(arm, q)
    legs = cfg.get("legs")
    started = time.perf_counter()
    try:
        async with async_session_maker() as session:
            service = IndexingService(session)
            results = await service.hybrid_search_rrf(
                user_id=user_id,
                query=q.query,
                file_type=cfg.get("file_type"),
                limit=arm.limit,
                legs=legs,
            )
    except Exception as exc:
        return RunResult(
            arm=arm.name,
            system=arm.system,
            query_id=q.id,
            ranked=[],
            raw_count=0,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
    latency_ms = (time.perf_counter() - started) * 1000

    ranked: list[str] = []
    detail: list[dict[str, Any]] = []
    for r in results:
        file_row = r["file"]
        drive_file_id = file_row.drive_file_id or corpus.distill_file_to_drive.get(
            str(file_row.id)
        )
        if not drive_file_id:
            continue
        ranked.append(drive_file_id)
        detail.append(
            {
                "drive_file_id": drive_file_id,
                "filename": file_row.filename,
                "hybrid_score": round(float(r["hybrid_score"]), 6),
                "legs": {
                    leg: round(float(r[f"{leg}_score"]), 6)
                    for leg in HYBRID_SEARCH_LEGS
                    if r.get(f"{leg}_score")
                },
            }
        )
    return RunResult(
        arm=arm.name,
        system=arm.system,
        query_id=q.id,
        ranked=ranked,
        raw_count=len(results),
        latency_ms=latency_ms,
        detail=detail,
    )


async def run_twelvelabs(
    arm: Arm,
    q: EvalQuery,
    client: Any,
    corpus: Corpus,
) -> RunResult:
    """One TwelveLabs search, paging until we have arm.limit distinct videos."""
    cfg = _tl_cfg(arm, q)
    started = time.perf_counter()
    ranked: list[str] = []
    detail: list[dict[str, Any]] = []
    raw_count = 0
    try:
        kwargs: dict[str, Any] = {
            "index_id": corpus.index_id,
            "search_options": _tl_search_options(cfg),
            "query_text": q.query,
            "group_by": cfg.get("group_by") or "video",
            "operator": cfg.get("operator") or "or",
            "page_limit": min(max(arm.limit, 1), 50),
            "include_user_metadata": True,
        }
        if cfg.get("filter"):
            kwargs["filter"] = cfg["filter"]
        if cfg.get("transcription_options"):
            kwargs["transcription_options"] = list(cfg["transcription_options"])

        pager = await client.search.query(**kwargs)
        seen: set[str] = set()
        async for item in pager:
            raw_count += 1
            drive_file_id = _tl_item_drive_id(item, corpus)
            if not drive_file_id or drive_file_id in seen:
                continue
            seen.add(drive_file_id)
            ranked.append(drive_file_id)
            detail.append(
                {
                    "drive_file_id": drive_file_id,
                    "tl_video_id": getattr(item, "video_id", None) or getattr(item, "id", None),
                    "rank": getattr(item, "rank", None),
                    "start": getattr(item, "start", None),
                    "end": getattr(item, "end", None),
                    "thumbnail_url": _tl_item_thumbnail_url(item),
                }
            )
            if len(ranked) >= arm.limit:
                break
    except Exception as exc:
        return RunResult(
            arm=arm.name,
            system=arm.system,
            query_id=q.id,
            ranked=ranked,
            raw_count=raw_count,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=f"{type(exc).__name__}: {exc}",
        )
    return RunResult(
        arm=arm.name,
        system=arm.system,
        query_id=q.id,
        ranked=ranked,
        raw_count=raw_count,
        latency_ms=(time.perf_counter() - started) * 1000,
        detail=detail,
    )


def _tl_item_drive_id(item: Any, corpus: Corpus) -> Optional[str]:
    """
    Resolve a TL hit to a drive_file_id.

    Prefers the user_metadata written at index time, which is authoritative,
    and falls back to the id map from the sync state.
    """
    meta = getattr(item, "user_metadata", None)
    if isinstance(meta, dict):
        candidate = meta.get("drive_file_id")
        if candidate:
            return str(candidate)
    for key in ("video_id", "id"):
        value = getattr(item, key, None)
        if value and str(value) in corpus.tl_video_to_drive:
            return corpus.tl_video_to_drive[str(value)]
    return None


def _tl_item_thumbnail_url(item: Any) -> Optional[str]:
    """
    Thumbnail URL from a search hit when the index has the thumbnail addon.

    group_by=video may nest clips; take the first URL we find.
    """
    direct = getattr(item, "thumbnail_url", None)
    if direct:
        return str(direct)
    clips = getattr(item, "clips", None) or getattr(item, "clips_data", None) or []
    for clip in clips:
        url = getattr(clip, "thumbnail_url", None)
        if url:
            return str(url)
        if isinstance(clip, dict) and clip.get("thumbnail_url"):
            return str(clip["thumbnail_url"])
    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    lines = [
        "  ".join(c.ljust(widths[c]) for c in columns),
        "  ".join("-" * widths[c] for c in columns),
    ]
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns))
    return "\n".join(lines)


def fmt(value: Optional[float], places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--user", required=True, help="Distill account: email or user UUID.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="Only run this arm. Repeatable. Default: every arm in the file.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        default=[],
        help="Only run this query id. Repeatable.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Only run queries whose tags[] include this Distill-leg name.",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--out", type=Path, default=None, help="Write full results JSON here.")
    parser.add_argument("--verbose", action="store_true", help="Print per-query rankings.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, corpus and judgments; run no searches.",
    )
    return parser.parse_args()


def select(arms: list[Arm], queries: list[EvalQuery], args) -> tuple[list[Arm], list[EvalQuery]]:
    if args.arm:
        wanted = set(args.arm)
        unknown = wanted - {a.name for a in arms}
        if unknown:
            raise SystemExit(f"Unknown --arm value(s): {sorted(unknown)}")
        arms = [a for a in arms if a.name in wanted]
    if args.query_id:
        wanted_q = set(args.query_id)
        unknown_q = wanted_q - {q.id for q in queries}
        if unknown_q:
            raise SystemExit(f"Unknown --query-id value(s): {sorted(unknown_q)}")
        queries = [q for q in queries if q.id in wanted_q]
    if args.tag:
        tags = set(args.tag)
        queries = [q for q in queries if tags & set(q.tags)]
        if not queries:
            raise SystemExit(f"No queries carry any of the tags {sorted(tags)}")
    return arms, queries


def validate(arms: list[Arm], queries: list[EvalQuery], corpus: Corpus) -> list[str]:
    """Return human-readable warnings; hard errors raise instead."""
    warnings: list[str] = []
    for arm in arms:
        if arm.system == "distill":
            for q in queries:
                legs = _distill_cfg(arm, q).get("legs")
                if legs is not None:
                    unknown = set(legs) - set(HYBRID_SEARCH_LEGS)
                    if unknown:
                        raise SystemExit(
                            f"arm {arm.name!r} query {q.id}: unknown leg(s) "
                            f"{sorted(unknown)}. Valid: {list(HYBRID_SEARCH_LEGS)}"
                        )
        elif not corpus.index_id:
            raise SystemExit(
                f"arm {arm.name!r} is a TwelveLabs arm but the state file has no "
                "twelvelabs_index_id. Run python -m eval.create_twelvelabs_index first."
            )
        else:
            for q in queries:
                options = _tl_search_options(_tl_cfg(arm, q))
                if not options:
                    raise SystemExit(
                        f"arm {arm.name!r} query {q.id}: tags map to no TwelveLabs "
                        "search_options (text/color have no TL analog). Add a visual "
                        "or speech tag, or set twelvelabs.search_options on the query."
                    )

    judged = {doc for q in queries for doc in q.relevant}
    missing = judged - corpus.ready_drive_ids
    if missing:
        warnings.append(
            f"{len(missing)} judged video(s) are not status=ready in the sync state and "
            f"cannot be returned by TwelveLabs: {sorted(missing)[:5]}"
            f"{' ...' if len(missing) > 5 else ''}. Recall is capped for TL arms until "
            "python -m eval.sync_twelvelabs_index indexes them."
        )
    unjudged = [q.id for q in queries if not q.relevant]
    if unjudged:
        warnings.append(
            f"{len(unjudged)} query/queries have no relevant[] judgments; every metric "
            f"is undefined for them and they are excluded from averages: {unjudged[:5]}"
        )
    if len(corpus.ready_drive_ids) < 10:
        warnings.append(
            f"Only {len(corpus.ready_drive_ids)} video(s) are indexed on the TwelveLabs "
            "side. Ranking metrics on a corpus this small are dominated by noise — "
            "treat results as a smoke test, not a benchmark."
        )
    return warnings


async def check_tl_index(
    client: Any,
    arms: list[Arm],
    queries: list[EvalQuery],
    index_id: str,
) -> None:
    """
    Fail before spending queries if any (arm, query) asks for a modality the
    index lacks — including options derived from tags[], not only arm defaults.

    search_options must be a subset of the index's model_options; the API
    otherwise rejects the request per query.
    """
    index = await client.indexes.retrieve(index_id)
    addons = set()
    raw_addons = getattr(index, "addons", None) or []
    for addon in raw_addons:
        if isinstance(addon, str):
            addons.add(addon)
        else:
            name = getattr(addon, "name", None) or getattr(addon, "id", None)
            if name:
                addons.add(str(name))
    if "thumbnail" not in addons:
        raise SystemExit(
            f"index {index_id} has no thumbnail addon (addons={sorted(addons) or 'none'}). "
            "Thumbnail is enabled at index creation (addons=['thumbnail']), not via "
            "search_options. Create a new index with that addon and re-sync."
        )
    available: set[str] = set()
    for model in getattr(index, "models", None) or []:
        available.update(getattr(model, "model_options", None) or [])
    if not available:
        return
    for arm in arms:
        if arm.system != "twelvelabs":
            continue
        for q in queries:
            requested = set(_tl_search_options(_tl_cfg(arm, q)))
            missing = requested - available
            if missing:
                raise SystemExit(
                    f"arm {arm.name!r} query {q.id} requests search_options "
                    f"{sorted(missing)} but index {index_id} was built with "
                    f"model_options {sorted(available)}. Modalities are fixed at "
                    "index creation — create a new index and re-sync to benchmark them."
                )


async def run(args: argparse.Namespace) -> int:
    if not args.queries.is_file():
        print(f"Queries file not found: {args.queries}", file=sys.stderr)
        return 2
    if not args.state.is_file():
        print(f"State file not found: {args.state}", file=sys.stderr)
        return 2

    arms, queries, k_values = load_queries(args.queries)
    arms, queries = select(arms, queries, args)
    corpus = build_corpus(load_state(args.state))
    warnings = validate(arms, queries, corpus)

    print(f"Arms:    {', '.join(a.name for a in arms)}")
    print(f"Queries: {len(queries)}")
    print(f"Corpus:  {len(corpus.ready_drive_ids)} video(s) ready in TL index {corpus.index_id}")
    print(f"k:       {k_values}")
    for w in warnings:
        print(f"WARNING: {w}")

    needs_tl = any(a.system == "twelvelabs" for a in arms)
    if args.dry_run:
        print("\nDry run - no searches executed.")
        return 0

    client = None
    if needs_tl:
        try:
            from twelvelabs import AsyncTwelveLabs
        except ImportError as exc:
            raise SystemExit(
                "The twelvelabs package is required. Install with: pip install twelvelabs"
            ) from exc
        client = AsyncTwelveLabs(api_key=get_api_key())
        await check_tl_index(client, arms, queries, str(corpus.index_id))

    async with async_session_maker() as session:
        user = await resolve_user(session, args.user)
    user_id = user.id
    print(f"User:    {user.email} ({user_id})\n")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def one(arm: Arm, q: EvalQuery) -> RunResult:
        async with semaphore:
            if arm.system == "distill":
                return await run_distill(arm, q, user_id, corpus)
            return await run_twelvelabs(arm, q, client, corpus)

    tasks = [one(arm, q) for arm in arms for q in queries]
    started = time.perf_counter()
    results: list[RunResult] = await asyncio.gather(*tasks)
    wall_ms = (time.perf_counter() - started) * 1000

    if client is not None and hasattr(client, "close"):
        maybe = client.close()
        if asyncio.iscoroutine(maybe):
            await maybe

    return report(args, arms, queries, k_values, results, corpus, warnings, wall_ms)


def report(
    args: argparse.Namespace,
    arms: list[Arm],
    queries: list[EvalQuery],
    k_values: list[int],
    results: list[RunResult],
    corpus: Corpus,
    warnings: list[str],
    wall_ms: float,
) -> int:
    by_query = {q.id: q for q in queries}
    per_arm: dict[str, list[dict[str, Any]]] = {a.name: [] for a in arms}
    latencies: dict[str, list[float]] = {a.name: [] for a in arms}
    errors: list[RunResult] = []
    scored: list[dict[str, Any]] = []

    for r in results:
        latencies[r.arm].append(r.latency_ms)
        if r.error:
            errors.append(r)
            continue
        q = by_query[r.query_id]
        if not q.relevant:
            continue
        metrics = evaluate_query(r.ranked, q.relevant, k_values, graded=q.graded or None)
        per_arm[r.arm].append(metrics)
        scored.append(
            {
                "arm": r.arm,
                "system": r.system,
                "query_id": r.query_id,
                "query": q.query,
                "ranked": r.ranked,
                "relevant": q.relevant,
                "raw_count": r.raw_count,
                "latency_ms": round(r.latency_ms, 1),
                "metrics": metrics,
                "detail": r.detail,
            }
        )

    if args.verbose:
        print("Per-query rankings")
        for row in sorted(scored, key=lambda x: (x["query_id"], x["arm"])):
            hits = ["*" if d in row["relevant"] else " " for d in row["ranked"]]
            marked = ", ".join(f"{h}{d}" for h, d in zip(hits, row["ranked"])) or "(empty)"
            print(f"  [{row['query_id']}] {row['arm']}: {marked}")
        print()

    summary_rows = []
    for arm in arms:
        agg = aggregate(per_arm[arm.name])
        lat = latencies[arm.name]
        row: dict[str, Any] = {
            "arm": arm.name,
            "system": arm.system,
            "n": len(per_arm[arm.name]),
            "mrr": fmt(agg.get("mrr")),
        }
        for k in k_values:
            row[f"recall@{k}"] = fmt(agg.get(f"recall@{k}"))
        for k in k_values:
            row[f"ndcg@{k}"] = fmt(agg.get(f"ndcg@{k}"))
        row["p50_ms"] = fmt(percentile(lat, 50), 0)
        row["p95_ms"] = fmt(percentile(lat, 95), 0)
        summary_rows.append(row)

    columns = (
        ["arm", "system", "n", "mrr"]
        + [f"recall@{k}" for k in k_values]
        + [f"ndcg@{k}" for k in k_values]
        + ["p50_ms", "p95_ms"]
    )
    print(format_table(summary_rows, columns))
    print(f"\n{len(results)} search(es) in {wall_ms / 1000:.1f}s wall clock.")

    if errors:
        print(f"\n{len(errors)} search(es) failed:")
        for e in errors[:10]:
            print(f"  [{e.query_id}] {e.arm}: {e.error}")

    if args.out:
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "twelvelabs_index_id": corpus.index_id,
            "corpus_size": len(corpus.ready_drive_ids),
            "k_values": k_values,
            "warnings": warnings,
            "arms": [
                {
                    "name": a.name,
                    "system": a.system,
                    "limit": a.limit,
                    "config": a.distill if a.system == "distill" else a.twelvelabs,
                    "summary": aggregate(per_arm[a.name]),
                    "latency_ms": {
                        "p50": percentile(latencies[a.name], 50),
                        "p95": percentile(latencies[a.name], 95),
                    },
                }
                for a in arms
            ],
            "runs": scored,
            "errors": [
                {"arm": e.arm, "query_id": e.query_id, "error": e.error} for e in errors
            ],
        }
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")

    return 1 if errors else 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    sys.exit(main())

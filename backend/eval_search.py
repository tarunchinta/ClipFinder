"""
Search relevance eval for Distill hybrid search.

Runs a fixed set of queries against IndexingService.hybrid_search_rrf for a
dev account with pre-existing indexed files, compares the ranked results to
hand-written relevance judgments, and reports recall/precision/MRR/nDCG plus
per-leg attribution.

Because hybrid_search_rrf fuses six retrieval legs (filename trigram, poster
thumbnail embeddings, video frame embeddings, caption FTS/embeddings,
transcript lexical + semantic, color signature), the aggregate score alone
does not tell you which leg earned a hit. This eval also reports leg coverage:
the share of relevant files each leg retrieved on its own.

Usage:
    cd backend
    python eval_search.py --user dev@example.com --queries eval_queries.json

    # metrics at a different cutoff, deeper candidate list, tuned RRF constant
    python eval_search.py --user dev@example.com --k 5 --limit 100 --rrf-k 30

    # machine-readable report for diffing between runs
    python eval_search.py --user dev@example.com --json-out report.json

Requires the same .env the app uses (DATABASE_URL at minimum). Without Gemini
credentials the two semantic legs return nothing and the eval silently measures
lexical-only search, so the script refuses to run unless you pass
--allow-unconfigured-vision.
"""

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import UUID

# Add the backend directory to the path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import select

from app.database import async_session_maker
from app.models.indexed_file import IndexedFile
from app.models.user import User
from app.services.indexing import IndexingService
from app.services.vision_embedding import get_vision_embedding_service


# ---------------------------------------------------------------------------
# Query set
# ---------------------------------------------------------------------------


@dataclass
class Judgment:
    """One relevant file for a query, keyed by the stable per-user file id."""

    drive_file_id: str
    gain: float = 1.0
    file_id: Optional[UUID] = None  # resolved against the dev account at load


@dataclass
class QueryCase:
    query: str
    relevant: list[Judgment]
    file_type: Optional[str] = None
    notes: str = ""
    # None = do not check. False = color_score must be 0 on every hit
    # (query has no color language). True = the color leg must fire.
    expect_color_leg: Optional[bool] = None


@dataclass
class QuerySet:
    queries: list[QueryCase]
    description: str = ""
    default_k: int = 10


def load_query_set(path: Path) -> QuerySet:
    """
    Load a query set from JSON. Schema:

    {
      "description": "smoke set for the dev account",
      "default_k": 10,
      "queries": [
        {
          "query": "sourdough starter explainer",
          "file_type": "video",
          "notes": "the two baking reels, the first is the better answer",
          "expect_color_leg": false,
          "relevant": [
            {"drive_file_id": "instagram:DAcuKpJyzZq", "gain": 3},
            {"drive_file_id": "instagram:C8xY1zQrLmN"}
          ]
        }
      ]
    }

    "gain" is optional and defaults to 1.0, which makes judgments binary. Use
    graded gains (3 = ideal answer, 1 = acceptable) to make nDCG meaningful;
    recall/precision/MRR treat any listed file as relevant either way.

    "expect_color_leg" is optional. false asserts the color RRF leg contributed
    nothing (content-only query). true asserts it fired. Omit to skip the check.
    """
    raw = json.loads(path.read_text())
    queries = []
    for i, case in enumerate(raw.get("queries", [])):
        if not case.get("query", "").strip():
            raise ValueError(f"queries[{i}] has an empty query string")
        judgments = [
            Judgment(
                drive_file_id=j["drive_file_id"],
                gain=float(j.get("gain", 1.0)),
            )
            for j in case.get("relevant", [])
        ]
        if not judgments:
            raise ValueError(f"queries[{i}] ({case['query']!r}) lists no relevant files")
        expect = case.get("expect_color_leg")
        if expect is not None:
            expect = bool(expect)
        queries.append(
            QueryCase(
                query=case["query"],
                relevant=judgments,
                file_type=case.get("file_type"),
                notes=case.get("notes", ""),
                expect_color_leg=expect,
            )
        )
    if not queries:
        raise ValueError(f"{path} contains no queries")
    return QuerySet(
        queries=queries,
        description=raw.get("description", ""),
        default_k=int(raw.get("default_k", 10)),
    )


# ---------------------------------------------------------------------------
# Metrics — pure functions over a ranked list of file ids
# ---------------------------------------------------------------------------


def recall_at_k(ranked: list[UUID], relevant: set[UUID], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: list[UUID], relevant: set[UUID], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked: list[UUID], relevant: set[UUID]) -> float:
    """1/rank of the first relevant hit over the whole ranked list, else 0."""
    for i, file_id in enumerate(ranked):
        if file_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[UUID], gains: dict[UUID, float], k: int) -> float:
    """
    Graded nDCG@k with the standard log2 discount. The ideal ranking is every
    judged file sorted by gain descending, so a query whose relevant files all
    rank in the top k scores 1.0 regardless of their order among themselves
    only when their gains are equal.
    """
    dcg = sum(
        gains.get(file_id, 0.0) / math.log2(i + 2)
        for i, file_id in enumerate(ranked[:k])
    )
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-query execution
# ---------------------------------------------------------------------------

LEGS = ("text", "thumbnail", "frame", "caption", "transcript", "color")


@dataclass
class QueryResult:
    case: QueryCase
    ranked: list[UUID]
    recall: float
    precision: float
    mrr: float
    ndcg: float
    # leg -> number of this query's relevant files that leg retrieved at all
    leg_hits: dict[str, int] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    resolved_count: int = 0
    color_leg_fired: bool = False


async def resolve_user(session, identifier: str) -> User:
    """Look the dev account up by email, or by UUID if the identifier parses as one."""
    try:
        stmt = select(User).where(User.id == UUID(identifier))
    except ValueError:
        stmt = select(User).where(User.email == identifier)
    user = (await session.execute(stmt)).unique().scalar_one_or_none()
    if user is None:
        raise SystemExit(f"No user matches {identifier!r}. Pass an email or a user UUID.")
    return user


async def resolve_judgments(session, user_id: UUID, query_set: QuerySet) -> dict[str, UUID]:
    """
    Map every drive_file_id named in the query set to its indexed_files row for
    this user. Judgments that do not resolve are reported rather than silently
    dropped: they mean the ground truth references a file the dev account has
    not indexed, which is a data problem, not a search result.
    """
    wanted = {j.drive_file_id for case in query_set.queries for j in case.relevant}
    rows = (
        await session.execute(
            select(IndexedFile.drive_file_id, IndexedFile.id).where(
                IndexedFile.user_id == user_id,
                IndexedFile.drive_file_id.in_(wanted),
            )
        )
    ).all()
    return {drive_file_id: file_id for drive_file_id, file_id in rows}


async def run_case(
    service: IndexingService,
    user_id: UUID,
    case: QueryCase,
    resolved: dict[str, UUID],
    k: int,
    limit: int,
    rrf_k: int,
) -> QueryResult:
    """Run one query and score it against its judgments."""
    unresolved = [j.drive_file_id for j in case.relevant if j.drive_file_id not in resolved]
    gains = {
        resolved[j.drive_file_id]: j.gain
        for j in case.relevant
        if j.drive_file_id in resolved
    }
    relevant_ids = set(gains)

    results = await service.hybrid_search_rrf(
        user_id=user_id,
        query=case.query,
        file_type=case.file_type,
        limit=limit,
        rrf_k=rrf_k,
    )
    ranked = [r["file"].id for r in results]

    # A leg "retrieved" a relevant file when it contributed a nonzero RRF score
    # for it, anywhere in the candidate list — not only inside the top k.
    leg_hits = {leg: 0 for leg in LEGS}
    for r in results:
        if r["file"].id not in relevant_ids:
            continue
        for leg in LEGS:
            if r[f"{leg}_score"] > 0:
                leg_hits[leg] += 1

    return QueryResult(
        case=case,
        ranked=ranked,
        recall=recall_at_k(ranked, relevant_ids, k),
        precision=precision_at_k(ranked, relevant_ids, k),
        mrr=reciprocal_rank(ranked, relevant_ids),
        ndcg=ndcg_at_k(ranked, gains, k),
        leg_hits=leg_hits,
        unresolved=unresolved,
        resolved_count=len(relevant_ids),
        color_leg_fired=any(r.get("color_score", 0.0) > 0 for r in results),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[QueryResult], k: int) -> None:
    print(f"\n{'query':<44} {'R@k':>6} {'P@k':>6} {'MRR':>6} {'nDCG':>6}")
    print("-" * 72)
    for r in results:
        label = r.case.query if len(r.case.query) <= 43 else r.case.query[:40] + "..."
        print(f"{label:<44} {r.recall:>6.3f} {r.precision:>6.3f} {r.mrr:>6.3f} {r.ndcg:>6.3f}")

    n = len(results) or 1
    print("-" * 72)
    print(
        f"{'MEAN (' + str(len(results)) + ' queries, k=' + str(k) + ')':<44} "
        f"{sum(r.recall for r in results) / n:>6.3f} "
        f"{sum(r.precision for r in results) / n:>6.3f} "
        f"{sum(r.mrr for r in results) / n:>6.3f} "
        f"{sum(r.ndcg for r in results) / n:>6.3f}"
    )

    total_relevant = sum(r.resolved_count for r in results)
    if total_relevant:
        print("\nLeg coverage — share of relevant files each leg retrieved:")
        for leg in LEGS:
            hits = sum(r.leg_hits.get(leg, 0) for r in results)
            print(f"  {leg:<12} {hits:>4}/{total_relevant} ({hits / total_relevant:.1%})")

    unresolved = {fid for r in results for fid in r.unresolved}
    if unresolved:
        print(
            f"\nWARNING: {len(unresolved)} judged file(s) are not indexed for this "
            "account and were excluded from every metric:"
        )
        for fid in sorted(unresolved):
            print(f"  - {fid}")

    contracts = [r for r in results if r.case.expect_color_leg is not None]
    if contracts:
        print("\nColor-leg contract (expect_color_leg):")
        for r in contracts:
            expected = r.case.expect_color_leg
            ok = r.color_leg_fired is expected
            status = "PASS" if ok else "FAIL"
            print(
                f"  {status}  fired={r.color_leg_fired} expected={expected}  "
                f"{r.case.query}"
            )


def build_json_report(results: list[QueryResult], k: int, meta: dict) -> dict:
    n = len(results) or 1
    return {
        "meta": meta,
        "summary": {
            "queries": len(results),
            "k": k,
            "recall_at_k": sum(r.recall for r in results) / n,
            "precision_at_k": sum(r.precision for r in results) / n,
            "mrr": sum(r.mrr for r in results) / n,
            "ndcg_at_k": sum(r.ndcg for r in results) / n,
            "leg_coverage": {
                leg: sum(r.leg_hits.get(leg, 0) for r in results)
                for leg in LEGS
            },
            "relevant_resolved": sum(r.resolved_count for r in results),
            "unresolved": sorted({fid for r in results for fid in r.unresolved}),
        },
        "queries": [
            {
                "query": r.case.query,
                "file_type": r.case.file_type,
                "notes": r.case.notes,
                "recall_at_k": r.recall,
                "precision_at_k": r.precision,
                "mrr": r.mrr,
                "ndcg_at_k": r.ndcg,
                "leg_hits": r.leg_hits,
                "color_leg_fired": r.color_leg_fired,
                "expect_color_leg": r.case.expect_color_leg,
                "unresolved": r.unresolved,
                "top_results": [str(fid) for fid in r.ranked[:k]],
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--user",
        required=True,
        help="Dev account to search as: email address or user UUID.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(__file__).parent / "eval_queries.json",
        help="Path to the query set JSON (default: eval_queries.json).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Metric cutoff. Defaults to default_k from the query set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Candidates to retrieve per query; must be >= k (default 50).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=IndexingService.RRF_K,
        help=f"RRF rank constant to evaluate (default {IndexingService.RRF_K}).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the full report as JSON to this path.",
    )
    parser.add_argument(
        "--allow-unconfigured-vision",
        action="store_true",
        help="Run even without Gemini credentials, measuring lexical legs only.",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()

    if not args.queries.is_file():
        print(f"Query set not found: {args.queries}", file=sys.stderr)
        return 2
    query_set = load_query_set(args.queries)
    k = args.k if args.k is not None else query_set.default_k
    if args.limit < k:
        print(f"--limit ({args.limit}) must be >= k ({k})", file=sys.stderr)
        return 2

    if not get_vision_embedding_service().is_configured:
        if not args.allow_unconfigured_vision:
            print(
                "Gemini embeddings are not configured, so the thumbnail, frame, caption-\n"
                "semantic, and transcript-semantic legs would return nothing and the\n"
                "scores below would describe lexical-only search. Set GEMINI_API_KEY, or\n"
                "pass --allow-unconfigured-vision if that is what you meant to measure.",
                file=sys.stderr,
            )
            return 2
        print("NOTE: running without Gemini — semantic legs are inactive.\n")

    async with async_session_maker() as session:
        user = await resolve_user(session, args.user)
        resolved = await resolve_judgments(session, user.id, query_set)
        service = IndexingService(session)

        print(f"Evaluating {len(query_set.queries)} queries as {user.email} (k={k}, rrf_k={args.rrf_k})")
        if query_set.description:
            print(f"Query set: {query_set.description}")

        results = []
        for case in query_set.queries:
            results.append(
                await run_case(service, user.id, case, resolved, k, args.limit, args.rrf_k)
            )

    print_report(results, k)

    if args.json_out:
        report = build_json_report(
            results,
            k,
            meta={
                "user": user.email,
                "query_set": str(args.queries),
                "limit": args.limit,
                "rrf_k": args.rrf_k,
            },
        )
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

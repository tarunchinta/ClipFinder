"""
Ranking metrics for the Distill vs TwelveLabs benchmark.

Pure functions over ranked ID lists — no database, no network — so they can be
unit-tested directly. `run_eval.py` normalizes both systems' responses into
ranked lists of drive_file_ids and hands them here.

Binary relevance drives recall/precision/MRR/hit-rate; graded relevance (when
the query supplies it) drives nDCG, falling back to gain 1 for anything in the
binary relevant set.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Iterable, Mapping, Optional, Sequence


def _dedup(ranked: Sequence[str]) -> list[str]:
    """Keep first occurrence only; both systems can return a doc more than once."""
    seen: set[str] = set()
    out: list[str] = []
    for item in ranked:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> Optional[float]:
    """
    Fraction of relevant docs retrieved in the top k.

    Returns None when the query has no relevant docs — an undefined value that
    must be excluded from averages rather than counted as 0.0.
    """
    rel = set(relevant)
    if not rel:
        return None
    top = set(_dedup(ranked)[:k])
    return len(top & rel) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> Optional[float]:
    """Fraction of the top k that is relevant. None when k <= 0."""
    if k <= 0:
        return None
    rel = set(relevant)
    top = _dedup(ranked)[:k]
    if not top:
        return 0.0
    # Divide by k, not len(top): a system returning 2 results for k=10 is not
    # more precise than one returning 10, it just retrieved less.
    return sum(1 for doc in top if doc in rel) / k


def hit_rate_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> Optional[float]:
    """1.0 if any relevant doc appears in the top k, else 0.0."""
    rel = set(relevant)
    if not rel:
        return None
    return 1.0 if rel & set(_dedup(ranked)[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> Optional[float]:
    """1 / rank of the first relevant doc (1-indexed); 0.0 if none is retrieved."""
    rel = set(relevant)
    if not rel:
        return None
    for i, doc in enumerate(_dedup(ranked), start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], relevant: Iterable[str], k: int) -> Optional[float]:
    """
    Mean of precision@i over the positions i <= k holding a relevant doc.

    Normalized by min(len(relevant), k) so a query with more relevant docs than
    k can still reach 1.0.
    """
    rel = set(relevant)
    if not rel:
        return None
    top = _dedup(ranked)[:k]
    hits = 0
    total = 0.0
    for i, doc in enumerate(top, start=1):
        if doc in rel:
            hits += 1
            total += hits / i
    denom = min(len(rel), k)
    return total / denom if denom else 0.0


def ndcg_at_k(
    ranked: Sequence[str],
    relevant: Iterable[str],
    k: int,
    graded: Optional[Mapping[str, float]] = None,
) -> Optional[float]:
    """
    Normalized discounted cumulative gain with binary or graded relevance.

    Gains come from `graded` when supplied, else 1.0 for every doc in
    `relevant`. The ideal ranking is the same gains sorted descending.
    """
    rel = set(relevant)
    gains: dict[str, float] = {doc: 1.0 for doc in rel}
    if graded:
        gains.update({doc: float(g) for doc, g in graded.items() if float(g) > 0})
    if not gains:
        return None

    def dcg(scores: Sequence[float]) -> float:
        return sum(s / math.log2(i + 1) for i, s in enumerate(scores, start=1))

    actual = [gains.get(doc, 0.0) for doc in _dedup(ranked)[:k]]
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(actual) / idcg if idcg else None


def evaluate_query(
    ranked: Sequence[str],
    relevant: Iterable[str],
    k_values: Sequence[int],
    graded: Optional[Mapping[str, float]] = None,
) -> dict[str, Optional[float]]:
    """Every metric for one (arm, query) pair, keyed as e.g. "recall@5"."""
    rel = list(relevant)
    out: dict[str, Optional[float]] = {"mrr": reciprocal_rank(ranked, rel)}
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(ranked, rel, k)
        out[f"precision@{k}"] = precision_at_k(ranked, rel, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(ranked, rel, k)
        out[f"map@{k}"] = average_precision(ranked, rel, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k, graded=graded)
    return out


def aggregate(
    per_query: Sequence[Mapping[str, Optional[float]]],
) -> dict[str, Optional[float]]:
    """
    Macro-average each metric across queries, skipping None (undefined) values.

    A metric that was undefined for every query stays None rather than becoming
    0.0, so "no ground truth" never reads as "scored zero".
    """
    keys: list[str] = []
    for row in per_query:
        for key in row:
            if key not in keys:
                keys.append(key)
    out: dict[str, Optional[float]] = {}
    for key in keys:
        values = [row[key] for row in per_query if row.get(key) is not None]
        out[key] = mean(values) if values else None  # type: ignore[arg-type]
    return out


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Linear-interpolated percentile (pct in 0..100). None for an empty input."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)

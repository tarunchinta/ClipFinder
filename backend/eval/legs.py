"""
Map queries.json fields to Distill hybrid-search legs and TwelveLabs
search_options.

This is the file to read for "what does this query field do on each system."
eval.run and eval.search import the helpers below; do not duplicate the
table.

Query JSON field -> Distill hybrid_search_rrf leg, TwelveLabs search_options
------------------------------------------------------------------------------
tags[] values:
  "text"        Distill: filename trigram ("text")     TL: none
  "thumbnail"   Distill: poster embedding              TL: none (index addon, not a search filter)
  "frame"       Distill: video frame embeddings        TL: "visual"
  "caption"     Distill: caption lexical+semantic      TL: "audio"
  "transcript"  Distill: transcript lexical+semantic   TL: "audio"
expect_color_leg:
  true          Distill: color signature ("color")     TL: none

TwelveLabs thumbnail is addons=["thumbnail"] at index creation. Search hits
may include thumbnail_url; do not pass "thumbnail" in search_options.

Do not put "color" in tags[]; use expect_color_leg. High-level aliases like
"visual" are not accepted — name each Distill leg.

When tags is empty and expect_color_leg is not true, the helpers return None
so the caller keeps arm/defaults config (eval.search then runs every leg).
"""

from __future__ import annotations

from typing import Iterable, Optional

from app.services.indexing import HYBRID_SEARCH_LEGS

# tags[] value -> Distill hybrid_search_rrf leg name
TAG_TO_DISTILL_LEG: dict[str, str] = {
    "text": "text",  # filename trigram
    "thumbnail": "thumbnail",  # poster embedding
    "frame": "frame",  # video frame embeddings
    "caption": "caption",  # caption lexical + semantic
    "transcript": "transcript",  # transcript lexical + semantic
}

# tags[] value -> TwelveLabs search.query search_options
# text has no TL analog (filename search). Color is expect_color_leg, also no TL analog.
TAG_TO_TL_OPTIONS: dict[str, tuple[str, ...]] = {
    "text": (),  # no TL analog (filename search)
    "thumbnail": (),  # TL thumbnail is an index addon, not search_options
    "frame": ("visual",),
    "caption": ("audio",),
    "transcript": ("audio",),
}

# expect_color_leg=True -> Distill "color" (Lab histogram). TL: none.


def validate_query_tags(
    tags: Iterable[str],
    *,
    query_id: Optional[str] = None,
) -> list[str]:
    """
    Normalize tags and fail loudly on unknown names or "color" in tags[].

    Color is gated by expect_color_leg, not a tag, so a typo cannot silently
    enable the wrong Distill path.
    """
    prefix = f"query {query_id}: " if query_id else ""
    normalized = [str(t).strip() for t in tags or [] if str(t).strip()]
    if "color" in normalized:
        raise ValueError(
            f"{prefix}do not put 'color' in tags[]; set expect_color_leg true "
            "to enable Distill's color signature leg (TwelveLabs has no analog)."
        )
    unknown = [t for t in normalized if t not in TAG_TO_DISTILL_LEG]
    if unknown:
        raise ValueError(
            f"{prefix}unknown tag(s) {unknown}. Valid tags are Distill leg "
            f"names: {list(TAG_TO_DISTILL_LEG)}. Color is expect_color_leg, "
            "not a tag."
        )
    return normalized


def _query_selects_legs(
    tags: Iterable[str],
    expect_color_leg: Optional[bool],
) -> bool:
    """True when the query JSON is choosing modalities instead of arm defaults."""
    return bool(list(tags or [])) or expect_color_leg is True


def distill_legs_for_query(
    tags: Iterable[str],
    expect_color_leg: Optional[bool] = None,
    *,
    query_id: Optional[str] = None,
) -> Optional[list[str]]:
    """
    Distill hybrid_search_rrf legs implied by tags[] and expect_color_leg.

    None means the caller should keep arm/defaults (typically every leg).
    expect_color_leg true appends "color"; false/None does not.
    """
    normalized = validate_query_tags(tags, query_id=query_id)
    if not _query_selects_legs(normalized, expect_color_leg):
        return None
    legs: list[str] = []
    seen: set[str] = set()
    for tag in normalized:
        leg = TAG_TO_DISTILL_LEG[tag]
        if leg not in seen:
            seen.add(leg)
            legs.append(leg)
    if expect_color_leg is True and "color" not in seen:
        legs.append("color")
        seen.add("color")
    extra = seen - set(HYBRID_SEARCH_LEGS)
    if extra:
        raise ValueError(f"mapped Distill leg(s) are not in HYBRID_SEARCH_LEGS: {sorted(extra)}")
    return legs


def tl_search_options_for_query(
    tags: Iterable[str],
    expect_color_leg: Optional[bool] = None,
    *,
    query_id: Optional[str] = None,
) -> Optional[list[str]]:
    """
    TwelveLabs search_options implied by the same tags[] / expect_color_leg.

    None means keep arm/defaults. Color and text add no TL modalities, so a
    query that only sets those can yield an empty list (not None).
    """
    normalized = validate_query_tags(tags, query_id=query_id)
    if not _query_selects_legs(normalized, expect_color_leg):
        return None
    options: list[str] = []
    seen: set[str] = set()
    for tag in normalized:
        for option in TAG_TO_TL_OPTIONS[tag]:
            if option not in seen:
                seen.add(option)
                options.append(option)
    return options

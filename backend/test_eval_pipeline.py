"""Unit tests for the Distill vs TwelveLabs eval pipeline (metrics + config wiring)."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent))

from app.services.indexing import HYBRID_SEARCH_LEGS, IndexingService
from eval_legs import distill_legs_for_query, tl_search_options_for_query
from eval_metrics import (
    aggregate,
    average_precision,
    evaluate_query,
    hit_rate_at_k,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
import eval_search
from run_eval import (
    Arm,
    EvalQuery,
    _distill_cfg,
    _merge,
    _parse_judgments,
    _tl_cfg,
    _tl_item_drive_id,
    build_corpus,
    load_queries,
)


class RecallTests(unittest.TestCase):
    def test_partial_recall(self):
        ranked = ["a", "b", "c", "d"]
        self.assertAlmostEqual(recall_at_k(ranked, ["a", "z"], 4), 0.5)

    def test_k_truncates(self):
        self.assertEqual(recall_at_k(["a", "b"], ["b"], 1), 0.0)
        self.assertEqual(recall_at_k(["a", "b"], ["b"], 2), 1.0)

    def test_no_judgments_is_undefined_not_zero(self):
        self.assertIsNone(recall_at_k(["a"], [], 5))

    def test_duplicates_collapse_before_truncation(self):
        # TL returns several clips per video. Dedup runs before the top-k cut, so
        # three clips of "a" occupy one slot and "b" still makes the top 2.
        self.assertAlmostEqual(recall_at_k(["a", "a", "a", "b"], ["a", "b"], 2), 1.0)
        # Without dedup-first, "b" would fall outside k=2 and recall would be 0.5.
        self.assertAlmostEqual(recall_at_k(["a", "z", "y", "b"], ["a", "b"], 2), 0.5)


class PrecisionTests(unittest.TestCase):
    def test_divides_by_k_not_result_count(self):
        # 1 relevant hit in a 2-result response is precision@10 = 0.1, not 0.5.
        self.assertAlmostEqual(precision_at_k(["a", "z"], ["a"], 10), 0.1)

    def test_empty_ranking(self):
        self.assertEqual(precision_at_k([], ["a"], 5), 0.0)


class RankTests(unittest.TestCase):
    def test_reciprocal_rank_uses_first_hit(self):
        self.assertAlmostEqual(reciprocal_rank(["x", "y", "a"], ["a", "y"]), 0.5)

    def test_reciprocal_rank_miss(self):
        self.assertEqual(reciprocal_rank(["x", "y"], ["a"]), 0.0)

    def test_hit_rate_is_binary(self):
        self.assertEqual(hit_rate_at_k(["x", "a"], ["a"], 2), 1.0)
        self.assertEqual(hit_rate_at_k(["x", "a"], ["a"], 1), 0.0)

    def test_average_precision_rewards_early_hits(self):
        early = average_precision(["a", "b", "x", "y"], ["a", "b"], 4)
        late = average_precision(["x", "y", "a", "b"], ["a", "b"], 4)
        self.assertGreater(early, late)
        self.assertAlmostEqual(early, 1.0)


class NdcgTests(unittest.TestCase):
    def test_perfect_ranking_is_one(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], ["a", "b"], 2), 1.0)

    def test_graded_prefers_higher_gain_first(self):
        graded = {"a": 3.0, "b": 1.0}
        good = ndcg_at_k(["a", "b"], ["a", "b"], 2, graded=graded)
        bad = ndcg_at_k(["b", "a"], ["a", "b"], 2, graded=graded)
        self.assertGreater(good, bad)

    def test_undefined_without_judgments(self):
        self.assertIsNone(ndcg_at_k(["a"], [], 5))


class AggregateTests(unittest.TestCase):
    def test_skips_undefined_values(self):
        rows = [{"recall@5": 1.0}, {"recall@5": None}, {"recall@5": 0.0}]
        self.assertAlmostEqual(aggregate(rows)["recall@5"], 0.5)

    def test_all_undefined_stays_none(self):
        self.assertIsNone(aggregate([{"recall@5": None}])["recall@5"])

    def test_evaluate_query_covers_every_k(self):
        metrics = evaluate_query(["a"], ["a"], [1, 3])
        for key in ("recall@1", "precision@3", "hit_rate@1", "map@3", "ndcg@1", "mrr"):
            self.assertIn(key, metrics)

    def test_percentile(self):
        self.assertAlmostEqual(percentile([10.0, 20.0, 30.0], 50), 20.0)
        self.assertIsNone(percentile([], 50))


class LegResolutionTests(unittest.TestCase):
    def test_none_means_every_leg(self):
        self.assertEqual(IndexingService._resolve_legs(None), frozenset(HYBRID_SEARCH_LEGS))

    def test_subset_is_preserved(self):
        self.assertEqual(IndexingService._resolve_legs(["frame"]), frozenset({"frame"}))

    def test_typo_raises_rather_than_silently_dropping(self):
        with self.assertRaises(ValueError):
            IndexingService._resolve_legs(["frames"])


class LegGatingTests(unittest.TestCase):
    """Disabled legs must not be queried at all, not merely zero-weighted."""

    LEG_METHODS = {
        "text": "search_files_with_scores",
        "caption": "description_lexical_search",
        "transcript": "transcript_lexical_search",
        "color": "color_search",
        "thumbnail": "thumbnail_semantic_search_by_embedding",
        "frame": "_vision_search_video_frames",
    }
    SEMANTIC_METHODS = (
        "description_semantic_search_by_embedding",
        "transcript_semantic_search_by_embedding",
    )

    def _run(self, legs):
        service = IndexingService(session=object())
        mocks = {}
        patchers = []
        for name in list(self.LEG_METHODS.values()) + list(self.SEMANTIC_METHODS):
            mock = AsyncMock(return_value=[])
            mocks[name] = mock
            patchers.append(patch.object(service, name, mock))

        vision = AsyncMock()
        vision.is_configured = True
        vision.generate_text_embedding = AsyncMock(return_value=[0.1, 0.2])

        for pt in patchers:
            pt.start()
        try:
            with patch(
                "app.services.indexing.get_vision_embedding_service", return_value=vision
            ):
                asyncio.run(
                    service.hybrid_search_rrf(user_id=uuid4(), query="a dog", legs=legs)
                )
        finally:
            for pt in patchers:
                pt.stop()
        return {name: mock.await_count for name, mock in mocks.items()}, vision

    def test_default_runs_every_leg(self):
        calls, vision = self._run(None)
        for method in self.LEG_METHODS.values():
            self.assertEqual(calls[method], 1, f"{method} should have run")
        self.assertEqual(vision.generate_text_embedding.await_count, 1)

    def test_frame_only_skips_the_rest(self):
        calls, _ = self._run(["frame"])
        self.assertEqual(calls["_vision_search_video_frames"], 1)
        for leg, method in self.LEG_METHODS.items():
            if leg != "frame":
                self.assertEqual(calls[method], 0, f"{method} should not have run")
        for method in self.SEMANTIC_METHODS:
            self.assertEqual(calls[method], 0)

    def test_lexical_only_skips_the_embedding_call(self):
        # No embedding leg enabled means we never pay for the query embedding.
        calls, vision = self._run(["text", "color"])
        self.assertEqual(vision.generate_text_embedding.await_count, 0)
        self.assertEqual(calls["search_files_with_scores"], 1)
        self.assertEqual(calls["color_search"], 1)

    def test_caption_leg_runs_both_lexical_and_semantic(self):
        calls, _ = self._run(["caption"])
        self.assertEqual(calls["description_lexical_search"], 1)
        self.assertEqual(calls["description_semantic_search_by_embedding"], 1)
        self.assertEqual(calls["transcript_lexical_search"], 0)

    def test_empty_leg_set_returns_nothing(self):
        service = IndexingService(session=object())
        result = asyncio.run(
            service.hybrid_search_rrf(user_id=uuid4(), query="a dog", legs=[])
        )
        self.assertEqual(result, [])

    def test_blank_query_short_circuits(self):
        service = IndexingService(session=object())
        self.assertEqual(
            asyncio.run(service.hybrid_search_rrf(user_id=uuid4(), query="  ")), []
        )


class ConfigTests(unittest.TestCase):
    def test_merge_ignores_none_overrides(self):
        merged = _merge({"a": 1, "b": 2}, {"b": None, "c": 3})
        self.assertEqual(merged, {"a": 1, "b": 2, "c": 3})

    def _write(self, payload) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "queries.json"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        return tmp

    def test_defaults_flow_into_arms_and_are_overridable(self):
        path = self._write(
            {
                "k_values": [1, 5],
                "defaults": {
                    "limit": 7,
                    "distill": {"legs": list(HYBRID_SEARCH_LEGS)},
                    "twelvelabs": {"search_options": ["visual", "audio"]},
                },
                "arms": [
                    {"name": "full", "system": "distill"},
                    {"name": "frames", "system": "distill", "distill": {"legs": ["frame"]}},
                    {
                        "name": "tl",
                        "system": "twelvelabs",
                        "twelvelabs": {"search_options": ["visual"]},
                    },
                ],
                "queries": [{"id": "q1", "query": "a dog", "relevant": ["d1"]}],
            }
        )
        arms, queries, k_values = load_queries(path)
        by_name = {a.name: a for a in arms}
        self.assertEqual(k_values, [1, 5])
        self.assertEqual(by_name["full"].limit, 7)
        self.assertEqual(by_name["full"].distill["legs"], list(HYBRID_SEARCH_LEGS))
        self.assertEqual(by_name["frames"].distill["legs"], ["frame"])
        self.assertEqual(by_name["tl"].twelvelabs["search_options"], ["visual"])
        self.assertEqual(queries[0].relevant, ["d1"])

    def test_bad_system_rejected(self):
        path = self._write(
            {
                "arms": [{"name": "x", "system": "elasticsearch"}],
                "queries": [{"id": "q1", "query": "a"}],
            }
        )
        with self.assertRaises(ValueError):
            load_queries(path)

    def test_empty_query_string_rejected(self):
        path = self._write(
            {
                "arms": [{"name": "x", "system": "distill"}],
                "queries": [{"id": "q1", "query": "   "}],
            }
        )
        with self.assertRaises(ValueError):
            load_queries(path)

    def test_judgments_accept_shared_schema_with_gains(self):
        relevant, graded = _parse_judgments(
            [{"drive_file_id": "d1", "gain": 3}, {"drive_file_id": "d2"}], "q1"
        )
        self.assertEqual(relevant, ["d1", "d2"])
        self.assertEqual(graded, {"d1": 3.0, "d2": 1.0})

    def test_judgments_accept_bare_strings(self):
        relevant, graded = _parse_judgments(["d1"], "q1")
        self.assertEqual(relevant, ["d1"])
        self.assertEqual(graded, {"d1": 1.0})

    def test_judgment_without_drive_file_id_rejected(self):
        with self.assertRaises(ValueError):
            _parse_judgments([{"gain": 2}], "q1")

    def test_k_values_fall_back_to_default_k(self):
        path = self._write(
            {
                "default_k": 20,
                "arms": [{"name": "a", "system": "distill"}],
                "queries": [{"id": "q1", "query": "x", "relevant": ["d1"]}],
            }
        )
        _, _, k_values = load_queries(path)
        self.assertEqual(k_values, [1, 3, 5, 20])

    def test_shipped_file_is_readable_by_both_tools(self):
        # eval_search.py defaults --queries to this same file, so the schema
        # has to satisfy both loaders or one of them breaks.
        path = Path(__file__).parent / "eval_queries.json"
        query_set = eval_search.load_query_set(path)
        arms, queries, _ = load_queries(path)
        self.assertEqual(len(query_set.queries), len(queries))
        self.assertEqual(
            [j.drive_file_id for j in query_set.queries[0].relevant],
            queries[0].relevant,
        )
        self.assertEqual(query_set.queries[0].tags, queries[0].tags)
        self.assertEqual(query_set.queries[0].expect_color_leg, queries[0].expect_color_leg)
        self.assertTrue(arms)

    def test_eval_search_shares_the_leg_constant(self):
        self.assertEqual(tuple(eval_search.LEGS), tuple(HYBRID_SEARCH_LEGS))

    def test_shipped_queries_file_parses(self):
        arms, queries, _ = load_queries(Path(__file__).parent / "eval_queries.json")
        self.assertEqual({a.name for a in arms}, {"distill", "twelvelabs"})
        self.assertTrue(queries)
        q001 = next(q for q in queries if q.id == "q001")
        self.assertEqual(
            q001.tags,
            ["text", "thumbnail", "frame", "caption", "transcript"],
        )
        self.assertTrue(q001.expect_color_leg)
        self.assertEqual(
            distill_legs_for_query(q001.tags, q001.expect_color_leg),
            list(HYBRID_SEARCH_LEGS),
        )
        self.assertEqual(
            tl_search_options_for_query(q001.tags, q001.expect_color_leg),
            ["visual", "audio"],
        )
        for arm in arms:
            if arm.system == "distill" and arm.distill.get("legs") is not None:
                self.assertFalse(set(arm.distill["legs"]) - set(HYBRID_SEARCH_LEGS))

    def test_unknown_query_tag_rejected_at_load(self):
        path = self._write(
            {
                "arms": [{"name": "a", "system": "distill"}],
                "queries": [
                    {"id": "q1", "query": "x", "tags": ["visual"], "relevant": ["d1"]}
                ],
            }
        )
        with self.assertRaises(ValueError):
            load_queries(path)

    def test_color_tag_rejected_at_load(self):
        path = self._write(
            {
                "arms": [{"name": "a", "system": "distill"}],
                "queries": [
                    {"id": "q1", "query": "x", "tags": ["color"], "relevant": ["d1"]}
                ],
            }
        )
        with self.assertRaises(ValueError):
            load_queries(path)


class TagMappingTests(unittest.TestCase):
    ALL_TAGS = ["text", "thumbnail", "frame", "caption", "transcript"]

    def test_all_tags_plus_color_enable_every_distill_leg_and_both_tl_options(self):
        self.assertEqual(
            distill_legs_for_query(self.ALL_TAGS, True),
            list(HYBRID_SEARCH_LEGS),
        )
        self.assertEqual(
            tl_search_options_for_query(self.ALL_TAGS, True),
            ["visual", "audio"],
        )

    def test_thumbnail_only_maps_to_visual(self):
        self.assertEqual(distill_legs_for_query(["thumbnail"], False), ["thumbnail"])
        self.assertEqual(tl_search_options_for_query(["thumbnail"], False), ["visual"])

    def test_unknown_tag_raises(self):
        with self.assertRaises(ValueError):
            distill_legs_for_query(["visual"], None)

    def test_color_in_tags_raises(self):
        with self.assertRaises(ValueError):
            distill_legs_for_query(["color"], None)

    def test_empty_tags_fall_back_to_none(self):
        self.assertIsNone(distill_legs_for_query([], False))
        self.assertIsNone(tl_search_options_for_query([], None))
        self.assertIsNone(distill_legs_for_query([], None))

    def test_expect_color_leg_alone_selects_color(self):
        self.assertEqual(distill_legs_for_query([], True), ["color"])
        self.assertEqual(tl_search_options_for_query([], True), [])

    def test_text_only_has_no_tl_options(self):
        self.assertEqual(distill_legs_for_query(["text"], False), ["text"])
        self.assertEqual(tl_search_options_for_query(["text"], False), [])

    def test_tags_override_arm_legs_unless_query_block_wins(self):
        arm = Arm(
            name="distill",
            system="distill",
            limit=10,
            distill={"legs": ["text"]},
        )
        tagged = EvalQuery(
            id="q1",
            query="a dog",
            relevant=["d1"],
            tags=["thumbnail", "frame"],
            expect_color_leg=False,
        )
        self.assertEqual(_distill_cfg(arm, tagged)["legs"], ["thumbnail", "frame"])
        override = EvalQuery(
            id="q1",
            query="a dog",
            relevant=["d1"],
            tags=["thumbnail", "frame"],
            distill={"legs": ["caption"]},
        )
        self.assertEqual(_distill_cfg(arm, override)["legs"], ["caption"])

    def test_tags_override_arm_tl_search_options(self):
        arm = Arm(
            name="twelvelabs",
            system="twelvelabs",
            limit=10,
            twelvelabs={"search_options": ["visual"]},
        )
        tagged = EvalQuery(
            id="q1",
            query="someone talking",
            relevant=["d1"],
            tags=["transcript"],
        )
        self.assertEqual(_tl_cfg(arm, tagged)["search_options"], ["audio"])


class CorpusTests(unittest.TestCase):
    STATE = {
        "twelvelabs_index_id": "idx1",
        "videos": {
            "instagram:AAA": {
                "status": "ready",
                "distill_file_id": "uuid-aaa",
                "tl_video_id": "tlv-aaa",
                "tl_indexed_asset_id": "tlv-aaa",
            },
            "instagram:BBB": {"status": "failed", "tl_video_id": "tlv-bbb"},
        },
    }

    def test_only_ready_videos_join(self):
        corpus = build_corpus(self.STATE)
        self.assertEqual(corpus.ready_drive_ids, {"instagram:AAA"})
        self.assertEqual(corpus.tl_video_to_drive["tlv-aaa"], "instagram:AAA")
        self.assertNotIn("tlv-bbb", corpus.tl_video_to_drive)
        self.assertEqual(corpus.distill_file_to_drive["uuid-aaa"], "instagram:AAA")

    def test_user_metadata_wins_over_id_map(self):
        corpus = build_corpus(self.STATE)

        class Hit:
            user_metadata = {"drive_file_id": "instagram:FROM_META"}
            video_id = "tlv-aaa"

        self.assertEqual(_tl_item_drive_id(Hit(), corpus), "instagram:FROM_META")

    def test_falls_back_to_state_id_map(self):
        corpus = build_corpus(self.STATE)

        class Hit:
            user_metadata = None
            video_id = "tlv-aaa"

        self.assertEqual(_tl_item_drive_id(Hit(), corpus), "instagram:AAA")

    def test_unknown_hit_is_dropped(self):
        corpus = build_corpus(self.STATE)

        class Hit:
            user_metadata = None
            video_id = "tlv-unknown"

        self.assertIsNone(_tl_item_drive_id(Hit(), corpus))


if __name__ == "__main__":
    unittest.main()

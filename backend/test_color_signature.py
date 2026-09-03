"""Unit tests for the color-signature extract/compare/lexicon encoder."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.services.color_signature import (
    HIST_DIM,
    encode_query,
    histogram_intersection,
    merge_signatures,
    rgb_uint8_to_lab,
    signature_from_rgb,
)


def _block(rgb, h=48, w=48) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = rgb
    return img


class LabConversionTests(unittest.TestCase):
    def test_teal_is_negative_a_and_b(self):
        lab = rgb_uint8_to_lab(_block([20, 90, 100]))
        self.assertLess(float(lab[..., 1].mean()), 0.0)
        self.assertLess(float(lab[..., 2].mean()), 0.0)

    def test_orange_is_positive_a_and_b(self):
        lab = rgb_uint8_to_lab(_block([230, 130, 40]))
        self.assertGreater(float(lab[..., 1].mean()), 0.0)
        self.assertGreater(float(lab[..., 2].mean()), 0.0)


class SignatureExtractTests(unittest.TestCase):
    def test_split_tone_palette_has_shadow_and_highlight(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:32] = [18, 80, 90]  # dark teal
        img[32:] = [255, 200, 80]  # bright orange (L* > 70)
        sig = signature_from_rgb(img)
        self.assertIsNotNone(sig)
        self.assertEqual(len(sig.histogram), HIST_DIM)
        zones = {s["zone"] for s in sig.palette}
        self.assertIn("shadows", zones)
        self.assertIn("highlights", zones)
        shadows = next(s for s in sig.palette if s["zone"] == "shadows")
        highlights = next(s for s in sig.palette if s["zone"] == "highlights")
        self.assertLess(shadows["a"], 0.0)
        self.assertGreater(highlights["b"], 0.0)

    def test_letterbox_black_is_ignored(self):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[10:30, 10:30] = [230, 140, 50]
        sig = signature_from_rgb(img)
        self.assertIsNotNone(sig)
        self.assertGreater(sig.mean_l, 40.0)

    def test_merge_averages_histograms(self):
        a = signature_from_rgb(_block([20, 90, 100]))
        b = signature_from_rgb(_block([230, 140, 50]))
        merged = merge_signatures([a, b])
        self.assertIsNotNone(merged)
        self.assertAlmostEqual(sum(merged.histogram), 1.0, places=5)


class QueryEncoderTests(unittest.TestCase):
    def test_no_color_language_skips(self):
        self.assertIsNone(encode_query("person walking"))
        self.assertIsNone(encode_query("sourdough starter explainer"))
        self.assertIsNone(encode_query(""))

    def test_eval_color_queries_match_encoder_contract(self):
        import json

        path = Path(__file__).parent / "eval_color_queries.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for case in data["queries"]:
            probe = encode_query(case["query"])
            if case.get("expect_color_leg") is True:
                self.assertIsNotNone(probe, case["query"])
            elif case.get("expect_color_leg") is False:
                self.assertIsNone(probe, case["query"])

    def test_teal_and_orange_fires(self):
        probe = encode_query("teal and orange person walking")
        self.assertIsNotNone(probe)
        zones = {s["zone"] for s in probe.palette}
        self.assertIn("shadows", zones)
        self.assertIn("highlights", zones)

    def test_teal_and_orange_ranks_split_tone_above_magenta(self):
        probe = encode_query("find footage with that teal-and-orange grade")
        self.assertIsNotNone(probe)
        split = np.zeros((64, 64, 3), dtype=np.uint8)
        split[:32] = [18, 80, 90]
        split[32:] = [230, 140, 50]
        magenta = _block([180, 40, 160], h=64, w=64)
        split_sig = signature_from_rgb(split)
        mag_sig = signature_from_rgb(magenta)
        split_score = histogram_intersection(probe.histogram, split_sig.histogram)
        mag_score = histogram_intersection(probe.histogram, mag_sig.histogram)
        self.assertGreater(split_score, mag_score)


if __name__ == "__main__":
    unittest.main()

"""Per-clip color signatures for the look/grade RRF leg.

Stores the grade (Lab split-tone + chroma histogram), not a Gemini embedding.
Query text is mapped through a color lexicon; queries with no color language
skip this leg so it cannot pollute content search.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

HIST_BINS = 16
HIST_DIM = HIST_BINS * HIST_BINS
AB_MIN = -128.0
AB_MAX = 127.0
L_CLIP_LO = 5.0
L_CLIP_HI = 97.0
SHADOW_L = 40.0
HIGHLIGHT_L = 70.0
GRAY_CHROMA = 8.0
QUERY_SIGMA_LAB = 22.0

_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


@dataclass(frozen=True)
class LabSwatch:
    l: float
    a: float
    b: float
    zone: str  # shadows | midtones | highlights


@dataclass
class ColorSignature:
    histogram: list[float]
    palette: list[dict]
    mean_l: float
    std_l: float
    mean_a: float
    mean_b: float


def color_signature_values(sig: ColorSignature) -> dict:
    """Signature as plain column values, for the PostgREST worker path."""
    return {
        "histogram": [float(x) for x in sig.histogram],
        "palette": sig.palette,
        "mean_l": float(sig.mean_l),
        "std_l": float(sig.std_l),
        "mean_a": float(sig.mean_a),
        "mean_b": float(sig.mean_b),
    }


def apply_color_signature(row, sig: ColorSignature) -> None:
    """Write signature columns onto an IndexedFile-like ORM row."""
    values = color_signature_values(sig)
    row.color_histogram = values["histogram"]
    row.color_palette = values["palette"]
    row.color_mean_l = values["mean_l"]
    row.color_std_l = values["std_l"]
    row.color_mean_a = values["mean_a"]
    row.color_mean_b = values["mean_b"]


def signature_from_row(row) -> Optional[ColorSignature]:
    hist = _coerce_histogram(getattr(row, "color_histogram", None))
    if hist is None:
        return None
    return ColorSignature(
        histogram=hist,
        palette=list(getattr(row, "color_palette", None) or []),
        mean_l=float(getattr(row, "color_mean_l", 0.0) or 0.0),
        std_l=float(getattr(row, "color_std_l", 0.0) or 0.0),
        mean_a=float(getattr(row, "color_mean_a", 0.0) or 0.0),
        mean_b=float(getattr(row, "color_mean_b", 0.0) or 0.0),
    )


def histogram_intersection(a: list[float], b: list[float]) -> float:
    """L1-normalized histogram intersection in [0, 1]."""
    if len(a) != HIST_DIM or len(b) != HIST_DIM:
        return 0.0
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.minimum(aa, bb).sum())


def rgb_uint8_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert HxWx3 uint8 sRGB to HxWx3 Lab (D65)."""
    linear = _srgb_to_linear(rgb.astype(np.float64) / 255.0)
    xyz = linear @ _SRGB_TO_XYZ.T
    xyz_n = xyz / _D65
    delta = 6.0 / 29.0
    t = np.where(
        xyz_n > delta**3,
        np.cbrt(np.clip(xyz_n, 0.0, None)),
        xyz_n / (3.0 * delta**2) + 4.0 / 29.0,
    )
    fx, fy, fz = t[..., 0], t[..., 1], t[..., 2]
    lab = np.empty(rgb.shape, dtype=np.float64)
    lab[..., 0] = 116.0 * fy - 16.0
    lab[..., 1] = 500.0 * (fx - fy)
    lab[..., 2] = 200.0 * (fy - fz)
    return lab


def signature_from_rgb(rgb: np.ndarray) -> Optional[ColorSignature]:
    """Build a signature from an HxWx3 uint8 RGB array."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        return None
    lab = rgb_uint8_to_lab(rgb)
    L = lab[..., 0].ravel()
    a = lab[..., 1].ravel()
    b = lab[..., 2].ravel()
    valid = (L > L_CLIP_LO) & (L < L_CLIP_HI)
    if not np.any(valid):
        return None
    L, a, b = L[valid], a[valid], b[valid]
    chroma = np.sqrt(a * a + b * b)
    weights = np.clip(chroma / GRAY_CHROMA, 0.0, 1.0)
    if float(weights.sum()) <= 1e-9:
        weights = np.ones_like(weights)

    hist = _weighted_ab_histogram(a, b, weights)
    palette = _split_tone_palette(L, a, b, weights)
    return ColorSignature(
        histogram=hist,
        palette=palette,
        mean_l=float(L.mean()),
        std_l=float(L.std()),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
    )


def signature_from_image_bytes(image_bytes: bytes) -> Optional[ColorSignature]:
    rgb = _rgb_from_image_bytes(image_bytes)
    if rgb is None:
        return None
    return signature_from_rgb(rgb)


def signature_from_jpeg_path(path: str) -> Optional[ColorSignature]:
    try:
        with open(path, "rb") as f:
            return signature_from_image_bytes(f.read())
    except OSError as e:
        logger.warning("Failed to read JPEG for color signature %s: %s", path, e)
        return None


def merge_signatures(sigs: list[ColorSignature]) -> Optional[ColorSignature]:
    """Average histograms / tone stats and pool split-tone palettes across frames."""
    if not sigs:
        return None
    if len(sigs) == 1:
        return sigs[0]
    hist = np.mean(
        np.stack([np.asarray(s.histogram, dtype=np.float64) for s in sigs]),
        axis=0,
    )
    hist = _l1_normalize(hist)
    zones = ("shadows", "midtones", "highlights")
    palette: list[dict] = []
    for zone in zones:
        ls, as_, bs, ws = [], [], [], []
        for sig in sigs:
            for swatch in sig.palette:
                if swatch.get("zone") == zone and float(swatch.get("weight") or 0) > 0:
                    ls.append(float(swatch["l"]))
                    as_.append(float(swatch["a"]))
                    bs.append(float(swatch["b"]))
                    ws.append(float(swatch["weight"]))
        if not ws:
            continue
        w = np.asarray(ws, dtype=np.float64)
        w_sum = float(w.sum())
        palette.append(
            {
                "zone": zone,
                "l": float(np.average(ls, weights=w)),
                "a": float(np.average(as_, weights=w)),
                "b": float(np.average(bs, weights=w)),
                "weight": w_sum / len(sigs),
            }
        )
    return ColorSignature(
        histogram=hist.tolist(),
        palette=palette,
        mean_l=float(np.mean([s.mean_l for s in sigs])),
        std_l=float(np.mean([s.std_l for s in sigs])),
        mean_a=float(np.mean([s.mean_a for s in sigs])),
        mean_b=float(np.mean([s.mean_b for s in sigs])),
    )


def encode_query(query: str) -> Optional[ColorSignature]:
    """Map query text to a color probe, or None when there is no color language."""
    matches = _lexicon_matches(query)
    if not matches:
        return None
    swatches: list[LabSwatch] = []
    chroma_scale = 1.0
    l_shift = 0.0
    for entry in matches:
        chroma_scale *= entry.chroma_scale
        l_shift += entry.l_shift
        swatches.extend(entry.swatches)
    if not swatches:
        return None
    scaled = [
        LabSwatch(
            l=float(np.clip(s.l + l_shift, 0.0, 100.0)),
            a=s.a * chroma_scale,
            b=s.b * chroma_scale,
            zone=s.zone,
        )
        for s in swatches
    ]
    hist = _synthesize_histogram(scaled)
    palette = _palette_from_swatches(scaled)
    Ls = np.array([s.l for s in scaled], dtype=np.float64)
    As = np.array([s.a for s in scaled], dtype=np.float64)
    Bs = np.array([s.b for s in scaled], dtype=np.float64)
    return ColorSignature(
        histogram=hist,
        palette=palette,
        mean_l=float(Ls.mean()),
        std_l=float(Ls.std()) if len(Ls) > 1 else 15.0,
        mean_a=float(As.mean()),
        mean_b=float(Bs.mean()),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def _rgb_from_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    if not image_bytes:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as e:
        logger.warning("Failed to decode image for color signature: %s", e)
        return None
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    h, w = rgb.shape[:2]
    max_side = max(h, w)
    if max_side > 336:
        scale = 336 / max_side
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        with Image.fromarray(rgb) as resized:
            rgb = np.asarray(resized.resize(new_size, Image.Resampling.BILINEAR), dtype=np.uint8)
    return rgb


def _weighted_ab_histogram(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> list[float]:
    a_idx = np.clip(
        ((a - AB_MIN) / (AB_MAX - AB_MIN) * HIST_BINS).astype(np.int32),
        0,
        HIST_BINS - 1,
    )
    b_idx = np.clip(
        ((b - AB_MIN) / (AB_MAX - AB_MIN) * HIST_BINS).astype(np.int32),
        0,
        HIST_BINS - 1,
    )
    flat = a_idx * HIST_BINS + b_idx
    hist = np.bincount(flat, weights=weights, minlength=HIST_DIM).astype(np.float64)
    return _l1_normalize(hist).tolist()


def _split_tone_palette(
    L: np.ndarray, a: np.ndarray, b: np.ndarray, weights: np.ndarray
) -> list[dict]:
    zones = (
        ("shadows", L < SHADOW_L),
        ("midtones", (L >= SHADOW_L) & (L <= HIGHLIGHT_L)),
        ("highlights", L > HIGHLIGHT_L),
    )
    total_w = float(weights.sum()) or 1.0
    palette: list[dict] = []
    for name, mask in zones:
        if not np.any(mask):
            continue
        w = weights[mask]
        w_sum = float(w.sum())
        if w_sum <= 1e-9:
            continue
        palette.append(
            {
                "zone": name,
                "l": float(np.average(L[mask], weights=w)),
                "a": float(np.average(a[mask], weights=w)),
                "b": float(np.average(b[mask], weights=w)),
                "weight": w_sum / total_w,
            }
        )
    return palette


def _l1_normalize(hist: np.ndarray) -> np.ndarray:
    total = float(hist.sum())
    if total <= 1e-12:
        out = np.zeros(HIST_DIM, dtype=np.float64)
        # Neutral / B&W fallback: all mass at a=0, b=0.
        mid = (HIST_BINS // 2) * HIST_BINS + (HIST_BINS // 2)
        out[mid] = 1.0
        return out
    return hist / total


def _coerce_histogram(value) -> Optional[list[float]]:
    if value is None:
        return None
    hist = [float(x) for x in value]
    if len(hist) != HIST_DIM:
        return None
    return hist


def _synthesize_histogram(swatches: list[LabSwatch]) -> list[float]:
    hist = np.zeros(HIST_DIM, dtype=np.float64)
    sigma = QUERY_SIGMA_LAB
    a_centers = AB_MIN + (np.arange(HIST_BINS) + 0.5) * (AB_MAX - AB_MIN) / HIST_BINS
    b_centers = a_centers
    aa, bb = np.meshgrid(a_centers, b_centers, indexing="ij")
    for swatch in swatches:
        bump = np.exp(
            -((aa - swatch.a) ** 2 + (bb - swatch.b) ** 2) / (2.0 * sigma * sigma)
        )
        hist += bump.ravel()
    return _l1_normalize(hist).tolist()


def _palette_from_swatches(swatches: list[LabSwatch]) -> list[dict]:
    by_zone: dict[str, list[LabSwatch]] = {}
    for s in swatches:
        by_zone.setdefault(s.zone, []).append(s)
    n = max(len(swatches), 1)
    palette = []
    for zone, items in by_zone.items():
        palette.append(
            {
                "zone": zone,
                "l": float(np.mean([s.l for s in items])),
                "a": float(np.mean([s.a for s in items])),
                "b": float(np.mean([s.b for s in items])),
                "weight": len(items) / n,
            }
        )
    return palette


@dataclass(frozen=True)
class _LexiconEntry:
    phrases: tuple[str, ...]
    swatches: tuple[LabSwatch, ...]
    chroma_scale: float = 1.0
    l_shift: float = 0.0
    fires: bool = True


def _entry(phrases, swatches, *, chroma_scale=1.0, l_shift=0.0, fires=True) -> _LexiconEntry:
    return _LexiconEntry(
        phrases=tuple(phrases),
        swatches=tuple(swatches),
        chroma_scale=chroma_scale,
        l_shift=l_shift,
        fires=fires,
    )


_LEXICON: tuple[_LexiconEntry, ...] = (
    _entry(
        ("teal and orange", "teal-and-orange", "orange and teal", "teal orange"),
        (
            LabSwatch(32, -18, -14, "shadows"),
            LabSwatch(68, 28, 48, "highlights"),
        ),
    ),
    _entry(("golden hour", "goldenhour"), (LabSwatch(62, 18, 52, "highlights"),)),
    _entry(
        ("bleach bypass", "bleach-bypass"),
        (LabSwatch(45, 2, 2, "midtones"),),
        chroma_scale=0.35,
    ),
    _entry(
        ("black and white", "black & white", "b&w", "b and w", "monochrome", "grayscale", "greyscale"),
        (LabSwatch(50, 0, 0, "midtones"),),
        chroma_scale=0.05,
    ),
    _entry(("high contrast",), (), l_shift=0.0, fires=False),
    _entry(("teal", "cyan"), (LabSwatch(42, -22, -16, "shadows"),)),
    _entry(("orange",), (LabSwatch(65, 32, 52, "highlights"),)),
    _entry(("amber",), (LabSwatch(62, 22, 58, "highlights"),)),
    _entry(("tungsten", "incandescent"), (LabSwatch(52, 16, 32, "midtones"),)),
    _entry(("daylight", "day light"), (LabSwatch(68, -6, -12, "midtones"),)),
    _entry(("magenta",), (LabSwatch(52, 55, -18, "midtones"),)),
    _entry(("pink",), (LabSwatch(72, 38, 8, "highlights"),)),
    _entry(("red",), (LabSwatch(48, 62, 42, "midtones"),)),
    _entry(("yellow",), (LabSwatch(82, -8, 72, "highlights"),)),
    _entry(("green",), (LabSwatch(52, -42, 38, "midtones"),)),
    _entry(("blue",), (LabSwatch(38, 18, -48, "shadows"),)),
    _entry(("purple", "violet"), (LabSwatch(40, 42, -38, "midtones"),)),
    _entry(("gold", "golden"), (LabSwatch(72, 8, 62, "highlights"),)),
    _entry(("sepia",), (LabSwatch(52, 12, 28, "midtones"),)),
    _entry(("warm",), (LabSwatch(58, 14, 28, "midtones"),)),
    _entry(("cool", "cold"), (LabSwatch(52, -8, -18, "midtones"),)),
    _entry(("neon",), (LabSwatch(60, 40, -20, "midtones"),), chroma_scale=1.6),
    _entry(("pastel",), (LabSwatch(78, 12, 10, "highlights"),), chroma_scale=0.55),
    _entry(("saturated", "saturation"), (), chroma_scale=1.45, fires=False),
    _entry(("faded", "washed out", "washed-out"), (), chroma_scale=0.55, l_shift=8.0, fires=False),
    _entry(("crushed", "crushed blacks"), (), l_shift=-12.0, fires=False),
)


def _lexicon_matches(query: str) -> list[_LexiconEntry]:
    text = " ".join((query or "").lower().split())
    if not text:
        return []
    occupied: list[tuple[int, int]] = []
    matched: list[_LexiconEntry] = []
    # Longest phrase first so "teal and orange" wins over "teal" + "orange".
    ranked = sorted(_LEXICON, key=lambda e: max(len(p) for p in e.phrases), reverse=True)
    for entry in ranked:
        hit = False
        for phrase in entry.phrases:
            for m in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
                span = (m.start(), m.end())
                if any(not (span[1] <= s or span[0] >= e) for s, e in occupied):
                    continue
                occupied.append(span)
                hit = True
                break
            if hit:
                break
        if hit:
            matched.append(entry)
    if not any(e.fires and e.swatches for e in matched):
        return []
    return matched

"""M3.7.3: מדדי האיכות (`fixpic.quality`) מבחינים בין מילוי תקין לבין צל מטושטש, תפרים והפרש צבע בגבול, על תמונות סינתטיות."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from fixpic.errors import EmptyMaskError, InvalidInputError
from fixpic.quality import (QualityReport, boundary_color_diff, measure, seam_ratio, stripe_score,
                            texture_ratio)

H, W = 900, 1200
BOX = (300, 250, 900, 700)  # x0, y0, x1, y1


def background(h=H, w=W, seed=1):
    """רקע "טבעי": מעבר צבע איטי + רעש קורלטיבי (מקור סטטיסטי אחד, כדי שהמילוי התקין יהיה זהה סטטיסטית לסביבה)."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    base = 100 + 40 * np.sin(xx / 170) + 30 * np.cos(yy / 230)
    n = cv2.GaussianBlur(r.normal(0, 7, (h, w)).astype(np.float32), (0, 0), 1.0) * 2.0
    return np.clip(np.stack([base + n + i * 10 for i in range(3)], -1), 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def mask():
    m = np.zeros((H, W), np.uint8)
    x0, y0, x1, y1 = BOX
    m[y0:y1, x0:x1] = 255
    return m


@pytest.fixture(scope="module", params=[1, 2, 3])
def good(request):
    return background(seed=request.param)


def shifted(img, mask, delta, region=None):
    out = img.copy()
    sel = mask > 0
    if region is not None:
        x0, y0, x1, y1 = region
        sel = np.zeros_like(sel)
        sel[y0:y1, x0:x1] = True
    out[sel] = np.clip(out[sel].astype(int) + delta, 0, 255).astype(np.uint8)
    return out


def smudged(img, mask, sigma):
    out = img.copy()
    out[mask > 0] = cv2.GaussianBlur(img, (0, 0), sigma)[mask > 0]
    return out


# ---------------------------------------------------------------- צל מטושטש

def test_texture_ratio_good_fill_is_about_one(good, mask):
    ratio, rin, rout = texture_ratio(good, mask)
    assert 0.9 < ratio < 1.1
    assert rin > 1 and rout > 1


@pytest.mark.parametrize("sigma", [3, 8])
def test_texture_ratio_detects_smudge(good, mask, sigma):
    ratio, rin, rout = texture_ratio(smudged(good, mask, sigma), mask)
    assert ratio < 0.4
    assert rin < rout


def test_texture_ratio_excess_noise_above_one(good, mask):
    out = good.copy()
    noise = np.random.default_rng(9).normal(0, 25, good.shape)
    out[mask > 0] = np.clip(out[mask > 0] + noise[mask > 0], 0, 255).astype(np.uint8)
    assert texture_ratio(out, mask)[0] > 1.8


def test_texture_ratio_flat_both_sides_is_one(mask):
    flat = np.full((H, W, 3), 120, np.uint8)
    assert texture_ratio(flat, mask)[0] == pytest.approx(1.0, abs=1e-6)


def test_texture_ratio_explicit_ring_and_small_mask(good):
    m = np.zeros((H, W), np.uint8)
    m[400:412, 500:530] = 255  # אזור צר: הפנים נופל חזרה למסכה עצמה
    r1 = texture_ratio(good, m)[0]
    r2 = texture_ratio(good, m, ring_px=12)[0]
    assert 0.5 < r1 < 1.6 and 0.5 < r2 < 1.6


# ---------------------------------------------------------------- פסים / תפרים

def test_stripe_score_low_without_seam(good, mask):
    assert stripe_score(good, mask) < 6


@pytest.mark.parametrize("delta", [6, 14])
def test_stripe_score_detects_vertical_seam(good, mask, delta):
    assert stripe_score(shifted(good, mask, delta, (600, 250, 900, 700)), mask) > 15


def test_stripe_score_detects_horizontal_seam(good, mask):
    assert stripe_score(shifted(good, mask, 12, (300, 480, 900, 700)), mask) > 15


def test_stripe_score_ignores_mask_edges_and_smudge(good, mask):
    assert stripe_score(smudged(good, mask, 8), mask) < 7


def test_stripe_score_tiny_mask_is_zero(good):
    m = np.zeros((H, W), np.uint8)
    m[100:110, 100:110] = 255
    assert stripe_score(good, m) == 0.0


def test_seam_ratio_known_lines(good, mask):
    with_seam = shifted(good, mask, 14, (600, 250, 900, 700))
    v = [("v", 600, 250, 700)]
    assert seam_ratio(with_seam, v) > 3
    assert seam_ratio(good, v) < 1.5
    # קו אופקי על תמונה עם תפר אופקי
    hs = shifted(good, mask, 14, (300, 480, 900, 700))
    assert seam_ratio(hs, [("h", 480, 300, 900)]) > 3
    # קו שנבחר במקום אחר (בלי תפר) לא מדווח
    assert seam_ratio(with_seam, [("v", 450, 250, 700)]) < 1.5


def test_seam_ratio_errors(good):
    with pytest.raises(InvalidInputError):
        seam_ratio(good, [])
    with pytest.raises(InvalidInputError):
        seam_ratio(good, [("v", 5000, 0, 10)])
    with pytest.raises(InvalidInputError):
        seam_ratio(good, [("x", 5, 0, 10)])


# ---------------------------------------------------------------- הפרש צבע בגבול

def test_boundary_color_diff_ordering(good, mask):
    base = boundary_color_diff(good, mask)[0]
    d8 = boundary_color_diff(shifted(good, mask, 8), mask)[0]
    d25 = boundary_color_diff(shifted(good, mask, 25), mask)[0]
    assert base < 4
    assert base + 3 < d8 < base + 10
    assert d25 > base + 18
    assert d25 > d8 > base


def test_boundary_color_diff_p95_at_least_mean(good, mask):
    mean, p95 = boundary_color_diff(shifted(good, mask, 12), mask)
    assert p95 >= mean


def test_boundary_color_diff_ignores_texture_loss(good, mask):
    """צל מטושטש אינו הפרש צבע: המדד לא אמור להגיב לו (ההפרדה בין המדדים)."""
    assert abs(boundary_color_diff(smudged(good, mask, 8), mask)[0] - boundary_color_diff(good, mask)[0]) < 1.0


# ---------------------------------------------------------------- מקבץ, קלט, ביצועים

def test_measure_separates_the_three_symptoms(good, mask):
    ok = measure(good, mask)
    sm = measure(smudged(good, mask, 8), mask)
    se = measure(shifted(good, mask, 14, (600, 250, 900, 700)), mask)
    co = measure(shifted(good, mask, 25), mask)
    assert isinstance(ok, QualityReport)
    assert ok.texture_ratio > 0.9 and ok.stripe_score < 6 and ok.boundary_color_diff < 4
    assert sm.texture_ratio < 0.4 and sm.stripe_score < 7
    assert se.stripe_score > 15 and se.texture_ratio > 0.9
    assert co.boundary_color_diff > 20 and co.stripe_score < 6 and co.texture_ratio > 0.9
    assert set(ok.as_dict()) == {"texture_ratio", "texture_rms_inside", "texture_rms_ring", "stripe_score",
                                 "boundary_color_diff", "boundary_color_diff_p95"}


def test_input_validation(good, mask):
    with pytest.raises(EmptyMaskError):
        measure(good, np.zeros((H, W), np.uint8))
    with pytest.raises(InvalidInputError):
        measure(good, mask[:-1])
    with pytest.raises(InvalidInputError):
        measure(good.astype(np.float32), mask)
    with pytest.raises(InvalidInputError):
        measure(good[..., :1], mask)


def test_full_image_mask_has_no_surroundings(good):
    with pytest.raises(InvalidInputError):
        texture_ratio(good, np.full((H, W), 255, np.uint8))
    with pytest.raises(InvalidInputError):
        boundary_color_diff(good, np.full((H, W), 255, np.uint8))


def test_mask_touching_image_edge_works(good):
    m = np.zeros((H, W), np.uint8)
    m[0:300, 0:400] = 255
    r = measure(good, m)
    assert 0.8 < r.texture_ratio < 1.2


def test_large_mask_in_24mp_image_is_fast():
    """מסכה של 6MP בתמונת 24MP: הכול על התיבה (כ-0.9 שנייה בסביבת הפיתוח; נמדד 9 שניות לפני מעבר למרחקים והקטנה)."""
    big = background(4000, 6000, 1)
    m = np.zeros((4000, 6000), np.uint8)
    m[500:3000, 1000:3400] = 255
    t = time.perf_counter()
    r = measure(big, m)
    assert time.perf_counter() - t < 4.0
    assert 0.9 < r.texture_ratio < 1.1 and r.stripe_score < 6

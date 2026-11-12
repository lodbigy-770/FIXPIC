import cv2
import numpy as np
import pytest

from fixpic.pipeline import Pipeline, PipelineConfig
from fixpic.pipeline.feather import feather_alpha
from fixpic.pipeline.grouping import group_mask
from fixpic.pipeline.roi import tile_cores

from test_engine import FILL, FakeModel, rnd


def blocks(h, w, *rects):
    m = np.zeros((h, w), np.uint8)
    for y0, y1, x0, x1 in rects:
        m[y0:y1, x0:x1] = 255
    return m


# ---------------------------------------------------------------- grouping
def test_far_components_are_separate_groups_and_near_are_merged():
    far = (blocks(2000, 2000, (100, 140, 100, 140), (1800, 1840, 1800, 1840)) > 0).astype(np.uint8)
    assert len(group_mask(far, 384)) == 2
    near = (blocks(2000, 2000, (100, 140, 100, 140), (100, 140, 190, 230)) > 0).astype(np.uint8)
    g = group_mask(near, 384)
    assert len(g) == 1 and g[0].mask.sum() == near.sum()


def test_group_masks_partition_the_mask():
    m = (blocks(1500, 1500, (10, 40, 10, 40), (700, 740, 700, 740), (1400, 1440, 1300, 1340)) > 0).astype(np.uint8)
    gs = group_mask(m, 256)
    assert len(gs) == 3 and sum(int(g.mask.sum()) for g in gs) == int(m.sum())


def test_many_components_fall_back_to_single_group():
    m = np.zeros((600, 600), np.uint8)
    m[::4, ::4] = 1
    assert len(group_mask(m, 256, max_components=100)) == 1


def test_pipeline_far_masks_use_small_windows():
    img = rnd(3000, 4000)
    mask = blocks(3000, 4000, (100, 160, 100, 160), (2800, 2860, 3800, 3860))
    fm = FakeModel()
    res = Pipeline(fm, PipelineConfig(feather_px=0)).run(img, mask, 256)
    assert res.groups == 2 and res.model_calls == 2 and not any(r.downscaled for r in res.rois)
    assert all(s == ((256, 256, 3), (256, 256)) for s in fm.shapes)
    eff = res.effective_mask.astype(bool)
    assert (res.image[eff] == FILL).all() and np.array_equal(res.image[~eff], img[~eff])


# ----------------------------------------------------------------- tiling
def test_tile_cores_cover_box_and_fit_window():
    box = (50, 70, 1450, 1170)
    cores = tile_cores(box, 512, 64)
    cover = np.zeros((1170, 1450), np.uint8)
    for x0, y0, x1, y1 in cores:
        assert x1 - x0 + 128 <= 512 and y1 - y0 + 128 <= 512
        cover[y0:y1, x0:x1] += 1
    assert (cover[70:1170, 50:1450] == 1).all()


@pytest.mark.parametrize("hw", [(1800, 2400), (700, 3000)])
def test_tiling_fills_whole_mask_exactly_and_keeps_shapes(hw):
    h, w = hw
    img = rnd(h, w)
    mask = blocks(h, w, (h // 10, h - h // 10, w // 10, w - w // 10))
    fm = FakeModel()
    cfg = PipelineConfig(large_strategy="tile", feather_px=0, mask_dilate_px=0, max_tiles=500)
    res = Pipeline(fm, cfg).run(img, mask, 512)
    assert res.model_calls > 4 and all(not r.downscaled for r in res.rois)
    assert all(s == ((512, 512, 3), (512, 512)) for s in fm.shapes)
    eff = res.effective_mask.astype(bool)
    assert (res.image[eff] == FILL).all()                    # בלי חורים בין אריחים
    assert np.array_equal(res.image[~eff], img[~eff])


def test_auto_strategy_downscale_vs_tile():
    img = rnd(2000, 3000)
    mild = blocks(2000, 3000, (500, 1000, 500, 1000))        # scale ~0.5+ : הקטנה
    r1 = Pipeline(FakeModel(), PipelineConfig(feather_px=0)).run(img, mild, 512)
    assert r1.model_calls == 1 and r1.rois[0].downscaled
    huge = blocks(2000, 3000, (100, 1900, 100, 2900))        # scale קטן: אריחים
    r2 = Pipeline(FakeModel(), PipelineConfig(feather_px=0)).run(img, huge, 512)
    assert r2.model_calls > 1 and not r2.rois[0].downscaled


def test_max_tiles_cap_falls_back_to_downscale():
    img = rnd(2000, 3000)
    huge = blocks(2000, 3000, (100, 1900, 100, 2900))
    r = Pipeline(FakeModel(), PipelineConfig(large_strategy="tile", max_tiles=4)).run(img, huge, 512)
    assert r.model_calls == 1 and r.rois[0].downscaled


def test_tiles_see_previously_filled_context():
    """אריח מאוחר רואה פיקסלים שמולאו באריח קודם (רציפות), ולא את האובייקט המקורי."""
    seen = []

    class Spy(FakeModel):
        def run(self, image, mask, cancel=None):
            seen.append(int((image == FILL).all(axis=2).sum()))
            return super().run(image, mask, cancel)

    img = rnd(1200, 1800)
    mask = blocks(1200, 1800, (100, 1100, 100, 1700))
    Pipeline(Spy(), PipelineConfig(large_strategy="tile", feather_px=0, mask_dilate_px=0, max_tiles=200)).run(img, mask, 512)
    assert seen[0] == 0 and max(seen) > 0


# ---------------------------------------------------------------- feather
def test_feather_alpha_bounds_and_zero_outside():
    m = (blocks(200, 200, (50, 150, 50, 150)) > 0).astype(np.uint8)
    a = feather_alpha(m, 5)
    assert a.dtype == np.float32 and a.min() == 0 and a.max() == 1
    assert (a[m == 0] == 0).all() and (a[60:140, 60:140] == 1).all()
    assert 0 < a[50, 100] < 1


def test_feather_thin_line_center_fully_replaced():
    m = (blocks(100, 200, (50, 53, 10, 190)) > 0).astype(np.uint8)       # עובי 3
    a = feather_alpha(m, 8)
    assert a[51, 100] == 1.0 and (a[m == 0] == 0).all()


def test_feather_off_equals_mask():
    m = (blocks(50, 50, (10, 20, 10, 20)) > 0).astype(np.uint8)
    assert np.array_equal(feather_alpha(m, 0), m.astype(np.float32))


def test_feather_pipeline_outside_identical_and_rim_blended():
    img = rnd(400, 400)
    mask = blocks(400, 400, (150, 250, 150, 250))
    res = Pipeline(FakeModel(), PipelineConfig(mask_dilate_px=0, feather_px=4)).run(img, mask, 384)
    eff = res.effective_mask.astype(bool)
    assert np.array_equal(res.image[~eff], img[~eff])
    assert (res.image[170:230, 170:230] == FILL).all()
    rim = res.image[150, 200]
    assert not np.array_equal(rim, FILL) and not np.array_equal(rim, img[150, 200])

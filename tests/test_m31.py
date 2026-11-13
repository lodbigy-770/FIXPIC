"""M3.1: ניהול תמונות (טעינה/שמירה, נתיבי עברית, Alpha, EXIF/ICC, שמירה אטומית). ללא Qt."""
from __future__ import annotations

import io
import os
import unicodedata
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps

from fixpic.errors import FixPicError, ImageIOError, InvalidInputError
from fixpic.imgio import (MAX_PIXELS, composite_on, decode_image, encode_image, image_from_array, load_image,
                          save_image)
from fixpic.imgio import images as images_mod


def rnd(h=40, w=60, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def pil_bytes(img: Image.Image, fmt: str, **kw) -> bytes:
    b = io.BytesIO()
    img.save(b, fmt, **kw)
    return b.getvalue()


def srgb_icc() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


# ------------------------------------------------------------------------------------------ מעגל שמירה/טעינה
def test_png_roundtrip_exact(tmp_path):
    a = rnd()
    save_image(tmp_path / "a.png", a)
    li = load_image(tmp_path / "a.png")
    assert np.array_equal(li.rgb, a) and li.alpha is None and li.source_format == "PNG"
    assert li.rgb.dtype == np.uint8 and li.rgb.flags.c_contiguous and li.rgb.flags.writeable


def test_webp_lossless_roundtrip_exact(tmp_path):
    a = rnd()
    r = save_image(tmp_path / "a.webp", a, lossless=True)
    assert r.format == "WEBP" and np.array_equal(load_image(tmp_path / "a.webp").rgb, a)


@pytest.mark.parametrize("ext,fmt", [(".jpg", "JPEG"), (".JPEG", "JPEG"), (".webp", "WEBP")])
def test_lossy_roundtrip_close(tmp_path, ext, fmt):
    yy, xx = np.mgrid[0:64, 0:96]
    a = np.stack([xx * 2, yy * 3, (xx + yy)], -1).clip(0, 255).astype(np.uint8)   # גרדיאנט חלק
    r = save_image(tmp_path / f"a{ext}", a, quality=95)
    assert r.format == fmt
    back = load_image(tmp_path / f"a{ext}").rgb
    assert back.shape == a.shape and np.abs(back.astype(int) - a).mean() < 3


def test_quality_changes_size(tmp_path):
    a = rnd(120, 160)
    lo = save_image(tmp_path / "lo.jpg", a, quality=20).bytes_written
    hi = save_image(tmp_path / "hi.jpg", a, quality=95).bytes_written
    assert lo < hi


@pytest.mark.parametrize("q", [0, 101, -5])
def test_quality_range_checked(tmp_path, q):
    with pytest.raises(InvalidInputError):
        save_image(tmp_path / "a.jpg", rnd(), quality=q)
    assert not (tmp_path / "a.jpg").exists()


# ------------------------------------------------------------------------------------------ נתיבים בעברית
@pytest.mark.parametrize("folder,name", [
    ("תיקייה בעברית", "תמונה.png"),
    ("תיקייה עם רווחים", "שם עם   רווחים ו-מקף.jpg"),
    ("נקודות.ושם", "שלום עולם 😀.webp"),
    (unicodedata.normalize("NFD", "פֶּרֶק"), unicodedata.normalize("NFD", "קֶבֶץ.png")),   # ניקוד/צירופים
])
def test_unicode_paths_roundtrip(tmp_path, folder, name):
    d = tmp_path / folder
    d.mkdir()
    a = rnd()
    r = save_image(d / name, a, lossless=True)
    assert r.path == d / name and (d / name).exists()
    assert load_image(d / name).size == (60, 40)
    assert sorted(x.name for x in d.iterdir()) == [name]       # בלי קובץ זמני שנשאר


def test_accepts_str_and_pathlike(tmp_path):
    save_image(str(tmp_path / "א.png"), rnd())
    assert load_image(str(tmp_path / "א.png")).size == (60, 40)


# ------------------------------------------------------------------------------------------ Alpha
@pytest.mark.parametrize("name", ["a.png", "a.webp"])
def test_alpha_roundtrip(tmp_path, name):
    a = rnd()
    al = np.linspace(0, 255, 40 * 60).reshape(40, 60).astype(np.uint8)
    save_image(tmp_path / name, a, al, lossless=True)
    li = load_image(tmp_path / name)
    assert li.alpha is not None and np.array_equal(li.alpha, al)
    assert li.alpha.flags.c_contiguous and li.rgb.shape == (40, 60, 3)
    # WebP lossless שומר גם RGB מוסתר רק בחלק מהמימושים; בודקים היכן שהתמונה גלויה לגמרי
    vis = al == 255
    assert np.array_equal(li.rgb[vis], a[vis])


def test_png_alpha_keeps_hidden_rgb(tmp_path):
    a, al = rnd(), np.zeros((40, 60), np.uint8)
    save_image(tmp_path / "a.png", a, al)
    li = load_image(tmp_path / "a.png")
    assert np.array_equal(li.rgb, a) and np.array_equal(li.alpha, al)


def test_fully_opaque_alpha_is_dropped(tmp_path):
    a = rnd()
    save_image(tmp_path / "a.png", a, np.full((40, 60), 255, np.uint8))
    assert load_image(tmp_path / "a.png").alpha is None


def test_jpeg_flattens_alpha_on_white(tmp_path):
    a = np.full((8, 8, 3), 100, np.uint8)
    al = np.zeros((8, 8), np.uint8)
    al[:, 4:] = 255
    r = save_image(tmp_path / "a.jpg", a, al, quality=100)
    assert r.alpha_flattened
    back = load_image(tmp_path / "a.jpg").rgb
    assert back[:, :4].mean() > 245 and abs(back[:, 4:].mean() - 100) < 4


def test_composite_on_values():
    rgb = np.array([[[0, 0, 0], [200, 100, 50], [10, 20, 30]]], np.uint8)
    al = np.array([[0, 255, 128]], np.uint8)
    out = composite_on(rgb, al)
    assert out[0, 0].tolist() == [255, 255, 255] and out[0, 1].tolist() == [200, 100, 50]
    assert all(abs(int(out[0, 2, c]) - ((int(rgb[0, 2, c]) * 128 + 255 * 127 + 127) // 255)) <= 1 for c in range(3))
    assert composite_on(rgb, None) is rgb


def test_palette_png_with_transparency(tmp_path):
    p = Image.new("P", (10, 10))
    p.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    p.putpixel((0, 0), 1)
    p.info["transparency"] = 0
    (tmp_path / "p.png").write_bytes(pil_bytes(p, "PNG", transparency=0))
    li = load_image(tmp_path / "p.png")
    assert li.alpha is not None and li.alpha[5, 5] == 0 and li.alpha[0, 0] == 255


# ------------------------------------------------------------------------------------------ מצבי צבע ועומק
def test_grayscale_and_1bit_become_rgb(tmp_path):
    g = Image.fromarray((np.arange(100).reshape(10, 10) * 2).astype(np.uint8))
    li = decode_image(pil_bytes(g, "PNG"))
    assert li.rgb.shape == (10, 10, 3) and np.array_equal(li.rgb[..., 0], li.rgb[..., 2])
    bw = Image.fromarray(np.tri(8, 8, dtype=bool))
    assert decode_image(pil_bytes(bw, "PNG")).rgb.shape == (8, 8, 3)


def test_16bit_gray_png_reduced_to_8(tmp_path):
    a16 = np.array([[0, 257, 32768, 65535]], np.uint16)
    li = decode_image(pil_bytes(Image.fromarray(a16), "PNG"))
    assert li.depth_reduced and li.rgb.shape == (1, 4, 3)
    assert li.rgb[0, :, 0].tolist() == [0, 1, 128, 255]


def test_cmyk_jpeg_loads_as_rgb_without_icc():
    c = Image.new("CMYK", (16, 16), (0, 255, 255, 0))
    li = decode_image(pil_bytes(c, "JPEG", icc_profile=srgb_icc()))
    assert li.rgb.shape == (16, 16, 3) and li.icc_profile is None
    assert li.rgb[..., 0].mean() > li.rgb[..., 2].mean()        # אדום, לא כחול


def test_icc_dropped_for_gray_kept_for_rgb():
    g = Image.new("L", (8, 8), 90)
    assert decode_image(pil_bytes(g, "PNG", icc_profile=srgb_icc())).icc_profile is None
    assert decode_image(pil_bytes(Image.new("RGB", (8, 8)), "PNG", icc_profile=srgb_icc())).icc_profile


# ------------------------------------------------------------------------------------------ EXIF / ICC
def exif_with(orientation: int, **extra) -> Image.Exif:
    ex = Image.Exif()
    ex[0x0112] = orientation
    for k, v in extra.items():
        ex[int(k, 16)] = v
    return ex


@pytest.mark.parametrize("orient", [2, 3, 4, 5, 6, 7, 8])
def test_exif_orientation_applied_and_reset(tmp_path, orient):
    a = np.zeros((20, 30, 3), np.uint8)
    a[:5, :8] = (255, 0, 0)                      # פינה שמאלית-עליונה אדומה, כדי לראות את הסיבוב
    data = pil_bytes(Image.fromarray(a), "PNG", exif=exif_with(orient))
    li = decode_image(data)
    exp = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(data))))
    assert li.orientation_applied == orient
    assert np.array_equal(li.rgb, exp)
    assert li.size == ((20, 30) if orient in (5, 6, 7, 8) else (30, 20))
    # EXIF שחזר אינו כולל Orientation, כך ששמירה וטעינה חוזרת לא מסובבות שוב
    if li.exif:
        got = Image.Exif()
        got.load(li.exif)
        assert 0x0112 not in got
    save_image(tmp_path / "o.jpg", li.rgb, exif=li.exif, quality=100)
    assert load_image(tmp_path / "o.jpg").size == li.size


def test_exif_other_tags_preserved_through_save(tmp_path):
    ex = exif_with(1)
    ex[0x010F] = "FIXPIC-Test"                   # Make
    ex[0x0131] = "unit"                          # Software
    save_image(tmp_path / "a.jpg", rnd(), exif=ex.tobytes())
    li = load_image(tmp_path / "a.jpg")
    assert li.exif is not None
    got = Image.Exif()
    got.load(li.exif)
    assert got[0x010F] == "FIXPIC-Test" and got[0x0131] == "unit"
    for name in ("b.png", "b.webp"):
        save_image(tmp_path / name, rnd(), exif=li.exif, lossless=True)
        g2 = load_image(tmp_path / name)
        assert g2.exif is not None


@pytest.mark.parametrize("name", ["a.png", "a.jpg", "a.webp"])
def test_icc_preserved_through_save(tmp_path, name):
    icc = srgb_icc()
    save_image(tmp_path / name, rnd(), icc_profile=icc)
    assert load_image(tmp_path / name).icc_profile == icc


def test_mpo_reported_as_jpeg():
    # JPEG רגיל מזוהה כ-JPEG; MPO (טלפונים) ממופה לאותו שם
    assert decode_image(pil_bytes(Image.fromarray(rnd()), "JPEG")).source_format == "JPEG"


# ------------------------------------------------------------------------------------------ קלט פגום / לא נתמך
def test_missing_file_is_io_error(tmp_path):
    with pytest.raises(ImageIOError):
        load_image(tmp_path / "אין.png")


def test_directory_is_io_error(tmp_path):
    with pytest.raises(ImageIOError):
        load_image(tmp_path)


@pytest.mark.parametrize("payload", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, b"%PDF-1.4\n"])
def test_garbage_is_invalid_input(payload):
    with pytest.raises(InvalidInputError):
        decode_image(payload)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_truncated_file_is_invalid_input(fmt):
    data = pil_bytes(Image.fromarray(rnd(200, 300)), fmt)
    with pytest.raises(InvalidInputError):
        decode_image(data[: len(data) // 3])


def test_unsupported_format_rejected():
    with pytest.raises(InvalidInputError, match="פורמט"):
        decode_image(pil_bytes(Image.fromarray(rnd(16, 16)), "ICO", sizes=[(16, 16)]))


def test_too_many_pixels_rejected(monkeypatch):
    monkeypatch.setattr(images_mod, "MAX_PIXELS", 1000)
    with pytest.raises(InvalidInputError, match="גדולה מדי"):
        decode_image(pil_bytes(Image.fromarray(rnd(40, 60)), "PNG"))


def test_animated_gif_first_frame_only():
    f1 = Image.new("RGB", (12, 10), (255, 0, 0))
    f2 = Image.new("RGB", (12, 10), (0, 0, 255))
    b = io.BytesIO()
    f1.save(b, "GIF", save_all=True, append_images=[f2], duration=50, loop=0)
    li = decode_image(b.getvalue())
    assert li.animated and li.size == (12, 10) and li.rgb[..., 0].mean() > 200 and li.source_format == "GIF"


def test_all_errors_are_fixpic_errors(tmp_path):
    for call in (lambda: load_image(tmp_path / "x"), lambda: decode_image(b"zz"),
                 lambda: save_image(tmp_path / "x.bmp", rnd())):
        with pytest.raises(FixPicError):
            call()


# ------------------------------------------------------------------------------------------ שמירה: קלט ושגיאות
def test_unsupported_extension_rejected(tmp_path):
    with pytest.raises(InvalidInputError, match="סיומת"):
        save_image(tmp_path / "a.tiff", rnd())
    with pytest.raises(InvalidInputError):
        save_image(tmp_path / "noext", rnd())
    assert list(tmp_path.iterdir()) == []


def test_explicit_fmt_overrides_extension(tmp_path):
    r = save_image(tmp_path / "x.dat", rnd(), fmt="PNG")
    assert r.format == "PNG" and decode_image((tmp_path / "x.dat").read_bytes()).source_format == "PNG"


@pytest.mark.parametrize("bad", [
    np.zeros((4, 4), np.uint8), np.zeros((4, 4, 4), np.uint8), np.zeros((4, 4, 3), np.float32),
    np.zeros((0, 4, 3), np.uint8), [[1, 2, 3]], None,
])
def test_save_rejects_bad_rgb(tmp_path, bad):
    with pytest.raises(InvalidInputError):
        save_image(tmp_path / "a.png", bad)


def test_save_rejects_bad_alpha(tmp_path):
    for al in (np.zeros((3, 3), np.uint8), np.zeros((40, 60), np.float32), np.zeros((40, 60, 1), np.uint8)):
        with pytest.raises(InvalidInputError):
            save_image(tmp_path / "a.png", rnd(), al)


def test_webp_dimension_limit(tmp_path):
    with pytest.raises(InvalidInputError, match="16383"):
        encode_image(np.zeros((1, 16384, 3), np.uint8), fmt="WEBP")


def test_missing_parent_folder_is_io_error(tmp_path):
    with pytest.raises(ImageIOError, match="לא קיימת"):
        save_image(tmp_path / "אין" / "a.png", rnd())


def test_overwrite_existing_file(tmp_path):
    save_image(tmp_path / "a.png", rnd(seed=1))
    new = rnd(seed=2)
    save_image(tmp_path / "a.png", new)
    assert np.array_equal(load_image(tmp_path / "a.png").rgb, new)


def test_failed_replace_leaves_original_and_no_temp(tmp_path, monkeypatch):
    old = rnd(seed=1)
    save_image(tmp_path / "a.png", old)
    before = (tmp_path / "a.png").read_bytes()

    def boom(*a, **k):
        raise PermissionError("הקובץ נעול")
    monkeypatch.setattr(images_mod.os, "replace", boom)
    with pytest.raises(ImageIOError, match="נכשלה"):
        save_image(tmp_path / "a.png", rnd(seed=2))
    assert (tmp_path / "a.png").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["a.png"]


def test_failed_write_leaves_original_and_no_temp(tmp_path, monkeypatch):
    save_image(tmp_path / "a.png", rnd(seed=1))
    before = (tmp_path / "a.png").read_bytes()
    real_fsync = os.fsync

    def disk_full(fd):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(images_mod.os, "fsync", disk_full)
    with pytest.raises(ImageIOError):
        save_image(tmp_path / "a.png", rnd(seed=2))
    monkeypatch.setattr(images_mod.os, "fsync", real_fsync)
    assert (tmp_path / "a.png").read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["a.png"]


def test_encode_failure_does_not_touch_disk(tmp_path, monkeypatch):
    save_image(tmp_path / "a.png", rnd(seed=1))
    before = (tmp_path / "a.png").read_bytes()

    def bad_save(self, *a, **k):
        raise ValueError("קידוד שבור")
    monkeypatch.setattr(Image.Image, "save", bad_save)
    with pytest.raises(InvalidInputError, match="קידוד"):
        save_image(tmp_path / "a.png", rnd(seed=2))
    assert (tmp_path / "a.png").read_bytes() == before and len(list(tmp_path.iterdir())) == 1


def test_input_arrays_not_modified_by_save(tmp_path):
    a, al = rnd(), np.full((40, 60), 128, np.uint8)
    a0, al0 = a.copy(), al.copy()
    save_image(tmp_path / "a.jpg", a, al)
    save_image(tmp_path / "a.png", a, al)
    assert np.array_equal(a, a0) and np.array_equal(al, al0)


# ------------------------------------------------------------------------------------------ מערך -> תמונה (הדבקה)
def test_image_from_array_variants():
    rgb = rnd()
    assert np.array_equal(image_from_array(rgb).rgb, rgb)
    rgba = np.dstack([rgb, np.full((40, 60), 7, np.uint8)])
    li = image_from_array(rgba)
    assert np.array_equal(li.rgb, rgb) and (li.alpha == 7).all() and li.alpha.flags.c_contiguous
    assert image_from_array(np.dstack([rgb, np.full((40, 60), 255, np.uint8)])).alpha is None
    g = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert image_from_array(g).rgb.shape == (3, 4, 3) and image_from_array(g[..., None]).rgb.shape == (3, 4, 3)
    sliced = image_from_array(rgba[:, ::2])                         # קלט לא רציף
    assert sliced.rgb.flags.c_contiguous and sliced.rgb.shape == (40, 30, 3)


@pytest.mark.parametrize("bad", [np.zeros((4, 4, 2), np.uint8), np.zeros((4, 4), np.float64), np.zeros((0, 3, 3), np.uint8),
                                 np.zeros((4,), np.uint8), "x", None])
def test_image_from_array_rejects(bad):
    with pytest.raises(InvalidInputError):
        image_from_array(bad)


def test_full_cycle_load_edit_save_keeps_metadata(tmp_path):
    """תרחיש בסיס: טעינה, עריכה בזיכרון, שמירה בפורמט אחר עם ICC ו-EXIF."""
    ex = exif_with(6)
    ex[0x010F] = "מצלמה"
    src = tmp_path / "מקור.jpg"
    src.write_bytes(pil_bytes(Image.fromarray(rnd(30, 50)), "JPEG", exif=ex, icc_profile=srgb_icc()))
    li = load_image(src)
    assert li.size == (30, 50) and li.orientation_applied == 6 and li.icc_profile
    li.rgb[:5] = 0                                                   # ניתן לכתיבה
    out = tmp_path / "פלט.png"
    save_image(out, li.rgb, li.alpha, icc_profile=li.icc_profile, exif=li.exif)
    back = load_image(out)
    assert back.size == (30, 50) and back.icc_profile == li.icc_profile and back.orientation_applied == 1
    assert (back.rgb[:5] == 0).all()


def test_max_pixels_constant_matches_docs():
    assert MAX_PIXELS == 150_000_000

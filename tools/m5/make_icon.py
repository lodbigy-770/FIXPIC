"""יצירת אייקון היישום (PNG ו-ICO) מקוד, ללא קבצי מקור חיצוניים: `python tools/m5/make_icon.py`.
עיצוב: ריבוע מעוגל בגרדיאנט כחול-ירוק, וממחק לבן באלכסון עם נצנוץ (הסרת אלמנט). רנדור ב-1024 והקטנה (LANCZOS)."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parents[2] / "src" / "fixpic" / "ui" / "assets"
SIZES = (16, 24, 32, 48, 64, 128, 256)
S = 1024


def _gradient(size: int) -> Image.Image:
    top, bot = (37, 99, 235), (20, 184, 166)
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(top, bot))
    return img


def render() -> Image.Image:
    bg = _gradient(S)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S - 1, S - 1), radius=int(S * 0.22), fill=255)
    icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    icon.paste(bg, (0, 0), mask)

    # ממחק: מלבן מעוגל מסובב ב-40 מעלות, חלק עליון לבן וחלק תחתון (ידית) כהה
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w, h, cx, cy = 560, 300, S // 2, S // 2 + 20
    box = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
    d.rounded_rectangle(box, radius=60, fill=(255, 255, 255, 255))
    d.rounded_rectangle((box[0], box[1], box[0] + 230, box[3]), radius=60, fill=(251, 191, 189, 255))
    d.rectangle((box[0] + 170, box[1], box[0] + 230, box[3]), fill=(251, 191, 189, 255))
    layer = layer.rotate(40, resample=Image.BICUBIC, center=(cx, cy))
    shadow = layer.filter(ImageFilter.GaussianBlur(22))
    shadow = Image.merge("RGBA", (*(Image.new("L", (S, S), 0),) * 3, shadow.getchannel("A").point(lambda v: v * 0.35)))
    icon.alpha_composite(shadow, (0, 14))
    icon.alpha_composite(layer)

    # נצנוץ (כוכב ארבע קרניים) בפינה השמאלית-עליונה, מסמל "נוקה"
    d = ImageDraw.Draw(icon)
    sx, sy, r, k = 255, 265, 120, 28
    d.polygon([(sx, sy - r), (sx + k, sy - k), (sx + r, sy), (sx + k, sy + k),
               (sx, sy + r), (sx - k, sy + k), (sx - r, sy), (sx - k, sy - k)], fill=(255, 255, 255, 255))
    sx, sy, r, k = 780, 790, 64, 15
    d.polygon([(sx, sy - r), (sx + k, sy - k), (sx + r, sy), (sx + k, sy + k),
               (sx, sy + r), (sx - k, sy + k), (sx - r, sy), (sx - k, sy - k)], fill=(255, 255, 255, 235))
    return icon


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    big = render()
    big.resize((256, 256), Image.LANCZOS).save(OUT / "fixpic.png")
    big.save(OUT / "fixpic.ico", sizes=[(s, s) for s in SIZES])
    print("נוצרו:", OUT / "fixpic.png", OUT / "fixpic.ico")
    return 0


if __name__ == "__main__":
    sys.exit(main())

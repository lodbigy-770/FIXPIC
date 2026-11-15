#!/usr/bin/env python3
"""מוריד שני מועמדי MobileSAM (ONNX) ל-`models/spike/` (מחוץ ל-git) ומדפיס את פקודת `model_spike.py`.

הרצה מהשורש:  python tools/m4/fetch_spike_models.py
הסקריפט רק מוריד קבצים מ-Hugging Face (לפי כתובות `resolve/main`), מדפיס גודל ו-SHA256, ואינו מריץ מדידה.
**[לא אומת]** שהחוזה של המפענחים תואם ל-`model_spike.py`: אם המדידה עוצרת עם רשימת שמות קלט/פלט, שלחו אותה.
מועמדים (הרישיון לפי כרטיס המודל ב-Hugging Face; לאמת במקור לפני כל שימוש מעבר למדידה):
  vietanhdev  = `vietanhdev/segment-anything-onnx-models`, `mobile_sam_20230629.zip` (encoder.onnx + decoder.onnx), Apache-2.0
  acly        = `Acly/MobileSAM`, `mobile_sam_image_encoder.onnx` + `sam_mask_decoder_single.onnx`, MIT
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://huggingface.co/{repo}/resolve/main/{name}"

FILES = {
    "vietanhdev": ("vietanhdev/segment-anything-onnx-models", ["mobile_sam_20230629.zip"]),
    "acly": ("Acly/MobileSAM", ["mobile_sam_image_encoder.onnx", "sam_mask_decoder_single.onnx"]),
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"קיים: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"מוריד {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r  {got / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(FILES), help="להוריד מועמד אחד בלבד")
    ap.add_argument("--out", default=str(ROOT / "models" / "spike"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    pairs: dict[str, tuple[Path, Path]] = {}
    for name, (repo, files) in FILES.items():
        if a.only and a.only != name:
            continue
        d = out / name
        for f in files:
            download(BASE.format(repo=repo, name=f), d / f)
        if name == "vietanhdev":
            z = d / files[0]
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
            enc = next(d.rglob("encoder.onnx"), None)
            dec = next((p for p in d.rglob("*.onnx") if "decoder" in p.name), None)
            if enc is None or dec is None:
                print(f"לא נמצאו encoder/decoder בתוך {z}; תוכן: {[str(p) for p in d.rglob('*')]}", file=sys.stderr)
                return 2
        else:
            enc, dec = d / files[0], d / files[1]
        pairs[name] = (enc, dec)
        for p in (enc, dec):
            print(f"{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}  {p.stat().st_size / 1e6:.1f} MB  sha256={sha256(p)[:16]}…")
    cmd = " ".join(f'--candidate {n} "{e}" "{d}"' for n, (e, d) in pairs.items())
    print("\nפקודת המדידה (מהשורש):\n  python tools/m4/model_spike.py " + cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())

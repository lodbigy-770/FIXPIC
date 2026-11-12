"""Model Manager (ARCHITECTURE §11): `models/<name>/model.json` -> טעינה, מטא-דאטה, בדיקת hash.
ה-sha256 ב-model.json של Big LaMa נלקח מה-LFS pointer (MODEL_CARD)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from ..errors import ModelLoadError
from .auto import AutoModel
from .base import ModelConfig, ModelInfo

log = logging.getLogger("fixpic.models")
SCHEMA = 1


@dataclass(frozen=True)
class Manifest:
    name: str
    version: str
    license: str
    path: Path             # נתיב מלא לקובץ המודל
    sha256: str | None
    size_bytes: int | None
    info: ModelInfo

    @classmethod
    def read(cls, folder: Path) -> "Manifest":
        mj = Path(folder) / "model.json"
        if not mj.is_file():
            raise ModelLoadError(f"model.json חסר ב-{folder}")
        try:
            d = json.loads(mj.read_text(encoding="utf-8"))
            if d.get("schema") != SCHEMA:
                raise ValueError(f"schema לא נתמך: {d.get('schema')!r}")
            if not isinstance(d.get("name"), str) or not d["name"]:
                raise ValueError("name חסר או לא תקין")
            inp = d.get("input", {})
            info = ModelInfo(name=d["name"], multiple=int(inp.get("multiple", 8)),
                             min_size=int(inp.get("min_size", 64)), max_size=int(inp.get("max_size", 4096)))
            if info.multiple <= 0 or info.min_size > info.max_size:
                raise ValueError("ערכי input לא תקינים")
            file = Path(d["file"])
            if file.is_absolute() or ".." in file.parts:
                raise ValueError("שדה file חייב להיות נתיב יחסי בתוך תיקיית המודל")
            sha = d.get("sha256")
            if sha is not None and (len(sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha)):
                raise ValueError("sha256 לא תקין")
            size = d.get("size_bytes")
            return cls(d["name"], str(d.get("version", "")), str(d.get("license", "")),
                       Path(folder) / file, sha, int(size) if size is not None else None, info)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            raise ModelLoadError(f"model.json לא תקין ב-{folder}: {e}") from e


class ModelManager:
    def __init__(self, models_dir: str | Path) -> None:
        self.dir = Path(models_dir)

    def list(self) -> list[Manifest]:
        """מודלים תקינים בלבד; תיקייה פגומה נרשמת ביומן ואינה מפילה את הרשימה."""
        out = []
        for d in sorted(p for p in self.dir.iterdir() if p.is_dir()) if self.dir.is_dir() else []:
            if (d / "model.json").is_file():
                try:
                    out.append(Manifest.read(d))
                except ModelLoadError as e:
                    log.warning("מודל מדולג: %s", e)
        return out

    def manifest(self, name: str) -> Manifest:
        return Manifest.read(self.dir / name)

    def load(self, name: str, threads: int = 3, model=None) -> AutoModel:
        """טוען מודל מוכן להרצה. אימות גודל ו-hash לפני יצירת Session; כשל = ModelLoadError."""
        mf = self.manifest(name)
        if mf.size_bytes is not None and mf.path.is_file() and mf.path.stat().st_size != mf.size_bytes:
            with open(mf.path, "rb") as f:
                is_lfs = f.read(64).startswith(b"version https://git-lfs")   # הודעת LFS מפורשת נותנת check_model_file
            if not is_lfs:
                raise ModelLoadError(f"גודל קובץ המודל {mf.path.stat().st_size} שונה מהצפוי {mf.size_bytes}")
        m = model if model is not None else AutoModel(mf.info)
        m.load(ModelConfig(mf.path, threads=threads, sha256=mf.sha256))   # כולל LFS pointer ו-hash
        return m

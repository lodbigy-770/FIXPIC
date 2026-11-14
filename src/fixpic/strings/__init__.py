"""מחרוזות הממשק (עברית בלבד, ARCHITECTURE §13). המחרוזות בקובץ JSON נפרד כדי שאפשר יהיה להוסיף שפה בעתיד.

`t("מפתח", שם=ערך)` מחזיר את המחרוזת ומחיל `str.format`. מפתח חסר = KeyError (לא מחרוזת ריקה שקטה), ובדיקה
(`tests/test_m33.py`) מוודאת שכל מפתח שהקוד משתמש בו קיים."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

LANG = "he"


@lru_cache(maxsize=None)
def _load(lang: str) -> dict[str, str]:
    p = Path(__file__).with_name(f"{lang}.json")
    return json.loads(p.read_text(encoding="utf-8"))


def t(key: str, **fmt) -> str:
    try:
        s = _load(LANG)[key]
    except KeyError:
        raise KeyError(f"מחרוזת חסרה: {key}") from None
    return s.format(**fmt) if fmt else s


def all_keys() -> set[str]:
    return set(_load(LANG))

"""קיצור: `python -m fixpic.app` (ARCHITECTURE §13: app.py = נקודת הכניסה)."""
from .ui.app import create_app, main

__all__ = ["create_app", "main"]

if __name__ == "__main__":
    raise SystemExit(main())

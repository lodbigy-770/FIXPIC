import pytest


@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path, monkeypatch):
    """בדיקות לא קוראות ולא כותבות את הגדרות המשתמש האמיתיות (קיצורי מקלדת)."""
    monkeypatch.setenv("FIXPIC_CONFIG_DIR", str(tmp_path / "cfg"))

"""בחירת אזור אוטומטית (M4, AUTO-SELECT.md), ללא UI: טיפוסים, `SelectionProvider`, `MagicWandProvider`, `SelectionService`, `GrabCutProvider`, `stroke_to_points`/`with_stroke` (משיכה → נקודות)."""
from .base import SelectionProvider
from .grabcut import GrabCutProvider
from .service import RequestState, SelectionRequest, SelectionService
from .stroke import stroke_to_points, with_stroke
from .types import Context, Point, Prompts, Proposal, SelectionError
from .wand import MagicWandProvider

__all__ = ["Context", "GrabCutProvider", "MagicWandProvider", "Point", "Prompts", "Proposal", "RequestState", "SelectionError",
           "SelectionProvider", "SelectionRequest", "SelectionService", "stroke_to_points", "with_stroke"]

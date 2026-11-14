"""בחירת אזור אוטומטית (M4, AUTO-SELECT.md), ללא UI: טיפוסים, `SelectionProvider`, `MagicWandProvider`, `SelectionService`."""
from .base import SelectionProvider
from .service import RequestState, SelectionRequest, SelectionService
from .types import Context, Point, Prompts, Proposal, SelectionError
from .wand import MagicWandProvider

__all__ = ["Context", "MagicWandProvider", "Point", "Prompts", "Proposal", "RequestState", "SelectionError",
           "SelectionProvider", "SelectionRequest", "SelectionService"]

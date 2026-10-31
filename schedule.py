"""Risk ranking and spaced 7-day scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

SEVERITY = {"conceptual": 3, "procedural": 2, "careless": 1, "unreadable": 0}
DAYS = (1, 3, 7)


def recency_weight(days_since: int) -> float:
    if days_since <= 3:
        return 1.0
    if days_since <= 7:
        return 0.8
    return 0.5


def risk(repeat_count: int, days_since: int, error_type: str, cleared_count: int = 0) -> float:
    base = max(1, int(repeat_count)) * recency_weight(int(days_since)) * SEVERITY.get(error_type, 1)
    return round(base * (0.5 ** int(cleared_count)), 4)


@dataclass
class Gap:
    knowledge_point: str
    error_type: str = "procedural"
    repeat_count: int = 1
    days_since: int = 0
    cleared_count: int = 0
    items: List[Dict[str, str]] = field(default_factory=list)

    @property
    def risk_score(self) -> float:
        return risk(self.repeat_count, self.days_since, self.error_type, self.cleared_count)


def rank_gaps(gaps: List[Gap]) -> List[Gap]:
    """Highest risk first; conceptual errors outrank careless ones at equal risk."""
    return sorted(gaps, key=lambda g: (-g.risk_score, -SEVERITY.get(g.error_type, 0), g.knowledge_point))


def initial_path(items: List[Dict[str, str]]) -> List[Dict[str, object]]:
    """A single fresh mistake: one item per spaced day, never more."""
    path = []
    for day, item in zip(DAYS, items[: len(DAYS)]):
        if item:
            path.append({"day": day, "item_id": item.get("id"), "text": item.get("text"), "difficulty": item.get("difficulty")})
    return path


def build_path(gaps: List[Gap], max_per_day: int = 3, today: Optional[date] = None) -> List[Dict[str, object]]:
    """Rank gaps, spread them across days 1/3/7, cap the daily load."""
    today = today or date.today()
    ranked = rank_gaps(gaps)
    slots: Dict[int, List[Dict[str, object]]] = {d: [] for d in DAYS}
    for gap in ranked:
        pending = [i for i in gap.items]
        for offset, day in enumerate(DAYS):
            if not pending:
                break
            if len(slots[day]) >= max_per_day:
                continue
            item = pending.pop(0)
            slots[day].append(
                {
                    "knowledge_point": gap.knowledge_point,
                    "risk": gap.risk_score,
                    "item_id": item.get("id"),
                    "text": item.get("text"),
                    "difficulty": item.get("difficulty"),
                }
            )
    def due_iso(day: int) -> str:
        return date.fromordinal(today.toordinal() + day - 1).isoformat()

    return [{"day": day, "date": due_iso(day), "items": slots[day]} for day in DAYS]

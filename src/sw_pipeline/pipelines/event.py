from __future__ import annotations

from ..app import run_event
from ..models import EventSpec


def run_event_pipeline(event_spec: EventSpec, include_fetch: bool = False) -> None:
    run_event(event_spec, include_fetch=include_fetch)

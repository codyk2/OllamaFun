"""Economic event calendar with trading blackout windows.

Blocks trading around high-impact news events to avoid volatility spikes:
  FOMC:  30 min before, 60 min after
  NFP:   15 min before, 45 min after
  CPI:   15 min before, 30 min after

Events are defined as recurring schedules. For production use,
these should be loaded from an external calendar API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

import pytz

from src.core.logging import get_logger

logger = get_logger("news_calendar")

ET = pytz.timezone("America/New_York")


@dataclass
class EconomicEvent:
    """A scheduled economic event with blackout buffer."""

    name: str
    event_time: datetime  # ET timezone
    pre_buffer_minutes: int = 15
    post_buffer_minutes: int = 30

    @property
    def blackout_start(self) -> datetime:
        return self.event_time - timedelta(minutes=self.pre_buffer_minutes)

    @property
    def blackout_end(self) -> datetime:
        return self.event_time + timedelta(minutes=self.post_buffer_minutes)

    def is_blocked(self, now: datetime) -> bool:
        """Check if the given time falls within this event's blackout window."""
        if now.tzinfo is None:
            now = ET.localize(now)
        else:
            now = now.astimezone(ET)

        return self.blackout_start <= now <= self.blackout_end


# Standard event configurations
FOMC_BUFFER = {"pre_buffer_minutes": 30, "post_buffer_minutes": 60}
NFP_BUFFER = {"pre_buffer_minutes": 15, "post_buffer_minutes": 45}
CPI_BUFFER = {"pre_buffer_minutes": 15, "post_buffer_minutes": 30}


class NewsCalendar:
    """Manages economic events and checks for blackout periods.

    Events can be added manually or loaded from external sources.
    """

    def __init__(self, events: list[EconomicEvent] | None = None) -> None:
        self.events: list[EconomicEvent] = events or []

    def add_event(self, event: EconomicEvent) -> None:
        self.events.append(event)

    def add_fomc(self, event_time: datetime) -> None:
        """Add an FOMC decision event (2:00 PM ET typically)."""
        self.events.append(EconomicEvent(
            name="FOMC", event_time=event_time, **FOMC_BUFFER,
        ))

    def add_nfp(self, event_time: datetime) -> None:
        """Add an NFP report event (8:30 AM ET, first Friday of month)."""
        self.events.append(EconomicEvent(
            name="NFP", event_time=event_time, **NFP_BUFFER,
        ))

    def add_cpi(self, event_time: datetime) -> None:
        """Add a CPI report event (8:30 AM ET)."""
        self.events.append(EconomicEvent(
            name="CPI", event_time=event_time, **CPI_BUFFER,
        ))

    def is_blocked(self, now: datetime) -> tuple[bool, str]:
        """Check if trading is blocked at the given time.

        Returns (blocked, reason).
        """
        if now.tzinfo is None:
            now = ET.localize(now)
        else:
            now = now.astimezone(ET)

        for event in self.events:
            if event.is_blocked(now):
                return True, (
                    f"News blackout: {event.name} "
                    f"({event.blackout_start.strftime('%H:%M')}-"
                    f"{event.blackout_end.strftime('%H:%M')} ET)"
                )

        return False, ""

    def next_event(self, now: datetime) -> EconomicEvent | None:
        """Find the next upcoming event after the given time."""
        if now.tzinfo is None:
            now = ET.localize(now)
        else:
            now = now.astimezone(ET)

        future_events = [e for e in self.events if e.event_time > now]
        if not future_events:
            return None
        return min(future_events, key=lambda e: e.event_time)

    def clear_past_events(self, now: datetime) -> int:
        """Remove events whose blackout window has fully passed."""
        if now.tzinfo is None:
            now = ET.localize(now)
        else:
            now = now.astimezone(ET)

        before = len(self.events)
        self.events = [e for e in self.events if e.blackout_end > now]
        removed = before - len(self.events)
        if removed:
            logger.info("cleared_past_events", count=removed)
        return removed

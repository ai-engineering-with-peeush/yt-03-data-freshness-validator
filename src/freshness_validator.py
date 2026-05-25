"""
freshness_validator.py
======================
A lightweight data freshness validation layer for ML pipelines.

Companion code for: "How to Build a Data Freshness Validation Layer in Python
— Silent Data Source Failure Fix" (AI Engineering with Peeush, Video 3)

Problem it solves
-----------------
Your ML pipeline is too resilient. When an external data source stops updating,
stale data flows through silently — model outputs degrade, nobody notices.
This validator adds a gate: check freshness before the model ever sees the data.

Usage
-----
    from freshness_validator import DataSource, FreshnessValidator
    from datetime import timedelta

    validator = FreshnessValidator()

    source = DataSource(
        name="customer_events_api",
        expected_freshness=timedelta(hours=1),
        get_last_updated=lambda: fetch_last_event_timestamp(),
    )

    result = validator.check(source)
    if not result.is_fresh:
        # Don't run inference — alert and stop
        raise RuntimeError(result.message)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DataSource:
    """
    Describes a single data source and its freshness contract.

    Attributes
    ----------
    name : str
        Human-readable identifier (used in logs and alerts).
    expected_freshness : timedelta
        How recently the source must have updated to be considered fresh.
        Example: timedelta(hours=1) means "must have updated in the last hour".
    get_last_updated : Callable[[], datetime]
        Zero-argument function that returns the source's last-updated timestamp.
        Must return a timezone-aware datetime (UTC recommended).
    """
    name: str
    expected_freshness: timedelta
    get_last_updated: Callable[[], datetime]


@dataclass
class FreshnessResult:
    """
    The outcome of a single freshness check.

    Attributes
    ----------
    source_name : str
        Name of the DataSource that was checked.
    last_updated : datetime
        Timestamp returned by the source's get_last_updated function.
    checked_at : datetime
        UTC timestamp when this check was performed.
    is_fresh : bool
        True if data is within the expected freshness window.
    staleness : timedelta
        How long ago the source last updated (0 if fresh).
    message : str
        Human-readable summary — safe to log or send to an alert system.
    """
    source_name: str
    last_updated: datetime
    checked_at: datetime
    is_fresh: bool
    staleness: timedelta
    message: str


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class FreshnessValidator:
    """
    Checks one or more DataSources against their freshness contracts.

    Parameters
    ----------
    alert_fn : Callable[[FreshnessResult], None], optional
        Called whenever a stale result is detected. Defaults to a log warning.
        Replace with your own function to send Slack messages, PagerDuty alerts,
        write to a monitoring DB, etc.

    Example
    -------
        def my_alert(result: FreshnessResult):
            slack_client.post(f":warning: {result.message}")

        validator = FreshnessValidator(alert_fn=my_alert)
    """

    def __init__(self, alert_fn: Optional[Callable[[FreshnessResult], None]] = None):
        self._alert_fn = alert_fn or self._default_alert

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, source: DataSource) -> FreshnessResult:
        """
        Check a single DataSource for freshness.

        Returns a FreshnessResult. If the source is stale, the alert function
        is called automatically — you don't need to call it yourself.
        """
        now = datetime.now(tz=timezone.utc)
        last_updated = source.get_last_updated()

        # Normalise: if the returned datetime has no timezone, assume UTC
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)

        age = now - last_updated
        is_fresh = age <= source.expected_freshness
        staleness = timedelta(0) if is_fresh else age

        if is_fresh:
            message = (
                f"[FRESH] {source.name} — "
                f"last updated {_fmt_duration(age)} ago "
                f"(threshold: {_fmt_duration(source.expected_freshness)})"
            )
        else:
            message = (
                f"[STALE] {source.name} — "
                f"last updated {_fmt_duration(age)} ago, "
                f"expected within {_fmt_duration(source.expected_freshness)}. "
                f"Staleness: {_fmt_duration(staleness)}"
            )

        result = FreshnessResult(
            source_name=source.name,
            last_updated=last_updated,
            checked_at=now,
            is_fresh=is_fresh,
            staleness=staleness,
            message=message,
        )

        if not is_fresh:
            self._alert_fn(result)
        else:
            logger.info(result.message)

        return result

    def check_all(self, sources: List[DataSource]) -> List[FreshnessResult]:
        """
        Check every DataSource in the list.

        All sources are checked regardless of individual failures — you get a
        complete picture of your pipeline's data health in one call.

        Returns a list of FreshnessResult objects in the same order as `sources`.
        """
        results = []
        for source in sources:
            try:
                result = self.check(source)
            except Exception as exc:
                # If the get_last_updated call itself fails, treat as stale
                now = datetime.now(tz=timezone.utc)
                error_message = (
                    f"[ERROR] {source.name} — could not retrieve last-updated "
                    f"timestamp: {exc}"
                )
                logger.error(error_message)
                result = FreshnessResult(
                    source_name=source.name,
                    last_updated=datetime.min.replace(tzinfo=timezone.utc),
                    checked_at=now,
                    is_fresh=False,
                    staleness=now - datetime.min.replace(tzinfo=timezone.utc),
                    message=error_message,
                )
                self._alert_fn(result)
            results.append(result)
        return results

    def all_fresh(self, sources: List[DataSource]) -> bool:
        """
        Convenience method: returns True only if every source passes.

        Use this as a pipeline gate:

            if not validator.all_fresh(sources):
                raise RuntimeError("Stale data detected — aborting inference.")
        """
        results = self.check_all(sources)
        return all(r.is_fresh for r in results)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_alert(result: FreshnessResult) -> None:
        """Default alert: emit a WARNING log. Replace for production use."""
        logger.warning(result.message)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_duration(delta: timedelta) -> str:
    """Return a human-readable duration string, e.g. '2h 15m' or '45s'."""
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "0s"

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)

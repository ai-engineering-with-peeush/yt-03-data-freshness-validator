"""
pipeline_example.py
===================
Real-world usage: wiring the FreshnessValidator into an ML inference pipeline.

This is the "after" state — what your pipeline looks like once you've added
the data freshness validation layer.

Scenario
--------
A production ML pipeline that:
  1. Fetches data from three sources (a SQL DB, a REST API, a file export)
  2. Validates that all sources are fresh BEFORE running inference
  3. Sends a Slack alert if any source is stale (and aborts the run)
  4. Proceeds to feature engineering and model inference only if all sources pass

Run this file to see the validator in action with simulated data sources.
"""

from __future__ import annotations

import logging
import sys
import os
from datetime import datetime, timedelta, timezone

# Add parent directory to path (for running from examples/ folder)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from freshness_validator import DataSource, FreshnessResult, FreshnessValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Define your alert function
# ---------------------------------------------------------------------------
# In production, this would call your Slack webhook, PagerDuty, or write to
# a monitoring DB. For this demo it just prints to the console.


def alert_team(result: FreshnessResult) -> None:
    """Send a Slack / PagerDuty alert when a source is stale."""
    logger.warning("🚨 ALERT: %s", result.message)
    # In production you'd add something like:
    # slack_client.chat_postMessage(
    #     channel="#ml-alerts",
    #     text=f":warning: *Stale data detected*\n{result.message}"
    # )


# ---------------------------------------------------------------------------
# Step 2 — Simulate data sources
# ---------------------------------------------------------------------------
# In production these would be real adapters from adapters.py.
# Here we use simple lambdas to simulate different freshness states.


def simulate_customer_events_db() -> datetime:
    """Simulates a SQL DB that updated 20 minutes ago → FRESH."""
    return datetime.now(tz=timezone.utc) - timedelta(minutes=20)


def simulate_product_catalog_api() -> datetime:
    """Simulates a REST API that updated 3 hours ago → STALE (threshold: 1h)."""
    return datetime.now(tz=timezone.utc) - timedelta(hours=3)


def simulate_feature_store_export() -> datetime:
    """Simulates a file export that updated 10 minutes ago → FRESH."""
    return datetime.now(tz=timezone.utc) - timedelta(minutes=10)


# ---------------------------------------------------------------------------
# Step 3 — Define your data sources
# ---------------------------------------------------------------------------

DATA_SOURCES = [
    DataSource(
        name="customer_events_db",
        expected_freshness=timedelta(hours=1),
        get_last_updated=simulate_customer_events_db,
    ),
    DataSource(
        name="product_catalog_api",
        expected_freshness=timedelta(hours=1),  # stale — last update was 3h ago
        get_last_updated=simulate_product_catalog_api,
    ),
    DataSource(
        name="feature_store_export",
        expected_freshness=timedelta(hours=2),
        get_last_updated=simulate_feature_store_export,
    ),
]


# ---------------------------------------------------------------------------
# Step 4 — The pipeline
# ---------------------------------------------------------------------------


def run_inference_pipeline() -> None:
    """
    ML inference pipeline with a data freshness gate.

    ┌─────────────────────────────────┐
    │  1. Validate data freshness     │ ← NEW: blocks stale data
    └──────────────┬──────────────────┘
                   │ All fresh?
                   ▼
    ┌─────────────────────────────────┐
    │  2. Fetch raw data              │
    └──────────────┬──────────────────┘
                   ▼
    ┌─────────────────────────────────┐
    │  3. Feature engineering         │
    └──────────────┬──────────────────┘
                   ▼
    ┌─────────────────────────────────┐
    │  4. Model inference             │
    └──────────────┬──────────────────┘
                   ▼
    ┌─────────────────────────────────┐
    │  5. Write outputs               │
    └─────────────────────────────────┘
    """

    logger.info("=" * 60)
    logger.info("Starting ML inference pipeline")
    logger.info("=" * 60)

    # ── GATE: Check all data sources before doing any work ──────────────────
    logger.info(
        "Step 1: Validating data freshness across %d sources...", len(DATA_SOURCES)
    )

    validator = FreshnessValidator(alert_fn=alert_team)
    results = validator.check_all(DATA_SOURCES)

    stale_sources = [r for r in results if not r.is_fresh]

    if stale_sources:
        logger.error(
            "Pipeline aborted — %d source(s) are stale: %s",
            len(stale_sources),
            [r.source_name for r in stale_sources],
        )
        logger.error(
            "Inference will NOT run. Investigate the stale sources before retrying."
        )
        return  # ← Stop here. Don't proceed with bad data.

    logger.info("✅ All sources are fresh. Proceeding with inference.")

    # ── Step 2: Fetch raw data ───────────────────────────────────────────────
    logger.info("Step 2: Fetching raw data from all sources...")
    raw_data = _fetch_raw_data()
    logger.info("  Fetched %d records.", len(raw_data))

    # ── Step 3: Feature engineering ─────────────────────────────────────────
    logger.info("Step 3: Running feature engineering...")
    features = _engineer_features(raw_data)
    logger.info("  Engineered %d feature vectors.", len(features))

    # ── Step 4: Model inference ──────────────────────────────────────────────
    logger.info("Step 4: Running model inference...")
    predictions = _run_model(features)
    logger.info("  Generated %d predictions.", len(predictions))

    # ── Step 5: Write outputs ────────────────────────────────────────────────
    logger.info("Step 5: Writing outputs...")
    _write_outputs(predictions)

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully.")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Placeholder pipeline steps (replace with your real code)
# ---------------------------------------------------------------------------


def _fetch_raw_data() -> list[dict]:
    """Fetch data from all sources (simplified for demo)."""
    return [{"customer_id": i, "event_count": i * 3} for i in range(1, 101)]


def _engineer_features(raw_data: list[dict]) -> list[dict]:
    """Transform raw data into model-ready features."""
    return [
        {
            "customer_id": row["customer_id"],
            "log_event_count": row["event_count"] ** 0.5,
        }
        for row in raw_data
    ]


def _run_model(features: list[dict]) -> list[dict]:
    """Run the ML model (placeholder — replace with your actual model call)."""
    return [
        {
            "customer_id": f["customer_id"],
            "churn_probability": min(1.0, f["log_event_count"] / 20),
        }
        for f in features
    ]


def _write_outputs(predictions: list[dict]) -> None:
    """Persist predictions to the output store."""
    logger.info("  Writing %d predictions to output DB...", len(predictions))
    # In production: write to SQL, Cosmos DB, etc.


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_inference_pipeline()

    print()
    print("─" * 60)
    print("Try this: change simulate_product_catalog_api() to return")
    print("  datetime.now(tz=timezone.utc) - timedelta(minutes=30)")
    print("and re-run — the pipeline will proceed with all-fresh data.")
    print("─" * 60)

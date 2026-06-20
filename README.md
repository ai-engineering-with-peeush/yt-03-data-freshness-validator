# Data Freshness Validation Layer in Python

> Companion code for: **[How to Build a Data Freshness Validation Layer in Python](https://youtu.be/xW1xgx28D_w)**  
> Part of the [Debugging ML in Production](https://www.youtube.com/playlist?list=PLOszX3Fd4bgccKUnq6cBaZbQ3V7245isU) series on YouTube.

---

## What this builds

A lightweight, drop-in `FreshnessValidator` that checks whether your ML pipeline's data sources are fresh before you run inference — and aborts cleanly if any source is stale.

This is the hands-on fix for **Failure Mode 1: Silent Data Source Failure**, introduced in [Part 1 of Debugging ML in Production](https://youtu.be/GsxQQvXGzDs).

```python
from src.freshness_validator import FreshnessValidator, DataSource
from datetime import timedelta

validator = FreshnessValidator(max_staleness=timedelta(hours=1))

sources = [
    DataSource(name="customer_events_db", last_updated=...),
    DataSource(name="product_catalog_api", last_updated=...),
]

results = validator.check_all(sources)

if not validator.all_fresh(results):
    raise RuntimeError("Stale data detected — aborting pipeline")
```

---

## What you'll learn

- How to model data source freshness as a typed, testable Python object
- How to build a `FreshnessValidator` that checks one or all sources
- How to plug it into a real ML inference pipeline so stale data aborts the run
- How to write adapters for SQL, REST APIs, files, and Cosmos DB

---

## Getting started

**Requirements:** Python 3.11+, no external dependencies for the core module.

```bash
git clone https://github.com/ai-engineering-with-peeush/yt-03-data-freshness-validator.git
cd yt-03-data-freshness-validator

# Run the end-to-end pipeline demo
python src/examples/pipeline_example.py

# Run the test suite
python -m unittest discover src/tests
```

---

## Code structure

```
src/
  ├── freshness_validator.py   # Core: DataSource, FreshnessResult, FreshnessValidator
  ├── adapters.py              # Real-world adapters: SQL, REST API, file, Cosmos DB
  ├── examples/
  │   └── pipeline_example.py  # End-to-end demo — 3-source ML pipeline
  └── tests/
      └── test_freshness_validator.py  # 22 unit tests
```

### Key classes

| Class | Description |
|-------|-------------|
| `DataSource` | Holds source name and `last_updated` timestamp |
| `FreshnessResult` | Immutable result of a single freshness check (fresh/stale, age, staleness duration) |
| `FreshnessValidator` | Checks one or all sources; configurable `max_staleness` |

---

## Series context

This repo is part of the **Debugging ML in Production** series:

| Video | Link |
|-------|------|
| Part 1 — 5 Failure Modes (Theory) | [Watch](https://youtu.be/GsxQQvXGzDs) |
| Part 2 — Failure Modes 4 & 5 (Theory) | [Watch](https://youtu.be/j_IjpiZE_4k) |
| **Part 3 — Data Freshness Validator (this repo)** | [Watch](https://youtu.be/xW1xgx28D_w) |
| Part 4 — Step-Level Observability | [Watch](https://youtu.be/0-SEzeCT9qE) |

---

## Channel

**[AI Engineering with Peeush](https://www.youtube.com/@AIEngineeringWithPeeush)** — hands-on production ML engineering, one video at a time.

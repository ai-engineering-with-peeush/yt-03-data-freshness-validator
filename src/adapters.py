"""
adapters.py
===========
Real-world adapters: plug the FreshnessValidator into your actual data sources.

Each function returns a zero-argument callable suitable for use as the
`get_last_updated` field on a DataSource. This keeps your DataSource
declarations clean and the connection/query logic isolated here.

Adapters included
-----------------
1. sql_adapter       — last-updated timestamp from a database table
2. rest_api_adapter  — last-updated field from a REST API JSON response
3. file_adapter      — file modification time (local file or NFS mount)
4. cosmos_db_adapter — Azure Cosmos DB (common in Azure ML stacks)

Usage example
-------------
    from freshness_validator import DataSource, FreshnessValidator
    from adapters import sql_adapter, rest_api_adapter
    from datetime import timedelta
    import pyodbc

    conn = pyodbc.connect(CONNECTION_STRING)

    sources = [
        DataSource(
            name="customer_events_db",
            expected_freshness=timedelta(hours=1),
            get_last_updated=sql_adapter(
                connection=conn,
                table="customer_events",
                timestamp_column="event_time",
            ),
        ),
        DataSource(
            name="product_catalog_api",
            expected_freshness=timedelta(hours=6),
            get_last_updated=rest_api_adapter(
                url="https://api.example.com/products/last-updated",
                timestamp_field="updated_at",
            ),
        ),
    ]

    validator = FreshnessValidator()
    results = validator.check_all(sources)
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# 1. SQL adapter
# ---------------------------------------------------------------------------

def sql_adapter(
    connection,
    table: str,
    timestamp_column: str,
    schema: str = "dbo",
) -> Callable[[], datetime]:
    """
    Returns a callable that queries MAX(<timestamp_column>) from a SQL table.

    Works with any DB-API 2.0 connection (pyodbc, psycopg2, sqlite3, etc.).

    Parameters
    ----------
    connection : DB-API 2.0 connection object
        An open database connection.
    table : str
        Table name to query.
    timestamp_column : str
        Column containing the row's last-updated timestamp.
    schema : str
        Schema name (default: "dbo" for SQL Server).

    Returns
    -------
    Callable[[], datetime]
        Zero-argument function that returns a UTC-aware datetime.
    """
    def get_last_updated() -> datetime:
        cursor = connection.cursor()
        cursor.execute(
            f"SELECT MAX([{timestamp_column}]) FROM [{schema}].[{table}]"
        )
        row = cursor.fetchone()
        if row is None or row[0] is None:
            raise ValueError(
                f"No rows found in {schema}.{table}.{timestamp_column} — "
                "table may be empty."
            )
        ts = row[0]
        # If the DB driver returns a naive datetime, assume UTC
        if isinstance(ts, datetime) and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts

    return get_last_updated


# ---------------------------------------------------------------------------
# 2. REST API adapter
# ---------------------------------------------------------------------------

def rest_api_adapter(
    url: str,
    timestamp_field: str,
    timestamp_format: str = "%Y-%m-%dT%H:%M:%SZ",
    headers: Optional[dict] = None,
) -> Callable[[], datetime]:
    """
    Returns a callable that fetches a JSON endpoint and parses a timestamp field.

    Suitable for any API that returns a JSON body with a last-updated field.

    Parameters
    ----------
    url : str
        Full URL of the API endpoint.
    timestamp_field : str
        Top-level JSON key containing the timestamp string.
        For nested keys, use dot notation: "data.last_updated"
    timestamp_format : str
        strptime format string (default: ISO 8601 UTC).
    headers : dict, optional
        HTTP headers to include (e.g. Authorization tokens).

    Returns
    -------
    Callable[[], datetime]
        Zero-argument function that returns a UTC-aware datetime.

    Example
    -------
        # API response: {"status": "ok", "data": {"last_updated": "2026-05-16T08:00:00Z"}}
        adapter = rest_api_adapter(
            url="https://api.example.com/status",
            timestamp_field="data.last_updated",
        )
    """
    def get_last_updated() -> datetime:
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))

        # Support dot-notation for nested keys: "data.last_updated"
        value = body
        for key in timestamp_field.split("."):
            value = value[key]

        ts = datetime.strptime(value, timestamp_format)

        # Ensure UTC-aware
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        return ts

    return get_last_updated


# ---------------------------------------------------------------------------
# 3. File adapter
# ---------------------------------------------------------------------------

def file_adapter(path: str) -> Callable[[], datetime]:
    """
    Returns a callable that reads a file's last-modified timestamp.

    Useful for pipelines that write output to a file (CSV, Parquet, JSON)
    and you want to verify the file was recently updated.

    Parameters
    ----------
    path : str
        Absolute or relative path to the file.

    Returns
    -------
    Callable[[], datetime]
        Zero-argument function that returns a UTC-aware datetime.
    """
    def get_last_updated() -> datetime:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime, tz=timezone.utc)

    return get_last_updated


# ---------------------------------------------------------------------------
# 4. Azure Cosmos DB adapter
# ---------------------------------------------------------------------------

def cosmos_db_adapter(
    endpoint: str,
    key: str,
    database_name: str,
    container_name: str,
    query: str,
    timestamp_field: str,
) -> Callable[[], datetime]:
    """
    Returns a callable that queries Azure Cosmos DB for the latest timestamp.

    Requires: pip install azure-cosmos

    Parameters
    ----------
    endpoint : str
        Cosmos DB account endpoint URL.
    key : str
        Account key (use environment variable — never hardcode).
    database_name : str
        Cosmos DB database name.
    container_name : str
        Container (collection) name.
    query : str
        SQL query returning a single row with the timestamp field.
        Example: "SELECT TOP 1 c.updated_at FROM c ORDER BY c.updated_at DESC"
    timestamp_field : str
        Field name in the query result containing the timestamp.

    Returns
    -------
    Callable[[], datetime]
        Zero-argument function that returns a UTC-aware datetime.
    """
    def get_last_updated() -> datetime:
        try:
            from azure.cosmos import CosmosClient  # type: ignore
        except ImportError:
            raise ImportError(
                "azure-cosmos is required for cosmos_db_adapter. "
                "Install it with: pip install azure-cosmos"
            )

        client = CosmosClient(url=endpoint, credential=key)
        container = (
            client
            .get_database_client(database_name)
            .get_container_client(container_name)
        )
        items = list(container.query_items(query=query, enable_cross_partition_query=True))

        if not items:
            raise ValueError(
                f"No results returned from Cosmos DB container '{container_name}'"
            )

        raw_ts = items[0][timestamp_field]

        # Cosmos stores timestamps as ISO 8601 strings
        ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        return ts

    return get_last_updated

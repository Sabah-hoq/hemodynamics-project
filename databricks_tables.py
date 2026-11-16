"""
Send a pandas DataFrame to Databricks as a real table.

Experiment tracking (tracking.py) stores runs and models. This module
stores the *data itself*, so it shows up under Catalog in the workspace and
can be queried in a notebook or charted on a Databricks dashboard.

It uses only the workspace URL and access token the app already has, via
two Databricks REST APIs:

  SQL warehouses        to find a warehouse that can run SQL
  SQL statement API     to CREATE the table and INSERT the rows

No extra Python package or cluster is needed.
"""

import math
import time

import pandas as pd
import requests

TIMEOUT = 30          # seconds per HTTP request
ROWS_PER_INSERT = 500
STATEMENT_WAIT_S = 300  # a stopped warehouse can take a few minutes to start


class DatabricksError(RuntimeError):
    pass


def _call(method, host, token, path, **kwargs):
    try:
        resp = requests.request(
            method, f"{host.rstrip('/')}{path}",
            headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT, **kwargs,
        )
    except requests.RequestException as e:
        raise DatabricksError(f"Could not reach {host} ({type(e).__name__}).") from e
    if resp.status_code in (401, 403):
        raise DatabricksError(
            f"Databricks refused the request (HTTP {resp.status_code}). The token may "
            "have expired, or your account may not be allowed to do this."
        )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("message", resp.text)
        except ValueError:
            detail = resp.text
        raise DatabricksError(f"Databricks error (HTTP {resp.status_code}): {detail[:300]}")
    return resp.json() if resp.content else {}


def list_warehouses(host, token):
    """SQL warehouses in the workspace as [{'id', 'name', 'state'}], running ones first."""
    found = _call("GET", host, token, "/api/2.0/sql/warehouses").get("warehouses", [])
    out = [{"id": w["id"], "name": w.get("name", w["id"]), "state": w.get("state", "")}
           for w in found]
    return sorted(out, key=lambda w: w["state"] != "RUNNING")


def default_schema(host, token):
    """A sensible 'catalog.schema' to write to: the workspace's own catalog
    if there is one, otherwise the first that is not a system catalog."""
    try:
        catalogs = [c["name"] for c in
                    _call("GET", host, token, "/api/2.1/unity-catalog/catalogs").get("catalogs", [])]
    except DatabricksError:
        return "hive_metastore.default"
    for preferred in ("workspace", "main"):
        if preferred in catalogs:
            return f"{preferred}.default"
    usable = [c for c in catalogs if c not in ("system", "samples")]
    return f"{usable[0]}.default" if usable else "hive_metastore.default"


def _run_sql(host, token, warehouse_id, statement):
    """Run one SQL statement and wait for it to finish."""
    body = {"warehouse_id": warehouse_id, "statement": statement,
            "wait_timeout": "30s", "on_wait_timeout": "CONTINUE"}
    result = _call("POST", host, token, "/api/2.0/sql/statements", json=body)
    deadline = time.time() + STATEMENT_WAIT_S
    while True:
        state = result.get("status", {}).get("state")
        if state == "SUCCEEDED":
            return result
        if state in ("FAILED", "CANCELED", "CLOSED"):
            message = result.get("status", {}).get("error", {}).get("message", state)
            raise DatabricksError(f"Databricks could not run the SQL: {message[:400]}")
        if time.time() > deadline:
            raise DatabricksError("Databricks took too long to answer. The SQL warehouse "
                                  "may still be starting; try again in a minute.")
        time.sleep(2)
        result = _call("GET", host, token,
                       f"/api/2.0/sql/statements/{result['statement_id']}")


def _ident(name):
    """Quote one identifier; backticks inside it are doubled."""
    return "`" + str(name).replace("`", "``") + "`"


def _sql_type(series):
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    return "STRING"


def _literal(value):
    if value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float) and not math.isfinite(value):
        return "NULL"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def qualified_name(schema, table):
    """`catalog`.`schema`.`table` from 'catalog.schema' and a table name."""
    parts = [p for p in schema.split(".") if p.strip()]
    if len(parts) != 2:
        raise DatabricksError("Destination must look like catalog.schema, "
                              "for example workspace.default")
    return ".".join(_ident(p.strip()) for p in parts + [table])


def upload_dataframe(df, host, token, warehouse_id, schema, table, progress=None):
    """Create (or replace) `schema.table` in Databricks holding the rows of df.

    progress, if given, is called with a fraction between 0 and 1.
    Returns the number of rows written.
    """
    if df.empty:
        raise DatabricksError("There are no rows to send.")
    target = qualified_name(schema, table)
    columns = ", ".join(f"{_ident(c)} {_sql_type(df[c])}" for c in df.columns)
    _run_sql(host, token, warehouse_id, f"CREATE OR REPLACE TABLE {target} ({columns})")

    # plain Python values, so numpy types and missing values serialise cleanly
    records = df.astype(object).where(df.notna(), None).values.tolist()
    for start in range(0, len(records), ROWS_PER_INSERT):
        chunk = records[start:start + ROWS_PER_INSERT]
        values = ",\n".join("(" + ", ".join(_literal(v) for v in row) + ")" for row in chunk)
        _run_sql(host, token, warehouse_id, f"INSERT INTO {target} VALUES\n{values}")
        if progress:
            progress(min(1.0, (start + len(chunk)) / len(records)))
    return len(records)


def table_url(host, schema, table):
    catalog, sch = [p.strip() for p in schema.split(".")]
    return f"{host.rstrip('/')}/explore/data/{catalog}/{sch}/{table}"

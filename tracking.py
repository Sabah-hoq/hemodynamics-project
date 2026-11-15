"""
Experiment tracking with MLflow, stored either locally or in Databricks.

Every time the detector is trained, the settings that produced it (network
geometry, noise levels, seed), the scores it reached (accuracy, ROC AUC,
chosen threshold, sensitivity, specificity) and optionally the trained
model itself are recorded as one "run". Runs can then be compared side by
side instead of being lost when the app is closed.

Two backends, same code path:

  local       a SQLite file under outputs/ (no account needed)
  databricks  the Managed MLflow service inside a Databricks workspace,
              reached with a workspace URL and a personal access token

MLflow is the open-source tracking library Databricks maintains, which is
why switching between the two is one setting rather than a rewrite.
"""

import hashlib
import json
import os
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
import requests
from mlflow.tracking import MlflowClient

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_DIR, "outputs")
LOCAL_DB = os.path.join(OUTPUT_DIR, "mlflow.db")
LOCAL_ARTIFACTS = os.path.join(OUTPUT_DIR, "mlartifacts")

LOCAL_EXPERIMENT = "hemodynamics-flow-anomaly"
DEFAULT_DATABRICKS_EXPERIMENT = "/Shared/hemodynamics-flow-anomaly"


def configure_local():
    """Point MLflow at the SQLite store under outputs/. Returns the experiment name."""
    os.makedirs(LOCAL_ARTIFACTS, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{LOCAL_DB}")
    if mlflow.get_experiment_by_name(LOCAL_EXPERIMENT) is None:
        mlflow.create_experiment(
            LOCAL_EXPERIMENT, artifact_location=f"file://{LOCAL_ARTIFACTS}"
        )
    mlflow.set_experiment(LOCAL_EXPERIMENT)
    return LOCAL_EXPERIMENT


def configure_databricks(host, token, experiment_path=DEFAULT_DATABRICKS_EXPERIMENT):
    """Point MLflow at a Databricks workspace. Returns the experiment path.

    host             workspace URL, e.g. https://dbc-1234abcd-5678.cloud.databricks.com
    token            a personal access token (User Settings > Developer > Access tokens)
    experiment_path  absolute workspace path, e.g. /Users/you@example.com/hemodynamics

    The credentials are kept in this process's environment only; nothing is
    written to disk. Raises if the workspace cannot be reached or rejects
    the token, so a caller can show the reason.
    """
    host = host.strip().rstrip("/")
    if not host.startswith("https://"):
        raise ValueError("Workspace URL must start with https://")
    if not token.strip():
        raise ValueError("A personal access token is required")
    if not experiment_path.startswith("/"):
        raise ValueError("Experiment path must be absolute, e.g. /Shared/hemodynamics")

    token = token.strip()
    # Check the URL and token with one short request before handing over to
    # MLflow. MLflow and the Databricks SDK both retry an unreachable host
    # for minutes by default, which would freeze the app on a mistyped URL.
    try:
        resp = requests.get(
            f"{host}/api/2.0/mlflow/experiments/search",
            headers={"Authorization": f"Bearer {token}"},
            params={"max_results": 1}, timeout=15,
        )
    except requests.RequestException as e:
        raise ConnectionError(
            f"Could not reach {host}. Check the workspace URL and your network. "
            f"({type(e).__name__})"
        ) from e
    if resp.status_code in (401, 403):
        raise PermissionError(
            "The workspace rejected the access token "
            f"(HTTP {resp.status_code}). Generate a new one under "
            "User Settings > Developer > Access tokens."
        )
    if resp.status_code != 200:
        raise ConnectionError(
            f"{host} answered HTTP {resp.status_code}, which is not what a "
            "Databricks workspace returns. Check the workspace URL."
        )

    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "20"
    os.environ["DATABRICKS_RETRY_TIMEOUT_SECONDS"] = "20"
    os.environ["DATABRICKS_HTTP_TIMEOUT_SECONDS"] = "20"
    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token
    mlflow.set_tracking_uri("databricks")
    # the first real request: fails here if the URL or token is wrong
    mlflow.set_experiment(experiment_path)
    return experiment_path


def _config_hash(params):
    canon = json.dumps({k: str(v) for k, v in params.items()}, sort_keys=True)
    return hashlib.sha1(canon.encode()).hexdigest()[:16]


def log_training_run(params, metrics, model=None, run_name=None, tags=None):
    """Record one training run in the currently configured backend.

    params   dict of settings (logged as strings)
    metrics  dict of name -> float
    model    optional fitted scikit-learn estimator to store with the run
    Returns the run id.

    Training is seeded, so the same settings always give the same scores.
    If a run with identical settings is already stored, its id is returned
    and nothing new is written.
    """
    config = _config_hash(params)
    existing = mlflow.search_runs(
        filter_string=f"tags.config_hash = '{config}'", max_results=1
    )
    if not existing.empty:
        return existing.iloc[0]["run_id"]

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.set_tags({"config_hash": config, **(tags or {})})
        if model is not None:
            # MLflow saves scikit-learn models in the skops format, which
            # refuses types it cannot vouch for. Random-forest node storage
            # is one of them; it is declared trusted here because the model
            # was fitted in this process a moment ago, not loaded from a file.
            mlflow.sklearn.log_model(
                model, name="binary_detector",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
        return run.info.run_id


def list_runs(experiment_name, max_results=200):
    """All runs of an experiment as a tidy DataFrame, newest first.

    Column prefixes are stripped ("metrics.roc_auc" -> "roc_auc") and
    numeric-looking params are converted so they can be plotted.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=max_results,
        order_by=["attributes.start_time DESC"],
    )
    if runs.empty:
        return runs

    keep = {"run_id": "run_id", "start_time": "started",
            "tags.mlflow.runName": "run_name"}
    out = pd.DataFrame({new: runs[old] for old, new in keep.items() if old in runs})
    for col in runs.columns:
        if col.startswith("metrics."):
            out[col[len("metrics."):]] = runs[col]
        elif col.startswith("params."):
            as_num = pd.to_numeric(runs[col], errors="coerce")
            out[col[len("params."):]] = as_num if as_num.notna().all() else runs[col]
    if "started" in out:
        local_tz = datetime.now().astimezone().tzinfo
        out["started"] = (pd.to_datetime(out["started"], utc=True)
                          .dt.tz_convert(local_tz).dt.strftime("%Y-%m-%d %H:%M:%S"))
    return out


def delete_run(run_id):
    MlflowClient().delete_run(run_id)


_active = None


def activate(backend, host=None, token=None, experiment_path=None):
    """Make `backend` ("local" or "databricks") the current store, doing the
    setup work only when the choice differs from the last call. Streamlit
    reruns the whole script on every click, so this is called constantly.
    Returns the experiment name to pass to list_runs()."""
    global _active
    key = (backend, host, experiment_path)
    if _active and _active[0] == key:
        return _active[1]
    _active = None  # a failed switch must not leave a stale "already set up"
    if backend == "databricks":
        name = configure_databricks(host, token, experiment_path)
    else:
        name = configure_local()
    _active = (key, name)
    return name

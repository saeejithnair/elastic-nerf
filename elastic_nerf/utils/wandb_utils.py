import concurrent.futures
import functools
import glob
import json
import os
import pickle
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gonas.utils.logging_utils as lu
import pandas as pd
from tomlkit import table
from wandb.apis.public import Api, File
from wandb.apis.public import Run as WandbPublicRun
from wandb.apis.public import Sweep
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.artifacts.exceptions import ArtifactNotLoggedError
from wandb.sdk.wandb_init import init
from wandb.sdk.wandb_run import Run as WandbInternalRun

import wandb

CACHE_DIR: str = "/home/user/shared/wandb_cache"
PROJECT: str = "elastic-nerf"
ENTITY: str = "saeejithn"
PROJECT_NAME: str = f"{ENTITY}/{PROJECT}"


def retry(num_retries=3, initial_delay=5, backoff_factor=2):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(num_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == num_retries - 1:  # Last attempt
                        raise e
                    time.sleep(delay)
                    delay *= backoff_factor  # Increase delay for the next attempt
                    lu.error(
                        f"Retry {attempt + 1} for function {func.__name__} after error: {e}"
                    )

        return wrapper

    return decorator_retry


if not Path(CACHE_DIR).exists():
    from elastic_nerf.configs.hosts import HOSTNAMES_TO_CACHE_DIRS

    hostname = os.environ["HOSTNAME"]
    wandb_cache_dir = Path(HOSTNAMES_TO_CACHE_DIRS[hostname]) / "wandb_cache"
    # Make wandb_cache_dir if it doesn't exist already.
    wandb_cache_dir.mkdir(parents=True, exist_ok=True)
    lu.warning(
        f"WANDB_CACHE_DIR {CACHE_DIR} does not exist, defaulting to {wandb_cache_dir}"
    )
    CACHE_DIR = wandb_cache_dir.as_posix()

api = Api(timeout=19)


class RunResult:
    """Encapsulates the attributes we care about from a run."""

    def __init__(
        self,
        run: WandbPublicRun,
        download_history: bool = False,
        tables: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.run_id = run.id
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = cache_dir
        try:
            self.config = run.config
            self.state = run.state
            if download_history:
                self.history = self.download_history(run)
            self.summary = self.download_summary(run)
            if tables is not None:
                if self.cache_dir is None:
                    raise ValueError(
                        "Cache dir must be provided to download tables for a run."
                    )
                self.tables = self.download_tables(run, tables, self.cache_dir)
        except Exception as e:
            lu.error(f"An error occurred while fetching run {run.id}: {e}")
            raise e

    def update_missing_attributes(
        self,
        run: WandbPublicRun,
        download_history: bool,
        tables: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
    ) -> bool:
        attributes_updated = False
        # Update history if needed
        if download_history and not hasattr(self, "history"):
            print(f"Missing attribute history for run {run.id}.")
            self.history = self.download_history(run)
            attributes_updated = True

        # Update tables if needed
        if tables:
            if cache_dir is None:
                raise ValueError(
                    "Cache dir must be provided to download tables for a run."
                )

            if not hasattr(self, "tables"):
                self.tables = self.download_tables(run, tables, cache_dir)
                attributes_updated = True
            else:
                for table_name in tables:
                    if table_name not in self.tables or self.tables[table_name] is None:
                        df_tables = self.download_table(run, table_name, cache_dir)
                        if df_tables is None:
                            self.tables[table_name] = None
                            attributes_updated = True
                        else:
                            self.tables.update(df_tables)
                            attributes_updated = True

        return attributes_updated

    @staticmethod
    @retry(num_retries=3)
    def download_summary(run: WandbPublicRun) -> Dict:
        """Downloads the wandb-summary.json file for a run and returns the summary dict."""
        file: File = run.file("wandb-summary.json")  # type: ignore
        # Create a temporary directory with summary file
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "wandb-summary.json")
            # Download the wandb-summary.json to the temporary directory
            file.download(replace=True, root=temp_dir)

            # Open and read the temporary file
            with open(file_path, "r") as f:
                summary_dict = json.load(f)

            return summary_dict

    @staticmethod
    @retry(num_retries=3)
    def download_table(
        run: WandbPublicRun,
        table_name: str,
        cache_dir: Path,
        versions: Optional[List[int]] = None,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        artifact_name = f"run-{run.id}-{table_name}"
        artifact = try_fetch_artifact(artifact_name=artifact_name)
        if artifact is None:
            lu.error(f"Unable to find artifact {artifact_name} on WandB.")
            return None

        artifact_versions = list(artifact.collection.artifacts())[::-1]

        if versions is not None:
            artifacts = [artifact_versions[i] for i in versions]
        else:
            artifacts = [artifact_versions[-1]]
            versions = [len(artifact_versions) - 1]

        tables = {}
        for version_idx, table in zip(versions, artifacts):
            version = f"v{version_idx}"
            table_name_versioned = f"{table_name}_{version}"
            table_dir = cache_dir / table_name_versioned
            print(f"Downloading table {artifact_name} for run {run.id} to {table_dir}")
            table_path = table.download(root=table_dir.as_posix())
            table_json_path = f"{table_path}/**/**.table.json"
            table_json = glob.glob(table_json_path, recursive=True)
            assert (
                len(table_json) == 1
            ), f"Expected 1 table.json file, found {len(table_json)} files in {table_json_path}."

            tables[table_name_versioned] = parse_wandb_table_json_to_dataframe(
                table_json[0]
            )

        return tables

    @staticmethod
    def download_tables(
        run: WandbPublicRun,
        tables: List[str],
        cache_dir: Path,
        versions: Optional[List[int]] = None,
    ) -> Dict:
        """Downloads the table artifacts for a run and returns the table dict."""
        table_dict = {}
        for table_name in tables:
            df_tables = RunResult.download_table(run, table_name, cache_dir, versions)
            if df_tables is None:
                table_dict[table_name] = None
            else:
                table_dict.update(df_tables)

        return table_dict

    @staticmethod
    @retry(num_retries=3)
    def download_history(run: WandbPublicRun) -> pd.DataFrame:
        """Processes the run history from a WandB HistoryScan object into a pandas DataFrame."""
        # List to store the data for each step
        history_data = []

        # Iterate through the run history
        print(f"Downloading history for run {run.id}")
        for entry in run.scan_history(page_size=100000):
            # Append the current entry (row) to the history_data
            history_data.append(entry)

        # Convert the list of dictionaries into a DataFrame
        history_df = pd.DataFrame(history_data)
        print(f"Downloaded history for run {run.id}")

        return history_df


def fetch_run_result(
    run_id: str,
    project_name: str = PROJECT_NAME,
    cache: bool = True,
    refresh_cache: bool = False,
    download_history: bool = False,
    tables: Optional[List[str]] = None,
) -> RunResult:
    """Fetch a run from W&B."""

    # Define the path to the cache file
    cache_dir = Path(os.path.join(CACHE_DIR, "runs"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{run_id}_results.pkl")
    run_assets_dir = cache_dir / run_id

    run = fetch_run(run_id, project_name)
    # If a cached result exists, load and return it
    if cache and not refresh_cache and os.path.exists(cache_file):
        lu.warning(f"Found cached results for run {run_id}")
        with open(cache_file, "rb") as f:
            cached_result: RunResult = pickle.load(f)

        attributes_updated = cached_result.update_missing_attributes(
            run, download_history, tables, cache_dir=run_assets_dir
        )
        if attributes_updated:
            lu.warning(f"Updated missing attributes for run {run_id}.")
            with open(cache_file, "wb") as fw:
                pickle.dump(cached_result, fw)

        return cached_result

    run_result = RunResult(
        run,
        download_history=download_history,
        tables=tables,
        cache_dir=run_assets_dir,
    )

    if cache:
        with open(cache_file, "wb") as f:
            pickle.dump(run_result, f)
    return run_result


def fetch_sweep(sweep_id: str, project_name: str = PROJECT_NAME) -> Sweep:
    """Fetch a sweep from W&B."""
    return api.sweep(f"{project_name}/{sweep_id}")


def remove_sweep_results_cache(sweep: Union[str, Sweep]):
    """Remove the cache file for a sweep."""
    if isinstance(sweep, str):
        sweep_id = sweep
    else:
        sweep_id = sweep.id

    cache_file = os.path.join(CACHE_DIR, "sweeps", f"{sweep_id}_results.pkl")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        lu.info(f"Removed cached results for sweep {sweep_id}.")


def fetch_sweep_results(
    sweep: Union[str, Sweep],
    cache: bool = True,
    refresh_cache: bool = False,
    download_history: bool = False,
    tables: Optional[List[str]] = None,
) -> List[RunResult]:
    """Fetch summary data for all runs in a sweep, with optional caching."""
    if isinstance(sweep, str):
        sweep_id = sweep
        sweep = fetch_sweep(sweep)
    else:
        sweep_id = sweep.id

    # Define the path to the cache file
    cache_dir = Path(os.path.join(CACHE_DIR, "sweeps"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{sweep_id}_results.pkl")

    # Load from cache if available and not refreshing
    if cache and not refresh_cache and os.path.exists(cache_file):
        lu.warning(f"Using cached results for sweep {sweep_id}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Fetch the results
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                fetch_run_result,
                run.id,
                cache=cache,
                refresh_cache=refresh_cache,
                download_history=download_history,
                tables=tables,
            )
            for run in sweep.runs
            if run.state == "finished"
        ]
        runs = []
        for future in concurrent.futures.as_completed(futures):
            try:
                run_result = future.result()
                runs.append(run_result)
            except Exception as e:
                lu.error(f"Failed to fetch a run due to: {e}")

    lu.info(f"Fetched {len(runs)} finished runs out of {len(sweep.runs)} total runs.")

    # Optionally cache the results
    if cache:
        with open(cache_file, "wb") as f:
            pickle.dump(runs, f)
            lu.info(f"Cached results for sweep {sweep_id}.")

    return runs


def fetch_all_sweep_results(sweeps: List[str]) -> Dict[str, List[RunResult]]:
    # Define a function that fetches results for a given sweep ID
    def fetch_results(sweep_id: str) -> List[RunResult]:
        return fetch_sweep_results(sweep_id)

    # Dictionary to store the results
    all_results = {}

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Submit the fetch_results function for each sweep ID
        future_to_sweep = {
            executor.submit(fetch_results, sweep_id): sweep_id for sweep_id in sweeps
        }

        # Collect the results as they complete
        for future in concurrent.futures.as_completed(future_to_sweep):
            sweep_id = future_to_sweep[future]
            try:
                result = future.result()
                all_results[sweep_id] = result
            except Exception as e:
                lu.error(
                    f"An error occurred while fetching results for {sweep_id}: {e}"
                )

    return all_results


def flatten_dict(d, parent_key="", sep="_"):
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def prepare_data(results: List[RunResult]) -> pd.DataFrame:
    """Prepare data for analysis from updated RunResult summaries."""
    data = []

    for result in results:
        # Flatten the summary dictionary
        flattened_summary = flatten_dict(result.summary)
        flattened_config = flatten_dict(result.config)

        # Include the Run ID in the data
        run_data = {"Run ID": result.run_id}
        run_data.update(flattened_summary)
        run_data.update(flattened_config)

        # Append the collected data for this run to the main data list
        data.append(run_data)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)
    return df


def fetch_run(run_id: str, project_name: str = PROJECT_NAME) -> WandbPublicRun:
    """Returns a reference to an existing WandB run."""
    run: WandbPublicRun = api.run(f"{project_name}/{run_id}")
    return run


def download_file(
    run_id: str,
    remote_filename: str,
    local_filename: Optional[str] = None,
    cache_dir=CACHE_DIR,
    project_name: str = PROJECT_NAME,
    results_subdir: str = "results",
) -> Path:
    """
    Downloads a file associated with a particular run.

    Args:
        run_id (str): ID associated with the run.
        remote_filename (str): Name of the file on the remote server.
        local_filename (Optional[str]): Optional name to rename the downloaded file.
        cache_dir (str): Directory to store cached files.
        project_name (str): Name of the project.
        results_subdir (str): Subdirectory name for results.

    Returns:
        Path: Path to the downloaded (or already existing) local file.
    """
    local_filepath = determine_local_filepath(
        cache_dir, results_subdir, run_id, remote_filename, local_filename
    )
    if os.path.exists(local_filepath):
        lu.warning(f"File {local_filepath} already exists, skipping download.")
        return local_filepath

    run = fetch_run(run_id, project_name)
    target_file: File = run.file(remote_filename)  # type: ignore
    local_filepath.parent.mkdir(parents=True, exist_ok=True)
    target_file.download(root=local_filepath.parent.as_posix(), exist_ok=True)

    if local_filename:
        downloaded_filepath = os.path.join(local_filepath.parent, remote_filename)
        os.rename(downloaded_filepath, local_filepath)

    lu.info(f"Downloaded file {remote_filename} for run {run_id} to {local_filepath}")
    return local_filepath


def determine_local_filepath(
    cache_dir: str,
    results_subdir: str,
    run_id: str,
    remote_filename: str,
    local_filename: Optional[str] = None,
) -> Path:
    """
    Determine the local path where the file will be stored or was stored previously.

    Args:
        cache_dir (str): Base cache directory.
        results_subdir (str): Subdirectory name for results.
        run_id (str): ID associated with the run.
        remote_filename (str): Name of the file on the remote server.
        local_filename (Optional[str]): Optional name to rename the downloaded file.

    Returns:
        str: Path for the local file.
    """
    target_dir = os.path.join(cache_dir, results_subdir, run_id)
    filename = local_filename or remote_filename
    return Path(os.path.join(target_dir, filename))


def initialize_run(
    config: Optional[Dict] = None,
    id: Optional[str] = None,
    resume: Optional[Union[bool, str]] = None,
    reinit: Optional[bool] = False,
    project: str = PROJECT,
    entity: str = ENTITY,
) -> WandbInternalRun:
    """Initializes a WandB run."""
    run = init(
        id=id,
        config=config,
        project=os.environ.get("WANDB_PROJECT", project),
        entity=entity,
        dir=os.environ.get("WANDB_DIR", f"{CACHE_DIR}/tmp"),
        reinit=reinit,
        resume=resume,
    )
    assert isinstance(run, WandbInternalRun)
    return run


def get_active_run(
    run: Optional[WandbInternalRun] = None,
    run_id: Optional[str] = None,
    project=PROJECT,
    entity=ENTITY,
) -> WandbInternalRun:
    """Gets active WandB run."""
    if run is None:
        # If no run_id, then let WandB create a new run.
        resume = None

        # If run_id is passed, then resume and get handle to existing run.
        # NOTE: Since resume type is _must_, this will raise error if a run with
        # this ID does not exist.
        if run_id is not None:
            resume = "must"
        run = initialize_run(id=run_id, resume=resume, project=project, entity=entity)

    return run


def save_artifact(
    artifact_name: str,
    artifact_type: str,
    artifact_path: str,
    project=PROJECT,
    entity=ENTITY,
    run: Optional[WandbInternalRun] = None,
    run_id: Optional[str] = None,
) -> None:
    """Upload an artifact to W&B.
    Artifact is uploaded to to either:
        * A current run (if a run is passed)
        * An existing run (if run_id is passed)
        * A new run if both run_id and run are None.
    """
    active_run = get_active_run(run_id=run_id, run=run, project=project, entity=entity)

    with active_run:
        artifact = Artifact(artifact_name, type=artifact_type)
        artifact.add_file(artifact_path)
        active_run.log_artifact(artifact)


def save_file(
    file_path: Path,
    project=PROJECT,
    entity=ENTITY,
    run: Optional[WandbInternalRun] = None,
    run_id: Optional[str] = None,
):
    """Upload a file to W&B.
    Artifact is uploaded to to either:
        * A current run (if a run is passed)
        * An existing run (if run_id is passed)
        * A new run if both run_id and run are None.
    """
    active_run = get_active_run(run_id=run_id, run=run, project=project, entity=entity)

    with active_run:
        active_run.save(str(file_path), policy="now")


def try_fetch_artifact(
    artifact_name: str, project=PROJECT, entity=ENTITY, version: str = "latest"
) -> Union[Artifact, None]:
    try:
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}:{version}")
        return artifact
    except Exception as e:
        return None


def download_artifact(
    run_id: str,
    remote_filename: str,
    artifact_name: str,
    local_filename: Optional[str] = None,
    cache_dir=CACHE_DIR,
    results_subdir: str = "results/artifacts",
) -> Union[Path, None]:
    """
    Downloads an artifact.

    Args:
        run_id: Run ID corresponding to the artifact.
        remote_filename: Name of the file on the remote server.
        artifact: Wandb artifact to download. This can only be optional if you
            know for sure that it's already downloaded. Otherwise will raise error.
        local_filename: Optional name to rename the downloaded file.
        cache_dir: Directory to store cached files.
        results_subdir: Subdirectory name for results.

    Returns:
        Path: Path to the downloaded (or already existing) local file.
    """
    local_filepath = determine_local_filepath(
        cache_dir, results_subdir, run_id, remote_filename, local_filename
    )
    if os.path.exists(local_filepath):
        lu.warning(f"File {local_filepath} already exists, skipping download.")
        return local_filepath

    artifact = try_fetch_artifact(artifact_name)
    if artifact is None:
        lu.error(
            f"Unable to find artifact {artifact_name} on WandB. Failed to download to {local_filepath}."
        )
        return None

    local_filepath.parent.mkdir(parents=True, exist_ok=True)
    artifact.download(root=local_filepath.parent.as_posix())

    if local_filename:
        downloaded_filepath = os.path.join(local_filepath.parent, remote_filename)
        os.rename(downloaded_filepath, local_filepath)

    lu.info(f"Downloaded file {remote_filename} for run {run_id} to {local_filepath}")
    return local_filepath


def download_checkpoint(run_id: str, load_step: int) -> Path:
    """Downloads checkpoint from WandB for run."""
    checkpoint_filename = f"step-{load_step:09d}.ckpt"
    checkpoint_path = download_file(
        run_id, checkpoint_filename, results_subdir="results/runs"
    )
    return checkpoint_path


def get_metric_from_config(metric_name: str, config: Dict) -> Any:
    try:
        value = config[metric_name]
        return value
    except:
        raise ValueError(f"Metric key {metric_name} was not found in config.")


def get_metric_from_run(metric_name: str, run: Union[RunResult, WandbPublicRun]) -> Any:
    try:
        config = run.config
    except:
        raise ValueError(f"Configuration is missing from {type(run)} Run object.")

    return get_metric_from_config(metric_name, config)


def compute_latest_checkpoint_load_step(run_result: RunResult) -> int:
    max_num_iterations = get_metric_from_run("max_num_iterations", run_result)
    load_step = max_num_iterations - 1

    return load_step


def download_latest_checkpoint(run_id: str) -> Path:
    """Downloads latest checkpoint from WandB for run."""
    run_result = fetch_run_result(run_id)
    load_step = compute_latest_checkpoint_load_step(run_result)

    return download_checkpoint(run_id, load_step)


def compose_artifact_name(run_id: str, artifact_suffix: str) -> str:
    return f"{run_id}-{artifact_suffix}"


def parse_wandb_table_json_to_dataframe(json_file_path):
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    # Extract the column names and the data
    columns = json_data["columns"]
    data = json_data["data"]

    # Create a DataFrame
    df = pd.DataFrame(data=data, columns=columns)

    return df

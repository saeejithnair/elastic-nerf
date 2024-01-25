import concurrent.futures
import json
import os
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gonas.utils.logging_utils as lu
import pandas as pd
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

    def __init__(self, run: WandbPublicRun, download_history: bool = False):
        self.run_id = run.id
        try:
            self.config = run.config
            self.state = run.state
            if download_history:
                self.history = self.download_history(run)
            self.summary = self.download_summary(run)
        except Exception as e:
            lu.error(f"An error occurred while fetching run {run.id}: {e}")
            raise e

    @staticmethod
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

        return history_df


def fetch_run_result(
    run_id: str,
    project_name: str = PROJECT_NAME,
    cache: bool = True,
    download_history: bool = False,
) -> RunResult:
    """Fetch a run from W&B."""

    # Define the path to the cache file
    cache_dir = Path(os.path.join(CACHE_DIR, "runs"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{run_id}_results.pkl")

    # If a cached result exists, load and return it
    if cache and os.path.exists(cache_file):
        lu.warning(f"Using cached results for run {run_id}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    run = fetch_run(run_id, project_name)
    run_result = RunResult(run, download_history=download_history)

    if cache:
        with open(cache_file, "wb") as f:
            pickle.dump(run_result, f)
    return run_result


def fetch_sweep(sweep_id: str, project_name: str = PROJECT_NAME) -> Sweep:
    """Fetch a sweep from W&B."""
    return api.sweep(f"{project_name}/{sweep_id}")


def fetch_sweep_results(
    sweep: Union[str, Sweep], cache: bool = True
) -> List[RunResult]:
    """Fetch summary data for all runs in a sweep."""
    if isinstance(sweep, str):
        sweep_id = sweep
        sweep = fetch_sweep(sweep)
    else:
        sweep_id = sweep.id

    # Define the path to the cache file
    cache_dir = Path(os.path.join(CACHE_DIR, "sweeps"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{sweep_id}_results.pkl")

    # If a cached result exists, load and return it
    if cache and os.path.exists(cache_file):
        lu.warning(f"Using cached results for sweep {sweep_id}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Otherwise, fetch the results
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fetch_run_result, run.id)
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
    # Cache the results
    if cache:
        with open(cache_file, "wb") as f:
            pickle.dump(runs, f)

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
    artifact_name: str, project=PROJECT, entity=ENTITY
) -> Union[Artifact, None]:
    try:
        artifact = api.artifact(f"{entity}/{project}/{artifact_name}:latest")
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

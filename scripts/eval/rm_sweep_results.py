import shutil
import tyro
from elastic_nerf.utils import wandb_utils as wu
from pathlib import Path
from tqdm import tqdm as tqdm


def remove_dir(path: Path):
    """Print out the directory path and remove it."""
    if path.exists():
        shutil.rmtree(path)


def main(sweep_id: str):
    """Remove results folder for each run in a sweep from the cache."""
    log_dir = Path("/home/user/shared/results/elastic-nerf")
    wandb_dir = Path("/home/user/shared/wandb_cache/elastic-nerf")
    sweep = wu.fetch_sweep(sweep_id)
    for run in tqdm(sweep.runs):
        remove_dir(log_dir / run.id)
        remove_dir(wandb_dir / run.id)


# %%
if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(main)

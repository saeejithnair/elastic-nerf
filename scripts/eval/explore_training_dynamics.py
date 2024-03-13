# %%
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from tqdm import tqdm as tqdm

import wandb
from elastic_nerf.nerfacc.radiance_fields.ngp import (
    NGPRadianceField,
    NGPRadianceFieldConfig,
)

# Add the elastic-nerf root directory to the python path
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from elastic_nerf.utils import dataset_utils as du
from elastic_nerf.utils import notebook_utils as nu
from elastic_nerf.utils import plotting_utils as pu
from elastic_nerf.utils import results_utils as ru
from elastic_nerf.utils import wandb_utils as wu
from elastic_nerf.utils.experiments.sweeps import SWEEPS
from tyro.extras._serialization import from_yaml

# pio.renderers.default = "plotly_mimetype+notebook_connected"
# pd.set_option("display.max_columns", None)
sweep_id = "6b4xxk3c"
"""
Goals:
- Visualize the training dynamics of the model
- We already have the training checkpoints saved to the cache directory
- Run Result should 

"""

# Get runs from sweep

# For each run in sweep, iterate through the runs and create Run Results

# Each RunResult should hold pointers to the weights_grads (optional) and checkpoints

# Each RunResult should also download the files from wandb (is this necessary?) Why waste
# bandwidth doing this since the files are already in the wandb cache directory?
# since that's where they were cached locally before being uploaded to wandb.
# So we should just use the files there if it's available, and only download from
# the server if its missing.

# We serialize the RunResult to disk

# Since each RunResult now also has the corresponding config, it should be easy
# to create a Trainer/Model that matches the exact configuration we used for
# the training (all the same model hyperparameters/architecture configs, etc.)
# Then using this identical model, we can load the weights from the correct checkpoint.

# So for each run, we iterate through all the saved steps, and load the weights/grads for
# each step to that model. And then we SHOULD query the model for N different
# elastic representations for each step. Now we _could_ get away with dynamically
# splicing the elastic weights similar to what we did during training/eval, but I
# really need to validate that my code is not buggy. Hence why I need to actually
# splice more compact elastic representations and then compute the dynamics/eval results
# for each one separately so that we know for sure there's no leakage going on.

# Then we need to plot the dynamics (for a collection of distance/norm metrics) as a function
# of training step and across various experiments.

# We need to stop thinking of things as a singular cache dir entity. We have
# multiple types of caches. We have the wandb cache, we have a results cache.
# Rename this shit.

"""
results_cache_dir: "/home/user/shared/results"
    - "/home/user/shared/results/elastic-nerf/x8yiupdo/checkpoints"
    - "/home/user/shared/results/elastic-nerf/x8yiupdo/checkpoints"
    - "/home/user/shared/results/elastic-nerf/x8yiupdo/config.yaml"
wandb_cache_dir: "/home/user/shared/wandb_cache"
    - "/home/user/shared/wandb_cache/elastic-nerf/x8yiupdo/wandb/debug-internal.log"
    - "/home/user/shared/wandb_cache/elastic-nerf/x8yiupdo/wandb/debug.log"
    - "/home/user/shared/wandb_cache/elastic-nerf/x8yiupdo/wandb/latest-run"
    - "/home/user/shared/wandb_cache/elastic-nerf/x8yiupdo/wandb/run-20240227_074310-x8yiupdo"
        - "/home/user/shared/wandb_cache/elastic-nerf/x8yiupdo/wandb/run-20240227_074310-x8yiupdo/files"
"""

sweep_id = "6b4xxk3c"
sweep = wu.fetch_sweep(sweep_id)

for run in sweep.runs:
    print(run.id)
# %%


def get_checkpoints(run_id: str, results_dir: Path):
    ckpt_dir = results_dir / run_id / "checkpoints"
    ckpt_files = list(ckpt_dir.glob("*.pt"))
    # Parse checkpoint files based on the step number.
    # E.g. "0tlyjl6x_ship_10000.pt" -> 10000
    ckpt_steps = [int(f.stem.split("_")[-1]) for f in ckpt_files]
    # Create a dictionary of step -> checkpoint file
    ckpt_dict = dict(zip(ckpt_steps, ckpt_files))

    return ckpt_dict


def get_weights_grads(run_id: str, results_dir: Path):
    weights_grads_dir = results_dir / run_id / "weights_grads"
    weights_grads_files = list(weights_grads_dir.glob("*.pt"))
    # Parse weights_grads files based on the step number and model.
    # E.g. "radiance_field_step_10500.pt" -> ('radiance_field', 10500)
    weights_grads_info = []
    for f in weights_grads_files:
        model, step = f.stem.split("_step_")
        step = int(step)
        weights_grads_info.append((model, step))

    # Create a dictionary of {model: {step -> weights_grads file}}
    weights_grads_dict = defaultdict(dict)
    for i, (model, step) in enumerate(weights_grads_info):
        weights_grads_dict[model][step] = weights_grads_files[i]

    return weights_grads_dict


def get_config(run_id: str, results_dir: Path):
    config_file = results_dir / run_id / "config.yaml"
    # Load yaml file from string
    with open(config_file, "r") as file:
        yaml_string = file.read()
    return yaml_string


# %%

import functools
import math
import os
import subprocess
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm
import tyro
from gonas.configs.base_config import InstantiateConfig, PrintableConfig
from lpips import LPIPS
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.scripts.downloads.download_data import BlenderDownload
from torchmetrics.functional import structural_similarity_index_measure
from tyro.extras._serialization import to_yaml

import wandb
from elastic_nerf.nerfacc.datasets.nerf_360_v2 import SubjectLoader as MipNerf360Loader
from elastic_nerf.nerfacc.datasets.nerf_synthetic import (
    SubjectLoader as BlenderSyntheticLoader,
)
from elastic_nerf.nerfacc.radiance_fields.ngp import (
    NGPRadianceField,
    NGPRadianceFieldConfig,
)
from elastic_nerf.nerfacc.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from elastic_nerf.utils import logging_utils as lu
from scripts.train.train_elastic_ngp_occ import NGPOccTrainer

set_random_seed(42)
results_cache_dir = Path("/home/user/shared/results/elastic-nerf")
results_dict = {}

run_id = "d85nyk68"
run_results = {}
run_results["checkpoints"] = get_checkpoints(run_id, results_cache_dir)
run_results["config"] = get_config(run_id, results_cache_dir)
run_results["weights_grads"] = get_weights_grads(run_id, results_cache_dir)
print(run_results)


# run_id = sweep.runs[0].id
log_dir = Path("/home/user/shared/results/elastic-nerf") / run_id
wandb_dir = Path("/home/user/shared/wandb_cache/elastic-nerf") / run_id
config = run_results["config"]
ckpt = run_results["weights_grads"]["radiance_field"][500]
trainer = NGPOccTrainer.load_trainer(config, log_dir=log_dir, wandb_dir=wandb_dir)

trainer.setup_logging()
# %%
# trainer.load_elastic_width(8)
# trainer.eval([8])
# # %%
# trainer.load_elastic_width(64)
# trainer.eval([64])
# %%
checkpoints = get_checkpoints(run_id, results_cache_dir)
trainer.config.model_path = checkpoints[20000]
trainer.load_checkpoint()
# %%
trainer.eval([64])
# %%
trainer.load_elastic_width(8)
trainer.eval([8])

# %%
trainer.load_elastic_width(12)
trainer.eval([12])

# %%
trainer.load_elastic_width(4)
trainer.eval([4])

# %%
trainer.load_elastic_width(7)
trainer.eval([7])

# %%
trainer.load_elastic_width(2)
trainer.eval([2])

# %%
trainer.load_elastic_width(37)
trainer.eval([37])

# %%

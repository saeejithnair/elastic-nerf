"""
Copyright (c) 2023 Saeejith Nair, University of Waterloo.
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import functools
import math
import os
import subprocess
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
from flask import g

import numpy as np
from sympy import false
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
from tyro.extras._serialization import to_yaml, from_yaml
import copy
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
from nerfacc.estimators.prop_net import PropNetEstimator, get_proposal_requires_grad_fn
from elastic_nerf.nerfacc.configs.datasets.base import (
    NGPBaseDatasetConfig,
)
from elastic_nerf.nerfacc.configs.datasets.blender import (
    BlenderSyntheticDatasetOccConfig,
    BlenderSyntheticDatasetPropConfig,
)
from elastic_nerf.nerfacc.configs.datasets.mipnerf360 import (
    MipNerf360DatasetOccConfig,
    MipNerf360DatasetPropConfig,
)


@dataclass
class NGPBaseTrainerConfig(PrintableConfig):
    """Configurations for training the model."""

    exp_date: str = field(default_factory=lambda: time.strftime("%Y-%m-%d-%H-%M-%S"))
    """The date of the experiment."""
    exp_name: str = field(
        default_factory=lambda: os.environ.get(
            "WANDB_RUN_ID", time.strftime("%Y-%m-%d-%H-%M-%S")
        )
    )
    """The name of the experiment."""
    project_name: str = "elastic-nerf"
    """The name of the project."""
    dataset_name: Literal["blender", "mipnerf360"] = "blender"
    """Which dataset to use."""
    dataset: Optional[NGPBaseDatasetConfig] = None
    """The dataset configuration. Will be dynamically set based on the dataset_name."""
    scene: str = "lego"
    """Which scene to use."""
    model_path: Optional[Path] = None
    """The path of the pretrained model."""
    seed: int = 42
    """The random seed."""
    sampling_strategy: Literal[
        "uniform",
        "exp-optimal",
        "exp-optimal-reverse",
        "exp",
        "exp-reverse",
        "matroyshka",
        "matroyshka-reverse",
        "sequential",
    ] = "uniform"
    """Sampling strategy for widths."""
    loss_weight_strategy: Literal[
        "uniform", "uniform-inv", "matroyshka", "matroyshka-inv", "exp", "exp-inv"
    ] = "uniform"
    """Loss upweighting strategy."""
    normalize_loss_weights: bool = False
    """Whether to normalize the loss weights."""
    hidden_dim: int = 64
    """The hidden dimension of the MLP."""
    num_train_widths: int = 4
    """Number of widths to use for training."""
    duplicate_train_batch_across_widths: bool = True
    """Whether to duplicate the training batch across different widths."""
    adjust_lr_for_duplicate_train_batch: bool = False
    """Whether to adjust the learning rate for duplicated training batch."""
    use_elastic_loss: bool = False
    """Whether to use elastic loss."""
    use_elastic_lr: bool = False
    """Whether to use elastic learning rate."""
    optimizer_lr: Optional[float] = 0.0725
    """The optimizer learning rate. This was tuned for the Counter scene."""
    use_mup: bool = True
    """Whether to use Maximal Update Parameterization."""
    num_widths_to_sample: int = 1
    """Number of widths to sample for each training step."""
    eval_elastic_widths: List[int] = field(default_factory=lambda: [64, 32, 16, 8])
    """Number of widths to use for evaluation."""
    num_eval_elastic_widths: Optional[int] = None
    """Number of widths to use for evaluation. If set, will use the first n widths used for training."""
    max_steps: int = 20000
    """Maximum number of training steps."""
    fused_eval: bool = True
    """Whether to convert elastic modules to TCNN Fused before eval."""
    num_eval_all_steps: int = 20000
    """Number of iterations after which to perform evaluation during training."""
    num_checkpoint_steps: int = 5000
    """Number of iterations after which to save a checkpoint."""
    num_log_steps: int = 1000
    """Number of iterations after which to log training information."""
    num_weights_grads_steps: int = 1000
    """Number of iterations after which to log weights and gradients."""
    weights_grads_warmup: int = 1000
    """Number of iterations to wait before logging weights and gradients less frequently."""
    device: str = "cuda:0"
    """The device to use."""
    log_dir: Optional[Path] = None
    """The directory to store the logs."""
    wandb_dir: Optional[Path] = None
    """The directory containing wandb cache."""
    host_machine: str = os.environ["HOSTNAME"]
    """Name of the host machine"""
    enable_logging: bool = True
    """Whether to enable logging."""

    def __post_init__(self):
        if self.log_dir is None:
            self.log_dir = (
                Path(os.environ.get("RESULTS_CACHE_DIR", "./results"))
                / self.project_name
                / self.exp_name
            )

        if self.wandb_dir is None:
            self.wandb_dir = (
                Path(os.environ.get("WANDB_CACHE_DIR", "./wandb_cache"))
                / self.project_name
                / self.exp_name
            )


class NGPTrainer:
    radiance_field: NGPRadianceField
    estimator: Union[OccGridEstimator, PropNetEstimator]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    grad_scaler: torch.cuda.amp.GradScaler
    models_to_watch: Dict[str, torch.nn.Module]
    frozen: Dict[str, Union[torch.nn.Module, Sequence[torch.nn.Module]]]

    def __init__(self, config: NGPBaseTrainerConfig):
        self.config = config
        self.device = torch.device(config.device)

    def setup(self):
        set_random_seed(self.config.seed)

        # Set up the training and testing datasets
        self.train_dataset, self.test_dataset = self.setup_datasets()

        # Set up the sampling schedule.
        self.setup_sampling_schedule()

        self.initialize_model()

        # Load parameters for radiance field, estimator, optimizer, and scheduler
        # from the checkpoint if specified.
        self.step = self.load_checkpoint()
        self.lpips_fn, self.ssim_fn = self.setup_evaluation_tools()

        # Set up wandb
        self.setup_logging()

        # Set up the frozen models for evaluation
        self.freeze()

    def setup_sampling_schedule(self):
        train_granularities = []

        for i in range(self.config.num_train_widths):
            train_granularities.append(self.config.hidden_dim // (2**i))
        self.train_elastic_widths = torch.tensor(train_granularities)

        if not self.use_elastic():
            assert (
                self.config.num_eval_elastic_widths == 1
            ), "Elastic widths to eval should be 1 for non-elastic models."

        # If we only want to evaluate on the first n widths corresponding to the
        # hidden dimension used for training (this is useful for benchmarking a
        # smaller width representation of the baseline architecture.)
        if self.config.num_eval_elastic_widths is not None:
            num_eval_elastic_widths = min(
                len(self.train_elastic_widths), self.config.num_eval_elastic_widths
            )
            self.eval_elastic_widths = self.train_elastic_widths[
                :num_eval_elastic_widths
            ]
        else:
            self.eval_elastic_widths = torch.tensor(self.config.eval_elastic_widths)

        # The sampling weights determine the probability of a elastic_width
        # being selected for a forward pass.
        self.elastic_width_sampling_weights: Optional[torch.Tensor] = (
            (
                self.get_elastic_width_sampling_weights(
                    num_granularities=len(self.train_elastic_widths),
                )
            )
            if self.config.sampling_strategy != "sequential"
            else None
        )

        # Keep track of how many samples we've seen for each elastic_width.
        # Create torch tensor that maps from elastic width to tensor index.
        # Initialize the elastic width indices and sample counts
        unique_elastic_widths, _ = torch.sort(
            torch.unique(
                torch.cat([self.eval_elastic_widths, self.train_elastic_widths])
            )
        )
        self.elastic_width_indices = unique_elastic_widths
        self.elastic_width_sample_counts = torch.zeros_like(self.elastic_width_indices)
        self.num_updates_skipped = {
            int(elastic_width): 0 for elastic_width in unique_elastic_widths
        }
        self.granularity_target_num_rays = {
            int(elastic_width): 0 for elastic_width in unique_elastic_widths
        }

        # Precompute the granularities to sample for each step
        # (helps minimize randomness between runs).
        self.sampling_schedule = []
        train_indices_to_sample = []
        for step in range(self.config.max_steps + 1):
            granularities_to_sample, granularity_loss_weights = (
                self.sample_granularities(step)
            )
            self.sampling_schedule.append(
                (granularities_to_sample, granularity_loss_weights)
            )
            # Precompute the train dataset indices for the granularities at each step.
            if self.config.duplicate_train_batch_across_widths:
                # Sample the same indices for each step across granularities.
                train_indices_to_sample.append(
                    torch.randint(0, len(self.train_dataset), (1,)).repeat(
                        1, len(granularities_to_sample)
                    )
                )
            else:
                # Sample different indices for each granularity on each step.
                train_indices_to_sample.append(
                    torch.randint(
                        0, len(self.train_dataset), (len(granularities_to_sample),)
                    ).unsqueeze(0)
                )

        self.train_indices_to_sample = torch.cat(train_indices_to_sample)

        self.validate_elastic_compatibility()

    def use_elastic(self):
        raise NotImplementedError

    def setup_logging(self):
        self.wandb_dir = self.config.wandb_dir
        self.wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = self.wandb_dir.as_posix()

        self.log_dir = self.config.log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Save serialized config to log directory.
        config_serialized = to_yaml(self.config)
        with open(self.log_dir / "config.yaml", "w") as f:
            f.write(config_serialized)

        # Save sampling schedule and training schedule to log directory.
        with open(self.log_dir / "schedules.pt", "wb") as f:
            torch.save(
                {
                    "sampling_schedule": self.sampling_schedule,
                    "train_indices_to_sample": self.train_indices_to_sample,
                },
                f,
            )

        config = asdict(self.config)
        mode = "disabled" if not self.config.enable_logging else None
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", self.config.project_name),
            dir=self.wandb_dir.as_posix(),
            name=os.environ.get("WANDB_NAME", self.config.exp_name),
            reinit=True,
            config=config,
            mode=mode,
        )
        # wandb.watch(list(self.models_to_watch.values()), log="all")

        self.exp_config_columns = {
            "Scene": self.config.scene,
            "Dataset": self.config.dataset_name,
            "Hidden Dim": self.config.hidden_dim,
            "Elastic": self.use_elastic(),
            "Train Widths": self.config.num_train_widths,
            "Sampling Strategy": self.config.sampling_strategy,
            "Num Samples": self.config.num_widths_to_sample,
        }

        self.eval_table_columns = [
            "Step",
            "Index",
            "Width",
            "PSNR",
            "SSIM",
            "LPIPS",
            "GT",
            "RGB",
            "Depth",
            "Error",
            "Acc",
        ]
        self.eval_summary_table_columns = [
            "Step",
            "Width",
            "PSNR Avg",
            "SSIM Avg",
            "LPIPS Avg",
        ]
        self.start_time = time.time()
        self.num_weights_grads_steps = 10

    def setup_datasets(self, num_rays_scaler: int = 1):
        """Setup training and testing datasets."""
        assert (
            self.config.dataset is not None
        ), "Dataset must be provided in the config."
        self.dataset: NGPBaseDatasetConfig = self.config.dataset
        # Check if dataset exists at provided path.
        # If not download it to its parent.
        if not self.dataset.data_root.exists():
            self.dataset.download()

        train_dataset = self.dataset.setup(
            subject_id=self.config.scene,
            root_fp=self.dataset.data_root.as_posix(),
            split=self.dataset.train_split,
            num_rays=self.dataset.init_batch_size // num_rays_scaler,
            device=self.device,
            **self.dataset.train_dataset_kwargs,
        )
        test_dataset = self.dataset.setup(
            subject_id=self.config.scene,
            root_fp=self.dataset.data_root.as_posix(),
            split="test",
            num_rays=None,
            device=self.device,
            **self.dataset.test_dataset_kwargs,
        )

        return train_dataset, test_dataset

    def get_aabb(self, estimator=None):
        raise NotImplementedError

    def validate_elastic_compatibility(self):
        raise NotImplementedError

    def initialize_model(self):
        raise NotImplementedError

    def load_checkpoint(self):
        """Load model from checkpoint if available."""
        if self.config.model_path is not None:
            checkpoint = torch.load(self.config.model_path)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.estimator.load_state_dict(checkpoint["estimator_state_dict"])

            for name, model in self.models_to_watch.items():
                model.load_state_dict(checkpoint[f"{name}_state_dict"])
            step = checkpoint["step"]
            print(f"Loaded checkpoint from {self.config.model_path} at step {step}")
        else:
            step = 0
        return step

    def setup_evaluation_tools(self):
        """Setup tools required for evaluation."""
        lpips_net = LPIPS(net="vgg").to(self.device)
        lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
        lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()
        ssim_fn = lambda x, y: structural_similarity_index_measure(
            x.unsqueeze(0).permute(0, 3, 1, 2), y.unsqueeze(0).permute(0, 3, 1, 2)
        )
        return lpips_fn, ssim_fn

    def get_elastic_width_sampling_weights(self, num_granularities: int):
        """Generates normalized weights for sampling widths."""
        weight_strategies = {
            "exp-optimal": lambda i: math.exp(0.1 * i),
            "exp": lambda i: math.exp(i),
            "matroyshka": lambda i: math.sqrt(2**i),
        }
        # Get the specified strategy or default to uniform.
        strategy = weight_strategies.get(
            self.config.sampling_strategy.replace("-reverse", ""), lambda i: 1
        )
        weights = torch.tensor([strategy(i) for i in range(num_granularities)])
        if "reverse" in self.config.sampling_strategy:
            weights = weights.flip(0)
        return weights / weights.sum()

    def update_elastic_width_sample_counts(self, sampled_granularities):
        for elastic_width in sampled_granularities:
            # Find the index in elastic_width_indices that corresponds to elastic_width
            index = torch.nonzero(self.elastic_width_indices == elastic_width).item()
            self.elastic_width_sample_counts[index] += 1

    def get_elastic_width_sample_counts(self, elastic_width):
        index = torch.nonzero(self.elastic_width_indices == elastic_width).item()
        return int(self.elastic_width_sample_counts[index])

    def log_metric(
        self,
        metric_key: str,
        metric_value: Any,
        axis_value: Optional[int] = None,
        axis_key: Optional[str] = None,
        commit=False,
    ):
        log_dict = {metric_key: metric_value}
        if axis_key is not None:
            assert (
                axis_value is not None
            ), f"axis_value must be provided for axis_key: {axis_key}"
            log_dict[axis_key] = axis_value

        wandb.log(log_dict, step=self.step, commit=commit)

    def log_metrics(
        self,
        metrics_dict,
        axis_value: Optional[int] = None,
        axis_key: Optional[str] = None,
        mode="Train",
        commit=False,
    ):
        elapsed_time = time.time() - self.start_time
        log_dict = {}

        for granularity_label in metrics_dict:
            for metric_name, metric_value in metrics_dict[granularity_label].items():
                log_dict[f"{mode}/{metric_name}/{granularity_label}"] = metric_value

        for elastic_width, sample_count in zip(
            self.elastic_width_indices, self.elastic_width_sample_counts
        ):
            elastic_width, sample_count = int(elastic_width), int(sample_count)
            granularity_label = f"elastic_{elastic_width}"
            log_dict[f"{mode}/num_sampled_times/{granularity_label}"] = sample_count
            log_dict[f"{mode}/num_updates_skipped/{granularity_label}"] = (
                self.num_updates_skipped[elastic_width]
            )
            log_dict[f"{mode}/target_num_rays/{granularity_label}"] = (
                self.granularity_target_num_rays[elastic_width]
            )

        log_dict[f"{mode}/elapsed_time"] = elapsed_time
        if axis_key is not None:
            assert (
                axis_value is not None
            ), f"axis_value must be provided for axis_key: {axis_key}"
            log_dict[axis_key] = axis_value

        if mode == "Train":
            log_dict.update({"Train/total_time": float(self.total_train_time)})

        wandb.log(log_dict, step=self.step, commit=commit)

    def preprocess_image(self, image):
        if len(image.shape) == 2:  # For images like 'error' with no channel dimension
            image = image.unsqueeze(2)  # Add a channel dimension
        if image.shape[2] == 1:  # For images with one channel
            image = image.repeat(1, 1, 3)  # Repeat along the channel dimension

        # Convert to [C, H, W]
        image = image.permute(2, 0, 1)

        # Rescale the image to be in the range [0, 255]
        if image.dtype == torch.float32:
            # Find the maximum value
            max_val = torch.max(image)
            # Rescale so that the max value is 255
            image = image / max_val * 255.0
            # Convert to uint8
            image = image.to(torch.uint8)

        return image

    def log_images(
        self,
        images_dict,
        axis_value: Optional[int] = None,
        axis_key: Optional[str] = None,
        mode="Train",
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        """Images will not be committed by default. Log a metric with commit=True to commit them."""
        preprocessed_images_dict = defaultdict(dict)
        for granularity_label in images_dict:
            images_list = []

            for image_name, image in images_dict[granularity_label].items():
                processed_image = self.preprocess_image(image)
                images_list.append(processed_image)
                preprocessed_images_dict[granularity_label][
                    image_name
                ] = processed_image

            # Create caption by joining the image names.
            caption = ", ".join(list(images_dict[granularity_label].keys()))
            image_grid = torchvision.utils.make_grid(images_list, nrow=len(images_list))

            log_dict: Dict[str, Any] = {
                f"{mode}/imgs/{granularity_label}": wandb.Image(
                    image_grid, caption=caption
                )
            }
            if axis_key is not None:
                assert (
                    axis_value is not None
                ), f"axis_value must be provided for axis_key {axis_key}"
                log_dict[axis_key] = axis_value

            wandb.log(
                log_dict,
                step=self.step,
                commit=False,
            )

        return preprocessed_images_dict

    def log_weights_and_gradients(self):
        # Log weights and gradients for the model asynchronously.
        checkpoints_dir = self.log_dir / "weights_grads"
        checkpoints_dir.mkdir(exist_ok=True, parents=True)
        for name, model in self.models_to_watch.items():
            params = model.state_dict()
            gradients = {
                name: p.grad.clone().detach()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            file_path = checkpoints_dir / f"{name}_step_{self.step}.pt"
            lu.robust_torch_save(
                {"step": self.step, "params": params, "gradients": gradients}, file_path
            )
            print(f"Saved weights and gradients at step {self.step} to {file_path}")

    def log_checkpoint(self):
        # Log checkpoint asynchronously.
        checkpoints_dir = self.log_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True, parents=True)
        checkpoint_fp = (
            checkpoints_dir
            / f"{self.config.exp_name}_{self.config.scene}_{self.step}.pt"
        )
        save_dict = {
            "step": self.step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "estimator_state_dict": self.estimator.state_dict(),
        }
        for name, model in self.models_to_watch.items():
            save_dict[f"{name}_state_dict"] = model.state_dict()

        lu.robust_torch_save(save_dict, checkpoint_fp)
        print(f"Saved checkpoint at step {self.step} to {checkpoint_fp}")

    def get_elastic_forward_kwargs(self, elastic_width, eval=False):
        raise NotImplementedError

    def render(self, rays, render_bkgd, **kwargs):
        raise NotImplementedError

    def eval_image(
        self, img_idx, elastic_width, **modules_for_eval
    ) -> Tuple[Dict, Dict]:
        data = self.test_dataset[img_idx]
        rays, pixels, render_bkgd = data["rays"], data["pixels"], data["color_bkgd"]

        elastic_width = int(elastic_width)
        kwargs = self.get_elastic_forward_kwargs(elastic_width, eval=True)
        rgb, acc, depth, extras = self.render(
            rays, render_bkgd, **modules_for_eval, **kwargs
        )

        if isinstance(extras, dict):
            # Last argument is dict for propnet, and n_rendering_samples
            # for occ.
            n_rendering_samples = math.nan
        else:
            n_rendering_samples = extras

        loss, mse_loss, psnr = self.compute_losses(rgb, pixels)
        lpips = self.lpips_fn(rgb, pixels)
        ssim = self.ssim_fn(rgb, pixels)
        metrics_dict = {
            "loss": float(loss.item()),
            "mse_loss": float(mse_loss.item()),
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),  # type: ignore
            "lpips": float(lpips.item()),
            "max_depth": float(depth.max()),
            "num_rays": int(len(pixels)),
            "n_rendering_samples": n_rendering_samples,
        }
        images_dict = {
            "gt": pixels.cpu(),
            "rgb": rgb.cpu(),
            "acc": acc.cpu(),
            "depth": depth.cpu(),
            "error": (rgb - pixels).norm(dim=-1).cpu(),
        }

        return metrics_dict, images_dict

    def load_width(self, elastic_width, module):
        """Load the model with a specific width."""
        module_copy = copy.deepcopy(module)
        if hasattr(module, "load_width"):
            module_copy.load_width(elastic_width, load_fused=self.config.fused_eval)

        return module_copy.to(self.device)

    def probe_models(self):
        print(f"Probing models...")
        for name, module in self.models_to_watch.items():
            for n, p in module.named_parameters():
                if "layer" not in n or "norm" in n:
                    continue
                if p.ndim == 1:
                    p = p.unsqueeze(-1)
                norm = torch.linalg.matrix_norm(p, ord=2).item()
                var = p.var().item()
                print(name, n, p.shape, f"Norm: {norm:.3f}", f"Var: {var:.3f}")

    def get_modules_for_eval(self, elastic_width):
        modules_for_eval = {}
        for name, module in self.frozen.items():
            if isinstance(module, Sequence):
                modules_for_eval[name] = [
                    self.load_width(elastic_width, m).eval() for m in module
                ]
            elif isinstance(module, torch.nn.Module):
                modules_for_eval[name] = self.load_width(elastic_width, module).eval()
            else:
                raise ValueError(f"Unknown module type {type(module)}")

        return modules_for_eval

    def eval_width(self, elastic_width: int):
        psnrs = defaultdict(list)
        lpips = defaultdict(list)
        ssims = defaultdict(list)
        times = defaultdict(list)

        self.set_mode(train=False)

        modules_for_eval = self.get_modules_for_eval(elastic_width)
        print(f"Evaluating with elastic width {elastic_width}...")

        for i in tqdm.tqdm(
            range(len(self.test_dataset)), desc="Test Dataset", leave=True
        ):
            granularity_label = f"elastic_{elastic_width}"
            start_time = time.time()
            metrics_dict, images_dict = self.eval_image(
                i, elastic_width, **modules_for_eval
            )
            times[granularity_label].append(time.time() - start_time)
            psnrs[granularity_label].append(metrics_dict["psnr"])
            lpips[granularity_label].append(metrics_dict["lpips"])
            ssims[granularity_label].append(metrics_dict["ssim"])

            if i == 0:
                preprocessed_images_dict = self.log_images(
                    {granularity_label: images_dict},
                    axis_value=i,
                    axis_key=f"Test Image/{granularity_label}",
                    mode="Eval Results",
                )

                assert preprocessed_images_dict is not None

            self.log_metrics(
                {granularity_label: metrics_dict},
                axis_value=i,
                axis_key=f"Test Image/{granularity_label}",
                mode="Eval Results",
                commit=False,
            )

        return psnrs, lpips, ssims, times

    @torch.no_grad()
    def eval(self, eval_elastic_widths=None):
        psnrs_history = {}
        lpips_history = {}
        ssim_history = {}
        elapsed_times = {}

        if eval_elastic_widths is None:
            eval_elastic_widths = self.eval_elastic_widths

        eval_summary_table = wandb.Table(
            columns=self.eval_summary_table_columns
            + list(self.exp_config_columns.keys())
        )
        # Freeze modules for evaluation. This is so that we can load the
        # copies of the frozen modules at different widths for evaluation
        # without affecting the already registered training parameters in the optimizer.
        self.freeze()

        for elastic_width in eval_elastic_widths:
            psnrs, lpips, ssims, times = self.eval_width(int(elastic_width))
            psnrs_history.update(psnrs)
            lpips_history.update(lpips)
            ssim_history.update(ssims)
            elapsed_times.update(times)

        avg_metrics_dict = {}
        for granularity_label in psnrs_history:
            avg_metrics_dict[granularity_label] = {
                "psnr_avg": np.mean(psnrs_history[granularity_label]),
                "lpips_avg": np.mean(lpips_history[granularity_label]),
                "ssim_avg": np.mean(ssim_history[granularity_label]),
                "time_avg": np.mean(elapsed_times[granularity_label]),
            }

        print("Eval Results Summary")
        print(avg_metrics_dict)
        eval_summary_table = self.log_to_table(
            avg_metrics_dict,
            eval_summary_table,
            self.eval_summary_table_columns,
        )
        wandb.log(
            {f"Eval Results Summary/table": eval_summary_table},
            step=self.step,
            commit=False,
        )
        self.log_metrics(
            avg_metrics_dict,
            axis_value=len(self.test_dataset),
            axis_key="Test Image",
            mode="Eval Results Summary",
            commit=False,
        )

    def log_to_table(
        self,
        metrics_dict: Dict[str, Dict[str, Any]],
        table: wandb.Table,
        columns: List[str],
        images_dict: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        index: Optional[int] = None,
    ) -> wandb.Table:
        """Log metrics and images to the wandb table."""
        for width, metrics in metrics_dict.items():
            images = images_dict[width] if images_dict else {}
            data = []
            for col in columns:
                col = col.lower().replace(" ", "_")
                if col == "step":
                    data.append(self.step)
                elif col == "index":
                    data.append(index)
                elif col == "width":
                    data.append(width)
                elif col in metrics:
                    data.append(metrics[col])
                elif col in images:
                    data.append(wandb.Image(images[col]))
                else:
                    raise ValueError(f"Invalid column {col}")
            data += list(self.exp_config_columns.values())
            table.add_data(*data)

        return table

    def step_check(self, step, step_size, run_at_zero=False) -> bool:
        """Returns true based on current step and step interval."""
        if step_size == 0:
            return False
        return (run_at_zero or step != 0) and step % step_size == 0

    def train(self):
        """Train the model."""
        # Log initial weights and gradients.
        self.log_weights_and_gradients()
        self.log_checkpoint()

        pbar = tqdm.tqdm(
            total=self.config.max_steps + 1, desc="Training Steps", leave=True
        )
        self.total_train_time = 0
        start_step = self.step
        for step in range(start_step, self.config.max_steps):
            # Perform a single training step and increment self.step.
            train_start = time.time()
            metrics_dict, gradient_updated = self.train_step()
            self.total_train_time += time.time() - train_start

            if self.step_check(self.step, self.num_weights_grads_steps):
                # Log weights and gradients at specified intervals.
                self.log_weights_and_gradients()

                if self.step < self.config.weights_grads_warmup:
                    if self.step < 100:
                        self.num_weights_grads_steps = 10
                    elif self.step < 500:
                        self.num_weights_grads_steps = 50
                    elif self.step < 1000:
                        self.num_weights_grads_steps = 100
                    else:
                        self.num_weights_grads_steps = 500
                elif self.step == self.config.weights_grads_warmup:
                    self.num_weights_grads_steps = self.config.num_weights_grads_steps

            if self.step_check(self.step, self.config.num_checkpoint_steps):
                # Save a checkpoint at specified intervals.
                self.log_checkpoint()

            if step % 100 == 0:
                pbar.update(
                    100
                    if step + 100 <= self.config.max_steps + 1
                    else self.config.max_steps + 1 - step
                )

            # Perform evaluation at specified intervals.
            if self.step_check(self.step, self.config.num_eval_all_steps):
                self.eval()

            if (
                self.step_check(self.step, self.config.num_log_steps)
                or not gradient_updated
            ):
                # Log metrics at specified intervals or if gradient was not
                # updated because loss was 0.
                self.log_metrics(metrics_dict, commit=True)

            self.step += 1

        pbar.close()

        # Final checkpoint logging and evaluation.
        self.log_weights_and_gradients()
        self.log_checkpoint()
        self.eval()
        wandb.log({}, step=self.step, commit=True)

    def get_train_data_idx(self, step: int, granularity_idx: int) -> int:
        return int(self.train_indices_to_sample[step, granularity_idx].item())

    def get_train_data(self, train_data_idx: int):
        data = self.train_dataset[train_data_idx]
        rays, pixels, render_bkgd = data["rays"], data["pixels"], data["color_bkgd"]

        return rays, pixels, render_bkgd

    def train_granular_step(
        self, elastic_width: int, granularity_loss_weight: float
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        raise NotImplementedError

    def train_step(
        self,
    ) -> Tuple[Dict[str, Union[float, int]], bool]:
        raise NotImplementedError

    def compute_losses(self, rgb, pixels):
        """Compute losses based on the outputs from a forward pass."""
        loss = F.smooth_l1_loss(rgb, pixels)
        mse_loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse_loss) / np.log(10.0)
        return loss, mse_loss, psnr

    def get_loss_weight(self, elastic_width, num_widths_to_sample):
        matroyshka_weights_map = {
            8: 1,
            16: math.sqrt(2),
            32: 2,
            64: math.sqrt(8),
        }
        matroyshka_inv_weights_map = {
            8: 1.0 / math.sqrt(8),
            16: 1.0 / 2.0,
            32: 1.0 / math.sqrt(2),
            64: 1.0,
        }
        exp_weights = [math.exp(i) for i in range(4)]
        widths = [8, 16, 32, 64]
        exp_weights_map = {width: weight for width, weight in zip(widths, exp_weights)}
        exp_inv_weights_map = {
            width: 1.0 / weight for width, weight in zip(widths, reversed(exp_weights))
        }
        if self.config.loss_weight_strategy == "uniform":
            return 1 / num_widths_to_sample
        elif self.config.loss_weight_strategy == "uniform-inv":
            return 1
        elif self.config.loss_weight_strategy == "matroyshka":
            return matroyshka_weights_map[int(elastic_width)]
        elif self.config.loss_weight_strategy == "matroyshka-inv":
            return matroyshka_inv_weights_map[int(elastic_width)]
        elif self.config.loss_weight_strategy == "exp":
            return exp_weights_map[int(elastic_width)]
        elif self.config.loss_weight_strategy == "exp-inv":
            return exp_inv_weights_map[int(elastic_width)]
        else:
            raise ValueError(
                f"Invalid loss weight strategy: {self.config.loss_weight_strategy}"
            )

    def compute_lr(self, optimizer_lr):
        """Adjusts the learning rate based on the number of samples."""
        if (
            self.config.duplicate_train_batch_across_widths
            and self.config.adjust_lr_for_duplicate_train_batch
        ):
            # If we're duplicating the training batch across widths, this
            # may be the same as decreasing the batch size.
            num_widths_to_sample = min(
                len(self.train_elastic_widths), self.config.num_widths_to_sample
            )
            return optimizer_lr / float(num_widths_to_sample)
        else:
            return optimizer_lr

    def sample_granularities(self, step: int):
        """Sample widths for training."""
        if self.config.sampling_strategy == "sequential":
            # Sequentially sample the widths at each step.
            sampling_idx = step % len(self.train_elastic_widths)
            elastic_width = self.train_elastic_widths[sampling_idx]
            # For sequential, we only sample a single width.
            granularity_loss_weight = self.get_loss_weight(
                elastic_width, num_widths_to_sample=1
            )
            return (
                torch.tensor([elastic_width]),
                torch.tensor([granularity_loss_weight]),
            )

        assert self.elastic_width_sampling_weights is not None, (
            "Elastic width sampling weights must be provided "
            "for non-sequential sampling strategies."
        )
        num_widths_to_sample = min(
            len(self.train_elastic_widths), self.config.num_widths_to_sample
        )
        elastic_width_indices = torch.multinomial(
            self.elastic_width_sampling_weights,
            num_widths_to_sample,
            replacement=False,
        )
        granularities_to_sample = self.train_elastic_widths[elastic_width_indices]

        granularity_loss_weights = []
        for elastic_width in granularities_to_sample:
            granularity_loss_weights.append(
                self.get_loss_weight(elastic_width, num_widths_to_sample)
            )

        granularity_loss_weights = torch.tensor(granularity_loss_weights)
        if self.config.normalize_loss_weights:
            granularity_loss_weights = (
                granularity_loss_weights / granularity_loss_weights.sum()
            )
        return granularities_to_sample, granularity_loss_weights

    def set_mode(self, train: bool = True):
        """Set the mode of the model."""
        self.estimator.train(train)
        for name, model in self.models_to_watch.items():
            model.train(train)
            for p in model.parameters():
                p.requires_grad = train

    def load_weights_grads(self, weights_grads_path: Path, module_name: str):
        ckpt = torch.load(weights_grads_path)
        try:
            module = self.models_to_watch[module_name]
            module.load_state_dict(ckpt["params"])
        except Exception as e:
            print(
                f"Error loading weights and grads for {module_name} from {ckpt.keys()} into module {module}"
            )
            raise e

        for name, param in module.named_parameters():
            if name in ckpt["gradients"]:
                grad = ckpt["gradients"][name]
                try:
                    grad = grad.to(param.device)
                    grad = grad.to(param.dtype)
                    param.grad = grad
                except Exception as e:
                    print(
                        f"param.dtype = {param.dtype}, grad = {grad}, param.grad = {param.grad}"
                    )
                    raise e

    def freeze(self):
        """Saves a deepcopy of models to be used for evaluation."""
        raise NotImplementedError

    @staticmethod
    def load_trainer(
        config: str,
        log_dir: Path,
        wandb_dir: Path,
        config_type: Type,
        ckpt_path: Optional[Path] = None,
    ):
        # Load model from config
        trainer_config = from_yaml(config_type, config)
        trainer_config.log_dir = log_dir
        trainer_config.wandb_dir = wandb_dir
        trainer_config.enable_logging = False
        if ckpt_path is not None:
            trainer_config.model_path = ckpt_path
        trainer = trainer_config.setup()

        return trainer

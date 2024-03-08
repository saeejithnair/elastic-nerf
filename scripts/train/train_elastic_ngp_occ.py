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

set_random_seed(42)


@dataclass
class NGPDatasetConfig(PrintableConfig):
    """Dataset/scene specific configurations."""

    data_root: Path
    """The root directory of the dataset."""
    scene: str
    """Which scene to use."""
    init_batch_size: int = 1024
    """Initial batch size."""
    target_sample_batch_size: int = 1 << 18
    """Target sample batch size."""
    weight_decay: float = 0.0
    """Weight decay."""
    optimizer_lr: float = 1e-2
    """Learning rate for the optimizer."""
    optimizer_eps: float = 1e-15
    """Epsilon for the optimizer."""
    scheduler_start_factor: float = 0.01
    """Start factor for the scheduler."""
    scheduler_total_iters: int = 100
    """Total iterations for the scheduler."""
    scheduler_gamma: float = 0.33
    """Gamma for the scheduler."""
    aabb_coeffs: Tuple[float, float, float, float, float, float] = (
        -1.5,
        -1.5,
        -1.5,
        1.5,
        1.5,
        1.5,
    )
    """Coefficients for the AABB."""
    near_plane: float = 0.0
    """Near plane."""
    far_plane: float = 1.0e10
    """Far plane."""
    train_dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the train dataset."""
    test_dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the test dataset."""
    grid_resolution: int = 128
    """Resolution of the grid."""
    grid_nlvl: int = 1
    """Number of levels of the grid."""
    render_step_size: float = 5e-3
    """Step size for rendering."""
    alpha_thre: float = 0.0
    """Threshold for alpha."""
    cone_angle: float = 0.0
    """Cone angle."""
    train_split: Literal["train", "trainval"] = "train"
    """Which train split to use."""
    test_chunk_size: int = 8192
    """Chunk size for testing."""
    num_dynamic_batch_warmup_steps: int = 20
    """Number of warmup steps for dynamic batch size."""

    def get_dataloader(self):
        raise NotImplementedError


@dataclass
class BlenderSyntheticDatasetConfig(NGPDatasetConfig):
    """Dataset/scene specific configurations for Blender Synthetic dataset."""

    data_root: Path = field(
        default_factory=lambda: Path(os.environ["NERFSTUDIO_CACHE_DIR"])
        / "data/blender"
    )
    """The root directory of the dataset."""
    scene: Literal[
        "chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"
    ] = "lego"
    """Which scene to use."""

    def __post_init__(self):
        self.weight_decay = (
            1e-5 if self.scene in ["materials", "ficus", "drums"] else 1e-6
        )

    def setup(self, **kwargs) -> BlenderSyntheticLoader:
        return BlenderSyntheticLoader(**kwargs)


@dataclass
class MipNerf360DatasetConfig(NGPDatasetConfig):
    """Dataset/scene specific configurations for Mip-NeRF 360 dataset."""

    data_root: Path = field(
        default_factory=lambda: Path(os.environ["NERFSTUDIO_CACHE_DIR"]) / "data/360_v2"
    )
    """The root directory of the dataset."""
    scene: Literal[
        "garden",
        "bicycle",
        "bonsai",
        "counter",
        "kitchen",
        "room",
        "stump",
    ] = "bicycle"
    """Which scene to use."""
    aabb_coeffs: Tuple[float, float, float, float, float, float] = (
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
    )
    """Coefficients for the AABB."""
    near_plane: float = 0.2
    """Near plane."""
    train_dataset_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"color_bkgd_aug": "random", "factor": 4}
    )
    """Keyword arguments for the train dataset."""
    test_dataset_kwargs: Dict[str, Any] = field(default_factory=lambda: {"factor": 4})
    """Keyword arguments for the test dataset."""
    grid_nlvl: int = 4
    """Number of levels of the grid."""
    render_step_size: float = 1e-3
    """Step size for rendering."""
    alpha_thre: float = 1e-2
    """Threshold for alpha."""
    cone_angle: float = 0.004
    """Cone angle."""

    def setup(self, **kwargs) -> MipNerf360Loader:
        return MipNerf360Loader(**kwargs)


@dataclass
class NGPOccTrainerConfig(InstantiateConfig):
    """Configurations for training the model."""

    _target: Type = field(default_factory=lambda: NGPOccTrainer)
    """The target class to instantiate."""

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
    dataset: Optional[NGPDatasetConfig] = None
    """The dataset configuration. Will be dynamically set based on the dataset_name."""
    scene: str = "lego"
    """Which scene to use."""
    model_path: Optional[Path] = None
    """The path of the pretrained model."""
    sampling_strategy: Literal[
        "uniform",
        "exp-optimal",
        "exp-optimal-reverse",
        "exp",
        "exp-reverse",
        "matroyshka",
        "matroyshka-reverse",
    ] = "exp-reverse"
    """Sampling strategy for widths."""
    hidden_dim: int = 64
    """The hidden dimension of the MLP."""
    num_train_widths: int = 4
    """Number of widths to use for training."""
    num_widths_to_sample: int = 1
    """Number of widths to sample for each training step."""
    eval_elastic_widths: List[int] = field(default_factory=lambda: [64, 32, 16, 8])
    """Number of widths to use for evaluation."""
    max_steps: int = 20000
    """Maximum number of training steps."""
    num_eval_all_steps: int = 5000
    """Number of iterations after which to perform evaluation during training."""
    num_checkpoint_steps: int = 10000
    """Number of iterations after which to save a checkpoint."""
    num_log_steps: int = 500
    """Number of iterations after which to log training information."""
    radiance_field: NGPRadianceFieldConfig = field(
        default_factory=lambda: NGPRadianceFieldConfig()
    )
    """The configuration for the elastic MLP."""
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

        if self.dataset_name == "blender":
            self.dataset = BlenderSyntheticDatasetConfig(scene=self.scene)
        elif self.dataset_name == "mipnerf360":
            self.dataset = MipNerf360DatasetConfig(scene=self.scene)
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")


class NGPOccTrainer:
    def __init__(self, config: NGPOccTrainerConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set up the training and testing datasets
        self.train_dataset, self.test_dataset = self.setup_datasets()

        aabb = torch.tensor([*self.dataset.aabb_coeffs], device=self.device)
        self.estimator = OccGridEstimator(
            roi_aabb=aabb,
            resolution=self.dataset.grid_resolution,
            levels=self.dataset.grid_nlvl,
        ).to(self.device)
        # Set up the radiance field, optimizer and scheduler
        (
            self.radiance_field,
            self.optimizer,
            self.scheduler,
            self.grad_scaler,
        ) = self.initialize_model()

        # Load parameters for radiance field, estimator, optimizer, and scheduler
        # from the checkpoint if specified.
        self.step = self.load_checkpoint()
        self.lpips_fn, self.ssim_fn = self.setup_evaluation_tools()

        train_granularities = []
        if not config.radiance_field.use_elastic:
            assert config.num_train_widths == 1
            assert config.num_widths_to_sample == 1
            assert config.eval_elastic_widths == [config.hidden_dim]

        for i in range(config.num_train_widths):
            train_granularities.append(config.hidden_dim // (2**i))
        self.train_elastic_widths = torch.tensor(train_granularities)
        self.eval_elastic_widths = torch.tensor(config.eval_elastic_widths)

        # The sampling weights determine the probability of a elastic_width
        # being selected for a forward pass.
        self.elastic_width_sampling_weights: torch.Tensor = (
            self.get_elastic_width_sampling_weights(
                num_granularities=len(self.train_elastic_widths),
            )
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

        # Set up wandb
        if self.config.enable_logging:
            self.setup_logging()

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

        config = asdict(self.config)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", self.config.project_name),
            dir=self.wandb_dir.as_posix(),
            name=os.environ.get("WANDB_NAME", self.config.exp_name),
            reinit=True,
            config=config,
        )
        self.models_to_watch = {
            "radiance_field": self.radiance_field,
        }
        wandb.watch(self.radiance_field, log="gradients", log_graph=True, log_freq=500)

        self.exp_config_columns = {
            "Scene": self.config.scene,
            "Dataset": self.config.dataset_name,
            "Hidden Dim": self.config.hidden_dim,
            "Elastic": self.config.radiance_field.use_elastic,
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

    def setup_datasets(self):
        """Setup training and testing datasets."""
        self.dataset: Union[BlenderSyntheticDatasetConfig, MipNerf360DatasetConfig] = (
            self.config.dataset
        )
        # Check if dataset exists at provided path.
        # If not download it to its parent.
        if not self.dataset.data_root.exists():
            if isinstance(self.dataset, BlenderSyntheticDatasetConfig):
                # Download Blender dataset.
                downloader = BlenderDownload(save_dir=self.dataset.data_root.parent)
                downloader.download(save_dir=self.dataset.data_root.parent)
            elif isinstance(self.dataset, MipNerf360DatasetConfig):
                # Download MipNerf360 dataset TODO
                dataset_url = (
                    "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
                )
                zip_path = Path(self.dataset.data_root.parent, "360_v2.zip")
                subprocess.run(["wget", "-O", str(zip_path), dataset_url], check=True)

                # Unzip the file to self.dataset.data_root.parent / "360_v2"
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    self.dataset.data_root.mkdir(parents=True, exist_ok=True)
                    zip_ref.extractall(self.dataset.data_root)

                # Optionally, delete the zip file after extraction
                zip_path.unlink()
            else:
                raise ValueError(
                    f"Unknown dataset type {type(self.dataset)} for dataset {self.dataset.data_root.as_posix()}."
                )

        train_dataset = self.dataset.setup(
            subject_id=self.config.scene,
            root_fp=self.dataset.data_root.as_posix(),
            split=self.dataset.train_split,
            num_rays=self.dataset.init_batch_size,
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

    def initialize_model(self):
        """Initialize the radiance field and optimizer."""
        grad_scaler = torch.cuda.amp.GradScaler(2**10)
        radiance_field: NGPRadianceField = self.config.radiance_field.setup(
            aabb=self.estimator.aabbs[-1]
        ).to(self.device)
        optimizer = torch.optim.Adam(
            radiance_field.parameters(),
            lr=self.dataset.optimizer_lr,
            eps=self.dataset.optimizer_eps,
            weight_decay=self.dataset.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.dataset.scheduler_start_factor,
                    total_iters=self.dataset.scheduler_total_iters,
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        self.config.max_steps // 2,
                        self.config.max_steps * 3 // 4,
                        self.config.max_steps * 9 // 10,
                    ],
                    gamma=self.dataset.scheduler_gamma,
                ),
            ]
        )

        return radiance_field, optimizer, scheduler, grad_scaler

    def load_checkpoint(self):
        """Load model from checkpoint if available."""
        if self.config.model_path is not None:
            checkpoint = torch.load(self.config.model_path)
            self.radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.estimator.load_state_dict(checkpoint["estimator_state_dict"])
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
            self.log_weights_and_gradients()

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
        """Log weights and gradients for the models to WandB."""
        checkpoints_dir = self.log_dir / "weights_grads"
        checkpoints_dir.mkdir(exist_ok=True, parents=True)
        for name, model in self.models_to_watch.items():
            # Extract the state_dict for parameters
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
            print(f"Saved weights and gradients for model '{name}' to '{file_path}'")

    def log_checkpoint(self):
        checkpoints_dir = self.log_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True, parents=True)
        checkpoint_fp = (
            checkpoints_dir
            / f"{self.config.exp_name}_{self.config.scene}_{self.step}.pt"
        )
        lu.robust_torch_save(
            {
                "step": self.step,
                "radiance_field_state_dict": self.radiance_field.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "estimator_state_dict": self.estimator.state_dict(),
            },
            checkpoint_fp,
        )
        print(f"Saved model to {checkpoint_fp}")

    @torch.no_grad()
    def eval_image(self, img_idx, elastic_width) -> Tuple[Dict, Dict]:
        data = self.test_dataset[img_idx]
        rays, pixels, render_bkgd = data["rays"], data["pixels"], data["color_bkgd"]

        elastic_width = int(elastic_width)
        kwargs = {}
        if self.config.radiance_field.use_elastic:
            kwargs["active_neurons"] = elastic_width
        rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
            self.radiance_field,
            self.estimator,
            rays,
            # rendering options
            near_plane=self.dataset.near_plane,
            render_step_size=self.dataset.render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=self.dataset.cone_angle,
            alpha_thre=self.dataset.alpha_thre,
            # test options
            test_chunk_size=self.dataset.test_chunk_size,
            **kwargs,
        )
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

    def eval_width(self, elastic_width: int):
        psnrs = defaultdict(list)
        lpips = defaultdict(list)
        ssims = defaultdict(list)
        self.radiance_field.eval()
        self.estimator.eval()

        for i in tqdm.tqdm(
            range(len(self.test_dataset)), desc="Test Dataset", leave=True
        ):
            granularity_label = f"elastic_{elastic_width}"
            metrics_dict, images_dict = self.eval_image(i, elastic_width)

            psnrs[granularity_label].append(metrics_dict["psnr"])
            lpips[granularity_label].append(metrics_dict["lpips"])
            ssims[granularity_label].append(metrics_dict["ssim"])

            if i % 10 == 0:
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

        return psnrs, lpips, ssims

    @torch.no_grad()
    def eval(self, eval_elastic_widths=None):
        psnrs_history = {}
        lpips_history = {}
        ssim_history = {}

        if eval_elastic_widths is None:
            eval_elastic_widths = self.eval_elastic_widths

        eval_summary_table = wandb.Table(
            columns=self.eval_summary_table_columns
            + list(self.exp_config_columns.keys())
        )
        for elastic_width in eval_elastic_widths:
            torch.cuda.empty_cache()
            psnrs, lpips, ssims = self.eval_width(int(elastic_width))
            psnrs_history.update(psnrs)
            lpips_history.update(lpips)
            ssim_history.update(ssims)

        avg_metrics_dict = {}
        for granularity_label in psnrs_history:
            avg_metrics_dict[granularity_label] = {
                "psnr_avg": np.mean(psnrs_history[granularity_label]),
                "lpips_avg": np.mean(lpips_history[granularity_label]),
                "ssim_avg": np.mean(ssim_history[granularity_label]),
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
        for step in range(self.step, self.config.max_steps):
            # Perform a single training step and increment self.step.
            metrics_dict, gradient_updated = self.train_step()

            if (
                self.step_check(self.step, self.config.num_log_steps)
                or not gradient_updated
            ):
                # Log metrics at specified intervals or if gradient was not
                # updated because loss was 0.
                self.log_metrics(metrics_dict, commit=False)

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

            wandb.log({}, step=self.step)
            self.step += 1

        pbar.close()

        # Final checkpoint logging and evaluation.
        self.log_weights_and_gradients()
        self.log_checkpoint()
        self.eval()
        wandb.log({}, step=self.step)

    def train_granular_step(
        self, elastic_width: int, granularity_loss_weight: float
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Perform a single training step on a single width."""
        if self.dataset.target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = self.granularity_target_num_rays[elastic_width]
            num_rays = (
                num_rays
                if (
                    num_rays > 0
                    and (
                        self.get_elastic_width_sample_counts(elastic_width)
                        > self.dataset.num_dynamic_batch_warmup_steps
                    )
                )
                else (self.dataset.init_batch_size // self.config.num_widths_to_sample)
            )
            self.train_dataset.update_num_rays(num_rays)

        data = self.train_dataset[
            torch.randint(0, len(self.train_dataset), (1,)).item()
        ]
        rays, pixels, render_bkgd = data["rays"], data["pixels"], data["color_bkgd"]

        kwargs = {}
        if self.config.radiance_field.use_elastic:
            kwargs["active_neurons"] = elastic_width
        rgb, _, depth, n_rendering_samples = render_image_with_occgrid(
            self.radiance_field,
            self.estimator,
            rays,
            # rendering options
            near_plane=self.dataset.near_plane,
            render_step_size=self.dataset.render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=self.dataset.cone_angle,
            alpha_thre=self.dataset.alpha_thre,
            **kwargs,
        )

        if n_rendering_samples == 0:
            metrics = {
                "n_rendering_samples": int(n_rendering_samples),
                "num_rays": int(len(pixels)),
            }
            return None, metrics

        loss, mse_loss, psnr = self.compute_losses(rgb, pixels)
        loss = loss * granularity_loss_weight
        metrics = {
            "loss": float(loss),
            "mse_loss": float(mse_loss),
            "psnr": float(psnr),
            "max_depth": float(depth.max()),
            "num_rays": int(len(pixels)),
            "n_rendering_samples": n_rendering_samples,
        }

        num_rays = int(
            len(pixels)
            * (
                (
                    self.dataset.target_sample_batch_size
                    / self.config.num_widths_to_sample
                )
                / n_rendering_samples
            )
        )
        self.granularity_target_num_rays[int(elastic_width)] = num_rays

        return loss, metrics

    def train_step(
        self,
    ) -> Tuple[Dict[str, Union[float, int]], bool]:
        """Perform a single training step."""
        self.radiance_field.train()
        self.estimator.train()
        granularities_to_sample = self.sample_granularities()
        granularity_loss_weight = 1 / len(granularities_to_sample)

        def occ_eval_fn(x):
            if not self.config.radiance_field.use_elastic:
                density = self.radiance_field.query_density(x)
                return density * self.dataset.render_step_size

            density_sum = None
            count = 0
            # Compute the mean density across all elastic widths.
            for elastic_width in granularities_to_sample:
                density = self.radiance_field.query_density(
                    x, active_neurons=int(elastic_width)
                )
                if density_sum is None:
                    density_sum = density
                else:
                    density_sum = torch.add(density_sum, density)
                count += 1

            mean_density = density_sum / count
            return mean_density * self.dataset.render_step_size

        self.estimator.update_every_n_steps(
            step=self.step, occ_eval_fn=occ_eval_fn, occ_thre=1e-2
        )

        loss_dict = {}
        metrics_dict = {}
        for elastic_width in granularities_to_sample:
            granularity_label = f"elastic_{elastic_width}"
            torch.cuda.empty_cache()
            loss, metrics = self.train_granular_step(
                int(elastic_width), granularity_loss_weight
            )
            metrics_dict[granularity_label] = metrics

            if loss is None:
                # No points were sampled for this width.
                gradient_updated = False
                for width in granularities_to_sample:
                    self.num_updates_skipped[int(width)] += 1

                del loss_dict
                return metrics_dict, gradient_updated

            loss_dict[granularity_label] = loss

        loss = functools.reduce(torch.add, loss_dict.values())
        self.optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        self.grad_scaler.scale(loss).backward()
        self.optimizer.step()
        self.scheduler.step()
        gradient_updated = True
        self.update_elastic_width_sample_counts(granularities_to_sample)

        return metrics_dict, gradient_updated

    def compute_losses(self, rgb, pixels):
        """Compute losses based on the outputs from a forward pass."""
        loss = F.smooth_l1_loss(rgb, pixels)
        mse_loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse_loss) / np.log(10.0)
        return loss, mse_loss, psnr

    def sample_granularities(self):
        """Sample widths for training."""
        num_widths_to_sample = min(
            len(self.train_elastic_widths), self.config.num_widths_to_sample
        )
        elastic_width_indices = torch.multinomial(
            self.elastic_width_sampling_weights,
            num_widths_to_sample,
            replacement=False,
        )
        return self.train_elastic_widths[elastic_width_indices]

    def load_weights_grads(self, weights_grads_path: Path, module_name: str):
        ckpt = torch.load(weights_grads_path)
        module = getattr(self, module_name)
        module.load_state_dict(ckpt["params"])

        for name, param in module.named_parameters():
            if name in ckpt["gradients"]:
                param.grad = ckpt["gradients"][name]

    @staticmethod
    def load_trainer(
        config: str,
        log_dir: Path,
        wandb_dir: Path,
        ckpt_path: Optional[Path] = None,
    ) -> "NGPOccTrainer":
        # Load model from config
        trainer_config: NGPOccTrainerConfig = from_yaml(NGPOccTrainerConfig, config)
        trainer_config.log_dir = log_dir
        trainer_config.wandb_dir = wandb_dir
        trainer_config.enable_logging = False
        if ckpt_path is not None:
            trainer_config.model_path = ckpt_path
        trainer = trainer_config.setup()

        return trainer

    def load_elastic_width(self, elastic_width: int):
        """Load the model with the specified elastic width."""
        if not hasattr(self, "full_width_radiance_field"):
            self.full_width_radiance_field = copy.deepcopy(
                self.radiance_field.to(torch.device("cpu"))
            )

        new_width_elastic_net = (
            self.full_width_radiance_field.mlp_base.elastic_mlp.get_sliced_net(
                elastic_width
            )
        )
        self.radiance_field.mlp_base.elastic_mlp = new_width_elastic_net

        self.radiance_field.to(self.device)


def main(config: NGPOccTrainerConfig):
    trainer: NGPOccTrainer = config.setup()
    error_occurred = False  # Track if an error occurred

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Interrupted by user, finishing up...")
        error_occurred = True  # Update the flag
        raise  # Re-raise the KeyboardInterrupt exception
    except Exception as e:  # Catch other exceptions
        print(f"An error occurred: {e}")
        error_occurred = True  # Update the flag
        raise  # Re-raise the exception
    finally:
        exit_code = 1 if error_occurred else 0  # Determine the exit code
        wandb.finish(exit_code=exit_code)  # Pass the exit code


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
                tyro.conf.FlagConversionOff[NGPOccTrainerConfig]
            ],
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()

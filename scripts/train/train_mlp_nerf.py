"""
Copyright (c) 2023 Saeejith Nair, University of Waterloo.
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import functools
import math
import os
import time
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

import wandb
from elastic_nerf.nerfacc.datasets.nerf_synthetic import SubjectLoader
from elastic_nerf.nerfacc.radiance_fields.mlp import (
    VanillaNeRFRadianceField,
    VanillaNeRFRadianceFieldConfig,
    get_field_granular_state_dict,
)
from elastic_nerf.nerfacc.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)

set_random_seed(42)


@dataclass
class TrainerConfig(InstantiateConfig):
    """Configurations for training the model."""

    _target: Type = field(default_factory=lambda: Trainer)
    """The target class to instantiate."""

    exp_name: str = field(default_factory=lambda: time.strftime("%Y-%m-%d-%H-%M-%S"))
    """The name of the experiment."""
    project_name: str = "elastic-nerf"
    """The name of the project."""
    data_root: Path = field(
        default_factory=lambda: Path(os.environ["NERFSTUDIO_CACHE_DIR"])
        / "data/blender"
    )
    """The root directory of the dataset."""
    train_split: Literal["train", "trainval"] = "train"
    """Which train split to use."""
    model_path: Optional[Path] = None
    """The path of the pretrained model."""
    scene: Literal[
        "chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"
    ] = "lego"
    """Which scene to use."""
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
    hidden_dim: int = 256
    """The hidden dimension of the MLP."""
    test_chunk_size: int = 4096
    """The chunk size for testing."""
    init_batch_size: int = 1024
    """Initial batch size."""
    target_sample_batch_size: int = 1 << 16
    """Target sample batch size."""
    num_train_widths: int = 6
    """Number of widths to use for training."""
    num_widths_to_sample: int = 1
    """Number of widths to sample for each training step."""
    eval_elastic_widths: List[int] = field(
        default_factory=lambda: [256, 128, 64, 32, 16, 8]
    )
    """Number of widths to use for evaluation."""
    max_steps: int = 50000
    """Maximum number of training steps."""
    num_eval_all_steps: int = 50000
    """Number of iterations after which to perform evaluation during training."""
    num_checkpoint_steps: int = 10000
    """Number of iterations after which to save a checkpoint."""
    num_log_steps: int = 500
    """Number of iterations after which to log training information."""
    radiance_field: VanillaNeRFRadianceFieldConfig = field(
        default_factory=lambda: VanillaNeRFRadianceFieldConfig()
    )
    """The configuration for the elastic MLP."""
    near_plane: float = 0.0
    """The near plane of the camera."""
    far_plane: float = 1.0e10
    """The far plane of the camera."""
    grid_resolution: int = 128
    """The resolution of the grid."""
    grid_nlvl: int = 1
    """The number of levels of the grid."""
    render_step_size: float = 5e-3
    """The step size for rendering."""
    device: str = "cuda:0"
    """The device to use."""
    log_dir: Path = field(
        default_factory=lambda: Path(os.environ["NERFSTUDIO_CACHE_DIR"]) / "wandb_cache"
    )
    """The directory to store the logs."""
    host_machine: str = os.environ["HOSTNAME"]
    """Name of the host machine"""


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set up the training and testing datasets
        self.train_dataset, self.test_dataset = self.setup_datasets()

        aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=self.device)
        self.estimator = OccGridEstimator(
            roi_aabb=aabb,
            grid_resolution=self.config.grid_resolution,
            grid_nlvl=self.config.grid_nlvl,
            near_plane=self.config.near_plane,
            far_plane=self.config.far_plane,
        ).to(self.device)
        # Set up the radiance field, optimizer and scheduler
        self.radiance_field, self.optimizer, self.scheduler = self.initialize_model()

        # Load parameters for radiance field, estimator, optimizer, and scheduler
        # from the checkpoint if specified.
        self.step = self.load_checkpoint()
        self.lpips_fn, self.ssim_fn = self.setup_evaluation_tools()

        train_granularities = []
        if not config.radiance_field.mlp.use_elastic:
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
        # The keys for this should be the eval widths so that in our
        # logs, we can see for certain that the non train widths have
        # 0 samples.
        self.elastic_width_sample_counts = {
            int(elastic_width): 0
            for elastic_width in torch.unique(
                torch.cat([self.eval_elastic_widths, self.train_elastic_widths])
            )
        }
        self.num_updates_skipped = {
            int(elastic_width): 0
            for elastic_width in torch.unique(
                torch.cat([self.eval_elastic_widths, self.train_elastic_widths])
            )
        }

        # Set up wandb
        self.setup_wandb()

    def setup_wandb(self):
        self.log_dir = self.config.log_dir / self.config.exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", self.config.project_name),
            dir=os.environ.get("WANDB_DIR", self.log_dir.as_posix()),
            name=os.environ.get("WANDB_NAME", self.config.exp_name),
            reinit=True,
            config=asdict(self.config),
        )
        wandb.watch(self.radiance_field, log="all", log_graph=True, log_freq=500)

        self.eval_table_columns = [
            "Step",
            "Index",
            "Granularity",
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
            "Granularity",
            "PSNR Avg",
            "SSIM Avg",
            "LPIPS Avg",
        ]

    def setup_datasets(self):
        """Setup training and testing datasets."""
        # Check if dataset exists at provided path.
        # If not download it to its parent.
        if not self.config.data_root.exists():
            # Download Blender dataset.
            downloader = BlenderDownload(save_dir=self.config.data_root.parent)
            downloader.download(save_dir=self.config.data_root.parent)

        train_dataset = SubjectLoader(
            subject_id=self.config.scene,
            root_fp=self.config.data_root.as_posix(),
            split=self.config.train_split,
            num_rays=self.config.init_batch_size,
            device=self.device,
        )
        test_dataset = SubjectLoader(
            subject_id=self.config.scene,
            root_fp=self.config.data_root.as_posix(),
            split="test",
            num_rays=None,
            device=self.device,
        )
        return train_dataset, test_dataset

    def initialize_model(self):
        """Initialize the radiance field and optimizer."""
        radiance_field: VanillaNeRFRadianceField = (
            self.config.radiance_field.setup().to(self.device)
        )
        optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                self.config.max_steps // 2,
                self.config.max_steps * 3 // 4,
                self.config.max_steps * 5 // 6,
                self.config.max_steps * 9 // 10,
            ],
            gamma=0.33,
        )

        return radiance_field, optimizer, scheduler

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
            self.elastic_width_sample_counts[int(elastic_width)] += 1

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

        for elastic_width in self.elastic_width_sample_counts:
            granularity_label = f"elastic_{elastic_width}"
            log_dict[f"{mode}/num_sampled_times/{granularity_label}"] = (
                self.elastic_width_sample_counts[elastic_width]
            )
            log_dict[f"{mode}/num_updates_skipped/{granularity_label}"] = (
                self.num_updates_skipped[elastic_width]
            )

        log_dict[f"{mode}/elapsed_time"] = elapsed_time
        if axis_key is not None:
            assert (
                axis_value is not None
            ), f"axis_value must be provided for axis_key: {axis_key}"
            log_dict[axis_key] = axis_value
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

    def log_checkpoint(self):
        checkpoints_dir = self.config.log_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True, parents=True)
        checkpoint_fp = (
            checkpoints_dir
            / f"{self.config.exp_name}_{self.config.scene}_{self.step}.pt"
        )
        torch.save(
            {
                "step": self.step,
                "radiance_field_state_dict": self.radiance_field.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "estimator_state_dict": self.estimator.state_dict(),
            },
            checkpoint_fp,
        )
        wandb.save(str(checkpoint_fp))
        print(f"Saved model to {checkpoint_fp}")

    @torch.no_grad()
    def eval_image(self, img_idx) -> Tuple[Dict, Dict]:
        self.radiance_field.eval()
        self.estimator.eval()
        data = self.test_dataset[img_idx]
        rays, pixels, render_bkgd = data["rays"], data["pixels"], data["color_bkgd"]
        metrics_dict = {}
        images_dict = {}

        # Rendering for different widths.
        for elastic_width in tqdm.tqdm(
            self.eval_elastic_widths, desc="Granular Widths", leave=False
        ):
            torch.cuda.empty_cache()
            elastic_width = int(elastic_width)
            granularity_label = f"elastic_{elastic_width}"
            kwargs = {}
            if self.config.radiance_field.mlp.use_elastic:
                kwargs["active_neurons"] = elastic_width
            rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
                self.radiance_field,
                self.estimator,
                rays,
                # rendering options
                near_plane=self.config.near_plane,
                render_step_size=self.config.render_step_size,
                render_bkgd=render_bkgd,
                # test options
                test_chunk_size=self.config.test_chunk_size,
                **kwargs,
            )
            loss, mse_loss, psnr = self.compute_losses(rgb, pixels)
            lpips = self.lpips_fn(rgb, pixels)
            ssim = self.ssim_fn(rgb, pixels)
            metrics_dict[granularity_label] = {
                "loss": float(loss.item()),
                "mse_loss": float(mse_loss.item()),
                "psnr": float(psnr.item()),
                "ssim": float(ssim.item()),  # type: ignore
                "lpips": float(lpips.item()),
                "max_depth": float(depth.max()),
                "num_rays": int(len(pixels)),
                "n_rendering_samples": n_rendering_samples,
            }
            images_dict[granularity_label] = {
                "gt": pixels.cpu(),
                "rgb": rgb.cpu(),
                "acc": acc.cpu(),
                "depth": depth.cpu(),
                "error": (rgb - pixels).norm(dim=-1).cpu(),
            }

        return metrics_dict, images_dict

    @torch.no_grad()
    def eval(self):
        psnrs_history = defaultdict(list)
        lpips_history = defaultdict(list)
        ssim_history = defaultdict(list)

        eval_table = wandb.Table(columns=self.eval_table_columns)
        eval_summary_table = wandb.Table(columns=self.eval_summary_table_columns)
        for i in tqdm.tqdm(
            range(len(self.test_dataset)), desc="Test Dataset", leave=True
        ):
            metrics_dict, images_dict = self.eval_image(i)

            for granularity_label in metrics_dict:
                psnrs_history[granularity_label].append(
                    metrics_dict[granularity_label]["psnr"]
                )
                lpips_history[granularity_label].append(
                    metrics_dict[granularity_label]["lpips"]
                )
                ssim_history[granularity_label].append(
                    metrics_dict[granularity_label]["ssim"]
                )

            if i % 10 == 0:
                preprocessed_images_dict = self.log_images(
                    images_dict,
                    axis_value=i,
                    axis_key="Test Image",
                    mode="Eval Results",
                )

                assert preprocessed_images_dict is not None
                eval_table = self.log_to_table(
                    metrics_dict,
                    images_dict=preprocessed_images_dict,
                    table=eval_table,
                    columns=self.eval_table_columns,
                    index=i,
                )
            self.log_metrics(
                metrics_dict,
                axis_value=i,
                axis_key="Test Image",
                mode="Eval Results",
                commit=False,
            )
        wandb.log(
            {f"Eval Results/table": eval_table},
            step=self.step,
            commit=False,
        )

        avg_metrics_dict = {}
        for granularity_label in psnrs_history:
            avg_metrics_dict[granularity_label] = {
                "psnr_avg": np.mean(psnrs_history[granularity_label]),
                "lpips_avg": np.mean(lpips_history[granularity_label]),
                "ssim_avg": np.mean(ssim_history[granularity_label]),
            }

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
        for granularity, metrics in metrics_dict.items():
            images = images_dict[granularity] if images_dict else {}
            data = []
            for col in columns:
                col = col.lower().replace(" ", "_")
                if col == "step":
                    data.append(self.step)
                elif col == "index":
                    data.append(index)
                elif col == "granularity":
                    data.append(granularity)
                elif col in metrics:
                    data.append(metrics[col])
                elif col in images:
                    data.append(wandb.Image(images[col]))
                else:
                    raise ValueError(f"Invalid column {col}")
            table.add_data(*data)

        return table

    def step_check(self, step, step_size, run_at_zero=False) -> bool:
        """Returns true based on current step and step interval."""
        if step_size == 0:
            return False
        return (run_at_zero or step != 0) and step % step_size == 0

    def train(self):
        """Train the model."""
        self.start_time = time.time()
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
                # Log metrics at specified intervals.
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
        self.log_checkpoint()
        self.eval()
        wandb.log({}, step=self.step)

    def train_step(
        self,
    ) -> Tuple[Dict[str, Union[float, int]], bool]:
        """Perform a single training step."""
        self.radiance_field.train()
        self.estimator.train()

        data = self.train_dataset[
            torch.randint(0, len(self.train_dataset), (1,)).item()
        ]
        rays, pixels, render_bkgd = data["rays"], data["pixels"], data["color_bkgd"]
        granularities_to_sample = self.sample_granularities()
        granularity_loss_weight = 1 / len(granularities_to_sample)

        def occ_eval_fn(x):
            if not self.config.radiance_field.mlp.use_elastic:
                return (
                    self.radiance_field.query_density(x) * self.config.render_step_size
                )

            density_sum = None
            count = 0
            # Compute the mean density across all granular widths.
            for elastic_width in granularities_to_sample:
                density = self.radiance_field.query_density(
                    x, active_neurons=int(elastic_width)
                )
                if density_sum is None:
                    density_sum = density
                else:
                    density_sum = torch.add(density_sum, density)
                count += 1

            # Compute the mean density.
            mean_density = density_sum / count
            return mean_density * self.config.render_step_size

        self.estimator.update_every_n_steps(
            step=self.step, occ_eval_fn=occ_eval_fn, occ_thre=1e-2
        )

        loss_dict = {}
        metrics_dict = {}
        for elastic_width in granularities_to_sample:
            torch.cuda.empty_cache()
            elastic_width = int(elastic_width)
            granularity_label = f"elastic_{elastic_width}"

            kwargs = {}
            if self.config.radiance_field.mlp.use_elastic:
                kwargs["active_neurons"] = elastic_width
            rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
                self.radiance_field,
                self.estimator,
                rays,
                # rendering options
                near_plane=self.config.near_plane,
                render_step_size=self.config.render_step_size,
                render_bkgd=render_bkgd,
                **kwargs,
            )

            if n_rendering_samples == 0:
                metrics_dict[granularity_label] = {
                    "n_rendering_samples": int(n_rendering_samples),
                    "num_rays": int(len(pixels)),
                }
                gradient_updated = False
                for width in granularities_to_sample:
                    self.num_updates_skipped[int(width)] += 1

                del (
                    loss_dict,
                    pixels,
                    rgb,
                    rays,
                    render_bkgd,
                    acc,
                    depth,
                    n_rendering_samples,
                )
                return metrics_dict, gradient_updated

            loss, mse_loss, psnr = self.compute_losses(rgb, pixels)
            loss_dict[granularity_label] = loss * granularity_loss_weight
            metrics_dict[granularity_label] = {
                "loss": float(loss),
                "mse_loss": float(mse_loss),
                "psnr": float(psnr),
                "max_depth": float(depth.max()),
                "num_rays": int(len(pixels)),
                "n_rendering_samples": n_rendering_samples,
            }

        if self.config.target_sample_batch_size > 0:
            # dynamic batch size for rays to keep sample batch size constant.
            mean_n_rendering_samples = sum(
                [
                    metrics_dict[granularity_label]["n_rendering_samples"]
                    for granularity_label in metrics_dict
                ]
            ) / float(len(metrics_dict))
            num_rays = len(pixels)
            num_rays = int(
                num_rays
                * (self.config.target_sample_batch_size / mean_n_rendering_samples)
            )
            self.train_dataset.update_num_rays(num_rays)

        loss = functools.reduce(torch.add, loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        gradient_updated = True
        self.update_elastic_width_sample_counts(granularities_to_sample)

        del pixels, rgb, rays, render_bkgd, acc, depth, n_rendering_samples

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


def main(config: TrainerConfig):
    trainer: Trainer = config.setup()
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
                tyro.conf.FlagConversionOff[TrainerConfig]
            ],
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()

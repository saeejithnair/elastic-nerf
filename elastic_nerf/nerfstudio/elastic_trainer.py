# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple, Type, cast

import torch
from torch.cuda.amp.grad_scaler import GradScaler

import wandb
from rich import box, style
from rich.panel import Panel
from rich.table import Table

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter

# Updated viewer imports for Nerfstudio 1.1.4
from nerfstudio.viewer.viewer import Viewer as ViewerState  # New Viewer
from nerfstudio.viewer_legacy.server.viewer_state import (
    ViewerLegacyState,
)  # Deprecated Legacy Viewer

TRAIN_INTERATION_OUTPUT = Tuple[
    torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
]
TORCH_DEVICE = str


@dataclass
class ElasticTrainerConfig(TrainerConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: ElasticTrainer)
    """target class to instantiate"""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images_quick: Optional[int] = None
    """Number of steps between eval all images (quick)."""
    steps_per_eval_all_images: int = 70000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    use_grad_scaler: bool = False
    """Use gradient scaler even if the automatic mixed precision is disabled."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    load_checkpoint: Optional[Path] = None
    """Path to checkpoint file."""
    log_gradients: bool = False
    """Optionally log gradients during training"""
    gradient_accumulation_steps: Dict[str, int] = field(default_factory=lambda: {})
    """Number of steps to accumulate gradients over."""


class ElasticTrainer(Trainer):
    """ElasticTrainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(
        self, config: ElasticTrainerConfig, local_rank: int = 0, world_size: int = 1
    ) -> None:
        self.train_lock = Lock()
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device: TORCH_DEVICE = config.machine.device_type
        if self.device == "cuda":
            self.device += f":{local_rank}"
        self.mixed_precision: bool = self.config.mixed_precision
        self.use_grad_scaler: bool = self.mixed_precision or self.config.use_grad_scaler
        self.training_state: Literal["training", "paused", "completed"] = "training"
        self.gradient_accumulation_steps: Dict[str, int] = (
            self.config.gradient_accumulation_steps
        )

        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step: int = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)

        self.base_dir: Path = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir: Path = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")

        self.viewer_state = None

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = self.setup_optimizers()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info

        # Optionally support Legacy Viewer if needed
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerLegacyState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Legacy viewer at: {self.viewer_state.viewer_url}"]

        self._check_viewer_warnings()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
                trainer=self,  # Ensure trainer is passed if required
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging,
            max_iter=self.config.max_num_iterations,
            banner_messages=banner_messages,
        )
        writer.put_config(
            name="config", config_dict=dataclasses.asdict(self.config), step=0
        )
        profiler.setup_profiler(self.config.logging, writer_log_path)
        self.setup_wandb()

    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers

        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        param_groups = self.pipeline.get_param_groups()
        camera_optimizer_config = self.config.pipeline.datamanager.camera_optimizer
        if (
            camera_optimizer_config is not None
            and camera_optimizer_config.mode != "off"
        ):
            assert camera_optimizer_config.param_group not in optimizer_config
            optimizer_config[camera_optimizer_config.param_group] = {
                "optimizer": camera_optimizer_config.optimizer,
                "scheduler": camera_optimizer_config.scheduler,
            }
        return Optimizers(optimizer_config, param_groups)

    def setup_wandb(self) -> None:
        """Configures wandb if enabled."""
        if not self.config.is_wandb_enabled():
            return

        self.setup_model_watcher()
        self.wandb_tables = {}
        self.wandb_table_columns = {}
        self.wandb_val_table_queue = defaultdict(list)
        self.wandb_val_table_expected_groups = ["Eval Images", "Eval Batch"]

    def setup_model_watcher(self) -> None:
        """Configures wandb to log the model weights and gradients."""
        if self.config.is_wandb_enabled():
            wandb.watch(self.pipeline.model, log="all", log_graph=True, log_freq=200)
        else:
            raise ValueError("Wandb must be enabled to use model watcher.")

    def update_val_table(self, table_type: str, **kwargs) -> None:
        """Updates the val overview table in wandb if enabled.
        Args:
            table_name: name of the table to update
            push_changes: whether to push changes to wandb
            kwargs: key-value pairs to update in table
        """
        self.wandb_val_table_queue[table_type].append(kwargs)

        # If the table_type is not yet in the wandb_tables, it means
        # that we haven't seen all the expected val table groups yet.
        # This is because different groups execute after different number
        # of training steps.
        if table_type not in self.wandb_tables:
            # Add the kwarg keys to the table columns if they are not already there
            self.wandb_table_columns["Validation Overview"] = list(
                set(
                    self.wandb_table_columns["Validation Overview"]
                    + list(kwargs.keys())
                )
            )
            # Check if all the expected val types are in the table queue
            if all(
                len(self.wandb_val_table_queue[val_type]) > 0
                for val_type in self.wandb_val_table_expected_groups
            ):
                self.wandb_tables[table_type] = wandb.Table(
                    columns=self.wandb_table_columns["Validation Overview"]
                )

        # Check if all the expected val types are in the table queue
        if all(
            len(self.wandb_val_table_queue[val_type]) > 0
            for val_type in self.wandb_val_table_expected_groups
        ):
            # Now populate the table with the queued data
            rows = defaultdict(list)
            for val_type in self.wandb_val_table_expected_groups:
                group_data_over_steps: List[Dict] = self.wandb_val_table_queue[val_type]
                for group_data in group_data_over_steps:
                    rows[group_data["step"]].append(group_data)

            for step, row in rows.items():
                row_data = {}
                for group in row:
                    row_data.update(group)

                for key in self.wandb_table_columns["Validation Overview"]:
                    if key not in row_data:
                        row_data[key] = None
                self.wandb_tables[table_type].add_data(**row_data)
            # Reset the table queue
            self.wandb_val_table_queue = defaultdict(list)
            wandb.log({"Validation Overview": self.wandb_tables[table_type]})

    def train(self) -> None:
        """Train the model."""
        assert (
            self.pipeline.datamanager.train_dataset is not None
        ), "Missing DatasetInputs"

        # don't want to call save_dataparser_transform if pipeline's datamanager does not have a dataparser
        if isinstance(self.pipeline.datamanager, VanillaDataManager):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )

        self._init_viewer_state()

        # train_table_initialized = False
        # eval_table_initialized = False
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            step = 0
            for step in range(self._start_step, self._start_step + num_iterations):
                while self.training_state == "paused":
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(
                        writer, EventName.ITER_TRAIN_TIME, step=step
                    ) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step,
                                location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION,
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step,
                                location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION,
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(
                    step, self.config.logging.steps_per_log, run_at_zero=True
                ):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(
                        name="Train Loss Dict", scalar_dict=loss_dict, step=step
                    )
                    writer.put_dict(
                        name="Train Metrics Dict", scalar_dict=metrics_dict, step=step
                    )
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)",
                        scalar=torch.cuda.max_memory_allocated() / (1024**2),
                        step=step,
                    )

                    # Uncomment and adjust if you want to retain training overview logging with wandb
                    # if self.config.is_wandb_enabled():
                    #     if not train_table_initialized:
                    #         train_table_initialized = True
                    #         train_table_columns = (
                    #             ["step", "Train Loss"]
                    #             + list(loss_dict.keys())
                    #             + list(metrics_dict.keys())
                    #         )
                    #         self.wandb_tables["Training Overview"] = wandb.Table(
                    #             columns=train_table_columns
                    #         )

                    #     wandb_train_table_data = [step, loss]
                    #     for k in train_table_columns:
                    #         if k in loss_dict and k in metrics_dict:
                    #             raise ValueError(
                    #                 f"Duplicate key ({k}) in both loss dict and metrics_dict"
                    #             )

                    #         if k in loss_dict:
                    #             wandb_train_table_data.append(loss_dict[k])
                    #         elif k in metrics_dict:
                    #             wandb_train_table_data.append(metrics_dict[k])
                    #         else:
                    #             wandb_train_table_data.append(None)

                    #     print(train_table_columns)
                    #     print(wandb_train_table_data)
                    #     self.wandb_tables["Training Overview"].add_data(
                    #         *wandb_train_table_data
                    #     )
                    #     wandb.log(
                    #         {
                    #             "Training Overview": self.wandb_tables[
                    #                 "Training Overview"
                    #             ]
                    #         },
                    #         step=step,
                    #     )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(
            Panel(
                table,
                title="[bold][green]:tada: Training Finished :tada:[/bold]",
                expand=False,
            )
        )

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(
                step=step, location=TrainingCallbackLocation.AFTER_TRAIN
            )

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """Helper to print out any warnings regarding the way the viewer/loggers are enabled"""
        if (
            (self.config.is_viewer_enabled() or self.config.is_viewer_legacy_enabled())
            and not self.config.is_tensorboard_enabled()
            and not self.config.is_wandb_enabled()
            and not self.config.is_comet_enabled()
        ):
            string: str = (
                "[NOTE] Not running eval iterations since only viewer is enabled.\n"
                "Use [yellow]--vis {wandb, tensorboard, viewer+wandb, viewer+tensorboard}[/yellow] to run with eval."
            )
            CONSOLE.print(f"{string}")

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state and self.pipeline.datamanager.train_dataset
        self.viewer_state.init_scene(
            train_dataset=self.pipeline.datamanager.train_dataset,
            train_state="training",
            eval_dataset=self.pipeline.datamanager.eval_dataset,
        )

    @check_viewer_enabled
    def _update_viewer_state(self, step: int) -> None:
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        num_rays_per_batch: int = self.pipeline.datamanager.get_train_rays_per_batch()
        try:
            self.viewer_state.update_scene(step, num_rays_per_batch)
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")

    @check_viewer_enabled
    def _train_complete_viewer(self) -> None:
        """Let the viewer know that the training is complete"""
        assert self.viewer_state is not None
        self.training_state = "completed"
        try:
            self.viewer_state.training_complete()
        except RuntimeError:
            time.sleep(0.03)  # sleep to allow buffer to reset
            CONSOLE.log("Viewer failed. Continuing training.")
        CONSOLE.print("Use ctrl+c to quit", justify="center")
        while True:
            time.sleep(0.01)

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(
        self, train_t: TimeWriter, vis_t: TimeWriter, step: int
    ) -> None:
        """Performs update on rays/sec calculation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch: int = (
            self.pipeline.datamanager.get_train_rays_per_batch()
        )
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=self.world_size
            * train_num_rays_per_batch
            / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(
                    int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir)
                )[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scaler
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert (
                load_checkpoint.exists()
            ), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scaler
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path: Path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {
                    k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()
                },
                "schedulers": {
                    k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()
                },
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        needs_zero = [
            group
            for group in self.optimizers.parameters.keys()
            if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        needs_step = [
            group
            for group in self.optimizers.parameters.keys()
            if step % self.gradient_accumulation_steps[group]
            == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore

    @profiler.time_function
    def log_elastic_params(self, step: int) -> None:
        """Log elastic parameters and heatmaps."""
        (
            elastic_weight_norms_dict,
            elastic_weight_heatmaps,
        ) = self.pipeline.get_elastic_params_and_heatmaps()
        writer.put_dict(
            name="Elastic Params Dict",
            scalar_dict=elastic_weight_norms_dict,
            step=step,
        )
        group = "Elastic Params"
        for image_name, image in elastic_weight_heatmaps.items():
            writer.put_image(name=group + "/" + image_name, image=image, step=step)

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(
                step=step
            )
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(
                name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step
            )
            writer.put_dict(
                name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step
            )
            # eval_batch_data = eval_loss_dict.copy()
            # eval_batch_data.update(eval_metrics_dict)
            # eval_batch_data["Eval Loss"] = eval_loss
            # self.update_val_table(table_type="Eval Batch", step=step, **eval_batch_data)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                (
                    metrics_dict,
                    images_dict,
                ) = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(
                name="Eval Images Metrics", scalar_dict=metrics_dict, step=step
            )
            group = "Eval Images"
            # eval_image_data = metrics_dict.copy()
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        if step_check(step, self.config.steps_per_eval_image, run_at_zero=True):
            self.log_elastic_params(step)
            # self.update_val_table(
            #     table_type="Eval Images", step=step, **eval_batch_data
            # )

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images) or (
            self.config.steps_per_eval_all_images_quick
            and step == self.config.steps_per_eval_all_images_quick
        ):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(
                name="Eval Images Metrics Dict (all images)",
                scalar_dict=metrics_dict,
                step=step,
            )

            # Uncomment and adjust if you want to retain test overview logging with wandb
            # if self.config.is_wandb_enabled():
            #     test_table_columns = [
            #         "step",
            #     ] + list(metrics_dict.keys())
            #     if "Test Overview" not in self.wandb_tables:
            #         self.wandb_tables["Test Overview"] = wandb.Table(
            #             columns=test_table_columns
            #         )

            #     test_metrics = {"step": step}
            #     test_metrics.update(metrics_dict)

            #     self.wandb_tables["Test Overview"].add_data(**test_metrics)
            #     wandb.log(
            #         {"Test Overview": self.wandb_tables["Test Overview"]}, step=step
            #     )

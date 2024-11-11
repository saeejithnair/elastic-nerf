from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Type, Union, cast

import torch
import torch.distributed as dist
from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
    VanillaDataManager,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    Pipeline,
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass
class ElasticPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: ElasticPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = DataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""


class ElasticPipeline(VanillaPipeline):
    """The pipeline class for the vanilla nerf setup of multiple cameras for one or a few scenes.

    Args:
        config: configuration to instantiate pipeline
        device: location to place model and data
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        world_size: total number of machines available
        local_rank: rank of current machine
        grad_scaler: gradient scaler used in the trainer

    Attributes:
        datamanager: The data manager that will be used
        model: The model that will be used
    """

    model: Model

    def flatten_elastic_dict(self, elastic_metrics_dict):
        """Flatten the elastic metrics dict.

        Args:
            elastic_metrics_dict: elastic metrics dict

        Returns:
            flattened_metrics_dict: flattened metrics dict
        """
        flattened_metrics_dict = {}
        for key, val in elastic_metrics_dict.items():
            if isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    flattened_metrics_dict[f"{key}/{sub_key}"] = sub_val
            else:
                flattened_metrics_dict[key] = val

        return flattened_metrics_dict

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        model_outputs, loss_dict, elastic_metrics_dict = super().get_train_loss_dict(
            step
        )
        metrics_dict = self.flatten_elastic_dict(elastic_metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(
        self, step: int
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        model_outputs, loss_dict, elastic_metrics_dict = super().get_eval_loss_dict(
            step
        )
        metrics_dict = self.flatten_elastic_dict(elastic_metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        (
            elastic_metrics_dict,
            elastic_images_dict,
        ) = super().get_eval_image_metrics_and_images(step)
        metrics_dict = self.flatten_elastic_dict(elastic_metrics_dict)
        images_dict = self.flatten_elastic_dict(elastic_images_dict)
        return metrics_dict, images_dict

    # @profiler.time_function
    # def get_average_eval_image_metrics(
    #     self,
    #     step: Optional[int] = None,
    #     output_path: Optional[Path] = None,
    #     get_std: bool = False,
    # ):
    #     """Iterate over all the images in the eval dataset and get the average.

    #     Args:
    #         step: current training step
    #         output_path: optional path to save rendered images to
    #         get_std: Set True if you want to return std with the mean metric.

    #     Returns:
    #         metrics_dict: dictionary of metrics
    #     """
    #     elastic_metrics_dict = super().get_average_eval_image_metrics(
    #         step=step, output_path=output_path, get_std=get_std
    #     )
    #     metrics_dict = self.flatten_elastic_dict(elastic_metrics_dict)
    #     return metrics_dict

    @profiler.time_function
    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        assert isinstance(self.datamanager, VanillaDataManager)
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[green]Evaluating all eval images...", total=num_images
            )
            for (
                camera_ray_bundle,
                batch,
            ) in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(
                    camera_ray_bundle, evaluate_all_granularities=True
                )
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(
                    outputs, batch
                )

                if output_path is not None:
                    camera_indices = camera_ray_bundle.camera_indices
                    assert camera_indices is not None
                    for key, val in images_dict.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            output_path
                            / "{0:06d}-{1}.jpg".format(
                                int(camera_indices[0, 0, 0]), key
                            )
                        )
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (
                    height * width
                )
                metrics_dict = self.flatten_elastic_dict(metrics_dict)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor(
                        [metrics_dict[key] for metrics_dict in metrics_dict_list]
                    )
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor(
                            [metrics_dict[key] for metrics_dict in metrics_dict_list]
                        )
                    )
                )
        self.train()
        return metrics_dict

    def get_elastic_params_and_heatmaps(self):
        (
            elastic_params_dict,
            elastic_params_heatmap,
        ) = self.model.get_elastic_params_and_heatmaps()
        elastic_params_dict = self.flatten_elastic_dict(elastic_params_dict)

        return elastic_params_dict, elastic_params_heatmap

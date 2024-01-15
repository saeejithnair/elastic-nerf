import glob
import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from tabnanny import check
from typing import List, Optional, Tuple, Union

import numpy as np
import tyro
from nerfstudio import data
from PIL import Image

import elastic_nerf.utils.logging_utils as lu
import elastic_nerf.utils.wandb_utils as wu


def evaluate_model_image_metrics_and_images_batches(
    blender_scene,
    optimal_model,
    batch_indices_to_eval,
    checkpoint_dir,
    load_step,
    cache_dir,
    image_prefix,
    keys=["img"],
):
    batch_indices_filtered = []
    cache_files = {key: [] for key in keys}
    for i in batch_indices_to_eval:
        for key in keys:
            # Check if the cache directory already contains the images. If so, skip.
            cache_file = glob.glob(
                os.path.join(
                    cache_dir,
                    f"{key}_{image_prefix}_{i}.png",
                )
            )
            if len(cache_file) == 0:
                batch_indices_filtered.append(i)
                break

            cache_files[key] += cache_file

    batch_indices_filtered = sorted(list(set(batch_indices_filtered)))
    if len(batch_indices_filtered) == 0:
        lu.info("All images already cached. Skipping.")
        return cache_files
    else:
        lu.info(
            f"{len(batch_indices_filtered)}/{len(batch_indices_to_eval)} batch indices left to evaluate."
        )

    config = GenNerfBenchmarkConfig(
        blender_scene=blender_scene,
        experiment_name="blender_debug_eval",
        optimal_model=optimal_model,
    )
    config.batch_indices_to_eval = batch_indices_filtered
    config.vis = "tensorboard"

    trainer_cfg = config.make_trainer()
    trainer_cfg.load_dir = checkpoint_dir
    trainer_cfg.load_step = load_step
    trainer = trainer_cfg.setup(local_rank=0, world_size=1)
    trainer.setup()

    (
        batch_metrics_dict,
        batch_images_dict,
    ) = trainer.pipeline.get_eval_image_metrics_and_images_batches(
        config.batch_indices_to_eval
    )

    for key in keys:
        if key not in cache_files:
            cache_files[key] = []
        images_list = batch_images_dict[key]
        for i, image_tensor in enumerate(images_list):
            # Convert PyTorch tensor to NumPy array
            image_tensor = image_tensor
            image_array = image_tensor.cpu().numpy()
            image_array = (image_array * 255).astype(np.uint8)  # Convert to 0-255 scale

            # Convert the NumPy array to a PIL Image
            image_pil = Image.fromarray(image_array)

            # Save the image
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(
                cache_dir,
                f"{key}_{image_prefix}_{batch_indices_filtered[i]}.png",
            )
            image_pil.save(cache_file, format="png")
            lu.info(f"Saved image to {cache_file}")
            cache_files[key].append(cache_file)

    return cache_files


@dataclass
class ModelEvalImageBatchesConfig:
    arch_id: str
    """Model ID to evaluate (e.g. drums_SSIM_0.8673)."""
    run_id: str
    """WandB run ID."""
    arch_label: str
    """Human readable label for the architecture (e.g. 'NAS-NeRF-S')."""
    batch_indices_to_eval: List[int] = field(
        default_factory=lambda: list(range(0, 100))
    )
    """List of batch indices to evaluate."""
    cache_root: Path = Path(wu.CACHE_DIR)
    load_step: Optional[int] = None

    def main(self) -> None:
        run_result: wu.RunResult = wu.fetch_run_result(self.run_id)
        config = run_result.config

        if self.load_step is None and "max_num_iterations" not in config:
            raise ValueError(
                f"Need to specify --load_step or max_num_iterations must be available in the config."
            )
        elif self.load_step is None:
            self.load_step = int(config["max_num_iterations"]) - 1

        checkpoint_filename = f"step-{self.load_step:09d}.ckpt"
        checkpoint_path = wu.download_file(
            self.run_id, checkpoint_filename, results_subdir="results/runs"
        )
        checkpoint_dir = checkpoint_path.parent
        dataset_name = config["data"].split("/")[-1]
        image_prefix = f"{dataset_name}_{self.arch_label}"
        evaluate_model_image_metrics_and_images_batches(
            blender_scene=dataset_name,
            optimal_model=self.arch_id,
            batch_indices_to_eval=self.batch_indices_to_eval,
            checkpoint_dir=checkpoint_dir,
            load_step=self.load_step,
            cache_dir=checkpoint_dir,
            image_prefix=image_prefix,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ModelEvalImageBatchesConfig).main()


if __name__ == "__main__":
    entrypoint()

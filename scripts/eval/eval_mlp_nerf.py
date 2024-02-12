"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import functools
import math
import pathlib
import time
from collections import defaultdict
from tkinter import HIDDEN

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from nerfacc.estimators.occ_grid import OccGridEstimator

from elastic_nerf.nerfacc.datasets.nerf_synthetic import SubjectLoader
from elastic_nerf.nerfacc.radiance_fields.mlp import (
    VanillaNeRFRadianceField,
    get_field_granular_state_dict,
)
from elastic_nerf.nerfacc.utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)

device = "cuda:0"
set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="the path of the pretrained model",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES,
    help="which scene to use",
)

SAMPLING_STRATEGIES = [
    "uniform",
    "exp-optimal",
    "exp-optimal-reverse",
    "exp",
    "exp-reverse",
    "matroyshka",
    "matroyshka-reverse",
]
parser.add_argument(
    "--sampling_strategy",
    type=str,
    default="exp-reverse",
    choices=SAMPLING_STRATEGIES,
    help="sampling strategy for widths",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=4096,
)
parser.add_argument(
    "--num_train_widths",
    type=int,
    default=6,
)
parser.add_argument(
    "--num_widths_to_sample",
    type=int,
    default=1,
)
parser.add_argument(
    "--num_eval_granularities",
    type=int,
    default=6,
)
parser.add_argument(
    "--granular_norm", type=str, default="None", choices=["None", "var", "std"]
)
parser.add_argument(
    "--use_elastic",
    type=bool,
    default=False,
)

args = parser.parse_args()


# args.data_root = "/pub2/shared/nerf/nerfstudio/data/blender"
# args.scene = "drums"
# args.use_elastic = True
# args.num_widths_to_sample = 6
# args.granular_norm = "var"
# args.model_path = "/home/smnair/work/nerf/nerf-optimization/src/gen-nerf/scripts/kinder/mlp_nerf_drums_train_exp-reverse_6_6_6_var_True_50000.pt"

if args.granular_norm == "None":
    args.granular_norm = None

args_label = f"{args.scene}_{args.train_split}_{args.sampling_strategy}_{args.num_train_widths}_{args.num_widths_to_sample}_{args.num_eval_granularities}_{args.granular_norm}_{args.use_elastic}"
# training parameters
max_steps = 50000
init_batch_size = 1024
target_sample_batch_size = 1 << 16
# scene parameters
aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
near_plane = 0.0
far_plane = 1.0e10
# model parameters
grid_resolution = 128
grid_nlvl = 1
# render parameters
render_step_size = 5e-3
model_name = args.model_path.split(".pt")[0].split("/")[-1]

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
)

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)


# setup the radiance field we want to train.
radiance_field = VanillaNeRFRadianceField(
    use_elastic=args.use_elastic,
    num_granularities=args.num_train_widths,
    granular_norm=args.granular_norm,
).to(device)
optimizer = torch.optim.Adam(radiance_field.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[
        max_steps // 2,
        max_steps * 3 // 4,
        max_steps * 5 // 6,
        max_steps * 9 // 10,
    ],
    gamma=0.33,
)

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

if args.model_path is not None:
    checkpoint = torch.load(args.model_path)
    radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    estimator.load_state_dict(checkpoint["estimator_state_dict"])
    step = checkpoint["step"]
else:
    step = 0


def get_granularity_sampling_weights(num_granularities: int) -> torch.Tensor:
    """Generates normalized weights for sampling widths."""
    if args.sampling_strategy == "exp-optimal":
        weights = torch.tensor([math.exp(0.1 * i) for i in range(num_granularities)])
    elif args.sampling_strategy == "exp-optimal-reverse":
        weights = torch.tensor(
            [math.exp(0.1 * i) for i in range(num_granularities)]
        ).flip(
            0,
        )
    elif args.sampling_strategy == "exp":
        weights = torch.tensor([math.exp(i) for i in range(num_granularities)])
    elif args.sampling_strategy == "exp-reverse":
        weights = torch.tensor([math.exp(i) for i in range(num_granularities)]).flip(
            0,
        )
    elif args.sampling_strategy == "matroyshka":
        weights = torch.tensor([math.sqrt(2**i) for i in range(num_granularities)])
    elif args.sampling_strategy == "matroyshka-reverse":
        weights = torch.tensor(
            [math.sqrt(2**i) for i in range(num_granularities)]
        ).flip(
            0,
        )
    elif args.sampling_strategy == "uniform":
        weights = torch.ones(num_granularities)
    else:
        raise NotImplementedError

    return weights / weights.sum()


train_granularities = torch.tensor([2**i for i in range(args.num_train_widths)])
eval_granularities = torch.tensor([2**i for i in range(args.num_eval_granularities)])

# The sampling weights determine the probability of a granularity
# being selected for a forward pass.
granularity_sampling_weights: torch.Tensor = get_granularity_sampling_weights(
    len(train_granularities)
)

# To save time during the frequent eval metrics dict computations,
# we only use the widths associated with that forward pass.
# However, this field gets set to True when we evaluate all images.
evaluate_all_granularities = False

# Keep track of how many samples we've seen for each granularity.
# The keys for this should be the eval widths so that in our
# logs, we can see for certain that the non train widths have
# 0 samples.
granularity_sample_counts = {int(granularity): 0 for granularity in eval_granularities}

# training
tic = time.time()
elastic_outputs = {}
HIDDEN_DIM = 256

# evaluation
radiance_field.eval()
estimator.eval()

psnrs = defaultdict(list)
lpips = defaultdict(list)
with torch.no_grad():
    for i in tqdm.tqdm(range(len(test_dataset)), desc="Test Dataset", leave=True):
        data = test_dataset[i]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]
        pixels = data["pixels"]

        # rendering
        for granularity in tqdm.tqdm(
            eval_granularities, desc="Granularities", leave=False
        ):
            granularity = int(granularity)
            granularity_label = f"elastic_{granularity}"
            active_neurons = HIDDEN_DIM // granularity
            rgb, acc, depth, _ = render_image_with_occgrid(
                radiance_field,
                estimator,
                rays,
                # rendering options
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                # test options
                test_chunk_size=args.test_chunk_size,
                active_neurons=active_neurons,
            )
            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs[granularity_label].append(float(psnr.item()))
            lpips[granularity_label].append(float(lpips_fn(rgb, pixels).item()))
            tqdm.tqdm.write(
                f"evaluation {granularity_label} | psnr={psnrs[granularity_label][-1]} | psnr_avg={np.mean(psnrs[granularity_label])} | lpips={lpips[granularity_label][-1]} lpips_avg={np.mean(lpips[granularity_label])}"
            )
            if i == 0:
                imageio.imwrite(
                    f"rgb_test-{model_name}-{granularity_label}.png",
                    (rgb.cpu().numpy() * 255).astype(np.uint8),
                )
                imageio.imwrite(
                    f"rgb_error-{model_name}-{granularity_label}.png",
                    ((rgb - pixels).norm(dim=-1).cpu().numpy() * 255).astype(np.uint8),
                )
            # Explicitly delete the tensors to free up memory.
            del rgb, acc, depth

for granularity in eval_granularities:
    granularity = int(granularity)
    granularity_label = f"elastic_{granularity}"
    psnr_avg = sum(psnrs[granularity_label]) / len(psnrs[granularity_label])
    lpips_avg = sum(lpips[granularity_label]) / len(lpips[granularity_label])
    print(f"evaluation {granularity_label}: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
    active_neurons = HIDDEN_DIM // granularity

    model_save_path = f"{model_name}_{granularity_label}.pt"
    radiance_field_state_dict = get_field_granular_state_dict(
        radiance_field, active_neurons
    )
    torch.save(
        {
            "step": step,
            "radiance_field_state_dict": radiance_field_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "estimator_state_dict": estimator.state_dict(),
        },
        model_save_path,
    )
    print(f"saved {granularity_label} model to {model_save_path}")

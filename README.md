# Elastic-Nerf

Elastic Neural Radiance Fields

## Setup

### Clone Repo

```
git clone https://github.com/saeejithnair/elastic-nerf.git
```

### Update submodules

```
git submodule update --init --recursive
```

### Initialize Workspace

Run the initialization script at the root of the directory. This will generate an **.env** file in the repo root which will be sourced by docker compose and hold your user specific configurations and secret keys for logging and tracking experiments.

```
./initialize.sh
```

## Build Process

Elastic-NeRF can be run either in a docker container or in a conda environment.

### Building Docker Container

#### Setting CUDA Architectures

By default, the container is built for CUDA Architecture 86 which supports the Nvidia RTX A6000 GPU. The container can be easily customized for other architectures by prefixing any docker compose command with `CUDA_ARCHITECTURES=ARCH_ID`. For example, to build a container for a 75 series architecture, you can set the environment variable at build time as follows (various examples):

```
CUDA_ARCHITECTURES=75 docker compose build gen-nerf
CUDA_ARCHITECTURES=75 docker compose build gen-nerf-dev
CUDA_ARCHITECTURES=75 docker compose up --build gen-nerf-dev
```

#### Modes

The container can be built in one of two modes: _developer mode_ or _serving mode_. The core idea is that developer mode is intended for building on the codebase and automatically enables things like pre-commit hooks for linting, auto-formatting, type checking, etc but these bog down the container start up time if you only want to run a model. Hence for cases like running experiments (including training and inference after you're finished prototyping) or hyperparameter searches with WandB sweeps, try serving mode as it's slightly more lightweight.

##### Developer Mode

1. Spin up development container by running `docker compose up --build elastic-nerf-dev`
2. Once it finishes building (wait for an output message like _elastic-nerf-smnair-elastic-nerf-dev-1 | Starting container and waiting forever.._), attach to the container using the VS-Code docker extension
3. Start coding!

##### Serving Mode

To just build the non-dev service, run

```
docker compose build elastic-nerf
```

To build and start the service as a detached instance, and then run a specific command

```
docker compose run -d --build elastic-nerf {your command here}
```

A common use case is to start a pre-configured WandB sweep on a specific GPU. This can be done by running:

```
GPU_IDS=0 CUDA_ARCHITECTURES=75 docker compose run -d --build elastic-nerf wandb agent wandb-entity-name/elastic-nerf/abc0defg
```

### Building Conda Environment

TODO

## Datasets

Nerfstudio hosts a number of NeRF related datasets which can be easily downloaded by running Nerfstudio's `ns-download` command from within a container or environment. E.g. the following command downloads the Blender Synthetic dataset.

```
ns-download-data blender --save-dir /home/user/shared/nerfstudio/data/
```

You can also download the dataset to a filesystem drive that has been mounted to a container running in serving mode.

```
docker compose run -d --build elastic-nerf ns-download-data blender --save-dir /home/user/shared/nerfstudio/data/
```

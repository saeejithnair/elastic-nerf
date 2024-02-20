## -----------------------------------------
## Stage 1: Base dependencies installation.
## -----------------------------------------
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as base

ENV DEBIAN_FRONTEND noninteractive
# Set timezone as it is required by some packages.
ENV TZ=America/Toronto
# CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"
# Use new faster buildkit.
ENV DOCKER_BUILDKIT=1
# Project name.
ENV PROJECT="elastic-nerf"
# Path to host dependencies location.
ENV HOST_DEPS="./docker/deps"
# Path to docker dependencies location.
ENV DOCKER_DEPS="/docker/deps"

# Update system packages and install the basic ones
RUN apt-get -y update && \
    apt-get -y install --no-install-recommends \
    apt-utils \
    software-properties-common

# Add deadsnakes PPA
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get -y update

# Install Python and related packages
RUN apt-get -y install --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv \
    python3-pip \
    python-is-python3

# Copy apt dependencies file
COPY $HOST_DEPS/apt_requirements.txt $DOCKER_DEPS/apt_requirements.txt
# Install additional dependencies from requirements file
RUN xargs -d '\n' -a $DOCKER_DEPS/apt_requirements.txt \
    apt-get -y install \
    --no-install-recommends

# Clean up to reduce image size
RUN rm -rf /var/lib/apt/lists/*

## -----------------------------------------
## Stage 2: User setup.
## -----------------------------------------
FROM base AS setup

# User and group ids to map the docker user to the host user.
ARG USER_ID=1000
ARG GROUP_ID=1000
# Group ID to add the docker user to a shared group the host user is part of.
ARG SHARED_GROUP_ID

ARG DOCKER_USERNAME="user"
ARG HOST_USERNAME="null"
ARG HOSTNAME
ENV USER_NAME $DOCKER_USERNAME
ENV HOST_USERNAME $HOST_USERNAME
ENV HOSTNAME $HOSTNAME
ENV USER_HOME="/home/${USER_NAME}"

# Add a non-root docker user with passwordless sudo permissions and map this
# docker user to the host user so that created files in the docker container
# are owned by the host user.
RUN addgroup --gid $GROUP_ID $USER_NAME && \
    adduser \
    --uid $USER_ID \
    --ingroup $USER_NAME \
    --home $USER_HOME \
    --shell /bin/bash \
    --disabled-password \
    --gecos "" \
    $USER_NAME && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Create shared group with same GID as a shared host group (e.g. vip_user) and
# add docker user to this secondary group so that they can access any shared
# dirs that might need to be mounted.
ENV SHARED_GROUP_NAME="host_shared"
RUN if [ ! -z $SHARED_GROUP_ID ]; then \
    groupadd -g $SHARED_GROUP_ID $SHARED_GROUP_NAME && \
    usermod -a -G $SHARED_GROUP_NAME $USER_NAME; \
    fi

USER $USER_NAME
WORKDIR $USER_HOME

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:${USER_HOME}/.local/bin"

# Set default shell to bash.
SHELL ["/bin/bash", "-c"]

## -----------------------------------------
## Stage 3: Application building.
## -----------------------------------------
FROM setup AS app

# CUDA architecture, required by tiny-cuda-nn.
ARG CUDA_ARCHITECTURES=86
# Virtual environment path.
ENV VIRTUAL_ENV="$USER_HOME/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Set the target architectures as env var for tiny-cuda-nn.
ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}

# Install python dependencies.
RUN python3.9 -m venv $VIRTUAL_ENV

# RUN python -m pip install \
#     torch==1.13.1+cu117 \
#     torchvision==0.14.1+cu117 \
#     torchaudio==0.13.1 \
#     --extra-index-url https://download.pytorch.org/whl/cu117
RUN python -m pip install \
    torch==2.0.1+cu117 \
    torchvision==0.15.2+cu117 \
    torchaudio==2.0.2 \
    --extra-index-url https://download.pytorch.org/whl/cu117

COPY $HOST_DEPS/pip_requirements.txt $DOCKER_DEPS/pip_requirements.txt
RUN python -m pip install -r $DOCKER_DEPS/pip_requirements.txt

RUN TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} python -m pip install \
    git+https://github.com/saeejithnair/tiny-cuda-nn/#subdirectory=bindings/torch

# Install precompiled nerfacc from wheel and nerfstudio from pip.
RUN python -m pip install nerfacc==0.5.2 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu117/nerfacc-0.5.2%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
RUN python -m pip install nerfstudio==0.3.4

## -----------------------------------------
## Stage 4: Runtime setup.
## -----------------------------------------
FROM app AS runtime
WORKDIR $USER_HOME

# Weights and Biases login info for MLOps.
ARG USER_WANDB_MODE
ARG USER_WANDB_KEY
ARG WANDB_ENTITY
ENV WANDB_MODE $USER_WANDB_MODE
ENV WANDB_ENTITY $WANDB_ENTITY

# Log into Weights and Biases if a key is provided.
RUN if [ ! -z $USER_WANDB_KEY ]; then wandb login $USER_WANDB_KEY; fi

# HuggingFace login info for dataset management.
ARG USER_HF_MODE
ARG USER_HF_TOKEN
ENV HF_MODE $USER_HF_MODE
# Log into HuggingFace CLI if a token is provided.
RUN if [ ! -z $USER_HF_TOKEN ]; then huggingface-cli login --token $USER_HF_TOKEN; fi

# Workspace path will have been set if user wants to enable developer mode.
ARG DEVELOPER_MODE
ENV DEVELOPER_MODE $DEVELOPER_MODE
ARG TARGET_PROJECT_DIR
ENV TARGET_PROJECT_DIR $TARGET_PROJECT_DIR
ARG NERFSTUDIO_CACHE_DIR
ENV NERFSTUDIO_CACHE_DIR $NERFSTUDIO_CACHE_DIR

# Initialize pre-commit hooks during build so we don't have to run this every
# single time on container startup.
ARG DUMMY_GIT_REPO_PATH="$USER_HOME/dummy_repo"
ENV DUMMY_GIT_REPO_PATH $DUMMY_GIT_REPO_PATH

ARG PRECOMMIT_CONFIG_FILE="$DUMMY_GIT_REPO_PATH/.pre-commit-config.yaml"
ENV PRECOMMIT_CONFIG_FILE $PRECOMMIT_CONFIG_FILE

COPY .pre-commit-config.yaml $PRECOMMIT_CONFIG_FILE

WORKDIR $USER_HOME
ENV ENTRYPOINT_SCRIPT="$USER_HOME/entrypoint.sh"
COPY $HOST_DEPS/entrypoint.sh $ENTRYPOINT_SCRIPT

USER root
RUN chmod +x $ENTRYPOINT_SCRIPT && \
    chown $USER_NAME:$USER_NAME $ENTRYPOINT_SCRIPT

USER $USER_NAME
ENTRYPOINT ["/home/user/entrypoint.sh"]

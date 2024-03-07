## -----------------------------------------
## Stage 1: Base dependencies installation.
## -----------------------------------------
ARG OS_VERSION=22.04
ARG CUDA_VERSION=11.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${OS_VERSION} as base

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

# Virtual environment path.
ENV VIRTUAL_ENV="$USER_HOME/venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install packages.
RUN python3.10 -m venv $VIRTUAL_ENV
RUN python3.10 -m pip install --no-cache-dir --upgrade \
    pip setuptools pathtools promise pybind11

# Install pytorch and submodules
RUN export CUDA_VER=$(echo ${CUDA_VERSION%.*} | tr -d '.') && \
    python3.10 -m pip install --no-cache-dir \
    torch==2.0.1+cu${CUDA_VER} \
    torchvision==0.15.2+cu${CUDA_VER} \
        --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VER}

COPY $HOST_DEPS/pip_requirements.txt $DOCKER_DEPS/pip_requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r $DOCKER_DEPS/pip_requirements.txt

# Install tynyCUDNN (we need to set the target architectures as environment variable first).
# CUDA architecture, required by tiny-cuda-nn.
ARG CUDA_ARCHITECTURES
# COPY $HOST_DEPS/detect_cuda_arch.sh $DOCKER_DEPS/detect_cuda_arch.sh
# RUN chmod +x $DOCKER_DEPS/detect_cuda_arch.sh
# RUN export $($DOCKER_DEPS/detect_cuda_arch.sh) && \
#     echo "Using CUDA Architecture: $TCNN_CUDA_ARCHITECTURES"

ENV TCNN_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}
RUN python3.10 -m pip install --no-cache-dir git+https://github.com/NVlabs/tiny-cuda-nn.git@v1.6#subdirectory=bindings/torch

# Install precompiled nerfacc from wheel and nerfstudio from pip.
RUN python3.10 -m pip install --no-cache-dir nerfacc==0.5.2 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118/nerfacc-0.5.2%2Bpt20cu118-cp310-cp310-linux_x86_64.whl
RUN python3.10 -m pip install --no-cache-dir nerfstudio==0.3.4

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
ARG RESULTS_CACHE_DIR
ENV RESULTS_CACHE_DIR $RESULTS_CACHE_DIR
ARG WANDB_CACHE_DIR
ENV WANDB_CACHE_DIR $WANDB_CACHE_DIR

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

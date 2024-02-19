#!/bin/bash

print_error() {
    echo -e "\033[31m[ERROR]: $1\033[0m"
}

print_warning() {
    echo -e "\033[33m[WARNING]: $1\033[0m"
}

print_info() {
    echo -e "\033[32m[INFO]: $1\033[0m"
}

SHARED_GROUP_NAME="vip_user"
SHARED_GROUP_ID=$(getent group $SHARED_GROUP_NAME | cut -d: -f3)

# Get UID and GID.
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# TODO(snair): Deprecate this.
FIXUID=$(id -u)
FIXGID=$(id -g)

# Validate that the user is a member of the group.
if ! groups $USER | grep &>/dev/null "\b$SHARED_GROUP_NAME\b"; then
    echo "User ${USER} with UID ${USER_ID} is not a member of group ${SHARED_GROUP_NAME} with GID ${SHARED_GROUP_ID}"
    exit 1
fi

BASE_PORT=${BASE_PORT:-$(($(id -u)*20))}
GUI_TOOLS_VNC_PORT=${GUI_TOOLS_VNC_PORT:-$((BASE_PORT++))}
WANDB_MODE="online"
HF_MODE="online"
PROJECT_NAME="elastic-nerf"
host_workspace_path=""
wandb_key=""
hf_token=""
WANDB_CACHE_DIR="/nfs0/shared/nerf/wandb"

# Host machine setup.
HOSTNAME=$(hostname)

HUGGINGFACE_CACHE_DIR="/nfs0/shared/hf_cache"
NERFSTUDIO_CACHE_DIR="/nfs0/shared/nerf/nerfstudio"

# Check if cache directories exist, and if not, prompt the user.
if [ ! -d "$HUGGINGFACE_CACHE_DIR" ]; then
    print_warning "Warning: HuggingFace cache directory does not exist: $HUGGINGFACE_CACHE_DIR"
    read -p "Do you want to create this directory? [Y/n]: " yn_hf
    if [[ $yn_hf =~ [Yy] ]]; then
        mkdir -p "$HUGGINGFACE_CACHE_DIR" && echo "Directory $HUGGINGFACE_CACHE_DIR created."
    else
        print_warning "Huggingface cache directory is required, please make sure to create it later."
    fi
fi

if [ ! -d "$NERFSTUDIO_CACHE_DIR" ]; then
    print_warning "Warning: Nerfstudio cache directory does not exist: $NERFSTUDIO_CACHE_DIR"
    read -p "Do you want to create this directory? [Y/n]: " yn_nerf
    if [[ $yn_nerf =~ [Yy] ]]; then
        mkdir -p "$NERFSTUDIO_CACHE_DIR" && echo "Directory $NERFSTUDIO_CACHE_DIR created."
    else
        print_warning "Nerfstudio cache directory is required, please make sure to create it later."
    fi
fi

# WandB setup.
while true; do
    read -p "Do you wish to use Weights and Biases for tracking experiments? [Y/n]: " yn
    case $yn in
        [Yy]* ) WANDB_MODE="online"; read -p "Enter your WandB API key: " wandb_key ; break;;
        [Nn]* ) WANDB_MODE="offline"; print_warning "WandB disabled"; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# HuggingFace setup.
while true; do
    read -p "Do you wish to use HuggingFace for downloading datasets? [Y/n]: " yn
    case $yn in
        [Yy]* ) HF_MODE="online"; read -p "Enter your HuggingFace user access token: " hf_token ; break;;
        [Nn]* ) HF_MODE="offline"; print_warning "HuggingFace disabled"; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# Set username for non-root docker user.
DOCKER_USERNAME="user"

# Workspace setup for development.
workspace_prompt="Do you wish to enable developer mode by configuring your workspace path? \
This is the directory containing the repo and will be mounted automatically by the ${PROJECT_NAME}-dev service \
to enable easier development within a container. [Y/n]: "
# Replace newlines and multiple spaces with a single space
workspace_prompt=$(echo $workspace_prompt | tr -s ' ')
while true; do
    read -p "$workspace_prompt " yn
    case $yn in
        [Yy]* ) read -p "Enter the path to your development workspace: " host_workspace_path ; break;;
        [Nn]* ) echo "Docker compose context will be used as default workspace"; break;;
        * ) echo "Please answer yes or no.";;
    esac
done

# If the host workspace path was set, validate that it exists, and that it
# contains the project repo.
TARGET_REPO_MOUNTDIR="/home/$DOCKER_USERNAME/workspace"
TARGET_PROJECT_DIR="$TARGET_REPO_MOUNTDIR/$PROJECT_NAME"
if [[ -n $host_workspace_path ]]; then
    if [[ ! -d $host_workspace_path ]]; then
        print_error "Host workspace path does not exist: $host_workspace_path"
        exit 1
    fi
    if [[ ! -d $host_workspace_path/$PROJECT_NAME ]]; then
        print_error "Host workspace path does not contain $PROJECT_NAME repo: $host_workspace_path/$PROJECT_NAME"
        print_error "Unsupported workspace path: $host_workspace_path"
        exit 1
    fi
fi


> ".env"
echo "COMPOSE_PROJECT_NAME=$PROJECT_NAME-${USER}" >> ".env"
echo "USER_ID=$USER_ID" >> ".env"
echo "GROUP_ID=$GROUP_ID" >> ".env"
echo "SHARED_GROUP_NAME=$SHARED_GROUP_NAME" >> ".env"
echo "SHARED_GROUP_ID=$SHARED_GROUP_ID" >> ".env"
echo "GUI_TOOLS_VNC_PORT=$GUI_TOOLS_VNC_PORT" >> ".env"
echo "USERNAME=${USER}" >> ".env"
echo "DOCKER_USERNAME=${DOCKER_USERNAME}" >> ".env"
echo "WANDB_MODE=${WANDB_MODE}" >> ".env"
echo "HF_MODE=${HF_MODE}" >> ".env"
echo "HOST_WORKSPACE_PATH=${host_workspace_path}" >> ".env"
echo "TARGET_REPO_MOUNTDIR=${TARGET_REPO_MOUNTDIR}" >> ".env"
echo "TARGET_PROJECT_DIR=${TARGET_PROJECT_DIR}" >> ".env"
echo "HUGGINGFACE_CACHE_DIR=${HUGGINGFACE_CACHE_DIR}" >> ".env"
echo "NERFSTUDIO_CACHE_DIR=${NERFSTUDIO_CACHE_DIR}" >> ".env"
echo "WANDB_CACHE_DIR=${WANDB_CACHE_DIR}" >> ".env"
echo "HOSTNAME=${HOSTNAME}" >> ".env"

# TODO(snair): Deprecate this.
echo "FIXUID=$FIXUID" >> ".env"
echo "FIXGID=$FIXGID" >> ".env"

if [ $WANDB_MODE == "online" ]; then
    echo "WANDB_KEY=${wandb_key}" >> ".env"
    echo "WANDB_ENTITY=saeejithn" >> ".env"
fi

if [ $HF_MODE == "online" ]; then
    echo "HF_TOKEN=${hf_token}" >> ".env"
fi

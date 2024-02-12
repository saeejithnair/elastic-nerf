#!/bin/bash

# Pre-Installation Steps:
# 1. Install CUDA 11.X (11.8 if Ada, 11.7 otherwise) to /usr/local/cuda-11.X
# 2. Install gcc-10 and g++-10 (`sudo apt install gcc-10 g++-10`).
# 3. If you have other gcc/g++ versions installed, use update-alternatives to mange versions.
# 3a. `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10`
# 3b. `sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10`
# 3c. `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 20`
# 3d. `sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 20`
# 3e. `sudo update-alternatives --config gcc`
# 3f. `sudo update-alternatives --config g++`

# Use the first argument as the CUDA architecture version, default to 86.
CUDA_ARCHITECTURES=${1:-86}
PACKAGE_NAME="elastic_nerf"

# Use the second argument as the Conda environment name, default to package name.
CONDA_ENV_NAME=${2:-"$PACKAGE_NAME"}

PYTORCH_VERSION=${3:-"1.13.1"}

# Select CUDA version based on architecture.
if [ "$CUDA_ARCHITECTURES" == "89" ]; then
  # Ada architecture only supports CUDA 11.8 and above.
  CUDA_VERSION="11.8"
else
  CUDA_VERSION="11.7"
fi

# Check for selected CUDA installation.
if [ ! -d "/usr/local/cuda-$CUDA_VERSION" ]; then
  echo "Error: CUDA $CUDA_VERSION is not installed in /usr/local/cuda-$CUDA_VERSION. Please follow the pre-installation steps."
  exit 1
fi
# Check for gcc-10 and g++-10
GCC_VERSION=$(gcc -dumpversion | cut -d. -f1-2)
GXX_VERSION=$(g++ -dumpversion | cut -d. -f1-2)

if [ "$GCC_VERSION" != "10" ] || [ "$GXX_VERSION" != "10" ]; then
  echo "Error: gcc-10 and g++-10 must be installed and selected as the default versions. Please follow the pre-installation steps."
  exit 1
fi

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Initialize Conda in bash script.
source /home/$USER/miniconda3/etc/profile.d/conda.sh

# Print the CUDA architecture version.
echo "Building for CUDA architecture version: $CUDA_ARCHITECTURES"

# Create the conda environment with the given name.
conda create --name $CONDA_ENV_NAME -y python=3.9
conda activate $CONDA_ENV_NAME

# Create directories for the conda activate and deactivate scripts.
mkdir -p ${CONDA_PREFIX}/etc/conda/activate.d
mkdir -p ${CONDA_PREFIX}/etc/conda/deactivate.d

# Create and modify activate script
echo '#!/bin/sh' > ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
echo 'set -o allexport' >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
echo "source $SCRIPT_DIR/../../.env" >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
echo 'set +o allexport' >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
echo "export OLD_PATH=$PATH" >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
echo "export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
echo "export PATH=$PATH:/usr/local/cuda-$CUDA_VERSION/bin" >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-$CUDA_VERSION/lib64" >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
echo "export TCNN_CUDA_ARCHITECTURE=$CUDA_ARCHITECTURES" >> ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh
# Source the activate script immediately.
source ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh

# Create and modify deactivate script.
echo '#!/bin/sh' > ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh
echo "export PATH=$OLD_PATH" >> ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH" >> ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh
echo "unset OLD_PATH" >> ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh
echo "unset OLD_LD_LIBRARY_PATH" >> ${CONDA_PREFIX}/etc/conda/deactivate.d/env_vars.sh

# Install Pytorch based on specified argument
echo "Installing Pytorch $PYTORCH_VERSION"
if [ "$PYTORCH_VERSION" == "1.13.1" ]; then
  conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
      pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
else
  conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
      pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia
fi

python -m pip install --upgrade pip

pip install cmake lit

pip install -r docker/deps/pip_requirements.txt

# SHA corresponds to release tag v1.6 for tiny-cuda-nn.
pip install ninja \
    git+https://github.com/NVlabs/tiny-cuda-nn/@8e6e242f36dd197134c9b9275a8e5108a8e3af78#subdirectory=bindings/torch

if [ "$PYTORCH_VERSION" != "1.13.1" ]; then
  if [ "$CUDA_VERSION" == "11.8" ]; then
    echo "Installing nerfacc==0.5.2+pt20cu118"
    pip install nerfacc==0.5.2 \
      -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu118/nerfacc-0.5.2%2Bpt20cu118-cp39-cp39-linux_x86_64.whl
  else
    echo "Installing nerfacc==0.5.2+pt20cu117"
    pip install nerfacc==0.5.2 \
      -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-2.0.0_cu117/nerfacc-0.5.2%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
  fi
fi
# Install nerfstudio, pin to v0.3.4
pip install nerfstudio==0.3.4

if [ ! -z $WANDB_KEY ]; then wandb login $WANDB_KEY; fi

if [ ! -z $HF_TOKEN ]; then huggingface-cli login --token $HF_TOKEN; fi

if pip show $PACKAGE_NAME &> /dev/null; then
  # Check if the package is already installed.
  echo "$PACKAGE_NAME is already installed."
else
  # Install package if not already installed.
  echo "$PACKAGE_NAME is not installed. Installing ${PACKAGE_NAME}..."
  git submodule update --init --recursive
  HOST_PROJECT_DIR="${HOST_WORKSPACE_PATH}/elastic-nerf/elastic_nerf"
  cd ${HOST_PROJECT_DIR} && \
    pip install -e .

  # Install gonas.
  cd ${HOST_PROJECT_DIR}/third-party/gonas && \
    pip install -e .
fi

#!/bin/bash

echo "Running entrypoint.sh"
if [[ $DEVELOPER_MODE == 1 ]]; then
    echo "Setting up developer mode..."
    cd ${USER_HOME}/workspace/$PROJECT
    nbdev_install_hooks
    nbdev_install_quarto
    quarto install tinytex
    quarto tools install chromium
    pre-commit install

    # Added Dockerfile commands.
    echo "Updating file permissions and ownership..."
    sudo chmod +x $PRECOMMIT_CONFIG_FILE
    sudo chown -R $USER_NAME:$USER_NAME $DUMMY_GIT_REPO_PATH

    echo "Initializing Git repository and installing hooks..."
    cd $DUMMY_GIT_REPO_PATH
    git init .
    pre-commit install --hook-type pre-push \
                       --hook-type post-checkout \
                       --hook-type pre-commit

    sudo rm -rf $DUMMY_GIT_REPO_PATH
fi

PACKAGE_NAME="elastic_nerf"
if pip show $PACKAGE_NAME &> /dev/null; then
  # Check if the package is already installed
  echo "$PACKAGE_NAME is already installed."
else
  # Install package if not already installed.
  echo "$PACKAGE_NAME is not installed. Installing ${PROJECT}..."
  cd ${TARGET_PROJECT_DIR} && \
    pip install -e . && \
    dvc pull && dvc checkout

  # Install gonas.
  cd ${TARGET_PROJECT_DIR}/third-party/nas-sandbox && \
    pip install -e .
fi

cd ${USER_HOME}/workspace/$PROJECT



# Run the CMD passed as command-line arguments or keep container open.
if [ $# -eq 0 ]; then
  # Install Nerfstudio CLI.
  echo "Installing Nerfstudio CLI."
  ns-install-cli --mode install

  echo "Starting container and waiting forever..."
  exec sleep inf
else
  echo "Running command: $@"
  exec "$@"
fi

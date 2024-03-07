#!/bin/bash

# Use nvidia-smi to get the Compute Capability and format it by removing the decimal point
COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//')

# Echo the formatted Compute Capability for Docker to use
echo "$COMPUTE_CAPABILITY"

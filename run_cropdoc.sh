#!/bin/bash

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Running with CUDA support."
    NVIDIA_FLAGS="--nv"
    
    # Check if nvidia-container-cli is available
    if [ -x "$(command -v nvidia-container-cli)" ]; then
        echo "nvidia-container-cli detected. Adding --nvccli flag."
        NVIDIA_FLAGS="$NVIDIA_FLAGS --nvccli"
    else
        echo "nvidia-container-cli not found. Proceeding without --nvccli flag."
    fi
else
    echo "No NVIDIA GPU detected. Running without CUDA support."
    NVIDIA_FLAGS=""
fi

# Run Apptainer shell command
apptainer shell \
    $NVIDIA_FLAGS \
    --writable-tmpfs \
    --bind ./CropDoc:/CropDoc \
    --pwd /CropDoc \
    cropdoc-backend.sif
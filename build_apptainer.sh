#!/bin/bash

# Ensure we're in the project root directory
cd "$(dirname "$0")"

# Function to print usage information
print_usage() {
    echo "Usage: $0 [--frontend] [--backend]"
    echo "  --frontend  Build only the frontend container"
    echo "  --backend   Build only the backend container"
    echo "  If no flag is provided, both containers will be built."
}

# Initialize flags
build_frontend=false
build_backend=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --frontend) build_frontend=true ;;
        --backend) build_backend=true ;;
        --help) print_usage; exit 0 ;;
        *) echo "Unknown parameter: $1"; print_usage; exit 1 ;;
    esac
    shift
done

# If no flags are provided, build both
if ! $build_frontend && ! $build_backend; then
    build_frontend=true
    build_backend=true
fi

# Build frontend if flag is set
if $build_frontend; then
    echo "Building frontend container..."
    apptainer build --fakeroot cropdoc-frontend.sif Apptainer.frontend
fi

# Build backend if flag is set
if $build_backend; then
    echo "Building backend container..."
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Building with CUDA support..."
        apptainer build --fakeroot --nv cropdoc-backend.sif Apptainer.backend
    else
        echo "No NVIDIA GPU detected. Building without CUDA support..."
        apptainer build --fakeroot cropdoc-backend.sif Apptainer.backend
    fi
fi

echo "Build complete."
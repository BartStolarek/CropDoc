#!/bin/bash

# Check if the .sif files exist
if [ ! -f cropdoc-backend.sif ] || [ ! -f cropdoc-frontend.sif ]; then
    echo "Apptainer image files not found. Please run build_apptainer.sh first."
    exit 1
fi

# Run the backend
echo "Starting backend..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Running backend with CUDA support"
    NVIDIA_FLAGS="--nv"
    if [ -x "$(command -v nvidia-container-cli)" ]; then
        NVIDIA_FLAGS="$NVIDIA_FLAGS --nvccli"
    fi
    apptainer run --fakeroot $NVIDIA_FLAGS --writable-tmpfs --bind ./CropDoc:/CropDoc:rw cropdoc-backend.sif &
else
    echo "No NVIDIA GPU detected. Running backend without CUDA support"
    apptainer run --fakeroot --writable-tmpfs --bind ./CropDoc:/CropDoc:rw cropdoc-backend.sif &
fi

# Run the frontend
echo "Starting frontend..."
apptainer run --bind ./frontend:/frontend cropdoc-frontend.sif &

echo "Both services started. Use Ctrl+C to stop."

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
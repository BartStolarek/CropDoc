#!/bin/bash

# Check if the .sif files exist
if [ ! -f cropdoc-backend.sif ] || [ ! -f cropdoc-frontend.sif ]; then
    echo "Apptainer image files not found. Please run build_apptainer.sh first."
    exit 1
fi

# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    CUDA_FLAG="--nv"
    echo "CUDA is available. Adding flag to backend run to run with GPU support."
    nvidia-smi
else
    CUDA_FLAG=""
    echo "CUDA is not available. Running without GPU support."
fi

# Run the backend
echo "Starting backend..."
apptainer run $CUDA_FLAG --writable-tmpfs \
    --bind ./CropDoc:/CropDoc \
    --bind /usr/local/cuda-12.2:/usr/local/cuda-12.2 \
    cropdoc-backend.sif &

# Run the frontend
echo "Starting frontend..."
apptainer run --bind ./frontend:/frontend cropdoc-frontend.sif &

echo "Both services started. Use Ctrl+C to stop."

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
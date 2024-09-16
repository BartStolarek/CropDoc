#!/bin/bash

# Check if the .sif files exist
if [ ! -f cropdoc-backend.sif ] || [ ! -f cropdoc-frontend.sif ]; then
    echo "Apptainer image files not found. Please run build_apptainer.sh first."
    exit 1
fi

# Run the backend
echo "Starting backend..."
apptainer run --nv --bind ./CropDoc:/CropDoc cropdoc-backend.sif &

# Run the frontend
echo "Starting frontend..."
apptainer run --bind ./frontend:/frontend cropdoc-frontend.sif &

echo "Both services started. Use Ctrl+C to stop."

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
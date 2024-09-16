#!/bin/bash

# Build the Apptainer containers
apptainer build cropdoc-backend.sif Apptainer.backend
apptainer build cropdoc-frontend.sif Apptainer.frontend

# Run the backend
apptainer run --nv --bind ./CropDoc:/CropDoc cropdoc-backend.sif &

# Run the frontend
apptainer run --bind ./frontend:/frontend cropdoc-frontend.sif &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
#!/bin/bash

# Build the Apptainer containers
echo "Building backend container..."
apptainer build cropdoc-backend.sif Apptainer.backend

echo "Building frontend container..."
apptainer build cropdoc-frontend.sif Apptainer.frontend

echo "Build complete."
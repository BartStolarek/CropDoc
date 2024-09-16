#!/bin/bash

# Ensure we're in the project root directory
cd "$(dirname "$0")"

# Build the Apptainer containers
echo "Building frontend container..."
apptainer build --fakeroot cropdoc-frontend.sif Apptainer.frontend


echo "Building backend container..."
apptainer build --fakeroot cropdoc-backend.sif Apptainer.backend



echo "Build complete."
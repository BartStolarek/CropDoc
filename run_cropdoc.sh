#!/bin/bash

apptainer shell \
    --nv \
    --nvccli \
    --writable-tmpfs \
    --bind ./CropDoc:/CropDoc \
    --pwd /CropDoc \
    cropdoc-backend.sif
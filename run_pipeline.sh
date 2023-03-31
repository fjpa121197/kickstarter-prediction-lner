#!/bin/bash
# Navigate to pipeline directory
cd ml_pipeline
# Start docker steps, starting with building image and then running container
docker build --tag lner-ml-pipeline .
docker run lner-ml-pipeline
# Access docker container files and copy artifacts (in this case, the trained model)
CONTAINER_ID=$(docker ps -alq)
mkdir -p artifacts
docker cp $CONTAINER_ID:ml-pipeline/model_1.h5 artifacts/
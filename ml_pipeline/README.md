# ML Pipeline

This directory serves to run an ML pipeline, focused on the following steps: simple data preparation, training (neural network), evaluation and storing of trained model.

## Content

- `run_pipeline.py`: main entry point file, which executes and orchestrates steps of ML pipeline, and calls other modules in pipeline directory.

- `pipeline/`: pipeline folder containing preprocessing, training and evaluation modules.

- `utils.py`: python script containing helper functions that are used inside main pipeline script.

- `Dockerfile`: Dockerfile image used to create container to execute pipeline.

- `requirements.txt`: requirements file containing modules used to execute pipeline. Used in Dockerfile to install correct libraries, and can be used to run locally (to create a virtual environment).

- `run_pipeline.sh`: bash script to be used for building docker image and running container.

## Running locally

In order to run this pipeline locally, proceed with cloning this repository to your local machine.

### Using Docker

Given that Docker is properly installed in your machine and running, follow these steps:

1. Change directory to `ml_pipeline`.
2. Open a terminal (you can use a WSL if using windows) and run the following command: `bash run_pipeline.sh`

>**Note:**
>
>If using Windows, it is recommended to use WSL and you will need to change some setting in Docker Desktop, please follow this [guide](https://docs.docker.com/desktop/windows/wsl/).

## Without Docker

It is possible to run the ML pipeline locally, without the need of Docker. These are the steps to do so:

1. Create a virtual environment (using pipenv or conda) and install libraries in `requirements.txt`.

2. Open terminal/command prompt and activate virtual environment previously created.

3. Change directory to `ml_pipeline`

3. Run the following command `python run_pipeline.py -d REPLACE_WITH_DATA_PATH_TO_USE`
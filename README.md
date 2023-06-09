# kickstarter-prediction-lner

Code repo holding code of a training production pipeline, as part of the hiring process for the MLOps position at LNER. The objective is to create a production pipeline using the code provided by the data scientist in a jupyter notebook.

## Content

- `ml_pipeline/`: folder containing the code for the pipeline.

- `notebooks_ds/`: notebooks folder, inside you can find the notebook provided by the data scientist which pipeline is based on.

- `run_pipeline.sh`: bash script used to trigger the whole ml pipeline and run it using a docker container.

- `.github/workflows/`: folder containing github workflow files that automatically test and run pipeline for new pushes to repo.

## Running locally

In order to run this pipeline locally, proceed with cloning this repository to your local machine.

Do these steps:

1. After cloning repository, proceed to go to the `ml_pipeline` folder, and unzip the `kickstarter_train_test_onehot.zip` into the same folder (inside `ml_pipeline/data`).

2. Proceed to change to the root directory.

### Using Docker

Given that Docker is properly installed in your machine and running, follow these steps:

1. Open a terminal (you can use a WSL if using windows) and run the following command: `bash run_pipeline.sh`

>**Note:**
>
>If using Windows, it is recommended to use WSL and you will need to change some setting in Docker Desktop, please follow this [guide](https://docs.docker.com/desktop/windows/wsl/). Additionally, it editing is done to this .sh file, windows might add a different end of line.

This script should make a Docker image, installing the correct dependencies to run the ML pipeline, then run a container that executes it, and save artifacts back to the artifacts directory.

### Without Docker

It is possible to run the ML pipeline locally, without the need of Docker. These are the steps to do so:

1. Create a virtual environment (using pipenv or conda) and install libraries in `ml_pipeline/requirements.txt`.

2. Open terminal/command prompt and activate virtual environment previously created.

3. Change directory to `ml_pipeline`

4. Run the following command `python run_pipeline.py -d REPLACE_WITH_DATA_PATH_TO_USE`.

>**Note:**
>
>Make sure the data zip files have been unzipped.

Dont feel like running it locally? See section below.

## CI/CD - Using Github actions

The files inside `.github/workflows/` folder, have a series of workflows/set of instructions to run, this is all run and hosted using remote resources. This automates the testing and deployment of code, so when new changes are made to different components of the ML pipeline, this can be tested automatically, if successfull, deploy (run) the pipeline from end-to-end.

Follow these steps to trigger the workflows:

1. Clone repo to your local computer.
2. Push repositories to Github.
3. If you push to main branch, it will trigger a CI&CD workflow, where it will test the pipeline and then run it using Docker and locally.
4. If you push to another branch, it will trigger a CI workflow only, where it will only test the pipeline.

If you head to the github repositories, and go to the Actions tab, you should see that the workflow(s) are running. For the workflow that does CI&CD, after finishing, it will create artifacts associated to it, and there, you can find my-model-artifacts, which contains the trained model.

# ML Pipeline

This directory serves to run an ML pipeline, focused on the following steps: simple data preparation, training (neural network), evaluation and storing of trained model.

## Content

- `run_pipeline.py`: main entry point file, which executes and orchestrates steps of ML pipeline, and calls other modules in pipeline directory.

- `pipeline/`: pipeline folder containing preprocessing, training and evaluation modules.

- `Dockerfile`: Dockerfile image used to create container to execute pipeline.

- `requirements.txt`: requirements file containing modules used to execute pipeline. Used in Dockerfile to install correct libraries, and can be used to run locally (to create a virtual environment).

- `artifacts/`: folder containing training artifacts (as for now, only stores trained model).

- `test/`: folder containing test cases for testing single components of pipeline (preprocess and train modules) and their operations. See below for more reference.

## Testing locally

Currently, `pytest` is used to test the pipeline. The test cases are inside `tests/`, and there are 2 files for testing the preprocessing and training module (`pipeline/train.py` and `pipeline/preprocess.py`).

In order to run the tests, these are the steps:

1. Create a local virtual environment with the dependencies in `requirements.txt`, and activate it.

2. Make sure you are in the correct directory (inside `ml_pipeline`).

3. Run the following command: `pytest -vv`.

>**Note**
>
>Right now, it only consists of unit tests, focused in testing the methods of each class object in train.py and preprocess.py.
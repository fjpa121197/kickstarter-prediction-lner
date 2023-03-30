# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /ml-pipeline

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . ./ml-pipeline

CMD [ "python", "ml-pipeline/run_pipeline.py", "-d", "ml-pipeline/data/kickstarter_train_test_onehot.csv"]

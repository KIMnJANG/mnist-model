FROM tensorflow/tensorflow

ARG GITHUB_OWNER_ARG
ENV GITHUB_OWNER=${GITHUB_OWNER_ARG}

ARG GITHUB_REPO_ARG
ENV GITHUB_REPO=${GITHUB_REPO_ARG}

ARG GITHUB_WORKFLOW_ARG
ENV GITHUB_WORKFLOW=${GITHUB_WORKFLOW_ARG}

ARG GITHUB_TOKEN_ARG
ENV GITHUB_TOKEN=${GITHUB_TOKEN_ARG}

ARG GCS_BUCKET_ARG
ENV GCS_BUCKET=${GCS_BUCKET_ARG}

ARG MODEL_TAG_ARG
ENV MODEL_TAG=${MODEL_TAG_ARG}

RUN apt-get -y update
RUN apt-get -y install git
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install dvc
RUN pip install 'dvc[gs]'
RUN pip install typing

RUN mkdir -p /app
ADD train.py /app/
ADD utils.py /app/

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "app/train.py"]
# FROM tensorflow/tensorflow

# ADD train.py /
FROM tensorflow/tensorflow

ARG GCS_BUCKET_ARG
ENV GCS_BUCKET=${GCS_BUCKET_ARG}

RUN apt-get -y update
RUN apt-get -y install git
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install dvc
RUN pip install 'dvc[gs]'
RUN pip install typing

RUN mkdir -p /app
ADD train.py /app/

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "app/train.py"]
# FROM tensorflow/tensorflow

# ADD train.py /
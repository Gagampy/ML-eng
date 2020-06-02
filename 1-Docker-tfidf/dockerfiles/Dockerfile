FROM python:3.7-slim-buster

USER 1000

FROM continuumio/miniconda3

RUN mkdir /usr/src/tf_idf/

RUN conda update -n base -c defaults conda

WORKDIR /usr/src/tf_idf/

COPY ./1-Docker-tfidf/ /usr/src/tf_idf/

RUN conda env create -f /usr/src/tf_idf/environment.yml

CMD ["conda", "run", "-n", "tfidf-server", "python", "tfidf_server.py"]

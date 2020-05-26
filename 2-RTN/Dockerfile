FROM python:3.7-slim-buster

RUN mkdir -p /usr/src/rtn/

WORKDIR /usr/src/rtn/

COPY ./ /usr/src/rtn/

RUN conda env create -f enviroment.yml && conda activate rtn-title-prediction && pip3 install .


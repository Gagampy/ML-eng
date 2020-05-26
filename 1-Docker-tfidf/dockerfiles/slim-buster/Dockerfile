FROM tfidf-server-baseimage

FROM continuumio/miniconda3

CMD ["/bin/bash"]

RUN mkdir /usr/src/tf_idf/

RUN conda update -n base -c defaults conda

WORKDIR /usr/src/tf_idf/

COPY ./1-Docker-tfidf/ /usr/src/tf_idf/

RUN conda env create -f /usr/src/tf_idf/environment.yml

SHELL ["conda", "run", "-n", "tfidf-server", "/bin/bash", "-c"]

RUN chown 1000 /usr/src/tf_idf/model

CMD ["conda", "run", "-n", "tfidf-server", "python", "tfidf_server.py"]
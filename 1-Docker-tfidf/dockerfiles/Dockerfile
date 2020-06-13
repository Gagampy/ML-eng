FROM continuumio/miniconda3

ARG NB_USER="user1000"
ARG NB_UID="1000"
ARG NB_GID="100"

ENV CONDA_DIR=/opt/conda \
    HOME=/home/$NB_USER

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER &&\
    mkdir -p $CONDA_DIR && \
    chown $NB_USER:$NB_GID $CONDA_DIR

WORKDIR $HOME

RUN mkdir /home/$NB_USER/tf_idf

RUN chown -R $NB_USER:$NB_GID /home/$NB_USER

WORKDIR /home/$NB_USER/tf_idf

USER $NB_UID

COPY ./1-Docker-tfidf/ /home/$NB_USER/tf_idf

RUN conda env create -f /home/$NB_USER/tf_idf/environment.yml

CMD ["conda", "run", "-n", "tfidf-server", "python", "tfidf_server.py"]


FROM tfidf-server-baseimage

CMD ["/bin/bash"]

RUN mkdir -p /usr/src/tf_idf/

WORKDIR /usr/src/tf_idf/

COPY ./1-Docker-tfidf/ /usr/src/tf_idf/

RUN pip3 install -r reqs.txt

CMD ["python", "tfidf_server.py"]
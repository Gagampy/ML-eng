FROM ubuntu:latest

RUN apt-get update

RUN apt-get install -y python3-pip

RUN mkdir -p /usr/src/tf_idf/

WORKDIR /usr/src/tf_idf/

COPY ./ /usr/src/tf_idf/

RUN pip3 install -r reqs.txt

CMD ["python3", "tfidf_server.py"]

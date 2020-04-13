FROM python:3.7-slim-buster

RUN mkdir -p /usr/src/tf_idf/

WORKDIR /usr/src/tf_idf/

COPY ./. /usr/src/tf_idf/

RUN pip3 install -r reqs.txt

CMD ["python", "tfidf_server.py"]
## 1st task:

Build Docker python-slim-buster image (Debian+python) and install dependencies:
* `docker build -t tfidf-server-image -f 1-Docker-tfidf/dockerfiles/Dockerfile .`

Run a container with running flask server with Tf-Idf transformer in quiet mode:
* `docker run --mount type=volume,source=tf-idf-model,target=/usr/src/tf_idf/model/ --name tfidf-server-cont -p 5000:5000 -d tfidf-server-image`

Send data there and get Tf-Idf of it:
* `python 1-Docker-tfidf/get_tfidf.py --file 1-Docker-tfidf/data/[1-2-3].txt`

And observe the result in `1-Docker-tfidf/tf_idf folder`. 

To inspect volume files during container running:
* `sudo ls -l /var/lib/docker/volumes/tf-idf-model/_data`

Environment for the dataset chosen is in `2-RTN`.

## 2st task:
Inspect 2-RTN for environment and DVC stages. 


## 3rd task:

Build Docker image with conda for client:
* `docker build -t rtn-mlflow-client-img -f 2-RTN/dockerfiles/Dockerfile_client .`

Build Docker image with mlflow for server:
* `docker build -t rtn-mlflow-serv-img -f 2-RTN/dockerfiles/Dockerfile_mlflow_serv .`

Create `rtn-net` network if it's absent in your network list:
* `docker network create rtn-net`

Run server container:
* `docker run --rm --name rtn-mlflow-serv -p 5000:5000 --network rtn-net -it rtn-mlflow-serv-img mlflow server --host 0.0.0.0`

Run client container:
* `docker run --rm  --name rtn-mlflow-client --network rtn-net -it rtn-mlflow-client-img`

To train models run:
* `conda run -n client_env python src/train_models.py`

To get metric on test set run with a model chosen by a git branch:
* `conda run -n client_env python src/predict_by_model.py`

(Optional) To install package as `rtn`:
* `pip install .`
## 1st task:

Build Docker ubuntu image and install python3.7 with pip on top of it:
* `docker build -t tf-idf-server-ubuntu -f dockerfiles/ubuntu/Dockerfile .`

Build Docker python-slim-buster image (Debian+python):
* `docker build -t tf-idf-server-buster -f dockerfiles/slim-buster/Dockerfile .`

Run a container with flask server with Tf-Idf transformer:
* `docker run --name tf-idf-server-[ubuntu/buster]-cont -p 5000:5000 tf-idf-server-[ubuntu/buster]`

Send data there and get Tf-Idf of it:
* `python get_tfidf.py --file ./data/[1-2-3].txt`

And observe the result in `./tf_idf folder`. 

To build an image for the dataset presented in ./TSE/ run:
* `docker build -t twitter-sent-extr .`


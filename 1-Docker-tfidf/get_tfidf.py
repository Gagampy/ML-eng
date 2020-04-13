import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--url', default='http://127.0.0.1:5000/get_tf-idf', type=str)

parser = parser.parse_args()

with open(parser.file, 'r') as f:
    doc = {'sents': [''.join(f.readlines())]}

result = requests.get(parser.url, json=doc)

with open(f'./tf_idf/{parser.file.split("/")[2].split(".")[0]}_tfidf.txt', 'w') as f:
    f.write(' '.join([str(el) for el in result.json()['output'][0]]))
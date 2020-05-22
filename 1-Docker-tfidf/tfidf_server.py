import pickle
from flask import Flask, request
import os

app = Flask(__name__)


@app.route('/get_tf-idf', methods=['GET'])
def getter():
    recieved_docs = request.get_json(force=True)['sents']
    tf_idf_matrix = tf_idf.transform(recieved_docs).toarray().tolist()
    return {'output': tf_idf_matrix}


if __name__ == "__main__":

    with open('model/tfidf_model.pkl', 'rb') as f:
        tf_idf = pickle.load(f)

    print(os.getcwd())

    with open('model/logfile.log', 'w') as f:
        f.writelines('Server started.')

    app.run(host='0.0.0.0')

    with open('model/logfile.log', 'w') as f:
        f.writelines('Server closed.')

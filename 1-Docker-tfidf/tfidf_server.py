import pickle
from flask import Flask, request

app = Flask(__name__)


@app.route('/get_tf-idf', methods=['GET'])
def print_square():
    recieved_docs = request.get_json(force=True)['sents']
    tf_idf_matrix = tf_idf.transform(recieved_docs).toarray().tolist()
    return {'output': tf_idf_matrix}


if __name__ == "__main__":

    with open('./tfidf_model.pkl', 'rb') as f:
        tf_idf = pickle.load(f)

    app.run(host='0.0.0.0')

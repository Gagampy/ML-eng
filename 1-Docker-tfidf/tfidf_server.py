import pickle
print(help("modules"))
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

    # stat_info = os.stat('model')
    # uid = stat_info.st_uid
    # gid = stat_info.st_gid
    # print(uid, gid)
    # print(os.getuid())
    # print(os.getcwd())

    try:
        with open('model/logfile.log', 'a+') as f:
            f.writelines('Server started.')
    except Exception as e:
        print(e)

    try:
        with open('model/logfile.log', 'r') as f:
            print(f.readlines())
    except Exception as e:
        print(e)

    app.run(host='0.0.0.0')

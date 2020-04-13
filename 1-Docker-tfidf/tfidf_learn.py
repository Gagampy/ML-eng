from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def main():

    tfidf = TfidfVectorizer(max_df=0.8)
    with open('data/shakespeare_input.txt', 'r') as f:
        docs = ''.join(f.readlines()).split('\n')

    tfidf.fit(docs)
    with open('./tfidf_model.pkl', 'wb') as file:
        pickle.dump(tfidf, file=file)


if __name__ == '__main__':
    main()

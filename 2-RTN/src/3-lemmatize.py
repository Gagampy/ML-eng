import argparse
from nltk.stem import WordNetLemmatizer
import pickle

DATAPATH = '2-RTN/data/tokenized/ria-tokenized.pkl'
DATASAVE = '2-RTN/data/lemmatized/ria-lemmatized.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', default=DATAPATH)
    parser.add_argument('-sp', default=DATASAVE)
    parser = parser.parse_args()

    with open(parser.fp, 'rb') as f:
        dataset = pickle.load(f)

    lemm = WordNetLemmatizer()
    dataset = {title: [lemm.lemmatize(token) for token in tokens] for title, tokens in zip(dataset.keys(),
                                                                                           dataset.values()
                                                                                           )
               }

    with open(parser.sp, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    main()

import pandas as pd
import argparse
from nltk.tokenize import word_tokenize
import pickle
import nltk
nltk.download('punkt')

DATAPATH = '2-RTN/data/cleaned/ria-1k-clean.csv'
DATASAVE = '2-RTN/data/tokenized/ria-tokenized.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', default=DATAPATH)
    parser.add_argument('-sp', default=DATASAVE)
    parser = parser.parse_args()

    dataset = pd.read_csv(parser.fp, encoding='utf-8')
    result = {title: tokens for title, tokens in zip(dataset['title'],
                                                     dataset['text'].apply(lambda x: word_tokenize(x)).values.tolist())}
    with open(parser.sp, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()

import pandas as pd
import argparse
import rutokenizer
import rupostagger
import rulemma
import json
from tqdm import tqdm
from pathlib import Path

DATAPATH = '2-RTN/data/cleaned/ria-1k-clean.csv'
DATASAVE = '2-RTN/data/lemmatized/ria-lemmatized.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', default=DATAPATH)
    parser.add_argument('-sp', default=DATASAVE)
    parser = parser.parse_args()

    Path('2-RTN/data/lemmatized/').mkdir(parents=True, exist_ok=True)

    lemmatizer = rulemma.Lemmatizer()
    lemmatizer.load()

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    tagger = rupostagger.RuPosTagger()
    tagger.load()

    dataset = pd.read_csv(parser.fp, encoding='utf-8')

    result = {'text': [], 'title': []}
    for title, text in tqdm(zip(dataset['title'], dataset['text'])):
        result['title'].append([lemma for _, _, lemma, *_ in lemmatizer.lemmatize(tagger.tag(tokenizer.tokenize(title)))])
        result['text'].append([lemma for _, _, lemma, *_ in lemmatizer.lemmatize(tagger.tag(tokenizer.tokenize(text)))])

    with open(parser.sp, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()

import pandas as pd
import argparse
from pymystem3 import Mystem
from tqdm import tqdm
from pathlib import Path

DATAPATH = '2-RTN/data/cleaned/ria-cleaned.csv'
DATASAVE = '2-RTN/data/lemmatized/ria-lemmatized.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', default=DATAPATH)
    parser.add_argument('-sp', default=DATASAVE)
    parser = parser.parse_args()

    Path('2-RTN/data/lemmatized/').mkdir(parents=True, exist_ok=True)

    lemmatizer = Mystem()
    dataset = pd.read_csv(parser.fp, encoding='utf-8')

    result = {'text': [], 'title': []}
    for title, text in tqdm(zip(dataset['title'], dataset['text'])):
        result['title'].append([lemma for lemma in lemmatizer.lemmatize(title)])
        result['text'].append([lemma for lemma in lemmatizer.lemmatize(text)])

    pd.DataFrame.from_dict(result).to_json(parser.sp, orient='records', lines=True)


if __name__ == '__main__':
    main()

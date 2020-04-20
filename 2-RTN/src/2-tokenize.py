import pandas as pd
import argparse
import rutokenizer
# import json
from tqdm import tqdm
from pathlib import Path

DATAPATH = '2-RTN/data/cleaned/ria-1k-clean.csv'
DATASAVE = '2-RTN/data/tokenized/ria-tokenized.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', default=DATAPATH)
    parser.add_argument('-sp', default=DATASAVE)
    parser = parser.parse_args()

    Path('2-RTN/data/tokenized/').mkdir(parents=True, exist_ok=True)

    tokenizer = rutokenizer.Tokenizer()
    tokenizer.load()

    dataset = pd.read_csv(parser.fp, encoding='utf-8')

    result = {'text': [], 'title': []}
    for title, text in tqdm(zip(dataset['title'], dataset['text'])):
        result['title'].append([word.lower() for word in tokenizer.tokenize(title)])
        result['text'].append([word.lower() for word in tokenizer.tokenize(text)])

    pd.DataFrame.from_dict(result).to_json(parser.sp, orient='records', lines=True)
    # with open(parser.sp, 'w') as f:
    #     json.dump(result, f)


if __name__ == '__main__':
    main()
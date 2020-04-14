import re
import argparse
import pandas as pd


DATAPATH = './2-RTN/data/ria_1k.json'
DATASAVE = './2-RTN/data/cleaned/ria-1k-clean.csv'


def clean_text(text: str) -> str:
    """
    Processes text removing all raw html headers data and replacing non-encoded symbols.

    :param text: Text to process.
    :return: str -- Processed text.

    """

    text = re.sub('</*p.*?>|\\n|<strong.*strong>|<img.*?}-]\" />', '', text).strip()
    text = re.sub('\&nbsp;', ' ', text)
    text = re.sub('\&ndash;', '-', text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', default=DATAPATH, type='str')
    parser.add_argument('-sp', default=DATASAVE, type='str')
    parser = parser.parse_args()

    dataset = pd.read_json(parser.fp, lines=True, encoding='utf-8')
    dataset['text'] = dataset['text'].apply(clean_text)
    dataset.to_csv(parser.sp, index=False)


if __name__ == '__main__':
    main()

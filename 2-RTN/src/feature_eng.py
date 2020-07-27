import pandas as pd
from collections import defaultdict
from pathlib import Path

import pymorphy2
from string import punctuation

import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords


DATAPATH = '2-RTN/data/lemmatized/lemmatized.json'
SAVEPATH = '2-RTN/data/feature_eng/dataset_fe.csv'
morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")


def get_text_length(x):
    return len([token for token in x if token not in punctuation.split() + [' '] + russian_stopwords])


def count_pos_tags(x: [str]):
    pos_tags = defaultdict(int)

    for token in x:
        p = morph.parse(token)[0]
        pos_tag = p.tag.POS
        pos_tags[pos_tag] += 1

    return pos_tags


def main():
    Path('2-RTN/data/feature_eng/').mkdir(parents=True, exist_ok=True)

    dataset = pd.read_json(DATAPATH, lines=True)

    dataset['text_len'] = dataset['text'].apply(lambda x: get_text_length(x))
    dataset['title_len'] = dataset['title'].apply(lambda x: get_text_length(x))

    dataset['n_punct'] = dataset['text'].apply(lambda x: len([t for t in x if t in punctuation]))
    dataset['pos_tags'] = dataset['text'].apply(count_pos_tags)

    pos_df = pd.DataFrame()
    for v in dataset.pos_tags.values:
        pos_df = pos_df.append(pd.DataFrame.from_dict(v, orient='index').T)

    pos_df.index = dataset.index
    dataset = dataset.join(pos_df)

    to_drop = ['PRTS', 'INTJ', 'GRND', 'COMP', 'PRED', 'PRTF', 'ADJS', 'NUMR', 'VERB', None]
    dataset = dataset.drop(to_drop, axis=1)

    new_columns = ['PRCL', 'NPRO', 'n_punct', 'title_len']
    dataset = dataset[new_columns].fillna(dataset[new_columns].median())
    dataset.to_csv(SAVEPATH, index=False)


if __name__ == '__main__':
    main()

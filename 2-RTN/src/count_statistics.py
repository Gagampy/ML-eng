import pandas as pd
import re
import argparse
from pathlib import Path
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords

RUSSIAN_SW = set(stopwords.words("russian"))
DATAPATH = "2-RTN/data/lemmatized/lemmatized.json"
DATASAVE = "2-RTN/metrics/token-statistics.metric"


def count_stopwords_in_sentence(
    tokenized_sentence: [str], stopwords_: set = RUSSIAN_SW
):
    return len(stopwords_.intersection(set(tokenized_sentence)))


def count_words_in_sentence(tokenized_sentence: [str]):
    return len([word for word in tokenized_sentence if word.isalpha()])


def count_symbols_in_sentence(tokenized_sentence: [str]):
    return len(
        [word for word in tokenized_sentence if len(re.findall("[\w\d]", word)) == 0]
    )


def get_stopwords_ratio_in_document(
    document: [[str]], stopwords_: set = RUSSIAN_SW, round_to=3
):
    n_stopwords = 0
    n_words = 0
    for sentence in document:
        n_words += count_words_in_sentence(sentence)
        n_stopwords += count_stopwords_in_sentence(sentence, stopwords_)

    return round(n_stopwords / n_words, round_to)


def get_symbols_ratio_in_document(document: [[str]], round_to=3):
    n_words = 0
    n_symbols = 0
    for sentence in document:
        n_words += count_words_in_sentence(sentence)
        n_symbols += count_symbols_in_sentence(sentence)

    return round(n_symbols / n_words, round_to)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fp", default=DATAPATH)
    parser.add_argument("-sp", default=DATASAVE)
    parser = parser.parse_args()

    Path("2-RTN/metrics/").mkdir(parents=True, exist_ok=True)

    dataset = pd.read_json(parser.fp, lines=True)

    sw_ratio_texts = get_stopwords_ratio_in_document(dataset.values[:, 0])
    sw_ratio_titles = get_stopwords_ratio_in_document(dataset.values[:, 1])
    symb_ratio_texts = get_symbols_ratio_in_document(dataset.values[:, 0])
    symb_ratio_titles = get_symbols_ratio_in_document(dataset.values[:, 1])

    statistics_message = """
    Stopwords ratios:
    
    -- Texts: {:<10} Titles: {}
    =====================================
    
    Symbols (non word and digit) ratios:
    
    -- Texts: {:<10} Titles: {}
    =====================================
    """.format(
        sw_ratio_texts, sw_ratio_titles, symb_ratio_texts, symb_ratio_titles
    ).strip()

    with open(parser.sp, "w") as f:
        f.writelines(statistics_message)


if __name__ == "__main__":
    main()

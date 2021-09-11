import os
import nltk
import pymorphy2
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

np.seterr(all="ignore")
morph = pymorphy2.MorphAnalyzer()
nltk.download("stopwords")
stopwords = stopwords.words("russian")

names = {'Моника': ['моника', 'мон'],
         'Рэйчел': ['рейчел', 'рейч'],
         'Чендлер': ['чендлер', 'чэндлер', 'чен'],
         'Фиби': ['фиби', 'фибс'],
         'Росс': ['росс'],
         'Джоуи': ['джоуи', 'джои', 'джо']}


def preprocess(filename):
    with open(filename, encoding='utf-8-sig') as f:
        text = f.read()
    tokens = [token for token in nltk.word_tokenize(text.lower())
              if token.isalpha()
              and token not in stopwords]
    lemmed_text = [morph.parse(word)[0].normal_form for word in tokens]
    return ' '.join(lemmed_text)


def indexing(script_folder):
    vectorizer = CountVectorizer(analyzer='word')
    corpus = []
    curr_dir = os.getcwd()
    for root, dirs, files in os.walk(os.path.join(curr_dir, script_folder)):
        for file in files:
            filename = os.path.join(root, file)
            corpus.append(preprocess(filename))
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def explore(script_folder):
    X, vectorizer = indexing(script_folder)

    # most frequent word
    matrix_freq = np.asarray(X.sum(axis=0)).ravel()
    final_df = pd.DataFrame([matrix_freq], columns=np.array(vectorizer.get_feature_names()))
    print(f'самое частотное слово: {final_df.idxmax(axis=1)[0]}, '
          f'{final_df[final_df.idxmax(axis=1)].values[0][0]} вхождений\n')

    # least frequent word
    print(f'самое редкое слово: {final_df.idxmin(axis=1)[0]}, '
          f'{final_df[final_df.idxmin(axis=1)].values[0][0]} вхождений\n')

    # words in all documents
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())
    all_doc_words = [col for col in df if 0 not in df[col].tolist()]
    print(f'набор слов во всех документах коллекции: {", ".join(all_doc_words)}.'
          f'\nвсего {len(all_doc_words)} слов\n')

    # popular name
    name_freqs = {}
    for name in names.keys():
        total = 0
        for nickname in names[name]:
            num = vectorizer.vocabulary_.get(morph.parse(nickname)[0].normal_form)
            if num:
                total += num
        name_freqs[name] = total
    print(f'самое популярное имя: {max(name_freqs, key=name_freqs.get)} (включая варианты), '
          f'{name_freqs[max(name_freqs, key=name_freqs.get)]} вхождений')


explore('friends-data')

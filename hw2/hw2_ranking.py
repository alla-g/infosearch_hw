import os
import warnings
import re

import nltk
import pymorphy2
import numpy as np

from pprint import pprint
from string import digits

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
morph = pymorphy2.MorphAnalyzer()


def preprocess(text):
    tokens = [token for token in nltk.word_tokenize(text.lower())
              if token.isalpha() or token in digits]
    lemmed_text = [morph.parse(word)[0].normal_form for word in tokens]
    return ' '.join(lemmed_text)


def index_corpus(script_folder):
    vectorizer = TfidfVectorizer()
    corpus = []
    filenames = []
    curr_dir = os.getcwd()
    for root, dirs, files in os.walk(os.path.join(curr_dir, script_folder)):
        for file in files:
            filename = os.path.join(root, file)
            filenames.append(re.sub(r'.+?- (.+?)\..+', r'\g<1>', file))
            with open(filename, encoding='utf-8-sig') as f:
                text = f.read()
            corpus.append(preprocess(text))
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer, filenames


def index_query(query_text, vectorizer):
    if not isinstance(query_text, str):
        raise ValueError('Неверный формат запроса')
    query_vector = vectorizer.transform([preprocess(query_text)]).toarray()
    return query_vector


def find_sim(X, query_vector):
    dists = cosine_similarity(X, query_vector)
    return dists


def search(query_text, X, vectorizer, filenames):
    query_vector = index_query(query_text, vectorizer)
    dists = find_sim(X, query_vector)
    if np.all(dists == 0):  # all zeros
        return 'Ничего не нашлось:с'
    else:
        order = (-dists).argsort(axis=0)
        sorted_docs = [filenames[idx[0]] for idx in order]
        return sorted_docs


X, vectorizer, filenames = index_corpus('friends-data')
while True:
    query = input('Введите поисковый запрос: ')
    pprint(search(query, X, vectorizer, filenames))
    answer = input('Хотите продолжить поиск? да/нет ')
    if answer.lower() == 'нет':
        break

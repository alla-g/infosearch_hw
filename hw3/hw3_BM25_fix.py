import json
import warnings

import numpy as np
import nltk
import pymorphy2

from string import digits
from pprint import pprint
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")
morph = pymorphy2.MorphAnalyzer()


def preprocess(text):
    tokens = [token for token in nltk.word_tokenize(text.lower())
              if token.isalpha() or token in digits]
    lemmed_text = [morph.parse(word)[0].normal_form for word in tokens]
    return ' '.join(lemmed_text)


def collect_answers(data):
    corpus = []
    raw_answers = []
    for question in data:
        item = json.loads(question)
        if item['answers']:
            ids = [int(answer['author_rating']['value']) for answer in item['answers']
                   if answer['author_rating']['value']]
            id_max = max(zip(ids, range(len(ids))))[1]
            answer = item['answers'][id_max]['text']
            corpus.append(preprocess(answer))
            raw_answers.append(answer)
    raw_answers = np.array(raw_answers)

    return corpus, raw_answers


def index_corpus(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        data = list(f)[:50000]
    corpus, raw_answers = collect_answers(data)

    count_vectorizer = CountVectorizer()
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

    x_count_vec = count_vectorizer.fit_transform(corpus)
    x_tf_vec = tf_vectorizer.fit_transform(corpus)
    tfidf_vectorizer.fit_transform(corpus)

    idf = tfidf_vectorizer.idf_
    tf = x_tf_vec

    k = 2
    b = 0.75
    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()
    B_1 = (k * (1 - b + b * len_d / avdl)).A1

    vals = []
    rows = []
    cols = []
    for i, j in zip(*tf.nonzero()):
        A = idf[j] * tf[i, j] * (k + 1)
        B = tf[i, j] + B_1[i]
        vals.append(A / B)
        rows.append(i)
        cols.append(j)
    matrix = sparse.csr_matrix((vals, (rows, cols)))

    return matrix, count_vectorizer, raw_answers


def index_query(query_text, vectorizer):
    if not isinstance(query_text, str):
        raise ValueError('Неверный формат запроса')
    query_vector = vectorizer.transform([preprocess(query_text)])
    return query_vector


def count_sim(matrix, query_vector):
    scores = matrix @ query_vector.T
    return scores


def search(query_text, matrix, vectorizer, answers):
    query_vector = index_query(query_text, vectorizer)
    scores = count_sim(matrix, query_vector)
    if scores.nnz == 0:
        return 'Ничего не нашлось:с'
    else:
        order = np.argsort(scores.toarray(), axis=0)[::-1]
        # на этом моменте нехватка памяти, поэтому
        # в выдачу сохраняются только первые 5к текстов:
        sorted_answers = answers[order[:5000].ravel()]
    # list() для аккуратного pprint
    return list(sorted_answers)


def start(filename):
    matrix, vectorizer, answers = index_corpus(filename)
    while True:
        query = input('Введите поисковый запрос: ')
        output = search(query, matrix, vectorizer, answers)
        pprint(output)
        answer = input('Хотите продолжить поиск? да/нет ')
        if answer.lower() == 'нет':
            break


if __name__ == '__main__':
    start('questions_about_love.jsonl')

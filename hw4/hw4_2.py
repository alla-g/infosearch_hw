import json
import warnings

import torch
import numpy as np
import nltk
import pymorphy2

from string import digits
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors

warnings.filterwarnings("ignore")
morph = pymorphy2.MorphAnalyzer()


def norm(vec):
    return vec / np.linalg.norm(vec)


def cls_pooling(model_output):
    return model_output[0][:, 0]


def preprocess(text):
    tokens = [token for token in nltk.word_tokenize(text.lower())
              if token.isalpha() or token in digits]
    lemmed_text = [morph.parse(word)[0].normal_form for word in tokens]
    return ' '.join(lemmed_text)


def collect_dataset(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        data = list(f)[:50000]  # 50000
    proc_answers, proc_questions = [], []
    raw_answers, raw_questions = [], []

    for question in data:
        item = json.loads(question)
        if item['answers']:
            ids = [int(answer['author_rating']['value']) for answer in item['answers']
                   if answer['author_rating']['value']]
            id_max = max(zip(ids, range(len(ids))))[1]
            answer = item['answers'][id_max]['text']
            proc_answer = preprocess(answer)
            proc_question = preprocess(item['question'])
            # проверка, что и вопрос, и ответ не пустые после лемматизации
            if (proc_answer != '') & (proc_question != ''):
                proc_answers.append(proc_answer)
                proc_questions.append(proc_question)
                raw_answers.append(answer)
                raw_questions.append(item['question'])

    return proc_answers, proc_questions, raw_answers, raw_questions


def index_corpus_cv(proc_corpus):
    c_vectorizer = CountVectorizer(analyzer='word')
    matrix = c_vectorizer.fit_transform(proc_corpus)
    matrix = normalize(matrix)

    return matrix, c_vectorizer


def index_query_cv(proc_query, c_vectorizer):
    query_matrix = c_vectorizer.transform(proc_query)
    query_matrix = normalize(query_matrix)

    return query_matrix


def index_corpus_tfidf(proc_corpus):
    tfidf_vectorizer = TfidfVectorizer()
    matrix = tfidf_vectorizer.fit_transform(proc_corpus)

    return matrix, tfidf_vectorizer


def index_query_tfidf(proc_query, tfidf_vectorizer):
    query_matrix = tfidf_vectorizer.transform(proc_query)

    return query_matrix


def index_corpus_bm25(proc_corpus):
    count_vectorizer = CountVectorizer()
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

    x_count_vec = count_vectorizer.fit_transform(proc_corpus)
    x_tf_vec = tf_vectorizer.fit_transform(proc_corpus)
    tfidf_vectorizer.fit_transform(proc_corpus)

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

    return matrix, count_vectorizer


def index_query_bm25(proc_query, vectorizer):
    query_matrix = vectorizer.transform(proc_query)

    return query_matrix


def index_corpus_fasttext(proc_corpus):
    model_name = 'araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
    fasttext_model = KeyedVectors.load(model_name)

    matrix = np.zeros((len(proc_corpus), fasttext_model.vector_size))
    for doc_id, document in enumerate(proc_corpus):
        lemmas = document.split()
        lem_vectors = np.zeros((len(lemmas), fasttext_model.vector_size))

        for lem_id, lemma in enumerate(lemmas):
            if lemma in fasttext_model:
                lem_vectors[lem_id] = fasttext_model[lemma]

        doc_vector = np.mean(lem_vectors, axis=0)
        matrix[doc_id] = norm(doc_vector)

    return matrix, fasttext_model


def index_query_fasttext(proc_query, fasttext_model):
    query_matrix = np.zeros((len(proc_query), fasttext_model.vector_size))
    for doc_id, document in enumerate(proc_query):
        lemmas = document.split()
        lem_vectors = np.zeros((len(lemmas), fasttext_model.vector_size))

        for lem_id, lemma in enumerate(lemmas):
            if lemma in fasttext_model:
                lem_vectors[lem_id] = fasttext_model[lemma]
        doc_vector = np.mean(lem_vectors, axis=0)
        query_matrix[doc_id] = norm(doc_vector)

    return query_matrix


def index_corpus_bert(raw_corpus):
    bert_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    bert_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

    encoded_input = bert_tokenizer(raw_corpus, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        bert_model_output = bert_model(**encoded_input)
    matrix = cls_pooling(bert_model_output)
    matrix = normalize(matrix)
    model = {'bert_tokenizer': bert_tokenizer,
             'bert_model': bert_model}

    return matrix, model


def index_query_bert(raw_query, bert_tokenizer, bert_model):
    encoded_input = bert_tokenizer(raw_query, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        bert_model_output = bert_model(**encoded_input)
    query_matrix = cls_pooling(bert_model_output)
    query_matrix = normalize(query_matrix)

    return query_matrix


def count_sim(matrix, query_matrix):
    scores = np.dot(matrix, query_matrix.T)
    return scores


def count_sim_sparse(matrix, query_matrix):
    scores = matrix @ query_matrix.T
    return scores.toarray()


def order(row):
    order = np.argsort(row, axis=0)[::-1]
    return order


def metric_cv(proc_questions, proc_answers):
    # проиндексировать корпус
    answers_matrix, c_vectorizer = index_corpus_cv(proc_answers)

    # проиндексировать запросы
    questions_matrix = index_query_cv(proc_questions, c_vectorizer)

    # умножить матрицу на матрицу
    scores = count_sim(answers_matrix, questions_matrix)

    # сортировка и подсчёт
    score = 0
    for idx, row in enumerate(scores):
        # n-ной строке должен соответствовать n-ный столбец,
        # т.к. одинаковое кол-во вопросов и ответов
        if idx in order(row)[:5]:
            score += 1

    return score / answers_matrix.shape[0]


def metric_tfidf(proc_questions, proc_answers):
    # проиндексировать корпус
    answers_matrix, tfidf_vectorizer = index_corpus_tfidf(proc_answers)

    # проиндексировать запросы
    questions_matrix = index_query_tfidf(proc_questions, tfidf_vectorizer)

    # умножить матрицу на матрицу
    scores = count_sim_sparse(answers_matrix, questions_matrix)

    # сортировка и подсчёт
    score = 0
    for idx, row in enumerate(scores):
        # n-ной строке должен соответствовать n-ный столбец,
        # т.к. одинаковое кол-во вопросов и ответов
        if idx in order(row)[:5]:
            score += 1

    return score / answers_matrix.shape[0]


def metric_bm25(proc_questions, proc_answers):
    # проиндексировать корпус
    answers_matrix, c_vectorizer = index_corpus_bm25(proc_answers)

    # проиндексировать запросы
    questions_matrix = index_query_bm25(proc_questions, c_vectorizer)

    # умножить матрицу на матрицу
    scores = count_sim_sparse(answers_matrix, questions_matrix)

    # сортировка и подсчёт
    score = 0
    for idx, row in enumerate(scores):
        # n-ной строке должен соответствовать n-ный столбец,
        # т.к. одинаковое кол-во вопросов и ответов
        if idx in order(row)[:5]:
            score += 1

    return score / answers_matrix.shape[0]


def metric_fasttext(proc_questions, proc_answers):
    # проиндексировать корпус
    answers_matrix, fasttext_model = index_corpus_fasttext(proc_answers)

    # проиндексировать запросы
    questions_matrix = index_query_fasttext(proc_questions, fasttext_model)

    # умножить матрицу на матрицу
    scores = count_sim(answers_matrix, questions_matrix)

    # сортировка и подсчёт
    score = 0
    for idx, row in enumerate(scores):
        # n-ной строке должен соответствовать n-ный столбец,
        # т.к. одинаковое кол-во вопросов и ответов
        if idx in order(row)[:5]:
            score += 1

    return score / answers_matrix.shape[0]


def metric_bert(raw_questions, raw_answers):
    # проиндексировать корпус
    answers_matrix, bert_model = index_corpus_bert(raw_answers)

    # проиндексировать запросы
    questions_matrix = index_query_bert(raw_questions, **bert_model)

    # умножить матрицу на матрицу
    scores = count_sim(answers_matrix, questions_matrix)

    # сортировка и подсчёт
    score = 0
    for idx, row in enumerate(scores):
        # n-ной строке должен соответствовать n-ный столбец,
        # т.к. одинаковое кол-во вопросов и ответов
        if idx in order(row)[:5]:
            score += 1

    return score / answers_matrix.shape[0]


def compute_metrics(filename):
    proc_answers, proc_questions, raw_answers, raw_questions = collect_dataset(filename)
    print(f'Count: {metric_cv(proc_questions, proc_answers)}')
    print(f'Tf-Idf: {metric_tfidf(proc_questions, proc_answers)}')
    print(f'BM-25: {metric_bm25(proc_questions, proc_answers)}')
    print(f'FastText: {metric_fasttext(proc_questions, proc_answers)}')
    print(f'BERT: {metric_bert(raw_questions, raw_answers)}')


if __name__ == '__main__':
    compute_metrics('questions_about_love.jsonl')

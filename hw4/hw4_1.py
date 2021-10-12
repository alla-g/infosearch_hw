import json
import warnings
import torch

import numpy as np
import nltk
import pymorphy2

from string import digits
from pprint import pprint
from sklearn.preprocessing import normalize

from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors

warnings.filterwarnings("ignore")
morph = pymorphy2.MorphAnalyzer()


def cls_pooling(model_output):
    return model_output[0][:, 0]


def preprocess(text):
    tokens = [token for token in nltk.word_tokenize(text.lower())
              if token.isalpha() or token in digits]
    lemmed_text = [morph.parse(word)[0].normal_form for word in tokens]
    return ' '.join(lemmed_text)


def collect_answers(filename):
    with open(filename, 'r', encoding='UTF-8') as f:
        data = list(f)[:50000]  # 50000
    proc_corpus, raw_answers = [], []
    for question in data:
        item = json.loads(question)
        if item['answers']:
            ids = [int(answer['author_rating']['value']) for answer in item['answers']
                   if answer['author_rating']['value']]
            id_max = max(zip(ids, range(len(ids))))[1]
            answer = item['answers'][id_max]['text']
            proc_answer = preprocess(answer)
            if proc_answer:
                proc_corpus.append(proc_answer)
                raw_answers.append(answer)

    return proc_corpus, raw_answers


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
        matrix[doc_id] = doc_vector
        matrix = normalize(matrix)

    return matrix, fasttext_model


def index_query_fasttext(proc_query, fasttext_model):
    lemmas = proc_query.split()
    lem_vectors = np.zeros((len(lemmas), fasttext_model.vector_size))
    for lem_id, lemma in enumerate(lemmas):
        if lemma in fasttext_model:
            lem_vectors[lem_id] = fasttext_model[lemma]
    query_vector = np.array([np.mean(lem_vectors, axis=0)])
    query_vector = normalize(query_vector)
    return query_vector


def index_corpus_bert(raw_corpus):
    bert_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    bert_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

    encoded_input = bert_tokenizer(raw_corpus, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        bert_model_output = bert_model(**encoded_input)
    matrix = cls_pooling(bert_model_output)
    model = {'bert_tokenizer': bert_tokenizer,
             'bert_model': bert_model}
    matrix = normalize(matrix)

    return matrix, model


def index_query_bert(raw_query, bert_tokenizer, bert_model):
    encoded_input = bert_tokenizer(raw_query, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        bert_model_output = bert_model(**encoded_input)
    query_vector = cls_pooling(bert_model_output)
    query_vector = normalize(query_vector)

    return query_vector


def count_sim(matrix, query_vector):
    scores = np.dot(matrix, query_vector.T)

    return scores


def search(query, matrix, model, raw_answers, vectorizer):
    if vectorizer == 'fasttext':
        proc_query = preprocess(query)
        query_vector = index_query_fasttext(proc_query, model)
    elif vectorizer == 'bert':
        query_vector = index_query_bert(query, **model)
    else:
        raise TypeError('Unexpected vectorizer')
    scores = count_sim(matrix, query_vector)
    if np.all(scores == 0):
        return 'Ничего не нашлось:с'
    else:
        order = np.argsort(scores, axis=0)[::-1]
        arr_answers = np.array(raw_answers)
        sorted_answers = arr_answers[order[:5].ravel()]

    return list(sorted_answers)


def start(filename, vectorizer):
    proc_corpus, raw_answers = collect_answers(filename)
    if vectorizer == 'fasttext':
        matrix, model = index_corpus_fasttext(proc_corpus)
    elif vectorizer == 'bert':
        matrix, model = index_corpus_bert(raw_answers)
    else:
        raise TypeError('Unexpected vectorizer')

    while True:
        query = input('Введите поисковый запрос: ')
        output = search(query, matrix, model, raw_answers, vectorizer)
        pprint(output)
        answer = input('Хотите продолжить поиск? да/нет ')
        if answer.lower() == 'нет':
            break


if __name__ == '__main__':
    # 'fasttext' or 'bert'
    start('questions_about_love.jsonl', 'fasttext')

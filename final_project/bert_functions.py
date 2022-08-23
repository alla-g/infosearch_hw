import torch
import numpy as np
import torch.nn.functional as t
from common_functions import count_sim


def cls_pooling(model_output):
    return model_output[0][:, 0]


def index_corpus_bert(raw_corpus, bert_tokenizer, bert_model):
    encoded_input = bert_tokenizer(raw_corpus, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        bert_model_output = bert_model(**encoded_input)
    matrix = cls_pooling(bert_model_output)
    matrix = t.normalize(matrix)
    # сохранить матрицу
    torch.save(matrix, 'data/bert_matrix.pt')


def index_query_bert(raw_query, bert_tokenizer, bert_model):
    encoded_input = bert_tokenizer(raw_query, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        bert_model_output = bert_model(**encoded_input)
    query_vector = cls_pooling(bert_model_output)
    query_vector = t.normalize(query_vector)

    return query_vector


def search_bert(raw_query, **bert_model):
    matrix = torch.load('data/bert_matrix.pt')
    query_vector = index_query_bert(raw_query, **bert_model)
    scores = count_sim(matrix, query_vector)
    if np.all(scores == 0):
        return np.array([])
    else:
        order = np.argsort(scores, axis=0)[::-1][:5]
        return order

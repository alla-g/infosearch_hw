import numpy as np
from common_functions import count_sim, norm


def index_corpus_fasttext(proc_corpus, fasttext_model):
    matrix = np.zeros((len(proc_corpus), fasttext_model.vector_size))
    for doc_id, document in enumerate(proc_corpus):
        lemmas = document.split()
        lem_vectors = np.zeros((len(lemmas), fasttext_model.vector_size))
        for lem_id, lemma in enumerate(lemmas):
            if lemma in fasttext_model:
                lem_vectors[lem_id] = fasttext_model[lemma]
        doc_vector = np.mean(lem_vectors, axis=0)
        matrix[doc_id] = norm(doc_vector)
    # сохранить матрицу
    np.save('data/fasttext_matrix.npy', matrix)


def index_query_fasttext(proc_query, fasttext_model):
    lemmas = proc_query.split()
    lem_vectors = np.zeros((len(lemmas), fasttext_model.vector_size))
    for lem_id, lemma in enumerate(lemmas):
        if lemma in fasttext_model:
            lem_vectors[lem_id] = fasttext_model[lemma]
    query_vector = np.array([np.mean(lem_vectors, axis=0)])
    query_vector = norm(query_vector)

    return query_vector


def search_fasttext(proc_query, fasttext_model):
    matrix = np.load('data/fasttext_matrix.npy')
    query_vector = index_query_fasttext(proc_query, fasttext_model)
    scores = count_sim(matrix, query_vector)
    if np.all(scores == 0):
        return np.array([])
    else:
        order = np.argsort(scores, axis=0)[::-1][:5]
        return order

import pickle
import numpy as np
from common_functions import count_sim_sparse
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


def index_corpus_count(proc_corpus):
    c_vectorizer = CountVectorizer(analyzer='word')
    matrix = c_vectorizer.fit_transform(proc_corpus)
    matrix = normalize(matrix)
    # сохранить векторайзер
    pickle.dump(c_vectorizer, open('data/c_vectorizer.pickle', 'wb'))
    # сохранить матрицу
    sparse.save_npz('data/cv_matrix.npz', matrix)


def index_query_count(proc_query):
    c_vectorizer = pickle.load(open('data/c_vectorizer.pickle', 'rb'))
    query_vector = c_vectorizer.transform([proc_query])
    query_vector = normalize(query_vector)

    return query_vector


def search_count(proc_query):
    matrix = sparse.load_npz('data/cv_matrix.npz')
    query_vector = index_query_count(proc_query)
    scores = count_sim_sparse(matrix, query_vector)
    if np.all(scores == 0):
        return np.array([])
    else:
        order = np.argsort(scores, axis=0)[::-1][:5]
        return order
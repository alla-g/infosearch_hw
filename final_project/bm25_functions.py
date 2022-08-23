import pickle
import numpy as np
from common_functions import count_sim_sparse
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def index_corpus_bm25(proc_corpus):
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = pickle.load(open('data/tfidf_vectorizer.pickle', 'rb'))

    x_count_vec = sparse.load_npz('data/cv_matrix.npz')
    x_tf_vec = tf_vectorizer.fit_transform(proc_corpus)

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
    # сохранить матрицу
    sparse.save_npz('data/bm25_matrix.npz', matrix)


def index_query_bm25(proc_query):
    tfidf_vectorizer = pickle.load(open('data/tfidf_vectorizer.pickle', 'rb'))
    query_vector = tfidf_vectorizer.transform([proc_query])

    return query_vector


def search_bm25(proc_query):
    matrix = sparse.load_npz('data/bm25_matrix.npz')
    query_vector = index_query_bm25(proc_query)
    scores = count_sim_sparse(matrix, query_vector)
    if np.all(scores == 0):
        return np.array([])
    else:
        order = np.argsort(scores, axis=0)[::-1][:5]
        return order

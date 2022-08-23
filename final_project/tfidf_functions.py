import pickle
import numpy as np
from common_functions import count_sim_sparse
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def index_corpus_tfidf(proc_corpus):
    tfidf_vectorizer = TfidfVectorizer()
    matrix = tfidf_vectorizer.fit_transform(proc_corpus)
    # сохранить векторайзер
    pickle.dump(tfidf_vectorizer, open('data/tfidf_vectorizer.pickle', 'wb'))
    # сохранить матрицу
    sparse.save_npz('data/tfidf_matrix.npz', matrix)


def index_query_tfidf(proc_query):
    tfidf_vectorizer = pickle.load(open('data/tfidf_vectorizer.pickle', 'rb'))
    query_vector = tfidf_vectorizer.transform([proc_query])

    return query_vector


def search_tfidf(proc_query):
    matrix = sparse.load_npz('data/tfidf_matrix.npz')
    query_vector = index_query_tfidf(proc_query)
    scores = count_sim_sparse(matrix, query_vector)
    if np.all(scores == 0):
        return np.array([])
    else:
        order = np.argsort(scores, axis=0)[::-1][:5]
        return order

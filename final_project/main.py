import json
import time
import base64
import numpy as np
import streamlit as st

from transformers import AutoTokenizer, AutoModel
from gensim.models import KeyedVectors

from common_functions import collect_answers, preprocess
from count_functions import index_corpus_count, search_count
from tfidf_functions import index_corpus_tfidf, search_tfidf
from bm25_functions import index_corpus_bm25, search_bm25
from fasttext_functions import index_corpus_fasttext, search_fasttext
from bert_functions import index_corpus_bert, search_bert


@st.cache(allow_output_mutation=True)
def prestart(filename):
    """загрузить модели, посчитать и создать все нужные файлы"""
    ft_model_path = 'araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model'
    fasttext_model = KeyedVectors.load(ft_model_path)
    bert_tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    bert_model = {'bert_tokenizer': bert_tokenizer,
                  'bert_model': bert_model}
    # collect_answers(filename)
    with open('data/processed_corpus.json', 'r', encoding='UTF-8') as f:
        data = json.load(f)
    raw_answers = data['raw_answers']
    # одноразовая индексация корпуса:
    # raw_questions = data['raw_questions']
    # proc_questions = data['proc_questions']
    # index_corpus_count(proc_questions)
    # index_corpus_tfidf(proc_questions)
    # index_corpus_bm25(proc_questions)
    # index_corpus_fasttext(proc_questions, fasttext_model)
    # index_corpus_bert(raw_questions, **bert_model)

    return raw_answers, fasttext_model, bert_model


raw_answers, fasttext_model, bert_model = prestart('data/questions_about_love.jsonl')
arr_answers = np.array(raw_answers)


def start(query, vectorizer):
    if vectorizer == 'count':
        proc_query = preprocess(query)
        order = search_count(proc_query)
    elif vectorizer == 'tfidf':
        proc_query = preprocess(query)
        order = search_tfidf(proc_query)
    elif vectorizer == 'bm25':
        proc_query = preprocess(query)
        order = search_bm25(proc_query)
    elif vectorizer == 'fasttext':
        proc_query = preprocess(query)
        order = search_fasttext(proc_query, fasttext_model)
    elif vectorizer == 'bert':
        order = search_bert(query, **bert_model)
    else:
        raise TypeError('Unknown vectorizer')
    if order.size:
        sorted_answers = arr_answers[order.ravel()]
        return sorted_answers
    else:
        return(['О нет, наши мудрецы ещё не задавались таким вопросом. '
                'Спросите что-нибудь ещё'])


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("data:image/png;base64,%s");
      background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('data/background.png')

st.title('поисковик ответов про любовь')
st.header('Что вы хотите узнать?')
st.info('Здесь можно задать самые сокровенные вопросы лучшим умам '
        'человечества, чью мудрость навечно сохранил сайт ответы маил ру. '
        'Попробуйте')
place_zero = st.empty()
place_one = st.empty()
place_two = st.empty()
place_three = st.empty()
place_four = st.empty()


def search_function():
    query = ' '
    query = place_zero.text_input('Ваш тревожащий вопрос:', query)
    options = ['count', 'tfidf', 'bm25', 'fasttext', 'bert']
    display = ['Count (считаем слова)',
               'Tf-idf (считаем важные слова)',
               'BM-25 (считаем важные слова по-сложному)',
               'FastText (крутая векторная модель)',
               'BERT (очень крутая векторная модель)']
    vectorizer = place_one.selectbox('Надо выбрать метод поиска:',
                                     options, format_func=lambda x: display[options.index(x)])
    button = place_two.button('Хочу ответ!')

    if button:
        start_time = time.time()
        with st.spinner('Ищем...'):
            answers = start(query, vectorizer)
        place_one.caption('Наш ответ:')
        place_two.write('  \n'.join(answers))
        end_time = time.time()
        seconds = end_time - start_time
        place_three.caption(f'Поиск сработал за {seconds:.4f} сек')
        new_button = place_four.button('Спросить ещё!')

        if new_button:
            search_function()


search_function()
